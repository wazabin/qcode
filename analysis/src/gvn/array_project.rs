//! Element projection out of array-producing nodes — one sub-pass holding the
//! `Range`-of-`Map`, `Range`-of-`enumerate`, and `Extract`-of-`Tuple` rewrites.
//!
//! A byte slice of such an array — `Range(arr, k·esz, esz)`, the shape a
//! caller's `buf[k]` reload takes after store-to-load forwarding — is the `k`-th
//! lane of the result. Recovering that one element without materializing the
//! whole array is the projectability half of `ARGPROMOTE_ARRAY_MAP.md` and lets
//! `enumerate` survive as a returned value.
//!
//! * **`Range(Map(body, src), k·osz, osz)` ⇒ `body(src[k], captures…)`.** The
//!   `map` is unary in its element, so the `k`-th output lane is the body applied
//!   to `src[k]`. Since `Call` is a terminator we cannot emit a call
//!   mid-expression; instead we **inline** `body`'s pure straight-line
//!   computation with its element param bound to `src[k]`. Constant folding then
//!   reduces the inlined expression (to a literal when `src[k]` is constant). The
//!   output lane width `osz` (the body's return type) may differ from the input
//!   element width used to slice `src` — e.g. a map over `enumerate(arr)`.
//! * **`Range(enumerate(src), k·tsz, tsz)` ⇒ `(index: k, elem: src[k])`.** The
//!   `k`-th lane of an `enumerate` is the tuple of its index and element; no body
//!   to inline.
//! * **`Extract(Tuple{fields…}, i)` ⇒ `fields[i]`.** Reads a field straight out
//!   of a freshly-packed tuple, so `enumerate(arr)[k].elem` reduces to `src[k]`.

use qcode::{
    context::Context,
    value::{
        BasicBlock, InstructionRef, ValueId,
        insn::{Extract, Mnemonic, Range, Tuple},
    },
};

use crate::calls::inline_pure_body;

use std::any::Any;

use super::walk::{Claim, Editor, InsnCtx, SubPass};

/// Projecting a lane out of a `map` inlines the pure *body callee*'s IR, so this
/// runs only on the module host (dispatched by the
/// [`concretize`](super::concretize) module pass).
pub(super) struct ArrayProject;

impl<'a, 'str> SubPass<'str, &'a mut Context<'str>> for ArrayProject {
    fn init_state(&self) -> Box<dyn Any> {
        Box::new(())
    }

    fn clone_state(&self, _state: &dyn Any) -> Box<dyn Any> {
        Box::new(())
    }

    fn on_insn(
        &self,
        host: &mut &'a mut Context<'str>,
        _state: &mut dyn Any,
        ic: &InsnCtx,
        ed: &mut Editor,
    ) -> Claim {
        let ctx: &mut Context = host;
        match *ic.mnemonic {
            Mnemonic::Range(Range { src, start, size }) => {
                self.project_range(ctx, ic, ed, src, start, size)
            }
            Mnemonic::Extract(Extract { agg, index }) => {
                self.fold_extract_tuple(ctx, ic, ed, agg, index)
            }
            _ => Claim::Pass,
        }
    }
}

impl ArrayProject {
    /// Project a single lane out of a `Range` whose source is a `map` or an
    /// `enumerate`.
    fn project_range(
        &self,
        ctx: &mut Context,
        ic: &InsnCtx,
        ed: &mut Editor,
        src: ValueId,
        start: usize,
        size: usize,
    ) -> Claim {
        let ValueId::Instruction(src_id) = src else {
            return Claim::Pass;
        };
        match ctx.get_insn(src_id).mnemonic().clone() {
            Mnemonic::Map(_) => self.project_map(ctx, ic, ed, src_id, start, size),
            Mnemonic::Intrinsic(intr) if intr.id.name() == "enumerate" => {
                self.project_enumerate(ctx, ic, ed, src, intr.args[0], start, size)
            }
            Mnemonic::Intrinsic(intr) if intr.id.name() == "concat" => {
                self.project_concat(ctx, ic, ed, &intr.args, start, size)
            }
            // A `scan` is deliberately *not* projected: lane `k` is the `k`-th
            // accumulator, which depends on the whole prefix `0..=k`, not just
            // `src[k]`, so there is no cheap single-element extract.
            _ => Claim::Pass,
        }
    }

    /// `Range(Map(body, src), k·osz, osz)` ⇒ inlined `body(src[k], captures…)`.
    fn project_map(
        &self,
        mut ctx: &mut Context,
        ic: &InsnCtx,
        ed: &mut Editor,
        map_id: qcode::value::InstructionId,
        start: usize,
        size: usize,
    ) -> Claim {
        let Mnemonic::Map(map) = ctx.get_insn(map_id).mnemonic().clone() else {
            return Claim::Pass;
        };

        // Lane alignment is against the map's *output* element width `osz` (the
        // body's result type), which need not equal the input element width — a
        // `map` over `enumerate(arr)` consumes `(index, elem)` tuples but produces
        // bare elements.
        let out_ty = ctx.type_of(ValueId::Instruction(map_id));
        let Some((out_elem, _)) = ctx.types.array_of(out_ty) else {
            return Claim::Pass;
        };
        let osz = ctx.types.size_of(out_elem);
        if osz == 0 || size != osz || !start.is_multiple_of(osz) {
            return Claim::Pass;
        }
        let k = (start / osz) as u64;

        // `src[k]` — the element fed to the body — is sliced at the *input*
        // element width `isz`.
        let in_ty = ctx.type_of(map.src);
        let Some((in_elem, _)) = ctx.types.array_of(in_ty) else {
            return Claim::Pass;
        };
        let isz = ctx.types.size_of(in_elem);
        if isz == 0 {
            return Claim::Pass;
        }
        let element = {
            let r = InstructionRef::from_mnemonic(
                ctx,
                ic.block_id.func,
                Mnemonic::Range(Range {
                    src: map.src,
                    start: (k as usize) * isz,
                    size: isz,
                }),
                isz,
            )
            .id;
            BasicBlock::from_id_mut(ctx, ic.block_id).insert_insn_before(ic.insn_id, r);
            ValueId::Instruction(r)
        };

        // body(src[k], captures…), inlined before this instruction. The body is
        // unary in the element; index-aware bodies take an `enumerate` tuple as
        // that element and unpack it internally.
        let mut args = vec![element];
        args.extend(map.captures.iter().copied());
        let Some(result) = inline_pure_body(ctx, map.body, &args, ic.block_id, ic.insn_id) else {
            return Claim::Pass;
        };

        ed.replace(&mut ctx, ic.insn_id, result);
        Claim::Done
    }

    /// `Range(enumerate(src), k·tsz, tsz)` ⇒ `pack(index = k, elem = src[k])`.
    #[allow(clippy::too_many_arguments)]
    fn project_enumerate(
        &self,
        mut ctx: &mut Context,
        ic: &InsnCtx,
        ed: &mut Editor,
        enum_val: ValueId,
        src: ValueId,
        start: usize,
        size: usize,
    ) -> Claim {
        // The enumerate result type `[(index: i64, elem: T); N]` gives the tuple
        // width `tsz`; the operand array `[T; N]` gives the element width `esz`.
        let enum_ty = ctx.type_of(enum_val);
        let Some((tuple_ty, _count)) = ctx.types.array_of(enum_ty) else {
            return Claim::Pass;
        };
        let tsz = ctx.types.size_of(tuple_ty);
        let src_ty = ctx.type_of(src);
        let Some((elem_ty, _)) = ctx.types.array_of(src_ty) else {
            return Claim::Pass;
        };
        let esz = ctx.types.size_of(elem_ty);

        // Lane alignment: exactly one tuple wide, starting on a lane boundary.
        if tsz == 0 || size != tsz || !start.is_multiple_of(tsz) {
            return Claim::Pass;
        }
        let k = (start / tsz) as u64;

        // `index = k`, an i64 (the field width fixed by `enumerate`).
        let index_ty = ctx
            .types
            .field_type(tuple_ty, 0)
            .expect("enumerate tuple has an index field");
        let index = ctx.get_const(k, ctx.types.size_of(index_ty)).id();

        // `elem = src[k]`, a one-element byte slice of the operand array.
        let element = {
            let r = InstructionRef::from_mnemonic(
                ctx,
                ic.block_id.func,
                Mnemonic::Range(Range {
                    src,
                    start: (k as usize) * esz,
                    size: esz,
                }),
                esz,
            )
            .id;
            BasicBlock::from_id_mut(ctx, ic.block_id).insert_insn_before(ic.insn_id, r);
            ValueId::Instruction(r)
        };

        // pack the `(index, elem)` tuple, typed as the enumerate element type.
        let tuple = {
            let t = InstructionRef::from_mnemonic_with_type(
                ctx,
                ic.block_id.func,
                Mnemonic::Tuple(Tuple {
                    fields: vec![index, element],
                }),
                tuple_ty,
            )
            .id;
            BasicBlock::from_id_mut(ctx, ic.block_id).insert_insn_before(ic.insn_id, t);
            ValueId::Instruction(t)
        };

        ed.replace(&mut ctx, ic.insn_id, tuple);
        Claim::Done
    }

    /// `Range(concat(a, b), off, size)` ⇒ `Range(a, off, size)` when wholly in
    /// `a`, or `Range(b, off - sizeof(a), size)` when wholly in `b`.
    fn project_concat(
        &self,
        mut ctx: &mut Context,
        ic: &InsnCtx,
        ed: &mut Editor,
        args: &[ValueId],
        start: usize,
        size: usize,
    ) -> Claim {
        let [lhs, rhs] = args else {
            return Claim::Pass;
        };
        let lhs_ty = ctx.type_of(*lhs);
        let Some((lhs_elem, lhs_len, _)) = ctx.types.seq_of(lhs_ty) else {
            return Claim::Pass;
        };
        let lhs_bytes = ctx.types.size_of(lhs_elem) * lhs_len;

        let (src, rel_start) = if start + size <= lhs_bytes {
            (*lhs, start)
        } else if start >= lhs_bytes {
            (*rhs, start - lhs_bytes)
        } else {
            return Claim::Pass;
        };

        let r = InstructionRef::from_mnemonic(
            ctx,
            ic.block_id.func,
            Mnemonic::Range(Range {
                src,
                start: rel_start,
                size,
            }),
            size,
        )
        .id;
        BasicBlock::from_id_mut(ctx, ic.block_id).insert_insn_before(ic.insn_id, r);
        ed.replace(&mut ctx, ic.insn_id, ValueId::Instruction(r));
        Claim::Done
    }

    /// `Extract(Tuple{fields…}, i)` ⇒ `fields[i]`.
    fn fold_extract_tuple(
        &self,
        mut ctx: &mut Context,
        ic: &InsnCtx,
        ed: &mut Editor,
        agg: ValueId,
        index: usize,
    ) -> Claim {
        let ValueId::Instruction(tuple_id) = agg else {
            return Claim::Pass;
        };
        let Mnemonic::Tuple(Tuple { fields }) = ctx.get_insn(tuple_id).mnemonic().clone() else {
            return Claim::Pass;
        };
        let Some(&field) = fields.get(index) else {
            return Claim::Pass;
        };
        ed.replace(&mut ctx, ic.insn_id, field);
        Claim::Done
    }
}

#[cfg(test)]
mod tests {
    use qcode::{
        builder::Builder,
        testing::TestContext,
        types::TypeId,
        value::{
            BasicBlock, Function, FunctionId, Instruction, Value, ValueId,
            insn::{IntrinsicId, Mnemonic, Return, Store},
        },
    };

    /// `body(elem: i8) -> elem + 1`, marked pure. A unary map body.
    fn build_inc_body(tc: &mut TestContext) -> FunctionId {
        let fid = Function::make(&mut tc.ctx, "inc".into()).unwrap().id;
        let entry = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x1000, __f)
        };
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let (inc, ptr, ret);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let elem = b.push_param(1).id();
            let one = b.context_mut().get_const(1, 1).id();
            inc = b.push_add(elem, one).id();
            ptr = b.context_mut().get_const(0, 8).id();
            ret = b.push_return(ptr).id();
            unsafe { b.dont_finalize() };
        }
        let ValueId::Instruction(rid) = ret else {
            unreachable!()
        };
        tc.ctx.replace_instruction_mnemonic(
            rid,
            Mnemonic::Return(Return {
                ptr,
                value: Some(inc),
            }),
        );
        Function::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);
        fid
    }

    /// `Range(Map(inc, src), 2, 1)` projects to the inlined `src[2] + 1`.
    #[test]
    fn projects_lane_from_map() {
        let mut tc = TestContext::new();
        let body = build_inc_body(&mut tc);

        let host = Function::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x5000, __f)
        };
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let i8 = tc.ctx.types.get_or_make_int(1);
        let arr_ty = tc.ctx.types.get_or_make_array(i8, 4);

        // src param, typed as the array *before* the map captures its type.
        let src = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_param(4).id()
        };
        if let ValueId::BlockParam(pid) = src {
            tc.ctx.values.block_param_mut(pid).type_id = arr_ty;
        }
        let reg_space = tc.reg_space;
        let r0 = tc.r0;
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let map = b.push_map(body, src, Vec::new()).id();
            // src[2]: a 1-byte slice at byte offset 2.
            let lane = b.get_range(map, 2..3).unwrap().id();
            b.push_store(lane, ValueId::Varnode(r0), reg_space);
            let ptr = b.context_mut().get_const(0, 8).id();
            b.push_return(ptr);
        }

        super::super::concretize::concretize_function(&mut tc.ctx, host);

        // No Range-of-Map remains; the body was inlined (an int_add appears), and
        // the surviving Range now slices the array source directly (`src[2]`).
        let insns: Vec<Mnemonic> = BasicBlock::from_id(&tc.ctx, entry)
            .iter()
            .map(|i| i.mnemonic().clone())
            .collect();
        let range_over_map = insns.iter().any(|m| {
            matches!(m, Mnemonic::Range(r) if matches!(r.src, ValueId::Instruction(id)
                if matches!(BasicBlock::from_id(&tc.ctx, entry).iter().find(|i| i.id == id).map(|i| i.mnemonic().clone()), Some(Mnemonic::Map(_)))))
        });
        assert!(!range_over_map, "the Range-of-Map must be projected away");
        assert!(
            insns.iter().any(|m| matches!(m, Mnemonic::Binop(_))),
            "the inlined body (elem + 1) must appear"
        );
        let slices_src = insns
            .iter()
            .any(|m| matches!(m, Mnemonic::Range(r) if r.src == src));
        assert!(slices_src, "a Range now slices the array source directly");
    }

    /// The literal value stored into varnode `reg` in `block`, if any.
    fn stored_const(
        tc: &TestContext,
        block: qcode::value::block::BlockId,
        reg: ValueId,
    ) -> Option<u64> {
        BasicBlock::from_id(&tc.ctx, block).iter().find_map(|i| {
            let Mnemonic::Store(Store { ptr, src, .. }) = i.mnemonic() else {
                return None;
            };
            if *ptr != reg {
                return None;
            }
            match src {
                ValueId::Literal(lid) => Some(tc.ctx.values.literals[*lid].value),
                _ => None,
            }
        })
    }

    /// `enumerate(src)[2]` projects to `(index: 2, elem: src[2])`, and the field
    /// extracts fold through the freshly-packed tuple: `.index → 2`, `.elem →
    /// src[2]`.
    #[test]
    fn projects_lane_from_enumerate() {
        let mut tc = TestContext::new();
        let enum_id = IntrinsicId::from_name("enumerate").unwrap();

        let host = Function::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x6000, __f)
        };
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let i8 = tc.ctx.types.get_or_make_int(1);
        let arr_ty = tc.ctx.types.get_or_make_array(i8, 4);

        // src param, typed as the array `[i8; 4]`.
        let src = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_param(4).id()
        };
        if let ValueId::BlockParam(pid) = src {
            tc.ctx.values.block_param_mut(pid).type_id = arr_ty;
        }

        // enumerate(src): `[(index: i64, elem: i8); 4]`, tuple width 9.
        let en = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_intrinsic(enum_id, vec![src]).id()
        };
        let en_ty = tc.ctx.type_of(en);
        let (tuple_ty, _) = tc.ctx.types.array_of(en_ty).unwrap();
        let tsz = tc.ctx.types.size_of(tuple_ty);

        // The k=2 lane: a tuple-typed byte slice `enumerate(src)[2]`.
        let lane = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.get_range(en, (2 * tsz)..(3 * tsz)).unwrap().id()
        };
        if let ValueId::Instruction(lid) = lane {
            Instruction::from_id_mut(&mut tc.ctx, lid).set_type(tuple_ty);
        }

        let (r0, r1, reg_space) = (tc.r0, tc.r1, tc.reg_space);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let index = b.push_extract(lane, 0).id();
            let elem = b.push_extract(lane, 1).id();
            b.push_store(index, ValueId::Varnode(r0), reg_space);
            b.push_store(elem, ValueId::Varnode(r1), reg_space);
            let ptr = b.context_mut().get_const(0, 8).id();
            b.push_return(ptr);
        }

        super::super::concretize::concretize_function(&mut tc.ctx, host);

        let insns: Vec<Mnemonic> = BasicBlock::from_id(&tc.ctx, entry)
            .iter()
            .map(|i| i.mnemonic().clone())
            .collect();

        // The Range-of-enumerate is projected away.
        let range_over_enum = insns
            .iter()
            .any(|m| matches!(m, Mnemonic::Range(r) if r.src == en));
        assert!(
            !range_over_enum,
            "the Range-of-enumerate must be projected away"
        );

        // `.index` folded to the constant lane index 2.
        assert_eq!(
            stored_const(&tc, entry, ValueId::Varnode(r0)),
            Some(2),
            "the index field must fold to the constant lane 2"
        );

        // `.elem` folded to a direct slice of the source array, `src[2]`.
        let elem_slices_src = insns
            .iter()
            .any(|m| matches!(m, Mnemonic::Range(r) if r.src == src && r.start == 2));
        assert!(elem_slices_src, "the elem field must slice src[2] directly");
    }

    /// `concat(a, b)[4]` where `a` has 3 lanes projects to `b[1]`.
    #[test]
    fn projects_lane_from_concat_rhs() {
        let mut tc = TestContext::new();
        let concat_id = IntrinsicId::from_name("concat").unwrap();

        let host = Function::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x6800, __f)
        };
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let i8 = tc.ctx.types.get_or_make_int(1);
        let a_ty = tc.ctx.types.get_or_make_array(i8, 3);
        let b_ty = tc.ctx.types.get_or_make_array(i8, 5);

        let (a, b_src) = {
            let mut builder = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            (builder.push_param(3).id(), builder.push_param(5).id())
        };
        if let ValueId::BlockParam(pid) = a {
            tc.ctx.values.block_param_mut(pid).type_id = a_ty;
        }
        if let ValueId::BlockParam(pid) = b_src {
            tc.ctx.values.block_param_mut(pid).type_id = b_ty;
        }

        let reg_space = tc.reg_space;
        let r0 = tc.r0;
        {
            let mut builder = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let concat = builder.push_intrinsic(concat_id, vec![a, b_src]).id();
            let lane = builder.get_range(concat, 4..5).unwrap().id();
            builder.push_store(lane, ValueId::Varnode(r0), reg_space);
            let ptr = builder.context_mut().get_const(0, 8).id();
            builder.push_return(ptr);
        }

        super::super::concretize::concretize_function(&mut tc.ctx, host);

        let insns: Vec<Mnemonic> = BasicBlock::from_id(&tc.ctx, entry)
            .iter()
            .map(|i| i.mnemonic().clone())
            .collect();
        assert!(
            insns
                .iter()
                .any(|m| matches!(m, Mnemonic::Range(r) if r.src == b_src && r.start == 1)),
            "concat lane 4 should project to rhs lane 1"
        );
    }

    /// `body(t: (index: i64, elem: i8)) -> t.elem`, marked pure. The unary,
    /// index-aware map body: it takes the `enumerate` tuple and unpacks it.
    fn build_unpack_elem_body(tc: &mut TestContext, tuple_ty: TypeId) -> FunctionId {
        let fid = Function::make(&mut tc.ctx, "unpack".into()).unwrap().id;
        let entry = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x2000, __f)
        };
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let tsz = tc.ctx.types.size_of(tuple_ty);
        let t = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_param(tsz).id()
        };
        if let ValueId::BlockParam(pid) = t {
            tc.ctx.values.block_param_mut(pid).type_id = tuple_ty;
        }
        let (elem, ptr, ret) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let elem = b.push_extract(t, 1).id();
            let ptr = b.context_mut().get_const(0, 8).id();
            let ret = b.push_return(ptr).id();
            unsafe { b.dont_finalize() };
            (elem, ptr, ret)
        };
        let ValueId::Instruction(rid) = ret else {
            unreachable!()
        };
        tc.ctx.replace_instruction_mnemonic(
            rid,
            Mnemonic::Return(Return {
                ptr,
                value: Some(elem),
            }),
        );
        Function::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);
        fid
    }

    /// `map(unpack, enumerate(src))[2]` projects through the unary map and the
    /// enumerate tuple: `unpack(enumerate(src)[2]) = enumerate(src)[2].elem =
    /// src[2]`. Exercises project_map (unary) ∘ project_enumerate ∘ Extract-fold.
    #[test]
    fn projects_lane_from_map_over_enumerate() {
        let mut tc = TestContext::new();
        let enum_id = IntrinsicId::from_name("enumerate").unwrap();

        let host = Function::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x7000, __f)
        };
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let i8 = tc.ctx.types.get_or_make_int(1);
        let arr_ty = tc.ctx.types.get_or_make_array(i8, 4);

        // The enumerate tuple type `(index: i64, elem: i8)`, via enumerate's own
        // result-type rule, so the body's param matches the lane the map yields.
        let enum_result_ty = enum_id.desc().result_type(&tc.ctx.types, &[arr_ty]);
        let (tuple_ty, _) = tc.ctx.types.array_of(enum_result_ty).unwrap();
        let body = build_unpack_elem_body(&mut tc, tuple_ty);

        // src param `[i8; 4]`.
        let src = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_param(4).id()
        };
        if let ValueId::BlockParam(pid) = src {
            tc.ctx.values.block_param_mut(pid).type_id = arr_ty;
        }

        // map(unpack, enumerate(src)). The body returns `i8`, so `push_map` types
        // the result `[i8; 4]` (from the body's return), not the `[tuple; 4]`
        // source.
        let (en, map_val) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let en = b.push_intrinsic(enum_id, vec![src]).id();
            let map_val = b.push_map(body, en, Vec::new()).id();
            (en, map_val)
        };

        let reg_space = tc.reg_space;
        let r0 = tc.r0;
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            // map(...)[2]: a 1-byte output slice at lane 2.
            let lane = b.get_range(map_val, 2..3).unwrap().id();
            b.push_store(lane, ValueId::Varnode(r0), reg_space);
            let ptr = b.context_mut().get_const(0, 8).id();
            b.push_return(ptr);
        }

        // Each projection inserts instructions a later GVN sweep reduces (the map
        // lane inlines the body, whose `Extract(enumerate[2])` then projects and
        // folds), so iterate to a fixpoint as the real pass pipeline does.
        super::super::concretize::concretize_function(&mut tc.ctx, host);

        let insns: Vec<Mnemonic> = BasicBlock::from_id(&tc.ctx, entry)
            .iter()
            .map(|i| i.mnemonic().clone())
            .collect();

        // Neither the map nor the enumerate is sliced any more — both projected
        // through to a direct `src[2]`.
        let range_over_derived = insns
            .iter()
            .any(|m| matches!(m, Mnemonic::Range(r) if r.src == map_val || r.src == en));
        assert!(
            !range_over_derived,
            "map/enumerate slices must project away"
        );
        let slices_src = insns
            .iter()
            .any(|m| matches!(m, Mnemonic::Range(r) if r.src == src && r.start == 2));
        assert!(slices_src, "the lane must reduce to src[2]");
    }
}
