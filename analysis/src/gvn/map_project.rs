//! Element projection out of a `map`: the `Extract`/`Range`-of-`Map` rewrite.
//!
//! A byte slice of a mapped array — `Range(Map(body, src), k·esz, esz)`, the
//! shape a caller's `buf[k]` reload takes after store-to-load forwarding — is the
//! `k`-th lane of the result, i.e. `body(k, src[k], captures…)`. Since `Call` is
//! a terminator, we cannot emit a call mid-expression; instead we **inline**
//! `body`'s pure straight-line computation with its index bound to the constant
//! `k` and its element bound to `src[k]`. Constant folding then reduces the
//! inlined expression (to a literal when `src[k]` is itself constant).
//!
//! This is the projectability half of `ARGPROMOTE_ARRAY_MAP.md`: it recovers one
//! element as its actual expression without materializing the whole array.

use qcode::{
    context::Context,
    value::{
        BasicBlock, Function, InstructionRef, ValueId,
        insn::{Mnemonic, Range},
    },
};

use crate::calls::inline_pure_body;

use super::walk::{Claim, Editor, InsnCtx, SubPass};

pub(super) struct MapProject;

impl SubPass for MapProject {
    type State = ();

    fn on_insn(&self, ctx: &mut Context, _state: &mut (), ic: &InsnCtx, ed: &mut Editor) -> Claim {
        // A byte-range slice of some value.
        let Mnemonic::Range(Range { src, start, size }) = *ic.mnemonic else {
            return Claim::Pass;
        };
        // …whose source is a `map`.
        let ValueId::Instruction(map_id) = src else {
            return Claim::Pass;
        };
        let Mnemonic::Map(map) = ctx.get_insn(map_id).mnemonic().clone() else {
            return Claim::Pass;
        };

        // Lane alignment: the slice must be exactly one element `esz` wide and
        // start on a lane boundary, giving lane index `k`.
        let map_ty = ctx.type_of(map.src);
        let Some((elem_ty, _count)) = ctx.types.array_of(map_ty) else {
            return Claim::Pass;
        };
        let esz = ctx.types.size_of(elem_ty);
        if esz == 0 || size != esz || start % esz != 0 {
            return Claim::Pass;
        }
        let k = (start / esz) as u64;

        // `src[k]` — the element fed to the body.
        let element = {
            let r = InstructionRef::from_mnemonic(
                ctx,
                Mnemonic::Range(Range {
                    src: map.src,
                    start,
                    size,
                }),
                size,
            )
            .id;
            BasicBlock::from_id_mut(ctx, ic.block_id).insert_insn_before(ic.insn_id, r);
            ValueId::Instruction(r)
        };

        // The body's first param is the index; size the constant `k` to it.
        let Some(root) = Function::from_id(ctx, map.body).root().map(|b| b.id) else {
            return Claim::Pass;
        };
        let Some(index_size) = BasicBlock::from_id(ctx, root).params().next().map(|p| p.size())
        else {
            return Claim::Pass;
        };
        let k_const = ctx.get_const(k, index_size).id();

        // body(k, src[k], captures…), inlined before this instruction.
        let mut args = vec![k_const, element];
        args.extend(map.captures.iter().copied());
        let Some(result) = inline_pure_body(ctx, map.body, &args, ic.block_id, ic.insn_id) else {
            return Claim::Pass;
        };

        ed.replace(ctx, ic.insn_id, result);
        Claim::Done
    }
}

#[cfg(test)]
mod tests {
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{
            BasicBlock, Function, FunctionId, Value, ValueId,
            insn::{Mnemonic, Return},
        },
    };

    /// `body(idx: i64, elem: i8) -> elem + 1`, marked pure.
    fn build_inc_body(tc: &mut TestContext) -> FunctionId {
        let fid = Function::make(&mut tc.ctx, "inc".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let (inc, ptr, ret);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let _idx = b.push_param(8).id();
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
        let entry = tc.ctx.get_or_make_block(0x5000);
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
            tc.ctx.values.block_params[pid].type_id = arr_ty;
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

        let aliases = crate::AliasResult::simple(&tc.ctx);
        super::super::gvn_function(&mut tc.ctx, host, Some(&aliases));

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
}
