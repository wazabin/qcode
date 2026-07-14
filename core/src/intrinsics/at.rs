//! The `at` intrinsic: `[T; N], i64 -> T` — a dynamic-lane array read.
//!
//! `at(arr, i)` reads lane `i` of a sequence at a value-level (possibly dynamic)
//! index — the counterpart of the constant-index [`Extract`](crate::value::insn::Extract),
//! needed when a promoted loop forwards a previous lane it just wrote
//! (`mt[i] = f(mt[i-1], i)` reads `at(arr, i-1)`). It forwards through `insert`:
//!
//! * `at(insert(a, i, v), i) = v`   (same index — provably equal),
//! * `at(insert(a, i, v), j) = at(a, j)`   (indices provably distinct),
//! * `at(Bytes, const j) = arr[j]`   (constant read out of a constant blob),
//! * `at(singleton(v), _) = v`   (a one-lane array has only lane 0),
//! * `at(concat(a, b), const j) = at(a, j)` / `at(b, j - len a)`   (side pick).

use crate::register_intrinsic;
use crate::types::{TypeId, TypeManager};
use crate::value::ValueId;
use crate::value::insn::{Intrinsic, IntrinsicId, Mnemonic, Simplified};
use crate::value::{BodyView, QCodeView};

/// `at` — read a sequence lane at a dynamic index.
struct At;

/// Whether two index operands are provably equal, provably distinct, or unknown.
enum IdxRel {
    Equal,
    Distinct,
    Unknown,
}

fn index_rel(view: BodyView<'_, '_>, a: ValueId, b: ValueId) -> IdxRel {
    if a == b {
        return IdxRel::Equal;
    }
    if let (ValueId::Literal(x), ValueId::Literal(y)) = (a, b) {
        return if view.shared().values.literals[x].value == view.shared().values.literals[y].value {
            IdxRel::Equal
        } else {
            IdxRel::Distinct
        };
    }
    IdxRel::Unknown
}

impl Intrinsic for At {
    fn name(&self) -> &'static str {
        "at"
    }

    fn arity(&self) -> usize {
        2
    }

    fn result_type(&self, types: &TypeManager, args: &[TypeId]) -> TypeId {
        // The sequence's element type. Falls back to the operand type itself for a
        // non-sequence (defensive; the builder only emits `at` over sequences).
        types.seq_elem_of(args[0]).unwrap_or(args[0])
    }

    fn eval(&self, _args: &[(u128, usize)], _out_size: usize) -> Option<u128> {
        // Reads an array lane; not expressible through scalar `eval` (the array is
        // not a scalar operand). Constant reads are handled in `simplify`.
        None
    }

    fn simplify(
        &self,
        view: BodyView<'_, '_>,
        _id: IntrinsicId,
        out_size: usize,
        args: &[ValueId],
    ) -> Option<Simplified> {
        let &[arr, index] = args else {
            return None;
        };
        simplify_at(view, out_size, arr, index)
    }
}

/// Forward `at(arr, index)` through the array constructors, recursively: a
/// `concat` side-pick or `insert` distinct-index re-issue lands on a smaller
/// array which is itself simplified in the same step. Without the recursion the
/// re-issued `at` would be materialized verbatim (e.g. `at(concat(singleton(v),
/// scan), 0)` narrows to `at(singleton(v), 0)` but stops there instead of folding
/// to `v`), leaving the seed read un-concretized and its array live.
fn simplify_at(
    view: BodyView<'_, '_>,
    out_size: usize,
    arr: ValueId,
    index: ValueId,
) -> Option<Simplified> {
    {
        // Read straight out of a constant `Bytes` array at a constant index.
        if let (ValueId::Bytes(bid), ValueId::Literal(ilit)) = (arr, index) {
            let arr_ty = view.shared().values.bytes[bid].type_id;
            if let Some((elem, count)) = view.shared().types.array_of(arr_ty) {
                let esz = view.shared().types.size_of(elem);
                let i = view.shared().values.literals[ilit].value as usize;
                if i < count {
                    let off = i * esz;
                    let data = &view.shared().values.bytes[bid].data;
                    let mut buf = [0u8; 8];
                    buf[..esz].copy_from_slice(&data[off..off + esz]);
                    let v = u64::from_le_bytes(buf);
                    return Some(Simplified::Value(view.shared().get_const(v, out_size)));
                }
            }
        }

        // Forward through the array constructors: `insert`, `singleton`, `concat`.
        let ValueId::Instruction(iid) = arr else {
            return None;
        };
        let Mnemonic::Intrinsic(app) = view.instruction(iid).mnemonic() else {
            return None;
        };
        let name = app.id.name();
        // Operands are stored bare-local; qualify with the intrinsic's own function
        // so the rest of this simplifier speaks the boundary `ValueId`.
        let args: Vec<ValueId> = app.args.iter().map(|a| a.qualify(iid.func)).collect();
        match name {
            // `at(insert(a, i, v), j)`.
            "insert" => {
                let (base, ins_idx, val) = (args[0], args[1], args[2]);
                match index_rel(view, ins_idx, index) {
                    IdxRel::Equal => Some(Simplified::Value(val)),
                    // Re-issue the read against the underlying array.
                    IdxRel::Distinct => Some(forward(view, out_size, base, index)),
                    IdxRel::Unknown => None,
                }
            }
            // A one-lane array has only lane 0 (any other index is UB anyway).
            "singleton" => Some(Simplified::Value(args[0])),
            // Every lane of `splat(v, n)` is `v`, regardless of the index.
            "splat" => Some(Simplified::Value(args[0])),
            // `at(concat(a, b), const j)`: pick the side `j` falls in.
            "concat" => {
                let (a, b) = (args[0], args[1]);
                let ValueId::Literal(jlit) = index else {
                    return None;
                };
                let j = view.shared().values.literals[jlit].value;
                let a_ty = view.type_of(a);
                let (_, len_a) = view.shared().types.array_of(a_ty)?;
                if j < len_a as u64 {
                    Some(forward(view, out_size, a, index))
                } else {
                    let shifted = view.shared().get_const(j - len_a as u64, 8);
                    Some(forward(view, out_size, b, shifted))
                }
            }
            _ => None,
        }
    }
}

/// Simplify `at(base, index)` one more level, falling back to the verbatim
/// re-issued `at` expression when no further constructor forwarding applies. So
/// a chain like `at(concat(singleton(v), s), 0)` collapses all the way to `v`
/// rather than stalling at `at(singleton(v), 0)`.
fn forward(view: BodyView<'_, '_>, out_size: usize, base: ValueId, index: ValueId) -> Simplified {
    simplify_at(view, out_size, base, index).unwrap_or_else(|| {
        Simplified::Expression(Mnemonic::Intrinsic(crate::value::insn::IntrinsicApp {
            id: IntrinsicId::from_name("at").unwrap(),
            // The expression's operands are all in `base`/`index`'s own body; strip
            // the func to store them bare-local (no ambient func needed here).
            args: vec![base.strip_func(), index.strip_func()],
        }))
    })
}

register_intrinsic!(At);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::Builder;
    use crate::context::Context;
    use crate::value::insn::IntrinsicId;
    use crate::value::{BasicBlock, BodyView, FunctionId, LocalValueId, ValueId};

    fn body_view<'ctx, 'str: 'ctx>(
        ctx: &'ctx Context<'str>,
        function: FunctionId,
    ) -> BodyView<'ctx, 'str> {
        BodyView::new(&ctx.bodies[function], &ctx.shared, &ctx.interfaces)
    }

    fn at_id() -> IntrinsicId {
        IntrinsicId::from_name("at").unwrap()
    }

    #[test]
    fn result_type_is_element_type() {
        let mut types = TypeManager::default();
        let i32 = types.get_or_make_int(4);
        let arr = types.get_or_make_array(i32, 5);
        let i64 = types.get_or_make_int(8);
        assert_eq!(at_id().desc().result_type(&types, &[arr, i64]), i32);
    }

    /// `at(insert(a, i, v), i) = v`.
    #[test]
    fn at_forwards_same_index() {
        let mut ctx = Context::new();
        let i32 = ctx.shared.types.get_or_make_int(4);
        let arr_ty = ctx.shared.types.get_or_make_array(i32, 4);
        let blk = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let a = BasicBlock::from_id_mut(&mut ctx, blk).push_param(16).id;
        ctx.block_param_mut(a).type_id = arr_ty;
        let i = ctx.get_const(2, 8).id();
        let v = ctx.get_const(0x77, 4).id();
        let insert_id = IntrinsicId::from_name("insert").unwrap();
        let ins = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, blk));
            b.push_intrinsic(insert_id, vec![ValueId::BlockParam(a), i, v])
                .id()
        };
        match at_id()
            .desc()
            .simplify(body_view(&ctx, blk.func), at_id(), 4, &[ins, i])
        {
            Some(Simplified::Value(got)) => assert_eq!(got, v),
            other => panic!("expected v, got {other:?}"),
        }
    }

    /// `at(insert(a, i, v), j) = at(a, j)` for provably distinct constant indices.
    #[test]
    fn at_bypasses_distinct_index() {
        let mut ctx = Context::new();
        let i32 = ctx.shared.types.get_or_make_int(4);
        let arr_ty = ctx.shared.types.get_or_make_array(i32, 4);
        let blk = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let a = BasicBlock::from_id_mut(&mut ctx, blk).push_param(16).id;
        ctx.block_param_mut(a).type_id = arr_ty;
        let i = ctx.get_const(2, 8).id();
        let j = ctx.get_const(3, 8).id();
        let v = ctx.get_const(0x77, 4).id();
        let insert_id = IntrinsicId::from_name("insert").unwrap();
        let ins = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, blk));
            b.push_intrinsic(insert_id, vec![ValueId::BlockParam(a), i, v])
                .id()
        };
        match at_id()
            .desc()
            .simplify(body_view(&ctx, blk.func), at_id(), 4, &[ins, j])
        {
            Some(Simplified::Expression(Mnemonic::Intrinsic(app))) => {
                assert_eq!(app.id.name(), "at");
                assert_eq!(
                    app.args,
                    vec![ValueId::BlockParam(a).strip_func(), j.strip_func()]
                );
            }
            other => panic!("expected at(a, j), got {other:?}"),
        }
    }

    /// `at(singleton(v), _) = v` regardless of the index.
    #[test]
    fn at_forwards_through_singleton() {
        let mut ctx = Context::new();
        let v = ctx.get_const(0x99, 4).id();
        let sing_id = IntrinsicId::from_name("singleton").unwrap();
        let blk = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let sing = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, blk));
            b.push_intrinsic(sing_id, vec![v]).id()
        };
        let idx = ctx.get_const(0, 8).id();
        match at_id()
            .desc()
            .simplify(body_view(&ctx, blk.func), at_id(), 4, &[sing, idx])
        {
            Some(Simplified::Value(got)) => assert_eq!(got, v),
            other => panic!("expected v, got {other:?}"),
        }
    }

    /// `at(concat(a, b), j)` picks the side `j` falls in, shifting the index into
    /// the right operand.
    #[test]
    fn at_picks_concat_side() {
        let mut ctx = Context::new();
        let i32 = ctx.shared.types.get_or_make_int(4);
        let a_ty = ctx.shared.types.get_or_make_array(i32, 1);
        let b_ty = ctx.shared.types.get_or_make_array(i32, 3);
        let blk = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let a = BasicBlock::from_id_mut(&mut ctx, blk).push_param(4).id;
        ctx.block_param_mut(a).type_id = a_ty;
        let b = BasicBlock::from_id_mut(&mut ctx, blk).push_param(12).id;
        ctx.block_param_mut(b).type_id = b_ty;
        let concat_id = IntrinsicId::from_name("concat").unwrap();
        let cat = {
            let mut bl = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, blk));
            bl.push_intrinsic(
                concat_id,
                vec![ValueId::BlockParam(a), ValueId::BlockParam(b)],
            )
            .id()
        };
        // Lane 0 → left operand `a`, same index.
        let j0 = ctx.get_const(0, 8).id();
        match at_id()
            .desc()
            .simplify(body_view(&ctx, blk.func), at_id(), 4, &[cat, j0])
        {
            Some(Simplified::Expression(Mnemonic::Intrinsic(app))) => {
                assert_eq!(app.id.name(), "at");
                assert_eq!(
                    app.args,
                    vec![ValueId::BlockParam(a).strip_func(), j0.strip_func()]
                );
            }
            other => panic!("expected at(a, 0), got {other:?}"),
        }
        // Lane 2 → right operand `b`, index shifted by len(a)=1 → 1.
        let j2 = ctx.get_const(2, 8).id();
        match at_id()
            .desc()
            .simplify(body_view(&ctx, blk.func), at_id(), 4, &[cat, j2])
        {
            Some(Simplified::Expression(Mnemonic::Intrinsic(app))) => {
                assert_eq!(app.id.name(), "at");
                let LocalValueId::Literal(l) = app.args[1] else {
                    panic!("expected literal shifted index");
                };
                assert_eq!(app.args[0], ValueId::BlockParam(b).strip_func());
                assert_eq!(ctx.shared.values.literals[l].value, 1);
            }
            other => panic!("expected at(b, 1), got {other:?}"),
        }
    }

    #[test]
    fn at_reads_constant_bytes() {
        let mut ctx = Context::new();
        let function = ctx.anon_function();
        let i32 = ctx.shared.types.get_or_make_int(4);
        let arr_ty = ctx.shared.types.get_or_make_array(i32, 3);
        let mut data = Vec::new();
        for w in [0x11u32, 0x22, 0x33] {
            data.extend_from_slice(&w.to_le_bytes());
        }
        let bid = ctx.get_bytes(data).id();
        if let ValueId::Bytes(b) = bid {
            ctx.shared.values.bytes[b].type_id = arr_ty;
        }
        let idx = ctx.get_const(2, 8).id();
        match at_id()
            .desc()
            .simplify(body_view(&ctx, function), at_id(), 4, &[bid, idx])
        {
            Some(Simplified::Value(ValueId::Literal(l))) => {
                assert_eq!(ctx.shared.values.literals[l].value, 0x33);
            }
            other => panic!("expected literal 0x33, got {other:?}"),
        }
    }
}
