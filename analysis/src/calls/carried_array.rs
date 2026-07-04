//! Shared structural matcher for the **single-carried-array** loop form that
//! [`array_promote`](crate::mem::array_promote) produces.
//!
//! Both [`loop_to_scan`](super::loop_to_scan) and
//! [`loop_to_map`](super::loop_to_map) anchor on the same shape: an array-typed
//! header param `@arr:[elem;N]` threaded through the loop, whose two incomings
//! are one **carry** insert (`insert(arr_param, index, val)`) on the back-edge
//! and, on the preheader edge, either a lane-0 **seed** insert
//! (`insert(arr0, 0, seed_val)`, the scan shape) or a plain **init** array
//! (splat / wide `Load` / zero literal, the map shape). This module owns finding
//! that carried array and classifying its parts; the recognizers layer their
//! own semantics on top.

use qcode::{
    context::Context,
    types::TypeId,
    value::{
        BasicBlock, BlockId, Function, FunctionId, ValueId, ValueRef,
        insn::{Binary, Binop, InstructionId, IntBinop, IntrinsicApp, IntrinsicId, Mnemonic},
    },
};

/// `c` if `v` is the integer literal `c`, else `None`.
pub(crate) fn literal(ctx: &Context, v: ValueId) -> Option<u64> {
    match ValueRef::new(v, ctx) {
        ValueRef::Literal(l) => Some(l.value()),
        _ => None,
    }
}

/// `true` if `v` is `idx + 1` (either operand order) — a unit step of `idx`.
pub(crate) fn is_increment(ctx: &Context, v: ValueId, idx: ValueId) -> bool {
    let ValueId::Instruction(id) = v else {
        return false;
    };
    let Mnemonic::Binop(Binary { lhs, rhs, op }) = ctx.get_insn(id).mnemonic() else {
        return false;
    };
    let one = |x: ValueId| literal(ctx, x) == Some(1);
    matches!(op, Binop::Int(IntBinop::Add))
        && ((*lhs == idx && one(*rhs)) || (*rhs == idx && one(*lhs)))
}

/// `true` if `v` is `idx - 1`, expressed either as `idx - 1` or as `idx + (-1)`
/// (the wrapping representation `array_promote` emits for the `at` back-index).
pub(crate) fn is_decrement(ctx: &mut Context, v: ValueId, idx: ValueId) -> bool {
    let ValueId::Instruction(id) = v else {
        return false;
    };
    let Mnemonic::Binop(Binary { lhs, rhs, op }) = ctx.get_insn(id).mnemonic() else {
        return false;
    };
    let (lhs, rhs, op) = (*lhs, *rhs, *op);
    let idx_ty = ctx.type_of(idx);
    let width = ctx.types.size_of(idx_ty);
    let neg_one = if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    };
    match op {
        Binop::Int(IntBinop::Sub) => lhs == idx && literal(ctx, rhs) == Some(1),
        Binop::Int(IntBinop::Add) => {
            (lhs == idx && literal(ctx, rhs) == Some(neg_one))
                || (rhs == idx && literal(ctx, lhs) == Some(neg_one))
        }
        _ => false,
    }
}

/// Values feeding block-param index `k` of `block` from every predecessor edge.
pub(crate) fn incoming(ctx: &Context, block: BlockId, k: usize) -> Vec<ValueId> {
    let mut out = Vec::new();
    let preds: Vec<BlockId> = BasicBlock::from_id(ctx, block)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    for pred in preds {
        let Some(term) = BasicBlock::from_id(ctx, pred).iter().last() else {
            continue;
        };
        match term.mnemonic() {
            Mnemonic::Branch(b) => out.extend(b.args.get(k).copied()),
            Mnemonic::CBranch(cb) => {
                if cb.success_block == block {
                    out.extend(cb.success_args.get(k).copied());
                }
                if cb.failure_block == block {
                    out.extend(cb.failure_args.get(k).copied());
                }
            }
            _ => {}
        }
    }
    out
}

/// Index of block-param `p` within `block`'s parameter list.
pub(crate) fn param_pos(ctx: &Context, block: BlockId, p: ValueId) -> Option<usize> {
    BasicBlock::from_id(ctx, block)
        .params()
        .position(|q| q.id() == p)
}

/// Parent block of a block-param value.
pub(crate) fn param_parent(ctx: &Context, v: ValueId) -> Option<BlockId> {
    let ValueId::BlockParam(pid) = v else {
        return None;
    };
    ctx.values.block_params[pid].parent
}

/// A loop-carried array threaded through a loop header:
/// header param `arr_h : [elem_ty; count]` whose incomings are exactly one
/// carry insert (base is a block param) and at most one seed insert
/// `insert(arr0, 0, seed_val)` into a fresh non-param array.
pub(crate) struct CarriedArray {
    pub header: BlockId,
    /// The body block (parent of the carry insert's index param) — always a
    /// predecessor of `header` (the back-edge).
    pub body: BlockId,
    pub arr_h: ValueId,      // header array param
    pub arr_b: ValueId,      // the carry insert's base array param (body view)
    pub index: ValueId,      // the carry insert's index (body block param)
    pub stored_val: ValueId, // the carry insert's stored value
    /// `Some((seed_val, arr0))` when a lane-0 seed insert exists (scan shape);
    /// `None` for a carry whose init incoming is a plain array value
    /// (splat / wide Load — the seedless map shape).
    pub seed: Option<(ValueId, ValueId)>,
    /// The non-insert init incoming when `seed` is None (the map shape's
    /// initial array), else None. Consumed by `loop_to_map` (step 2).
    #[allow(dead_code)]
    pub init: Option<ValueId>,
    pub elem_ty: TypeId,
    pub count: usize,
}

/// Find the unique carried array in `fid`. Returns None when there is no
/// carried array OR more than one (not the canonical shape).
pub(crate) fn find_carried_array(ctx: &mut Context, fid: FunctionId) -> Option<CarriedArray> {
    let insert_id = IntrinsicId::from_name("insert")?;

    let mut found: Option<CarriedArray> = None;
    let blocks: Vec<BlockId> = Function::from_id(ctx, fid).iter().map(|b| b.id).collect();
    for header in blocks {
        let params: Vec<ValueId> = BasicBlock::from_id(ctx, header)
            .params()
            .map(|p| p.id())
            .collect();
        for arr_h in params {
            let arr_ty = ctx.type_of(arr_h);
            let Some((elem_ty, count)) = ctx.types.array_of(arr_ty) else {
                continue;
            };
            if count == 0 {
                continue;
            }
            let Some(k) = param_pos(ctx, header, arr_h) else {
                continue;
            };
            let incs = incoming(ctx, header, k);
            if incs.len() != 2 {
                continue;
            }

            // Classify the two incomings: exactly one carry, and at most one of
            // {seed, init}. Anything else means this param is not a carried
            // array of the canonical shape.
            let mut carry: Option<(ValueId, ValueId, ValueId)> = None; // (arr_b, val, idx)
            let mut seed: Option<(ValueId, ValueId)> = None; // (seed_val, arr0)
            let mut init: Option<ValueId> = None;
            let mut ok = true;
            for &v in &incs {
                // `insert(arr0, idx, val)` splits into carry (param base) and
                // lane-0 seed (non-param base); every other array value is init.
                if let ValueId::Instruction(iid) = v {
                    if let Mnemonic::Intrinsic(IntrinsicApp { id, args }) =
                        ctx.get_insn(iid).mnemonic()
                    {
                        if *id == insert_id && args.len() == 3 {
                            let [arr0, idx, val] = args[..] else {
                                ok = false;
                                break;
                            };
                            if let ValueId::BlockParam(_) = arr0 {
                                if carry.is_some() {
                                    ok = false;
                                    break;
                                }
                                carry = Some((arr0, val, idx));
                            } else if literal(ctx, idx) == Some(0) {
                                if seed.is_some() || init.is_some() {
                                    ok = false;
                                    break;
                                }
                                seed = Some((val, arr0));
                            } else {
                                ok = false;
                                break;
                            }
                            continue;
                        }
                    }
                }
                // A non-insert array value (splat / wide Load / zero literal).
                if seed.is_some() || init.is_some() {
                    ok = false;
                    break;
                }
                init = Some(v);
            }
            if !ok {
                continue;
            }
            let Some((arr_b, stored_val, index)) = carry else {
                continue;
            };
            // Exactly one of {seed, init}.
            if seed.is_some() == init.is_some() {
                continue;
            }
            // The carry's index must be a block param of `arr_b`'s parent — the
            // loop body.
            let Some(body) = param_parent(ctx, arr_b) else {
                continue;
            };
            if !matches!(index, ValueId::BlockParam(_)) || param_parent(ctx, index) != Some(body) {
                continue;
            }
            // The body must be a predecessor of the header (the back-edge).
            let is_back_edge = BasicBlock::from_id(ctx, header)
                .predecessors()
                .any(|(_, p)| p == body);
            if !is_back_edge {
                continue;
            }
            if param_parent(ctx, arr_h) != Some(header) {
                continue;
            }

            if found.is_some() {
                return None; // more than one carried array — not the canonical shape
            }
            found = Some(CarriedArray {
                header,
                body,
                arr_h,
                arr_b,
                index,
                stored_val,
                seed,
                init,
                elem_ty,
                count,
            });
        }
    }
    found
}

/// The `at(arr_b, ·)` reads inside `body`, classified by index.
pub(crate) struct BodyReads {
    /// `at(arr_b, index-1)` — the accumulator carry read (scan only).
    pub prev: Option<InstructionId>,
    /// `at(arr_b, index)` — the own-lane original read.
    pub own: Option<ValueId>,
}

/// Classify the body's `at(arr_b, ·)` reads on the single carried array.
/// `None` if any `at(arr_b, ·)` has an unrecognized index, or a class repeats.
pub(crate) fn classify_body_reads(ctx: &mut Context, ca: &CarriedArray) -> Option<BodyReads> {
    let at_id = IntrinsicId::from_name("at")?;
    let arr_b_ats: Vec<(InstructionId, ValueId)> = BasicBlock::from_id(ctx, ca.body)
        .iter()
        .filter_map(|insn| match insn.mnemonic() {
            Mnemonic::Intrinsic(IntrinsicApp { id, args })
                if *id == at_id && args.len() == 2 && args[0] == ca.arr_b =>
            {
                Some((insn.id, args[1]))
            }
            _ => None,
        })
        .collect();
    let mut prev = None;
    let mut own = None;
    for (id, e_idx) in arr_b_ats {
        if is_decrement(ctx, e_idx, ca.index) {
            if prev.is_some() {
                return None;
            }
            prev = Some(id);
        } else if e_idx == ca.index {
            if own.is_some() {
                return None;
            }
            own = Some(ValueId::Instruction(id));
        } else {
            return None;
        }
    }
    Some(BodyReads { prev, own })
}

/// The value the exit block sees for the carried array: an exit pass-through
/// param copying `arr_h` on the header→exit edge, or `arr_h` itself when gvn
/// coalesced the pass-through.
pub(crate) fn exit_view(ctx: &Context, ca: &CarriedArray, exit: BlockId) -> ValueId {
    let exit_params: Vec<ValueId> = BasicBlock::from_id(ctx, exit)
        .params()
        .map(|p| p.id())
        .collect();
    for p in exit_params {
        if let Some(kp) = param_pos(ctx, exit, p) {
            if incoming(ctx, exit, kp)[..] == [ca.arr_h] {
                return p;
            }
        }
    }
    ca.arr_h
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use crate::mem::array_promote::ArrayPromote;
    use crate::test_util::run_function_pass;

    // Reuse ArrayPromote to *produce* the promoted single-array shape from the
    // prefix-sum IR, so the matcher is tested against the exact IR array_promote
    // emits rather than a hand-written approximation.
    #[test]
    fn find_carried_array_classifies_seed_and_carry() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn prefix:
            <entry @seed:i64 @base:i64>
                %e0 = @seed[0:4];
                store(ram:4, @base <- %e0);
                goto <head @i=1 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 624;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %jm1 = @j - 1;
                %roff = %jm1 * 4;
                %raddr = @b + %roff;
                %prev = load(ram:4, %raddr);
                %coff = @j * 4;
                %caddr = @b + %coff;
                %cur = load(ram:4, %caddr);
                %next = %cur + %prev;
                store(ram:4, %caddr <- %next);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        assert!(
            run_function_pass::<ArrayPromote>(&mut ctx, prefix).unwrap(),
            "array_promote should promote the prefix sum"
        );
        let ca = find_carried_array(&mut ctx, prefix).expect("carried array found");
        assert!(ca.seed.is_some(), "prefix sum is the seeded scan shape");
        assert!(ca.init.is_none());
        let reads = classify_body_reads(&mut ctx, &ca).expect("body reads classify");
        assert!(reads.prev.is_some(), "carry read at(arr, j-1)");
        assert!(reads.own.is_some(), "own-lane read at(arr, j)");
    }
}
