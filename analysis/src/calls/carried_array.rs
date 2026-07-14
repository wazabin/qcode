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
    types::TypeId,
    value::{
        BlockId, BlockRef, FunctionId, FunctionRef, InstructionRef, QCodeView, ValueId,
        insn::{InstructionId, IntrinsicApp, IntrinsicId, Mnemonic},
    },
};

use crate::loop_info::{incoming, is_decrement, literal, param_parent, param_pos};

/// A loop-carried array threaded through a loop header:
/// header param `arr_h : [elem_ty; count]` whose incomings are exactly one
/// carry insert (base is a block param) and at most one seed insert
/// `insert(arr0, 0, seed_val)` into a fresh non-param array.
pub(crate) struct CarriedArray {
    pub header: BlockId,
    /// The body block (parent of the carry insert's *base* array param) — always
    /// a predecessor of `header` (the back-edge).
    pub body: BlockId,
    pub arr_h: ValueId, // header array param
    pub arr_b: ValueId, // the carry insert's base array param (body view)
    /// The carry insert's index — the induction value **as the body uses it**: a
    /// param of `body` (body-copied) or of `header` itself (header-carried, the
    /// shape redundant-φ elimination leaves). Consumers compare it by `ValueId`
    /// and must not assume a defining block.
    pub index: ValueId,
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
pub(crate) fn find_carried_array<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    fid: FunctionId,
) -> Option<CarriedArray> {
    let insert_id = IntrinsicId::from_name("insert")?;

    let mut found: Option<CarriedArray> = None;
    let blocks: Vec<BlockId> = FunctionRef::new(host, fid).iter().map(|b| b.id).collect();
    for header in blocks {
        let params: Vec<ValueId> = BlockRef::new(host, header)
            .params()
            .map(|p| p.id())
            .collect();
        for arr_h in params {
            let arr_ty = host.type_of(arr_h);
            let Some((elem_ty, count)) = host.shared().types.array_of(arr_ty) else {
                continue;
            };
            if count == 0 {
                continue;
            }
            let Some(k) = param_pos(host, header, arr_h) else {
                continue;
            };
            let incs = incoming(host, header, k);
            if incs.len() != 2 {
                continue;
            }

            // Classify the two incomings: exactly one carry, and at most one of
            // {seed, init}. Anything else means this param is not a carried
            // array of the canonical shape.
            // Carry: `(arr_b, val, idx, insert_block)`. The insert's own parent
            // block is the loop body — recorded here rather than derived from
            // `param_parent(arr_b)`, because redundant-φ elimination may have
            // collapsed the minted body array param onto the header param, making
            // `arr_b` a *header* param even though the insert lives in the body.
            let mut carry: Option<(ValueId, ValueId, ValueId, BlockId)> = None;
            let mut seed: Option<(ValueId, ValueId)> = None; // (seed_val, arr0)
            let mut init: Option<ValueId> = None;
            let mut ok = true;
            for &v in &incs {
                // `insert(arr0, idx, val)` splits into carry (param base) and
                // lane-0 seed (non-param base); every other array value is init.
                if let ValueId::Instruction(iid) = v
                    && let Mnemonic::Intrinsic(IntrinsicApp { id, args }) =
                        host.instruction(iid).mnemonic()
                    && *id == insert_id
                    && args.len() == 3
                {
                    let [arr0, idx, val] = args[..] else {
                        ok = false;
                        break;
                    };
                    // Operands are stored bare-local; qualify with the insert's
                    // own function (== this pass's ambient function).
                    let (arr0, idx, val) = (
                        arr0.qualify(iid.func),
                        idx.qualify(iid.func),
                        val.qualify(iid.func),
                    );
                    if let ValueId::BlockParam(_) = arr0 {
                        if carry.is_some() {
                            ok = false;
                            break;
                        }
                        let Some(insert_block) =
                            InstructionRef::new(host, iid).parent().map(|b| b.id)
                        else {
                            ok = false;
                            break;
                        };
                        carry = Some((arr0, val, idx, insert_block));
                    } else if literal(host, idx) == Some(0) {
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
            let Some((arr_b, stored_val, index, body)) = carry else {
                continue;
            };
            // Exactly one of {seed, init}.
            if seed.is_some() == init.is_some() {
                continue;
            }
            // The carry base and index are the array/induction as the body uses
            // them: each a param of the body itself (body-copied) or of this
            // carry's header (header-carried — the shape redundant-φ elimination
            // leaves). Consumers compare both by `ValueId`, never by parent.
            let param_of_loop = |v: ValueId| {
                let p = param_parent(host, v);
                p == Some(body) || p == Some(header)
            };
            if !param_of_loop(arr_b) {
                continue;
            }
            if !matches!(index, ValueId::BlockParam(_)) || !param_of_loop(index) {
                continue;
            }
            // The body must be a predecessor of the header (the back-edge).
            let is_back_edge = BlockRef::new(host, header)
                .predecessors()
                .any(|(_, p)| p == body);
            if !is_back_edge {
                continue;
            }
            if param_parent(host, arr_h) != Some(header) {
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
pub(crate) fn classify_body_reads<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    ca: &CarriedArray,
) -> Option<BodyReads> {
    let at_id = IntrinsicId::from_name("at")?;
    let arr_b_ats: Vec<(InstructionId, ValueId)> = BlockRef::new(host, ca.body)
        .iter()
        .filter_map(|insn| match insn.mnemonic() {
            Mnemonic::Intrinsic(IntrinsicApp { id, args })
                if *id == at_id && args.len() == 2 && args[0].qualify(insn.id.func) == ca.arr_b =>
            {
                Some((insn.id, args[1].qualify(insn.id.func)))
            }
            _ => None,
        })
        .collect();
    let idx_ty = host.type_of(ca.index);
    let idx_width = host.shared().types.size_of(idx_ty);
    let mut prev = None;
    let mut own = None;
    for (id, e_idx) in arr_b_ats {
        if is_decrement(host, e_idx, ca.index, idx_width) {
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
pub(crate) fn exit_view<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    ca: &CarriedArray,
    exit: BlockId,
) -> ValueId {
    let exit_params: Vec<ValueId> = BlockRef::new(host, exit).params().map(|p| p.id()).collect();
    for p in exit_params {
        if let Some(kp) = param_pos(host, exit, p)
            && incoming(host, exit, kp)[..] == [ca.arr_h]
        {
            return p;
        }
    }
    ca.arr_h
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use qcode::{
        context::Context,
        value::{BasicBlock, FunctionBody},
    };

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
        let view = qcode::value::ModuleView::new(&ctx);
        let ca = find_carried_array(view, prefix).expect("carried array found");
        assert!(ca.seed.is_some(), "prefix sum is the seeded scan shape");
        assert!(ca.init.is_none());
        let reads = classify_body_reads(view, &ca).expect("body reads classify");
        assert!(reads.prev.is_some(), "carry read at(arr, j-1)");
        assert!(reads.own.is_some(), "own-lane read at(arr, j)");
    }

    /// The header-carried source of the promoted xor fill: the body reads the
    /// header induction param `@i` directly (no body index param — the shape
    /// redundant-φ elimination leaves). `array_promote` still threads the array
    /// through a fresh *body* param, so only the carry insert's index is
    /// header-carried; the matcher must accept it.
    fn promote_header_carried_fill(mut ctx: &mut Context) -> FunctionId {
        qcode!(
            ctx,
            "
            fn xorbuf:
            <entry @base:i64>
                goto <head @i=0>;
            <head @i:i64>
                %done = @i == 16;
                if %done goto <exit> else goto <body>;
            <body>
                %off = @i * 4;
                %addr = @base + %off;
                %x = load(ram:4, %addr);
                %v = %x ^ 0x5a;
                store(ram:4, %addr <- %v);
                %i1 = @i + 1;
                goto <head @i=%i1>;
            <exit>
                return at i64 0x0;
            "
        );
        assert!(
            run_function_pass::<ArrayPromote>(ctx, xorbuf).unwrap(),
            "the header-carried fill should promote"
        );
        xorbuf
    }

    #[test]
    fn header_carried_index_accepted() {
        let mut ctx = Context::new();
        let f = promote_header_carried_fill(&mut ctx);
        let view = qcode::value::ModuleView::new(&ctx);
        let ca = find_carried_array(view, f).expect("header-carried index accepted");
        assert_ne!(ca.header, ca.body, "split shape");
        assert_eq!(
            param_parent(qcode::value::ModuleView::new(&ctx), ca.index),
            Some(ca.header),
            "the index is the header induction param itself"
        );
        let reads = classify_body_reads(view, &ca).expect("body reads classify");
        assert!(reads.prev.is_none(), "no accumulator carry read");
        assert!(reads.own.is_some(), "own-lane read at(arr, i)");
    }

    /// An index param belonging to neither the body nor the carried header (here
    /// a function-entry param) is not a loop induction — the relaxed rule must
    /// still reject it.
    #[test]
    fn foreign_block_index_rejected() {
        let mut ctx = Context::new();
        let f = promote_header_carried_fill(&mut ctx);
        let ca = find_carried_array(qcode::value::ModuleView::new(&ctx), f)
            .expect("promoted shape matches");
        // Rewrite the carry insert's index to a fresh param of the entry block.
        let insert_id = IntrinsicId::from_name("insert").unwrap();
        let carry = BasicBlock::from_id(&ctx, ca.body)
            .iter()
            .find_map(|i| match i.mnemonic() {
                Mnemonic::Intrinsic(IntrinsicApp { id, args })
                    if *id == insert_id && args[0] == ca.arr_b.localize(i.id.func) =>
                {
                    Some(i.id)
                }
                _ => None,
            })
            .expect("carry insert present");
        let entry = FunctionBody::from_id(&ctx, f).root().unwrap().id;
        let foreign =
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut ctx, entry).push_param(8).id);
        let mut m = ctx.get_insn(carry).mnemonic().clone();
        let Mnemonic::Intrinsic(IntrinsicApp { args, .. }) = &mut m else {
            unreachable!("carry is an intrinsic");
        };
        args[1] = foreign.localize(carry.func);
        ctx.replace_instruction_mnemonic(carry, m);
        assert!(
            find_carried_array(qcode::value::ModuleView::new(&ctx), f).is_none(),
            "a foreign-block index param must not match"
        );
    }

    /// The rotated (do-while) shape — one self-looping block whose params carry
    /// both the array and the index — matches exactly as before the relaxation
    /// (`body == header`, so both index arms coincide).
    #[test]
    fn rotated_carry_unchanged() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn reg_rot:
            <entry @seed:i64 @base:i64>
                %e0 = trunc(i32, @seed);
                store(ram:4, @base <- %e0);
                goto <body @j=1 @b=@base @a=%e0>;
            <body @j:i64 @b:i64 @a:i32>
                %m = @a * 3;
                %jt = trunc(i32, @j);
                %next = %m + %jt;
                %woff = @j * 4;
                %waddr = @b + %woff;
                store(ram:4, %waddr <- %next);
                %j1 = @j + 1;
                %done = %j1 < 624;
                if %done goto <body @j=%j1 @b=@b @a=%next> else goto <exit>;
            <exit>
                return at i64 0x0;
            "
        );
        assert!(run_function_pass::<ArrayPromote>(&mut ctx, reg_rot).unwrap());
        let ca = find_carried_array(qcode::value::ModuleView::new(&ctx), reg_rot)
            .expect("rotated carry matches");
        assert_eq!(ca.header, ca.body, "rotated: the body is its own header");
        assert_eq!(
            param_parent(qcode::value::ModuleView::new(&ctx), ca.index),
            Some(ca.body)
        );
    }
}
