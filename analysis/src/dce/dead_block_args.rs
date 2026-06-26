//! Redundant block-argument elimination.
//!
//! A block parameter `p` is *redundant* when, across every predecessor edge, the
//! argument bound to it is either a single common value `v0` or `p` itself. Then
//! `p` is congruent to `v0` on every path that reaches the block, so `p` can be
//! replaced by `v0` everywhere and dropped, along with the matching positional
//! argument at each predecessor terminator.
//!
//! The canonical case is a loop-invariant value threaded around a back-edge: the
//! header param gets `v0` from the off-loop edge and itself (`p = p`) from the
//! latch edge. Since `v0` is the only non-self value and it dominates every
//! predecessor (it is used on each), it dominates the header too, so the
//! substitution is sound. This is exactly redundant-φ elimination.
//!
//! Removing one redundant param can expose another — e.g. two params that only
//! forward each other around a loop (`p ← {v0, q}`, `q ← {p}`): `q` collapses to
//! `p` first, which turns `p`'s latch argument into `p = p`, leaving `v0` as its
//! only non-self source. The sweep therefore iterates to a fixpoint.
//!
//! Unlike [`remove_unused_no_pred_block_params`](super::remove_unused_no_pred_block_params),
//! which drops zero-user params on blocks with *no* incoming edge, this works on
//! join/loop blocks by reconciling their incoming arguments. The function entry
//! (`root`) is intentionally left alone: its params are the function's interface
//! and must be removed through `remove_entry_param` to keep the ABI aligned.

use rustc_hash::FxHashSet as HashSet;

use qcode::{
    context::Context,
    value::{
        BasicBlock, BlockId, BlockParamId, ValueId,
        insn::{Branch, CBranch, Mnemonic},
    },
};

/// Collect the distinct values bound to position `index` of `block` across all
/// predecessor edges, excluding the param's own value `self_val` (self-edges
/// carry no new information).
///
/// Returns `Some(v0)` when exactly one such value remains — the replacement the
/// param collapses to — and `None` when the param is a genuine merge of two or
/// more distinct values (not redundant) or has no non-self incoming value.
fn unique_incoming(
    ctx: &Context,
    block: BlockId,
    index: usize,
    self_val: ValueId,
) -> Option<ValueId> {
    // Dedup predecessor blocks: a `CBranch` whose two edges both target `block`
    // shows up twice but its terminator is read once below (handling both arms).
    let preds: HashSet<BlockId> = BasicBlock::from_id(ctx, block)
        .predecessors()
        .map(|(_, b)| b)
        .collect();

    let mut found: Option<ValueId> = None;
    for pred in preds {
        let Some(&term_id) = ctx.values.basic_blocks[pred].instructions.last() else {
            continue;
        };
        let mut consider = |arg: ValueId| -> bool {
            if arg == self_val {
                return true;
            }
            match found {
                Some(v) if v == arg => true,
                Some(_) => false, // a second distinct value: not redundant
                None => {
                    found = Some(arg);
                    true
                }
            }
        };

        let ok = match ctx.get_insn(term_id).mnemonic() {
            Mnemonic::Branch(b) if b.target == block => {
                b.args.get(index).copied().is_some_and(&mut consider)
            }
            Mnemonic::CBranch(c) => {
                let mut ok = true;
                if c.success_block == block {
                    ok &= c
                        .success_args
                        .get(index)
                        .copied()
                        .is_some_and(&mut consider);
                }
                if c.failure_block == block {
                    ok &= c
                        .failure_args
                        .get(index)
                        .copied()
                        .is_some_and(&mut consider);
                }
                ok
            }
            // An indirect terminator carries no per-target argument list, so a
            // block reached that way has no incoming value to reconcile.
            _ => return None,
        };
        if !ok {
            return None;
        }
    }

    found
}

/// Replace every block parameter across `block_ids` that is redundant — bound to
/// a single common value (or itself) on all incoming edges — with that value,
/// then drop the param and its predecessor arguments. Iterates to a fixpoint so
/// chains and cycles collapse. Leaves `root`'s params untouched.
///
/// Returns whether anything was removed.
pub fn remove_dead_block_args(
    ctx: &mut Context,
    block_ids: &[BlockId],
    root: Option<BlockId>,
) -> bool {
    let mut changed = false;
    loop {
        let Some((block, index, param, repl)) = find_redundant_param(ctx, block_ids, root) else {
            break;
        };

        // `p ≡ repl`: rewrite every use, then strip the param and the now-removed
        // column of arguments from each predecessor.
        ctx.replace_all_uses_with(ValueId::BlockParam(param), repl);
        remove_params_from_block(ctx, block, &HashSet::from_iter([index]));
        changed = true;
    }
    changed
}

/// Scan for the first redundant param: a non-root, non-protected param on a block
/// with predecessors whose incoming arguments reduce to a single value `repl`.
fn find_redundant_param(
    ctx: &Context,
    block_ids: &[BlockId],
    root: Option<BlockId>,
) -> Option<(BlockId, usize, BlockParamId, ValueId)> {
    for &block in block_ids {
        if Some(block) == root {
            continue;
        }
        if BasicBlock::from_id(ctx, block)
            .predecessors()
            .next()
            .is_none()
        {
            continue;
        }
        let params = ctx.values.basic_blocks[block].params.clone();
        for (index, &param) in params.iter().enumerate() {
            if ctx.values.block_params[param].protected {
                continue;
            }
            if let Some(repl) = unique_incoming(ctx, block, index, ValueId::BlockParam(param)) {
                return Some((block, index, param, repl));
            }
        }
    }
    None
}

/// Drop the params at `dead_indices` from `block`, reindexing the survivors, and
/// strip the matching positional argument from every predecessor terminator.
fn remove_params_from_block(ctx: &mut Context, block: BlockId, dead_indices: &HashSet<usize>) {
    let params = ctx.values.basic_blocks[block].params.clone();
    let mut kept = Vec::with_capacity(params.len());
    for (i, &p) in params.iter().enumerate() {
        if dead_indices.contains(&i) {
            ctx.values.block_params[p].parent = None;
        } else {
            ctx.values.block_params[p].index = kept.len();
            kept.push(p);
        }
    }
    ctx.values.basic_blocks[block].params = kept;

    // A predecessor reaching `block` through both edges of a `CBranch` appears
    // twice; dedup so we rewrite its terminator exactly once.
    let preds: HashSet<BlockId> = BasicBlock::from_id(ctx, block)
        .predecessors()
        .map(|(_, b)| b)
        .collect();

    for pred in preds {
        let Some(&term_id) = ctx.values.basic_blocks[pred].instructions.last() else {
            continue;
        };
        let new = match ctx.get_insn(term_id).mnemonic().clone() {
            Mnemonic::Branch(b) if b.target == block => Mnemonic::Branch(Branch {
                target: b.target,
                args: filter_kept(&b.args, dead_indices),
            }),
            Mnemonic::CBranch(c) => Mnemonic::CBranch(CBranch {
                condition: c.condition,
                success_block: c.success_block,
                success_args: if c.success_block == block {
                    filter_kept(&c.success_args, dead_indices)
                } else {
                    c.success_args
                },
                failure_block: c.failure_block,
                failure_args: if c.failure_block == block {
                    filter_kept(&c.failure_args, dead_indices)
                } else {
                    c.failure_args
                },
            }),
            // Indirect terminators carry no per-target argument list, so a block
            // reached that way has no params to feed and never reaches here.
            _ => continue,
        };
        ctx.replace_instruction_mnemonic(term_id, new);
    }
}

/// Return `args` with the entries at `drop` positions removed.
fn filter_kept(args: &[ValueId], drop: &HashSet<usize>) -> Vec<ValueId> {
    args.iter()
        .enumerate()
        .filter(|(i, _)| !drop.contains(i))
        .map(|(_, &a)| a)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::FunctionRef;
    use qcode_macro::qcode;

    /// All block ids of `fun`, used to drive the standalone sweep in tests.
    fn block_ids(ctx: &Context, fun: qcode::value::FunctionId) -> Vec<BlockId> {
        FunctionRef::from_id(ctx, fun)
            .blocks()
            .map(|b| b.id)
            .collect()
    }

    fn root_of(ctx: &Context, fun: qcode::value::FunctionId) -> Option<BlockId> {
        FunctionRef::from_id(ctx, fun).root().map(|b| b.id)
    }

    fn num_params(ctx: &Context, block: BlockId) -> usize {
        BasicBlock::from_id(ctx, block).num_params()
    }

    fn param_names(ctx: &Context, block: BlockId) -> Vec<String> {
        BasicBlock::from_id(ctx, block)
            .params()
            .map(|p| p.name().unwrap_or("?").to_string())
            .collect()
    }

    /// A loop-invariant param threaded `%c` off-loop and forwarded to itself on
    /// the back-edge — even though it is *read* inside the loop — collapses to
    /// `%c` (mirrors `<40b3f0>`'s `@ESP`). The genuinely-varying induction param
    /// fed `%next` on the back-edge is a real merge and stays.
    #[test]
    fn loop_invariant_param_collapses_to_its_value() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                %c = load(i64, 0x2000);
                goto <header @inv=%c @ind=0x0>;
            <header @inv:i64 @ind:i64>
                %cond = @ind < 0x3;
                %next = @ind + @inv;
                if %cond goto <header @inv=@inv @ind=%next> else goto <exit>;
            <exit>
                return [0x0];
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(remove_dead_block_args(&mut ctx, &blocks, root));

        // @inv is redundant (only {%c, @inv}) and removed; @ind merges {0x0,%next}
        // and stays. Its uses of @inv are rewritten to %c.
        assert_eq!(param_names(&ctx, header), vec!["ind"]);
        let next = BasicBlock::from_id(&ctx, header)
            .iter()
            .find(|i| i.as_statement().to_string().contains('+'))
            .unwrap()
            .as_statement()
            .to_string();
        assert_eq!(
            next, "i64 %next = @ind + i64 %c;",
            "use of @inv rewritten to %c"
        );
    }

    /// The user's motivating case: two params that only feed each other around a
    /// loop. `@y` collapses to `@x` first, turning `@x`'s back-edge arg into
    /// `@x=@x`, after which `@x` collapses to its off-loop value `0x0`. Both go.
    #[test]
    fn mutually_recursive_params_both_removed() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <header @x=0x0>;
            <header @x:i64>
                %i = load(i8, 0x1000);
                if %i goto <latch @y=@x> else goto <exit>;
            <latch @y:i64>
                goto <header @x=@y>;
            <exit>
                return [0x0];
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(remove_dead_block_args(&mut ctx, &blocks, root));

        assert_eq!(num_params(&ctx, header), 0, "@x should be gone");
        assert_eq!(num_params(&ctx, latch), 0, "@y should be gone");
        let header_term = BasicBlock::from_id(&ctx, header)
            .iter()
            .last()
            .unwrap()
            .as_statement()
            .to_string();
        assert_eq!(header_term, "if i8 %i goto <latch> else goto <exit>;");
    }

    /// A real merge of two distinct values is not redundant and is kept.
    #[test]
    fn genuine_merge_is_kept() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                %c = load(i8, 0x1000);
                if %c goto <join @m=0x1> else goto <other>;
            <other>
                goto <join @m=0x2>;
            <join @m:i64>
                store(0x4000, i64 @m);
                return [0x0];
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(!remove_dead_block_args(&mut ctx, &blocks, root));
        assert_eq!(num_params(&ctx, join), 1);
    }

    /// Only the redundant column is removed from a two-predecessor join; the
    /// genuinely-merged column and both predecessors' live args survive,
    /// re-indexed.
    #[test]
    fn one_redundant_column_among_two_preds() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                %c = load(i8, 0x1000);
                if %c goto <join @same=0x7 @merge=0x2> else goto <other>;
            <other>
                goto <join @same=0x7 @merge=0x4>;
            <join @same:i64 @merge:i64>
                store(0x4000, i64 @merge);
                store(0x4008, i64 @same);
                return [0x0];
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(remove_dead_block_args(&mut ctx, &blocks, root));

        // @same is 0x7 on both edges → collapses; @merge differs → kept.
        assert_eq!(param_names(&ctx, join), vec!["merge"]);
        let other_term = BasicBlock::from_id(&ctx, other)
            .iter()
            .last()
            .unwrap()
            .as_statement()
            .to_string();
        assert_eq!(other_term, "goto <join @merge=0x4>;");
    }

    /// Root (entry) params are the function interface and must not be touched
    /// here even when a back-edge makes them look redundant.
    #[test]
    fn root_params_left_alone() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <root @x:i64>
                %c = load(i8, 0x1000);
                if %c goto <root @x=@x> else goto <exit>;
            <exit>
                return [0x0];
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(!remove_dead_block_args(&mut ctx, &blocks, root));
        assert_eq!(num_params(&ctx, root.unwrap()), 1);
    }

    /// A multi-block cycle. Block `b` has a single predecessor, so *both* its
    /// params are trivially redundant and forward into `a` (jump-threading). In
    /// `a`, the `@inv` column only ever carries `%seed` (off-loop) or itself and
    /// collapses to `%seed`, while `@merg` merges two distinct values and stays.
    #[test]
    fn chain_collapses_redundant_column_keeps_merged_one() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                %seed = load(i64, 0x2000);
                goto <a @inva=%seed @merga=0x0>;
            <a @inva:i64 @merga:i64>
                goto <b @invb=@inva @mergb=@merga>;
            <b @invb:i64 @mergb:i64>
                store(0x5000, i64 @invb);
                %n = @mergb + 0x1;
                %c = load(i8, 0x1000);
                if %c goto <a @inva=@invb @merga=%n> else goto <exit>;
            <exit>
                return [0x0];
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(remove_dead_block_args(&mut ctx, &blocks, root));

        // b's params both forward into a (single predecessor). In a, @inva
        // collapses to %seed and @merga merges {0x0, %n} → kept.
        assert_eq!(param_names(&ctx, a), vec!["merga"]);
        assert!(
            param_names(&ctx, b).is_empty(),
            "b's params all forward into a"
        );
        // The store now reads %seed directly (via @invb → @inva → %seed).
        let store = BasicBlock::from_id(&ctx, b)
            .iter()
            .map(|i| i.as_statement().to_string())
            .find(|s| s.contains("0x5000"))
            .expect("store on 0x5000 survives in b");
        assert!(store.contains("%seed"), "store rewritten to %seed: {store}");
    }
}
