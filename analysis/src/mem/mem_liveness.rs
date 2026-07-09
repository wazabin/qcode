//! Flow-sensitive memory-liveness dataflow.
//!
//! The alias analyses ([`crate::AliasResult`]) provide a flow-*insensitive*
//! may-alias relation. To remove register/temp-space stores that are dead
//! *across* basic blocks, that oracle is layered with a backward dataflow over
//! the CFG computing, per block:
//!
//! - `live_out`: a *may* (union) set of locations possibly read along some path
//!   after the block before being overwritten. Over-approximating keeps it
//!   sound — a live reader is never missed.
//! - `killed_out`: a *must* (intersection) set of byte intervals guaranteed to
//!   be overwritten by a covering store before any read on **every** successor
//!   path. Exit/`return` blocks contribute the empty set, so a store reaching a
//!   return is not considered killed (only an explicit `dead_reg` is removable
//!   at a function boundary).
//!
//! These are fed back into [`crate::dce::dead_load::dead_load_insns_seeded`] as scan
//! seeds. The fixpoint is computed from below (both sets start empty), which is
//! sound for loops — `killed_out` is under-approximated (fewer guaranteed
//! kills) and `live_out` over-approximated.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::value::{
    BlockId, FunctionId, FunctionRef, ValueId, block::BlockRef, util::base_ref::HostRef,
};

use crate::AliasResult;
use crate::dce::dead_load::{
    KilledInterval, KilledSet, LiveLoc, LiveSet, block_transfer, is_tracked_space,
};

/// Per-block memory liveness summaries, indexed by block.
pub struct MemLiveness {
    live_out: HashMap<BlockId, LiveSet>,
    killed_out: HashMap<BlockId, KilledSet>,
}

impl MemLiveness {
    pub(crate) fn live_out(&self, block: BlockId) -> &[LiveLoc] {
        self.live_out.get(&block).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn killed_out(&self, block: BlockId) -> &[KilledInterval] {
        self.killed_out.get(&block).map_or(&[], Vec::as_slice)
    }
}

fn successors(host: HostRef, block: BlockId) -> Vec<BlockId> {
    BlockRef::new(host, block)
        .successors()
        .map(|(_, succ)| succ)
        .collect()
}

/// `live_out(B) = ⋃ live_in(successor)`.
fn union_live(succs: &[BlockId], live_in: &HashMap<BlockId, LiveSet>) -> LiveSet {
    let mut acc: HashSet<LiveLoc> = HashSet::default();
    for &succ in succs {
        if let Some(set) = live_in.get(&succ) {
            acc.extend(set.iter().copied());
        }
    }
    acc.into_iter().collect()
}

/// `killed_out(B) = ⋂ killed_in(successor)`; empty when `B` has no successors.
fn intersect_killed(succs: &[BlockId], killed_in: &HashMap<BlockId, KilledSet>) -> KilledSet {
    let mut iter = succs.iter();
    let Some(&first) = iter.next() else {
        return Vec::new();
    };
    let mut acc: HashSet<KilledInterval> = killed_in
        .get(&first)
        .map(|set| set.iter().copied().collect())
        .unwrap_or_default();
    for &succ in iter {
        let other: HashSet<KilledInterval> = killed_in
            .get(&succ)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default();
        acc.retain(|item| other.contains(item));
    }
    acc.into_iter().collect()
}

/// Restrict liveness facts to register/temp spaces; RAM/global stores are never
/// propagated across blocks (they may be observed outside the function).
fn retain_tracked(host: HostRef, live: &mut LiveSet, killed: &mut KilledSet) {
    live.retain(|l| is_tracked_space(host.shared(), l.space));
    killed.retain(|k| is_tracked_space(host.shared(), k.space));
}

fn live_changed(old: &LiveSet, new: &LiveSet) -> bool {
    let old: HashSet<_> = old.iter().copied().collect();
    let new: HashSet<_> = new.iter().copied().collect();
    old != new
}

fn killed_changed(old: &KilledSet, new: &KilledSet) -> bool {
    let old: HashSet<_> = old.iter().copied().collect();
    let new: HashSet<_> = new.iter().copied().collect();
    old != new
}

/// Run the backward memory-liveness dataflow over `function_id`.
pub fn compute_memory_liveness<'a, 'str: 'a>(
    host: impl Into<HostRef<'a, 'str>>,
    function_id: FunctionId,
    aliases: &AliasResult,
    dead_regs: &[ValueId],
) -> MemLiveness {
    let host = host.into();
    let blocks: Vec<BlockId> = FunctionRef::new(host, function_id)
        .iter()
        .map(|block| block.id)
        .collect();

    let mut live_in: HashMap<BlockId, LiveSet> = blocks.iter().map(|&b| (b, Vec::new())).collect();
    let mut killed_in: HashMap<BlockId, KilledSet> =
        blocks.iter().map(|&b| (b, Vec::new())).collect();

    // Predecessor map: a block's in-sets feed its predecessors' `live_out` /
    // `killed_out`, so only the predecessors need recomputing when it changes.
    let mut preds: HashMap<BlockId, Vec<BlockId>> =
        blocks.iter().map(|&b| (b, Vec::new())).collect();
    for &block in &blocks {
        for succ in successors(host, block) {
            if let Some(entry) = preds.get_mut(&succ) {
                entry.push(block);
            }
        }
    }

    // Worklist fixpoint: recompute a block only when one of its successors'
    // in-sets changed, rather than re-sweeping every block to stability — the
    // same monotone equations and the same fixpoint, but work proportional to
    // actual changes instead of O(B²) on large functions.
    let mut worklist: Vec<BlockId> = blocks.clone();
    let mut queued: HashSet<BlockId> = blocks.iter().copied().collect();
    while let Some(block) = worklist.pop() {
        queued.remove(&block);
        let succs = successors(host, block);
        let live_out = union_live(&succs, &live_in);
        let killed_out = intersect_killed(&succs, &killed_in);

        let (mut new_live_in, mut new_killed_in) = block_transfer(
            host.shared(),
            block,
            aliases,
            dead_regs,
            &live_out,
            &killed_out,
        );
        retain_tracked(host, &mut new_live_in, &mut new_killed_in);

        let mut block_changed = false;
        if live_changed(&live_in[&block], &new_live_in) {
            live_in.insert(block, new_live_in);
            block_changed = true;
        }
        if killed_changed(&killed_in[&block], &new_killed_in) {
            killed_in.insert(block, new_killed_in);
            block_changed = true;
        }
        if block_changed {
            for &pred in &preds[&block] {
                if queued.insert(pred) {
                    worklist.push(pred);
                }
            }
        }
    }

    let mut live_out = HashMap::default();
    let mut killed_out = HashMap::default();
    for &block in &blocks {
        let succs = successors(host, block);
        live_out.insert(block, union_live(&succs, &live_in));
        killed_out.insert(block, intersect_killed(&succs, &killed_in));
    }

    MemLiveness {
        live_out,
        killed_out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AliasResult;
    use crate::dce::dead_load::dead_load_insns_seeded;
    use qcode::{
        context::Context,
        value::{
            BasicBlock,
            insn::{InstructionId, Mnemonic},
        },
    };
    use qcode_macro::qcode;
    use rustc_hash::FxHashSet as HashSet;

    /// Store instruction ids in `block`, in program order.
    fn store_ids(ctx: &Context, block: BlockId) -> Vec<InstructionId> {
        BasicBlock::from_id(ctx, block)
            .instruction_ids()
            .iter()
            .copied()
            .filter(|&id| matches!(ctx.get_insn(id).mnemonic(), Mnemonic::Store(_)))
            .collect()
    }

    /// Dead stores/loads detected in `block` using the cross-block seeds.
    fn dead_in(
        ctx: &Context,
        function: FunctionId,
        block: BlockId,
        aliases: &AliasResult,
        dead_regs: &[ValueId],
    ) -> HashSet<InstructionId> {
        let liveness = compute_memory_liveness(ctx, function, aliases, dead_regs);
        dead_load_insns_seeded(
            ctx,
            block,
            aliases,
            dead_regs,
            liveness.live_out(block),
            liveness.killed_out(block),
        )
    }

    #[test]
    fn cross_block_overwrite_is_dead() {
        // A written in bb1, unconditionally overwritten in bb2 before any read.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            fn test:
                <bb1>
                    store(A:8, &A <- i64 1);
                    goto <bb2>;
                <bb2>
                    store(A:8, &A <- i64 2);
                    return at 0;
            "
        );

        let aliases = AliasResult::simple(&ctx);
        let dead = dead_in(&ctx, test, bb1, &aliases, &[]);
        assert!(
            dead.contains(&store_ids(&ctx, bb1)[0]),
            "store to A in bb1 is overwritten in bb2 with no read, must be dead"
        );

        // The overwriting store reaches a return; it is not dead.
        let dead_bb2 = dead_in(&ctx, test, bb2, &aliases, &[]);
        assert!(
            !dead_bb2.contains(&store_ids(&ctx, bb2)[0]),
            "store to A in bb2 reaches a return, must not be dead"
        );
    }

    #[test]
    fn cross_block_live_read_keeps_store() {
        // A written in bb1, read in bb2 before being overwritten -> live.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            fn test:
                <bb1>
                    store(A:8, &A <- i64 1);
                    goto <bb2>;
                <bb2>
                    %v = load(A:8, &A);
                    store(B:8, &B <- %v);
                    return at 0;
            "
        );

        let aliases = AliasResult::simple(&ctx);
        let dead = dead_in(&ctx, test, bb1, &aliases, &[]);
        assert!(
            !dead.contains(&store_ids(&ctx, bb1)[0]),
            "store to A in bb1 is read in bb2, must not be dead"
        );
    }

    #[test]
    fn diamond_partial_read_keeps_store() {
        // A read on only one of two successor paths -> the union (may) live-out
        // keeps the store.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            fn test:
                <bb1>
                    store(A:8, &A <- i64 1);
                    if i8 1 goto <bb2> else goto <bb3>;
                <bb2>
                    %v = load(A:8, &A);
                    store(B:8, &B <- %v);
                    goto <bb4>;
                <bb3>
                    goto <bb4>;
                <bb4>
                    return at 0;
            "
        );

        let aliases = AliasResult::simple(&ctx);
        let dead = dead_in(&ctx, test, bb1, &aliases, &[]);
        assert!(
            !dead.contains(&store_ids(&ctx, bb1)[0]),
            "store to A is read on the bb2 path, must not be dead"
        );
    }

    #[test]
    fn diamond_partial_overwrite_keeps_store() {
        // A overwritten on only one path before the join -> killed_out is the
        // intersection (must), so bb1's store is not provably dead.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            fn test:
                <bb1>
                    store(A:8, &A <- i64 1);
                    if i8 1 goto <bb2> else goto <bb3>;
                <bb2>
                    store(A:8, &A <- i64 2);
                    goto <bb4>;
                <bb3>
                    goto <bb4>;
                <bb4>
                    return at 0;
            "
        );

        let aliases = AliasResult::simple(&ctx);
        let dead = dead_in(&ctx, test, bb1, &aliases, &[]);
        assert!(
            !dead.contains(&store_ids(&ctx, bb1)[0]),
            "A is overwritten on only one path, store must not be dead"
        );
    }

    #[test]
    fn cross_block_dead_reg_is_dead() {
        // A is marked as a dead register (live-out value never observable) and
        // is never read; its store is dead even without a covering overwrite.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            fn test:
                <bb1>
                    store(A:8, &A <- i64 1);
                    goto <bb2>;
                <bb2>
                    return at 0;
            "
        );

        let aliases = AliasResult::simple(&ctx);
        let dead_regs = vec![ValueId::from(A)];
        let dead = dead_in(&ctx, test, bb1, &aliases, &dead_regs);
        assert!(
            dead.contains(&store_ids(&ctx, bb1)[0]),
            "store to dead-reg A with no downstream read must be dead"
        );

        // Without the dead-reg hint it reaches a return and must be kept.
        let dead_no_hint = dead_in(&ctx, test, bb1, &aliases, &[]);
        assert!(
            !dead_no_hint.contains(&store_ids(&ctx, bb1)[0]),
            "without dead-reg hint, store reaching a return must not be dead"
        );
    }
}
