//! Re-deriving function boundaries on the clean IR by CFG ownership.
//!
//! Recursive disassembly is order-dependent: a function entry is only known once
//! something reaches it (a `call`, an export, a tail `jmp`). If a function `F` is
//! lifted while one of its blocks `B@a` is not yet known to be a function entry,
//! `F` *absorbs* `B` and everything downstream of it. When `a` later turns out to
//! be its own function `G` — most commonly a thunk `jmp realfunc` lifted before
//! anything `call`s `realfunc` — `G` is left a stub while its body lives inside
//! `F`. The GUI then shows `G` with a single (stub) instruction even though the
//! bytes decode fine in the flat listing.
//!
//! [`split_overlapping_functions`] repairs this on the clean IR by recomputing
//! ownership from the control-flow graph: each function claims the blocks
//! reachable from its entry, and a forward walk stops at any *other* function's
//! entry (a tail-call boundary). A block reachable from two different entries is
//! shared code and is promoted to its own function. Each block's `parent`, and
//! each function's `blocks`/`root`/`instruction_addrs`, are rewritten to match.
//! Because the optimized context is always re-derived from the clean IR, fixing
//! those fields is enough — the change invalidates and replays analysis exactly
//! like discovering new code does.
//!
//! This relies on the clean CFG actually encoding resolved control flow: the
//! jump-table pass records a real edge from each indirect branch to its targets
//! (see `Context::discover_code`), and the lifter materializes the fall-through
//! edge of every may-return call, so both a switch's case bodies and a call's
//! continuation are reachable from — and thus claimed by — their owning function.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::{BTreeSet, VecDeque};

use qcode::{
    context::Context,
    value::{BasicBlock, BlockId, Function, FunctionId, block::EdgeId},
};

/// Safety cap on fixpoint rounds (promotion + reattribution both monotonic).
const MAX_ROUNDS: usize = 64;

/// Recompute function ownership from the CFG until it settles. Returns `true` if
/// anything changed (so the caller replays analysis).
pub fn split_overlapping_functions(ctx: &mut Context) -> bool {
    let mut changed_any = false;
    for _ in 0..MAX_ROUNDS {
        // Shared code reachable from two function entries is promoted to its own
        // function first (it becomes a tail-call boundary); then every function
        // re-claims the blocks reachable from its entry. Both steps are monotonic.
        if promote_shared_blocks(ctx) {
            changed_any = true;
            continue;
        }
        if reattribute_blocks(ctx) {
            changed_any = true;
            continue;
        }
        break;
    }
    // Ownership has settled. A `jmp`/`jcc` into another function's entry (a tail
    // call, or the thunk `jmp realfunc` that seeded a boundary above) left an
    // inter-procedural CFG edge behind. Strip those so every function's
    // successor/predecessor graph is closed over its own blocks — see
    // [`remove_cross_function_edges`].
    if remove_cross_function_edges(ctx) {
        changed_any = true;
    }
    changed_any
}

/// Remove every CFG edge whose endpoints belong to different functions.
///
/// After [`reattribute_blocks`] settles ownership, a terminator that jumps to
/// another function's entry — a tail call, or a thunk's `jmp realfunc` — is an
/// *inter*-procedural control transfer. The terminator still records the target
/// (so thunk naming still sees it; the call graph is built from `Call`
/// instructions, not CFG edges, so it is unaffected either way), but the CFG
/// *edge* has no intra-procedural meaning: leaving it in makes every per-function
/// analysis that walks `successors`/`predecessors` (dominators, liveness, GVN,
/// mem2reg's rename walk) silently traverse into a foreign function's blocks.
/// The splitter's own reach
/// walk ([`claimed_from`]) already stops at these boundaries, so removing the
/// edges cannot change ownership — it only closes each function's CFG over its
/// own blocks. Returns `true` if any edge was removed.
fn remove_cross_function_edges(ctx: &mut Context) -> bool {
    let stale: Vec<EdgeId> = ctx
        .blocks()
        .filter_map(|block| {
            let owner = block.parent()?.id;
            let doomed: Vec<EdgeId> = block
                .successors()
                .filter(|&(_, succ)| {
                    // Keep an edge only when the successor is owned by the same
                    // function. A successor with no owner (an orphan) is left
                    // alone — orphans are handled by `reattribute_blocks`.
                    BasicBlock::from_id(ctx, succ)
                        .parent()
                        .is_some_and(|f| f.id != owner)
                })
                .map(|(edge, _)| edge)
                .collect();
            (!doomed.is_empty()).then_some(doomed)
        })
        .flatten()
        .collect();

    let changed = !stale.is_empty();
    for edge in stale {
        ctx.remove_cfg_edge(edge);
    }
    changed
}

/// Map every block that carries a machine address to its id. Addresses are unique
/// per block (the context rejects duplicates), so this is well-defined.
fn block_by_address(ctx: &Context) -> HashMap<u64, BlockId> {
    ctx.blocks()
        .filter_map(|b| b.address().map(|a| (a, b.id)))
        .collect()
}

/// The entry block of every non-external function that has one (the block at the
/// function's address), paired with its function.
fn function_entries(ctx: &Context, block_at: &HashMap<u64, BlockId>) -> Vec<(BlockId, FunctionId)> {
    let mut entries = Vec::new();
    for id in ctx.function_ids() {
        let f = Function::from_id(ctx, id);
        if f.is_external() {
            continue;
        }
        if let Some(addr) = f.address()
            && let Some(&entry) = block_at.get(&addr)
        {
            entries.push((entry, id));
        }
    }
    entries
}

/// Blocks reachable from a function entry along CFG edges, stopping at any other
/// function's entry — the tail-call boundary. The clean CFG already encodes a
/// call's fall-through (materialized at lift) and jump-table edges, so a plain
/// successor walk is exact.
fn claimed_from(
    ctx: &Context,
    entry: BlockId,
    owner: FunctionId,
    entry_of: &HashMap<BlockId, FunctionId>,
) -> HashSet<BlockId> {
    let mut seen = HashSet::default();
    seen.insert(entry);
    let mut queue = VecDeque::from([entry]);
    while let Some(b) = queue.pop_front() {
        for (_, succ) in BasicBlock::from_id(ctx, b).successors() {
            // A different function's entry is a boundary; never cross it.
            if entry_of.get(&succ).is_some_and(|&f| f != owner) {
                continue;
            }
            if seen.insert(succ) {
                queue.push_back(succ);
            }
        }
    }
    seen
}

/// Promote any block reachable from two different function entries to its own
/// function (shared code, e.g. a tail target two routines jump to). Returns
/// `true` if a function was created.
fn promote_shared_blocks(ctx: &mut Context) -> bool {
    let block_at = block_by_address(ctx);
    let entries = function_entries(ctx, &block_at);
    let entry_of: HashMap<BlockId, FunctionId> = entries.iter().copied().collect();

    // First reaching function wins the block; a second reach marks it contested.
    let mut owner_of: HashMap<BlockId, FunctionId> = HashMap::default();
    let mut contested: Vec<BlockId> = Vec::new();
    for &(entry, func) in &entries {
        for b in claimed_from(ctx, entry, func, &entry_of) {
            match owner_of.insert(b, func) {
                Some(prev) if prev != func && !entry_of.contains_key(&b) => contested.push(b),
                _ => {}
            }
        }
    }

    let mut promoted = false;
    for b in contested {
        if let Some(addr) = ctx.values.block(b).address
            && Function::from_addr(ctx, addr).is_none()
        {
            Function::make_at_addr(ctx, addr, None);
            log::debug!(target: "split", "promoted shared block {addr:#x} to its own function");
            promoted = true;
        }
    }
    promoted
}

/// Reassign each block to the function whose entry reaches it, and rebuild every
/// touched function's `blocks`, `root`, and `instruction_addrs`. Returns `true`
/// if any block's owner changed.
fn reattribute_blocks(ctx: &mut Context) -> bool {
    let block_at = block_by_address(ctx);
    let entries = function_entries(ctx, &block_at);
    let entry_of: HashMap<BlockId, FunctionId> = entries.iter().copied().collect();

    let mut new_owner: HashMap<BlockId, FunctionId> = HashMap::default();
    for &(entry, func) in &entries {
        for b in claimed_from(ctx, entry, func, &entry_of) {
            // A block reachable from several entries is shared; `promote_shared_blocks`
            // turns it into its own boundary first, so here we keep the first claim.
            new_owner.entry(b).or_insert(func);
        }
    }

    // Settle each block's owner: a block reachable from some entry belongs to that
    // function; an unreached block keeps whatever owner it already had (orphans
    // from earlier lifting are left alone, not silently dropped).
    let mut changed = false;
    let mut owners_changed: HashSet<FunctionId> = HashSet::default();
    for id in ctx.block_ids() {
        let cur = ctx.values.block(id).parent;
        let Some(desired) = new_owner.get(&id).copied().or(cur) else {
            continue;
        };
        if cur != Some(desired) {
            // `add_block` reassigns ownership: it drops the block from the
            // previous owner's roster, sets `parent`, and claims it here. Block
            // *storage* stays in the block's birth arena (`id.func`).
            Function::from_id_mut(ctx, desired).add_block(id);
            if let Some(prev) = cur {
                owners_changed.insert(prev);
            }
            owners_changed.insert(desired);
            changed = true;
        }
    }
    if !changed {
        return false;
    }

    // Fix roots and instruction addresses on the affected functions. The
    // ownership rosters were maintained by `add_block` above.
    for &(entry, func) in &entries {
        if owners_changed.contains(&func) && ctx.values.functions[func].root != Some(entry) {
            ctx.values.functions[func].root = Some(entry);
        }
    }
    // A function that only *lost* blocks (a `prev` owner) is in `owners_changed`
    // but may be absent from `entries` (it kept no block at its own address), so
    // the re-root above skips it. If its recorded `root` was one of the blocks
    // just reassigned away, it now points outside the owned set — an invariant
    // every consumer relies on (e.g. `compute_input_regs` indexes
    // `live_in[root]`). Drop such an orphaned root so the function reads as a
    // rootless stub instead of crashing analysis.
    for &func in &owners_changed {
        let live: HashSet<BlockId> = Function::from_id(ctx, func).block_ids().into_iter().collect();
        let root = ctx.values.functions[func].root;
        if root.is_some_and(|r| !live.contains(&r)) {
            ctx.values.functions[func].root = None;
        }
    }
    for func in owners_changed {
        recompute_instruction_addrs(ctx, func);
    }
    changed
}

/// Rebuild a function's `instruction_addrs` from the machine addresses of the
/// instructions in its (post-split) blocks. On the clean IR blocks are not yet
/// merged, so this is exact.
fn recompute_instruction_addrs(ctx: &mut Context, func: FunctionId) {
    let blocks: Vec<BlockId> = Function::from_id(ctx, func).block_ids();
    let mut addrs = BTreeSet::new();
    for b in blocks {
        for insn in BasicBlock::from_id(ctx, b).instructions() {
            if let Some(a) = insn.address() {
                addrs.insert(a);
            }
        }
    }
    ctx.values.functions[func].instruction_addrs = addrs;
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::builder::Builder;
    use qcode::value::{Function, FunctionId, Instruction};
    use std::borrow::Cow;

    fn block_at(ctx: &mut Context, addr: u64) -> BlockId {
        BasicBlock::make(ctx).with_address(addr).id
    }

    /// Append an unconditional branch to `block`, tagged with machine `addr`.
    fn branch_at(ctx: &mut Context, block: BlockId, target: BlockId, addr: u64) {
        let id = Builder::from_block(BasicBlock::from_id_mut(ctx, block))
            .push_branch(target)
            .id;
        Instruction::from_id_mut(ctx, id).set_address(addr);
    }

    /// Append a `return` to `block`, tagged with machine `addr`.
    fn return_at(ctx: &mut Context, block: BlockId, addr: u64) {
        let zero = ctx.get_const(0, 8).id();
        let id = Builder::from_block(BasicBlock::from_id_mut(ctx, block))
            .push_return(zero)
            .id;
        Instruction::from_id_mut(ctx, id).set_address(addr);
    }

    /// A thunk function `F` at 0x1000 (`jmp 0x2000`) that absorbed the body of the
    /// function later discovered at 0x2000: blocks 0x2000 -> 0x2005(ret). Once a
    /// stub `G` exists at 0x2000, the split must move 0x2000 and 0x2005 into `G`,
    /// leave only the thunk block in `F`, and rebuild both `instruction_addrs`.
    #[test]
    fn splits_absorbed_body_into_its_own_function() {
        let mut ctx = Context::new();

        let b0 = block_at(&mut ctx, 0x1000);
        let b1 = block_at(&mut ctx, 0x2000);
        let b2 = block_at(&mut ctx, 0x2005);

        // b0: jmp b1 ; b1: jmp b2 ; b2: ret
        branch_at(&mut ctx, b0, b1, 0x1000);
        branch_at(&mut ctx, b1, b2, 0x2000);
        return_at(&mut ctx, b2, 0x2005);
        ctx.add_cfg_edge(b0, b1);
        ctx.add_cfg_edge(b1, b2);

        // F owns all three blocks; its entry is the thunk block.
        let f = Function::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("thunk"))).id;
        {
            let mut func = Function::from_id_mut(&mut ctx, f);
            func.set_root(b0).unwrap();
            func.add_block(b1);
            func.add_block(b2);
            func.add_instruction_addr(0x1000);
            func.add_instruction_addr(0x2000);
            func.add_instruction_addr(0x2005);
        }

        // A later `call 0x2000` creates the stub for the real function.
        let g = Function::make_at_addr(&mut ctx, 0x2000, Some(Cow::Borrowed("real"))).id;

        assert!(split_overlapping_functions(&mut ctx));

        assert_block_set(&ctx, f, &[b0]);
        assert_block_set(&ctx, g, &[b1, b2]);
        assert_eq!(ctx.values.block(b1).parent, Some(g));
        assert_eq!(ctx.values.block(b2).parent, Some(g));
        assert_eq!(ctx.values.functions[g].root, Some(b1));

        assert_eq!(addrs(&ctx, f), vec![0x1000]);
        assert_eq!(addrs(&ctx, g), vec![0x2000, 0x2005]);

        // Idempotent: nothing left to split.
        assert!(!split_overlapping_functions(&mut ctx));
    }

    /// Append a `call callee` to `block`, tagged with machine `addr`. `Call` is a
    /// terminator with no fall-through operand, so the lifter materializes the
    /// continuation edge separately; the caller wires it with `add_cfg_edge`.
    fn call_at(ctx: &mut Context, block: BlockId, callee: FunctionId, addr: u64) {
        let id = Builder::from_block(BasicBlock::from_id_mut(ctx, block))
            .push_call(callee)
            .id;
        Instruction::from_id_mut(ctx, id).set_address(addr);
    }

    /// The clean CFG carries a materialized call fall-through edge; the splitter
    /// claims the continuation by following it like any other successor. Here
    /// `F`@0x1000 absorbed `G`@0x2000, whose body is `call; ret` — the `ret`
    /// block must end up in `G`, not `F`.
    #[test]
    fn claims_post_call_continuation_through_materialized_edge() {
        let mut ctx = Context::new();

        let callee = Function::make_at_addr(&mut ctx, 0x3000, Some(Cow::Borrowed("callee"))).id;
        {
            let ret = block_at(&mut ctx, 0x3000);
            return_at(&mut ctx, ret, 0x3000);
            Function::from_id_mut(&mut ctx, callee)
                .set_root(ret)
                .unwrap();
        }

        let b0 = block_at(&mut ctx, 0x1000); // thunk: jmp 0x2000
        let b1 = block_at(&mut ctx, 0x2000); // call callee  (materialized FT edge)
        let b2 = block_at(&mut ctx, 0x2005); // ret          (fall-through continuation)

        branch_at(&mut ctx, b0, b1, 0x1000);
        call_at(&mut ctx, b1, callee, 0x2000);
        return_at(&mut ctx, b2, 0x2005);
        ctx.add_cfg_edge(b0, b1);
        ctx.add_cfg_edge(b1, b2); // the lifter's materialized call fall-through

        let f = Function::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("thunk"))).id;
        {
            let mut func = Function::from_id_mut(&mut ctx, f);
            func.set_root(b0).unwrap();
            func.add_block(b1);
            func.add_block(b2);
        }
        let g = Function::make_at_addr(&mut ctx, 0x2000, Some(Cow::Borrowed("real"))).id;

        assert!(split_overlapping_functions(&mut ctx));

        assert_block_set(&ctx, f, &[b0]);
        assert_block_set(&ctx, g, &[b1, b2]);
        assert_eq!(
            ctx.values.block(b2).parent,
            Some(g),
            "the post-call block must be claimed via the materialized fall-through edge",
        );
    }

    /// A `jmp` from one function into another's entry (a tail call) leaves an
    /// inter-procedural CFG edge that must be stripped once ownership settles:
    /// the edge has no intra-procedural meaning and would make every per-function
    /// analysis wander into foreign blocks. The terminator's target is preserved,
    /// and intra-function edges are untouched.
    #[test]
    fn strips_tail_call_edge_into_another_function() {
        use qcode::value::insn::{Branch, Mnemonic};

        let mut ctx = Context::new();

        // F@0x1000: entry(0x1000) -> tail(0x1008); tail: jmp g_entry(0x2000).
        let entry = block_at(&mut ctx, 0x1000);
        let tail = block_at(&mut ctx, 0x1008);
        let g_entry = block_at(&mut ctx, 0x2000);

        // `branch_at` (via `push_branch`) already wires the CFG edge.
        branch_at(&mut ctx, entry, tail, 0x1000); // intra-F edge — kept
        branch_at(&mut ctx, tail, g_entry, 0x1008); // F -> G tail call — stripped
        return_at(&mut ctx, g_entry, 0x2000);

        let f = Function::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("f"))).id;
        {
            let mut func = Function::from_id_mut(&mut ctx, f);
            func.set_root(entry).unwrap();
            func.add_block(tail);
        }
        let g = Function::make_at_addr(&mut ctx, 0x2000, Some(Cow::Borrowed("g"))).id;
        Function::from_id_mut(&mut ctx, g)
            .set_root(g_entry)
            .unwrap();

        assert!(split_overlapping_functions(&mut ctx));

        // Ownership is already correct; the only change is the stripped edge.
        assert_block_set(&ctx, f, &[entry, tail]);
        assert_block_set(&ctx, g, &[g_entry]);

        // The cross-function edge is gone; the intra-function edge survives.
        assert_eq!(
            BasicBlock::from_id(&ctx, tail).successors().count(),
            0,
            "the tail-call edge into G must be removed",
        );
        assert_eq!(
            BasicBlock::from_id(&ctx, entry)
                .successors()
                .map(|(_, s)| s)
                .collect::<Vec<_>>(),
            vec![tail],
            "the intra-function edge must be preserved",
        );

        // The terminator still records the jump target — only the CFG edge went.
        let term = BasicBlock::from_id(&ctx, tail)
            .instructions()
            .last()
            .map(|i| i.mnemonic().clone());
        assert!(
            matches!(term, Some(Mnemonic::Branch(Branch { target, .. })) if target == g_entry),
            "the tail-call terminator target must be preserved, got {term:?}",
        );

        // Idempotent: nothing left to strip.
        assert!(!split_overlapping_functions(&mut ctx));
    }

    fn assert_block_set(ctx: &Context, func: FunctionId, expected: &[BlockId]) {
        let blocks = &ctx.values.functions[func].blocks;
        assert_eq!(
            blocks.len(),
            expected.len(),
            "block count mismatch for {func:?}"
        );
        for b in expected {
            assert!(blocks.contains(b), "{func:?} missing block {b:?}");
        }
    }

    fn addrs(ctx: &Context, func: FunctionId) -> Vec<u64> {
        ctx.values.functions[func]
            .instruction_addrs
            .iter()
            .copied()
            .collect()
    }
}
