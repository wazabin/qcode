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
    builder::Builder,
    context::Context,
    value::{BasicBlock, BlockId, Function, FunctionId, block::EdgeId},
};

/// Safety cap on fixpoint rounds (promotion + reattribution both monotonic).
const MAX_ROUNDS: usize = 64;

/// Whether any block references a block owned by (or stored in) a *different*
/// function — a cross-function CFG edge or a terminator whose static target block
/// belongs to another function (a thunk / tail-call `Branch`, a cross-function
/// `jcc`). This is precisely the state [`split_overlapping_functions`] normalizes
/// away (strict IR locality, context-split ruling 2); the optimization entry uses
/// it to skip a needless clone+normalize on IR that is already local (every
/// lifter-driven discovery round leaves the clean IR in that state).
pub fn has_cross_function_reference(ctx: &Context) -> bool {
    let owner_of = |b: BlockId| BasicBlock::from_id(ctx, b).parent().map(|f| f.id);
    ctx.blocks().any(|b| {
        let Some(owner) = b.parent().map(|f| f.id) else {
            return false;
        };
        let foreign = |t: BlockId| owner_of(t).is_some_and(|o| o != owner);
        b.successors().any(|(_, s)| foreign(s))
            || b.instructions()
                .any(|i| i.mnemonic().target_blocks().iter().any(|&t| foreign(t)))
    })
}

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
        // A terminator that targets the *middle* of another function (a `jmp`/`jcc`
        // into a non-entry block of a sibling routine) is a genuine inter-procedural
        // landing that the IR must not encode as a foreign `BlockId`. Force a
        // function split at that landing: promote it to its own function's entry so
        // `reattribute_blocks` re-claims the tail below it, and the incoming edge
        // becomes an into-entry tail call the conversion step can honestly encode.
        if promote_cross_function_landings(ctx) {
            changed_any = true;
            continue;
        }
        break;
    }
    // Ownership has settled. A `jmp`/`jcc` into another function's entry (a tail
    // call, or the thunk `jmp realfunc` that seeded a boundary above) left an
    // inter-procedural CFG edge behind, and its terminator still carries a foreign
    // `BlockId`. Rewrite an unconditional `jmp` into another function's *entry* to
    // a function-level [`TailCall`] terminator (strict IR locality — no foreign
    // block reference), and strip every remaining cross-function CFG edge so each
    // function's successor/predecessor graph is closed over its own blocks — see
    // [`convert_cross_function_tail_calls`].
    if convert_cross_function_tail_calls(ctx) {
        changed_any = true;
    }
    // Ownership and the CFG are now settled and each function's graph is closed
    // over its own blocks, but `reattribute_blocks` moved *ownership* (a block's
    // `parent`/roster) without moving *storage* (`id.func`): a function can still
    // own a block that lives in another function's arena. Discharge that here so
    // split always hands back strictly local IR — every block self-stored, the
    // invariant `CheckedOut::new` asserts at every checkout. A pure storage move
    // (the IR is semantically identical), and a cheap no-op scan once storage
    // already matches ownership (the steady state after the first round).
    if ctx.normalize_block_storage() {
        changed_any = true;
    }
    changed_any
}

/// Rewrite cross-function tail jumps to [`TailCall`] terminators and remove every
/// CFG edge whose endpoints belong to different functions.
///
/// After [`reattribute_blocks`] settles ownership, a terminator that jumps to
/// another function's entry — a tail call, or a thunk's `jmp realfunc` — is an
/// *inter*-procedural control transfer. Two things must be repaired:
///
/// 1. **The terminator.** Its target is a foreign `BlockId` — exactly the
///    cross-function reference the context-split design (ruling 2) forbids. By the
///    time this runs, [`promote_cross_function_landings`] has forced every
///    mid-function landing to become its own function's entry, so every surviving
///    cross-function target is a foreign *entry*. An unconditional `Branch` into a
///    foreign entry (the thunk / tail-call shape) is rewritten to a function-level
///    [`TailCall`] carrying the callee's `FunctionId`. A *conditional* arm into a
///    foreign entry is routed through a fresh intra-function trampoline block that
///    ends in a `TailCall` (a `TailCall` is unconditional and cannot be a `CBranch`
///    arm). Either way the IR holds no foreign block reference and the reverse call
///    graph picks the edge up through `call_target`/`call_sites`.
/// 2. **The CFG edge.** Whether or not the terminator was rewritten, the
///    inter-procedural *edge* has no intra-procedural meaning: leaving it in makes
///    every per-function analysis that walks `successors`/`predecessors`
///    (dominators, liveness, GVN, mem2reg's rename walk) silently traverse into a
///    foreign function's blocks. The splitter's own reach walk ([`claimed_from`])
///    already stops at these boundaries, so removing the edges cannot change
///    ownership — it only closes each function's CFG over its own blocks.
///
/// Returns `true` if any terminator was rewritten or any edge removed.
fn convert_cross_function_tail_calls(ctx: &mut Context) -> bool {
    use qcode::value::insn::{Branch, CBranch, Mnemonic, TailCall};

    // An unconditional `Branch` into a foreign function's entry: the terminator
    // instruction and the callee it tail-jumps to. Rewritten in place to `TailCall`.
    let mut tail_calls: Vec<(qcode::value::insn::InstructionId, FunctionId)> = Vec::new();
    // A conditional arm (`CBranch`) into a foreign function's entry: the terminator,
    // the block that owns it, and — per arm — the callee that arm jumps to. Each
    // needs a fresh intra-function trampoline block ending in a `TailCall`, since a
    // `TailCall` is unconditional and cannot itself be a `CBranch` arm.
    let mut cond_calls: Vec<(qcode::value::insn::InstructionId, BlockId, FunctionId)> = Vec::new();
    // Every cross-function CFG edge, converted or residual, is stripped.
    let mut stale: Vec<EdgeId> = Vec::new();

    // Resolve a static terminator target to the foreign function whose *entry* it is
    // (the only cross-function shape that survives after `promote_cross_function_landings`
    // has forced every mid-function landing to become its own entry).
    let foreign_entry = |ctx: &Context, target: BlockId, owner: FunctionId| -> Option<FunctionId> {
        let callee = BasicBlock::from_id(ctx, target).parent().map(|f| f.id)?;
        (callee != owner && Function::from_id(ctx, callee).root().map(|r| r.id) == Some(target))
            .then_some(callee)
    };

    for block in ctx.blocks() {
        let Some(owner) = block.parent().map(|f| f.id) else {
            continue;
        };
        for (edge, succ) in block.successors() {
            // Keep an edge only when the successor is owned by the same function.
            // A successor with no owner (an orphan) is left alone.
            let Some(succ_owner) = BasicBlock::from_id(ctx, succ).parent().map(|f| f.id) else {
                continue;
            };
            if succ_owner != owner {
                stale.push(edge);
            }
        }

        match block.instructions().last().map(|t| (t.id, t.mnemonic())) {
            // Unconditional tail jump into another function's entry.
            Some((id, Mnemonic::Branch(Branch { target, .. }))) => {
                if let Some(callee) = foreign_entry(ctx, *target, owner) {
                    tail_calls.push((id, callee));
                }
            }
            // A conditional arm into another function's entry. Each arm is handled
            // independently; a self-loop / intra arm is left untouched.
            Some((
                id,
                Mnemonic::CBranch(CBranch {
                    success_block,
                    failure_block,
                    ..
                }),
            )) => {
                if let Some(callee) = foreign_entry(ctx, *success_block, owner) {
                    cond_calls.push((id, block.id, callee));
                }
                if let Some(callee) = foreign_entry(ctx, *failure_block, owner) {
                    cond_calls.push((id, block.id, callee));
                }
            }
            _ => {}
        }
    }

    let changed = !tail_calls.is_empty() || !cond_calls.is_empty() || !stale.is_empty();

    for (insn, callee) in tail_calls {
        ctx.replace_instruction_mnemonic(
            insn,
            Mnemonic::TailCall(TailCall {
                target: callee,
                args: vec![],
            }),
        );
    }
    // Route each conditional cross-function arm through a fresh trampoline block in
    // the arm's own function that ends in a `TailCall`. The `CBranch` arm is
    // repointed at the trampoline (an intra-function target); the trampoline's tail
    // call carries the callee's id, so no foreign `BlockId` survives. On the clean
    // IR (split runs pre-mem2reg) target blocks have no params, so the arm carries
    // no block arguments to forward.
    for (insn, owner_block, callee) in cond_calls {
        let owner = BasicBlock::from_id(ctx, owner_block)
            .parent()
            .map(|f| f.id)
            .expect("cbranch block has an owner");
        let tramp = BasicBlock::make(ctx, owner).id;
        Builder::from_block(BasicBlock::from_id_mut(ctx, tramp)).push_tail_call(callee);
        ctx.add_cfg_edge(owner_block, tramp);

        let Mnemonic::CBranch(mut cb) = ctx.values.instruction(insn).mnemonic().clone() else {
            continue;
        };
        // Repoint whichever arm(s) targeted this callee's entry at the trampoline.
        if foreign_entry(ctx, cb.success_block, owner) == Some(callee) {
            cb.success_block = tramp;
        }
        if foreign_entry(ctx, cb.failure_block, owner) == Some(callee) {
            cb.failure_block = tramp;
        }
        ctx.replace_instruction_mnemonic(insn, Mnemonic::CBranch(cb));
    }
    for edge in stale {
        ctx.remove_cfg_edge(edge);
    }
    changed
}

/// Force a function split at every terminator target that lands in the *middle* of
/// another function. After [`reattribute_blocks`] settles ownership, a terminator in
/// function `F` may still statically target a block owned by a different function
/// `G` that is not `G`'s entry — a `jmp`/`jcc` into the middle of `G`. Strict IR
/// locality (context-split ruling 2) forbids the resulting foreign `BlockId`, so the
/// landing is promoted to its own function: it becomes an entry, the next
/// [`reattribute_blocks`] round re-claims the tail below it, and
/// [`convert_cross_function_tail_calls`] then encodes the incoming edge as an
/// into-entry tail call (unconditional) or a trampoline (conditional). No block is
/// cloned. Returns `true` if a function was created.
fn promote_cross_function_landings(ctx: &mut Context) -> bool {
    let mut to_promote: Vec<u64> = Vec::new();
    let mut seen: HashSet<u64> = HashSet::default();

    for block in ctx.blocks() {
        let Some(owner) = block.parent().map(|f| f.id) else {
            continue;
        };
        let Some(term) = block.instructions().last() else {
            continue;
        };
        for target in term.mnemonic().target_blocks() {
            let tb = BasicBlock::from_id(ctx, target);
            let Some(g) = tb.parent().map(|f| f.id) else {
                continue;
            };
            // Only a cross-function target that is *not* already `G`'s entry needs a
            // forced split; an into-entry edge is handled by the conversion step.
            if g == owner || Function::from_id(ctx, g).root().map(|r| r.id) == Some(target) {
                continue;
            }
            if let Some(addr) = tb.address()
                && Function::from_addr(ctx, addr).is_none()
                && seen.insert(addr)
            {
                to_promote.push(addr);
            }
        }
    }

    let promoted = !to_promote.is_empty();
    for addr in to_promote {
        Function::make_at_addr(ctx, addr, None);
        log::debug!(target: "split", "forced function split at cross-function landing {addr:#x}");
    }
    promoted
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
        let live: HashSet<BlockId> = Function::from_id(ctx, func)
            .block_ids()
            .into_iter()
            .collect();
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
    use qcode::value::{Function, FunctionId, Instruction};
    use std::borrow::Cow;

    fn block_at(ctx: &mut Context, addr: u64) -> BlockId {
        let f = ctx.anon_function();
        BasicBlock::make(ctx, f).with_address(addr).id
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

        // Split settled ownership and then re-homed the absorbed blocks into G's
        // arena, so identify them by address (their pre-split ids are tombstoned).
        assert_block_addrs(&ctx, f, &[0x1000]);
        assert_block_addrs(&ctx, g, &[0x2000, 0x2005]);
        let g_entry = block_at_addr(&ctx, g, 0x2000);
        assert_eq!(ctx.values.block(g_entry).parent, Some(g));
        assert_eq!(
            ctx.values.block(block_at_addr(&ctx, g, 0x2005)).parent,
            Some(g)
        );
        assert_eq!(ctx.values.functions[g].root, Some(g_entry));

        assert_eq!(addrs(&ctx, f), vec![0x1000]);
        assert_eq!(addrs(&ctx, g), vec![0x2000, 0x2005]);
        let _ = (b0, b1, b2);

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

        // Blocks are re-homed into G by split's tail; identify them by address.
        assert_block_addrs(&ctx, f, &[0x1000]);
        assert_block_addrs(&ctx, g, &[0x2000, 0x2005]);
        assert_eq!(
            ctx.values.block(block_at_addr(&ctx, g, 0x2005)).parent,
            Some(g),
            "the post-call block must be claimed via the materialized fall-through edge",
        );
        let _ = (b0, b1, b2);
    }

    /// A `jmp` from one function into another's entry (a tail call) is rewritten
    /// to a function-level [`TailCall`] terminator carrying the callee's id, and
    /// its inter-procedural CFG edge is stripped: the edge has no intra-procedural
    /// meaning and would make every per-function analysis wander into foreign
    /// blocks. The IR then holds no cross-function block reference. Intra-function
    /// edges are untouched.
    #[test]
    fn strips_tail_call_edge_into_another_function() {
        use qcode::value::insn::{Mnemonic, TailCall};

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

        // Ownership was already correct; split stripped the cross-function edge and
        // re-homed the blocks. Identify them by address (pre-split ids are stale).
        assert_block_addrs(&ctx, f, &[0x1000, 0x1008]);
        assert_block_addrs(&ctx, g, &[0x2000]);
        let entry = block_at_addr(&ctx, f, 0x1000);
        let tail = block_at_addr(&ctx, f, 0x1008);

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

        // The terminator is now a function-level `TailCall` carrying the callee's
        // id — no foreign block reference survives in the IR.
        let term = BasicBlock::from_id(&ctx, tail)
            .instructions()
            .last()
            .map(|i| i.mnemonic().clone());
        assert!(
            matches!(term, Some(Mnemonic::TailCall(TailCall { target, .. })) if target == g),
            "the tail jump must become a TailCall to G, got {term:?}",
        );
        let _ = g_entry;

        // Idempotent: nothing left to strip.
        assert!(!split_overlapping_functions(&mut ctx));
    }

    /// Append a `cbranch` to `block`: taken -> `success`, fall-through -> `failure`,
    /// tagged with machine `addr`. Wires both CFG edges (via `push_cbranch`).
    fn cbranch_at(
        ctx: &mut Context,
        block: BlockId,
        success: BlockId,
        failure: BlockId,
        addr: u64,
    ) {
        let cond = ctx.get_const(1, 1).id();
        let id = Builder::from_block(BasicBlock::from_id_mut(ctx, block))
            .push_cbranch(cond, success, failure)
            .id;
        Instruction::from_id_mut(ctx, id).set_address(addr);
    }

    /// A *conditional* arm into another function's entry cannot become a `TailCall`
    /// in place (a tail call is unconditional). The splitter routes it through a
    /// fresh intra-function trampoline block ending in a `TailCall`: the `CBranch`
    /// arm is repointed at the trampoline, the cross-function edge is stripped, and
    /// no foreign `BlockId` survives. The intra-function fall-through arm is intact.
    #[test]
    fn routes_conditional_cross_function_arm_through_trampoline() {
        use qcode::value::insn::{CBranch, Mnemonic, TailCall};

        let mut ctx = Context::new();

        // F@0x1000: entry(0x1000) if cond -> g_entry(0x2000) else -> cont(0x1008).
        let entry = block_at(&mut ctx, 0x1000);
        let cont = block_at(&mut ctx, 0x1008);
        let g_entry = block_at(&mut ctx, 0x2000);

        cbranch_at(&mut ctx, entry, g_entry, cont, 0x1000);
        return_at(&mut ctx, cont, 0x1008);
        return_at(&mut ctx, g_entry, 0x2000);

        let f = Function::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("f"))).id;
        {
            let mut func = Function::from_id_mut(&mut ctx, f);
            func.set_root(entry).unwrap();
            func.add_block(cont);
        }
        let g = Function::make_at_addr(&mut ctx, 0x2000, Some(Cow::Borrowed("g"))).id;
        Function::from_id_mut(&mut ctx, g)
            .set_root(g_entry)
            .unwrap();

        assert!(split_overlapping_functions(&mut ctx));

        // A trampoline block was minted into F (entry + cont + trampoline = 3).
        // The absorbed blocks are re-homed by split's tail, so resolve by address.
        let f_blocks = Function::from_id(&ctx, f).block_ids();
        assert_eq!(f_blocks.len(), 3, "F gains one trampoline block");
        assert_block_addrs(&ctx, g, &[0x2000]);
        let entry = block_at_addr(&ctx, f, 0x1000);
        let cont = block_at_addr(&ctx, f, 0x1008);

        // The CBranch's success arm now points at an intra-F trampoline; the
        // fall-through arm is unchanged. No arm references a foreign block.
        let Mnemonic::CBranch(CBranch {
            success_block,
            failure_block,
            ..
        }) = BasicBlock::from_id(&ctx, entry)
            .instructions()
            .last()
            .unwrap()
            .mnemonic()
            .clone()
        else {
            panic!("entry must still end in a cbranch");
        };
        assert_eq!(failure_block, cont, "the intra-function arm is untouched");
        assert_ne!(
            BasicBlock::from_id(&ctx, success_block).address(),
            Some(0x2000),
            "the foreign arm was repointed away from G's entry",
        );
        assert_eq!(
            BasicBlock::from_id(&ctx, success_block)
                .parent()
                .map(|f| f.id),
            Some(f),
            "the trampoline lives in F",
        );

        // The trampoline ends in a TailCall to G; no foreign block reference.
        let term = BasicBlock::from_id(&ctx, success_block)
            .instructions()
            .last()
            .map(|i| i.mnemonic().clone());
        assert!(
            matches!(term, Some(Mnemonic::TailCall(TailCall { target, .. })) if target == g),
            "trampoline must tail-call G, got {term:?}",
        );

        // No cross-function CFG edge remains out of entry: it reaches the trampoline
        // and cont, both in F.
        for (_, s) in BasicBlock::from_id(&ctx, entry).successors() {
            assert_eq!(
                BasicBlock::from_id(&ctx, s).parent().map(|f| f.id),
                Some(f),
                "every successor of entry is owned by F",
            );
        }

        // Idempotent: nothing left to route.
        assert!(!split_overlapping_functions(&mut ctx));
    }

    /// Assert `func`'s roster is exactly the blocks at these machine addresses.
    /// Split's tail re-homes reattributed blocks into fresh arena slots, so tests
    /// must identify post-split blocks by their stable machine address, never by a
    /// pre-split `BlockId` (which relocation tombstones).
    fn assert_block_addrs(ctx: &Context, func: FunctionId, expected: &[u64]) {
        let mut got: Vec<u64> = Function::from_id(ctx, func)
            .block_ids()
            .into_iter()
            .filter_map(|b| ctx.values.block(b).address)
            .collect();
        got.sort_unstable();
        let mut want = expected.to_vec();
        want.sort_unstable();
        assert_eq!(got, want, "block-address set mismatch for {func:?}");
    }

    /// Resolve the (post-split, possibly relocated) block of `func` at `addr`.
    fn block_at_addr(ctx: &Context, func: FunctionId, addr: u64) -> BlockId {
        Function::from_id(ctx, func)
            .block_ids()
            .into_iter()
            .find(|b| ctx.values.block(*b).address == Some(addr))
            .unwrap_or_else(|| panic!("{func:?} has no block at {addr:#x}"))
    }

    fn addrs(ctx: &Context, func: FunctionId) -> Vec<u64> {
        ctx.values.functions[func]
            .instruction_addrs
            .iter()
            .copied()
            .collect()
    }

    /// A block *owned* by `F` but *stored* in another function's arena — exactly
    /// the reattributed state `reattribute_blocks`/`lift_block` produce — is
    /// re-homed into `F`'s arena by `split_overlapping_functions`'s tail so that
    /// split always hands back strictly local IR: ownership and storage agree, the
    /// IR (successors, opcodes) is intact, and every function is checkout-safe.
    #[test]
    fn split_rehomes_reattributed_block() {
        use qcode::value::insn::Mnemonic;
        use qcode::value::util::host_mut::CheckedOut;

        let mut ctx = Context::new();

        let f = Function::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("f"))).id;
        let g = Function::make_at_addr(&mut ctx, 0x2000, Some(Cow::Borrowed("g"))).id;

        // entry: stored in F, owned by F. body: born in *G*'s arena, then attached
        // to F (parent = F, storage = G) — the entanglement.
        let entry = BasicBlock::make(&mut ctx, f).with_address(0x1000).id;
        let body = BasicBlock::make(&mut ctx, g).with_address(0x1008).id;
        Function::from_id_mut(&mut ctx, f).add_block(body);
        Function::from_id_mut(&mut ctx, f).set_root(entry).unwrap();

        // `branch_at` (via `push_branch`) already wires the CFG edge entry -> body.
        branch_at(&mut ctx, entry, body, 0x1000);
        return_at(&mut ctx, body, 0x1008);

        // Precondition: `body` is reattributed (stored in G, owned by F).
        assert_eq!(body.func, g);
        assert_eq!(ctx.values.block(body).parent, Some(f));

        // Ownership already matches the CFG, so split's only work is discharging
        // strict locality at its tail — which reports a change (the storage move).
        assert!(
            split_overlapping_functions(&mut ctx),
            "expected split to re-home the reattributed block",
        );

        // (a) Every one of F's roster blocks is now self-stored.
        let f_blocks = Function::from_id(&ctx, f).block_ids();
        assert_eq!(f_blocks.len(), 2);
        for b in &f_blocks {
            assert_eq!(
                b.func, f,
                "F still owns a foreign-stored block {b:?} after normalization",
            );
            assert_eq!(ctx.values.block(*b).parent, Some(f));
        }
        // The old storage in G is tombstoned and no longer owned.
        assert!(ctx.values.block(body).deleted);

        // (b) The CFG and opcodes are preserved: entry still branches to a single
        // successor which is a `return` block, and F's root is still `entry`.
        assert_eq!(ctx.values.functions[f].root, Some(entry));
        let succ: Vec<BlockId> = BasicBlock::from_id(&ctx, entry)
            .successors()
            .map(|(_, s)| s)
            .collect();
        assert_eq!(succ.len(), 1, "entry must keep its single successor");
        let new_body = succ[0];
        assert_eq!(
            new_body.func, f,
            "the relocated body must live in F's arena"
        );
        assert_eq!(ctx.values.block(new_body).address, Some(0x1008));
        let last = BasicBlock::from_id(&ctx, new_body)
            .instructions()
            .last()
            .map(|i| i.mnemonic().clone());
        assert!(
            matches!(last, Some(Mnemonic::Return(_))),
            "relocated body must still end in a return, got {last:?}",
        );

        // (c) F is now checkout-safe: `CheckedOut::new`'s debug-assert holds.
        let mut fun = ctx.checkout_function(f);
        let _co = CheckedOut::new(&mut fun, f, &ctx);
        drop(_co);
        ctx.checkin_function(f, fun);
    }
}
