use std::collections::{HashMap, HashSet};

use jstd::graph::analysis::{DominatorTree, compute_dominators};

use crate::AliasResult;

use qcode::{
    context::Context,
    value::{
        BasicBlock, Function, ValueId,
        block::{BlockId, BlockMutRef},
        function::FunctionId,
        insn::{InstructionId, Mnemonic},
        util::base_ref::WithCtxMut,
    },
};

mod integer;
mod memory;

#[cfg(test)]
mod tests;

use integer::try_fold;
use memory::{gvn_block_inner, prune_clobbered_by_call, prune_loop_carried_loads};

/// Constant-fold every foldable instruction in `func_id` to interned literals,
/// iterating to a fixpoint.
///
/// Canonicalizes pointer arithmetic (e.g. the `stack_base - 8` then `+ offset`
/// chains left by brighten+mem2reg) into single stack-address literals so an
/// [`AliasResult`](crate::AliasResult) built *afterwards* sees per-slot locations
/// instead of collapsing every slot onto the shared `stack_base` root. This keeps
/// the oracle consistent with the pointers [`gvn_function`] reasons about: without
/// it, GVN folds these adds into fresh literals the precomputed oracle has never
/// seen, so loop-carried stack stores become invisible to
/// [`prune_loop_carried_loads`] and are wrongly forwarded across loop back-edges.
///
/// Folding only — no CSE or load/store forwarding. Returns `true` if anything
/// changed.
pub fn constant_fold_function(ctx: &mut Context, func_id: FunctionId) -> bool {
    let block_ids: Vec<BlockId> = Function::from_id(ctx, func_id)
        .iter()
        .map(|block| block.id)
        .collect();

    let mut changed_any = false;
    loop {
        let mut changed = false;
        for &block_id in &block_ids {
            let insns = BasicBlock::from_id(ctx, block_id)
                .instruction_ids()
                .to_vec();
            let mut redundant: HashSet<InstructionId> = HashSet::new();

            for insn_id in insns {
                let insn = ctx.get_insn(insn_id);
                let id = insn.id();
                let size = insn.size();
                let mnemonic = insn.mnemonic().clone();

                if mnemonic.is_terminator() || size == 0 {
                    continue;
                }

                if let Some(folded) = try_fold(ctx, &mnemonic, size) {
                    ctx.replace_all_uses_with(id, folded);
                    redundant.insert(insn_id);
                    changed = true;
                }
            }

            if !redundant.is_empty() {
                BasicBlock::from_id_mut(ctx, block_id)
                    .retain_insns(|insn| !redundant.contains(insn));
            }
        }
        changed_any |= changed;
        if !changed {
            break;
        }
    }
    changed_any
}

/// Single-block GVN pass (preserved for backward compatibility).
///
/// Processes instructions in registry (insertion) order. Loads are GVN-able
/// but are invalidated by intervening stores that may-alias the load pointer
/// according to `aliases`. Pure instructions are always GVN-able.
/// Terminators, calls, and `PCodeOp` are excluded.
pub fn gvn(block: &mut BlockMutRef, aliases: Option<&AliasResult>) {
    let block_id = block.id;
    gvn_block_inner(block.ctx_mut(), block_id, &HashMap::new(), aliases);
}

fn gvn_block_rec(
    ctx: &mut Context,
    block_id: BlockId,
    inherited: &HashMap<Mnemonic, ValueId>,
    tree: &DominatorTree<BlockId>,
    aliases: Option<&AliasResult>,
    shared: &HashSet<BlockId>,
    changed: &mut bool,
) {
    // A block reachable from more than one walk entry is not truly dominated by
    // anything in this walk's entry-local dominator tree (control can arrive
    // via the other entry), so no values may be forwarded into it.
    let empty = HashMap::new();
    let inherited = if shared.contains(&block_id) {
        &empty
    } else {
        inherited
    };
    let inherited = prune_loop_carried_loads(ctx, block_id, inherited, tree, aliases);
    let (updated, block_changed) = gvn_block_inner(ctx, block_id, inherited.as_ref(), aliases);
    *changed |= block_changed;
    // A block that ends in a call clobbers registers: its dominated children run
    // after the call, so register `Load` leaders the call clobbers must not be
    // forwarded into them (e.g. a caller's post-call `RAX` read is the callee's
    // result, not a value computed before the call).
    let for_children = prune_clobbered_by_call(ctx, block_id, &updated, aliases);
    for &child in tree.children_of(block_id) {
        gvn_block_rec(
            ctx,
            child,
            for_children.as_ref(),
            tree,
            aliases,
            shared,
            changed,
        );
    }
}

/// All blocks reachable from `entry` via CFG successor edges (including `entry`).
fn reachable_from(ctx: &Context, entry: BlockId) -> HashSet<BlockId> {
    let mut seen = HashSet::from([entry]);
    let mut stack = vec![entry];
    while let Some(block) = stack.pop() {
        for (_, succ) in BasicBlock::from_id(ctx, block).successors() {
            if seen.insert(succ) {
                stack.push(succ);
            }
        }
    }
    seen
}

/// Dominator-tree GVN over an entire function.
///
/// Walks the dominator tree in pre-order, propagating the value table from each
/// block to its dominated successors. A value computed in a dominator is always
/// available to every descendant, so redundant recomputations across blocks are
/// eliminated. Store/load invalidation follows the same alias-aware rules as the
/// single-block pass.
/// Returns `true` if anything changed.
pub fn gvn_function(ctx: &mut Context, func_id: FunctionId, aliases: Option<&AliasResult>) -> bool {
    let root = match ctx.values.functions[func_id].root {
        Some(r) => r,
        None => return false,
    };

    // Blocks unreachable from the function root are not covered by the root
    // walk. The common case is the fall-through after a `call`: a `Call` is a
    // block terminator, but the lifter records no CFG edge from the call site
    // back to its return block, so the entire post-call region (register reload
    // chains, the epilogue, ...) is orphaned. Optimize each such region with its
    // own dominator tree, rooted at its entry — an unreachable block with no
    // predecessor. Once calls grow a proper return edge this finds nothing to do.
    let root_reachable = reachable_from(ctx, root);
    let entries: Vec<BlockId> = Function::from_id(ctx, func_id)
        .iter()
        .filter(|block| !root_reachable.contains(&block.id))
        .filter(|block| block.predecessors().next().is_none())
        .map(|block| block.id)
        .collect();

    // Blocks reachable from more than one entry get no forwarded values: each
    // walk's dominator tree only sees its own entry's edges, so its dominance
    // claims are invalid for blocks the other entries can also reach.
    let mut seen_count: HashMap<BlockId, u32> = HashMap::new();
    for &entry in std::iter::once(&root).chain(&entries) {
        for block in reachable_from(ctx, entry) {
            *seen_count.entry(block).or_default() += 1;
        }
    }
    let shared: HashSet<BlockId> = seen_count
        .into_iter()
        .filter_map(|(block, count)| (count > 1).then_some(block))
        .collect();

    let mut changed = false;
    let tree = compute_dominators(ctx, root);
    gvn_block_rec(
        ctx,
        root,
        &HashMap::new(),
        &tree,
        aliases,
        &shared,
        &mut changed,
    );

    for entry in entries {
        let subtree = compute_dominators(ctx, entry);
        gvn_block_rec(
            ctx,
            entry,
            &HashMap::new(),
            &subtree,
            aliases,
            &shared,
            &mut changed,
        );
    }
    changed
}

// ----- passes ----------------------------------------------------------------

use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct ConstFold;

impl FunctionPass for ConstFold {
    const NAME: &'static str = "const_fold";
    fn description(&self) -> &'static str {
        "Fold pointer/integer arithmetic into literals"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        Ok(constant_fold_function(ctx, fun_id))
    }
}

crate::register_function_pass!(ConstFold);

#[derive(Default)]
pub struct Gvn;

impl FunctionPass for Gvn {
    const NAME: &'static str = "gvn";
    fn description(&self) -> &'static str {
        "Global value numbering and constant folding"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        // Canonicalize pointer arithmetic into literals *before* building the alias
        // oracle, so it sees per-slot stack locations rather than collapsing them
        // onto `stack_base`.
        let mut changed = constant_fold_function(ctx, fun_id);
        let aliases = AliasResult::simple(ctx);
        changed |= gvn_function(ctx, fun_id, Some(&aliases));
        Ok(changed)
    }
}

crate::register_function_pass!(Gvn);
