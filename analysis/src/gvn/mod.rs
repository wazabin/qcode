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

use integer::{algebraic_identity, constant_folding};
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

                if let Some(cst) = constant_folding(ctx, &mnemonic, size) {
                    ctx.replace_all_uses_with(id, cst);
                    redundant.insert(insn_id);
                    changed = true;
                } else if let Some(simplified) = algebraic_identity(ctx, &mnemonic, size) {
                    ctx.replace_all_uses_with(id, simplified);
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
) {
    let inherited = prune_loop_carried_loads(ctx, block_id, inherited, tree, aliases);
    let updated = gvn_block_inner(ctx, block_id, inherited.as_ref(), aliases);
    // A block that ends in a call clobbers registers: its dominated children run
    // after the call, so register `Load` leaders the call clobbers must not be
    // forwarded into them (e.g. a caller's post-call `RAX` read is the callee's
    // result, not a value computed before the call).
    let for_children = prune_clobbered_by_call(ctx, block_id, &updated, aliases);
    for &child in tree.children_of(block_id) {
        gvn_block_rec(ctx, child, for_children.as_ref(), tree, aliases);
    }
}

/// Dominator-tree GVN over an entire function.
///
/// Walks the dominator tree in pre-order, propagating the value table from each
/// block to its dominated successors. A value computed in a dominator is always
/// available to every descendant, so redundant recomputations across blocks are
/// eliminated. Store/load invalidation follows the same alias-aware rules as the
/// single-block pass.
pub fn gvn_function(ctx: &mut Context, func_id: FunctionId, aliases: Option<&AliasResult>) {
    let root = match ctx.values.functions[func_id].root {
        Some(r) => r,
        None => return,
    };

    let tree = compute_dominators(ctx, root);
    gvn_block_rec(ctx, root, &HashMap::new(), &tree, aliases);

    // Blocks unreachable from the function root are not covered by the walk
    // above. The common case is the fall-through after a `call`: a `Call` is a
    // block terminator, but the lifter records no CFG edge from the call site
    // back to its return block, so the entire post-call region (register reload
    // chains, the epilogue, ...) is orphaned. Optimize each such region with its
    // own dominator tree, rooted at its entry — an unreachable block with no
    // predecessor. Once calls grow a proper return edge this loop simply finds
    // nothing to do.
    let is_reachable = |b: BlockId| b == root || tree.dominates(root, b);
    let block_ids: Vec<BlockId> = Function::from_id(ctx, func_id)
        .iter()
        .map(|block| block.id)
        .collect();

    for block_id in block_ids {
        if is_reachable(block_id) {
            continue;
        }
        // Only start at region entries; interior blocks are reached by the
        // sub-walk from their entry.
        if BasicBlock::from_id(ctx, block_id)
            .predecessors()
            .next()
            .is_some()
        {
            continue;
        }
        let subtree = compute_dominators(ctx, block_id);
        gvn_block_rec(ctx, block_id, &HashMap::new(), &subtree, aliases);
    }
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
        constant_fold_function(ctx, fun_id);
        let aliases = AliasResult::simple(ctx);
        gvn_function(ctx, fun_id, Some(&aliases));
        Ok(false)
    }
}

crate::register_function_pass!(Gvn);
