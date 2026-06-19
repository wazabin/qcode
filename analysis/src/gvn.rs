//! Global value numbering, built from composable sub-passes.
//!
//! The pass is a chain of [`walk::SubPass`]es driven by the generic walkers in
//! [`walk`]: per instruction, each sub-pass is tried in order until one claims
//! it. To add a new sub-pass, implement [`walk::SubPass`] in its own module and
//! add it to the tuple in [`gvn_passes`] (order matters: earlier members see
//! the instruction first).

use crate::AliasResult;

use qcode::{
    context::Context,
    value::{block::BlockMutRef, function::FunctionId, util::base_ref::WithCtxMut},
};

mod affine;
mod cse;
mod flag_idiom;
mod fold;
mod identity;
mod intrinsics;
mod mem_forward;
mod memory;
mod walk;

use cse::Cse;
use flag_idiom::FlagIdiom;
use fold::Fold;
use identity::Identities;
use intrinsics::Recognize;
use memory::MemoryForwarding;
use walk::{run_dominator_walk, run_flat_fixpoint, run_single_block};

/// The full GVN sub-pass chain. Order is load-bearing: memory forwarding must
/// see loads/stores first, folding must run before idiom recognition (so shift
/// amounts and multipliers are constants), intrinsic recognition before the
/// algebraic identities that simplify the intrinsics it produces, and CSE last
/// over already-simplified mnemonics.
fn gvn_passes() -> (
    MemoryForwarding,
    Fold,
    Recognize,
    FlagIdiom,
    Identities,
    Cse,
) {
    (
        MemoryForwarding,
        Fold,
        Recognize,
        FlagIdiom,
        Identities,
        Cse,
    )
}

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
/// [`MemForward::prune_loop_carried`](mem_forward::MemForward::prune_loop_carried)
/// and are wrongly forwarded across loop back-edges.
///
/// Folding only — no CSE or load/store forwarding. Returns `true` if anything
/// changed.
pub fn constant_fold_function(ctx: &mut Context, func_id: FunctionId) -> bool {
    run_flat_fixpoint(ctx, func_id, &(Fold,))
}

/// Single-block GVN pass (preserved for backward compatibility).
///
/// Processes instructions in registry (insertion) order. Loads are GVN-able
/// but are invalidated by intervening stores that may-alias the load pointer
/// according to `aliases`. Pure instructions are always GVN-able.
/// Terminators, calls, and `PCodeOp` are excluded.
pub fn gvn(block: &mut BlockMutRef, aliases: Option<&AliasResult>) {
    let block_id = block.id;
    run_single_block(block.ctx_mut(), block_id, &gvn_passes(), aliases);
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
    run_dominator_walk(ctx, func_id, &gvn_passes(), aliases)
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
        // Per-function pass: build the alias oracle from this function's own
        // instructions, not the whole program, so cost stays O(function) per
        // call instead of O(program) once per function (O(functions × program)
        // across the stage, which dominated on large binaries).
        let aliases = AliasResult::simple_for_function(ctx, fun_id);
        changed |= gvn_function(ctx, fun_id, Some(&aliases));
        Ok(changed)
    }
}

crate::register_function_pass!(Gvn);
