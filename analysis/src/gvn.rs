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
    value::{
        block::BlockMutRef,
        function::FunctionId,
        util::{base_ref::WithCtxMut, host_mut::HostMut},
    },
};

pub(crate) mod affine;
mod array_project;
pub(crate) mod concretize;
pub(crate) mod congruence;
mod cse;
mod emulate_map;
mod flag_idiom;
mod fold;
mod identity;
mod intrinsics;
mod mem_forward;
mod memory;
mod narrow;
mod pure_call;
mod walk;

use cse::Cse;
use flag_idiom::FlagIdiom;
use fold::Fold;
use identity::Identities;
use intrinsics::Recognize;
use memory::MemoryForwarding;
use narrow::NarrowTrunc;
use walk::{run_dominator_walk, run_flat_fixpoint, run_single_block};

/// The full GVN sub-pass chain. Order is load-bearing: memory forwarding must
/// see loads/stores first, folding must run before idiom recognition (so shift
/// amounts and multipliers are constants), intrinsic recognition before the
/// algebraic identities that simplify the intrinsics it produces, and CSE last
/// over already-simplified mnemonics.
///
/// The three body-reading sub-passes (`PureCall`, `EmulateMap`, `ArrayProject`)
/// that used to sit between `NarrowTrunc` and `Recognize` are **not** here: they
/// read pure *callee* bodies, an interprocedural read the parallel-safe function
/// pass contract forbids, so they live in the [`concretize`] module pass.
fn gvn_passes<'str, H: HostMut<'str>>() -> Vec<Box<dyn walk::SubPass<'str, H>>> {
    vec![
        Box::new(MemoryForwarding),
        Box::new(Fold),
        Box::new(NarrowTrunc),
        Box::new(Recognize),
        Box::new(FlagIdiom),
        Box::new(Identities),
        Box::new(Cse),
    ]
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
pub fn constant_fold_function(mut ctx: &mut Context, func_id: FunctionId) -> bool {
    constant_fold_host(&mut ctx, func_id)
}

/// Host-generic core of [`constant_fold_function`]: runs the [`Fold`] sub-pass to
/// a fixpoint over either the whole module (`&mut Context`) or a single
/// checked-out function ([`CheckedOut`]).
///
/// [`CheckedOut`]: qcode::value::util::host_mut::CheckedOut
pub(crate) fn constant_fold_host<'str, H: HostMut<'str>>(
    host: &mut H,
    func_id: FunctionId,
) -> bool {
    run_flat_fixpoint(
        host,
        func_id,
        &[Box::new(Fold) as Box<dyn walk::SubPass<'str, H>>],
    )
}

/// Sink low-word truncations through arithmetic, cancelling widenings, to a
/// fixpoint. Standalone composition of the [`NarrowTrunc`] sub-pass — the same
/// shape as [`constant_fold_function`]. Returns `true` if anything changed.
pub fn narrow_function(mut ctx: &mut Context, func_id: FunctionId) -> bool {
    narrow_host(&mut ctx, func_id)
}

/// Host-generic core of [`narrow_function`]: runs the [`NarrowTrunc`] sub-pass to
/// a fixpoint over either the whole module or a single checked-out function.
pub(crate) fn narrow_host<'str, H: HostMut<'str>>(host: &mut H, func_id: FunctionId) -> bool {
    run_flat_fixpoint(
        host,
        func_id,
        &[Box::new(NarrowTrunc) as Box<dyn walk::SubPass<'str, H>>],
    )
}

/// Single-block GVN pass (preserved for backward compatibility).
///
/// Processes instructions in registry (insertion) order. Loads are GVN-able
/// but are invalidated by intervening stores that may-alias the load pointer
/// according to `aliases`. Pure instructions are always GVN-able.
/// Terminators, calls, and `PCodeOp` are excluded.
pub fn gvn(block: &mut BlockMutRef, aliases: Option<&AliasResult>) {
    let block_id = block.id;
    let mut ctx = block.ctx_mut();
    run_single_block(&mut ctx, block_id, &gvn_passes(), aliases);
}

/// Dominator-tree GVN over an entire function.
///
/// Walks the dominator tree in pre-order, propagating the value table from each
/// block to its dominated successors. A value computed in a dominator is always
/// available to every descendant, so redundant recomputations across blocks are
/// eliminated. Store/load invalidation follows the same alias-aware rules as the
/// single-block pass.
/// Returns `true` if anything changed.
pub fn gvn_function(
    mut ctx: &mut Context,
    func_id: FunctionId,
    aliases: Option<&AliasResult>,
) -> bool {
    gvn_host(&mut ctx, func_id, aliases)
}

/// Host-generic core of [`gvn_function`]: runs the full GVN sub-pass chain over
/// the dominator tree of `func_id`, on either the whole module (`&mut Context`)
/// or a single checked-out function ([`CheckedOut`]).
///
/// [`CheckedOut`]: qcode::value::util::host_mut::CheckedOut
pub(crate) fn gvn_host<'str, H: HostMut<'str>>(
    host: &mut H,
    func_id: FunctionId,
    aliases: Option<&AliasResult>,
) -> bool {
    run_dominator_walk(host, func_id, &gvn_passes(), aliases)
}

// ----- passes ----------------------------------------------------------------

use crate::{ContextView, FunctionBody, FunctionPass};
use qcode::value::util::base_ref::HostRef;

#[derive(Default)]
pub struct ConstFold;

impl FunctionPass for ConstFold {
    const NAME: &'static str = "const_fold";
    fn description(&self) -> &'static str {
        "Fold pointer/integer arithmetic into literals"
    }
    fn run<'str>(
        &self,
        f: &mut FunctionBody<'str>,
        m: ContextView<'_, 'str>,
    ) -> Result<bool, String> {
        let fun_id = f.id();
        let mut host = f.host(m);
        Ok(constant_fold_host(&mut host, fun_id))
    }
}

crate::register_function_pass!(ConstFold);

#[derive(Default)]
pub struct Narrow;

impl FunctionPass for Narrow {
    const NAME: &'static str = "narrow";
    fn description(&self) -> &'static str {
        "Sink low-word truncations through arithmetic, cancelling widenings"
    }
    fn run<'str>(
        &self,
        f: &mut FunctionBody<'str>,
        m: ContextView<'_, 'str>,
    ) -> Result<bool, String> {
        let fun_id = f.id();
        let mut host = f.host(m);
        Ok(narrow_host(&mut host, fun_id))
    }
}

crate::register_function_pass!(Narrow);

/// Build the per-function alias oracle exactly as the V1 `Gvn` pass did, but over
/// the checked-out body (read through `host`). Reuses the shared, function-
/// independent register/varnode alias base and finishes it for just this
/// function's pointers, then supplies the stack pointer so the oracle applies
/// frame freshness (a function's own locals never alias an incoming pointer). The
/// stack pointer is `None` in arch-agnostic envs, leaving frame freshness inert.
fn build_gvn_aliases<'a, 'str: 'a>(
    m: ContextView<'_, 'str>,
    host: HostRef<'a, 'str>,
    fun_id: FunctionId,
) -> AliasResult {
    let ctx = m.shared_ctx();
    let sp_reg = ctx.registers.get(&m.env().cfg.stack_pointer).copied();
    m.env()
        .alias_base(ctx)
        .for_function(host, fun_id)
        .with_frame_freshness(host, fun_id, sp_reg)
}

#[derive(Default)]
pub struct Gvn;

impl FunctionPass for Gvn {
    const NAME: &'static str = "gvn";
    fn description(&self) -> &'static str {
        "Global value numbering and constant folding"
    }
    fn run<'str>(
        &self,
        f: &mut FunctionBody<'str>,
        m: ContextView<'_, 'str>,
    ) -> Result<bool, String> {
        let fun_id = f.id();
        let mut host = f.host(m);
        // Canonicalize pointer arithmetic *before* building the alias oracle, so it
        // sees per-slot `@SP`-rooted stack locations.
        let mut changed = constant_fold_host(&mut host, fun_id);
        // Build the oracle over the (now-canonicalized) checked-out body, then run
        // the dominator-tree GVN against it.
        let aliases = build_gvn_aliases(m, host.read_host(), fun_id);
        changed |= gvn_host(&mut host, fun_id, Some(&aliases));
        Ok(changed)
    }
}

crate::register_function_pass!(Gvn);
