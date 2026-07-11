//! Global value numbering, built from composable sub-passes.
//!
//! The pass is a chain of [`walk::SubPass`]es driven by the generic walkers in
//! [`walk`]: per instruction, each sub-pass is tried in order until one claims
//! it. To add a new sub-pass, implement [`walk::SubPass`] in its own module and
//! add it to the tuple in [`gvn_passes`] (order matters: earlier members see
//! the instruction first).

use crate::{AliasResult, ArchConfig, CallingConvention, PipelineEnv};

use qcode::{
    context::Context,
    value::{
        RegisterId, VarnodeId, block::BlockMutRef, function::FunctionId, util::base_ref::WithCtxMut,
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
use walk::{SubPassC, run_dominator_walk_c, run_flat_fixpoint_c, run_single_block};

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
fn gvn_passes<'str>() -> Vec<Box<dyn walk::ModuleSubPass<'str>>> {
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

/// Concrete twin of [`gvn_passes`] (context-split stage 5b-ii): the same chain in
/// the same order, over the host-free [`SubPassC`] surface, driving the
/// function-pass GVN chain on a checked-out `(&mut FunctionBody, ContextView)`.
fn gvn_passes_c<'str>() -> Vec<Box<dyn SubPassC<'str>>> {
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

/// Bridge a whole-`Context` GVN entry point onto the concrete function-pass core.
///
/// The public entry points ([`gvn_function`], [`constant_fold_function`],
/// [`narrow_function`]) take a `&mut Context` and a bare `func_id`; their callers
/// (unit tests, a couple of module-pass helpers) hold neither a checked-out
/// [`FunctionBody`] nor a [`PipelineEnv`]. This check-out shim runs the concrete
/// `_body` core over `(&mut FunctionBody, ContextView)` — the *same* surface the
/// parallel function-pass driver uses — then checks the body back in and resyncs
/// its call sites, exactly as [`FunctionPassAdapter`](crate::FunctionPassAdapter)
/// does (minus minting: these entry points mint nothing).
///
/// The [`ContextView`]'s [`PipelineEnv`] is a throwaway: the GVN `_body` cores
/// never consult `env()` (their alias oracle is supplied by the caller). Because
/// the concrete `PassBacking` path debug-asserts the body is self-stored, every
/// caller must feed a function with no reattributed blocks — which, post the
/// driver's `split_overlapping_functions` normalization, every production
/// function is (audited 2026-07-11: all direct callers are `#[cfg(test)]`).
fn with_checked_out_body<'str, R>(
    ctx: &mut Context<'str>,
    func_id: FunctionId,
    f: impl FnOnce(&mut FunctionBody<'str>, ContextView<'_, 'str>) -> R,
) -> R {
    let env = throwaway_env();
    let before_targets = ctx.direct_call_targets(func_id);
    let fun = ctx.checkout_function(func_id);
    let mut body = FunctionBody::new(func_id, fun, Vec::new());
    let out = {
        let view = ContextView::new(ctx, &env);
        f(&mut body, view)
    };
    // GVN/fold/narrow buffer no effects and mint nothing, so `into_parts`'
    // effects/minted are empty — drop them (the in-place module walker these
    // entry points used had no effects concept at all).
    let (fun, _effects, _minted, _unused) = body.into_parts();
    ctx.checkin_function(func_id, fun);
    ctx.resync_call_sites(func_id, &before_targets);
    out
}

/// A throwaway [`PipelineEnv`] for the check-out shim: the GVN `_body` cores never
/// read it, so its stack pointer / ABI are placeholders.
fn throwaway_env() -> PipelineEnv {
    PipelineEnv::from_parts(
        ArchConfig {
            stack_pointer: RegisterId::from(0usize),
            dead_flag_regs: Vec::new(),
            abi: CallingConvention::default(),
            os: qcode::context::TargetOs::Unknown,
            bitness: 64,
        },
        VarnodeId::from(0usize),
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
    with_checked_out_body(ctx, func_id, |body, cx| {
        constant_fold_body(body, cx, func_id)
    })
}

/// Concrete core of [`constant_fold_function`] (context-split stage 5b-ii): runs
/// the [`Fold`] sub-pass to a fixpoint over a checked-out `(&mut FunctionBody,
/// ContextView)` with no threaded mutation host.
fn constant_fold_body<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    func_id: FunctionId,
) -> bool {
    run_flat_fixpoint_c(body, cx, func_id, &[Box::new(Fold) as Box<dyn SubPassC>])
}

/// Sink low-word truncations through arithmetic, cancelling widenings, to a
/// fixpoint. Standalone composition of the [`NarrowTrunc`] sub-pass — the same
/// shape as [`constant_fold_function`]. Returns `true` if anything changed.
pub fn narrow_function(ctx: &mut Context, func_id: FunctionId) -> bool {
    with_checked_out_body(ctx, func_id, |body, cx| narrow_body(body, cx, func_id))
}

/// Concrete core of [`narrow_function`] (context-split stage 5b-ii): runs the
/// [`NarrowTrunc`] sub-pass to a fixpoint over a checked-out `(&mut FunctionBody,
/// ContextView)`.
fn narrow_body<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    func_id: FunctionId,
) -> bool {
    run_flat_fixpoint_c(
        body,
        cx,
        func_id,
        &[Box::new(NarrowTrunc) as Box<dyn SubPassC>],
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
    let ctx = block.ctx_mut();
    run_single_block(ctx, block_id, &gvn_passes(), aliases);
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
    with_checked_out_body(ctx, func_id, |body, cx| {
        gvn_body(body, cx, func_id, aliases)
    })
}

/// Concrete core of [`gvn_function`] (context-split stage 5b-ii): runs the full
/// GVN sub-pass chain over the dominator tree of `func_id` on a checked-out
/// `(&mut FunctionBody, ContextView)` with no threaded mutation host.
fn gvn_body<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    func_id: FunctionId,
    aliases: Option<&AliasResult>,
) -> bool {
    run_dominator_walk_c(body, cx, func_id, &gvn_passes_c(), aliases)
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
        Ok(constant_fold_body(f, m, fun_id))
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
        Ok(narrow_body(f, m, fun_id))
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
    let shared = m.shr();
    let sp_reg = shared.registers.get(&m.env().cfg.stack_pointer).copied();
    m.env()
        .alias_base(shared)
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
        // Canonicalize pointer arithmetic *before* building the alias oracle, so it
        // sees per-slot `@SP`-rooted stack locations.
        let mut changed = constant_fold_body(f, m, fun_id);
        // Build the oracle over the (now-canonicalized) body, then run the
        // dominator-tree GVN against it.
        let aliases = build_gvn_aliases(m, f.read_host(m), fun_id);
        changed |= gvn_body(f, m, fun_id, Some(&aliases));
        Ok(changed)
    }
}

crate::register_function_pass!(Gvn);
