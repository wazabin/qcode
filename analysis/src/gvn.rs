//! Global value numbering, built from composable sub-passes.
//!
//! The pass is a chain of [`walk::SubPass`]es driven by the generic walkers in
//! [`walk`]: per instruction, each sub-pass is tried in order until one claims
//! it. To add a new sub-pass, implement [`walk::SubPass`] in its own module and
//! add it to the tuple in [`gvn_passes`] (order matters: earlier members see
//! the instruction first).

use crate::{AliasResult, with_checked_out_body};

use qcode::{
    context::Context,
    value::{
        ValueId,
        block::BlockMutRef,
        function::FunctionId,
        insn::{InstructionId, Mnemonic},
        util::base_ref::WithCtxMut,
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
use walk::{SubPassC, run_dominator_walk_c, run_flat_fixpoint_c, run_single_block_c};

/// Current instruction context for one deterministic module-pass sweep.
///
/// Unlike the GVN walker context, this carries no aliases, numbering, or
/// inherited state. Cross-body module passes snapshot IDs once, then read each
/// instruction immediately before visiting it so earlier use-rewrites are
/// visible. Instructions created during the sweep are left for the next
/// pipeline fixpoint iteration.
struct ModuleInsn {
    block_id: qcode::value::block::BlockId,
    insn_id: InstructionId,
    id: ValueId,
    size: usize,
    mnemonic: Mnemonic,
}

/// Snapshot every instruction ID in eligible functions in function-roster,
/// block-roster, and program order. This is traversal only, not another walker:
/// the independent module passes own all recognition and mutation.
fn module_instruction_snapshot(ctx: &Context) -> Vec<InstructionId> {
    let fun_ids: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| !f.is_external())
        .filter(|f| !ctx.is_function_ignored(f.address()))
        .map(|f| f.id)
        .collect();

    let mut snapshot = Vec::new();
    for fun_id in fun_ids {
        for block_id in FunctionBody::from_id(ctx, fun_id).block_ids() {
            for insn_id in qcode::value::BasicBlock::from_id(ctx, block_id).instruction_ids() {
                snapshot.push(insn_id);
            }
        }
    }
    snapshot
}

/// Read the instruction after any earlier rewrites in the same sweep.
fn module_insn(ctx: &Context, insn_id: InstructionId) -> ModuleInsn {
    let insn = ctx.insn_ref(insn_id);
    ModuleInsn {
        block_id: insn
            .parent()
            .expect("module-pass instruction is attached")
            .id,
        insn_id,
        id: ValueId::Instruction(insn_id),
        size: insn.size(),
        mnemonic: insn.mnemonic().clone(),
    }
}

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
///
/// The chain runs over the host-free [`SubPassC`] surface on a checked-out
/// `(&mut FunctionBody, ContextView)`; both the whole-function entry points and
/// the single-block [`gvn`] drive it through the check-out shim.
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
    // The block's owning (storage) function — self-stored, so `id.func` is the
    // function to check out and run the concrete single-block core against.
    let func_id = block_id.func;
    let ctx = block.ctx_mut();
    let _ = with_checked_out_body(ctx, func_id, |body, cx| {
        run_single_block_c(body, cx, block_id, &gvn_passes_c(), aliases)
    });
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

use crate::{ContextView, FunctionBody, FunctionPass, Outcome};
use qcode::value::QCodeView;

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
        _next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String> {
        let fun_id = f.id();
        Ok(Outcome::changed(constant_fold_body(f, m, fun_id)))
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
        _next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String> {
        let fun_id = f.id();
        Ok(Outcome::changed(narrow_body(f, m, fun_id)))
    }
}

crate::register_function_pass!(Narrow);

/// Build the per-function alias oracle exactly as the V1 `Gvn` pass did, but over
/// the checked-out body (read through `host`). Reuses the shared, function-
/// independent register/varnode alias base and finishes it for just this
/// function's pointers, then supplies the stack pointer so the oracle applies
/// frame freshness (a function's own locals never alias an incoming pointer). The
/// stack pointer is `None` in arch-agnostic envs, leaving frame freshness inert.
fn build_gvn_aliases<'ctx, 'str: 'ctx>(
    m: ContextView<'_, 'str>,
    host: impl QCodeView<'ctx, 'str>,
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
        _next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String> {
        let fun_id = f.id();
        // Canonicalize pointer arithmetic *before* building the alias oracle, so it
        // sees per-slot `@SP`-rooted stack locations.
        let mut changed = constant_fold_body(f, m, fun_id);
        // Build the oracle over the (now-canonicalized) body, then run the
        // dominator-tree GVN against it.
        let aliases = build_gvn_aliases(m, m.body_view(f), fun_id);
        changed |= gvn_body(f, m, fun_id, Some(&aliases));
        Ok(Outcome::changed(changed))
    }
}

crate::register_function_pass!(Gvn);
