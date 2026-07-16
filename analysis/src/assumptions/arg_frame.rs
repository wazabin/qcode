//! The [`Proposition::ArgsDisjointFromCallerFrame`] assumption: a function's
//! incoming pointer arguments never point into its own *caller-frame* slots
//! (`@SP + k`, `k ≥ 0` — the return-address slot and incoming stack arguments).
//!
//! - [`assume_args_disjoint_caller_frame`] is the *make* pass: it records the
//!   assumption `Assumed=true` for every eligible function so the frame-freshness
//!   alias rule (see [`crate::alias::AliasResult::provably_disjoint`]) can forward
//!   a caller-frame slot load across a store through an incoming pointer (the
//!   spilled-pointer reload idiom).
//! - [`verify_args_disjoint_caller_frame`] is the *verify* pass. It is **lenient
//!   by default**: a pointer argument is assumed not to collide with the argument
//!   slots unless a collision can be *explicitly proven*. The call passes the
//!   callee's frame base as its `@ESP` argument, so the callee's caller-frame is a
//!   constant `@SP`-relative interval in the *caller*; a dereferenced pointer arg
//!   that resolves to an overlapping interval is a provable collision and proves
//!   the assumption *false* (a contradiction the checkpoint+replay driver rolls
//!   back). Anything unresolvable (unknown caller, no caller `@SP`, a
//!   non-`@SP`-rooted arg) is *not* proof of collision and leaves the assumption
//!   standing.
//!
//! The assumption is **not** statically sound on its own — hence the verifier and
//! the replay safety net, exactly as for [`Proposition::FunctionReturns`].

use std::collections::{HashMap, HashSet};

use qcode::{
    assumption::{Certainty, Proposition},
    context::Context,
    pass_scope,
    value::{FunctionBody, FunctionId, Instruction, ValueId, VarnodeId, insn::Mnemonic},
};

use crate::gvn::affine::{Numbering, precompute_forms};
use crate::stack::frame::{FrameClass, frame_class, frame_offset, incoming_sp_param};
use crate::{CallGraph, calls::direct_call_sites};

/// Every function whose address is used as a value (and so may be reached by an
/// indirect call this pass cannot see). Computed in one pass over all
/// instructions; mirrors `argpromote`'s closed-world gate.
fn address_taken_set(ctx: &Context) -> HashSet<FunctionId> {
    let mut set = HashSet::new();
    for insn in ctx.instructions() {
        for arg in insn.mnemonic().args().iter().copied() {
            if let qcode::value::LocalValueId::Function(f) = arg {
                set.insert(f);
            }
        }
    }
    set
}

/// A function is eligible for the assumption when it has a body, is not
/// address-taken (so every caller is a direct call we can vet), and has an
/// incoming `@SP` param.
///
/// We deliberately do **not** require a non-`@SP` pointer param to already exist.
/// The *make* pass runs before the pipeline, and the incoming pointer params this
/// assumption protects (`@SP_val_k`, the by-ref→by-value slots) are minted later,
/// by the in-pipeline `argpromote` pass. Gating on params present at make-time
/// would reject exactly the cdecl functions argpromote is about to promote, so the
/// assumption would never be standing when memory forwarding needs it. Recording
/// it for every `@SP`-param function is harmless for those that never gain a
/// pointer param — the frame-freshness alias rule only consults it when querying an
/// input-derived pointer.
fn eligible(
    ctx: &Context,
    fid: FunctionId,
    sp_reg: VarnodeId,
    taken: &HashSet<FunctionId>,
) -> bool {
    let f = FunctionBody::from_id(ctx, fid);
    if f.is_external() || f.root().is_none() {
        return false;
    }
    if taken.contains(&fid) {
        return false;
    }
    incoming_sp_param(qcode::value::ModuleView::new(ctx), fid, sp_reg).is_some()
}

/// *Make* pass — assume [`Proposition::ArgsDisjointFromCallerFrame`] for every
/// eligible function. Returns how many were assumed this round.
pub fn assume_args_disjoint_caller_frame(ctx: &mut Context, sp_reg: Option<VarnodeId>) -> usize {
    assume_args_disjoint_caller_frame_changed_functions(ctx, sp_reg).len()
}

fn assume_args_disjoint_caller_frame_changed_functions(
    ctx: &mut Context,
    sp_reg: Option<VarnodeId>,
) -> rustc_hash::FxHashSet<FunctionId> {
    let _scope = pass_scope::enter("assume_args_disjoint_caller_frame");
    let Some(sp_reg) = sp_reg else {
        return rustc_hash::FxHashSet::default();
    };
    let taken = address_taken_set(ctx);
    let fids: Vec<FunctionId> = ctx.function_ids();
    let mut changed = rustc_hash::FxHashSet::default();
    for fid in fids {
        if eligible(ctx, fid, sp_reg, &taken) {
            let first = assume_true_if_new(ctx, Proposition::ArgsDisjointFromCallerFrame(fid));
            // Same eligibility and consumer (the memory-forwarding alias rule), so
            // record the loaded-pointer-vs-slot assumption here too — it unblocks
            // forwarding the spilled buffer-pointer reload argpromote depends on.
            let second = assume_true_if_new(ctx, Proposition::LoadedPointerDisjointFromSlot(fid));
            if first || second {
                changed.insert(fid);
            }
        }
    }
    qcode::pass_log!(
        debug,
        "assumed caller-frame facts for {} functions",
        changed.len()
    );
    changed
}

/// Record a true assumption only when the proposition has no truth yet.
///
/// [`Context::assume_true`] returns `true` for both a new assumption and an
/// existing truth with the same polarity. That is useful to callers asking
/// whether an assumption is accepted, but a pass outcome must report only an
/// actual truth-map mutation as changed.
fn assume_true_if_new(ctx: &mut Context, prop: Proposition) -> bool {
    ctx.truth(prop).is_none() && ctx.assume_true(prop)
}

/// *Verify* pass — prove every assumed `ArgsDisjointFromCallerFrame` true or
/// false via [`Context::set_known`](Context::set_known). Returns the number of
/// newly-proven facts (novel knowledge for the replay driver's convergence test).
pub fn verify_args_disjoint_caller_frame(ctx: &mut Context, sp_reg: Option<VarnodeId>) -> usize {
    let _scope = pass_scope::enter("verify_args_disjoint_caller_frame");
    let assumed: Vec<FunctionId> = ctx
        .truths()
        .filter(|(_, t)| t.certainty == Certainty::Assumed)
        .filter_map(|(p, _)| match p {
            Proposition::ArgsDisjointFromCallerFrame(f) => Some(f),
            _ => None,
        })
        .collect();
    if assumed.is_empty() {
        return 0;
    }
    // Cache each caller's `(@SP param, affine numbering)` — `None` if it has no
    // incoming `@SP` param (it touched no stack, so it cannot pass a stack address).
    let mut cache: HashMap<FunctionId, Option<(ValueId, Numbering)>> = HashMap::new();
    let graph = CallGraph::analyze(ctx);
    let mut novel = 0;
    for callee in assumed {
        // Default to holding: the assumption is refuted only by a *provable*
        // collision (a resolved interval overlap). An unresolvable caller/offset
        // is not proof of a collision, so it leaves the assumption standing.
        let holds =
            sp_reg.is_none_or(|sp| !args_provably_collide(ctx, &graph, callee, sp, &mut cache));
        if ctx.set_known(Proposition::ArgsDisjointFromCallerFrame(callee), holds) {
            novel += 1;
            qcode::pass_log!(
                debug,
                "proved ArgsDisjointFromCallerFrame({}) = {holds}",
                FunctionBody::from_id(ctx, callee).name(),
            );
        }
    }
    novel
}

/// Whether some direct call site *provably* passes a dereferenced pointer arg
/// that overlaps `callee`'s own caller-frame region — the only way to refute the
/// assumption. The call passes the callee's frame base as its `@ESP` argument, so
/// the callee caller-frame is a constant `@SP`-relative interval in the caller;
/// each dereferenced pointer arg is likewise a constant interval. A *resolved*
/// interval overlap is proof of collision (`true`); anything we cannot resolve
/// (unknown caller, no caller `@SP`, non-`@SP`-rooted arg) is **not** proof, so it
/// leaves the assumption standing — pointers are assumed disjoint from the
/// argument slots unless we can show otherwise.
fn args_provably_collide(
    ctx: &Context,
    graph: &CallGraph,
    callee: FunctionId,
    sp_reg: VarnodeId,
    cache: &mut HashMap<FunctionId, Option<(ValueId, Numbering)>>,
) -> bool {
    let Some(callee_sp) = incoming_sp_param(qcode::value::ModuleView::new(ctx), callee, sp_reg)
    else {
        return false;
    };
    let callee_numbering = precompute_forms(qcode::value::ModuleView::new(ctx), callee);
    // The callee's own caller-frame footprint, `[0, frame_ext)` from its `@ESP`.
    let frame_ext = caller_frame_extent(ctx, callee, callee_sp, &callee_numbering);
    // Root params (in lockstep with `Call.args`); the `@ESP` param index; and the
    // access extent through each param (`None` = not a dereferenced pointer).
    let params: Vec<ValueId> = FunctionBody::from_id(ctx, callee)
        .root()
        .map(|r| r.params().map(|p| p.id()).collect())
        .unwrap_or_default();
    let Some(esp_idx) = params.iter().position(|&p| p == callee_sp) else {
        return false;
    };
    let ptee: Vec<Option<i64>> = params
        .iter()
        .map(|&p| param_access_extent(ctx, callee, p, &callee_numbering))
        .collect();

    for call_id in direct_call_sites(ctx, graph, callee) {
        let Mnemonic::Call(c) = ctx.get_insn(call_id).mnemonic().clone() else {
            continue;
        };
        // Unknown enclosing function — can't locate a frame to collide with.
        let Some(caller) = Instruction::from_id(ctx, call_id)
            .block()
            .and_then(|b| b.function())
            .map(|f| f.id)
        else {
            continue;
        };
        let frame = cache.entry(caller).or_insert_with(|| {
            incoming_sp_param(qcode::value::ModuleView::new(ctx), caller, sp_reg).map(|sp| {
                (
                    sp,
                    precompute_forms(qcode::value::ModuleView::new(ctx), caller),
                )
            })
        });
        // No caller `@SP`: can't place the callee frame in caller offsets — no proof.
        let Some((caller_sp, numbering)) = frame else {
            continue;
        };
        // The callee's frame base in caller `@SP` offsets, from the `@ESP` argument.
        let Some(base_arg) = c.args.get(esp_idx).map(|a| a.qualify(call_id.func)) else {
            continue;
        };
        let Some(base_off) = frame_offset(
            qcode::value::ModuleView::new(ctx),
            numbering,
            *caller_sp,
            base_arg,
        ) else {
            continue;
        };
        for (i, &arg) in c.args.iter().enumerate() {
            if i == esp_idx {
                continue;
            }
            // Only a dereferenced-pointer param carries an aliasing concern.
            let Some(Some(pe)) = ptee.get(i).copied() else {
                continue;
            };
            // A non-`@SP`-rooted arg (global/heap) can't be shown to collide.
            let Some(a_off) = frame_offset(
                qcode::value::ModuleView::new(ctx),
                numbering,
                *caller_sp,
                arg.qualify(call_id.func),
            ) else {
                continue;
            };
            // Resolved overlap of `[a_off, a_off+pe)` with `[base_off, base_off+frame_ext)`
            // — a provable collision.
            if a_off < base_off + frame_ext && base_off < a_off + pe {
                return true;
            }
        }
    }
    false
}

/// Max `(offset + size)` over `fid`'s loads/stores whose pointer is an `@SP`-rooted
/// caller-frame slot (`@SP + k`, `k ≥ 0`): the callee's caller-frame footprint.
fn caller_frame_extent(ctx: &Context, fid: FunctionId, sp: ValueId, numbering: &Numbering) -> i64 {
    let mut ext = 0i64;
    for block in FunctionBody::from_id(ctx, fid).blocks() {
        for insn in block.iter() {
            let (ptr, size) = match insn.mnemonic() {
                Mnemonic::Load(l) => (l.ptr.qualify(insn.id.func), l.size),
                Mnemonic::Store(s) => (s.ptr.qualify(insn.id.func), s.size),
                _ => continue,
            };
            if frame_class(qcode::value::ModuleView::new(ctx), numbering, sp, ptr)
                == Some(FrameClass::CallerFrame)
                && let Some(off) =
                    frame_offset(qcode::value::ModuleView::new(ctx), numbering, sp, ptr)
            {
                ext = ext.max(off + size as i64);
            }
        }
    }
    ext
}

/// Max `(offset + size)` over `fid`'s loads/stores based on `param` (`param ± k`),
/// or `None` if `param` is never a load/store base — i.e. not a dereferenced
/// pointer, so it carries no aliasing concern.
fn param_access_extent(
    ctx: &Context,
    fid: FunctionId,
    param: ValueId,
    numbering: &Numbering,
) -> Option<i64> {
    let mut ext: Option<i64> = None;
    for block in FunctionBody::from_id(ctx, fid).blocks() {
        for insn in block.iter() {
            let (ptr, size) = match insn.mnemonic() {
                Mnemonic::Load(l) => (l.ptr.qualify(insn.id.func), l.size),
                Mnemonic::Store(s) => (s.ptr.qualify(insn.id.func), s.size),
                _ => continue,
            };
            let (base, off) = numbering
                .base_offset(qcode::value::ModuleView::new(ctx), ptr)
                .unwrap_or((ptr, 0));
            if base == param && off >= 0 {
                let e = off + size as i64;
                ext = Some(ext.map_or(e, |x| x.max(e)));
            }
        }
    }
    ext
}

// ----- pass -----------------------------------------------------------------

use crate::{Pass, PipelineEnv};

/// Module pass wrapper for [`assume_args_disjoint_caller_frame`].
///
/// It must run *inside* the pipeline, not in the pre-pipeline driver: the
/// incoming `@SP` param it keys on is minted by `argpromote_registers`/`mem2reg`
/// during the run, so before those stages a function has *no* params at all and
/// nothing is eligible. The pipeline schedules this immediately before each
/// `argpromote` stage — the consumer of the assumption — so the `@SP` param
/// exists and the truth is recorded before the memory forwarding that reads it.
#[derive(Default)]
pub struct AssumeArgFrame;

impl Pass for AssumeArgFrame {
    const NAME: &'static str = "assume_arg_frame";
    fn description(&self) -> &'static str {
        "Assume each function's incoming pointer args are disjoint from its caller-frame slots"
    }
    fn run(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        Ok(crate::ModulePassOutcome::functions(
            assume_args_disjoint_caller_frame_changed_functions(ctx, env.sp_varnode),
        )
        .preserving_global::<crate::CallGraphAnalysis>()
        .preserving_global::<crate::AddressAnalysis>())
    }
}

crate::register_module_pass!(AssumeArgFrame);

#[cfg(test)]
mod tests {
    use qcode::value::QCodeMut;
    use qcode::{
        testing::TestContext,
        value::{BasicBlock, BlockId, insn::Call},
    };
    use qcode_macro::qcode;

    use super::*;

    /// Patch the `idx`-th param of `block` to carry `origin = sp_reg`, so
    /// `incoming_sp_param` recognizes it as the `@SP` param.
    fn make_sp_param(tc: &mut TestContext, block: BlockId, idx: usize, sp_reg: VarnodeId) {
        let pv = BasicBlock::from_id(&tc.ctx, block)
            .params()
            .nth(idx)
            .unwrap()
            .id();
        if let ValueId::BlockParam(inner) = pv {
            tc.ctx
                .block_param_mut(inner)
                .set_origin_id(ValueId::Varnode(sp_reg).localize(inner.func));
        }
    }

    /// Give the (single) call in `block` a target and args.
    fn set_call(tc: &mut TestContext, block: BlockId, target: FunctionId, args: Vec<ValueId>) {
        let call_id = BasicBlock::from_id(&tc.ctx, block)
            .iter()
            .find(|i| matches!(i.mnemonic(), Mnemonic::Call(_)))
            .unwrap()
            .id;
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: qcode::value::insn::Callee::Real(target),
                args: args
                    .into_iter()
                    .map(|arg| arg.localize(call_id.func))
                    .collect(),
                clobbers: vec![],
            }),
        );
    }

    /// Push `base - sub` (8-byte) at the start of `block` and return its value.
    fn push_sp_minus(tc: &mut TestContext, block: BlockId, base: ValueId, sub: u64) -> ValueId {
        let mut b = (&mut tc.ctx).builder(block);
        b.set_insert_point_to_start();
        let k = b.shr().get_const(sub, 8);
        let v = b.push_sub(base, k).id();
        v
    }

    /// Build `f` (reads its caller-frame slot `@sp+4`, derefs pointer param `@p`)
    /// and a caller `g` with an `@SP` param and a `call <f>`. Returns
    /// `(f, g_entry, g_call, gsp)`.
    fn build_f_and_g(
        tc: &mut TestContext,
        sp_reg: VarnodeId,
    ) -> (FunctionId, BlockId, BlockId, ValueId) {
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @sp:i64 @p:i64>
                    %off = @sp + i64 0x4;
                    %x = load(ram:4, %off);
                    store(ram:4, @p <- %x);
                    return at i64 0;
            fn g:
                <g_entry @gsp:i64>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = g_cont;
        make_sp_param(tc, f_entry, 0, sp_reg); // @sp
        make_sp_param(tc, g_entry, 0, sp_reg); // @gsp
        let gsp = BasicBlock::from_id(&tc.ctx, g_entry)
            .params()
            .next()
            .unwrap()
            .id();
        let _ = g;
        (f, g_entry, g_call, gsp)
    }

    #[test]
    fn verifies_true_when_pointer_arg_is_non_stack() {
        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let (f, g_entry, g_call, gsp) = build_f_and_g(&mut tc, sp_reg);
        // @ESP arg = a real stack frame base; pointer arg = a global (non-stack).
        let base = push_sp_minus(&mut tc, g_entry, gsp, 0x14);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        set_call(&mut tc, g_call, f, vec![base, ptr]);

        // Both `f` and the caller `g` carry an `@SP` param, so both are eligible
        // (the assumption is recorded for every `@SP`-param function regardless of
        // whether a pointer param exists yet — see `eligible`).
        assert_eq!(
            assume_args_disjoint_caller_frame(&mut tc.ctx, Some(sp_reg)),
            2
        );
        assert_eq!(
            assume_args_disjoint_caller_frame(&mut tc.ctx, Some(sp_reg)),
            0,
            "repeating accepted assumptions must not report a change"
        );
        verify_args_disjoint_caller_frame(&mut tc.ctx, Some(sp_reg));
        assert_eq!(
            tc.ctx.known(Proposition::ArgsDisjointFromCallerFrame(f)),
            Some(true),
            "a global pointer arg cannot alias the caller's stack frame"
        );
    }

    #[test]
    fn verifies_true_when_stack_pointer_is_disjoint() {
        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let (f, g_entry, g_call, gsp) = build_f_and_g(&mut tc, sp_reg);
        // Callee frame base @ -0x14 (footprint [-0x14, -0xc)); buffer well below it.
        let base = push_sp_minus(&mut tc, g_entry, gsp, 0x14);
        let buf = push_sp_minus(&mut tc, g_entry, gsp, 0x40);
        set_call(&mut tc, g_call, f, vec![base, buf]);

        assume_args_disjoint_caller_frame(&mut tc.ctx, Some(sp_reg));
        verify_args_disjoint_caller_frame(&mut tc.ctx, Some(sp_reg));
        assert_eq!(
            tc.ctx.known(Proposition::ArgsDisjointFromCallerFrame(f)),
            Some(true),
            "a stack buffer disjoint from the arg-slot region is fine"
        );
    }

    #[test]
    fn verifies_false_when_stack_pointer_overlaps_frame() {
        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let (f, g_entry, g_call, gsp) = build_f_and_g(&mut tc, sp_reg);
        // Pointer arg lands exactly on the callee frame base → overlaps the slots.
        let base = push_sp_minus(&mut tc, g_entry, gsp, 0x14);
        let buf = push_sp_minus(&mut tc, g_entry, gsp, 0x14);
        set_call(&mut tc, g_call, f, vec![base, buf]);

        assume_args_disjoint_caller_frame(&mut tc.ctx, Some(sp_reg));
        verify_args_disjoint_caller_frame(&mut tc.ctx, Some(sp_reg));
        assert_eq!(
            tc.ctx.known(Proposition::ArgsDisjointFromCallerFrame(f)),
            Some(false),
            "a pointer into the callee's arg-slot region disproves the assumption"
        );
    }
}
