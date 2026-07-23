//! Architecture-independent call summaries: unbounded-stack-read facts.
//!
//! Everything here is derived from the IR itself — there are no calling-
//! convention tables. [`set_function_summaries`] flags each function that reads a
//! passed pointer (or its own frame) unboundedly on its
//! [`FunctionSignature`]'s `reads_unbounded_stack`, so a caller that hands such a
//! callee a pointer into its own frame keeps that frame in memory. The register
//! effect/interface channel lives in `FunctionEffects` on the function interface,
//! populated by the `argpromote` register passes.

use rustc_hash::FxHashSet as HashSet;

use qcode::{
    context::Context,
    value::{
        BasicBlock, BlockId, BlockParam, FunctionBody, FunctionId, LocalValueId, ValueId, Varnode,
        VarnodeId,
        insn::{Binop, IntBinop, Mnemonic},
    },
};

use crate::{
    gvn::affine::{Numbering, precompute_forms},
    mem::mem2reg::has_dynamic_stack_pointer_deref,
    stack::frame::{frame_offset, incoming_sp_param},
};

/// True when `function_id` makes a call that could read a pointer argument
/// unboundedly: an indirect call (unknown target), or a direct call to an
/// external function or one already flagged [`FunctionBody::reads_unbounded_stack`].
/// Such a function may forward a caller-supplied pointer into that read, so it is
/// itself treated as an unbounded reader.
fn function_makes_unbounded_call(ctx: &Context, function_id: FunctionId) -> bool {
    for block in FunctionBody::from_id(ctx, function_id).blocks() {
        for insn in block.iter() {
            match insn.mnemonic() {
                Mnemonic::CallInd(_) => return true,
                Mnemonic::Call(call) => {
                    let Some(target_id) = call.target.real() else {
                        return true;
                    };
                    let target = FunctionBody::from_id(ctx, target_id);
                    if target.is_external() || target.reads_unbounded_stack() {
                        return true;
                    }
                }
                _ => {}
            }
        }
    }
    false
}

/// True when `function_id` *writes* through a stack-passed pointer parameter at
/// an unbounded offset — e.g. walking the pointer across a loop. Such a store
/// mutates memory the *caller* owns at an offset this function does not bound, so
/// a caller that passes a pointer into its own frame must keep that frame in
/// memory. A bounded write (`*p` or `*(p + const)`) or a mere *read* through such
/// a pointer is not enough — only an unbounded write counts (see the unit tests).
fn function_writes_through_stack_arg(
    ctx: &Context,
    function_id: FunctionId,
    stack_ptr: VarnodeId,
) -> bool {
    let ram = ctx.shared.default_space;
    let frame = FrameCtx::new(ctx, function_id, stack_ptr);
    for block in FunctionBody::from_id(ctx, function_id).blocks() {
        for insn in block.iter() {
            if let Mnemonic::Store(s) = insn.mnemonic()
                && s.space == ram
            {
                let mut visited = HashSet::default();
                let (derives, dynamic) =
                    trace_stack_arg_pointer(ctx, &frame, s.ptr.qualify(insn.id.func), &mut visited);
                if derives && dynamic {
                    return true;
                }
            }
        }
    }
    false
}

/// The frame-offset context for tracing stack-passed pointers: the affine
/// numbering plus the entry stack-pointer `base` (the `@SP` param or the bare
/// stack-pointer varnode) and its width, so a caller-frame slot is recognised in
/// either the legacy `@stack_base + N` or the `@SP + N` representation.
struct FrameCtx {
    numbering: Numbering,
    base: ValueId,
    ptr_width: usize,
}

impl FrameCtx {
    fn new(ctx: &Context, function_id: FunctionId, stack_ptr: VarnodeId) -> Self {
        let base = incoming_sp_param(qcode::value::ModuleView::new(ctx), function_id, stack_ptr)
            .unwrap_or(ValueId::Varnode(stack_ptr));
        Self {
            numbering: precompute_forms(qcode::value::ModuleView::new(ctx), function_id),
            base,
            ptr_width: Varnode::from_id(ctx, stack_ptr).size(),
        }
    }

    /// The frame offset of `v`, in either stack-address representation.
    fn offset(&self, ctx: &Context, v: ValueId) -> Option<i64> {
        frame_offset(
            qcode::value::ModuleView::new(ctx),
            &self.numbering,
            self.base,
            v,
        )
    }
}

/// Trace `v` back toward a stack-passed pointer parameter. Returns
/// `(derives, dynamic)`: whether `v` derives from such a parameter, and whether
/// the path to it carries a non-constant (loop- or index-driven) offset.
fn trace_stack_arg_pointer(
    ctx: &Context,
    frame: &FrameCtx,
    v: ValueId,
    visited: &mut HashSet<ValueId>,
) -> (bool, bool) {
    if !visited.insert(v) {
        return (false, false);
    }
    match v {
        ValueId::BlockParam(pid) => {
            let param = BlockParam::from_id(ctx, pid);
            // A stack-input parameter: promoted from a caller-frame slot at or
            // above the return-address slot. This is the static pointer base.
            if let Some(origin) = param.origin()
                && let Some(offset) = frame.offset(ctx, origin)
                && offset >= frame.ptr_width as i64
            {
                return (true, false);
            }
            // A merge/loop-carried parameter: follow its incoming values. Reaching
            // a stack arg through such a param means the pointer varies (the
            // induction pointer of a loop), which is the unbounded case.
            let Some(block) = param.parent().map(|b| b.id) else {
                return (false, false);
            };
            let index = param.index();
            let derives = incoming_values(ctx, block, index)
                .into_iter()
                .any(|incoming| trace_stack_arg_pointer(ctx, frame, incoming, visited).0);
            (derives, derives)
        }
        ValueId::Instruction(iid) => match ctx.get_insn(iid).mnemonic().clone() {
            Mnemonic::Binop(b) if matches!(b.op, Binop::Int(IntBinop::Add | IntBinop::Sub)) => {
                let (ld, ldyn) =
                    trace_stack_arg_pointer(ctx, frame, b.lhs.qualify(iid.func), visited);
                let (rd, rdyn) =
                    trace_stack_arg_pointer(ctx, frame, b.rhs.qualify(iid.func), visited);
                let derives = ld || rd;
                // The operand that does not derive from the pointer is the offset;
                // a non-constant offset makes the access unbounded.
                let mut dynamic = ldyn || rdyn;
                if ld && !matches!(b.rhs, LocalValueId::Literal(_)) {
                    dynamic = true;
                }
                if rd && !matches!(b.lhs, LocalValueId::Literal(_)) {
                    dynamic = true;
                }
                (derives, derives && dynamic)
            }
            _ => (false, false),
        },
        _ => (false, false),
    }
}

/// The values flowing into parameter `index` of `block` from its predecessors'
/// branch arguments.
fn incoming_values(ctx: &Context, block: BlockId, index: usize) -> Vec<ValueId> {
    let preds: Vec<BlockId> = BasicBlock::from_id(ctx, block)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    let mut out = Vec::new();
    for pred in preds {
        let Some(term) = BasicBlock::from_id(ctx, pred).iter().last() else {
            continue;
        };
        let q = |t| BlockId::new(pred.func, t);
        match term.mnemonic() {
            Mnemonic::Branch(br) if q(br.target) == block => {
                if let Some(&a) = br.args.get(index) {
                    out.push(a.qualify(pred.func));
                }
            }
            Mnemonic::CBranch(cb) => {
                if q(cb.success_block) == block
                    && let Some(&a) = cb.success_args.get(index)
                {
                    out.push(a.qualify(pred.func));
                }
                if q(cb.failure_block) == block
                    && let Some(&a) = cb.failure_args.get(index)
                {
                    out.push(a.qualify(pred.func));
                }
            }
            _ => {}
        }
    }
    out
}

/// Flags `function_id` on its signature when it reads a passed pointer (or its
/// own frame) unboundedly: it has a dynamic stack access, or forwards a pointer
/// into an unbounded/indirect/external call, or writes through a stack-passed
/// pointer at an unbounded offset. A caller that hands such a callee a pointer
/// into its own frame cannot bound which of its stack slots the callee touches,
/// so it must keep its whole frame in memory (see `mem2reg`'s stack-escape
/// handling). The register effect/interface channel is owned by the `argpromote`
/// register passes via `FunctionEffects`, not recomputed here.
pub fn set_function_summaries(ctx: &mut Context, function_id: FunctionId, stack_ptr: VarnodeId) {
    // A functionalized (`pure_reg`) function's interface is owned by
    // `argpromote_registers`; its unbounded-read fact is likewise established
    // there. Leave a materialized function's signature untouched.
    if FunctionBody::from_id(ctx, function_id).is_reg_materialized() {
        return;
    }

    // Union with the seeded value so the fact only grows across checkpoint+replay
    // rounds.
    let reads_unbounded = FunctionBody::from_id(ctx, function_id).reads_unbounded_stack()
        || has_dynamic_stack_pointer_deref(ctx, function_id, stack_ptr)
        || function_makes_unbounded_call(ctx, function_id)
        || function_writes_through_stack_arg(ctx, function_id, stack_ptr);

    FunctionBody::from_id_mut(ctx, function_id).set_reads_unbounded_stack(reads_unbounded);
}

/// Runs [`set_function_summaries`] over every non-external function.
pub fn set_all_function_summaries(ctx: &mut Context, stack_ptr: VarnodeId) {
    let ids: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| !f.is_external())
        .map(|f| f.id)
        .collect();
    for id in ids {
        set_function_summaries(ctx, id, stack_ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AliasResult;
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{BasicBlock, FunctionBody, FunctionId, Value},
    };

    /// Build a function rooted at `addr` with an incoming `@SP` param (origin = the
    /// stack-pointer varnode `sp`), as `argpromote_registers` would mint it. The
    /// closure receives the builder and the `@SP` param `ValueId`, so a body can
    /// address its frame as `@SP ± N`.
    fn build_fn_with_sp(
        tc: &mut TestContext,
        name: &'static str,
        addr: u64,
        sp: VarnodeId,
        f: impl FnOnce(&mut Builder<'static, '_>, ValueId),
    ) -> FunctionId {
        let fun_id = FunctionBody::make(&mut tc.ctx, name.into()).unwrap().id;
        // Self-stored root block (see `build_fn`), so mem2reg's checked-out path
        // can run on it.
        let block_id = tc.ctx.get_or_make_block(addr, fun_id);
        FunctionBody::from_id_mut(&mut tc.ctx, fun_id)
            .set_root(block_id)
            .unwrap();
        let pid = BasicBlock::from_id_mut(&mut tc.ctx, block_id)
            .push_param(8)
            .id;
        tc.ctx
            .block_param_mut(pid)
            .set_origin_id(ValueId::Varnode(sp).localize(pid.func));
        let sp_param = ValueId::BlockParam(pid);
        let mut builder = tc.ctx.builder_at(addr);
        f(&mut builder, sp_param);
        drop(builder);
        fun_id
    }

    // -----------------------------------------------------------------------
    // Stack-passed parameters + frame-escape safety
    // -----------------------------------------------------------------------

    #[test]
    fn dynamic_stack_frame_pointer_read_is_unbounded_stack_reader() {
        let mut tc = TestContext::new();
        let sp = tc.r3;
        let ram = tc.ctx.shared.default_space;
        let reg = tc.reg_space;
        let r0 = tc.r0;

        let callee = build_fn_with_sp(&mut tc, "callee", 0x2000, sp, |b, sp_param| {
            let c8 = b.shr().get_const(8, 8);
            let local = b.push_sub(sp_param, c8).id(); // @SP - 8
            let idx = b.push_load::<false>(ValueId::Varnode(r0), 8, reg).id();
            let ptr = b.push_add(local, idx).id(); // (@SP - 8) + idx → dynamic
            let loaded = b.push_load::<false>(ptr, 1, ram).id();
            let ret = b.push_zext(loaded, 8).id();
            b.push_return(ret);
        });
        let aliases = AliasResult::simple_for_function(&tc.ctx, callee);
        crate::mem2reg(&mut tc.ctx, callee, &aliases);
        set_function_summaries(&mut tc.ctx, callee, sp);

        assert!(
            FunctionBody::from_id(&tc.ctx, callee).reads_unbounded_stack(),
            "a computed pointer into this function's own frame remains unbounded"
        );
    }

    #[test]
    fn frame_escape_flag_disables_stack_promotion() {
        use crate::mem::mem2reg::mem2reg_framed;
        use crate::stack::canonicalize::canonicalize_sp_slots;
        use crate::stack::frame::incoming_sp_param;

        // A normally-promotable local (stored then loaded) must stay in memory once
        // the function is flagged as letting a frame pointer escape unboundedly.
        let mut tc = TestContext::new();
        let ram = tc.ctx.shared.default_space;
        let sp = tc.r3;

        // The only loads in the body are the stack-slot reload, so a plain RAM-load
        // count tracks whether the local was promoted away.
        let ram_load_count = |ctx: &Context, fun: FunctionId| {
            FunctionBody::from_id(ctx, fun)
                .blocks()
                .flat_map(|b| b.iter().collect::<Vec<_>>())
                .filter(|i| matches!(i.mnemonic(), Mnemonic::Load(_)))
                .count()
        };

        let build_local_fn = |tc: &mut TestContext, name: &'static str, addr: u64| {
            build_fn_with_sp(tc, name, addr, sp, |b, sp_param| {
                let c8 = b.shr().get_const(8, 8);
                let c16 = b.shr().get_const(16, 8);
                let slot = b.push_sub(sp_param, c8).id(); // @SP - 8
                let v = b.shr().get_const(7u64, 8);
                b.push_store(v, slot, ram);
                let slot_reload = b.push_sub(sp_param, c8).id(); // reload @SP - 8
                let loaded = b.push_load::<false>(slot_reload, 8, ram).id();
                let sink_slot = b.push_sub(sp_param, c16).id(); // @SP - 16
                b.push_store(loaded, sink_slot, ram);
                let sink = b.shr().get_const(0u64, 8);
                b.push_return(sink);
            })
        };

        let promote = |tc: &mut TestContext, fun: FunctionId| {
            let sp_param =
                incoming_sp_param(qcode::value::ModuleView::new(&tc.ctx), fun, sp).unwrap();
            canonicalize_sp_slots(&mut tc.ctx, fun, sp);
            let aliases = AliasResult::simple_for_function(&tc.ctx, fun);
            mem2reg_framed(&mut tc.ctx, fun, &aliases, Some(sp_param));
        };

        // Baseline: the local is promoted away (no stack loads remain).
        let promoted = build_local_fn(&mut tc, "promoted", 0x1000);
        promote(&mut tc, promoted);
        assert_eq!(
            ram_load_count(&tc.ctx, promoted),
            0,
            "without the flag, the local stack slot is promoted to SSA"
        );

        // Flagged: promotion is disabled and the load survives.
        let escaping = build_local_fn(&mut tc, "escaping", 0x2000);
        FunctionBody::from_id_mut(&mut tc.ctx, escaping).set_frame_escapes_to_unbounded(true);
        promote(&mut tc, escaping);
        assert!(
            ram_load_count(&tc.ctx, escaping) > 0,
            "a frame-escaping function must keep its stack frame in memory"
        );
    }
}

// ----- passes ----------------------------------------------------------------

use crate::{Pass, PipelineEnv};

#[derive(Default)]
pub struct Summaries;

impl Pass for Summaries {
    const NAME: &'static str = "summaries";
    fn description(&self) -> &'static str {
        "Flag each function that reads a passed pointer or its own frame unboundedly"
    }
    fn run(
        &self,
        cone: &mut crate::ConeMut,
        env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        // CONE-HATCH: remove when summaries migrates.
        let targets = cone.cone_functions();
        let ctx = cone.bypass_cone_unmigrated_hatch();
        let Some(stack_ptr) = env.sp_varnode else {
            return Ok(crate::ModulePassOutcome::default());
        };
        let affected: Vec<FunctionId> = targets
            .iter()
            .copied()
            .filter(|&id| !FunctionBody::from_id(ctx, id).is_external())
            .collect();
        for id in affected.iter().copied() {
            set_function_summaries(ctx, id, stack_ptr);
        }
        Ok(crate::ModulePassOutcome::functions(affected)
            .preserving_global::<crate::CallGraphAnalysis>()
            .preserving_global::<crate::AddressAnalysis>())
    }
}

crate::register_module_pass!(Summaries);
