use qcode::{
    context::Context,
    space::{SpaceId, SpaceType},
    types::TypeId,
    value::{
        BasicBlock, FunctionBody, FunctionEffects, FunctionId, Instruction, LocalValueId, QCodeMut,
        RegisterInterfaceMap, Value, ValueId, Varnode, VarnodeId,
        insn::{InstructionId, Mnemonic},
    },
};

use rustc_hash::FxHashSet;

use crate::{AnalysisManager, CallGraph, CallGraphAnalysis, Pass, PipelineEnv};

use super::{
    add_input, append_outputs, called_function_set_from_graph,
    reg_summary::{RegChannel, finalize_register_effects},
    summary::{EffectSummaries, TopCause, solve_summaries},
};

pub(crate) fn is_register(ctx: &Context, vn: VarnodeId) -> bool {
    matches!(Varnode::from_id(ctx, vn).space().ty, SpaceType::Register)
}

// ===========================================================================
// Register channel (runs early, before mem2reg — see ARGPROMOTE_REGISTERS.md)
// ===========================================================================
//
// Unlike the RAM channel above, register effects are functionalized by a purely
// syntactic scan of the lifted body: a register *loaded* is an input, a register
// *stored* is an output. No shadow space is needed — every register access
// converts, so the body keeps operating on real register space and the *next*
// mem2reg run SSA-promotes it into a pure value function. Inputs become by-value
// params seeded into register space at entry; outputs are returned as a flat
// positional aggregate (slot i ↔ output register i) the caller replays.

/// A function's register interface, recovered by [`scan_register_effects`].
#[derive(Debug)]
pub(crate) struct RegisterEffects {
    /// Registers the body loads — each becomes a by-value input parameter
    /// (over-approximated: a written-first register yields a dead param that
    /// mem2reg/DCE prune). Sorted by `(address, size)` for a deterministic
    /// param/argument order shared with the caller rewrite.
    pub(crate) inputs: Vec<VarnodeId>,
    /// Registers the body stores, canonicalized to the coarsest register per
    /// overlap group so the caller's replay is order-independent. Sorted.
    pub(crate) outputs: Vec<VarnodeId>,
}

/// The byte interval `(space, start, end)` a register varnode occupies.
fn reg_interval(ctx: &Context, vn: VarnodeId) -> (SpaceId, i64, i64) {
    let v = Varnode::from_id(ctx, vn);
    let start = v.address();
    (v.space().id, start, start + v.size() as i64)
}

/// `true` if two register varnodes occupy overlapping bytes of the same space.
fn regs_overlap(ctx: &Context, a: VarnodeId, b: VarnodeId) -> bool {
    let (sa, a0, a1) = reg_interval(ctx, a);
    let (sb, b0, b1) = reg_interval(ctx, b);
    sa == sb && a0 < b1 && b0 < a1
}

/// Collapse a set of register varnodes into the coarsest register per overlap
/// group. Register files nest (AL ⊂ AX ⊂ EAX ⊂ RAX), so each overlap group has a
/// unique member whose interval contains the rest; for *outputs*, loading that
/// register at the return reads the merged final state of every sub-write, and
/// for *inputs* one seed of it covers every overlapping read. Returns `None` if
/// some group has no single covering register (partial overlap with no cover) —
/// that function is left on the conservative path.
pub(crate) fn canonicalize_to_coarsest(
    ctx: &Context,
    regs: &[VarnodeId],
) -> Option<Vec<VarnodeId>> {
    // Connected components under `regs_overlap` (tiny N, so O(N²) is fine).
    let mut group_of: Vec<usize> = (0..regs.len()).collect();
    for i in 0..regs.len() {
        for j in (i + 1)..regs.len() {
            if regs_overlap(ctx, regs[i], regs[j]) {
                let (gi, gj) = (group_of[i], group_of[j]);
                if gi != gj {
                    for g in &mut group_of {
                        if *g == gj {
                            *g = gi;
                        }
                    }
                }
            }
        }
    }

    let mut coarse: Vec<VarnodeId> = Vec::new();
    for g in 0..regs.len() {
        let members: Vec<VarnodeId> = (0..regs.len())
            .filter(|&i| group_of[i] == g)
            .map(|i| regs[i])
            .collect();
        if members.is_empty() {
            continue; // not a group representative
        }
        // The cover must contain every member's interval.
        let cover = members.iter().copied().find(|&m| {
            let (_, m0, m1) = reg_interval(ctx, m);
            members.iter().all(|&o| {
                let (_, o0, o1) = reg_interval(ctx, o);
                m0 <= o0 && m1 >= o1
            })
        })?;
        if !coarse.contains(&cover) {
            coarse.push(cover);
        }
    }
    Some(coarse)
}

/// Why a function is not register-pure (cannot be functionalized by
/// [`argpromote_registers`]). Mirrors the gating in [`try_promote_registers`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegPurityReason {
    /// External function — no body to functionalize.
    External,
    /// Lifted but has no root block (empty body).
    NoBody,
    /// Address-taken: reachable by an indirect call this pass can't rewrite.
    AddressTaken,
    /// Writes no registers, so there is nothing to functionalize.
    NoRegisterWrites,
    /// A register-write overlap group has no single covering register.
    NonCanonicalRegisters,
    /// No direct callers to thread by-value inputs / replayed outputs through.
    NoCallers,
    /// Contains an unresolved indirect call whose register effects are unknown.
    IndirectCall,
    /// A direct callee (recorded) is itself register-impure — its effects
    /// through the call cannot be represented in this function's interface.
    ImpureCallee(FunctionId),
    /// Reached through a non-`Call` site (tail call / apply) this pass cannot
    /// rewrite to the new interface.
    NonCallSite,
}

impl RegPurityReason {
    /// One-line human-readable explanation, for display in the GUI / headless dump.
    pub fn describe(self) -> &'static str {
        match self {
            RegPurityReason::External => "external function (no body to functionalize)",
            RegPurityReason::NoBody => "no function body",
            RegPurityReason::AddressTaken => {
                "address-taken (reachable by indirect calls this pass can't rewrite)"
            }
            RegPurityReason::NoRegisterWrites => "writes no registers (nothing to functionalize)",
            RegPurityReason::NonCanonicalRegisters => {
                "register writes don't canonicalize to a coarsest register"
            }
            RegPurityReason::NoCallers => "no direct callers to thread inputs/outputs through",
            RegPurityReason::IndirectCall => {
                "contains an indirect call with unknown register effects"
            }
            RegPurityReason::ImpureCallee(_) => "calls a register-impure function",
            RegPurityReason::NonCallSite => {
                "reached through a tail-call/apply site this pass can't rewrite"
            }
        }
    }
}

/// Report whether `fid` is eligible to be functionalized into a register-pure
/// function (`Ok`) or, if not, the gating reason (`Err`). A function whose
/// [`FunctionRef::is_pure_reg`](qcode::value::FunctionRef::is_pure_reg) is
/// already set is necessarily `Ok`; this is the
/// source of the "why not" shown for the rest.
///
/// Solves the whole-program effect-summary fixpoint on every call. A caller
/// querying many functions in a loop (e.g. the GUI loader) should solve it once
/// with [`RegPurityGates`] and use [`RegPurityGates::purity`].
pub fn reg_purity(ctx: &Context, fid: FunctionId) -> Result<(), RegPurityReason> {
    RegPurityGates::compute(ctx).purity(ctx, fid)
}

/// The whole-program facts a [`reg_purity`] query consults — the solved effect
/// summaries (see `summary.rs`), the direct-call-target set, and the call-graph
/// snapshot the site-shape gate needs. Building them once amortizes the
/// O(instructions) work across a per-function classification loop (the GUI
/// loader). All are invariant under register promotion (it threads data values
/// and rewrites interfaces but adds no `ValueId::Function` operand and no
/// `Call.target` edge), so a single instance is valid for the whole loop.
pub struct RegPurityGates {
    graph: CallGraph,
    summaries: EffectSummaries<RegChannel>,
    called: FxHashSet<FunctionId>,
    /// Functions whose address is taken: they keep precise summaries but the v1
    /// promoter must not rewrite them (indirect sites are unrewritable).
    address_taken: FxHashSet<FunctionId>,
}

/// Best-effort stack-pointer varnode for diagnostic queries that have no
/// [`PipelineEnv`] (the GUI loader): the conventional SP register name per
/// supported arch. The pass itself always uses `env.sp_varnode`.
fn guess_sp(ctx: &Context) -> Option<VarnodeId> {
    ["RSP", "ESP", "SP"].iter().find_map(|sp_name| {
        ctx.shared.registers.values().copied().find(|&vn| {
            Varnode::from_id(ctx, vn)
                .name()
                .is_some_and(|n| n.eq_ignore_ascii_case(sp_name))
        })
    })
}

impl RegPurityGates {
    /// Solve the summaries and gating sets for `ctx` once.
    pub fn compute(ctx: &Context) -> Self {
        let graph = CallGraph::analyze(ctx);
        let chan = RegChannel { sp: guess_sp(ctx) };
        let summaries = solve_summaries(ctx, &graph, &chan);
        let called = called_function_set_from_graph(ctx, &graph);
        let address_taken = super::address_taken_set(ctx);
        Self {
            graph,
            summaries,
            called,
            address_taken,
        }
    }

    /// Classify `fid` against the solved summaries — see [`reg_purity`].
    pub fn purity(&self, ctx: &Context, fid: FunctionId) -> Result<(), RegPurityReason> {
        let f = FunctionBody::from_id(ctx, fid);
        // Externals carry leaf summaries for composition, but are never
        // themselves functionalized (no body to rewrite).
        if f.is_external() {
            return Err(RegPurityReason::External);
        }
        // Address-taken functions carry precise summaries (so their effects
        // compose into callers), but the v1 promoter cannot rewrite the indirect
        // sites that reach them — leave them on the conservative path until the
        // dual-binding convention (pass 2/3) makes zero-arg indirect calls valid.
        if self.address_taken.contains(&fid) {
            return Err(RegPurityReason::AddressTaken);
        }
        let eff = match self.summaries.get(fid) {
            Ok(eff) => eff,
            Err(cause) => return Err(purity_reason_of_top(*cause)),
        };
        finalize_register_effects(ctx, eff)?;
        if !self.called.contains(&fid) {
            return Err(RegPurityReason::NoCallers);
        }
        if has_non_call_site(ctx, &self.graph, fid) {
            return Err(RegPurityReason::NonCallSite);
        }
        Ok(())
    }
}

/// Map an engine ⊤ cause onto the public gating reason.
fn purity_reason_of_top(cause: TopCause) -> RegPurityReason {
    match cause {
        TopCause::External | TopCause::Channel => RegPurityReason::External,
        TopCause::NoBody => RegPurityReason::NoBody,
        TopCause::IndirectCall => RegPurityReason::IndirectCall,
        TopCause::ImpureCallee(fid) => RegPurityReason::ImpureCallee(fid),
    }
}

/// `true` if `fid` is reachable through an incoming edge that is not a real
/// [`Mnemonic::Call`] targeting it — a tail call, `Apply`/`Map`/`Scan` site, or
/// a synthetic discovery edge. Such sites cannot be rewritten to the new
/// positional interface, so promoting the function would desync them.
fn has_non_call_site(ctx: &Context, graph: &CallGraph, fid: FunctionId) -> bool {
    graph.incoming_edges(fid).iter().any(|&eid| {
        let edge = graph.edge(eid);
        match edge.site {
            Some(site) => !matches!(
                ctx.get_insn(site).mnemonic(),
                Mnemonic::Call(call) if call.target.real() == Some(fid)
            ),
            None => true,
        }
    })
}

/// Scan `fid` for register reads (inputs) and writes (outputs). Returns `Err`
/// when there is no register write (nothing to functionalize) or an output
/// overlap group has no single covering register.
///
/// Superseded in the pass itself by the solved summaries (`reg_summary.rs` +
/// `finalize_register_effects`, which compose callee effects); kept as the
/// body-local scan the single-function unit tests drive `rewrite_registers`
/// with.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn scan_register_effects(
    ctx: &Context,
    fid: FunctionId,
) -> Result<RegisterEffects, RegPurityReason> {
    let mut loaded: Vec<VarnodeId> = Vec::new();
    let mut stored: Vec<VarnodeId> = Vec::new();
    for block in FunctionBody::from_id(ctx, fid).blocks() {
        for insn in block.iter() {
            match insn.mnemonic() {
                Mnemonic::Load(l) => {
                    if let qcode::value::LocalValueId::Varnode(vn) = l.ptr
                        && is_register(ctx, vn)
                        && !loaded.contains(&vn)
                    {
                        loaded.push(vn);
                    }
                }
                Mnemonic::Store(s) => {
                    if let qcode::value::LocalValueId::Varnode(vn) = s.ptr
                        && is_register(ctx, vn)
                        && !stored.contains(&vn)
                    {
                        stored.push(vn);
                    }
                }
                _ => {}
            }
        }
    }
    if stored.is_empty() {
        return Err(RegPurityReason::NoRegisterWrites);
    }
    let mut outputs =
        canonicalize_to_coarsest(ctx, &stored).ok_or(RegPurityReason::NonCanonicalRegisters)?;

    // The rewritten body reads not only the originally-loaded registers but also
    // every output (the return write-set loads each one). An output written on
    // only some paths is therefore read-before-write at a return on a no-write
    // path; seeding it from an input param makes that read the caller's incoming
    // value (replayed back as a no-op), keeping callee params and caller args in
    // sync. Always-written outputs just yield a dead seed that DCE prunes.
    let mut read_set = loaded;
    for &o in &outputs {
        if !read_set.contains(&o) {
            read_set.push(o);
        }
    }
    let mut inputs =
        canonicalize_to_coarsest(ctx, &read_set).ok_or(RegPurityReason::NonCanonicalRegisters)?;

    let key = |ctx: &Context, vn: &VarnodeId| {
        let v = Varnode::from_id(ctx, *vn);
        (v.address(), v.size())
    };
    inputs.sort_by_key(|vn| key(ctx, vn));
    outputs.sort_by_key(|vn| key(ctx, vn));
    Ok(RegisterEffects { inputs, outputs })
}

/// Pass 2 (materialize) — `ARGPROMOTE_REGISTERS_V2.md`: solve the register
/// effect summaries over the *unmutated* module, then materialize every eligible
/// function's interface (by-value params + return pack), recording the
/// param/pack ↔ register mapping in [`FunctionEffects`]. **No call site is
/// touched** — every caller keeps its valid zero-arg implicit-binding call.
/// Returns the set of materialized functions.
pub(crate) fn materialize_functions(
    ctx: &mut Context,
    graph: &CallGraph,
    targets: &[FunctionId],
    sp: Option<VarnodeId>,
) -> FxHashSet<FunctionId> {
    let chan = RegChannel { sp };
    let summaries = solve_summaries(ctx, graph, &chan);
    let mut changed = FxHashSet::default();
    for fid in targets.iter().copied() {
        {
            let f = FunctionBody::from_id(ctx, fid);
            // Externals have no body to materialize; an already-materialized
            // function's interface is final (its solved callees stay solved).
            if f.is_external()
                || f.root().is_none()
                || matches!(f.effects(), FunctionEffects::Materialized(_))
            {
                continue;
            }
        }
        let eff = match summaries.get(fid) {
            Ok(eff) => eff,
            Err(_) => {
                // ⊤: record it so alias analysis / the RAM gate see it, but do
                // not materialize.
                FunctionBody::from_id_mut(ctx, fid).set_effects(FunctionEffects::Top);
                continue;
            }
        };
        let Ok(reg_eff) = finalize_register_effects(ctx, eff) else {
            // Solved, but no register writes / non-canonical overlap: nothing to
            // materialize. Still a *solved* summary (the RAM gate keys on that).
            FunctionBody::from_id_mut(ctx, fid).set_effects(FunctionEffects::Solved);
            continue;
        };
        // The implicit (zero-arg) convention binds params from the register file
        // at call entry — which the emulator only does for real `Call`/`CallInd`
        // sites. A tail-call/apply site is not seeded, so materializing a
        // function reached that way would leave its entry seed-stores reading
        // unbound params. Leave those solved-but-unmaterialized. (Address-taken
        // functions reached by `CallInd` *are* seeded, so they materialize.)
        if has_non_call_site(ctx, graph, fid) {
            FunctionBody::from_id_mut(ctx, fid).set_effects(FunctionEffects::Solved);
            continue;
        }
        materialize_interface(ctx, fid, &reg_eff);
        changed.insert(fid);
    }
    changed
}

/// Materialize `fid`'s register interface from its solved effects `reg_eff`:
/// add one by-value input param per input register (seeded into register space
/// at entry) and append the outputs as a flat positional return pack — but
/// inject *nothing* at any caller (empty call-site slice). Records the
/// param/pack ↔ register mapping in [`FunctionEffects::Materialized`].
pub(crate) fn materialize_interface(ctx: &mut Context, fid: FunctionId, reg_eff: &RegisterEffects) {
    // --- inputs: one by-value param per input register, seeded at entry ---------
    let input_meta: Vec<InputMeta> = reg_eff
        .inputs
        .iter()
        .map(|&r| {
            let v = Varnode::from_id(&*ctx, r);
            (
                r,
                v.size(),
                v.space().id,
                v.name().map(str::to_owned),
                // Inherit a global varnode type override so the by-value entry
                // param carries the register's richer type, not `Int(size)`.
                ctx.stored_type_of(ValueId::Varnode(r)),
            )
        })
        .collect();
    for (r, size, space, name, ty) in input_meta {
        add_input(
            ctx,
            fid,
            size,
            // Name the param after its register so the implicit convention binds
            // it from the register file (`seed_entry_params` keys on the name).
            name,
            Some(ValueId::Varnode(r)),
            ty,
            // Empty call sites: add the param + entry seed, touch no caller.
            Some(&[]),
            space,
            move |_b| ValueId::Varnode(r),
            // Never invoked (no call sites), but the closure type is required.
            move |ctx, call_id, block| {
                let mut b = (ctx).builder(block);
                b.set_insert_point_before(call_id);
                b.push_load::<false>(ValueId::Varnode(r), size, space).id()
            },
        );
    }

    // --- outputs: a flat positional return pack, one field per output register --
    let outputs = output_meta(ctx, &reg_eff.outputs);
    append_outputs(
        ctx,
        fid,
        &outputs,
        Some(&[]),
        false,
        |_i, (_, _, _, name)| vec![name.clone()],
        |b, &(r, size, space, _)| vec![b.push_load::<false>(ValueId::Varnode(r), size, space).id()],
        |b, &(r, _, space, _), ext| {
            b.push_store(ext[0], ValueId::Varnode(r), space);
        },
    );

    // Record the interface mapping: param slot i ↔ inputs[i], pack slot i ↔
    // outputs[i]. `add_input`/`append_outputs` iterate in these same orders.
    FunctionBody::from_id_mut(ctx, fid).set_effects(FunctionEffects::Materialized(
        RegisterInterfaceMap {
            inputs: reg_eff.inputs.clone(),
            outputs: reg_eff.outputs.clone(),
        },
    ));
}

/// Pass 3 (regpure call sites): rewrite one direct, `Opaque` call to a
/// materialized callee into an explicit `regpure` call — materialize each input
/// register as a `load(register, R)` argument before the call, retype the result
/// to the callee's return pack, replay each output as `store(register, R)` in the
/// continuation, and tag the call [`CallTag::RegPure`]. Order-independent: a ⊤
/// caller can host regpure calls (its loads/stores just stay in its body).
pub(crate) fn rewrite_call_regpure(ctx: &mut Context, call_id: InstructionId, callee: FunctionId) {
    let map = match FunctionBody::from_id(ctx, callee).effects() {
        FunctionEffects::Materialized(m) => m.clone(),
        _ => return,
    };
    let Some(call_block) = ctx.get_insn(call_id).parent().map(|b| b.id) else {
        return;
    };
    // The callee's return-pack aggregate type (built by `append_outputs`).
    let Some(ret_ty) = ctx.shared.types.function_return(callee) else {
        return;
    };

    // --- inputs: load each input register into an argument before the call ------
    let input_meta: Vec<(VarnodeId, usize, SpaceId)> = map
        .inputs
        .iter()
        .map(|&r| {
            let v = Varnode::from_id(&*ctx, r);
            (r, v.size(), v.space().id)
        })
        .collect();
    let args: Vec<LocalValueId> = {
        let mut b = (ctx).builder(call_block);
        b.set_insert_point_before(call_id);
        input_meta
            .iter()
            .map(|&(r, size, space)| {
                b.push_load::<false>(ValueId::Varnode(r), size, space)
                    .id()
                    .localize(call_id.func)
            })
            .collect()
    };

    // Rewrite the call: explicit args, cleared clobbers (register-transparent),
    // tagged regpure. Retype the result to the callee's return pack.
    ctx.replace_instruction_mnemonic(
        call_id,
        Mnemonic::Call(qcode::value::insn::Call {
            target: qcode::value::insn::Callee::Real(callee),
            args,
            clobbers: vec![],
            tag: qcode::value::insn::CallTag::RegPure,
        }),
    );
    Instruction::from_id_mut(ctx, call_id).set_type(ret_ty);

    // --- outputs: replay each pack slot as a register store in the continuation -
    let Some(cont) = BasicBlock::from_id(ctx, call_block)
        .successors()
        .next()
        .map(|(_, b)| b)
    else {
        return;
    };
    let output_meta: Vec<(VarnodeId, SpaceId)> = map
        .outputs
        .iter()
        .map(|&r| (r, Varnode::from_id(&*ctx, r).space().id))
        .collect();
    let result = ValueId::Instruction(call_id);
    let mut b = (ctx).builder(cont);
    b.set_insert_point_to_start();
    for (i, &(r, space)) in output_meta.iter().enumerate() {
        let ext = ValueId::Instruction(b.push_extract(result, i).id);
        b.push_store(ext, ValueId::Varnode(r), space);
    }
}

/// Rewrite every direct `Opaque` call to a materialized function into a regpure
/// call (pass 3). Returns the set of *caller* functions changed.
pub(crate) fn regpure_all_sites(
    ctx: &mut Context,
    graph: &CallGraph,
    targets: &[FunctionId],
) -> FxHashSet<FunctionId> {
    let mut changed = FxHashSet::default();
    for callee in targets.iter().copied() {
        if !matches!(
            FunctionBody::from_id(ctx, callee).effects(),
            FunctionEffects::Materialized(_)
        ) {
            continue;
        }
        for site in crate::calls::direct_call_sites(ctx, graph, callee) {
            let opaque = matches!(
                ctx.get_insn(site).mnemonic(),
                Mnemonic::Call(c) if c.tag == qcode::value::insn::CallTag::Opaque
            );
            if opaque {
                rewrite_call_regpure(ctx, site, callee);
                changed.insert(site.func);
            }
        }
    }
    changed
}

/// Compatibility driver for the unit tests: materialize every eligible function
/// (pass 2) then flip its direct call sites to regpure (pass 3). Returns `true`
/// if anything changed.
pub fn argpromote_registers(ctx: &mut Context) -> bool {
    let graph = CallGraph::analyze(ctx);
    let targets = ctx.function_ids();
    let sp = guess_sp(ctx);
    let mut changed = materialize_functions(ctx, &graph, &targets, sp);
    let graph = CallGraph::analyze(ctx);
    changed.extend(regpure_all_sites(ctx, &graph, &targets));
    !changed.is_empty()
}

/// Per-output-register metadata: `(register, size, space, name)`. The name is the
/// register's own (e.g. `eax`) or a positional `output{n}` fallback.
fn output_meta(ctx: &Context, regs: &[VarnodeId]) -> Vec<(VarnodeId, usize, SpaceId, String)> {
    regs.iter()
        .map(|&r| {
            let v = Varnode::from_id(ctx, r);
            let name = v
                .name()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("output{}", usize::from(r) + 1));
            (r, v.size(), v.space().id, name)
        })
        .collect()
}

/// Precomputed `(register, size, space, name, type)` for one input register.
type InputMeta = (VarnodeId, usize, SpaceId, Option<String>, Option<TypeId>);

/// Materialize `fid`'s interface (pass 2) then flip its own direct call sites to
/// regpure (pass 3). The single-function driver the unit tests use to exercise
/// the register rewrite end to end.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn rewrite_registers(ctx: &mut Context, fid: FunctionId, eff: &RegisterEffects) {
    materialize_interface(ctx, fid, eff);
    let graph = CallGraph::analyze(ctx);
    for site in crate::calls::direct_call_sites(ctx, &graph, fid) {
        rewrite_call_regpure(ctx, site, fid);
    }
}

/// Pass 2 of `ARGPROMOTE_REGISTERS_V2.md`: materialize every eligible function's
/// register interface from the solved whole-program summaries, touching no call
/// sites. Scheduled immediately before [`ArgPromoteRegpureCalls`].
#[derive(Default)]
pub struct ArgPromoteMaterialize;

impl Pass for ArgPromoteMaterialize {
    const NAME: &'static str = "argpromote_materialize";
    fn description(&self) -> &'static str {
        "Materialize register interfaces (by-value params + return pack) from solved effects"
    }
    fn run(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
    ) -> Result<crate::ModulePassOutcome, String> {
        let graph = CallGraph::analyze(ctx);
        Ok(crate::ModulePassOutcome::functions(materialize_functions(
            ctx,
            &graph,
            targets,
            env.sp_varnode,
        ))
        .preserving_global::<CallGraphAnalysis>()
        .preserving_global::<crate::AddressAnalysis>())
    }

    fn run_with_analyses(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
        analyses: &mut AnalysisManager,
    ) -> Result<crate::ModulePassOutcome, String> {
        let graph = analyses.global::<CallGraphAnalysis>(ctx);
        Ok(crate::ModulePassOutcome::functions(materialize_functions(
            ctx,
            graph,
            targets,
            env.sp_varnode,
        ))
        .preserving_global::<CallGraphAnalysis>()
        .preserving_global::<crate::AddressAnalysis>())
    }
}

crate::register_module_pass!(ArgPromoteMaterialize);

/// Pass 3 of `ARGPROMOTE_REGISTERS_V2.md`: flip every direct `Opaque` call to a
/// materialized function into an explicit `regpure` call. Order-independent, so
/// it runs right after [`ArgPromoteMaterialize`].
#[derive(Default)]
pub struct ArgPromoteRegpureCalls;

impl Pass for ArgPromoteRegpureCalls {
    const NAME: &'static str = "argpromote_regpure_calls";
    fn description(&self) -> &'static str {
        "Rewrite direct calls to materialized functions into explicit regpure calls"
    }
    fn run(
        &self,
        ctx: &mut Context,
        _env: &PipelineEnv,
        targets: &[FunctionId],
    ) -> Result<crate::ModulePassOutcome, String> {
        let graph = CallGraph::analyze(ctx);
        Ok(
            crate::ModulePassOutcome::functions(regpure_all_sites(ctx, &graph, targets))
                .preserving_global::<CallGraphAnalysis>()
                .preserving_global::<crate::AddressAnalysis>(),
        )
    }

    fn run_with_analyses(
        &self,
        ctx: &mut Context,
        _env: &PipelineEnv,
        targets: &[FunctionId],
        analyses: &mut AnalysisManager,
    ) -> Result<crate::ModulePassOutcome, String> {
        let graph = analyses.global::<CallGraphAnalysis>(ctx);
        Ok(
            crate::ModulePassOutcome::functions(regpure_all_sites(ctx, graph, targets))
                .preserving_global::<CallGraphAnalysis>()
                .preserving_global::<crate::AddressAnalysis>(),
        )
    }
}

crate::register_module_pass!(ArgPromoteRegpureCalls);
