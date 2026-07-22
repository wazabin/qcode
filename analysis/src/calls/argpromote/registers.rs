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

/// The `ptr` operand for a materialized access to register cell `r`: the varnode
/// itself (`load/store(register, R)`). Every materialized interface cell is a
/// register now — globals moved to the RAM effect channel — so this is always a
/// varnode `ptr`.
fn access_ptr(_b: &mut qcode::builder::Builder, r: VarnodeId) -> ValueId {
    ValueId::Varnode(r)
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
/// [`FunctionRef::is_reg_materialized`](qcode::value::FunctionRef::is_reg_materialized) is
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

/// Body-local register interface of a single function: the [`RegChannel`] scan
/// finalized without callee composition. Test-only driver for the
/// single-function `rewrite_registers` unit tests; the pass itself uses the
/// solved summaries.
#[cfg(test)]
pub(crate) fn scan_register_effects(
    ctx: &Context,
    fid: FunctionId,
) -> Result<RegisterEffects, RegPurityReason> {
    use crate::calls::argpromote::summary::EffectChannel;
    let eff = RegChannel { sp: None }
        .scan(ctx, fid)
        .ok_or(RegPurityReason::NoBody)?;
    finalize_register_effects(ctx, &eff)
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
            // materialize. Still a *solved* summary (the RAM gate keys on that);
            // persist the solved sets so call classifiers stay precise.
            FunctionBody::from_id_mut(ctx, fid).set_effects(FunctionEffects::Solved(eff.to_sets()));
            continue;
        };
        // The implicit (zero-arg) convention binds params from the register file
        // at call entry — which the emulator only does for real `Call`/`CallInd`
        // sites. A tail-call/apply site is not seeded, so materializing a
        // function reached that way would leave its entry seed-stores reading
        // unbound params. Leave those solved-but-unmaterialized. (Address-taken
        // functions reached by `CallInd` *are* seeded, so they materialize.)
        if has_non_call_site(ctx, graph, fid) {
            FunctionBody::from_id_mut(ctx, fid).set_effects(FunctionEffects::Solved(eff.to_sets()));
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
                // Inherit a varnode type override so the by-value entry param
                // carries the register's richer type, not `Int(size)`.
                ctx.stored_type_of(ValueId::Varnode(r)),
            )
        })
        .collect();
    for (r, size, space, name, ty) in input_meta {
        // A register cell binds from the register file by name (implicit
        // convention), keyed on the register varnode.
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
            move |b| access_ptr(b, r),
            // Never invoked (no call sites), but the closure type is required.
            move |ctx, call_id, block| {
                let mut b = (ctx).builder(block);
                b.set_insert_point_before(call_id);
                let ptr = access_ptr(&mut b, r);
                b.push_load::<false>(ptr, size, space).id()
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
        |b, &(r, size, space, _)| {
            let ptr = access_ptr(b, r);
            vec![b.push_load::<false>(ptr, size, space).id()]
        },
        |b, &(r, _, space, _), ext, _args| {
            let ptr = access_ptr(b, r);
            b.push_store(ext[0], ptr, space);
        },
    );

    // Record the interface mapping: param slot i ↔ inputs[i], pack slot i ↔
    // outputs[i]. `add_input`/`append_outputs` iterate in these same orders.
    FunctionBody::from_id_mut(ctx, fid).set_effects(FunctionEffects::Materialized(
        RegisterInterfaceMap {
            globals: vec![],
            inputs: reg_eff.inputs.clone(),
            outputs: reg_eff.outputs.clone(),
            // Every pack slot of a bodied function is a real computed value.
            returns: reg_eff.outputs.len(),
        },
    ));
}

/// The block where a regpure call's output pack is replayed: the call block's
/// single fallthrough successor (multiple successors on a call terminator are an
/// invariant violation). The replayed `extract`/`store`s must execute exactly
/// when the call did, so if the continuation has other predecessors (the call
/// falls through into a join) the fallthrough edge is split with a fresh block
/// that branches on to the original continuation, and the replay lands there.
fn replay_block(
    ctx: &mut Context,
    call_block: qcode::value::BlockId,
) -> Option<qcode::value::BlockId> {
    let block = BasicBlock::from_id(ctx, call_block);
    let mut succs = block.successors();
    let (edge, cont) = succs.next()?;
    debug_assert!(
        succs.next().is_none(),
        "call terminator with multiple successor edges"
    );
    drop(succs);
    if BasicBlock::from_id(ctx, cont)
        .predecessors()
        .nth(1)
        .is_none()
    {
        return Some(cont);
    }
    let func = call_block.func;
    let split = BasicBlock::make(ctx, func).id;
    ctx.remove_cfg_edge(func, edge);
    ctx.add_cfg_edge(call_block, split);
    (ctx).builder(split).push_branch(cont);
    Some(split)
}

/// Pass 3 (regpure call sites): rewrite one direct, `Opaque` call to a
/// materialized callee into an explicit `regpure` call — materialize each input
/// register as a `load(register, R)` argument before the call, retype the result
/// to the callee's return pack, replay each output as `store(register, R)` in the
/// continuation, and tag the call [`CallTag::RegPure`]. Order-independent: a ⊤
/// caller can host regpure calls (its loads/stores just stay in its body).
pub(crate) fn rewrite_call_regpure(
    ctx: &mut Context,
    call_id: InstructionId,
    callee: FunctionId,
    sp: Option<VarnodeId>,
) {
    let map = match FunctionBody::from_id(ctx, callee).effects() {
        FunctionEffects::Materialized(m) => m.clone(),
        _ => return,
    };
    // An external (bodyless) materialized callee has no return-pack type built by
    // `append_outputs` (pass 2 skips externals) — pass 3 owns its whole rewrite,
    // including the poison clobber pack and the SP-relative stack-arg loads that
    // `bind_external_args` used to emit (design ruling 7a).
    if FunctionBody::from_id(ctx, callee).is_external() {
        rewrite_external_call_regpure(ctx, call_id, callee, &map, sp);
        return;
    }
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
    // Each input cell becomes a positional argument: a register loaded from the
    // register file.
    let args: Vec<LocalValueId> = {
        let mut b = (ctx).builder(call_block);
        b.set_insert_point_before(call_id);
        input_meta
            .iter()
            .map(|&(r, size, space)| {
                let ptr = access_ptr(&mut b, r);
                b.push_load::<false>(ptr, size, space)
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
    let Some(cont) = replay_block(ctx, call_block) else {
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
        let ptr = access_ptr(&mut b, r);
        b.push_store(ext, ptr, space);
    }
}

/// Pass 3 for a **bodyless external** materialized callee (design ruling 7a).
/// Mirrors [`rewrite_call_regpure`] for bodied callees but owns the whole
/// rewrite: register args become explicit regpure operands; the result is typed
/// as the pack aggregate (return ∪ clobbers); the continuation replays the return
/// register(s) from the pack and stores **poison** into every clobber register
/// (a static alias/dataflow device — externals have no runtime semantics here).
/// It also keeps emitting the SP-relative stack-arg loads `bind_external_args`
/// did (implicit RAM, left in the body — the call is `RegPure`, not `Pure`).
fn rewrite_external_call_regpure(
    ctx: &mut Context,
    call_id: InstructionId,
    callee: FunctionId,
    map: &RegisterInterfaceMap,
    sp: Option<VarnodeId>,
) {
    let Some(call_block) = ctx.get_insn(call_id).parent().map(|b| b.id) else {
        return;
    };

    // --- register inputs: load each into a regpure argument before the call -----
    let input_meta: Vec<(VarnodeId, usize, SpaceId)> = map
        .inputs
        .iter()
        .map(|&r| {
            let v = Varnode::from_id(&*ctx, r);
            (r, v.size(), v.space().id)
        })
        .collect();
    // Stack-arg slots (implicit RAM): emitted as standalone SP-relative loads
    // before the call, exactly as `bind_external_args` did. Left in the body
    // (not threaded as regpure operands, which must match `map.inputs` 1:1).
    let stack_slots: Vec<(i64, usize)> = FunctionBody::from_id(ctx, callee)
        .extern_interface()
        .map(|iface| {
            iface
                .args
                .iter()
                .filter_map(|arg| match arg.slot {
                    qcode::value::ExternSlot::Stack { offset, size } => Some((offset, size)),
                    qcode::value::ExternSlot::Reg(..) => None,
                })
                .collect()
        })
        .unwrap_or_default();
    let ptr_width = sp.map(|s| Varnode::from_id(&*ctx, s).size()).unwrap_or(8);
    let default_space = ctx.shared.default_space;

    let args: Vec<LocalValueId> = {
        let mut b = (ctx).builder(call_block);
        b.set_insert_point_before(call_id);
        // Register args become the regpure operands (in `map.inputs` order).
        let args: Vec<LocalValueId> = input_meta
            .iter()
            .map(|&(r, size, space)| {
                b.push_load::<false>(ValueId::Varnode(r), size, space)
                    .id()
                    .localize(call_id.func)
            })
            .collect();
        // Stack args: emit the SP-relative RAM loads (implicit RAM effect); not
        // added to `Call.args`.
        if let Some(sp) = sp {
            let sp_space = Varnode::from_id(b.shr(), sp).space().id;
            for &(offset, size) in &stack_slots {
                let sp_val = b
                    .push_load::<false>(ValueId::Varnode(sp), ptr_width, sp_space)
                    .id();
                let addr = if offset == 0 {
                    sp_val
                } else {
                    let off = b.shr().get_const(offset as u64, ptr_width);
                    b.push_add(sp_val, off).id()
                };
                b.push_load::<false>(addr, size, default_space);
            }
        }
        args
    };

    // The pack aggregate type: one field per output register (return ∪ clobbers).
    let field_types: Vec<TypeId> = map
        .outputs
        .iter()
        .map(|&r| {
            let size = Varnode::from_id(&*ctx, r).size();
            ctx.shared.types.get_or_make_int(size)
        })
        .collect();
    let ret_ty = ctx.shared.types.get_or_make_aggregate(field_types);

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

    // --- outputs: replay the return register(s) from the pack; poison clobbers ---
    let Some(cont) = replay_block(ctx, call_block) else {
        return;
    };
    // Returns-first pack ordering: slots `..map.returns` are real return
    // values, the tail is the clobber set (poison at the call site).
    let output_meta: Vec<(VarnodeId, usize, SpaceId, bool)> = map
        .outputs
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            let v = Varnode::from_id(&*ctx, r);
            (r, v.size(), v.space().id, i < map.returns)
        })
        .collect();
    let result = ValueId::Instruction(call_id);
    let mut b = (ctx).builder(cont);
    b.set_insert_point_to_start();
    for (i, &(r, size, space, is_return)) in output_meta.iter().enumerate() {
        let value = if is_return {
            ValueId::Instruction(b.push_extract(result, i).id)
        } else {
            // A clobber slot: the register holds an undefined value after the call.
            let ty = b.shr().types.get_or_make_int(size);
            ValueId::Poison(b.shr().values.push_poison(ty))
        };
        b.push_store(value, ValueId::Varnode(r), space);
    }
}

/// Rewrite every direct `Opaque` call to a materialized function into a regpure
/// call (pass 3). Returns the set of *caller* functions changed.
pub(crate) fn regpure_all_sites(
    ctx: &mut Context,
    graph: &CallGraph,
    targets: &[FunctionId],
    sp: Option<VarnodeId>,
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
                rewrite_call_regpure(ctx, site, callee, sp);
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
    changed.extend(regpure_all_sites(ctx, &graph, &targets, sp));
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
    let sp = guess_sp(ctx);
    for site in crate::calls::direct_call_sites(ctx, &graph, fid) {
        rewrite_call_regpure(ctx, site, fid, sp);
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
        env: &PipelineEnv,
        targets: &[FunctionId],
    ) -> Result<crate::ModulePassOutcome, String> {
        let graph = CallGraph::analyze(ctx);
        Ok(crate::ModulePassOutcome::functions(regpure_all_sites(
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
        Ok(crate::ModulePassOutcome::functions(regpure_all_sites(
            ctx,
            graph,
            targets,
            env.sp_varnode,
        ))
        .preserving_global::<CallGraphAnalysis>()
        .preserving_global::<crate::AddressAnalysis>())
    }
}

crate::register_module_pass!(ArgPromoteRegpureCalls);
