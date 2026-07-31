//! Register instantiation of the [`EffectChannel`](super::summary::EffectChannel)
//! fixpoint (see `ARGPROMOTE_REGISTERS_V2.md`).
//!
//! Effects are two grow-only sets over the (finite) register file — registers
//! loaded and registers stored — so the lattice height is trivially bounded.
//! The transfer is the identity: the register file is a global namespace, so a
//! callee's effects union straight into every caller. Resolved externals are
//! leaves: their caller-side rewrite (`argpromote_external`) materializes each
//! register argument as a `load(register, R)` and the ABI return as a
//! `store(register, R)` in the caller, and the external itself clobbers its
//! ABI caller-saved set — all of which must appear in the caller's summary
//! *before* the caller's interface is frozen.

use qcode::{
    context::Context,
    value::{
        BasicBlock, BlockId, ExternSlot, FunctionBody, FunctionId, RegisterChannelState,
        RegisterInterfaceMap, Varnode, VarnodeId, insn::Mnemonic,
    },
};
use rustc_hash::FxHashSet;

use crate::calls::{
    CallEdge,
    effect_engine::{Summary, TopCause},
};

use super::{
    registers::{RegPurityReason, RegisterEffects, is_register},
    summary::EffectChannel,
};

/// The register channel's effect element: registers this function (or its
/// callees, once joined) may read / may write. Raw varnode sets are retained
/// exactly through summary solving. At materialization, output keys fully
/// contained by another output are represented by that covering register.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RegEffects {
    pub(crate) reads: FxHashSet<VarnodeId>,
    pub(crate) writes: FxHashSet<VarnodeId>,
}

impl RegEffects {
    /// The persistable form of this solved lattice value: sorted, deduplicated
    /// load/store lists, stored on the interface as
    /// [`RegisterChannelState::Solved`](qcode::value::FunctionEffects).
    pub(crate) fn to_sets(&self) -> qcode::value::RegisterEffectSets {
        let mut reads: Vec<VarnodeId> = self.reads.iter().copied().collect();
        let mut writes: Vec<VarnodeId> = self.writes.iter().copied().collect();
        reads.sort_unstable();
        writes.sort_unstable();
        qcode::value::RegisterEffectSets { reads, writes }
    }
}

/// The register [`EffectChannel`]. `sp` is the stack-pointer varnode, needed
/// only to model the SP reload `argpromote_external` emits for stack-passed
/// external arguments; without it such externals are ⊤.
///
/// DEBT(sp-normalization): this field exists purely to keep the summary in sync
/// with `registers::rewrite_external_call_regpure`, which emits those SP-relative
/// loads. The two are coupled by convention, not by construction: change the
/// lowering and this summary silently under-reports. The generic replacement is
/// for the lowering to *report* the registers it materialized (an effect delta
/// on the rewrite) so the channel reads a fact instead of re-deriving one — at
/// which point the stack pointer needs no named slot here at all.
pub(crate) struct RegChannel {
    pub(crate) sp: Option<VarnodeId>,
    /// The interface an unresolved indirect (`CallInd`) callee is modelled with:
    /// the platform-ABI clobber leaf
    /// ([`abi_clobber_leaf`](crate::assumptions::abi_clobber_leaf)), the same one
    /// an unprototyped external gets. `None` when the architecture has no
    /// modelled convention (hand-written IR), which leaves indirect calls
    /// contributing nothing — see [`RegChannel::indirect_call_effects`].
    pub(crate) indirect: Option<RegisterInterfaceMap>,
}

impl RegChannel {
    fn materialized_effects(
        &self,
        ctx: &Context,
        fid: FunctionId,
        map: &RegisterInterfaceMap,
    ) -> RegEffects {
        let mut eff = RegEffects::default();
        eff.reads.extend(map.inputs.iter().copied());
        eff.writes.extend(map.outputs.iter().copied());
        // A stack-passed argument is materialized at the caller as an SP reload +
        // a ram load; only the SP read is a register effect (the ram load is the
        // other channel's business). Register args are already in `map.inputs`.
        if let Some(iface) = FunctionBody::from_id(ctx, fid).extern_interface()
            && iface
                .args
                .iter()
                .any(|arg| matches!(arg.slot, ExternSlot::Stack { .. }))
        {
            eff.reads.extend(self.sp);
        }
        eff
    }
}

/// Whether some reachable return preserves the incoming value of `register`.
///
/// The search state records whether the current path has already performed a
/// covering write. Re-visiting a loop in the same state cannot reveal a new
/// path, so `(block, written)` is a finite traversal even for cyclic CFGs.
fn has_return_path_without_write(ctx: &Context, fid: FunctionId, register: VarnodeId) -> bool {
    let Some(root) = FunctionBody::from_id(ctx, fid).root().map(|block| block.id) else {
        return false;
    };
    let target = Varnode::from_id(ctx, register);
    let target_space = target.space().id;
    let target_start = target.address();
    let target_end = target_start + target.size() as i64;
    let covers = |candidate: VarnodeId| {
        let candidate = Varnode::from_id(ctx, candidate);
        let start = candidate.address();
        candidate.space().id == target_space
            && start <= target_start
            && start + candidate.size() as i64 >= target_end
    };

    let mut pending = vec![(root, false)];
    let mut seen: FxHashSet<(BlockId, bool)> = FxHashSet::default();
    while let Some((block_id, mut written)) = pending.pop() {
        if !seen.insert((block_id, written)) {
            continue;
        }
        let block = BasicBlock::from_id(ctx, block_id);
        for insn in block.iter() {
            match insn.mnemonic() {
                Mnemonic::Store(store)
                    if matches!(
                        store.ptr,
                        qcode::value::LocalValueId::Varnode(vn) if covers(vn)
                    ) =>
                {
                    written = true;
                }
                Mnemonic::Return(_) if !written => return true,
                _ => {}
            }
        }
        pending.extend(
            block
                .successors()
                .map(|(_, successor)| (successor, written)),
        );
    }
    false
}

/// Converts an out-of-cone function's persisted register state into a fixed
/// effect-engine leaf. An unsolved state is not safe to reuse and therefore
/// conservatively becomes ⊤, just like a persisted explicit [`Top`](RegisterChannelState::Top).
#[allow(dead_code)] // Wired by the incremental cone solver in the next slice.
pub(crate) fn fixed_summary_from_register_state(
    channel: &RegChannel,
    ctx: &Context,
    fid: FunctionId,
    state: &RegisterChannelState,
) -> Summary<RegEffects> {
    match state {
        RegisterChannelState::Solved(sets) => Ok(RegEffects {
            reads: sets.reads.iter().copied().collect(),
            writes: sets.writes.iter().copied().collect(),
        }),
        RegisterChannelState::Materialized(map) => Ok(channel.materialized_effects(ctx, fid, map)),
        RegisterChannelState::Top | RegisterChannelState::Unsolved => Err(TopCause::Channel),
    }
}

impl EffectChannel for RegChannel {
    type Effects = RegEffects;

    fn scan(&self, ctx: &Context, fid: FunctionId) -> Option<RegEffects> {
        let mut eff = RegEffects::default();
        for block in FunctionBody::from_id(ctx, fid).blocks() {
            for insn in block.iter() {
                match insn.mnemonic() {
                    // Only true register varnodes enter the load/store effect
                    // sets. Globals (constant real-RAM addresses) are the RAM
                    // channel's concern (see `argpromote::ram`).
                    Mnemonic::Load(l) => {
                        if let qcode::value::LocalValueId::Varnode(vn) = l.ptr
                            && is_register(ctx, vn)
                        {
                            eff.reads.insert(vn);
                        }
                    }
                    Mnemonic::Store(s) => {
                        if let qcode::value::LocalValueId::Varnode(vn) = s.ptr
                            && is_register(ctx, vn)
                        {
                            eff.writes.insert(vn);
                        }
                    }
                    _ => {}
                }
            }
        }
        // A register that is only conditionally written must preserve its
        // incoming value on the other return path. That dependency is a
        // semantic read and is discovered here, before materialization maps
        // the exact read/write keys to interface slots.
        let preserved: Vec<_> = eff
            .writes
            .iter()
            .copied()
            .filter(|&register| has_return_path_without_write(ctx, fid, register))
            .collect();
        eff.reads.extend(preserved);
        Some(eff)
    }

    fn external_leaf(&self, ctx: &Context, fid: FunctionId) -> Option<RegEffects> {
        let f = FunctionBody::from_id(ctx, fid);
        // A prototyped external is materialized by `external_sigs`, which stamps
        // the single source of truth (ruling 6a): `effects = Materialized(map)`
        // with `map.inputs` = argument registers and `map.outputs` = return
        // register(s) ∪ ABI caller-saved clobbers. Derive the leaf effects from
        // that mapping rather than recomputing — crucially the stores now include
        // the clobber set, so a materialized caller's return pack covers them
        // (bug-2-external, ARGPROMOTE_REGISTERS_V2.md Phase 3).
        //
        // A *prototype-less* external is materialized too, from the calling
        // convention alone (`external_sig::apply_abi_fallback_signature`): "obeys
        // the platform ABI" is a sound assumption for an arbitrary unprototyped
        // symbol, so its inputs over-approximate to every argument register and
        // its outputs to the return register(s) ∪ every caller-saved register.
        //
        // The variant this branch still rejects — and the one an earlier comment
        // here argued against — is leaving such an external's clobbers as the
        // lifter's *in-body declared* clobbers while handing the engine an
        // empty-effects leaf. Declared-clobber semantics only cover the external's
        // *direct* caller's body, so a solved summary for that caller would omit
        // the external's real clobbers and let *its* callers forward caller-saved
        // registers across the call (bug-2-external one level up). The ABI
        // fallback avoids that precisely because the clobbers go *into* the
        // `Materialized` map and so propagate up the cone exactly like a
        // prototyped external's do.
        //
        // What remains ⊤ here is an external carrying no mapping at all: an
        // architecture with no modelled convention, or a context in which
        // `external_sigs` never ran.
        let RegisterChannelState::Materialized(map) = &f.effects().register else {
            return None;
        };
        Some(self.materialized_effects(ctx, fid, map))
    }

    /// An unresolved `CallInd` target is exactly an *unprototyped* callee, so it
    /// carries the same platform-ABI clobber leaf an unprototyped external gets
    /// (`self.indirect`): it reads every argument-passing register and writes the
    /// return register(s) ∪ every caller-saved register.
    ///
    /// This is not a hypothesis layered on top of the IR — it *mirrors* what
    /// `registers::regpure_indirect_sites` materializes into the body at every
    /// indirect call site (a `load(register, R)` per input, an
    /// `extract`/poison `store(register, R)` per output). The summary must
    /// therefore contain it, or the enclosing function's frozen interface would
    /// omit register traffic its own body performs — bug-2-external one level up.
    /// The earlier empty-effects modelling claimed an indirect call clobbers
    /// *nothing*, which let a caller forward caller-saved registers across it.
    ///
    /// With no modelled convention (`indirect == None`, e.g. hand-written IR)
    /// this stays empty — never ⊤, matching the pre-ABI-leaf behaviour. The RAM
    /// channel must *not* copy any of this: an unknown callee's memory effects
    /// have no in-body fallback there.
    fn indirect_call_effects(&self, _ctx: &Context, _fid: FunctionId) -> Option<RegEffects> {
        Some(match &self.indirect {
            Some(map) => RegEffects {
                reads: map.inputs.iter().copied().collect(),
                writes: map.outputs.iter().copied().collect(),
            },
            None => RegEffects::default(),
        })
    }

    fn transfer(
        &self,
        _ctx: &Context,
        _edge: &CallEdge,
        callee: &RegEffects,
    ) -> Option<RegEffects> {
        // Registers are a global namespace (identity transfer): a callee's
        // read/write of a register is a read/write of that same register in
        // every caller.
        Some(RegEffects {
            reads: callee.reads.clone(),
            writes: callee.writes.clone(),
        })
    }

    fn join(&self, into: &mut RegEffects, from: &RegEffects) -> bool {
        let before = (into.reads.len(), into.writes.len());
        into.reads.extend(from.reads.iter().copied());
        into.writes.extend(from.writes.iter().copied());
        (into.reads.len(), into.writes.len()) != before
    }
}

/// Finalize a solved summary into the interface [`RegisterEffects`] the rewrite
/// consumes: retain every read key and every maximal write interval, then fix a
/// deterministic `(address, size)` order shared with the caller rewrite.
///
/// A contained output is redundant: the final value loaded from its covering
/// register already contains those bytes. Partial overlaps remain distinct
/// because neither register represents the other's complete final value.
pub(crate) fn finalize_register_effects(
    ctx: &Context,
    eff: &RegEffects,
) -> Result<RegisterEffects, RegPurityReason> {
    if eff.writes.is_empty() {
        return Err(RegPurityReason::NoRegisterWrites);
    }
    let mut outputs: Vec<VarnodeId> = eff.writes.iter().copied().collect();
    let mut inputs: Vec<VarnodeId> = eff.reads.iter().copied().collect();
    outputs.retain(|&candidate| {
        let candidate = Varnode::from_id(ctx, candidate);
        !eff.writes.iter().copied().any(|other| {
            let other = Varnode::from_id(ctx, other);
            other.id != candidate.id
                && other.space().id == candidate.space().id
                && other.address() <= candidate.address()
                && other.address() + other.size() as i64
                    >= candidate.address() + candidate.size() as i64
        })
    });

    // Deterministic order shared by callee param creation and the regpure-site
    // argument threading: sort by `(space, address, size)`.
    let key = |ctx: &Context, vn: &VarnodeId| {
        let v = Varnode::from_id(ctx, *vn);
        (v.space().id, v.address(), v.size())
    };
    inputs.sort_by_key(|vn| key(ctx, vn));
    outputs.sort_by_key(|vn| key(ctx, vn));
    Ok(RegisterEffects { inputs, outputs })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CallGraph;
    use qcode::value::FunctionBody;
    use qcode::value::RegisterChannelState;
    use qcode_macro::qcode;

    use super::super::summary::{EffectSummaries, solve_summaries, solve_summaries_with_fixed};

    fn solve(tc: &qcode::testing::TestContext) -> EffectSummaries<RegChannel> {
        let graph = CallGraph::analyze(&tc.ctx);
        solve_summaries(
            &tc.ctx,
            &graph,
            &RegChannel {
                sp: None,
                indirect: None,
            },
        )
    }

    /// A leaf function's summary is exactly its own syntactic register traffic.
    #[test]
    fn leaf_scan() {
        let mut tc = qcode::testing::TestContext::new();
        let (r0, r1) = (tc.r0, tc.r1);
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry>
                    %v = load(register:8, {r1});
                    store(register:8, {r0} <- %v);
                    return at i64 0;
            "
        );
        let _ = f_entry;
        let s = solve(&tc);
        let eff = s.get(f).as_ref().unwrap();
        assert!(eff.reads.contains(&r1));
        assert!(eff.writes.contains(&r0));
    }

    /// A caller's summary is the *union* of its own effects and every callee's
    /// (bug 2 regression at the analysis level: the caller's output set can never
    /// silently omit a callee's clobber).
    #[test]
    fn caller_unions_callee() {
        let mut tc = qcode::testing::TestContext::new();
        let (r0, r1) = (tc.r0, tc.r1);
        qcode!(
            tc.ctx,
            "
            fn callee:
                <callee_entry>
                    store(register:8, {r0} <- i64 7);
                    return at i64 0;
            fn caller:
                <caller_entry>
                    store(register:8, {r1} <- i64 9);
                    call fn callee();
                <caller_cont>
                    return at i64 0;
            "
        );
        let _ = (callee_entry, caller_entry, caller_cont);
        let s = solve(&tc);
        let eff = s.get(caller).as_ref().unwrap();
        assert!(
            eff.writes.contains(&r0) && eff.writes.contains(&r1),
            "caller's write-set unions the callee's clobber (r0) with its own (r1)"
        );
    }

    /// Self-recursion and a 2-cycle both reach a fixpoint (the lattice is finite
    /// and the transfer monotone; cycles need no special casing).
    #[test]
    fn recursion_reaches_fixpoint() {
        let mut tc = qcode::testing::TestContext::new();
        let (r0, r1) = (tc.r0, tc.r1);
        qcode!(
            tc.ctx,
            "
            fn a:
                <a_entry>
                    store(register:8, {r0} <- i64 1);
                    call fn b();
                <a_cont>
                    return at i64 0;
            fn b:
                <b_entry>
                    store(register:8, {r1} <- i64 2);
                    call fn a();
                <b_cont>
                    return at i64 0;
            "
        );
        let _ = (a_entry, a_cont, b_entry, b_cont);
        let s = solve(&tc);
        for fid in [a, b] {
            let eff = s.get(fid).as_ref().unwrap();
            assert!(
                eff.writes.contains(&r0) && eff.writes.contains(&r1),
                "the 2-cycle unions both members' stores into each"
            );
        }
    }

    /// A `CallInd` keeps its in-body clobbers-all modelling, so for the register
    /// channel it contributes *empty* effects — the function is NOT ⊤.
    #[test]
    fn indirect_call_is_empty_not_top() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry>
                    %ptr = load(register:8, {r0});
                    call [%ptr]();
                <f_cont>
                    return at i64 0;
            "
        );
        let _ = (f_entry, f_cont);
        let s = solve(&tc);
        let eff = s
            .get(f)
            .as_ref()
            .expect("CallInd is empty-effects, not ⊤, for the register channel");
        // Its own `load(register, r0)` is still scanned; the indirect call adds
        // nothing (no stores).
        assert!(eff.reads.contains(&r0));
        assert!(eff.writes.is_empty());
    }

    /// An external carrying *no* interface mapping is ⊤: an empty-effects leaf
    /// would let a solved caller summary omit the external's real clobbers and
    /// mislead the caller's own callers (bug-2-external one level up). Note this
    /// is now the un-stamped case only (no modelled convention, or
    /// `external_sigs` never ran) — a prototype-less external under a known ABI
    /// is stamped `Materialized` from the convention alone.
    #[test]
    fn unmapped_external_is_top() {
        let mut tc = qcode::testing::TestContext::new();
        let ext = FunctionBody::make_external(&mut tc.ctx, 0x9000, Some("noproto".into())).id;
        let s = solve(&tc);
        assert!(
            matches!(
                s.get(ext),
                Err(crate::calls::effect_engine::TopCause::External)
            ),
            "an external with no interface mapping must be ⊤, not an empty-effects leaf"
        );
    }

    /// bug-2-external: a caller of a prototyped external inherits the external's
    /// ABI clobbers into its own solved write-set (via `external_leaf`'s
    /// `Materialized` outputs), so a materialized caller's return pack covers
    /// them. Here `ext` is a `Materialized` external clobbering `r0`; the caller
    /// `c` (which also stores `r1`) unions `r0` into its stores.
    #[test]
    fn caller_inherits_external_clobber() {
        use qcode::value::{
            QCodeMut, RegisterInterfaceMap, ValueId,
            insn::{Call, Callee, Mnemonic as M},
        };
        let mut tc = qcode::testing::TestContext::new();
        let (r0, r1) = (tc.r0, tc.r1);

        // The prototyped external, stamped Materialized with r0 in its clobber
        // (output) pack — what `external_sigs` would produce.
        let ext = FunctionBody::make_external(&mut tc.ctx, 0x9000, Some("ext".into())).id;
        FunctionBody::from_id_mut(&mut tc.ctx, ext).set_register_effects(
            RegisterChannelState::Materialized(RegisterInterfaceMap {
                inputs: vec![],
                outputs: vec![r0],
                returns: 0,
                projections: Vec::new(),
            }),
        );

        // Caller `c`: stores r1, then calls the external.
        let cid = FunctionBody::make(&mut tc.ctx, "c".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x4000, cid);
        let cont = tc.ctx.get_or_make_block(0x4100, cid);
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, cid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(cont);
        }
        let reg_space = tc.reg_space;
        let call_id;
        {
            let mut b = tc.ctx.builder(entry);
            let c9 = b.shr().get_const(9, 8);
            b.push_store(c9, ValueId::Varnode(r1), reg_space);
            let ValueId::Instruction(id) = b.push_call(ext).id() else {
                unreachable!()
            };
            call_id = id;
        }
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            M::Call(Call {
                target: Callee::Real(ext),
                args: vec![],
                clobbers: vec![],
                tag: Default::default(),
            }),
        );
        tc.ctx.add_cfg_edge(entry, cont);
        {
            let mut b = tc.ctx.builder(cont);
            let z = b.shr().get_const(0, 8);
            b.push_return(z);
        }

        let s = solve(&tc);
        let eff = s.get(cid).as_ref().unwrap();
        assert!(
            eff.writes.contains(&r0),
            "caller must inherit the external's r0 clobber into its write-set"
        );
        assert!(eff.writes.contains(&r1), "caller keeps its own r1 store");
    }

    /// A covering output subsumes contained register cells.
    #[test]
    fn finalize_collapses_contained_output_keys() {
        let tc = qcode::testing::TestContext::new();
        let (wide, lo32, byte1) = (tc.r0, tc.r0_lo32, tc.r0_byte1);
        let mut eff = RegEffects::default();
        eff.writes.insert(wide);
        eff.writes.insert(lo32);
        eff.writes.insert(byte1);
        let iface = finalize_register_effects(&tc.ctx, &eff).unwrap();
        assert_eq!(iface.outputs, vec![wide]);
    }

    #[test]
    fn persisted_states_convert_to_fixed_summaries_conservatively() {
        use qcode::value::{RegisterEffectSets, RegisterInterfaceMap};

        let mut tc = qcode::testing::TestContext::new();
        let fid = FunctionBody::make(&mut tc.ctx, "fixed".into()).unwrap().id;
        let channel = RegChannel {
            sp: None,
            indirect: None,
        };
        let (r0, r1) = (tc.r0, tc.r1);

        let solved = RegisterChannelState::Solved(RegisterEffectSets {
            reads: vec![r0],
            writes: vec![r1],
        });
        let effects = fixed_summary_from_register_state(&channel, &tc.ctx, fid, &solved).unwrap();
        assert_eq!(effects.reads, FxHashSet::from_iter([r0]));
        assert_eq!(effects.writes, FxHashSet::from_iter([r1]));

        let materialized = RegisterChannelState::Materialized(RegisterInterfaceMap {
            inputs: vec![r1],
            outputs: vec![r0],
            returns: 1,
            projections: Vec::new(),
        });
        let effects =
            fixed_summary_from_register_state(&channel, &tc.ctx, fid, &materialized).unwrap();
        assert_eq!(effects.reads, FxHashSet::from_iter([r1]));
        assert_eq!(effects.writes, FxHashSet::from_iter([r0]));

        for state in [RegisterChannelState::Top, RegisterChannelState::Unsolved] {
            assert_eq!(
                fixed_summary_from_register_state(&channel, &tc.ctx, fid, &state),
                Err(TopCause::Channel)
            );
        }
    }

    /// A write-only register stays write-only after materialization, so using a
    /// materialized function as an out-of-cone fixed leaf agrees with a full
    /// solve.
    #[test]
    fn partial_solve_with_materialized_bodied_leaf_matches_full_solve() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn callee:
                <callee_entry>
                    store(register:8, {r0} <- i64 7);
                    return at i64 0;
            fn caller:
                <caller_entry>
                    call fn callee();
                <caller_cont>
                    return at i64 0;
            "
        );
        let _ = (callee_entry, caller_entry, caller_cont);
        let graph = CallGraph::analyze(&tc.ctx);
        let channel = RegChannel {
            sp: None,
            indirect: None,
        };
        let full = solve_summaries(&tc.ctx, &graph, &channel)
            .get(caller)
            .clone();

        let summaries = solve_summaries(&tc.ctx, &graph, &channel);
        let callee_effects = summaries.get(callee).as_ref().unwrap();
        let interface = finalize_register_effects(&tc.ctx, callee_effects).unwrap();
        super::super::registers::materialize_interface(&mut tc.ctx, callee, &interface);
        let state = FunctionBody::from_id(&tc.ctx, callee)
            .effects()
            .register
            .clone();
        let RegisterChannelState::Materialized(map) = &state else {
            panic!("materialization must persist its interface map");
        };
        assert!(map.inputs.is_empty(), "write-only r0 must remain unread");
        assert_eq!(map.outputs, vec![r0]);

        let fixed = [(
            callee,
            fixed_summary_from_register_state(&channel, &tc.ctx, callee, &state),
        )]
        .into_iter()
        .collect();
        let partial = solve_summaries_with_fixed(&tc.ctx, &graph, &channel, &fixed)
            .get(caller)
            .clone();

        assert_eq!(
            partial, full,
            "partial solving must preserve the exact materialized read/write keys"
        );
    }

    #[test]
    fn covering_write_key_survives_materialization_and_partial_solve() {
        let mut tc = qcode::testing::TestContext::new();
        let (wide, narrow) = (tc.r0, tc.r0_lo32);
        qcode!(
            tc.ctx,
            "
            fn callee:
                <callee_entry>
                    store(register:4, {narrow} <- i32 7);
                    store(register:8, {wide} <- i64 9);
                    return at i64 0;
            fn caller:
                <caller_entry>
                    call fn callee();
                <caller_cont>
                    return at i64 0;
            "
        );
        let _ = (callee_entry, caller_entry, caller_cont);
        let graph = CallGraph::analyze(&tc.ctx);
        let channel = RegChannel {
            sp: None,
            indirect: None,
        };
        let summaries = solve_summaries(&tc.ctx, &graph, &channel);
        let full = summaries.get(caller).clone();
        let semantic = summaries.get(callee).as_ref().unwrap();
        assert_eq!(semantic.writes, [wide, narrow].into_iter().collect());

        let interface = finalize_register_effects(&tc.ctx, semantic).unwrap();
        super::super::registers::materialize_interface(&mut tc.ctx, callee, &interface);
        let state = FunctionBody::from_id(&tc.ctx, callee)
            .effects()
            .register
            .clone();
        let RegisterChannelState::Materialized(map) = &state else {
            panic!("materialization must persist its interface map");
        };
        assert_eq!(map.outputs, vec![wide]);
        assert_eq!(map.returns, 1, "the covering key needs one replayable slot");

        let fixed = [(
            callee,
            fixed_summary_from_register_state(&channel, &tc.ctx, callee, &state),
        )]
        .into_iter()
        .collect();
        let partial = solve_summaries_with_fixed(&tc.ctx, &graph, &channel, &fixed)
            .get(caller)
            .clone();
        let partial = partial.expect("materialized covering output remains solved");
        assert_eq!(partial.writes, [wide].into_iter().collect());
        assert!(
            full.as_ref().unwrap().writes.contains(&wide),
            "the full summary contains the same covering write"
        );
    }

    #[test]
    fn finalization_preserves_exact_read_write_sets() {
        let tc = qcode::testing::TestContext::new();
        let effects = RegEffects {
            reads: [tc.r1].into_iter().collect(),
            writes: [tc.r0].into_iter().collect(),
        };

        let interface = finalize_register_effects(&tc.ctx, &effects).unwrap();

        assert_eq!(interface.inputs, vec![tc.r1]);
        assert_eq!(interface.outputs, vec![tc.r0]);
    }

    #[test]
    fn unconditional_write_does_not_become_a_read() {
        let tc = qcode::testing::TestContext::new();
        let effects = RegEffects {
            reads: FxHashSet::default(),
            writes: [tc.r0].into_iter().collect(),
        };

        let interface = finalize_register_effects(&tc.ctx, &effects).unwrap();

        assert!(interface.inputs.is_empty());
        assert_eq!(interface.outputs, vec![tc.r0]);
    }

    #[test]
    fn conditional_write_is_also_a_read_before_materialization() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry @condition:i8>
                    if @condition goto <written> else goto <preserved>;
                <written>
                    store(register:8, {r0} <- i64 7);
                    return at i64 0;
                <preserved>
                    return at i64 0;
            "
        );
        let _ = (entry, written, preserved);
        let graph = CallGraph::analyze(&tc.ctx);
        let channel = RegChannel {
            sp: None,
            indirect: None,
        };
        let summaries = solve_summaries(&tc.ctx, &graph, &channel);
        let effects = summaries.get(f).as_ref().unwrap();

        assert!(effects.reads.contains(&r0));
        assert!(effects.writes.contains(&r0));

        let interface = finalize_register_effects(&tc.ctx, effects).unwrap();
        super::super::registers::materialize_interface(&mut tc.ctx, f, &interface);
        let RegisterChannelState::Materialized(map) =
            &FunctionBody::from_id(&tc.ctx, f).effects().register
        else {
            panic!("materialization must persist its interface map");
        };
        assert_eq!(map.inputs, vec![r0]);
        assert_eq!(map.outputs, vec![r0]);
    }
}
