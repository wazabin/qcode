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
    value::{ExternSlot, FunctionBody, FunctionId, Varnode, VarnodeId, insn::Mnemonic},
};
use rustc_hash::FxHashSet;

use crate::calls::CallEdge;

use super::{
    registers::{RegPurityReason, RegisterEffects, canonicalize_to_coarsest, is_register},
    summary::EffectChannel,
};

/// The register channel's effect element: registers this function (or its
/// callees, once joined) may read / may write. Raw varnode sets — overlap
/// canonicalization happens once, in [`finalize_register_effects`], after the
/// fixpoint has finished growing them.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RegEffects {
    pub(crate) loads: FxHashSet<VarnodeId>,
    pub(crate) stores: FxHashSet<VarnodeId>,
}

/// The register [`EffectChannel`]. `sp` is the stack-pointer varnode, needed
/// only to model the SP reload `argpromote_external` emits for stack-passed
/// external arguments; without it such externals are ⊤.
pub(crate) struct RegChannel {
    pub(crate) sp: Option<VarnodeId>,
}

impl EffectChannel for RegChannel {
    type Effects = RegEffects;

    fn scan(&self, ctx: &Context, fid: FunctionId) -> Option<RegEffects> {
        let mut eff = RegEffects::default();
        for block in FunctionBody::from_id(ctx, fid).blocks() {
            for insn in block.iter() {
                match insn.mnemonic() {
                    Mnemonic::Load(l) => {
                        if let qcode::value::LocalValueId::Varnode(vn) = l.ptr
                            && is_register(ctx, vn)
                        {
                            eff.loads.insert(vn);
                        }
                    }
                    Mnemonic::Store(s) => {
                        if let qcode::value::LocalValueId::Varnode(vn) = s.ptr
                            && is_register(ctx, vn)
                        {
                            eff.stores.insert(vn);
                        }
                    }
                    _ => {}
                }
            }
        }
        Some(eff)
    }

    fn external_leaf(&self, ctx: &Context, fid: FunctionId) -> Option<RegEffects> {
        let f = FunctionBody::from_id(ctx, fid);
        let mut eff = RegEffects::default();
        // A materialized interface predicts the caller-side rewrite
        // `argpromote_external` will perform: a `load(register, R)` per register
        // argument (an SP reload for stack-passed ones) and a `store(register,
        // R)` of the ABI return — count those so a promoted caller's interface
        // covers them before they are injected. In the default pipeline the
        // register channel runs *before* `external_sigs`, so the interface is
        // usually absent here; an external without one still contributes empty
        // effects rather than ⊤, because its call instruction stays in the
        // promoted body carrying the declared-clobber semantics mem2reg models
        // (v1-compatible; strict tainting is the Phase 3 pipeline reorder —
        // see `ARGPROMOTE_REGISTERS_V2.md`).
        if let Some(iface) = f.extern_interface() {
            for arg in &iface.args {
                match arg.slot {
                    ExternSlot::Reg(vn, _) => {
                        eff.loads.insert(vn);
                    }
                    // A stack-passed argument is materialized at the caller as
                    // an SP reload + a ram load; only the SP read is a register
                    // effect. Without a known SP there is nothing to record —
                    // the ram load is the other channel's business.
                    ExternSlot::Stack { .. } => {
                        eff.loads.extend(self.sp);
                    }
                }
            }
        }
        if let Some(sig) = f.signature() {
            // The ABI return register(s): stored back into the caller by
            // `bind_external_return`.
            eff.stores.extend(sig.outputs.iter().flatten().copied());
        }
        Some(eff)
    }

    /// A `CallInd` keeps its intrinsic clobbers-all modelling in the promoted
    /// body (mem2reg treats it as reading and clobbering the register file), so
    /// it contributes no interface effects of its own — v1-compatible. The RAM
    /// channel must *not* copy this: an unknown callee's memory effects have no
    /// in-body fallback there.
    fn indirect_call_effects(&self, _ctx: &Context, _fid: FunctionId) -> Option<RegEffects> {
        Some(RegEffects::default())
    }

    fn transfer(
        &self,
        _ctx: &Context,
        _edge: &CallEdge,
        callee: &RegEffects,
    ) -> Option<RegEffects> {
        Some(callee.clone())
    }

    fn join(&self, into: &mut RegEffects, from: &RegEffects) -> bool {
        let before = (into.loads.len(), into.stores.len());
        into.loads.extend(from.loads.iter().copied());
        into.stores.extend(from.stores.iter().copied());
        (into.loads.len(), into.stores.len()) != before
    }
}

/// Finalize a solved summary into the interface [`RegisterEffects`] the rewrite
/// consumes: canonicalize stores to the coarsest register per overlap group,
/// over-approximate every output as an input too (a no-write path reads the
/// caller's incoming value at the return pack — see `scan_register_effects`),
/// and fix a deterministic `(address, size)` order shared with the caller
/// rewrite. Mirrors v1's gating: no stores ⇒ nothing to functionalize; an
/// overlap group with no single covering register ⇒ not promotable.
pub(crate) fn finalize_register_effects(
    ctx: &Context,
    eff: &RegEffects,
) -> Result<RegisterEffects, RegPurityReason> {
    if eff.stores.is_empty() {
        return Err(RegPurityReason::NoRegisterWrites);
    }
    let mut stored: Vec<VarnodeId> = eff.stores.iter().copied().collect();
    stored.sort_unstable();
    let mut outputs =
        canonicalize_to_coarsest(ctx, &stored).ok_or(RegPurityReason::NonCanonicalRegisters)?;

    let mut read_set: Vec<VarnodeId> = eff.loads.iter().copied().collect();
    read_set.sort_unstable();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CallGraph;
    use qcode::value::FunctionBody;
    use qcode_macro::qcode;

    use super::super::summary::{EffectSummaries, solve_summaries};

    fn solve(tc: &qcode::testing::TestContext) -> EffectSummaries<RegChannel> {
        let graph = CallGraph::analyze(&tc.ctx);
        solve_summaries(&tc.ctx, &graph, &RegChannel { sp: None })
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
        assert!(eff.loads.contains(&r1));
        assert!(eff.stores.contains(&r0));
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
            eff.stores.contains(&r0) && eff.stores.contains(&r1),
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
                eff.stores.contains(&r0) && eff.stores.contains(&r1),
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
        assert!(eff.loads.contains(&r0));
        assert!(eff.stores.is_empty());
    }

    /// A prototype-less external is a leaf with empty effects (its call keeps the
    /// declared-clobber modelling in the caller's body), NOT ⊤.
    #[test]
    fn prototypeless_external_is_empty_not_top() {
        let mut tc = qcode::testing::TestContext::new();
        let ext = FunctionBody::make_external(&mut tc.ctx, 0x9000, Some("noproto".into())).id;
        let s = solve(&tc);
        let eff = s
            .get(ext)
            .as_ref()
            .expect("a prototype-less external is empty-effects, not ⊤");
        assert!(eff.loads.is_empty() && eff.stores.is_empty());
    }

    /// `finalize_register_effects` canonicalizes an overlapping write group to the
    /// single coarsest covering register.
    #[test]
    fn finalize_canonicalizes_overlap() {
        let tc = qcode::testing::TestContext::new();
        let (lo32, byte1) = (tc.r0_lo32, tc.r0_byte1);
        let mut eff = RegEffects::default();
        eff.stores.insert(lo32);
        eff.stores.insert(byte1);
        let iface = finalize_register_effects(&tc.ctx, &eff).unwrap();
        assert_eq!(
            iface.outputs,
            vec![lo32],
            "byte1 ⊂ lo32, so the group collapses to the coarsest cover"
        );
    }
}
