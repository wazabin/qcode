//! Channel-generic interprocedural effect-summary fixpoint (see
//! `ARGPROMOTE_REGISTERS_V2.md`).
//!
//! Every argpromote channel needs the same whole-program fact before it can
//! rewrite any interface: the *call-graph-closed* effect set of each function —
//! its own syntactic effects joined with everything its callees (and theirs)
//! do. v1 derived interfaces from a per-body scan at rewrite time, which is
//! order-sensitive: promoting a callee injects effect instructions into its
//! callers, stranding loads and dropping output writes in callers promoted
//! earlier. This module separates the analysis out: a monotone worklist
//! fixpoint over the [`CallGraph`], solved *before* any mutation, so the
//! rewrite sweep is order-independent by construction.
//!
//! The engine is parameterized by an [`EffectChannel`] — the per-channel
//! effect lattice and its transfer through a call site. Registers instantiate
//! it with global register sets and an identity transfer
//! ([`super::reg_summary`]); the RAM channel can later instantiate it with
//! param-relative region sets rebased through call arguments.
//!
//! ⊤ ("reg-impure" / not promotable) is represented as `Err(TopCause)` and is
//! absorbing: any ⊤ callee makes its callers ⊤. Termination: each channel's
//! effect type must have finite join height (register subsets trivially;
//! region sets via a size budget that overflows to ⊤ in the channel's
//! `transfer`/`join`).

use qcode::{
    context::Context,
    value::{FunctionBody, FunctionId},
};
use rustc_hash::FxHashMap;

use crate::{
    CallGraph,
    calls::{CallEdge, CallTarget},
};

/// One argpromote channel's effect domain: the lattice element, the syntactic
/// seed scan, external leaves, and the per-call-site transfer.
pub(crate) trait EffectChannel {
    /// The channel's effect set. Joins must be monotone with finite height
    /// (grow-only sets, or budgeted sets that overflow to ⊤ via `None`s below).
    type Effects: Clone;

    /// Syntactic per-body seed scan — this function's *own* effects only; the
    /// engine composes callee effects. `None` for a channel-specific ⊤ gate.
    fn scan(&self, ctx: &Context, fid: FunctionId) -> Option<Self::Effects>;

    /// Leaf summary for an external (bodyless) function, from its materialized
    /// interface. `None` when the interface is missing or unusable — the
    /// external is ⊤ and taints its callers. A channel whose effects remain
    /// soundly modelled by the call instruction itself (the register channel:
    /// external calls keep their declared-clobber semantics in the promoted
    /// body) may return empty effects instead of `None`.
    fn external_leaf(&self, ctx: &Context, fid: FunctionId) -> Option<Self::Effects>;

    /// The effects to assume for a function containing an *unresolved indirect
    /// call*. `None` (the default) is ⊤: the unknown callee poisons the
    /// function. A channel that can model the unknown callee may return effects
    /// to join into the seed scan instead — the register channel binds every
    /// `CallInd` against the platform-ABI clobber leaf and returns that leaf's
    /// reads/writes, which is exactly the traffic it materializes at the site.
    fn indirect_call_effects(&self, _ctx: &Context, _fid: FunctionId) -> Option<Self::Effects> {
        None
    }

    /// Map a callee's summary into the caller's frame at one call edge.
    /// Registers: identity (the register file is a global namespace). RAM:
    /// rebase param-relative regions through the call's actual arguments,
    /// `None` when a region isn't expressible in the caller (⊤).
    fn transfer(
        &self,
        ctx: &Context,
        edge: &CallEdge,
        callee: &Self::Effects,
    ) -> Option<Self::Effects>;

    /// Join `from` into `into`; report whether `into` grew. Must be monotone.
    fn join(&self, into: &mut Self::Effects, from: &Self::Effects) -> bool;
}

/// Why a function's summary is ⊤ (not promotable by the channel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TopCause {
    /// External without a usable materialized interface.
    External,
    /// Lifted but has no root block (empty body).
    NoBody,
    /// Contains an unresolved indirect call (`CallInd`).
    IndirectCall,
    /// The channel's own scan/transfer bailed (e.g. unrebasable RAM region).
    Channel,
    /// A direct callee is ⊤ — the first one found is recorded.
    ImpureCallee(FunctionId),
}

/// A solved per-function summary: the channel effects, or ⊤ with its cause.
pub(crate) type Summary<E> = Result<E, TopCause>;

/// Solved summaries for every function in `ctx`, keyed by function ID.
pub(crate) struct EffectSummaries<C: EffectChannel> {
    map: FxHashMap<FunctionId, Summary<C::Effects>>,
}

impl<C: EffectChannel> EffectSummaries<C> {
    /// The solved summary for `fid`. Every function present in the context at
    /// solve time has one.
    pub(crate) fn get(&self, fid: FunctionId) -> &Summary<C::Effects> {
        &self.map[&fid]
    }

    /// The solved summary for `fid`, or `None` for a function outside the
    /// snapshot the summaries were solved on (e.g. a call target discovered
    /// by a mutation after the solve).
    pub(crate) fn try_get(&self, fid: FunctionId) -> Option<&Summary<C::Effects>> {
        self.map.get(&fid)
    }
}

/// Solve the channel's effect summaries for the whole program to a fixpoint.
///
/// Seeds every function with its own effects ([`EffectChannel::scan`], or
/// [`EffectChannel::external_leaf`] for externals), applies the shared ⊤
/// gates (no body, contains an indirect call), then joins
/// callee summaries into callers through [`EffectChannel::transfer`] until
/// nothing grows. Cycles (recursion) need no special casing: the lattice is
/// finite-height and the transfer is monotone, so the worklist terminates.
pub(crate) fn solve_summaries<C: EffectChannel>(
    ctx: &Context,
    graph: &CallGraph,
    chan: &C,
) -> EffectSummaries<C> {
    solve_summaries_with_fixed(ctx, graph, chan, &FxHashMap::default())
}

/// Solve summaries while treating selected functions as fixed interface leaves.
///
/// The incremental driver resets the invalidated cone to bottom and supplies
/// every out-of-cone function here from its persisted
/// [`FunctionEffects`](qcode::value::FunctionEffects).
/// Fixed functions are never scanned and their callees are not joined into them;
/// callers inside the cone consume the supplied summary exactly as they would an
/// external leaf.
pub(crate) fn solve_summaries_with_fixed<C: EffectChannel>(
    ctx: &Context,
    graph: &CallGraph,
    chan: &C,
    fixed: &FxHashMap<FunctionId, Summary<C::Effects>>,
) -> EffectSummaries<C> {
    let fids: Vec<FunctionId> = ctx.function_ids();

    let mut map: FxHashMap<FunctionId, Summary<C::Effects>> = FxHashMap::default();
    for &fid in &fids {
        if let Some(summary) = fixed.get(&fid) {
            map.insert(fid, summary.clone());
            continue;
        }
        let f = FunctionBody::from_id(ctx, fid);
        let seed = if f.is_external() {
            chan.external_leaf(ctx, fid).ok_or(TopCause::External)
        } else if f.root().is_none() {
            Err(TopCause::NoBody)
        } else if graph.has_indirect_call(fid) {
            match (chan.indirect_call_effects(ctx, fid), chan.scan(ctx, fid)) {
                (Some(ind), Some(mut own)) => {
                    chan.join(&mut own, &ind);
                    Ok(own)
                }
                (None, _) => Err(TopCause::IndirectCall),
                (_, None) => Err(TopCause::Channel),
            }
        } else {
            chan.scan(ctx, fid).ok_or(TopCause::Channel)
        };
        map.insert(fid, seed);
    }

    // Monotone worklist: recompute a caller from its callees; requeue its own
    // callers whenever it grows. `in_list` dedupes queue membership.
    let mut worklist: Vec<FunctionId> = fids
        .iter()
        .copied()
        .filter(|fid| !fixed.contains_key(fid))
        .collect();
    let mut in_list: rustc_hash::FxHashSet<FunctionId> = worklist.iter().copied().collect();
    while let Some(fid) = worklist.pop() {
        in_list.remove(&fid);
        if map[&fid].is_err() {
            continue; // ⊤ is absorbing; callers were poisoned when it was set
        }
        let mut grew = false;
        let mut top: Option<TopCause> = None;
        for &eid in graph.outgoing_edges(fid) {
            let edge = *graph.edge(eid);
            let CallTarget::Function(callee) = edge.target else {
                // Unresolved indirect edges are already gated by the seed.
                continue;
            };
            match map.get(&callee) {
                Some(Ok(callee_eff)) => {
                    let Some(moved) = chan.transfer(ctx, &edge, callee_eff) else {
                        top = Some(TopCause::Channel);
                        break;
                    };
                    let Ok(cur) = map.get_mut(&fid).expect("seeded above") else {
                        unreachable!("checked Ok before the loop");
                    };
                    grew |= chan.join(cur, &moved);
                }
                Some(Err(_)) | None => {
                    // ⊤ callee (or a callee outside the context snapshot).
                    top = Some(TopCause::ImpureCallee(callee));
                    break;
                }
            }
        }
        if let Some(cause) = top {
            map.insert(fid, Err(cause));
            grew = true;
        }
        if grew {
            for caller in graph.callers(fid) {
                if !fixed.contains_key(&caller) && in_list.insert(caller) {
                    worklist.push(caller);
                }
            }
        }
    }

    EffectSummaries { map }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode_macro::qcode;
    use rustc_hash::FxHashSet;

    /// Toy channel pinning engine behavior independently of any real channel:
    /// effects = the set of function IDs whose body was scanned (identity
    /// transfer, set-union join), so a solved summary is exactly "which
    /// functions' effects reach me".
    struct ToyChannel {
        /// Functions the channel itself refuses (drives the `Channel` cause).
        reject: FxHashSet<FunctionId>,
    }

    impl EffectChannel for ToyChannel {
        type Effects = FxHashSet<FunctionId>;

        fn scan(&self, _ctx: &Context, fid: FunctionId) -> Option<Self::Effects> {
            (!self.reject.contains(&fid)).then(|| FxHashSet::from_iter([fid]))
        }

        fn external_leaf(&self, _ctx: &Context, fid: FunctionId) -> Option<Self::Effects> {
            (!self.reject.contains(&fid)).then(|| FxHashSet::from_iter([fid]))
        }

        fn transfer(
            &self,
            _ctx: &Context,
            _edge: &CallEdge,
            callee: &Self::Effects,
        ) -> Option<Self::Effects> {
            Some(callee.clone())
        }

        fn join(&self, into: &mut Self::Effects, from: &Self::Effects) -> bool {
            let before = into.len();
            into.extend(from.iter().copied());
            into.len() != before
        }
    }

    fn solve(
        ctx: &Context,
        reject: impl IntoIterator<Item = FunctionId>,
    ) -> EffectSummaries<ToyChannel> {
        let graph = CallGraph::analyze(ctx);
        let chan = ToyChannel {
            reject: reject.into_iter().collect(),
        };
        solve_summaries(ctx, &graph, &chan)
    }

    #[test]
    fn leaf_and_caller_union() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn callee:
                <callee_entry>
                    return at i64 0;
            fn caller:
                <caller_entry>
                    call fn callee();
                <caller_cont>
                    return at i64 0;
            "
        );
        let _ = (callee_entry, caller_entry, caller_cont);
        let s = solve(&tc.ctx, []);
        assert_eq!(s.get(callee).as_ref().unwrap().len(), 1);
        let caller_eff = s.get(caller).as_ref().unwrap();
        assert!(caller_eff.contains(&callee) && caller_eff.contains(&caller));
    }

    #[test]
    fn recursion_reaches_fixpoint() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn a:
                <a_entry>
                    call fn b();
                <a_cont>
                    return at i64 0;
            fn b:
                <b_entry>
                    call fn a();
                <b_cont>
                    return at i64 0;
            "
        );
        let _ = (a_entry, a_cont, b_entry, b_cont);
        let s = solve(&tc.ctx, []);
        for fid in [a, b] {
            let eff = s.get(fid).as_ref().unwrap();
            assert!(eff.contains(&a) && eff.contains(&b), "2-cycle unions both");
        }
    }

    #[test]
    fn self_recursion() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry>
                    call fn f();
                <f_cont>
                    return at i64 0;
            "
        );
        let _ = (f_entry, f_cont);
        let s = solve(&tc.ctx, []);
        assert_eq!(s.get(f).as_ref().unwrap().len(), 1);
    }

    #[test]
    fn channel_reject_poisons_callers_transitively() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn bad:
                <bad_entry>
                    return at i64 0;
            fn mid:
                <mid_entry>
                    call fn bad();
                <mid_cont>
                    return at i64 0;
            fn top_fn:
                <top_entry>
                    call fn mid();
                <top_cont>
                    return at i64 0;
            "
        );
        let _ = (bad_entry, mid_entry, mid_cont, top_entry, top_cont);
        let s = solve(&tc.ctx, [bad]);
        assert_eq!(*s.get(bad), Err(TopCause::Channel));
        assert_eq!(*s.get(mid), Err(TopCause::ImpureCallee(bad)));
        assert_eq!(*s.get(top_fn), Err(TopCause::ImpureCallee(mid)));
    }

    #[test]
    fn address_taken_is_not_top() {
        // Per the call-tag model (ARGPROMOTE_REGISTERS_V2.md) address-taken alone
        // is NOT a ⊤ cause in the engine: the function keeps a precise summary so
        // its effects compose into direct callers, and only the *indirect* sites
        // stay implicit (the promoter, not the engine, declines to rewrite it).
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn taken:
                <taken_entry>
                    return at i64 0;
            fn user:
                <user_entry>
                    return at i64 0;
            "
        );
        let _ = taken_entry;
        // Take `taken`'s address: store the function value somewhere in `user`.
        let addr = tc.ctx.get_const(0x9000, 8).id();
        {
            let mut b = tc.ctx.builder(user_entry);
            b.set_insert_point_to_start();
            b.push_store(qcode::value::ValueId::Function(taken), addr, tc.reg_space);
        }
        let s = solve(&tc.ctx, []);
        assert!(
            s.get(taken).is_ok(),
            "address-taken function keeps a solved summary"
        );
        let _ = user;
    }
}
