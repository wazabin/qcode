//! RAM instantiation of the [`EffectChannel`](super::summary::EffectChannel)
//! fixpoint — stage 1 of moving the RAM argpromote channel onto the effect
//! engine (see `argpromote-ram-effects-migration`).
//!
//! At this stage the effect element is the *unit* lattice: a function's summary
//! is either "touches no memory at all" (`Ok(RamEffects)`) or ⊤. That is exactly
//! the fact the legacy blocking-call gate derived by rescanning callee bodies at
//! rewrite time ([`super::ram::function_makes_blocking_call`] before this
//! module): a call composes with the shadow promotion only if the callee can
//! neither dereference a promoted pointer nor alias the shadow — i.e. it is
//! (transitively) memory-free. Solving it as an engine fixpoint *before* any
//! mutation makes the answer independent of the sweep's visit order, and —
//! unlike the legacy per-body scan — closes it over the call graph: a
//! memory-free callee that itself calls a memory-touching function is now
//! correctly ⊤ (the legacy gate inspected only the direct callee's body, a
//! transitive hole).
//!
//! Stage 2 grows `RamEffects` into param-relative read/write region sets with a
//! rebasing `transfer`, so a caller can promote while keeping a call to a
//! callee that dereferences the promoted pointer.

use qcode::{
    context::Context,
    value::{FunctionBody, FunctionId, insn::Mnemonic},
};

use crate::CallGraph;
use crate::calls::CallEdge;

use super::summary::{EffectChannel, EffectSummaries, solve_summaries};

/// Stage-1 effect element: carries no information beyond "not ⊤". A function
/// with an `Ok(RamEffects)` summary touches no memory in any space —
/// transitively, through every resolved call, tail-call, and array-op body.
#[derive(Clone, Debug, Default)]
pub(crate) struct RamEffects;

/// The RAM [`EffectChannel`] (stage 1: memory-free or ⊤).
pub(crate) struct RamChannel;

impl EffectChannel for RamChannel {
    type Effects = RamEffects;

    fn scan(&self, ctx: &Context, fid: FunctionId) -> Option<RamEffects> {
        // Any load or store — in *any* space, register file included — is ⊤.
        // The consumers of a memory-free verdict (the blocking-call gate) need
        // the callee inert toward every address the caller can name, and a
        // register access marks a function `argpromote_registers` has not (or
        // cannot) functionalize, which the legacy gate rejected identically.
        let memory_free = FunctionBody::from_id(ctx, fid).blocks().all(|b| {
            b.iter()
                .all(|i| !matches!(i.mnemonic(), Mnemonic::Load(_) | Mnemonic::Store(_)))
        });
        memory_free.then_some(RamEffects)
    }

    fn external_leaf(&self, _ctx: &Context, _fid: FunctionId) -> Option<RamEffects> {
        // Parity with the legacy gate: a resolved bodyless external was treated
        // as memory-free (`function_accesses_memory` over an empty body), and
        // in practice every external call site carries ABI clobbers, which the
        // caller-side clobber check rejects on its own. Stage 3 (persisted
        // memory effects) revisits this with the materialized interface.
        Some(RamEffects)
    }

    fn transfer(
        &self,
        _ctx: &Context,
        _edge: &CallEdge,
        _callee: &RamEffects,
    ) -> Option<RamEffects> {
        // Memory-free composes through any call edge unchanged; ⊤ callees never
        // reach transfer (the engine poisons the caller first).
        Some(RamEffects)
    }

    fn join(&self, _into: &mut RamEffects, _from: &RamEffects) -> bool {
        false // the unit lattice never grows
    }
}

/// Solve the memory-free summary for every function in `ctx`.
pub(crate) fn solve(ctx: &Context, graph: &CallGraph) -> EffectSummaries<RamChannel> {
    solve_summaries(ctx, graph, &RamChannel)
}

/// Whether `fid`'s solved summary says it is transitively memory-free.
/// `false` for ⊤ and for call targets outside the solved snapshot.
pub(crate) fn is_memory_free(summaries: &EffectSummaries<RamChannel>, fid: FunctionId) -> bool {
    summaries.try_get(fid).is_some_and(|s| s.is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode_macro::qcode;

    /// The transitive hole the legacy body-rescan gate had: `mid` is memory-free
    /// in its own body but calls a loading `leaf`, so its summary must be ⊤ —
    /// a caller handing `mid` a promoted pointer is not safe.
    #[test]
    fn memory_free_is_transitive() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn leaf:
                <leaf_entry>
                    %v = load(register:8, {r0});
                    return at i64 0;
            fn mid:
                <mid_entry>
                    call fn leaf();
                <mid_cont>
                    return at i64 0;
            fn pure:
                <pure_entry>
                    return at i64 0;
            "
        );
        let _ = (leaf_entry, mid_entry, mid_cont, pure_entry);
        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph);
        assert!(!is_memory_free(&s, leaf));
        assert!(!is_memory_free(&s, mid));
        assert!(is_memory_free(&s, pure));
    }
}
