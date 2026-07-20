//! RAM instantiation of the [`EffectChannel`](super::summary::EffectChannel)
//! fixpoint — stages 1–2a of moving the RAM argpromote channel onto the effect
//! engine (see `argpromote-ram-effects-migration`).
//!
//! The effect element is still the *unit* lattice: a function's summary is
//! either "no **outward** memory effects" (`Ok(RamEffects)`) or ⊤. Stage 1
//! solved the legacy blocking-call gate's fact ("touches no memory at all")
//! order-independently and closed over the call graph; stage 2a refines the
//! scan to what the gate actually needs — effects a *caller* could observe.
//! Two access classes are outward-invisible and no longer poison a summary:
//!
//! - **Function-private spaces** (`space.shared()` is `None`): a promoted
//!   callee's shadow is a body-local [`TempSpace`](qcode::value::TempSpace)
//!   seeded from its own by-value snapshot params, so it can neither observe
//!   nor alias anything the caller names. This unblocks a caller in the same
//!   pipeline round its callee was promoted, instead of waiting for gvn/dce to
//!   erase the callee's redirected shadow accesses.
//! - **Own-frame locals** (classified [`FrameClass::Local`](crate::stack::frame::FrameClass)
//!   against the callee's own incoming `@SP`): a fresh frame strictly below the
//!   caller's stack pointer is disjoint from every address the caller can pass
//!   or promote, by the same stack discipline frame-freshness already relies
//!   on, and it is dead at return.
//!
//! Everything else — shared-ram accesses (param-relative derefs included),
//! caller-frame slots, register traffic, other shared spaces — stays ⊤.
//! Stage 2b grows `RamEffects` into param-relative read/write region sets with
//! a rebasing `transfer`, so those become expressible instead of ⊤.

use qcode::{
    context::Context,
    value::{FunctionBody, FunctionId, VarnodeId, insn::Mnemonic},
};

use crate::CallGraph;
use crate::calls::CallEdge;

use super::ram::OwnFrame;
use super::summary::{EffectChannel, EffectSummaries, solve_summaries};

/// Stage-2a effect element: carries no information beyond "not ⊤". A function
/// with an `Ok(RamEffects)` summary has no caller-observable memory effect —
/// transitively, through every resolved call, tail-call, and array-op body.
#[derive(Clone, Debug, Default)]
pub(crate) struct RamEffects;

/// The RAM [`EffectChannel`] (outward-effect-free or ⊤). `sp` is the
/// stack-pointer varnode used to recognise own-frame locals; without it the
/// own-frame carve-out is inert and such accesses are ⊤.
pub(crate) struct RamChannel {
    pub(crate) sp: Option<VarnodeId>,
}

impl EffectChannel for RamChannel {
    type Effects = RamEffects;

    fn scan(&self, ctx: &Context, fid: FunctionId) -> Option<RamEffects> {
        // Built on the first shared-space access only: most memory-free
        // candidates have none, and the affine numbering behind the frame
        // classifier is the expensive part.
        let mut own_frame: Option<OwnFrame> = None;
        for block in FunctionBody::from_id(ctx, fid).blocks() {
            for insn in block.iter() {
                let (space, ptr) = match insn.mnemonic() {
                    Mnemonic::Load(l) => (l.space, l.ptr),
                    Mnemonic::Store(s) => (s.space, s.ptr),
                    _ => continue,
                };
                // Function-private (shadow/temp) space: outward-invisible.
                if space.shared().is_none() {
                    continue;
                }
                // Own-frame local in a shared ram space: outward-invisible.
                // Register accesses, caller-frame slots, and every other
                // shared-space access are caller-observable — ⊤.
                let frame = own_frame.get_or_insert_with(|| OwnFrame::new(ctx, fid, self.sp));
                if !frame.is_local(ctx, ptr.qualify(insn.id.func)) {
                    return None;
                }
            }
        }
        Some(RamEffects)
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
        // No outward effects compose through any call edge unchanged; ⊤
        // callees never reach transfer (the engine poisons the caller first).
        Some(RamEffects)
    }

    fn join(&self, _into: &mut RamEffects, _from: &RamEffects) -> bool {
        false // the unit lattice never grows
    }
}

/// Solve the outward-effect-free summary for every function in `ctx`.
pub(crate) fn solve(
    ctx: &Context,
    graph: &CallGraph,
    sp: Option<VarnodeId>,
) -> EffectSummaries<RamChannel> {
    solve_summaries(ctx, graph, &RamChannel { sp })
}

/// Whether `fid`'s solved summary says it has no caller-observable memory
/// effect. `false` for ⊤ and for call targets outside the solved snapshot.
pub(crate) fn is_memory_free(summaries: &EffectSummaries<RamChannel>, fid: FunctionId) -> bool {
    summaries.try_get(fid).is_some_and(|s| s.is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode_macro::qcode;

    /// The transitive hole the legacy body-rescan gate had: `mid` is memory-free
    /// in its own body but calls a loading `leaf`, so its summary must be ⊤ —
    /// a caller handing `mid` a promoted pointer is not safe. (A register
    /// access is also a caller-observable effect: still ⊤ after stage 2a.)
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
        let s = solve(&tc.ctx, &graph, None);
        assert!(!is_memory_free(&s, leaf));
        assert!(!is_memory_free(&s, mid));
        assert!(is_memory_free(&s, pure));
    }

    /// Stage 2a: accesses in a function-private (shadow/temp) space are
    /// outward-invisible — a freshly promoted callee whose loads were
    /// redirected into its shadow no longer blocks its callers.
    #[test]
    fn private_space_access_is_outward_invisible() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn shadowed:
                <entry @p:i64>
                    return at i64 0;
            "
        );
        let _ = entry;
        // Redirect-style access: store the param into a body-local temp space.
        let word = 1;
        let addr_size = 8;
        let shadow = tc.ctx.bodies[shadowed].push_temp_space(qcode::value::TempSpace::new(
            Some("test_shadow"),
            word,
            addr_size,
        ));
        let shadow = qcode::space::LocalMemorySpaceId::Temp(shadow.local);
        let root = qcode::value::FunctionBody::from_id(&tc.ctx, shadowed)
            .root()
            .unwrap()
            .id;
        let p = qcode::value::BasicBlock::from_id(&tc.ctx, root)
            .params()
            .next()
            .unwrap()
            .id();
        let mut b = (&mut tc.ctx).builder(root);
        b.set_insert_point_to_start();
        let a = b.shr().get_const(0x10, addr_size);
        b.push_store(p, a, shadow);

        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, None);
        assert!(
            is_memory_free(&s, shadowed),
            "a private-space store must not poison the summary"
        );
    }
}
