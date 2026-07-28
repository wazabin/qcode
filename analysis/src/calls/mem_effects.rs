//! Interprocedural memory-write summaries: which memory spaces each function may
//! store to, recorded on [`written_spaces`](qcode::value::FunctionRef::written_spaces).
//!
//! The consumer is store-to-load forwarding's call-prune
//! ([`crate::gvn::mem_forward`]): a call to a callee whose witnessed write-set
//! does **not** include a cell's space cannot clobber that cell, so the forwarded
//! value survives the call.
//!
//! Stage 3 of the RAM effects migration retired the standalone `SpaceChannel`:
//! the coarse space set is now the `written` component of the unified
//! [`RamChannel`](super::argpromote) summary — one engine solve feeds both the
//! argpromote blocking gate (`precise`) and this record (`written`). A function
//! whose effect cannot be bounded — an unresolved `BranchInd`, an external, an
//! indirect call — is recorded `None` (may write any space), the conservative
//! answer the prune already assumes.

use qcode::{
    context::Context,
    space::SpaceId,
    value::{FunctionBody, FunctionId},
};

/// Compute and record [`written_spaces`](qcode::value::FunctionRef::written_spaces)
/// for every function — externals included: the `written` component of the
/// unified [`RamChannel`](super::argpromote) summary (`ram_summary_solve`). A ⊤
/// summary (unbounded) records `None`, the conservative value the prune already
/// assumes, so an under-approximation is impossible.
///
/// Externals are stamped uniformly with the rest: a prototyped external's argmem
/// derivation bounds its `written` (e.g. `memset` writes only the default ram
/// space), an un-prototyped one records `Unbounded`. Stamping them keeps the
/// `written_spaces` verify rule consistent — it reads the same tri-state for a
/// callee whether internal or external, so a caller bounded over a prototyped
/// external no longer trips the "external is unbounded" arm when the external's
/// derived write-set nests inside the caller's.
///
/// Test-only: the pipeline stamps through the cone-checked [`SeedWrittenSpaces`]
/// pass, which calls [`written_space_updates`] directly. This whole-`Context`
/// form bypasses the cone, so it is not offered to production code — it exists
/// for tests that hold the whole context and want a one-call stamp.
#[cfg(test)]
pub(crate) fn set_all_written_spaces(ctx: &mut Context) {
    let targets = ctx.function_ids();
    set_written_spaces_targeted_with_sp(ctx, &targets, None);
}

/// Solve the whole-program RAM summary and compute the memory-channel stamp for
/// each `target` whose value would change. Read-only: the whole-program solve and
/// the change comparison both run against `&Context`. The caller stamps the
/// returned updates (through the cone-checked interface setter on the module path,
/// or `from_id_mut` on the whole-`Context` path).
fn written_space_updates(
    ctx: &Context,
    targets: impl IntoIterator<Item = FunctionId>,
    sp: Option<qcode::value::VarnodeId>,
) -> Vec<(
    FunctionId,
    qcode::value::WrittenSpacesState,
    Option<qcode::value::Footprint>,
)> {
    let graph = crate::CallGraph::analyze(ctx);
    let summaries = super::argpromote::ram_summary_solve(ctx, &graph, sp);

    let mut updates = Vec::new();
    for id in targets {
        let (summary, precise) = match summaries.get(id) {
            // ⊤: neither component is expressible.
            Err(_) => (None, None),
            Ok(eff) => (
                eff.written
                    .as_ref()
                    .map(|set| set.iter().copied().collect::<Vec<SpaceId>>()),
                eff.precise.clone(),
            ),
        };
        // A ⊤ summary now records *stamped unbounded* rather than clearing the
        // stamp, so a fresh mint transitioning Unstamped → Unbounded counts as a
        // change (its callers' bounds may need re-verifying). Change-detection is
        // a direct equality on each solved component (Unstamped ≠
        // Bounded/Unbounded, Unbounded == Unbounded, Bounded(a) == Bounded(b) iff
        // a == b) — the same verdict the hand-rolled `WrittenSpaces` match
        // produced.
        //
        // The precise footprint the same solve derived is stamped alongside the
        // coarse set, so a memory-channel effect delta can compare addresses and
        // not just space granularity.
        //
        // Only these two *solved* components are this pass's to write. The
        // channel's materialized interface belongs to the RAM channel's rewrite,
        // not to any lattice this solve computes, so it is neither recomputed nor
        // compared, and the stamp goes through the field-wise
        // `set_memory_solved` rather than replacing the whole channel.
        let coarse = match &summary {
            Some(b) => qcode::value::WrittenSpacesState::Bounded(b.clone()),
            None => qcode::value::WrittenSpacesState::Unbounded,
        };
        let current = &FunctionBody::from_id(ctx, id).effects().memory;
        if current.coarse != coarse || current.precise != precise {
            updates.push((id, coarse, precise));
        }
    }
    updates
}

#[cfg(test)]
fn set_written_spaces_targeted_with_sp(
    ctx: &mut Context,
    targets: &[FunctionId],
    sp: Option<qcode::value::VarnodeId>,
) -> rustc_hash::FxHashSet<FunctionId> {
    let updates = written_space_updates(ctx, targets.iter().copied(), sp);
    let changed_functions: rustc_hash::FxHashSet<FunctionId> =
        updates.iter().map(|(id, ..)| *id).collect();
    for (id, coarse, precise) in updates {
        FunctionBody::from_id_mut(ctx, id).set_memory_solved(coarse, precise);
    }
    changed_functions
}

// ----- pass ------------------------------------------------------------------

use crate::{Pass, PipelineEnv};

#[derive(Default)]
pub struct SeedWrittenSpaces;

impl Pass for SeedWrittenSpaces {
    const NAME: &'static str = "seed_written_spaces";
    fn description(&self) -> &'static str {
        "Infer each function's transitive memory-write space set (for the GVN forwarding call-prune)"
    }
    fn run(
        &self,
        cone: &mut crate::ConeMut,
        env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        // Whole-program solve reads through `&Context`; stamping iterates the cone
        // and writes through the cone-checked interface setter.
        let updates = written_space_updates(cone.ctx(), cone.cone_functions(), env.sp_varnode);
        let changed: rustc_hash::FxHashSet<FunctionId> =
            updates.iter().map(|(id, ..)| *id).collect();
        for (id, coarse, precise) in updates {
            cone.function_mut(id).set_memory_solved(coarse, precise);
        }
        Ok(crate::ModulePassOutcome::functions(changed)
            .preserving_global::<crate::CallGraphAnalysis>()
            .preserving_global::<crate::AddressAnalysis>())
    }
}

crate::register_module_pass!(SeedWrittenSpaces);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AliasResult, constant_fold_function, gvn_function};
    use qcode::{
        context::Context,
        lower::lower_str,
        testing::TestContext,
        value::{BasicBlock, FunctionBody, TempSpace, ValueId},
    };

    /// The motivating shape, reduced: a buffer pointer is spilled into a frame
    /// slot, the block ends in a `call` to a callee that writes only its own
    /// scratch space, the pointer is reloaded after the call and the buffer is
    /// filled with constants, with an unrelated frame-slot store in between, then
    /// read back by a wide load. The wide load can only concretize if the spilled
    /// reload forwards across the call — which requires the spill cell to survive
    /// the call-prune. It does precisely because the callee's witnessed write-set
    /// excludes the buffer's space.
    const SPILL_CALL_RELOAD: &str = "\
fn caller:
<entry @sp:i32>
    i32 %bufptr = i32 @sp - i32 0x1a0;
    i32 %slot = i32 @sp - i32 0x78;
    store(ram:4, i32 %slot <- i32 %bufptr);
    i32 %argp = i32 @sp - i32 0xc04;
    call fn callee(@cp=i32 %argp) // -> <resume>;
<resume>
    i32 %buf = load(ram:4, i32 %slot);
    store(ram:4, i32 %buf <- i32 0x97079551);
    i32 %a1 = i32 %buf + i32 0x4;
    store(ram:4, i32 %a1 <- i32 0x9306913e);
    i32 %a2 = i32 %buf + i32 0x8;
    store(ram:4, i32 %a2 <- i32 0x9f309d3b);
    i32 %fslot = i32 @sp - i32 0xc00;
    store(ram:4, i32 %fslot <- i32 0xdeadbeef);
    i96 %v = load(ram:12, i32 %buf);
    return i96 %v;
fn callee:
<centry @cp:i32>
    store(scratch:4, i32 @cp <- i32 0x0);
    i32 %x = i32 @cp + i32 0x4;
    store(scratch:4, i32 %x <- i32 0x1);
    return at i32 @cp;
";

    /// The same shape but with `<header>` a *loop* header (back-edge): the buffer
    /// fill now runs through the not-yet-forwarded reload inside the loop body, so
    /// the loop-carried prune would drop the spill cell — the chicken-and-egg —
    /// unless it peels the reload (gated on `LoadedPointerDisjointFromSlot`).
    const LOOP_SPILL_RELOAD: &str = "\
fn caller:
<entry @sp:i32>
    i32 %bufptr = i32 @sp - i32 0x1a0;
    i32 %slot = i32 @sp - i32 0x78;
    store(ram:4, i32 %slot <- i32 %bufptr);
    i32 %argp = i32 @sp - i32 0xc04;
    call fn callee(@cp=i32 %argp) // -> <header>;
<header @i:i32>
    i32 %buf = load(ram:4, i32 %slot);
    store(ram:4, i32 %buf <- i32 0x97079551);
    i32 %a1 = i32 %buf + i32 0x4;
    store(ram:4, i32 %a1 <- i32 0x9306913e);
    i32 %a2 = i32 %buf + i32 0x8;
    store(ram:4, i32 %a2 <- i32 0x9f309d3b);
    i32 %fslot = i32 @sp - i32 0xc00;
    store(ram:4, i32 %fslot <- i32 0xdeadbeef);
    i96 %v = load(ram:12, i32 %buf);
    i32 %n = i32 @i + i32 0x1;
    i8 %more = i32 %n < i32 0x10;
    if i8 %more goto <header @i=i32 %n> else goto <exit>;
<exit>
    return i96 %v;
fn callee:
<centry @cp:i32>
    store(scratch:4, i32 @cp <- i32 0x0);
    return at i32 @cp;
";

    /// Run fold+gvn to a small fixpoint and return the rendered `caller`.
    fn optimize_caller(seed_summaries: bool) -> String {
        let mut ctx = Context::new();
        let syms = lower_str(&mut ctx, SPILL_CALL_RELOAD).expect("parse");
        let caller = syms.functions["caller"];
        if seed_summaries {
            set_all_written_spaces(&mut ctx);
        }
        for _ in 0..2 {
            constant_fold_function(&mut ctx, caller);
            let aliases = AliasResult::simple_for_function(&ctx, caller);
            gvn_function(&mut ctx, caller, Some(&aliases));
        }
        format!("{}", FunctionBody::from_id(&ctx, caller))
    }

    /// Lower the loop shape, seed write-spaces, optionally record
    /// `LoadedPointerDisjointFromSlot`, run fold+gvn, return the rendered `caller`.
    fn optimize_loop_caller(assume_disjoint: bool) -> String {
        use qcode::assumption::Proposition;
        let mut ctx = Context::new();
        let syms = lower_str(&mut ctx, LOOP_SPILL_RELOAD).expect("parse");
        let caller = syms.functions["caller"];
        set_all_written_spaces(&mut ctx);
        if assume_disjoint {
            ctx.assume_true(Proposition::LoadedPointerDisjointFromSlot(caller));
        }
        for _ in 0..3 {
            constant_fold_function(&mut ctx, caller);
            let aliases = AliasResult::simple_for_function(&ctx, caller);
            gvn_function(&mut ctx, caller, Some(&aliases));
        }
        format!("{}", FunctionBody::from_id(&ctx, caller))
    }

    /// A function that ends in an unresolved indirect tail-branch (`goto [p]` —
    /// the shape a PLT stub lifts to) escapes to unknown code, so its write-set is
    /// unbounded (`None`), not the empty set. Otherwise a caller would forward
    /// constant stores across the call, folding away input-dependent values.
    #[test]
    fn indirect_tailbranch_is_unbounded() {
        use qcode_macro::qcode;
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn stub:
            <entry>
                i64 %p = load(ram:8, i64 0x3fa0);
                goto [i64 %p];
            "
        );
        let _ = entry;
        set_all_written_spaces(&mut ctx);
        assert!(
            FunctionBody::from_id(&ctx, stub).written_spaces().is_none(),
            "an unresolved indirect tail-branch must have an unbounded write-set"
        );
    }

    /// The witnessed write-set of the functionalized callee is exactly its scratch
    /// space — never the default `ram` the buffer lives in.
    #[test]
    fn callee_written_spaces_excludes_ram() {
        let mut ctx = Context::new();
        let syms = lower_str(&mut ctx, SPILL_CALL_RELOAD).expect("parse");
        set_all_written_spaces(&mut ctx);
        let callee = FunctionBody::from_id(&ctx, syms.functions["callee"]);
        let spaces = callee
            .written_spaces()
            .expect("callee write-set is bounded");
        assert!(
            !spaces.contains(&ctx.shared.default_space),
            "callee writes only scratch, never the default ram space"
        );
    }

    #[test]
    fn body_local_scratch_is_not_published() {
        let mut tc = TestContext::new();
        let fid = FunctionBody::make(&mut tc.ctx, "local_scratch".into())
            .unwrap()
            .id;
        let block = BasicBlock::make(&mut tc.ctx, fid).id;
        FunctionBody::from_id_mut(&mut tc.ctx, fid)
            .set_root(block)
            .unwrap();
        let scratch = tc.ctx.bodies[fid].push_temp_space(TempSpace::new(Some("scratch"), 1, 8));
        let ptr = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, block).push_param(8).id);
        {
            let mut b = tc.ctx.builder(block);
            let value = b.shr().get_const(1, 1);
            b.push_store(
                value,
                ptr,
                qcode::space::LocalMemorySpaceId::Temp(scratch.local),
            );
            b.push_return(value);
        }

        set_all_written_spaces(&mut tc.ctx);
        assert_eq!(
            FunctionBody::from_id(&tc.ctx, fid).written_spaces(),
            Some(&[][..]),
            "body-local scratch must not escape into a shared-space summary"
        );
    }

    /// With the write-set summary the spilled reload forwards across the call and
    /// the wide load coalesces to a constant byte blob.
    #[test]
    fn wide_load_concretizes_with_write_summary() {
        let out = optimize_caller(true);
        assert!(
            out.contains("b\"\\x51\\x95\\x07\\x97"),
            "wide load should concretize to a byte blob, got:\n{out}"
        );
    }

    /// Without it the call-prune conservatively drops the buffer cells, so the
    /// wide load stays an opaque memory read — the pre-fix behaviour.
    #[test]
    fn wide_load_stays_opaque_without_write_summary() {
        let out = optimize_caller(false);
        assert!(
            out.contains("load(ram:12"),
            "without the write summary the wide load must remain opaque, got:\n{out}"
        );
    }

    /// In the loop shape, under `LoadedPointerDisjointFromSlot` the loop-carried
    /// prune peels the buffer-fill stores' reload, so the spill cell survives, the
    /// reload forwards, and the wide load concretizes — breaking the chicken-and-egg.
    #[test]
    fn loop_wide_load_concretizes_under_assumption() {
        let out = optimize_loop_caller(true);
        assert!(
            out.contains("b\"\\x51\\x95\\x07\\x97"),
            "under the assumption the loop-carried wide load should concretize, got:\n{out}"
        );
    }

    /// Without the assumption the reload cannot be soundly peeled (a self-referential
    /// slot could be overwritten through it), so the prune stays conservative and the
    /// wide load remains opaque.
    #[test]
    fn loop_wide_load_stays_opaque_without_assumption() {
        let out = optimize_loop_caller(false);
        assert!(
            out.contains("load(ram:12"),
            "without the assumption the loop-carried wide load must remain opaque, got:\n{out}"
        );
    }

    /// Non-vacuity of the cone-gated write handle: running `seed_written_spaces`
    /// under a partial `Cone::Set` stamps the in-cone function's memory channel
    /// but leaves the out-of-cone function's channel exactly at its unstamped
    /// default — the cone restricts the stamping loop, not just the iteration.
    #[test]
    fn partial_cone_stamps_only_in_cone_functions() {
        use qcode_macro::qcode;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn writer_a:
            <a @p:i64>
                store(ram:8, i64 @p <- i64 0x1);
                return at i64 0;
            fn writer_b:
            <b @p:i64>
                store(ram:8, i64 @p <- i64 0x1);
                return at i64 0;
            "
        );

        let default_mem = qcode::value::MemoryChannelState::default();
        // Both functions start unstamped.
        assert_eq!(
            FunctionBody::from_id(&ctx, writer_a).effects().memory,
            default_mem
        );
        assert_eq!(
            FunctionBody::from_id(&ctx, writer_b).effects().memory,
            default_mem
        );

        // Run under a cone that excludes `writer_b`.
        let env = crate::PipelineEnv::headless(&ctx);
        let set: rustc_hash::FxHashSet<FunctionId> = [writer_a].into_iter().collect();
        {
            let mut cone = crate::ConeMut::new(&mut ctx, crate::Cone::Set(set));
            crate::Pass::run(&SeedWrittenSpaces, &mut cone, &env).unwrap();
        }

        // `writer_a` (in cone) was stamped; `writer_b` (out of cone) is untouched.
        assert_ne!(
            FunctionBody::from_id(&ctx, writer_a).effects().memory,
            default_mem,
            "in-cone function must be stamped"
        );
        assert_eq!(
            FunctionBody::from_id(&ctx, writer_b).effects().memory,
            default_mem,
            "out-of-cone function's memory effects must be untouched"
        );
    }

    /// The memory channel has two independent writers: this solve owns
    /// `coarse`/`precise`, the RAM channel's rewrite owns the materialized
    /// interface. Re-running the solve must not disturb the interface.
    ///
    /// Regression: `written_space_updates` used to build a whole
    /// `MemoryChannelState` and replace the stored one, so a single re-stamp
    /// silently dropped the interface — a lost update, not an idempotence or
    /// duplicate-pass failure (it converged, on the wrong value).
    #[test]
    fn restamping_written_spaces_preserves_the_materialized_memory_interface() {
        let mut ctx = Context::new();
        let fid = FunctionBody::make(&mut ctx, "keeps_memory_interface".into())
            .unwrap()
            .id;
        BasicBlock::make(&mut ctx, fid);
        let map = qcode::value::MemoryInterfaceMap {
            inputs: vec![qcode::value::InterfaceSlot {
                base: qcode::value::SlotBase::Global(0x1000),
                offset: 0,
                size: 4,
            }],
            outputs: vec![],
        };
        FunctionBody::from_id_mut(&mut ctx, fid).set_memory_interface(Some(map.clone()));

        // Two solves: the first stamps the coarse/precise components, the second
        // finds them unchanged. Neither may touch the interface.
        crate::calls::set_all_written_spaces(&mut ctx);
        assert_eq!(
            FunctionBody::from_id(&ctx, fid)
                .effects()
                .memory
                .materialized(),
            Some(&map),
            "first solve dropped the materialized memory interface"
        );
        crate::calls::set_all_written_spaces(&mut ctx);
        assert_eq!(
            FunctionBody::from_id(&ctx, fid)
                .effects()
                .memory
                .materialized(),
            Some(&map),
            "re-solve dropped the materialized memory interface"
        );
    }
}
