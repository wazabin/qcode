//! The analysis pipeline: a TOML-described sequence of passes plus the
//! whole-program checkpoint+replay driver that runs it.
//!
//! Passes implement the [`pass`] traits and are registered by name; the pass
//! order, scopes, and fixpoints live in `default_pipeline.toml` (parsed by
//! [`config`]). The checkpoint+replay loop in [`analyze_with_pipeline`] stays in
//! Rust: each round clones the baseline IR, seeds the facts proven in earlier
//! rounds, records call-return assumptions, runs the configured [`Pipeline`],
//! verifies the assumptions, and replays from baseline until a round proves no
//! novel fact and records no violation.
//!
//! `qcode_analysis` does not depend on any architecture crate, so the
//! architecture-specific registers the register-aware passes need are injected
//! via [`ArchConfig`]. Callers build one with `harbinger::arch::arch_config`.

mod analysis_manager;
mod cone;
mod config;
mod lifter;
mod module_view;
mod pass;

pub use config::{DEFAULT_PIPELINE_TOML, Pipeline};
#[cfg(not(target_arch = "wasm32"))]
pub use config::{
    PipelineFile, create_named_user_pipeline_from_default_in, create_user_pipeline_from_default_in,
    list_user_pipelines_in, load_named_user_pipeline_in,
};
pub use lifter::{LiftOutcome, LiftSummary, Lifter, PipelineServices};
pub use module_view::{
    ContextSplit, ContextView, Minted, Outcome, host_with_minted, mint_function,
};
pub use qcode::value::FunctionBody;

/// Install a set of minted functions and resolve the owner's placeholders
/// (test-only shim over the barrier, so `test_util::with_minting` can exercise
/// outlining helpers without the full driver). Panics on an invalid slot or
/// name collision.
#[cfg(test)]
pub(crate) fn install_minted_for_test<'str>(
    ctx: &mut qcode::context::Context<'str>,
    owner: qcode::value::FunctionId,
    minted: Vec<Minted<'str>>,
) {
    let installed = pass::install_minted(ctx, "test", minted).expect("minted install");
    pass::resolve_minted_callees(ctx, "test", owner, &installed).expect("minted callee resolution");
}
pub use analysis_manager::{
    AnalysisManager, GlobalAnalysis, LocalAnalysis, LocalAnalysisManager, PreservedAnalyses,
};
pub use cone::{Cone, ConeMut};
pub(crate) use pass::with_body_mut;
pub use pass::{
    DecompilePass, DynDecompilePass, DynFunctionPass, DynPass, FunctionPass, FunctionPassAdapter,
    ModulePassOutcome, Pass, PassRegistration, PipelineEnv, RegisteredPass, known_pass_names,
    make_pass,
};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    assumption::{PassName, Proposition},
    context::{Context, TargetOs},
    value::{FunctionId, RegisterId, ValueId, VarnodeId},
};

use crate::{
    assume_call_returns, learn_stack_facts, seed_stack_facts, verify_args_disjoint_caller_frame,
    verify_assumptions, verify_forced_returns,
};

/// A shared handle to the loaded binary, threaded from the lift entry points
/// into every [`PipelineEnv`] the driver builds. `None` for headless/textual
/// runs; reloaded snapshots wrap the deserialized `MemoryImage`.
pub type BinaryHandle = std::sync::Arc<dyn binfmt::BinaryFormat>;

/// The [`cabi::AbiTarget`] a binary's prototype tables are keyed by: its
/// container-format OS mapped to a [`cabi::Platform`], at `bits` pointer width.
///
/// The single home for the `TargetOs → Platform` mapping, consulted by
/// `external_sigs` (signature materialization) and the GUI loader (doc links).
pub fn cabi_abi_target(os: TargetOs, bits: u8) -> cabi::AbiTarget {
    let platform = match os {
        TargetOs::Windows => cabi::Platform::Windows,
        // ELF/unknown binaries use the SysV/libc (host) table.
        TargetOs::Linux | TargetOs::Unknown => cabi::Platform::Linux,
    };
    cabi::AbiTarget::new(platform, bits)
}

/// Maximum checkpoint+replay rounds the overrides-aware driver attempts before
/// giving up with [`PipelineError::NoConvergence`].
const MAX_OVERRIDE_ROUNDS: usize = 5;
/// Fallback analyze/lift round cap, used only when the driver has no wall-clock
/// deadline ([`PipelineServices::deadline`], e.g. on wasm). With a deadline the
/// rounds are unbounded in count and bounded in time instead.
const MAX_ANALYZE_LIFT_ROUNDS: usize = 100;

/// True when the driver's budget is exhausted: past the wall-clock deadline, or
/// past the fallback round cap when no deadline is set.
fn out_of_budget(deadline: Option<std::time::Instant>, round: usize) -> bool {
    match deadline {
        Some(d) => std::time::Instant::now() >= d,
        None => round >= MAX_ANALYZE_LIFT_ROUNDS,
    }
}

/// Failure modes of [`analyze_with_overrides_with_progress`].
#[derive(Debug, Clone)]
pub enum PipelineError {
    /// A user-forced fact was disproved by analysis: the forced value and the
    /// proven value disagree irreconcilably.
    Contradiction(qcode::assumption::KnownContradiction),
    /// The checkpoint+replay loop did not converge within the round cap.
    NoConvergence { rounds: usize },
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Contradiction(c) => write!(
                f,
                "forced assumption rejected: {:?} was set to {} but {} proved {}",
                c.prop, c.known, c.proven_pass, c.proven,
            ),
            PipelineError::NoConvergence { rounds } => {
                write!(f, "analysis did not converge after {rounds} rounds")
            }
        }
    }
}

impl std::error::Error for PipelineError {}

/// The architecture-specific registers the register-aware passes need.
///
/// Built by `harbinger::arch::arch_config`, which resolves these from the loaded
/// context's pointer width.
#[derive(Clone)]
pub struct ArchConfig {
    /// Stack-pointer register (RSP on x64, ESP on x86), used to resolve
    /// [`PipelineEnv::sp_varnode`](crate::PipelineEnv::sp_varnode) — the single
    /// handle every pass reads. Prefer [`ArchConfig::sp_varnode`] over touching
    /// this field.
    pub stack_pointer: RegisterId,
    /// Status-flag registers (CF/OF/SF/ZF/PF) treated as dead by dead-store.
    pub dead_flag_regs: Vec<ValueId>,
    /// Registers whose unused functional-signature slots may be removed.
    ///
    /// This is architecture policy rather than a generic DCE decision. For the
    /// x86 families it contains the status flags, stack/frame pointers, and
    /// instruction pointer.
    pub killable_registers: rustc_hash::FxHashSet<VarnodeId>,
    /// Calling-convention argument/return register layout, used to give known
    /// external (libc) functions signatures. Empty for unsupported arches.
    pub abi: CallingConvention,
    /// The loaded binary's operating system (from the container format).
    /// Platform-gated passes (e.g. TEB seeding) read this.
    pub os: TargetOs,
    /// Pointer width in bits (32 or 64), derived from the default space.
    pub bitness: u8,
    /// Opt-in: install the whole-program `AssumeCallingConvention` hypothesis
    /// (see [`Proposition::AssumeCallingConvention`](qcode::assumption::Proposition::AssumeCallingConvention)),
    /// giving indirect / unresolved calls an educated-guess register effect
    /// (reads = argument registers, writes = caller-saved) instead of
    /// clobbers-all. A deliberate, controllable unsoundness — **off by default**;
    /// the user turns it on. Consulted only by the `assume_calling_convention`
    /// install pass.
    pub assume_calling_convention: bool,
}

impl ArchConfig {
    /// Resolve [`Self::stack_pointer`] against `ctx`'s register table — the one
    /// place a register *identity* becomes an SP varnode handle.
    ///
    /// Passes never call this: they read the cached
    /// [`PipelineEnv::sp_varnode`](crate::PipelineEnv::sp_varnode). It exists for
    /// env construction and for the few pre-env driver/UI sites that need the
    /// handle before (or without) a `PipelineEnv`. `None` when the architecture's
    /// stack-pointer register is not present in `ctx` (hand-written IR).
    pub fn sp_varnode(&self, ctx: &Context) -> Option<VarnodeId> {
        ctx.shared.registers.get(&self.stack_pointer).copied()
    }
}

/// A general-purpose argument/return register exposed at several byte widths
/// (e.g. `RDI`/`EDI`/`DI`/`DIL` for the first integer argument).
#[derive(Clone)]
pub struct GpReg {
    /// `(byte_width, varnode)` views of one physical register.
    pub widths: Vec<(usize, VarnodeId)>,
}

impl GpReg {
    /// The sub-register view best matching a `bytes`-wide value: an exact-width
    /// match, else the smallest view at least that wide, else the widest view.
    pub fn for_bytes(&self, bytes: usize) -> Option<VarnodeId> {
        self.widths
            .iter()
            .find(|(w, _)| *w == bytes)
            .or_else(|| {
                self.widths
                    .iter()
                    .filter(|(w, _)| *w >= bytes)
                    .min_by_key(|(w, _)| *w)
            })
            .or_else(|| self.widths.iter().max_by_key(|(w, _)| *w))
            .map(|(_, v)| *v)
    }
}

/// The subset of a calling convention needed to assign argument and return
/// registers to a C prototype. Currently models x64 System V.
#[derive(Clone, Default)]
pub struct CallingConvention {
    /// Integer/pointer argument registers, in order (RDI, RSI, RDX, RCX, R8, R9).
    pub int_args: Vec<GpReg>,
    /// SSE (float/double) argument registers, in order (XMM0..XMM7).
    pub sse_args: Vec<VarnodeId>,
    /// Integer/pointer return register (RAX), by width.
    pub int_ret: Option<GpReg>,
    /// SSE return register (XMM0).
    pub sse_ret: Option<VarnodeId>,
    /// Caller-saved (volatile) registers: the registers a callee may clobber
    /// without preserving, which a caller must therefore treat as written across
    /// any call obeying this convention. Used to seed the clobber set of
    /// **resolved** external callees (whose body is unavailable), one varnode per
    /// physical register — typically its widest view (e.g. RAX, not EAX/AX/AL),
    /// since the clobber test is overlap-aware. Empty when the convention is
    /// unknown (the conservative every-register fallback then applies).
    pub caller_saved: Vec<VarnodeId>,
}

/// Progress events emitted while a pipeline runs, for the GUI/loader to display.
///
/// `stage`/`function` are `Arc<str>` so the driver can emit one event per pass
/// per fixpoint iteration without re-allocating the names each time.
#[derive(Clone, Debug)]
pub enum PipelineProgress {
    Started,
    AssumptionRound {
        round: usize,
    },
    AssumptionsRecorded {
        round: usize,
        count: usize,
    },
    /// A whole-program (module-scoped) pass, or a driver phase like verification.
    WholeProgramPhase {
        round: usize,
        stage: std::sync::Arc<str>,
        pass: &'static str,
    },
    /// A per-function pass running on `function` (`index` of `total`).
    FunctionPass {
        round: usize,
        stage: std::sync::Arc<str>,
        function: std::sync::Arc<str>,
        index: usize,
        total: usize,
        pass: &'static str,
    },
    Finished,
}

/// The default analysis pipeline system entry point: run the canonical
/// [`DEFAULT_PIPELINE_TOML`] pipeline over the whole program under checkpoint+replay.
///
/// `baseline` is the freshly-lifted IR (no speculation); the converged, optimized
/// context is returned.
pub fn analyze_default<'s>(
    baseline: &Context<'s>,
    cfg: &ArchConfig,
    binary: Option<BinaryHandle>,
) -> Context<'s> {
    analyze_default_with_progress(baseline, cfg, binary, |_| {})
}

pub fn analyze_default_with_progress<'s>(
    baseline: &Context<'s>,
    cfg: &ArchConfig,
    binary: Option<BinaryHandle>,
    progress: impl FnMut(PipelineProgress),
) -> Context<'s> {
    analyze_with_pipeline_with_progress(baseline, cfg, &Pipeline::default(), binary, progress)
}

/// Run an arbitrary parsed `pipeline` over the whole program under the same
/// checkpoint+replay driver as [`analyze_default`].
pub fn analyze_with_pipeline<'s>(
    baseline: &Context<'s>,
    cfg: &ArchConfig,
    pipeline: &Pipeline,
    binary: Option<BinaryHandle>,
) -> Context<'s> {
    analyze_with_pipeline_with_progress(baseline, cfg, pipeline, binary, |_| {})
}

pub fn analyze_with_pipeline_with_progress<'s>(
    baseline: &Context<'s>,
    cfg: &ArchConfig,
    pipeline: &Pipeline,
    binary: Option<BinaryHandle>,
    progress: impl FnMut(PipelineProgress),
) -> Context<'s> {
    analyze_and_lift_with_progress(
        baseline,
        cfg,
        pipeline,
        PipelineServices::none(),
        binary,
        progress,
    )
}

/// The checkpoint+replay driver, with two contexts and lifting folded in.
///
/// `clean` (the persistent clean IR) is cloned from `baseline` and grown by the
/// lifter until address discovery reaches a fixpoint. While it is still growing,
/// only function-local discovery analysis runs on disposable clones; the full
/// interprocedural checkpoint+replay pipeline runs after lifting is quiet.
///
/// `lifter == None` reproduces the original single-context behavior: the lifting
/// passes are inert, `clean` never grows, and discoveries (if any) ride out on the
/// returned context for an external driver to handle.
///
/// [`LIFT_BARRIER`]: config::LIFT_BARRIER
pub fn analyze_and_lift_with_progress<'s>(
    baseline: &Context<'s>,
    cfg: &ArchConfig,
    pipeline: &Pipeline,
    services: PipelineServices<'_>,
    binary: Option<BinaryHandle>,
    progress: impl FnMut(PipelineProgress),
) -> Context<'s> {
    analyze_with_progress(
        baseline,
        cfg,
        pipeline,
        services,
        binary,
        &std::collections::HashMap::default(),
        progress,
    )
    .expect("a run with no overrides is infallible")
}

/// The single analysis driver: discovery + lifting + assumption checkpoint/replay
/// folded into one code path.
///
/// `services` injects the lifter (whole-binary recursive disassembly grows the
/// persistent clean IR until address discovery reaches a fixpoint); `None`
/// reproduces single-context analysis of an already-lifted `baseline`. `overrides`
/// are user-forced facts (empty for the default analysis), honored on the lifting
/// path too, so a jump table that only resolves under a user override gets its
/// targets lifted.
///
/// Verify-then-use: the binary's memory protections are established *before* the
/// lift loop consults executability. Returns the converged optimized context, or a
/// [`PipelineError`] when a forced override is disproved or the bounded
/// (overrides) fixpoint fails to converge. With no overrides it is infallible.
pub fn analyze_with_progress<'s>(
    baseline: &Context<'s>,
    cfg: &ArchConfig,
    pipeline: &Pipeline,
    mut services: PipelineServices<'_>,
    binary: Option<BinaryHandle>,
    overrides: &std::collections::HashMap<Proposition, bool>,
    mut progress: impl FnMut(PipelineProgress),
) -> Result<Context<'s>, PipelineError> {
    let lifting = services.lifter.is_some();
    progress(PipelineProgress::Started);

    if !lifting {
        let analyzed =
            run_analysis_fixpoint(baseline, cfg, pipeline, &binary, overrides, &mut progress)?;
        progress(PipelineProgress::Finished);
        return Ok(analyzed);
    }

    // The persistent clean IR. Only the lifting passes mutate it; it grows
    // monotonically across discovery rounds.
    let mut clean = baseline.clone();
    let deadline = services.deadline;
    let mut discovery_round = 0usize;
    loop {
        let converged = lift_and_discover_until_quiet(
            &mut clean,
            cfg,
            pipeline,
            &mut services,
            &binary,
            &mut discovery_round,
            &mut progress,
        );

        let mut analyzed =
            run_analysis_fixpoint(&clean, cfg, pipeline, &binary, overrides, &mut progress)?;
        // Stop on a clean fixpoint, or bail out best-effort if lifting could not
        // converge (budget / error) — never panic on input-dependent paths.
        if !converged || analyzed.has_no_discoveries() {
            log_obligations(&services.obligations);
            progress(PipelineProgress::Finished);
            return Ok(analyzed);
        }

        // Budget check *after* the analysis so the returned context is always an
        // analyzed view of the most-grown IR; no further optimization round runs.
        if out_of_budget(deadline, discovery_round) {
            log::warn!(
                target: "pipeline",
                "analyze/lift budget exhausted after {discovery_round} rounds; returning best-effort analysis of the most-grown IR ({} pending discoveries)",
                analyzed.discoveries().count(),
            );
            log_obligations(&services.obligations);
            progress(PipelineProgress::Finished);
            return Ok(analyzed);
        }

        for discovery in analyzed.drain_discoveries() {
            clean.discover(discovery);
        }
    }
}

/// Grow the clean IR until address discovery is quiet for this analysis round.
/// Returns `true` on a clean fixpoint, `false` if it bailed best-effort (round
/// cap reached) so the caller can return the most-grown analysis without panicking.
fn lift_and_discover_until_quiet(
    clean: &mut Context<'_>,
    cfg: &ArchConfig,
    pipeline: &Pipeline,
    services: &mut PipelineServices<'_>,
    binary: &Option<BinaryHandle>,
    discovery_round: &mut usize,
    progress: &mut impl FnMut(PipelineProgress),
) -> bool {
    // Fingerprint of each function's clean IR as analyzed in the previous discovery
    // round. Address discovery is function-local and deterministic, so a function whose
    // clean IR is unchanged re-derives exactly the discoveries it produced last round
    // (already drained) — only changed/new functions need re-analysis. This drives the
    // `restrict` set below; on large binaries it turns a late round that re-optimizes
    // thousands of functions to find a few addresses into one that touches only those.
    let mut prev_bodies: HashMap<FunctionId, u64> = HashMap::default();
    loop {
        if out_of_budget(services.deadline, *discovery_round) {
            log::warn!(
                target: "pipeline",
                "lift/discover budget exhausted after {} rounds; stopping best-effort with {} pending discoveries",
                *discovery_round,
                clean.discoveries().count(),
            );
            return false;
        }
        *discovery_round += 1;
        let round = *discovery_round;
        progress(PipelineProgress::AssumptionRound { round });
        log::info!(target: "pipeline", "discovery round {round} starting");
        #[cfg(not(target_arch = "wasm32"))]
        let started = std::time::Instant::now();

        // Lifting phase: grow the raw clean IR through the TOML-configured
        // pre-barrier stages. Newly discovered addresses are drained by the
        // ordinary `lift_new_addresses` pass.
        let env = PipelineEnv::new(clean, cfg.clone(), binary.clone());
        if let Err(e) = pipeline.run_lifting_phase(clean, &env, services, round, progress) {
            log::warn!(target: "pipeline", "lifting phase failed, continuing best-effort: {e}");
        }

        // Restrict this round's address discovery to functions whose clean IR changed
        // (or appeared) since the previous round — every other function would re-derive
        // the same, already-drained discoveries. The fingerprints are taken after the
        // lifting phase so they reflect this round's newly lifted blocks, last round's
        // drained discoveries, and any function splits. Round 1 (empty map) analyzes
        // everything. Note `function_fingerprint` keys on the rendered body, so a
        // split or block edit deterministically shows up as a change next round.
        let bodies: HashMap<FunctionId, u64> = clean
            .functions()
            .filter(|f| !f.is_external())
            .map(|f| (f.id, config::cheap_function_fingerprint(clean, f.id)))
            .collect();
        let restrict: HashSet<FunctionId> = bodies
            .iter()
            .filter(|(f, fp)| prev_bodies.get(f) != Some(fp))
            .map(|(f, _)| *f)
            .collect();
        log::info!(
            target: "pipeline",
            "discovery round {round}: analyzing {}/{} functions changed since last round",
            restrict.len(),
            bodies.len(),
        );
        prev_bodies = bodies;

        // Derive a disposable function-local analysis context from clean IR.
        // Discoveries found here are the only durable output; analysis residue is
        // discarded so newly lifted blocks invalidate the whole owning function.
        let mut ctx = clean.clone();
        let analysis_env = PipelineEnv::new(clean, cfg.clone(), binary.clone());
        if let Err(e) = pipeline.run_address_discovery_phase(
            &mut ctx,
            &analysis_env,
            Some(&restrict),
            round,
            progress,
        ) {
            log::warn!(target: "pipeline", "address discovery pass failed, continuing best-effort: {e}");
        }

        for discovery in ctx.drain_discoveries() {
            clean.discover(discovery);
        }

        // Harvest reconstruction obligations before `ctx` is dropped. Two
        // sources, deliberately: enumerating the clone recovers the full set of
        // live indirect transfers (so a site no resolver touched is still
        // recorded), while the sink carries the outcomes resolvers reported.
        // `merge_round` reconciles them by status precedence.
        //
        // Enumeration runs on the analyzed clone rather than on `clean` because
        // that is the IR resolvers actually saw; a transfer already rewritten in
        // the clone has genuinely been discharged.
        let observed = crate::reconstruction::enumerate_obligations(&ctx)
            .into_iter()
            .chain(analysis_env.obligations.take());
        services.obligations.merge_round(observed);

        // Function boundaries are settled at construction: the lifter emits
        // strict-local IR (tail calls for inter-procedural transfers, function
        // splits at shared/mid-function landings — context-split ruling 2), so no
        // round-end re-derivation of ownership is needed.
        let pending = !clean.has_no_discoveries();

        log_round_stats(round);
        #[cfg(not(target_arch = "wasm32"))]
        log::info!(
            target: "pipeline",
            "discovery round {round} finished in {:.2?}: pending_lifts={pending}",
            started.elapsed(),
        );

        if !pending {
            return true;
        }
    }
}

/// The shared checkpoint+replay fixpoint over a stable `baseline` (clean IR).
///
/// Each round derives a *fresh* clone of `baseline`, seeds the accumulated
/// `knowledge` (and any user `overrides`), runs the full optimization pipeline
/// once, then verifies assumptions; it replays while a round produces novel facts
/// or violations. Knowledge only grows, so it terminates.
///
/// `overrides` are user-forced facts kept authoritative across replays. When they
/// are present the fixpoint is *bounded*: it additionally verifies forced returns,
/// returns [`PipelineError::Contradiction`] if analysis disproves an override, and
/// gives up with [`PipelineError::NoConvergence`] after [`MAX_OVERRIDE_ROUNDS`].
/// With no overrides it cannot fail (knowledge-growth termination, no forced
/// facts to contradict) and the loop is unbounded.
///
/// Emits no `Started`/`Finished` progress — the top-level driver brackets those.
fn run_analysis_fixpoint<'s>(
    baseline: &Context<'s>,
    cfg: &ArchConfig,
    pipeline: &Pipeline,
    binary: &Option<BinaryHandle>,
    overrides: &std::collections::HashMap<Proposition, bool>,
    progress: &mut impl FnMut(PipelineProgress),
) -> Result<Context<'s>, PipelineError> {
    // Strict IR locality (context-split ruling 2) is an invariant of construction:
    // the lifter emits tail calls for inter-procedural transfers and splits at
    // shared/mid-function landings, so `baseline` — whether lifted, parsed from
    // textual IR, or wasm-produced — already holds no foreign block reference. No
    // entry normalization is needed.
    let env = PipelineEnv::new(baseline, cfg.clone(), binary.clone());
    let bounded = !overrides.is_empty();
    // User overrides carry a synthetic "override" provenance so the converged
    // context attributes a forced fact to the user, not the re-seeding driver.
    let mut knowledge: HashMap<Proposition, (bool, PassName)> = overrides
        .iter()
        .map(|(&prop, &value)| (prop, (value, PassName("override"))))
        .collect();
    let mut round = 0usize;

    loop {
        round += 1;
        progress(PipelineProgress::AssumptionRound { round });
        log::info!(target: "pipeline", "assumption round {round} starting");
        #[cfg(not(target_arch = "wasm32"))]
        let started = std::time::Instant::now();

        // Invariant: each round derives a *fresh* clone of the raw `baseline`
        // (clean IR) and runs the full optimization pipeline on it exactly once.
        // Knowledge carries across rounds via `seed_known`/facts, never via
        // accumulated IR mutations, so no optimization pass is ever applied to its
        // own output — they need not be idempotent. Preserve this on refactors.
        let mut ctx = baseline.clone();
        for (&prop, &(value, pass)) in &knowledge {
            ctx.seed_known(prop, value, pass);
        }
        let count = assume_call_returns(&mut ctx);
        // Re-apply the stack-escape facts proven in earlier rounds so this round's
        // mem2reg/summary passes observe them (mirrors `assume_call_returns`).
        seed_stack_facts(&mut ctx);
        // The `ArgsDisjointFromCallerFrame` assumption is recorded *inside* the
        // pipeline (the `assume-arg-frame` stage, before each `argpromote`): the
        // `@SP` param it keys on is minted mid-pipeline, so it cannot be assumed
        // here on the raw baseline. It is verified below (and rolled back if a
        // caller is proven to pass a colliding pointer).
        let sp_reg = cfg.sp_varnode(&ctx);
        progress(PipelineProgress::AssumptionsRecorded { round, count });

        if let Err(e) = pipeline.run(&mut ctx, &env, round, progress) {
            // A `repeat_until` stage that never settles is an input-dependent
            // property of one function's IR, not a compiler bug. Rather than
            // panic and lose everything, halt the pipeline best-effort: the
            // aborting `?` already stopped the remaining stages, so `ctx` holds
            // the IR up to (and including) the non-converged stage — hand it back
            // so it can be viewed (e.g. in the GUI) instead of crashing. Drop any
            // pending discoveries so the outer lift/discover loop stops here rather
            // than replaying the same non-converging shape every round. Any other
            // error is a real failure and still aborts loudly.
            if config::is_nonconvergence(&e) {
                log::warn!(
                    target: "pipeline",
                    "round {round}: {e}; halting analysis best-effort and surfacing the current IR",
                );
                let _ = ctx.drain_discoveries();
                return Ok(ctx);
            }
            panic!("pipeline pass failed: {e}");
        }

        progress(PipelineProgress::WholeProgramPhase {
            round,
            stage: "verify".into(),
            pass: "verify_assumptions",
        });
        // Converge only once a round proves no novel fact — a violated
        // call-return assumption, a newly-learned unbounded stack reader, or a
        // frame-escaping caller all change what earlier passes would have done,
        // so the round must replay with the fact seeded. Knowledge only grows,
        // hence termination.
        let novel = verify_assumptions(&mut ctx)
            + learn_stack_facts(&mut ctx)
            + verify_args_disjoint_caller_frame(&mut ctx, sp_reg);

        if bounded {
            // verify_assumptions skips facts already *known* (the overrides), so
            // check each forced FunctionReturns against the body explicitly.
            verify_forced_returns(&mut ctx, overrides);
            // A user override the analysis disproved cannot be honored — abort.
            if let Some(c) = ctx.known_contradictions().first() {
                log::warn!(
                    target: "pipeline",
                    "round {round}: forced {:?}={} rejected by {} (proved {})",
                    c.prop, c.known, c.proven_pass, c.proven,
                );
                return Err(PipelineError::Contradiction(*c));
            }
        }

        for v in ctx.violations() {
            log::info!(
                target: "pipeline",
                "round {round}: {} violated {:?} (assumed {} by {}, proven {})",
                v.asserting_pass, v.prop, v.assumed, v.assuming_pass, !v.assumed,
            );
        }
        // Stable-arena churn probe: logical IDs remain issued after their dense
        // payloads are removed.
        let (total_insns, dead_insns) = ctx.instruction_arena_stats();
        log::info!(
            target: "pipeline",
            "round {round}: instruction arena {total_insns} IDs issued, {} live payloads, {dead_insns} removed IDs ({:.1}% removed)",
            total_insns - dead_insns,
            if total_insns == 0 { 0.0 } else { 100.0 * dead_insns as f64 / total_insns as f64 },
        );
        log_round_stats(round);
        #[cfg(not(target_arch = "wasm32"))]
        log::info!(
            target: "pipeline",
            "round {round} finished in {:.2?}: {novel} novel facts, {} violations",
            started.elapsed(),
            ctx.violations().len(),
        );

        if novel == 0 && ctx.violations().is_empty() {
            // Analysis churn leaves the body arenas at peak capacity; release
            // the slack once, at convergence, where no further mutation follows.
            ctx.shrink_bodies_to_fit();
            return Ok(ctx);
        }
        if bounded && round >= MAX_OVERRIDE_ROUNDS {
            return Err(PipelineError::NoConvergence { rounds: round });
        }

        knowledge.extend(ctx.known_facts().map(|(p, v, pass)| (p, (v, pass))));
        if bounded {
            // Keep the user's overrides authoritative over anything learned.
            for (&prop, &value) in overrides {
                knowledge.insert(prop, (value, PassName("override")));
            }
        }
    }
}

/// Like [`analyze_with_pipeline_with_progress`], but seeds a set of user-forced
/// `overrides` as known facts before round 1 and keeps them authoritative across
/// replays. Used by the GUI assumption panel: a user can pin a proposition to a
/// value and rerun.
///
/// An override is treated exactly like a machine-proven known fact. If analysis
/// disproves one (a [`KnownContradiction`](qcode::assumption::KnownContradiction)),
/// or the loop fails to converge within [`MAX_OVERRIDE_ROUNDS`], the driver
/// returns a [`PipelineError`] instead of looping or panicking.
pub fn analyze_with_overrides_with_progress<'s>(
    baseline: &Context<'s>,
    cfg: &ArchConfig,
    pipeline: &Pipeline,
    binary: Option<BinaryHandle>,
    overrides: &std::collections::HashMap<Proposition, bool>,
    progress: impl FnMut(PipelineProgress),
) -> Result<Context<'s>, PipelineError> {
    analyze_with_progress(
        baseline,
        cfg,
        pipeline,
        PipelineServices::none(),
        binary,
        overrides,
        progress,
    )
}

/// Drain the per-pass counters accumulated during this round (via
/// [`qcode::stat!`]) and log them as one `debug` table.
/// Report the reconstruction obligations left standing at the end of a run.
///
/// This is the milestone's "expose obligations in logs and headless reports"
/// surface. Outstanding obligations are indirect transfers a resolver could
/// have handled and did not; inert ones are indirect calls, which have no
/// resolver yet and are counted separately so a missing feature never reads as
/// a reconstruction failure.
fn log_obligations(db: &crate::reconstruction::ObligationDb) {
    if db.is_empty() {
        return;
    }

    let outstanding = db.outstanding().count();
    let inert = db.inert().count();
    let resolved = db.iter().filter(|o| o.status.is_resolved()).count();

    log::info!(
        target: "obligations",
        "reconstruction obligations: {resolved} resolved, {outstanding} outstanding, \
         {inert} inert (indirect calls, no resolver yet)",
    );
    qcode::stat!("obligations_outstanding", outstanding as u64);
    qcode::stat!("obligations_resolved", resolved as u64);

    // Per-obligation detail is debug-level: a large binary can carry thousands,
    // and the summary above is what a normal run needs.
    for obligation in db.outstanding() {
        log::debug!(target: "obligations", "  {}", obligation.explain());
    }
}

fn log_round_stats(round: usize) {
    let stats = qcode::pass_scope::drain_stats();
    if stats.is_empty() || !log::log_enabled!(target: "pipeline", log::Level::Debug) {
        return;
    }
    let mut table = String::new();
    for ((pass, key), n) in stats {
        table.push_str(&format!("\n  {pass:<24} {key:<32} {n}"));
    }
    log::debug!(target: "pipeline", "round {round} statistics:{table}");
}
