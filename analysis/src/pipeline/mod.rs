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

mod config;
mod pass;

pub use config::{DEFAULT_PIPELINE_TOML, Pipeline};
pub use pass::{
    DynFunctionPass, DynPass, FunctionPass, Pass, PassRegistration, PipelineEnv, RegisteredPass,
};

use std::collections::HashMap;

use qcode::{
    assumption::Proposition,
    context::Context,
    value::{RegisterId, ValueId, VarnodeId},
};

use crate::{assume_call_returns, learn_stack_facts, seed_stack_facts, verify_assumptions};

/// The architecture-specific registers the register-aware passes need.
///
/// Built by `harbinger::arch::arch_config`, which resolves these from the loaded
/// context's pointer width.
#[derive(Clone)]
pub struct ArchConfig {
    /// Stack-pointer register (RSP on x64, ESP on x86), used by brighten-stack.
    pub stack_pointer: RegisterId,
    /// Status-flag registers (CF/OF/SF/ZF/PF) treated as dead by dead-store.
    pub dead_flag_regs: Vec<ValueId>,
    /// Calling-convention argument/return register layout, used to give known
    /// external (libc) functions signatures. Empty for unsupported arches.
    pub abi: CallingConvention,
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
pub fn analyze_default<'s>(baseline: &Context<'s>, cfg: &ArchConfig) -> Context<'s> {
    analyze_default_with_progress(baseline, cfg, |_| {})
}

pub fn analyze_default_with_progress<'s>(
    baseline: &Context<'s>,
    cfg: &ArchConfig,
    progress: impl FnMut(PipelineProgress),
) -> Context<'s> {
    analyze_with_pipeline_with_progress(baseline, cfg, &Pipeline::default(), progress)
}

/// Run an arbitrary parsed `pipeline` over the whole program under the same
/// checkpoint+replay driver as [`analyze_default`].
pub fn analyze_with_pipeline<'s>(
    baseline: &Context<'s>,
    cfg: &ArchConfig,
    pipeline: &Pipeline,
) -> Context<'s> {
    analyze_with_pipeline_with_progress(baseline, cfg, pipeline, |_| {})
}

pub fn analyze_with_pipeline_with_progress<'s>(
    baseline: &Context<'s>,
    cfg: &ArchConfig,
    pipeline: &Pipeline,
    mut progress: impl FnMut(PipelineProgress),
) -> Context<'s> {
    let env = PipelineEnv::new(baseline, cfg.clone());
    let mut knowledge: HashMap<Proposition, bool> = HashMap::new();
    let mut round = 0usize;

    progress(PipelineProgress::Started);
    loop {
        round += 1;
        progress(PipelineProgress::AssumptionRound { round });
        log::info!(target: "pipeline", "assumption round {round} starting");
        let started = std::time::Instant::now();

        let mut ctx = baseline.clone();
        for (&prop, &value) in &knowledge {
            ctx.seed_known(prop, value);
        }
        let count = assume_call_returns(&mut ctx);
        // Re-apply the stack-escape facts proven in earlier rounds so this round's
        // mem2reg/summary passes observe them (mirrors `assume_call_returns`).
        seed_stack_facts(&mut ctx);
        progress(PipelineProgress::AssumptionsRecorded { round, count });

        pipeline
            .run(&mut ctx, &env, round, &mut progress)
            .unwrap_or_else(|e| panic!("pipeline pass failed: {e}"));

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
        let novel = verify_assumptions(&mut ctx) + learn_stack_facts(&mut ctx);

        for v in ctx.violations() {
            log::info!(
                target: "pipeline",
                "round {round}: {} violated {:?} (assumed {} by {}, proven {})",
                v.asserting_pass, v.prop, v.assumed, v.assuming_pass, !v.assumed,
            );
        }
        log_round_stats(round);
        log::info!(
            target: "pipeline",
            "round {round} finished in {:.2?}: {novel} novel facts, {} violations",
            started.elapsed(),
            ctx.violations().len(),
        );

        if novel == 0 && ctx.violations().is_empty() {
            progress(PipelineProgress::Finished);
            return ctx;
        }
        knowledge.extend(ctx.known_facts());
    }
}

/// Drain the per-pass counters accumulated during this round (via
/// [`qcode::stat!`]) and log them as one `debug` table.
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
