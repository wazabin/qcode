//! The default analysis pipeline: a single, shared definition of the canonical
//! `-O all` function pipeline and the whole-program driver that runs it.
//!
//! The pass ordering and per-pass setup (the mem2reg/const-fold fixpoint loop,
//! fresh [`AliasResult::simple`] oracles, register selection) live here once and
//! are reused by the GUI, the `opt` example, and the integration tests.
//!
//! `qcode_analysis` does not depend on any architecture crate, so the
//! architecture-specific registers the register-aware passes need are injected
//! via [`ArchConfig`]. Callers build one with `harbinger::arch::arch_config`.

use qcode::{
    assumption::AssumptionKnowledge,
    context::Context,
    value::{FunctionId, FunctionRef, RegisterId, ValueId, VarnodeId},
};

use crate::{
    AliasResult, apply_all_external_signatures, assume_call_returns, bind_all_call_args,
    brighten_stack, constant_fold_function, dead_load::remove_dead_load_insns, gvn_function,
    lower_stack, mem2reg, remove_dead_insns, set_all_call_clobbered_regs,
    set_all_function_summaries, simplify_cfg, verify_assumptions,
};

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

/// A single selectable analysis/optimization pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pass {
    BrightenStack,
    Mem2Reg,
    GVN,
    DeadStore,
    DeadLoad,
    DCE,
    Simplify,
    LowerStack,
}

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
    WholeProgramPhase {
        round: usize,
        name: &'static str,
    },
    FunctionPass {
        round: usize,
        stage: &'static str,
        function: String,
        index: usize,
        total: usize,
        pass: Pass,
    },
    Finished,
}

/// Every selectable pass, in the order `opt --opt all` applies them (with the
/// standalone dead-load slotted next to dead-store). Used to populate à-la-carte
/// pass selectors.
pub const ALL: &[Pass] = &[
    Pass::BrightenStack,
    Pass::Mem2Reg,
    Pass::GVN,
    Pass::DeadStore,
    Pass::DeadLoad,
    Pass::DCE,
    Pass::Simplify,
    Pass::LowerStack,
];

/// The canonical default pipeline. Dead-load is folded into dead-store here, so
/// it is not listed separately.
pub const DEFAULT: &[Pass] = &[
    Pass::BrightenStack,
    Pass::Mem2Reg,
    Pass::GVN,
    Pass::DeadStore,
    Pass::DCE,
    Pass::Simplify,
    Pass::LowerStack,
];

/// The passes run on every function *before* interprocedural call-argument
/// binding. Includes `Simplify` so the lifter's straight-line `goto`-chains are
/// merged into single blocks: the binder forwards each call's argument loads to
/// the reaching arg-setup store via gvn's *intra-block* store→load forwarding, so
/// the store and the call must share a block. `DeadStore`/`DCE` are deferred so
/// the arg-setup stores are still present when the binder forwards them.
const PRE_BIND: &[Pass] = &[
    Pass::BrightenStack,
    Pass::Mem2Reg,
    Pass::GVN,
    Pass::Simplify,
];

/// The eliminating passes run on every function *after* binding, to clean up the
/// now-dead arg-setup stores and the loads the binder forwarded.
const POST_BIND: &[Pass] = &[Pass::DeadStore, Pass::DCE];

/// The final lowering pass, run on every function after summaries have consumed
/// the `@stack_base` epilogue to recover each function's stack delta. The trailing
/// eliminating passes clear dead artifacts (e.g. a caller-side RSP relink whose
/// store is overwritten before use): DCE drops the dead arithmetic, DeadStore the
/// now-unused register load it fed, and a final DCE anything that exposes.
const LOWER: &[Pass] = &[Pass::LowerStack, Pass::DCE, Pass::DeadStore, Pass::DCE];

impl Pass {
    pub fn name(&self) -> &'static str {
        match self {
            Self::BrightenStack => "Brighten Stack",
            Self::Mem2Reg => "Mem2Reg",
            Self::GVN => "GVN",
            Self::DeadStore => "Dead Store",
            Self::DeadLoad => "Dead Load",
            Self::DCE => "DCE",
            Self::Simplify => "CFG Simplify",
            Self::LowerStack => "Lower Stack",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::BrightenStack => "Inject symbolic stack base store at function entry",
            Self::Mem2Reg => "Promote memory loads/stores to SSA block params (to fixpoint)",
            Self::GVN => "Global value numbering and constant folding",
            Self::DeadStore => "Remove dead register loads and overwritten flag stores",
            Self::DeadLoad => "Remove dead memory loads",
            Self::DCE => "Remove unused pure instructions",
            Self::Simplify => "Merge straight-line basic blocks",
            Self::LowerStack => "Rewrite @stack_base literals back onto the real stack pointer",
        }
    }

    /// Apply this pass to a single function.
    pub fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        cfg: &ArchConfig,
    ) -> Result<(), String> {
        match self {
            Self::BrightenStack => {
                brighten_stack(ctx, fun_id, cfg.stack_pointer).map_err(|e| e.to_string())?;
            }
            Self::Mem2Reg => {
                // Promote to a fixpoint. Each round may expose a register (e.g.
                // RBP) as the symbolic stack base; constant-folding then rewrites
                // RBP/RSP-relative arithmetic into concrete stack-slot literals,
                // which become promotable on the following round.
                //
                // mem2reg only queries register-vs-register aliasing, and the
                // register varnode set (hence its alias classes) is fixed for the
                // whole pass — folding only rewrites stack-pointer arithmetic into
                // literals, never registers — so one oracle is valid across the
                // entire fixpoint instead of being rebuilt each round.
                let aliases = AliasResult::simple(ctx);
                loop {
                    let promoted = mem2reg(ctx, fun_id, &aliases);
                    let folded = constant_fold_function(ctx, fun_id);
                    if !promoted && !folded {
                        break;
                    }
                }
            }
            Self::GVN => {
                // Canonicalize pointer arithmetic (e.g. `stack_base + offset`) into
                // literals *before* building the alias oracle, so it sees per-slot
                // stack locations rather than collapsing them onto `stack_base`.
                constant_fold_function(ctx, fun_id);
                let aliases = AliasResult::simple(ctx);
                gvn_function(ctx, fun_id, Some(&aliases));
            }
            Self::DeadStore => {
                let aliases = AliasResult::simple(ctx);
                remove_dead_load_insns(ctx, fun_id, Some(&aliases), &cfg.dead_flag_regs);
            }
            Self::DeadLoad => {
                let aliases = AliasResult::simple(ctx);
                remove_dead_load_insns(ctx, fun_id, Some(&aliases), &[]);
            }
            Self::DCE => {
                let block_ids: Vec<_> = FunctionRef::from_id(ctx, fun_id)
                    .blocks()
                    .map(|b| b.id)
                    .collect();
                for block_id in block_ids {
                    remove_dead_insns(ctx, block_id);
                }
            }
            Self::Simplify => simplify_cfg(ctx, fun_id),
            Self::LowerStack => {
                let sp = ctx.registers[&cfg.stack_pointer];
                lower_stack(ctx, fun_id, sp);
            }
        }
        Ok(())
    }
}

/// Run an ordered list of passes over a single function.
pub fn run_passes(
    ctx: &mut Context,
    fun_id: FunctionId,
    passes: &[Pass],
    cfg: &ArchConfig,
) -> Result<(), String> {
    run_passes_with_progress(ctx, fun_id, passes, cfg, |_| {})
}

/// Run an ordered list of passes over a single function and report each pass.
pub fn run_passes_with_progress(
    ctx: &mut Context,
    fun_id: FunctionId,
    passes: &[Pass],
    cfg: &ArchConfig,
    mut progress: impl FnMut(PipelineProgress),
) -> Result<(), String> {
    let function = FunctionRef::from_id(ctx, fun_id).name().to_string();
    let total = passes.len();
    for (index, pass) in passes.iter().enumerate() {
        progress(PipelineProgress::FunctionPass {
            round: 0,
            stage: "Pipeline",
            function: function.clone(),
            index: index + 1,
            total,
            pass: *pass,
        });
        pass.run(ctx, fun_id, cfg)
            .map_err(|e| format!("{}: {}", pass.name(), e))?;
    }
    Ok(())
}

/// Run the [`DEFAULT`] pipeline over every non-external function.
///
/// This is the "dependent" body wrapped by [`analyze_default`]: the passes the
/// checkpoint+replay driver runs on the assumed call-return edges.
pub fn run_default_all_functions(ctx: &mut Context, cfg: &ArchConfig) -> Result<(), String> {
    run_default_all_functions_with_progress(ctx, cfg, 0, &mut |_| {})
}

pub fn run_default_all_functions_with_progress(
    ctx: &mut Context,
    cfg: &ArchConfig,
    round: usize,
    progress: &mut impl FnMut(PipelineProgress),
) -> Result<(), String> {
    let fun_ids: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| !f.is_external())
        .map(|f| f.id)
        .collect();
    let total = fun_ids.len();

    // Seed each function's call-clobbered-register set from the lifted IR *before*
    // the value-producing passes run. mem2reg and GVN consult a callee's clobbers
    // to treat a call as a definition of those registers (so a post-call register
    // read is the call's output, not a forwarded pre-call value). This uses the
    // verified `written − read-before-written` set, which excludes a callee's
    // saved/restored frame pointer and incoming arg registers; the precise
    // summaries computed at binding (below) later overwrite it.
    progress(PipelineProgress::WholeProgramPhase {
        round,
        name: "Seed call clobbers",
    });
    set_all_call_clobbered_regs(ctx);

    // Value-producing passes first, on every function.
    for (index, fun_id) in fun_ids.iter().enumerate() {
        run_default_pass_group(
            ctx,
            *fun_id,
            PRE_BIND,
            cfg,
            PassGroupContext {
                round,
                stage: "Pre-bind",
                index,
                total,
            },
            progress,
        )?;
    }

    // Interprocedural binding, after every function is post-mem2reg/gvn but
    // *before* the eliminating passes: give known external (libc) callees
    // signatures from their C prototypes, infer each internal callee's
    // input/clobber/saved summary, then bind arguments and per-call alias sets at
    // every call site. Binding a caller reads its callees' summaries, so these are
    // whole-program phases rather than per-function passes. Running this before
    // DeadStore/DCE keeps the arg-setup stores alive long enough for the binder to
    // forward them into the call's argument list.
    let sp_varnode = ctx.registers[&cfg.stack_pointer];
    progress(PipelineProgress::WholeProgramPhase {
        round,
        name: "Apply external signatures",
    });
    apply_all_external_signatures(ctx, &cfg.abi);
    progress(PipelineProgress::WholeProgramPhase {
        round,
        name: "Set function summaries",
    });
    set_all_function_summaries(ctx, sp_varnode);
    progress(PipelineProgress::WholeProgramPhase {
        round,
        name: "Bind call arguments",
    });
    bind_all_call_args(ctx, sp_varnode);

    // Eliminating passes: drop the now-dead arg-setup stores and the loads the
    // binder forwarded (the CFG was already simplified in PRE_BIND).
    for (index, fun_id) in fun_ids.iter().enumerate() {
        run_default_pass_group(
            ctx,
            *fun_id,
            POST_BIND,
            cfg,
            PassGroupContext {
                round,
                stage: "Post-bind",
                index,
                total,
            },
            progress,
        )?;
    }

    // Lowering last: rewrite the surviving `@stack_base ± N` literals back onto
    // each function's real stack pointer. Runs after summaries/binding so the
    // `RSP = @stack_base + N` epilogue is still intact when the delta is read.
    for (index, fun_id) in fun_ids.iter().enumerate() {
        run_default_pass_group(
            ctx,
            *fun_id,
            LOWER,
            cfg,
            PassGroupContext {
                round,
                stage: "Lowering",
                index,
                total,
            },
            progress,
        )?;
    }
    Ok(())
}

/// Where a pass group sits within the overall pipeline run, for progress
/// reporting: which `round`, which named `stage`, and this function's `index`
/// among `total` functions.
struct PassGroupContext {
    round: usize,
    stage: &'static str,
    index: usize,
    total: usize,
}

fn run_default_pass_group(
    ctx: &mut Context,
    fun_id: FunctionId,
    passes: &[Pass],
    cfg: &ArchConfig,
    at: PassGroupContext,
    progress: &mut impl FnMut(PipelineProgress),
) -> Result<(), String> {
    let function = FunctionRef::from_id(ctx, fun_id).name().to_string();
    let total = at.total * passes.len();
    for (pass_index, pass) in passes.iter().enumerate() {
        progress(PipelineProgress::FunctionPass {
            round: at.round,
            stage: at.stage,
            function: function.clone(),
            index: at.index * passes.len() + pass_index + 1,
            total,
            pass: *pass,
        });
        pass.run(ctx, fun_id, cfg)
            .map_err(|e| format!("{}: {}", pass.name(), e))?;
    }
    Ok(())
}

/// The default analysis pipeline system entry point.
///
/// Runs the [`DEFAULT`] pipeline over the whole program under whole-program
/// checkpoint+replay (see [`analyze_with_assumptions`]), so call-return
/// assumptions are made before and verified after the passes. `baseline` is the
/// freshly-lifted IR (no speculation); the converged, optimized context is
/// returned.
///
/// Pass errors abort the analysis and surface through the panicking driver; the
/// passes only error on an unrecognised architecture, which `cfg` already pins.
pub fn analyze_default<'s>(baseline: &Context<'s>, cfg: &ArchConfig) -> Context<'s> {
    analyze_default_with_progress(baseline, cfg, |_| {})
}

pub fn analyze_default_with_progress<'s>(
    baseline: &Context<'s>,
    cfg: &ArchConfig,
    mut progress: impl FnMut(PipelineProgress),
) -> Context<'s> {
    let mut knowledge = AssumptionKnowledge::default();
    let mut round = 0usize;

    progress(PipelineProgress::Started);
    loop {
        round += 1;
        progress(PipelineProgress::AssumptionRound { round });

        let mut ctx = baseline.clone();
        let count = assume_call_returns(&mut ctx, &knowledge);
        progress(PipelineProgress::AssumptionsRecorded { round, count });

        run_default_all_functions_with_progress(&mut ctx, cfg, round, &mut progress)
            .expect("default pipeline pass failed");

        progress(PipelineProgress::WholeProgramPhase {
            round,
            name: "Verify assumptions",
        });
        let learned = verify_assumptions(&mut ctx);

        if learned.iter().all(|f| knowledge.noreturn.contains(f)) {
            progress(PipelineProgress::Finished);
            return ctx;
        }
        knowledge.noreturn.extend(learned);
    }
}
