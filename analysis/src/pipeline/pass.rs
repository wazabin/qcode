//! The pass traits, the per-function adapter boundary, and the name registry.
//!
//! Every pass that can appear in a pipeline TOML implements one of two traits:
//!
//! - [`FunctionPass`] — operates on a single function. Most optimizations
//!   (mem2reg, gvn, dce, …) are these. Function-scoped TOML stages run them
//!   function-major and can loop them to a per-function fixpoint.
//! - [`Pass`] — operates on the whole program (`Context`). The interprocedural
//!   "milestone" steps (binding, summaries, external signatures, clobber seeding)
//!   are these.
//!
//! Both pull their architecture-specific inputs from [`PipelineEnv`] at run time,
//! so the registry can construct every pass as a zero-argument unit struct.
//! Adding a pass means writing one `impl` and adding one arm to [`make_pass`].

use qcode::{
    context::Context,
    value::{FunctionId, FunctionRef, VarnodeId},
};

use super::ArchConfig;
use crate::{
    AliasResult, apply_all_external_signatures, bind_all_call_args, brighten_stack,
    constant_fold_function, dead_load::remove_dead_load_insns, gvn_function, lower_stack, mem2reg,
    remove_dead_insns, set_all_call_clobbered_regs, set_all_function_summaries, simplify_cfg,
};

/// The architecture-specific inputs the register-aware passes need, resolved once
/// per pipeline run and shared by reference with every pass.
pub struct PipelineEnv {
    /// Register layout / calling convention, built by `harbinger::arch::arch_config`.
    pub cfg: ArchConfig,
    /// The stack-pointer *varnode* (`cfg.stack_pointer` resolved through
    /// `ctx.registers`), cached so passes don't re-resolve it each call.
    pub sp_varnode: VarnodeId,
}

impl PipelineEnv {
    /// Resolve the stack-pointer varnode from `cfg` against `ctx` once.
    pub fn new(ctx: &Context, cfg: ArchConfig) -> Self {
        let sp_varnode = ctx.registers[&cfg.stack_pointer];
        Self { cfg, sp_varnode }
    }
}

/// A pass over a single function. `run` returns `Ok(true)` if it changed the IR,
/// so a function-scoped stage can loop it to a fixpoint. Passes that don't track
/// change return `Ok(false)` and must not be placed in a `repeat_until` stage.
pub trait FunctionPass {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn run(&self, ctx: &mut Context, fun_id: FunctionId, env: &PipelineEnv)
    -> Result<bool, String>;
}

/// A whole-program pass (an interprocedural milestone). `run` returns `Ok(true)`
/// if it changed anything; the only milestones today report `Ok(false)`.
pub trait Pass {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn run(&self, ctx: &mut Context, env: &PipelineEnv) -> Result<bool, String>;
}

// ----- per-function passes ---------------------------------------------------

pub struct Brighten;
impl FunctionPass for Brighten {
    fn name(&self) -> &'static str {
        "brighten"
    }
    fn description(&self) -> &'static str {
        "Inject symbolic stack base store at function entry"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        env: &PipelineEnv,
    ) -> Result<bool, String> {
        brighten_stack(ctx, fun_id, env.cfg.stack_pointer).map_err(|e| e.to_string())?;
        Ok(false)
    }
}

pub struct Mem2Reg;
impl FunctionPass for Mem2Reg {
    fn name(&self) -> &'static str {
        "mem2reg"
    }
    fn description(&self) -> &'static str {
        "Promote memory loads/stores to SSA block params"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        let aliases = AliasResult::simple(ctx);
        Ok(mem2reg(ctx, fun_id, &aliases))
    }
}

pub struct ConstFold;
impl FunctionPass for ConstFold {
    fn name(&self) -> &'static str {
        "const_fold"
    }
    fn description(&self) -> &'static str {
        "Fold pointer/integer arithmetic into literals"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        Ok(constant_fold_function(ctx, fun_id))
    }
}

pub struct Gvn;
impl FunctionPass for Gvn {
    fn name(&self) -> &'static str {
        "gvn"
    }
    fn description(&self) -> &'static str {
        "Global value numbering and constant folding"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        // Canonicalize pointer arithmetic into literals *before* building the alias
        // oracle, so it sees per-slot stack locations rather than collapsing them
        // onto `stack_base`.
        constant_fold_function(ctx, fun_id);
        let aliases = AliasResult::simple(ctx);
        gvn_function(ctx, fun_id, Some(&aliases));
        Ok(false)
    }
}

pub struct DeadStore;
impl FunctionPass for DeadStore {
    fn name(&self) -> &'static str {
        "dead_store"
    }
    fn description(&self) -> &'static str {
        "Remove dead register loads and overwritten flag stores"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        env: &PipelineEnv,
    ) -> Result<bool, String> {
        let aliases = AliasResult::simple(ctx);
        remove_dead_load_insns(ctx, fun_id, Some(&aliases), &env.cfg.dead_flag_regs);
        Ok(false)
    }
}

pub struct DeadLoad;
impl FunctionPass for DeadLoad {
    fn name(&self) -> &'static str {
        "dead_load"
    }
    fn description(&self) -> &'static str {
        "Remove dead memory loads"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        let aliases = AliasResult::simple(ctx);
        remove_dead_load_insns(ctx, fun_id, Some(&aliases), &[]);
        Ok(false)
    }
}

pub struct Dce;
impl FunctionPass for Dce {
    fn name(&self) -> &'static str {
        "dce"
    }
    fn description(&self) -> &'static str {
        "Remove unused pure instructions"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        let block_ids: Vec<_> = FunctionRef::from_id(ctx, fun_id)
            .blocks()
            .map(|b| b.id)
            .collect();
        for block_id in block_ids {
            remove_dead_insns(ctx, block_id);
        }
        Ok(false)
    }
}

pub struct Simplify;
impl FunctionPass for Simplify {
    fn name(&self) -> &'static str {
        "simplify"
    }
    fn description(&self) -> &'static str {
        "Merge straight-line basic blocks"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        simplify_cfg(ctx, fun_id);
        Ok(false)
    }
}

pub struct LowerStack;
impl FunctionPass for LowerStack {
    fn name(&self) -> &'static str {
        "lower_stack"
    }
    fn description(&self) -> &'static str {
        "Rewrite @stack_base literals back onto the real stack pointer"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        env: &PipelineEnv,
    ) -> Result<bool, String> {
        lower_stack(ctx, fun_id, env.sp_varnode);
        Ok(false)
    }
}

// ----- whole-program (module) milestones -------------------------------------

pub struct SeedClobbers;
impl Pass for SeedClobbers {
    fn name(&self) -> &'static str {
        "seed_clobbers"
    }
    fn description(&self) -> &'static str {
        "Seed each function's call-clobbered-register set from the lifted IR"
    }
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        set_all_call_clobbered_regs(ctx);
        Ok(false)
    }
}

pub struct ExternalSigs;
impl Pass for ExternalSigs {
    fn name(&self) -> &'static str {
        "external_sigs"
    }
    fn description(&self) -> &'static str {
        "Give known external (libc) functions signatures from their C prototypes"
    }
    fn run(&self, ctx: &mut Context, env: &PipelineEnv) -> Result<bool, String> {
        apply_all_external_signatures(ctx, &env.cfg.abi);
        Ok(false)
    }
}

pub struct Summaries;
impl Pass for Summaries {
    fn name(&self) -> &'static str {
        "summaries"
    }
    fn description(&self) -> &'static str {
        "Infer each function's input/clobber/saved summary and stack delta"
    }
    fn run(&self, ctx: &mut Context, env: &PipelineEnv) -> Result<bool, String> {
        set_all_function_summaries(ctx, env.sp_varnode);
        Ok(false)
    }
}

pub struct BindArgs;
impl Pass for BindArgs {
    fn name(&self) -> &'static str {
        "bind_args"
    }
    fn description(&self) -> &'static str {
        "Bind argument and per-call alias sets at every call site"
    }
    fn run(&self, ctx: &mut Context, env: &PipelineEnv) -> Result<bool, String> {
        bind_all_call_args(ctx, env.sp_varnode);
        Ok(false)
    }
}

// ----- registry --------------------------------------------------------------

/// A pass resolved from its TOML name, tagged by which scope it runs in.
pub enum RegisteredPass {
    Function(Box<dyn FunctionPass>),
    Module(Box<dyn Pass>),
}

/// Construct the pass registered under `name`, or `None` if unknown.
///
/// This `match` is the single place a new pass is registered.
pub fn make_pass(name: &str) -> Option<RegisteredPass> {
    use RegisteredPass::{Function as F, Module as M};
    Some(match name {
        "brighten" => F(Box::new(Brighten)),
        "mem2reg" => F(Box::new(Mem2Reg)),
        "const_fold" => F(Box::new(ConstFold)),
        "gvn" => F(Box::new(Gvn)),
        "dead_store" => F(Box::new(DeadStore)),
        "dead_load" => F(Box::new(DeadLoad)),
        "dce" => F(Box::new(Dce)),
        "simplify" => F(Box::new(Simplify)),
        "lower_stack" => F(Box::new(LowerStack)),
        "seed_clobbers" => M(Box::new(SeedClobbers)),
        "external_sigs" => M(Box::new(ExternalSigs)),
        "summaries" => M(Box::new(Summaries)),
        "bind_args" => M(Box::new(BindArgs)),
        _ => return None,
    })
}

/// Every registered pass name, for error messages when a pipeline names an
/// unknown pass.
pub const PASS_NAMES: &[&str] = &[
    "brighten",
    "mem2reg",
    "const_fold",
    "gvn",
    "dead_store",
    "dead_load",
    "dce",
    "simplify",
    "lower_stack",
    "seed_clobbers",
    "external_sigs",
    "summaries",
    "bind_args",
];
