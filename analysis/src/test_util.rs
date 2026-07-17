//! Small helpers for exercising individual passes in unit tests.

use qcode::{
    context::Context,
    value::{FunctionId, RegisterId, VarnodeId},
};

use crate::{
    ArchConfig, CallingConvention, DynFunctionPass, FunctionPass, FunctionPassAdapter, PipelineEnv,
};

/// A throwaway [`PipelineEnv`] for passes that don't touch architecture state
/// (no real stack pointer or ABI). Arch-aware passes should build a real env via
/// [`PipelineEnv::new`] with a context that has a registered stack pointer.
pub(crate) fn dummy_env() -> PipelineEnv {
    PipelineEnv::from_parts(
        ArchConfig {
            stack_pointer: RegisterId::from(0usize),
            dead_flag_regs: Vec::new(),
            abi: CallingConvention::default(),
            os: qcode::context::TargetOs::Unknown,
            bitness: 64,
            assume_calling_convention: false,
        },
        VarnodeId::from(0usize),
    )
}

/// Run a [`FunctionPass`] once over `fun` through the real [`FunctionPassAdapter`] path
/// (split → run → barrier → effect replay) with a [`dummy_env`], so a unit
/// test exercises the same plumbing the sequential driver uses.
pub(crate) fn run_function_pass<P: FunctionPass + Send + Sync>(
    ctx: &mut Context,
    fun: FunctionId,
) -> Result<bool, String> {
    DynFunctionPass::run_with_analyses(
        &FunctionPassAdapter::<P>::default(),
        ctx,
        fun,
        &dummy_env(),
        &mut crate::AnalysisManager::default(),
    )
}

/// Borrow `fun`'s body in place, run `f` against its [`FunctionBody`] and a
/// `minted` buffer, then install any minted functions
/// — the same split/mint/install dance the driver performs, so a test can exercise
/// the outlining helpers directly and inspect the minted function afterwards.
/// Returns whatever `f` returns.
pub(crate) fn with_minting<'str, R>(
    ctx: &mut Context<'str>,
    fun: FunctionId,
    f: impl for<'a> FnOnce(
        crate::pipeline::ContextView<'a, 'str>,
        &mut crate::pipeline::FunctionBody<'str>,
        &mut u32,
        &mut Vec<crate::pipeline::Minted<'str>>,
    ) -> R,
) -> R {
    use crate::pipeline::{ContextSplit, Minted};
    let env = dummy_env();
    let (out, minted) = {
        let (bodies, view) = ctx.split(&env);
        let mut next_minted = 0;
        let mut minted: Vec<Minted<'str>> = Vec::new();
        let out = f(view, &mut bodies[fun], &mut next_minted, &mut minted);
        (out, minted)
    };
    crate::pipeline::install_minted_for_test(ctx, fun, minted);
    out
}
