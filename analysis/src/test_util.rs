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
        },
        VarnodeId::from(0usize),
    )
}

/// Run a [`FunctionPass`] once over `fun` through the real [`FunctionPassAdapter`] path
/// (check-out → run → check-in → effect replay) with a [`dummy_env`], so a unit
/// test exercises the same plumbing the sequential driver uses.
pub(crate) fn run_function_pass<P: FunctionPass + Send + Sync>(
    ctx: &mut Context,
    fun: FunctionId,
) -> Result<bool, String> {
    DynFunctionPass::run(&FunctionPassAdapter::<P>::default(), ctx, fun, &dummy_env())
}

/// Check `fun` out, run `f` against its [`FunctionBody`] (carrying two reserved
/// minting ids), then install any minted functions and check it back in — the
/// same check-out/mint/install dance the driver performs, so a test can exercise
/// the outlining helpers directly and inspect the minted function afterwards.
/// Returns whatever `f` returns.
pub(crate) fn with_minting<'str, R>(
    ctx: &mut Context<'str>,
    fun: FunctionId,
    f: impl for<'a> FnOnce(
        crate::pipeline::ContextView<'a, 'str>,
        &mut crate::pipeline::FunctionBody<'_, 'str>,
    ) -> R,
) -> R {
    use crate::pipeline::{ContextSplit, FunctionBody};
    let reserved: Vec<FunctionId> = (0..2).map(|_| ctx.push_sentinel_function()).collect();
    let env = dummy_env();
    let (out, minted) = {
        let (bodies, view) = ctx.split(&env);
        let mut body = FunctionBody::new(fun, &mut bodies[fun], reserved);
        let out = f(view, &mut body);
        let (_effects, minted, _unused) = body.into_parts();
        (out, minted)
    };
    crate::pipeline::install_minted_for_test(ctx, minted);
    out
}
