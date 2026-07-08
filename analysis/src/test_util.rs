//! Small helpers for exercising individual passes in unit tests.

use qcode::{
    context::Context,
    value::{FunctionId, RegisterId, VarnodeId},
};

use crate::{
    ArchConfig, CallingConvention, DynFunctionPass, FunctionPass, FunctionPassV2, PipelineEnv,
    V2Adapter,
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

/// Construct `P` via [`Default`] and run it once over `fun` with a [`dummy_env`],
/// returning whether the pass reported a change.
pub(crate) fn run_function_pass<P: FunctionPass>(
    ctx: &mut Context,
    fun: FunctionId,
) -> Result<bool, String> {
    P::default().run(ctx, fun, &dummy_env())
}

/// Run a [`FunctionPassV2`] once over `fun` through the real [`V2Adapter`] path
/// (check-out → run → check-in → effect replay) with a [`dummy_env`], so a unit
/// test exercises the same plumbing the sequential driver uses.
pub(crate) fn run_function_pass_v2<P: FunctionPassV2>(
    ctx: &mut Context,
    fun: FunctionId,
) -> Result<bool, String> {
    DynFunctionPass::run(&V2Adapter::<P>::default(), ctx, fun, &dummy_env())
}
