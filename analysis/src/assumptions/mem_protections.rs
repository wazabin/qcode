//! Establishing the binary's memory protections.
//!
//! Until this runs, the lifter treats every readable byte as potentially
//! executable (default r/x), because it cannot prove non-executability on its
//! own. This pass marks the context's per-segment protection flags (loaded from
//! the binary format's section/segment permissions) as authoritative, so
//! [`Context::assume_executable`] narrows to the real flags — a discovered target
//! in a non-executable region is then skipped instead of decoded as phantom code.
//!
//! Executability is modeled as a [`Proposition::ExecutableMemory`]: an optimistic
//! assumption while protections are unknown, a proven fact once they are.
//!
//! [`Proposition::ExecutableMemory`]: qcode::assumption::Proposition::ExecutableMemory

use qcode::context::Context;

use crate::{Pass, PipelineEnv};

/// Mark the context's per-segment protection flags as authoritative. Idempotent.
/// Run this *before* lifting consults executability (verify-then-use), so the
/// permissive default narrows to the binary's real protections.
pub fn establish_memory_protections(ctx: &mut Context) {
    ctx.mark_protections_known();
}

#[derive(Default)]
pub struct MemoryProtections;

impl Pass for MemoryProtections {
    const NAME: &'static str = "memory_protections";
    fn description(&self) -> &'static str {
        "Establish the binary's real memory protections (narrow the lifter's default r/x)"
    }
    fn run(
        &self,
        ctx: &mut Context,
        _env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        establish_memory_protections(ctx);
        Ok(crate::ModulePassOutcome::module())
    }
}

crate::register_module_pass!(MemoryProtections);
