//! The opt-in [`Proposition::AssumeCallingConvention`] hypothesis: a
//! whole-program, tracked guess that every indirect (`CallInd`) and
//! unresolved-direct call obeys the module's calling convention.
//!
//! When active it gives such a call an *educated-guess* register effect —
//! reads = all convention argument registers, writes = the caller-saved set —
//! instead of the conservative clobbers-all. This is a **deliberate,
//! controllable unsoundness**: an indirect callee may violate the ABI. It is
//! **off by default** ([`ArchConfig::assume_calling_convention`]) and the user
//! opts in.
//!
//! It is a **downstream refinement only**. The install pass records the
//! whole-program hypothesis and caches its [`AssumedCallEffect`] on the context;
//! the mem2reg / alias register classifier (`classify_call_reg_effect`) then
//! consults it so callee-saved registers survive across such calls. It does
//! **not** feed back into the argpromote effect-summary fixpoint, which keeps
//! modelling `CallInd` as `Some(empty)` in the register channel (see
//! `reg_summary::RegChannel::indirect_call_effects`).
//!
//! No verifier: the hypothesis is never proven, so it is not discharged in v1;
//! the checkpoint+replay net never acts on it. Like [`super::arg_frame`] it is
//! reinstalled fresh each pipeline round (an `Assumed` truth does not survive a
//! replay clone, and the cached effect is explicitly not serialized).

use qcode::{
    assumption::{AssumedCallEffect, Proposition},
    context::Context,
    pass_scope,
    value::{FunctionId, VarnodeId},
};

use crate::{Pass, PipelineEnv};

/// Compute the [`AssumedCallEffect`] the hypothesis assigns to indirect /
/// unresolved calls from `env`'s calling convention: `reads` = every argument
/// register (integer, widest view, ∪ SSE), `writes` = the caller-saved set. The
/// stack pointer is excluded from both (consistent with the rest of the register
/// channel); the frame pointer is callee-saved and so never appears in either
/// ABI set to begin with.
fn convention_effect(env: &PipelineEnv) -> AssumedCallEffect {
    let abi = &env.cfg.abi;
    let excluded = |vn: VarnodeId| Some(vn) == env.sp_varnode;

    let mut reads: Vec<VarnodeId> = abi
        .int_args
        .iter()
        .filter_map(|gp| gp.widths.iter().max_by_key(|(w, _)| *w).map(|(_, v)| *v))
        .chain(abi.sse_args.iter().copied())
        .filter(|&vn| !excluded(vn))
        .collect();
    reads.sort_unstable();
    reads.dedup();

    let mut writes: Vec<VarnodeId> = abi
        .caller_saved
        .iter()
        .copied()
        .filter(|&vn| !excluded(vn))
        .collect();
    writes.sort_unstable();
    writes.dedup();

    AssumedCallEffect { reads, writes }
}

/// *Make* pass — when opted in, record [`Proposition::AssumeCallingConvention`]
/// and cache the convention's effect on the context. Returns `true` if the
/// truth-map entry was newly recorded this call (the cached effect is refreshed
/// unconditionally so it is present whenever the hypothesis is active).
pub fn assume_calling_convention(ctx: &mut Context, env: &PipelineEnv) -> bool {
    let _scope = pass_scope::enter("assume_calling_convention");
    if !env.cfg.assume_calling_convention {
        return false;
    }
    ctx.set_assumed_call_convention(Some(convention_effect(env)));
    // `assume_true` is idempotent (returns `true` only for a fresh record or an
    // existing same-polarity entry); a pass outcome must report only an actual
    // truth-map mutation as changed, so gate on the entry not existing yet.
    let novel = ctx.truth(Proposition::AssumeCallingConvention).is_none()
        && ctx.assume_true(Proposition::AssumeCallingConvention);
    if novel {
        qcode::pass_log!(
            debug,
            "assumed calling convention for indirect/unresolved calls"
        );
    }
    novel
}

/// Module pass wrapper for [`assume_calling_convention`].
///
/// Placed immediately after the register effect-calculation stage
/// (`argpromote-registers`) so the downstream `mem2reg` / alias classifier sees
/// the hypothesis. Inert unless [`ArchConfig::assume_calling_convention`] is set.
#[derive(Default)]
pub struct AssumeCallingConvention;

impl Pass for AssumeCallingConvention {
    const NAME: &'static str = "assume_calling_convention";
    fn description(&self) -> &'static str {
        "Assume indirect/unresolved calls obey the calling convention (opt-in, unsound)"
    }
    fn run(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        _targets: &[FunctionId],
    ) -> Result<crate::ModulePassOutcome, String> {
        let changed = assume_calling_convention(ctx, env);
        // The hypothesis refines only mem2reg/alias register clobbering; it adds
        // no CFG/address structure, so both cached global analyses survive.
        Ok(crate::ModulePassOutcome::module_if(changed)
            .preserving_global::<crate::CallGraphAnalysis>()
            .preserving_global::<crate::AddressAnalysis>())
    }
}

crate::register_module_pass!(AssumeCallingConvention);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArchConfig, CallingConvention, GpReg};
    use qcode::testing::TestContext;

    fn env_with_abi(tc: &TestContext, abi: CallingConvention, sp: VarnodeId) -> PipelineEnv {
        let cfg = ArchConfig {
            stack_pointer: qcode::value::RegisterId::from(0usize),
            dead_flag_regs: Vec::new(),
            abi,
            os: qcode::context::TargetOs::Unknown,
            bitness: 64,
            assume_calling_convention: true,
        };
        let _ = tc;
        PipelineEnv::from_parts(cfg, sp)
    }

    #[test]
    fn off_by_default_records_nothing() {
        let mut tc = TestContext::new();
        let abi = CallingConvention {
            int_args: vec![GpReg {
                widths: vec![(8, tc.r1)],
            }],
            caller_saved: vec![tc.r0],
            ..CallingConvention::default()
        };
        // Same ABI but the opt-in flag stays false.
        let cfg = ArchConfig {
            stack_pointer: qcode::value::RegisterId::from(0usize),
            dead_flag_regs: Vec::new(),
            abi,
            os: qcode::context::TargetOs::Unknown,
            bitness: 64,
            assume_calling_convention: false,
        };
        let env = PipelineEnv::from_parts(cfg, tc.r3);
        assert!(!assume_calling_convention(&mut tc.ctx, &env));
        assert_eq!(tc.ctx.truth(Proposition::AssumeCallingConvention), None);
        assert!(tc.ctx.assumed_call_convention().is_none());
    }

    #[test]
    fn installs_effect_and_excludes_sp() {
        let mut tc = TestContext::new();
        let (r0, r0_lo32, r1, r2, sp) = (tc.r0, tc.r0_lo32, tc.r1, tc.r2, tc.r3);
        let abi = CallingConvention {
            // r0 is an argument register exposed at 4- and 8-byte views; sp is
            // (wrongly) also listed as an arg register to prove it is filtered.
            int_args: vec![
                GpReg {
                    widths: vec![(4, r0_lo32), (8, r0)],
                },
                GpReg {
                    widths: vec![(8, sp)],
                },
            ],
            caller_saved: vec![r1, r2, sp],
            ..CallingConvention::default()
        };
        let env = env_with_abi(&tc, abi, sp);

        assert!(assume_calling_convention(&mut tc.ctx, &env));
        assert_eq!(
            tc.ctx
                .truth(Proposition::AssumeCallingConvention)
                .map(|t| t.value),
            Some(true)
        );
        // Idempotent: repeating an accepted assumption reports no change.
        assert!(!assume_calling_convention(&mut tc.ctx, &env));

        let eff = tc.ctx.assumed_call_convention().expect("effect installed");
        // reads = widest arg-register view (r0, not r0_lo32); sp excluded.
        assert!(eff.reads.contains(&r0), "widest arg register present");
        assert!(!eff.reads.contains(&r0_lo32), "narrow view not used");
        assert!(!eff.reads.contains(&sp), "SP never in the read set");
        // writes = caller-saved minus sp.
        assert!(eff.writes.contains(&r1) && eff.writes.contains(&r2));
        assert!(!eff.writes.contains(&sp), "SP never in the write set");
    }
}
