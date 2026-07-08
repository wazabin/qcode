//! Whole-program IR verifier.
//!
//! Each invariant rule lives in its own module and exposes a small, testable
//! function. The verifier is meant to run *between passes* (enabled by the
//! `QCODE_VERIFY` environment variable; see [`enabled`]) so that when a pass
//! corrupts the IR the failure is reported against the exact pass that produced
//! it, rather than surfacing far downstream as a confusing symptom.

mod block_terminators;
mod bool_typing;
mod dangling_refs;
mod intra_function_ssa;
mod pointer_spaces;
mod pure_function;
mod pure_reg_call_args;
mod users_map;

pub use block_terminators::verify_block_terminators;
pub use bool_typing::verify_bool_typing;
pub use dangling_refs::verify_no_dangling_refs;
pub use intra_function_ssa::verify_intra_function_ssa;
pub use pointer_spaces::verify_pointer_spaces;
pub use pure_function::{PureFunctionViolation, verify_pure_functions};
pub use pure_reg_call_args::{PureRegCallArgsViolation, verify_pure_reg_call_args};
pub use users_map::verify_users_map;

use std::sync::OnceLock;

use qcode::context::Context;

use crate::{Pass, PipelineEnv};

/// Run every verifier rule and return all diagnostics as human-readable strings.
/// An empty result means the IR is well-formed by the checks we have.
pub fn verify(ctx: &Context<'_>) -> Vec<String> {
    let mut diagnostics = Vec::new();
    diagnostics.extend(verify_block_terminators(ctx));
    diagnostics.extend(verify_no_dangling_refs(ctx));
    diagnostics.extend(verify_intra_function_ssa(ctx));
    diagnostics.extend(verify_users_map(ctx));
    diagnostics.extend(verify_pointer_spaces(ctx));
    diagnostics.extend(verify_bool_typing(ctx));
    diagnostics.extend(
        verify_pure_reg_call_args(ctx)
            .into_iter()
            .map(|v| v.diagnostic(ctx)),
    );
    diagnostics.extend(
        verify_pure_functions(ctx)
            .into_iter()
            .map(|v| v.diagnostic(ctx)),
    );
    diagnostics
}

/// Back-compat alias for [`verify`].
pub fn verify_ir(ctx: &Context) -> Vec<String> {
    verify(ctx)
}

/// Whether between-pass verification is enabled. Reads the `QCODE_VERIFY`
/// environment variable once (any non-empty value enables it), so `gui` /
/// `headless` opt in with e.g. `QCODE_VERIFY=1 just gui <binary>`.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("QCODE_VERIFY").is_ok_and(|v| !v.is_empty()))
}

/// Run [`verify`] and panic if it fails, naming `pass` — the pass that just ran.
/// A no-op unless [`enabled`]. Called by the pipeline driver after each pass so the
/// first invariant break is pinned to its culprit.
pub fn verify_after(ctx: &Context, pass: &str) {
    if !enabled() {
        return;
    }
    let violations = verify(ctx);
    assert!(
        violations.is_empty(),
        "IR verification failed after pass `{pass}`:\n  - {}",
        violations.join("\n  - ")
    );
}

/// A no-op-by-default module pass that fails if the IR violates an invariant.
/// Registered so it can also be dropped into a pipeline TOML stage explicitly;
/// the between-passes hook ([`verify_after`]) is the primary entry point.
#[derive(Default)]
pub struct Verify;

impl Pass for Verify {
    const NAME: &'static str = "verify";
    fn description(&self) -> &'static str {
        "Check IR structural invariants (fails on violation)"
    }
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        let violations = verify(ctx);
        if violations.is_empty() {
            Ok(false)
        } else {
            Err(format!(
                "IR verification failed:\n  - {}",
                violations.join("\n  - ")
            ))
        }
    }
}

crate::register_module_pass!(Verify);
