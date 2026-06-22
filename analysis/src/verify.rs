//! IR invariant verification.
//!
//! `verify_ir` walks the module and returns a list of structural-invariant
//! violations. It is meant to run *between passes* (enabled by the `QCODE_VERIFY`
//! environment variable; see [`enabled`]) so that when a pass corrupts the IR the
//! failure is reported against the exact pass that produced it, rather than
//! surfacing far downstream as a confusing symptom.
//!
//! This is intentionally a small placeholder: today it only checks that every
//! basic block ends in a terminator. Add `check_*` helpers as new invariants are
//! worth enforcing (e.g. no use of a value defined in a non-dominating /
//! deleted block, single-entry blocks, well-typed aggregates, …).

use std::sync::OnceLock;

use qcode::{context::Context, value::Function};

use crate::{Pass, PipelineEnv};

/// Returns every structural-invariant violation found in `ctx`, as human-readable
/// strings. An empty result means the IR is well-formed by the checks we have.
pub fn verify_ir(ctx: &Context) -> Vec<String> {
    let mut violations = Vec::new();
    check_blocks_end_with_terminator(ctx, &mut violations);
    violations
}

/// Every basic block must end in a terminator (branch / cbranch / return / …).
/// A block that is empty, or whose last instruction is an ordinary value op, has
/// fall-through control flow with no defined successor — a malformed CFG.
fn check_blocks_end_with_terminator(ctx: &Context, out: &mut Vec<String>) {
    for fid in ctx.function_ids() {
        for block in Function::from_id(ctx, fid).iter() {
            match block.iter().last() {
                None => out.push(format!("fn {fid:?} block {:?} is empty (no terminator)", block.id)),
                Some(last) if !last.mnemonic().is_terminator() => out.push(format!(
                    "fn {fid:?} block {:?} does not end in a terminator (last op: `{}`)",
                    block.id,
                    last.mnemonic().opcode()
                )),
                Some(_) => {}
            }
        }
    }
}

/// Whether between-pass verification is enabled. Reads the `QCODE_VERIFY`
/// environment variable once (any non-empty value enables it), so `gui` /
/// `headless` opt in with e.g. `QCODE_VERIFY=1 just gui <binary>`.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("QCODE_VERIFY").is_ok_and(|v| !v.is_empty()))
}

/// Run [`verify_ir`] and panic if it fails, naming `pass` — the pass that just ran.
/// A no-op unless [`enabled`]. Called by the pipeline driver after each pass so the
/// first invariant break is pinned to its culprit.
pub fn verify_after(ctx: &Context, pass: &str) {
    if !enabled() {
        return;
    }
    let violations = verify_ir(ctx);
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
        let violations = verify_ir(ctx);
        if violations.is_empty() {
            Ok(false)
        } else {
            Err(format!("IR verification failed:\n  - {}", violations.join("\n  - ")))
        }
    }
}

crate::register_module_pass!(Verify);
