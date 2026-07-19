//! Whole-program IR verifier.
//!
//! Each invariant rule lives in its own module and exposes a small, testable
//! function. The verifier is meant to run *between passes* (enabled by the
//! `QCODE_VERIFY` environment variable; see [`enabled`]) so that when a pass
//! corrupts the IR the failure is reported against the exact pass that produced
//! it, rather than surfacing far downstream as a confusing symptom.
//!
//! Between-pass runs are *scoped*: the driver knows which functions a pass
//! touched, and a pass can only break invariants involving those functions, so
//! re-verifying the rest of the module (already verified after the previous
//! pass) is redundant. [`Scope::Functions`] restricts every rule to the changed
//! set; the interface rule additionally re-checks call sites *into* the set,
//! since rewriting a callee's interface can invalidate unchanged callers.

mod arena_integrity;
mod block_terminators;
mod bool_typing;
mod call_edges;
mod dangling_refs;
mod intra_function_ssa;
mod materialized_interface;
mod pointer_spaces;
mod pure_function;
mod pure_reg_call_args;
mod users_map;

pub use arena_integrity::verify_body_arena_integrity;
pub use block_terminators::verify_block_terminators;
pub use bool_typing::verify_bool_typing;
pub use call_edges::verify_call_edges;
pub use dangling_refs::verify_no_dangling_refs;
pub use intra_function_ssa::verify_intra_function_ssa;
pub use materialized_interface::verify_materialized_interfaces;
pub use pointer_spaces::verify_pointer_spaces;
pub use pure_function::{PureFunctionViolation, verify_pure_functions};
pub use pure_reg_call_args::{PureRegCallArgsViolation, verify_pure_reg_call_args};
pub use users_map::verify_users_map;

use std::sync::OnceLock;

use rustc_hash::FxHashSet;

use qcode::{
    context::Context,
    value::{FunctionId, Instruction, insn::InstructionId},
};

use crate::{Pass, PipelineEnv};

/// The functions one verifier invocation inspects.
#[derive(Clone, Copy)]
pub enum Scope<'a> {
    /// Every function in the module.
    All,
    /// Only these functions — the set a pass just changed. Sound between passes
    /// because the rest of the module verified clean after the previous pass.
    Functions(&'a FxHashSet<FunctionId>),
}

impl Scope<'_> {
    pub(crate) fn contains(&self, function: FunctionId) -> bool {
        match self {
            Scope::All => true,
            Scope::Functions(set) => set.contains(&function),
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Scope::Functions(set) if set.is_empty())
    }

    /// The live functions in scope, in the context's stable order.
    pub(crate) fn function_ids(&self, ctx: &Context<'_>) -> Vec<FunctionId> {
        let mut ids = ctx.function_ids();
        if let Scope::Functions(set) = self {
            ids.retain(|id| set.contains(id));
        }
        ids
    }

    /// The live instructions of the in-scope functions, in stable logical-ID
    /// order (mirrors `Context::instructions`).
    pub(crate) fn instructions<'str, 'ctx>(
        &self,
        ctx: &'ctx Context<'str>,
    ) -> impl Iterator<Item = qcode::value::InstructionRef<'str, 'ctx>> {
        let mut ids: Vec<InstructionId> = self
            .function_ids(ctx)
            .into_iter()
            .flat_map(|id| qcode::value::FunctionBody::from_id(ctx, id).instruction_ids())
            .collect();
        ids.sort_unstable();
        ids.into_iter().map(move |id| Instruction::from_id(ctx, id))
    }
}

/// Run every verifier rule and return all diagnostics as human-readable strings.
/// An empty result means the IR is well-formed by the checks we have.
pub fn verify(ctx: &Context<'_>) -> Vec<String> {
    verify_scoped(ctx, Scope::All)
}

/// [`verify`], restricted to `scope` (see [`Scope`]). With
/// [`Scope::Functions`], only invariants involving the named functions are
/// checked — the between-pass fast path.
pub fn verify_scoped(ctx: &Context<'_>, scope: Scope<'_>) -> Vec<String> {
    if scope.is_empty() {
        return Vec::new();
    }
    let mut diagnostics = arena_integrity::verify_body_arena_integrity_scoped(ctx, scope);
    if !diagnostics.is_empty() {
        return diagnostics;
    }
    diagnostics.extend(block_terminators::verify_block_terminators_scoped(
        ctx, scope,
    ));
    diagnostics.extend(call_edges::verify_call_edges_scoped(ctx, scope));
    diagnostics.extend(dangling_refs::verify_no_dangling_refs_scoped(ctx, scope));
    diagnostics.extend(intra_function_ssa::verify_intra_function_ssa_scoped(
        ctx, scope,
    ));
    diagnostics.extend(users_map::verify_users_map_scoped(ctx, scope));
    diagnostics.extend(pointer_spaces::verify_pointer_spaces_scoped(ctx, scope));
    diagnostics.extend(bool_typing::verify_bool_typing_scoped(ctx, scope));
    diagnostics.extend(materialized_interface::verify_materialized_interfaces_scoped(ctx, scope));
    diagnostics.extend(
        pure_reg_call_args::verify_pure_reg_call_args_scoped(ctx, scope)
            .into_iter()
            .map(|v| v.diagnostic(ctx)),
    );
    diagnostics.extend(
        pure_function::verify_pure_functions_scoped(ctx, scope)
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

/// Run [`verify_scoped`] and panic if it fails, naming `pass` — the pass that
/// just ran. A no-op unless [`enabled`]. Called by the pipeline driver after
/// each pass with the functions that pass changed, so the first invariant break
/// is pinned to its culprit without re-verifying the untouched rest of the
/// module.
pub fn verify_after(ctx: &Context, pass: &str, scope: Scope<'_>) {
    if !enabled() {
        return;
    }
    let violations = verify_scoped(ctx, scope);
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
    fn run(
        &self,
        ctx: &mut Context,
        _env: &PipelineEnv,
        _targets: &[qcode::value::FunctionId],
    ) -> Result<crate::ModulePassOutcome, String> {
        let violations = verify(ctx);
        if violations.is_empty() {
            Ok(crate::ModulePassOutcome::default())
        } else {
            Err(format!(
                "IR verification failed:\n  - {}",
                violations.join("\n  - ")
            ))
        }
    }
}

crate::register_module_pass!(Verify);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::{BasicBlock, BlockId, FunctionBody};
    use qcode_macro::qcode;

    /// A module with `f` corrupted (its goto targets a removed block) and `g` clean.
    fn corrupted_f() -> (Context<'static>, FunctionId, FunctionId) {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <entry>
                goto <target>;
            <target>
                return at i64 0;
            fn g:
            <entry>
                return at i64 0;
            "
        );
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let edge = *ctx.block(entry).edges.iter().next().expect("edge");
        let target = BlockId::new(f, ctx.edge(f, edge).to);
        BasicBlock::from_id_mut(&mut ctx, target).delete();
        (ctx, f, g)
    }

    #[test]
    fn scoped_verify_catches_in_scope_corruption() {
        let (ctx, f, _) = corrupted_f();
        let scope: FxHashSet<_> = [f].into_iter().collect();
        let diagnostics = verify_scoped(&ctx, Scope::Functions(&scope));
        assert!(
            diagnostics
                .iter()
                .any(|d| d.contains("targets removed block")),
            "{diagnostics:#?}"
        );
    }

    #[test]
    fn scoped_verify_skips_out_of_scope_functions() {
        // The between-pass contract: a pass that only changed `g` cannot have
        // broken `f`, so scoping to `g` skips f's (pre-existing) corruption.
        let (ctx, _, g) = corrupted_f();
        let scope: FxHashSet<_> = [g].into_iter().collect();
        assert_eq!(
            verify_scoped(&ctx, Scope::Functions(&scope)),
            Vec::<String>::new()
        );
    }

    #[test]
    fn empty_scope_verifies_nothing() {
        let (ctx, _, _) = corrupted_f();
        let scope = FxHashSet::default();
        assert!(verify_scoped(&ctx, Scope::Functions(&scope)).is_empty());
    }
}
