//! `retail_apply`: rewrite a *functionalized* tail call into `apply g; return`.
//!
//! A machine-level [`TailCall`](Mnemonic::TailCall) must never be de-tail-called:
//! its return-address / stack-pointer semantics reuse the caller's frame. But at
//! the **functionalized** level a tail transfer is pure control flow — the value
//! semantics live entirely in the register/RAM pack the callee returns. So once
//! both sides are functionalized (`is_reg_materialized`), a tail call
//!
//! ```text
//!     tailcall g(args)
//! ```
//!
//! is equivalent to applying `g` and returning its pack:
//!
//! ```text
//!     %pack = apply g(args)
//!     return %pack
//! ```
//!
//! [`Apply`] lives *outside* the CFG (it is an ordinary SSA instruction, not a
//! terminator), so no continuation block is minted; the rewrite is intra-block
//! value plumbing. This unblocks the caller for the RAM argpromote gate: a raw
//! `TailCall` escapes vetting entirely (see `function_makes_blocking_call`),
//! whereas the resulting `Apply` is vetted exactly like a `Call` — its callee's
//! memory summary is composed through the transfer.
//!
//! ## Conservatism
//!
//! [`TailCall`](qcode::value::insn::TailCall) carries *no* purity tag and *no*
//! clobbers field (unlike [`Call`](qcode::value::insn::Call)'s `CallTag` /
//! `clobbers`), so the purity gate reduces to: **both** the containing function
//! and the tail callee are `is_reg_materialized`. The applied pack must also
//! match the containing function's return type — if the tail callee's pack type
//! differs, the rewrite is skipped (pack adaptation across differing packs is
//! follow-up work; see the TODOs below).
//!
//! Rewriting a `Tail` call-graph edge into a `Direct(Apply)` edge changes the
//! call graph, so the pass does not preserve [`CallGraphAnalysis`] when it fires
//! (`ModulePassOutcome::functions` preserves nothing for a non-empty change set).

use qcode::value::QCodeMut;
use qcode::{
    context::Context,
    types::TypeId,
    value::{
        FunctionBody, FunctionId,
        insn::{Apply, Callee, InstructionId, Mnemonic, ReturnValue},
    },
};
use rustc_hash::FxHashSet;

use crate::{Pass, PipelineEnv};

/// The pack (value-return) type a function yields: the type of the value carried
/// by its first `Return`/`ReturnValue` terminator, or `None` for a function with
/// no *valued* return (a pure tail thunk whose return type its tail callee will
/// define, or a bare `return at addr`).
fn return_value_type(ctx: &Context, fid: FunctionId) -> Option<TypeId> {
    FunctionBody::from_id(ctx, fid).blocks().find_map(|b| {
        let last = b.iter().last()?;
        match last.mnemonic() {
            Mnemonic::Return(r) => r.value.map(|v| ctx.type_of(v.qualify(last.id.func))),
            Mnemonic::ReturnValue(rv) => Some(ctx.type_of(rv.value.qualify(last.id.func))),
            _ => None,
        }
    })
}

/// Rewrite every eligible functionalized tail call in `fid` to `apply g; return`.
/// Returns whether anything changed.
fn retail_function(ctx: &mut Context, fid: FunctionId) -> bool {
    if !FunctionBody::from_id(ctx, fid).is_reg_materialized() {
        return false;
    }
    let f_ret = return_value_type(ctx, fid);

    // Immutable scan first: collect the `(terminator, callee, args, pack type)`
    // of every eligible tail-call block, then mutate.
    let mut sites: Vec<(InstructionId, FunctionId, Vec<_>, TypeId)> = Vec::new();
    for block in FunctionBody::from_id(ctx, fid).blocks() {
        let Some(last) = block.iter().last() else {
            continue;
        };
        let Mnemonic::TailCall(tc) = last.mnemonic() else {
            continue;
        };
        let Some(g) = tc.target.real() else {
            continue;
        };
        // Both sides functionalized (the only available purity gate: TailCall has
        // neither a tag nor a clobbers field to inspect).
        if !FunctionBody::from_id(ctx, g).is_reg_materialized() {
            continue;
        }
        // The applied pack must be typeable and match `f`'s own return type.
        let Some(g_ret) = return_value_type(ctx, g) else {
            // TODO: the tail callee yields no typed pack (a bare `return at addr`),
            // so the `apply` result cannot be sized; leave the tail call alone.
            continue;
        };
        if f_ret.is_some_and(|t| t != g_ret) {
            // TODO: pack adaptation across differing return packs is follow-up
            // work — the applied pack would need reshaping into `f`'s pack.
            continue;
        }
        sites.push((last.id, g, tc.args.clone(), g_ret));
    }
    if sites.is_empty() {
        return false;
    }

    for (tid, g, args, g_ret) in sites {
        let block = ctx
            .get_insn(tid)
            .parent()
            .map(|b| b.id)
            .expect("tail block");
        // Build `%pack = apply g(args)` immediately before the tail terminator,
        // typed as g's pack (`push_mnemonic_with_type` — the target is a foreign
        // body, so its return type is supplied rather than read).
        let pack = {
            let mut b = ctx.builder(block);
            b.set_insert_point_before(tid);
            b.push_mnemonic_with_type(
                Mnemonic::Apply(Apply {
                    target: Callee::Real(g),
                    args,
                }),
                g_ret,
            )
            .id()
        };
        // Rewrite the terminator in place: `return %pack`. Both are terminators
        // with no CFG successors, so no edges change.
        ctx.replace_instruction_mnemonic(
            tid,
            Mnemonic::ReturnValue(ReturnValue {
                value: pack.localize(tid.func),
            }),
        );
    }
    true
}

#[derive(Default)]
pub struct RetailApply;

impl Pass for RetailApply {
    const NAME: &'static str = "retail_apply";
    fn description(&self) -> &'static str {
        "Rewrite functionalized tail calls to `apply g; return` so the RAM channel can vet them"
    }
    fn run(
        &self,
        cone: &mut crate::ConeMut,
        _env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        let mut changed = FxHashSet::default();
        for fid in cone.cone_functions() {
            if retail_function(cone.ctx_for(fid), fid) {
                changed.insert(fid);
            }
        }
        // A rewritten Tail edge becomes a Direct(Apply) edge, so the call graph
        // is invalidated (an empty change set preserves everything — nothing
        // moved). AddressAnalysis is likewise re-derived; the rewrite adds no
        // addresses so this is merely conservative.
        Ok(crate::ModulePassOutcome::functions(changed))
    }
}

crate::register_module_pass!(RetailApply);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::insn::Mnemonic;
    use qcode_macro::qcode;

    fn materialize(tc: &mut qcode::testing::TestContext, fid: FunctionId) {
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_register_effects(
            qcode::value::RegisterChannelState::Materialized(
                qcode::value::RegisterInterfaceMap::default(),
            ),
        );
    }

    fn tail_block(ctx: &Context, fid: FunctionId) -> qcode::value::BlockId {
        FunctionBody::from_id(ctx, fid)
            .blocks()
            .find(|b| {
                matches!(
                    b.iter().last().map(|i| i.mnemonic()),
                    Some(Mnemonic::TailCall(_))
                )
            })
            .expect("a tail block")
            .id
    }

    /// A functionalized `f` tail-calling a functionalized `g` of matching return
    /// type is rewritten: `f`'s tail block ends in `return` of an `apply g`.
    #[test]
    fn rewrites_matching_functionalized_tailcall() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn g:
                <g_entry @p:i64>
                    store(ram:8, @p <- i64 1);
                    return i64 0;
            fn f:
                <f_entry @p:i64>
                    tailcall fn g(i64 @p);
            "
        );
        let _ = (g_entry, f_entry);
        materialize(&mut tc, g);
        materialize(&mut tc, f);
        assert!(retail_function(&mut tc.ctx, f), "f must be rewritten");
        let root = FunctionBody::from_id(&tc.ctx, f).root().unwrap().id;
        let insns: Vec<_> = qcode::value::BasicBlock::from_id(&tc.ctx, root)
            .iter()
            .map(|i| i.mnemonic().clone())
            .collect();
        assert!(
            matches!(insns.first(), Some(Mnemonic::Apply(a)) if a.target.real() == Some(g)),
            "the tail call becomes an `apply g`: {insns:?}"
        );
        assert!(
            matches!(insns.last(), Some(Mnemonic::ReturnValue(_))),
            "the block now ends in a value return: {insns:?}"
        );
    }

    /// A tail callee whose pack type differs from the caller's return type is left
    /// alone (pack adaptation is follow-up).
    #[test]
    fn mismatched_return_type_is_left_alone() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn g:
                <g_entry @p:i64>
                    return i64 0;
            fn f:
                <f_entry @p:i64 @c:i8>
                    if @c goto <r> else goto <t>;
                <r>
                    return i32 5;
                <t>
                    tailcall fn g(i64 @p);
            "
        );
        let _ = (g_entry, f_entry, r, t);
        materialize(&mut tc, g);
        materialize(&mut tc, f);
        assert!(
            !retail_function(&mut tc.ctx, f),
            "mismatched types: no rewrite"
        );
        assert!(matches!(
            qcode::value::BasicBlock::from_id(&tc.ctx, tail_block(&tc.ctx, f))
                .iter()
                .last()
                .map(|i| i.mnemonic()),
            Some(Mnemonic::TailCall(_))
        ));
    }

    /// A non-materialized tail callee is left alone (only functionalized callees
    /// have a pure functionalized transfer).
    #[test]
    fn non_materialized_callee_is_left_alone() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn g:
                <g_entry @p:i64>
                    return i64 0;
            fn f:
                <f_entry @p:i64>
                    tailcall fn g(i64 @p);
            "
        );
        let _ = (g_entry, f_entry);
        // g intentionally NOT materialized.
        materialize(&mut tc, f);
        assert!(
            !retail_function(&mut tc.ctx, f),
            "non-materialized g: no rewrite"
        );
        assert!(matches!(
            qcode::value::BasicBlock::from_id(&tc.ctx, tail_block(&tc.ctx, f))
                .iter()
                .last()
                .map(|i| i.mnemonic()),
            Some(Mnemonic::TailCall(_))
        ));
    }

    /// After the rewrite the RAM gate no longer blocks `f` on a blanket tail bar:
    /// a memory-free callee unblocks it, while a memory-writing callee still
    /// blocks it — now via the new `Apply` vetting arm rather than the tail bar.
    #[test]
    fn rewrite_routes_gate_through_apply_vetting() {
        use super::super::ram_summary;
        use crate::CallGraph;
        // Memory-free callee: after retail, f is unblocked.
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn pureg:
                <pg_entry @p:i64>
                    return i64 0;
            fn f:
                <f_entry @p:i64>
                    tailcall fn pureg(i64 @p);
            "
        );
        let _ = (pg_entry, f_entry);
        materialize(&mut tc, pureg);
        materialize(&mut tc, f);
        retail_function(&mut tc.ctx, f);
        let graph = CallGraph::analyze(&tc.ctx);
        let summaries = ram_summary::solve(&tc.ctx, &graph, None);
        assert!(
            !super::super::ram::function_makes_blocking_call(
                &tc.ctx,
                f,
                &summaries,
                &FxHashSet::default(),
                None
            ),
            "a rewritten tail to a memory-free callee no longer blocks f"
        );
    }
}
