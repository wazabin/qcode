//! Verify each function's stamped `written_spaces` over-approximates its body.
//!
//! `written_spaces` is a *witnessed bound*: `Some(spaces)` promises a call to
//! the function cannot store outside `spaces`, and `mem_forward`'s call-prune
//! forwards values across calls on that promise. The bound is stamped by
//! `seed_written_spaces` and must stay an over-approximation while later
//! passes mutate bodies — deleting stores shrinks the truth (fine); a pass
//! that *adds* a store, an escape, or an unbounded callee behind a stale
//! `Some` bound makes the prune unsound. Dirty-scoped: only changed functions
//! are re-derived.
//!
//! Three checks per in-scope function with a `Some(ws)` stamp (a `None` stamp
//! is the conservative answer and always passes):
//!
//! 1. every shared non-register space its body stores to is in `ws`;
//! 2. it contains no unbounded escape — `CallInd`, unresolved `BranchInd`, or
//!    a direct-like call/tail-call to an unresolved target;
//! 3. every resolved direct-like callee (`Call` or `TailCall`) nests. A callee
//!    with a `Bounded(cs)` stamp must satisfy `cs ⊆ ws`. A callee recorded
//!    *unbounded* (⊤) over a bounded caller is exactly the stale state this rule
//!    exists to catch, and the tri-state `written_spaces_state` now distinguishes
//!    a deliberately recorded `Unbounded` from a never-stamped `Unstamped` fresh
//!    mint. Any resolved callee stamped `Unbounded` — internal or external — is
//!    flagged under a bounded caller; a genuinely `Unstamped` callee (a freshly
//!    minted, not-yet-seeded function) is still skipped.
//!
//! Dirty-scoping: `written_spaces` is transitive, so growing a callee's
//! write-set stales every caller's stamp even when the caller is out of scope.
//! Like the sibling call-driven rules ([`super::pure_reg_call_args`],
//! [`super::materialized_interface`]) the checked set is expanded with the
//! callers of in-scope functions.

use rustc_hash::FxHashSet;

use qcode::{
    context::Context,
    space::{Space, SpaceType},
    value::{FunctionBody, FunctionId, insn::Mnemonic},
};

pub fn verify_written_spaces(ctx: &Context) -> Vec<String> {
    verify_written_spaces_scoped(ctx, super::Scope::All)
}

/// The functions to check: everything in `scope`, plus — for a scoped run —
/// every out-of-scope function with a direct-like call/tail-call into the scope,
/// since growing an in-scope callee's write-set stales those callers' stamps.
fn functions_to_check(ctx: &Context, scope: super::Scope<'_>) -> Vec<FunctionId> {
    let mut ids: Vec<FunctionId> = scope.function_ids(ctx);
    let super::Scope::Functions(set) = scope else {
        return ids; // `All` already covers every function.
    };
    let mut extra: FxHashSet<FunctionId> = FxHashSet::default();
    for insn in ctx.instructions() {
        if set.contains(&insn.id.func) {
            continue; // already in scope
        }
        let callee = match insn.mnemonic() {
            Mnemonic::Call(c) => c.target.real(),
            Mnemonic::TailCall(t) => t.target.real(),
            Mnemonic::Apply(a) => a.target.real(),
            _ => None,
        };
        if callee.is_some_and(|c| set.contains(&c)) {
            extra.insert(insn.id.func);
        }
    }
    ids.extend(extra.into_iter().filter(|id| !set.contains(id)));
    ids
}

pub(crate) fn verify_written_spaces_scoped(ctx: &Context, scope: super::Scope<'_>) -> Vec<String> {
    let mut out = Vec::new();
    for fid in functions_to_check(ctx, scope) {
        let f = FunctionBody::from_id(ctx, fid);
        let Some(ws) = f.written_spaces() else {
            continue;
        };
        let name = f.name().to_string();
        for block in FunctionBody::from_id(ctx, fid).blocks() {
            for insn in block.iter() {
                match insn.mnemonic() {
                    Mnemonic::Store(s) => {
                        let Some(shared) = s.space.shared() else {
                            continue;
                        };
                        if matches!(Space::from_id(ctx, shared).ty, SpaceType::Register) {
                            continue;
                        }
                        if !ws.contains(&shared) {
                            out.push(format!(
                                "{name}: body stores to space {shared:?} outside its \
                                 witnessed written_spaces bound {ws:?}"
                            ));
                        }
                    }
                    Mnemonic::CallInd(_) => {
                        out.push(format!(
                            "{name}: has an indirect call but a bounded written_spaces \
                             {ws:?} — an unknown callee may write any space"
                        ));
                    }
                    Mnemonic::BranchInd(_) => {
                        out.push(format!(
                            "{name}: has an unresolved indirect branch but a bounded \
                             written_spaces {ws:?} — an escape may write any space"
                        ));
                    }
                    Mnemonic::Call(c) => {
                        out.extend(check_callee(ctx, &name, ws, c.target.real()));
                    }
                    Mnemonic::TailCall(t) => {
                        out.extend(check_callee(ctx, &name, ws, t.target.real()));
                    }
                    // An `Apply` (e.g. a tail call rewritten to `apply g; return`)
                    // can carry a real memory effect, so its callee nests too.
                    // (`Map`/`Scan` stay unvetted: their bodies are pure by
                    // construction, never storing to a shared space.)
                    Mnemonic::Apply(a) => {
                        out.extend(check_callee(ctx, &name, ws, a.target.real()));
                    }
                    _ => {}
                }
            }
        }
    }
    out
}

/// Check one resolved direct-like call/tail-call against the caller's bound
/// `ws`. An unresolved target (`None`) is an escape; a `Bounded` callee must
/// nest; an `Unbounded` callee (internal or external) flags; a genuinely
/// `Unstamped` fresh mint is skipped (see the module doc).
fn check_callee(
    ctx: &Context,
    name: &str,
    ws: &[qcode::space::SpaceId],
    callee: Option<FunctionId>,
) -> Option<String> {
    use qcode::value::WrittenSpaces;
    let Some(callee) = callee else {
        return Some(format!(
            "{name}: calls an unresolved target but has a bounded written_spaces {ws:?}"
        ));
    };
    let cf = FunctionBody::from_id(ctx, callee);
    match cf.written_spaces_state() {
        WrittenSpaces::Bounded(cs) => cs.iter().find(|s| !ws.contains(s)).map(|bad| {
            format!(
                "{name}: callee {} may write space {bad:?} outside the caller's \
                 witnessed bound {ws:?}",
                cf.name(),
            )
        }),
        // Deliberately recorded unbounded under a bounded caller: the stale state
        // the rule catches. Now flagged for internal callees too — the tri-state
        // distinguishes this from an unstamped fresh mint.
        WrittenSpaces::Unbounded => Some(format!(
            "{name}: callee {} is recorded unbounded but the caller has a bounded \
             written_spaces {ws:?}",
            cf.name(),
        )),
        // A defensive fallback: `SeedWrittenSpaces` now stamps externals too (a
        // prototyped one gets a bounded argmem write-set, an un-prototyped one
        // `Unbounded`), so a stamped external takes the `Bounded`/`Unbounded`
        // arms above. An external that reaches here *unstamped* never ran through
        // the seed pass; it is genuinely unbounded — never a fresh mint — so it
        // still flags.
        WrittenSpaces::Unstamped if cf.is_external() => Some(format!(
            "{name}: external callee {} is unbounded but the caller has a bounded \
             written_spaces {ws:?}",
            cf.name(),
        )),
        // A never-stamped internal fresh mint: skipped, re-seeded on the next run.
        WrittenSpaces::Unstamped => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::FunctionId;
    use qcode_macro::qcode;

    #[test]
    fn stale_bound_is_flagged_and_none_passes() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn w:
                <entry @p:i64>
                    store(ram:8, @p <- i64 1);
                    return at i64 0;
            "
        );
        let _ = entry;
        // Unstamped (None): conservative, passes.
        assert!(verify_written_spaces(&tc.ctx).is_empty());
        // A correct bound passes.
        let ram = tc.ctx.shared.default_space;
        set_ws(&mut tc, w, Some(vec![ram]));
        assert!(verify_written_spaces(&tc.ctx).is_empty());
        // A stale empty bound under a real store is flagged.
        set_ws(&mut tc, w, Some(vec![]));
        let diags = verify_written_spaces(&tc.ctx);
        assert_eq!(diags.len(), 1, "{diags:?}");
    }

    fn set_ws(
        tc: &mut qcode::testing::TestContext,
        fid: FunctionId,
        ws: Option<Vec<qcode::space::SpaceId>>,
    ) {
        qcode::value::FunctionBody::from_id_mut(&mut tc.ctx, fid).set_written_spaces(ws);
    }

    /// FIX 4a: a `TailCall` to a resolved callee whose bounded write-set exceeds
    /// the caller's bound is flagged, just like a `Call`.
    #[test]
    fn tailcall_callee_over_bound_is_flagged() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn callee:
                <c_entry @p:i64>
                    store(ram:8, @p <- i64 1);
                    return at i64 0;
            fn caller:
                <k_entry>
                    tailcall fn callee();
            "
        );
        let _ = (c_entry, k_entry);
        let ram = tc.ctx.shared.default_space;
        set_ws(&mut tc, callee, Some(vec![ram]));
        // Caller's stamp omits ram though its tail callee writes it.
        set_ws(&mut tc, caller, Some(vec![]));
        let diags = verify_written_spaces(&tc.ctx);
        assert!(
            diags.iter().any(|d| d.contains("callee callee may write")),
            "a tail-call callee over the caller's bound must be flagged: {diags:?}"
        );
    }

    /// FIX 4b: a resolved direct callee that is an *external* recorded unbounded
    /// (`None`) under a bounded caller is flagged — the stale state the rule
    /// exists to catch, distinguishable from a fresh internal mint via
    /// `is_external`.
    #[test]
    fn unbounded_external_callee_over_bounded_caller_is_flagged() {
        use qcode::value::insn::{Call, CallTag, Callee, Mnemonic};
        use qcode::value::{BasicBlock, FunctionBody, QCodeMut};
        let mut tc = qcode::testing::TestContext::new();
        let ext = FunctionBody::make_external(&mut tc.ctx, 0x9000, Some("ext".into())).id;
        qcode!(
            tc.ctx,
            "
            fn caller:
                <k_entry>
                    call fn caller();
                <k_cont>
                    return at i64 0;
            "
        );
        let _ = (k_entry, k_cont);
        // Redirect the placeholder call to the external.
        let call_id = BasicBlock::from_id(&tc.ctx, k_entry)
            .iter()
            .find(|i| matches!(i.mnemonic(), Mnemonic::Call(_)))
            .unwrap()
            .id;
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: Callee::Real(ext),
                args: vec![],
                clobbers: vec![],
                tag: CallTag::Opaque,
            }),
        );
        // External keeps its default unbounded (None) write-set; caller is bounded.
        set_ws(&mut tc, caller, Some(vec![]));
        let diags = verify_written_spaces(&tc.ctx);
        assert!(
            diags.iter().any(|d| d.contains("external callee")),
            "an unbounded external under a bounded caller must be flagged: {diags:?}"
        );
    }

    /// ITEM 2: an *internal* (non-external) callee stamped unbounded (`None` via
    /// `set_written_spaces`, now recorded as `WrittenSpaces::Unbounded`) under a
    /// bounded caller is now caught — the tri-state distinguishes it from a
    /// never-stamped fresh mint, which stays skipped.
    #[test]
    fn unbounded_internal_callee_over_bounded_caller_is_flagged() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn callee:
                <c_entry @p:i64>
                    store(ram:8, @p <- i64 1);
                    return at i64 0;
            fn caller:
                <k_entry>
                    call <callee>;
                <k_cont>
                    return at i64 0;
            "
        );
        let _ = (c_entry, k_entry, k_cont);
        // A fresh (unstamped) internal callee under a bounded caller is skipped.
        set_ws(&mut tc, caller, Some(vec![]));
        assert!(
            verify_written_spaces(&tc.ctx).is_empty(),
            "an unstamped fresh-mint callee must not be flagged"
        );
        // Once the callee is deliberately stamped unbounded, it is caught.
        set_ws(&mut tc, callee, None);
        let diags = verify_written_spaces(&tc.ctx);
        assert!(
            diags.iter().any(|d| d.contains("is recorded unbounded")),
            "an internal callee recorded unbounded under a bounded caller must be flagged: {diags:?}"
        );
    }

    /// STAGE 3: `SeedWrittenSpaces` stamps a prototyped external with its argmem
    /// write-set ({ram} for a mutable-pointer external), so a caller bounded over
    /// it nests cleanly — no false "external is unbounded" flag.
    #[test]
    fn prototyped_external_written_set_nests_under_caller() {
        use qcode::value::insn::{Call, CallTag, Callee, Mnemonic};
        use qcode::value::{ArgMemKind, BasicBlock, ExternArgmem, FunctionBody, QCodeMut};
        let mut tc = qcode::testing::TestContext::new();
        let memset = FunctionBody::make_external(&mut tc.ctx, 0x9000, Some("memset".into())).id;
        FunctionBody::from_id_mut(&mut tc.ctx, memset).set_argmem(ExternArgmem {
            params: vec![ArgMemKind::MutPtr, ArgMemKind::NonPtr, ArgMemKind::NonPtr],
            variadic: false,
        });
        qcode!(
            tc.ctx,
            "
            fn caller:
                <k_entry @p:i64>
                    call fn caller();
                <k_cont>
                    return at i64 0;
            "
        );
        let _ = (k_entry, k_cont);
        let p = FunctionBody::from_id(&tc.ctx, caller)
            .root()
            .unwrap()
            .params()
            .next()
            .unwrap()
            .id();
        let call_id = BasicBlock::from_id(&tc.ctx, k_entry)
            .iter()
            .find(|i| matches!(i.mnemonic(), Mnemonic::Call(_)))
            .unwrap()
            .id;
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: Callee::Real(memset),
                args: vec![p.localize(call_id.func)],
                clobbers: vec![],
                tag: CallTag::RegPure,
            }),
        );
        crate::calls::set_all_written_spaces(&mut tc.ctx);

        use qcode::value::WrittenSpaces;
        let ram = tc.ctx.shared.default_space;
        assert!(
            matches!(
                FunctionBody::from_id(&tc.ctx, memset).written_spaces_state(),
                WrittenSpaces::Bounded(cs) if cs == [ram]
            ),
            "a mutable-pointer external is stamped Bounded({{ram}})"
        );
        assert!(
            verify_written_spaces(&tc.ctx).is_empty(),
            "a caller bounded over a prototyped external must not be flagged: {:?}",
            verify_written_spaces(&tc.ctx)
        );
    }

    /// FIX 4c: written_spaces is transitive — growing an in-scope callee's
    /// write-set stales an out-of-scope caller's stamp. The dirty-scoped run must
    /// sweep callers of in-scope functions and catch it.
    #[test]
    fn caller_sweep_catches_stale_caller_when_callee_in_scope() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn callee:
                <c_entry @p:i64>
                    store(ram:8, @p <- i64 1);
                    return at i64 0;
            fn caller:
                <k_entry>
                    call <callee>;
                <k_cont>
                    return at i64 0;
            "
        );
        let _ = (c_entry, k_entry, k_cont);
        let ram = tc.ctx.shared.default_space;
        set_ws(&mut tc, callee, Some(vec![ram]));
        // Stale bound: predates the callee gaining the ram store.
        set_ws(&mut tc, caller, Some(vec![]));
        // Scope only the callee — the caller is out of scope.
        let scope: FxHashSet<FunctionId> = [callee].into_iter().collect();
        let diags = verify_written_spaces_scoped(&tc.ctx, super::super::Scope::Functions(&scope));
        assert!(
            diags.iter().any(|d| d.contains("callee callee may write")),
            "the caller sweep must catch the out-of-scope stale caller: {diags:?}"
        );
    }
}
