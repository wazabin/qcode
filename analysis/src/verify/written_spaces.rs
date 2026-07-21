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
//!    a direct-like call to an unresolved target;
//! 3. every resolved direct-like callee with its own `Some(cs)` stamp nests:
//!    `cs ⊆ ws`. (A callee stamped `None` — e.g. a freshly minted, not yet
//!    re-seeded function — is not flagged here: mints are pure and the next
//!    seed run re-establishes the bound; flagging would false-positive every
//!    outline.)

use qcode::{
    context::Context,
    space::{Space, SpaceType},
    value::{FunctionBody, insn::Mnemonic},
};

pub fn verify_written_spaces(ctx: &Context) -> Vec<String> {
    verify_written_spaces_scoped(ctx, super::Scope::All)
}

pub(crate) fn verify_written_spaces_scoped(ctx: &Context, scope: super::Scope<'_>) -> Vec<String> {
    let mut out = Vec::new();
    for fid in scope.function_ids(ctx) {
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
                    Mnemonic::Call(c) => match c.target.real() {
                        None => out.push(format!(
                            "{name}: calls an unresolved target but has a bounded \
                             written_spaces {ws:?}"
                        )),
                        Some(callee) => {
                            if let Some(cs) = FunctionBody::from_id(ctx, callee).written_spaces()
                                && let Some(bad) = cs.iter().find(|s| !ws.contains(s))
                            {
                                out.push(format!(
                                    "{name}: callee {} may write space {bad:?} outside \
                                     the caller's witnessed bound {ws:?}",
                                    FunctionBody::from_id(ctx, callee).name(),
                                ));
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
    }
    out
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
}
