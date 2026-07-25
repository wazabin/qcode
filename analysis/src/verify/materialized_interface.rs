//! Verify a materialized function's root params still agree with its published
//! register interface map.
//!
//! `argpromote_materialize` establishes the two together (`registers.rs`
//! `materialize_interface`): it pushes one by-value root param per
//! `RegisterEffects::inputs` entry, then records the same vector as
//! `RegisterInterfaceMap::inputs`, so param slot `i` binds `inputs[i]`. Every
//! consumer of the interface — the regpure call-site rewrite, the emulator's
//! implicit zero-arg convention, and `verify::pure_reg_call_args` — assumes that
//! correspondence and none of them re-derives it.
//!
//! Nothing checked it. That gap is not merely a missed diagnosis: because
//! `pure_reg_call_args::interface_param_sizes` treats the *root params* as
//! authoritative for a bodied callee, a function that loses its root params while
//! keeping a populated `inputs` map reports an arity of zero, and the breakage
//! surfaces as a call-site violation blamed on whichever pass last pulled those
//! callers into verify scope — not on the pass that dropped the params. This rule
//! catches it at the callee, where the defect actually is.

use qcode::value::{BasicBlock, FunctionBody, RegisterChannelState, Varnode};
use qcode::{context::Context, value::FunctionId};

/// A materialized function's root params must start with its `inputs` mapping:
/// `param[i]` binds register `inputs[i]` and carries that register's width.
///
/// Containment, not equality — the RAM channel (`argpromote`)
/// appends by-value memory params *after* the register block, so the root is a
/// superset. Only the leading register prefix is constrained.
pub fn verify_materialized_interfaces(ctx: &Context) -> Vec<String> {
    verify_materialized_interfaces_scoped(ctx, super::Scope::All)
}

pub(crate) fn verify_materialized_interfaces_scoped(
    ctx: &Context,
    scope: super::Scope<'_>,
) -> Vec<String> {
    let mut out = Vec::new();
    for fid in scope.function_ids(ctx) {
        out.extend(check_function(ctx, fid));
    }
    out
}

fn check_function(ctx: &Context, fid: FunctionId) -> Option<String> {
    let function = FunctionBody::from_id(ctx, fid);
    let RegisterChannelState::Materialized(map) = &function.effects().register else {
        return None;
    };
    // A bodyless external has no root block; its `inputs` map *is* the interface
    // and there is nothing to cross-check it against.
    let root = function.root().map(|b| b.id)?;

    let params: Vec<usize> = BasicBlock::from_id(ctx, root)
        .params()
        .map(|p| p.size())
        .collect();
    let name = &ctx.interfaces[fid].name;

    if params.len() < map.inputs.len() {
        return Some(format!(
            "materialized interface desync: `{name}` has {} root params but its interface map \
             declares {} register inputs (root params must contain the register block as a \
             prefix; a pass dropped or replaced the root without updating the map)",
            params.len(),
            map.inputs.len(),
        ));
    }

    let mismatch = map.inputs.iter().enumerate().find_map(|(i, &vn)| {
        let expected = Varnode::from_id(ctx, vn).size();
        (params[i] != expected).then_some((i, expected, params[i]))
    })?;
    let (i, expected, actual) = mismatch;
    Some(format!(
        "materialized interface desync: `{name}` root param {i} is {actual} bytes but its \
         interface map binds it to a {expected}-byte register",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::RegisterInterfaceMap;

    /// Losing the root params while the map still declares inputs is reported
    /// against the callee — the case that previously only surfaced, misattributed,
    /// as a call-site arity error.
    #[test]
    fn dropped_root_params_are_reported_at_the_callee() {
        let (mut ctx, vn) = fixture();
        let f = materialized(&mut ctx, "victim", &[vn], 0);

        let diags = verify_materialized_interfaces(&ctx);
        assert_eq!(diags.len(), 1, "expected exactly one diagnostic: {diags:?}");
        assert!(
            diags[0].contains("`victim`") && diags[0].contains("0 root params"),
            "diagnostic must name the callee and its arity: {}",
            diags[0],
        );
        let _ = f;
    }

    /// The register block is a *prefix*, not the whole interface: extra params
    /// appended by the RAM channel are legal and must not be flagged.
    #[test]
    fn ram_channel_params_appended_after_the_register_block_are_allowed() {
        let (mut ctx, vn) = fixture();
        materialized(&mut ctx, "grown", &[vn], 3);
        assert!(
            verify_materialized_interfaces(&ctx).is_empty(),
            "appended by-value params must not trip the prefix check",
        );
    }

    /// An exact match verifies clean.
    #[test]
    fn matching_interface_verifies_clean() {
        let (mut ctx, vn) = fixture();
        materialized(&mut ctx, "healthy", &[vn], 1);
        assert!(verify_materialized_interfaces(&ctx).is_empty());
    }

    fn fixture() -> (Context<'static>, qcode::value::VarnodeId) {
        let mut ctx = Context::new();
        let space = ctx.shared.default_space;
        let vn = Varnode::make(&mut ctx, 0, 8, space).id;
        (ctx, vn)
    }

    /// A materialized function whose map declares `inputs`, with `params` root
    /// params actually present (so the two can be made to disagree).
    fn materialized(
        ctx: &mut Context<'static>,
        name: &'static str,
        inputs: &[qcode::value::VarnodeId],
        params: usize,
    ) -> FunctionId {
        let f = FunctionBody::make(ctx, name.into()).unwrap().id;
        let root = BasicBlock::make(ctx, f).id;
        FunctionBody::from_id_mut(ctx, f).set_root(root).unwrap();
        for _ in 0..params {
            BasicBlock::from_id_mut(ctx, root).push_param(8);
        }
        FunctionBody::from_id_mut(ctx, f).set_register_effects(RegisterChannelState::Materialized(
            RegisterInterfaceMap {
                inputs: inputs.to_vec(),
                outputs: vec![],
                returns: 0,
            },
        ));
        f
    }
}
