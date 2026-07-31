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
/// Containment, not equality — the RAM channel (`argpromote`, `promote_stack_args`)
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

    // A register is either packed or derived, never both — a consumer resolving
    // one must have exactly one place to look, and a register appearing in both
    // means a pass added a projection without dropping the pack slot it
    // replaces. Checked ahead of the root-param rules so a bodyless external's
    // projections are covered too.
    if let Some(diag) = check_projections(ctx, fid, map) {
        return Some(diag);
    }

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

/// The [`projections`](qcode::value::RegisterInterfaceMap::projections)
/// invariants: each derived register is absent from `outputs`, and no register
/// is derived twice.
fn check_projections(
    ctx: &Context,
    fid: FunctionId,
    map: &qcode::value::RegisterInterfaceMap,
) -> Option<String> {
    if map.projections.is_empty() {
        return None;
    }
    let name = &ctx.interfaces[fid].name;

    if let Some(derived) = map
        .projections
        .iter()
        .find(|d| map.outputs.contains(&d.register))
    {
        return Some(format!(
            "materialized interface desync: `{name}` lists register `{}` as both a return-pack \
             output and a derived projection; a register is packed or derived, never both",
            render_varnode(ctx, derived.register),
        ));
    }

    let mut seen = rustc_hash::FxHashSet::default();
    if let Some(derived) = map.projections.iter().find(|d| !seen.insert(d.register)) {
        return Some(format!(
            "materialized interface desync: `{name}` derives register `{}` twice; each derived \
             register has exactly one projection",
            render_varnode(ctx, derived.register),
        ));
    }

    None
}

/// A varnode's architectural name for a diagnostic, falling back to its label
/// and then its raw id.
fn render_varnode(ctx: &Context, id: qcode::value::VarnodeId) -> String {
    let varnode = Varnode::from_id(ctx, id);
    match (varnode.name(), varnode.label()) {
        (Some(name), _) => name.to_string(),
        (None, Some(label)) => format!("v{label}"),
        (None, None) => format!("varnode:{id}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::{DerivedOutput, RegisterInterfaceMap, insn::Callee};

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

    /// A register may be packed or derived, never both. This is the shape a pass
    /// produces by recording a projection and forgetting to drop the pack slot it
    /// replaces, which would leave two disagreeing answers for one register.
    #[test]
    fn a_register_both_packed_and_derived_is_reported() {
        let (mut ctx, vn) = fixture();
        let f = materialized(&mut ctx, "doubled", &[vn], 1);
        let proj = projection(&mut ctx, "doubled$vn");
        set_interface(&mut ctx, f, |map| {
            map.outputs = vec![vn];
            map.returns = 1;
            map.projections = vec![DerivedOutput {
                register: vn,
                projection: proj,
            }];
        });

        let diags = verify_materialized_interfaces(&ctx);
        assert_eq!(diags.len(), 1, "expected exactly one diagnostic: {diags:?}");
        assert!(
            diags[0].contains("`doubled`") && diags[0].contains("packed or derived"),
            "diagnostic must name the callee and the rule: {}",
            diags[0],
        );
    }

    /// Each derived register has exactly one projection.
    #[test]
    fn a_register_derived_twice_is_reported() {
        let (mut ctx, vn) = fixture();
        let f = materialized(&mut ctx, "twice", &[vn], 1);
        let (a, b) = (
            projection(&mut ctx, "twice$a"),
            projection(&mut ctx, "twice$b"),
        );
        set_interface(&mut ctx, f, |map| {
            map.projections = vec![
                DerivedOutput {
                    register: vn,
                    projection: a,
                },
                DerivedOutput {
                    register: vn,
                    projection: b,
                },
            ];
        });

        let diags = verify_materialized_interfaces(&ctx);
        assert_eq!(diags.len(), 1, "expected exactly one diagnostic: {diags:?}");
        assert!(
            diags[0].contains("`twice`") && diags[0].contains("twice"),
            "diagnostic must name the callee and the rule: {}",
            diags[0],
        );
    }

    /// A register that is derived and *absent* from the pack is the whole point
    /// of the record — the shape the stack-pointer axis produces — and verifies
    /// clean. Being an `inputs` entry as well is fine: SP is read and derived.
    #[test]
    fn a_derived_register_absent_from_the_pack_verifies_clean() {
        let (mut ctx, vn) = fixture();
        let f = materialized(&mut ctx, "derived", &[vn], 1);
        let proj = projection(&mut ctx, "derived$vn");
        set_interface(&mut ctx, f, |map| {
            map.projections = vec![DerivedOutput {
                register: vn,
                projection: proj,
            }];
        });
        assert!(verify_materialized_interfaces(&ctx).is_empty());
    }

    /// Overwrite `f`'s materialized interface map through `edit`.
    fn set_interface(
        ctx: &mut Context<'static>,
        f: FunctionId,
        edit: impl FnOnce(&mut RegisterInterfaceMap),
    ) {
        let RegisterChannelState::Materialized(mut map) =
            FunctionBody::from_id(ctx, f).effects().register.clone()
        else {
            panic!("fixture must be materialized");
        };
        edit(&mut map);
        FunctionBody::from_id_mut(ctx, f)
            .set_register_effects(RegisterChannelState::Materialized(map));
    }

    /// A stand-in projection function to link to.
    fn projection(ctx: &mut Context<'static>, name: &'static str) -> Callee {
        Callee::Real(FunctionBody::make(ctx, name.into()).unwrap().id)
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
                projections: Vec::new(),
            },
        ));
        f
    }
}
