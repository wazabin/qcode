//! Stack-escape fact seeding/harvesting for the checkpoint+replay driver. These
//! mirror `assume_call_returns`/`verify_assumptions` for the
//! `reads_unbounded_stack` / `frame_escapes_to_unbounded` function flags.

use qcode::{assumption::Proposition, context::Context, pass_scope, value::FunctionBody};

/// Seed the stack-escape facts proven in earlier checkpoint+replay rounds
/// (now sitting as known [`Proposition`]s on the freshly-cloned IR) back onto
/// the function flags, before the pipeline runs. `mem2reg` consumes
/// `frame_escapes_to_unbounded` (to keep an escaping caller's frame in memory)
/// and the summary/bind passes union onto `reads_unbounded_stack`. Mirrors
/// [`assume_call_returns`](crate::assume_call_returns) for these facts.
pub fn seed_stack_facts(ctx: &mut Context) {
    let facts: Vec<(Proposition, bool)> = ctx.known_facts().map(|(p, v, _pass)| (p, v)).collect();
    for (prop, value) in facts {
        if !value {
            continue;
        }
        match prop {
            Proposition::UnboundedStackReader(id) => {
                FunctionBody::from_id_mut(ctx, id).set_reads_unbounded_stack(true);
            }
            Proposition::FrameEscapingCaller(id) => {
                FunctionBody::from_id_mut(ctx, id).set_frame_escapes_to_unbounded(true);
            }
            _ => {}
        }
    }
}

/// Harvest the stack-escape facts the just-completed pipeline round established
/// into known [`Proposition`]s, for the checkpoint+replay driver to carry into
/// the next round. The flags only grow across rounds (the passes union onto the
/// seeded values), so replay converges.
///
/// Returns the number of newly-proven (novel) facts.
pub fn learn_stack_facts(ctx: &mut Context) -> usize {
    let _scope = pass_scope::enter("learn_stack_facts");
    let mut facts = Vec::new();
    for f in ctx.functions() {
        if f.reads_unbounded_stack() {
            facts.push(Proposition::UnboundedStackReader(f.id));
        }
        if f.frame_escapes_to_unbounded() {
            facts.push(Proposition::FrameEscapingCaller(f.id));
        }
    }
    facts
        .into_iter()
        .filter(|&prop| ctx.set_known(prop, true))
        .count()
}
