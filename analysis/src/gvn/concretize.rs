//! The `concretize` module pass: gvn's three body-reading sub-passes, hoisted
//! out of the per-function GVN into a whole-program milestone.
//!
//! [`PureCall`](super::pure_call::PureCall),
//! [`EmulateMap`](super::emulate_map::EmulateMap), and
//! [`ArrayProject`](super::array_project::ArrayProject) all read (emulate/inline)
//! a *pure callee's* body — an interprocedural read the parallel-safe
//! [`FunctionPassV2`](crate::FunctionPassV2) contract forbids a function pass
//! from doing. They are therefore a module pass: it owns `&mut Context` and may
//! read any function's body directly.
//!
//! The pass drives the existing GVN dominator walk ([`super::walk`]) restricted
//! to exactly these three sub-passes over every non-external function, iterating
//! each function to a local fixpoint so the projection cascade (a `Range`-of-map
//! inlines a body whose `Range`-of-`enumerate` then projects and its
//! `Extract`-of-tuple folds) settles. The literal/inlined expressions it leaves
//! are folded by the next `gvn` in the stage.

use qcode::{context::Context, value::function::FunctionId};

use super::array_project::ArrayProject;
use super::emulate_map::EmulateMap;
use super::pure_call::PureCall;
use super::walk::{self, run_dominator_walk};

use crate::{Pass, PipelineEnv};

/// The three body-reading sub-passes, in the same relative order they held in
/// [`super::gvn_passes`] (emulate-map before array-project before pure-call). All
/// three run on the module host only, so the walk is instantiated at
/// `H = &mut Context`.
fn concretize_passes<'a, 'str>() -> Vec<Box<dyn walk::SubPass<'str, &'a mut Context<'str>>>> {
    vec![
        Box::new(EmulateMap),
        Box::new(ArrayProject),
        Box::new(PureCall),
    ]
}

/// Run the three body-reading sub-passes over `fun_id` to a local fixpoint.
/// Returns whether anything changed. These sub-passes carry no dominance state
/// and use no alias oracle, so the walk runs with `None` aliases.
pub(crate) fn concretize_function(mut ctx: &mut Context, fun_id: FunctionId) -> bool {
    let mut changed = false;
    while run_dominator_walk(&mut ctx, fun_id, &concretize_passes(), None) {
        changed = true;
    }
    changed
}

/// Whole-program pure-callee emulation/inlining: gvn's body-reading sub-passes.
#[derive(Default)]
pub struct Concretize;

impl Pass for Concretize {
    const NAME: &'static str = "concretize";
    fn description(&self) -> &'static str {
        "Emulate/inline pure callee bodies at constant-arg call/map/range sites"
    }
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        // Match `module(gvn)`'s eligibility: every non-external function, honoring
        // `--ignore`. Bodyless functions fall out for free (the walk returns
        // `false` when there is no root block).
        let fun_ids: Vec<FunctionId> = ctx
            .functions()
            .filter(|f| !f.is_external())
            .filter(|f| !ctx.is_function_ignored(f.address()))
            .map(|f| f.id)
            .collect();
        let mut changed = false;
        for fun_id in fun_ids {
            changed |= concretize_function(ctx, fun_id);
        }
        Ok(changed)
    }
}

crate::register_module_pass!(Concretize);
