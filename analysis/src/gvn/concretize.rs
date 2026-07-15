//! Transitional no-op registration. The three former body-reading sub-passes
//! are now independent module passes; the next commit removes this legacy name
//! after replacing its remaining pipeline sites.

use qcode::context::Context;

use crate::{Pass, PipelineEnv};

#[derive(Default)]
pub struct Concretize;

impl Pass for Concretize {
    const NAME: &'static str = "concretize";

    fn description(&self) -> &'static str {
        "Legacy no-op; replaced by explicit GVN module passes"
    }

    fn run(&self, _ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        Ok(false)
    }
}

crate::register_module_pass!(Concretize);
