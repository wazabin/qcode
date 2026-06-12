//! Handles jump tables.
//!
//! The first version of this pass looks for indirect jumps that are built
//! using a constant and possibly an offset.
//!
//! We perform a quick value analysis on the offset to try and determine the
//! jump table's size.
//!
//! TODO: if we can't we should scan the addresses in the jump table and try and
//! find address "close to" one another

use qcode::{
    context::Context,
    value::{Function, FunctionId},
};

use crate::{FunctionPass, PipelineEnv};

pub struct HandleJumpTables;

impl Default for HandleJumpTables {
    fn default() -> Self {
        Self
    }
}

impl FunctionPass for HandleJumpTables {
    const NAME: &'static str = "handle_jump_tables";

    fn description(&self) -> &'static str {
        "Attempts to resolve jump tables"
    }

    // Returns true if a change was made, so it can be used in a `repeat_until` stage if needed.
    // This pass doesn't need to be, but it's good practice to track changes in case you later add more functionality
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        env: &PipelineEnv,
    ) -> Result<bool, String> {
        let mut function = Function::from_id_mut(ctx, fun_id);

        Ok(false)
    }
}

crate::register_function_pass!(HandleJumpTables);

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use crate::test_util::run_function_pass;
}
