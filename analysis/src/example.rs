//! Example pass.
use qcode::{
    context::Context,
    value::{Function, FunctionId},
};

use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct ExamplePass;

impl FunctionPass for ExamplePass {
    const NAME: &'static str = "example";

    fn description(&self) -> &'static str {
        "Example Pass, lists functions"
    }

    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        let function = Function::from_id_mut(ctx, fun_id);

        let _name = function.name().to_string();

        // TODO: log!(name)

        // This pass did not mutate the IR
        Ok(false)
    }
}

crate::register_function_pass!(ExamplePass);

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use crate::test_util::run_function_pass;

    #[test]
    fn named_functions() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                fn foo:
                    <entry>
                        return at 0x1000;
            "
        );

        run_function_pass::<ExamplePass>(&mut ctx, foo).unwrap();
    }
}
