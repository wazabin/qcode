//! Example pass. The smallest [`FunctionPassV2`]: it reads its own function's
//! interface and mutates nothing.
use crate::{FunctionBody, FunctionPassV2, ModuleView};

#[derive(Default)]
pub struct ExamplePass;

impl FunctionPassV2 for ExamplePass {
    const NAME: &'static str = "example";

    fn description(&self) -> &'static str {
        "Example Pass, lists functions"
    }

    fn run(&self, _m: &ModuleView, f: &mut FunctionBody) -> Result<bool, String> {
        let _name = f.function().name.to_string();

        // TODO: log!(name)

        // This pass did not mutate the IR
        Ok(false)
    }
}

crate::register_function_pass_v2!(ExamplePass);

#[cfg(test)]
mod tests {
    use qcode::context::Context;
    use qcode_macro::qcode;

    use super::*;
    use crate::test_util::run_function_pass_v2;

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

        run_function_pass_v2::<ExamplePass>(&mut ctx, foo).unwrap();
    }
}
