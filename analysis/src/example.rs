//! Example pass. The smallest [`FunctionPass`]: it reads its own function's
//! interface and mutates nothing.
use crate::{FunctionBody, FunctionPass, ModuleView};

#[derive(Default)]
pub struct ExamplePass;

impl FunctionPass for ExamplePass {
    const NAME: &'static str = "example";

    fn description(&self) -> &'static str {
        "Example Pass, lists functions"
    }

    fn run<'str>(
        &self,
        _m: &ModuleView<'_, 'str>,
        f: &mut FunctionBody<'str>,
    ) -> Result<bool, String> {
        let _name = f.function().name.to_string();

        // TODO: log!(name)

        // This pass did not mutate the IR
        Ok(false)
    }
}

crate::register_function_pass!(ExamplePass);

#[cfg(test)]
mod tests {
    use qcode::context::Context;
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
