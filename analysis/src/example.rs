//! Example pass. The smallest [`FunctionPass`]: it reads its own function's
//! interface and mutates nothing.
use crate::{ContextView, FunctionBody, FunctionPass, Outcome};

#[derive(Default)]
pub struct ExamplePass;

impl FunctionPass for ExamplePass {
    const NAME: &'static str = "example";

    fn description(&self) -> &'static str {
        "Example Pass, lists functions"
    }

    fn run<'str>(
        &self,
        f: &mut FunctionBody<'str>,
        _m: ContextView<'_, 'str>,
        _next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String> {
        let _name = _m.interface(f.id()).name.to_string();

        // TODO: log!(name)

        // This pass did not mutate the IR
        Ok(Outcome::unchanged())
    }
}

crate::register_function_pass!(ExamplePass);

#[cfg(test)]
mod tests {
    use qcode::context::Context;
    use wazabin_qcode_macro::qcode;

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
