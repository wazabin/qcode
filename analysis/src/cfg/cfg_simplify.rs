use crate::{ContextView, FunctionBody, FunctionPass, Outcome};

#[derive(Default)]
pub struct SimplifyCfg;

impl FunctionPass for SimplifyCfg {
    const NAME: &'static str = "simplify_cfg";

    fn description(&self) -> &'static str {
        "Merge straight-line blocks, drop empty forwarding blocks, fold degenerate branches"
    }

    fn run<'str>(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        _next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String> {
        let fid = body.id();
        Ok(Outcome::changed(simplify_cfg_body(body, cx.pass(), fid)))
    }
}

crate::register_function_pass!(SimplifyCfg);

// The transforms themselves live in `qcode_passes`: they are block-local,
// need no architecture configuration, and are used directly by the VM
// lifter. This module keeps only the pipeline-facing pass wrapper.
pub use qcode_passes::cfg::{absorb_straight_line, simplify_cfg, simplify_cfg_body};
