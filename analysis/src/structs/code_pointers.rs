//! `infer_code_pointers`: type the target of an indirect call as a code pointer.
//!
//! A value used as the `ptr` of a [`CallInd`] is, by construction, a function
//! address. This pass types such a value as [`TypeRepr::CodePointer`] when the
//! value is a stored-type carrier we can annotate — a root block **param**
//! (`call [@cb]`: a callback parameter) or a **varnode** (`call [ECX]`). SSA
//! instruction results have derived, not stored, types and are left alone.
//!
//! This is the honest-typing prototype from GLOBALS_AS_VARNODES.md's follow-up:
//! a param called through is a function pointer, not an `i32`. Typing it is the
//! precondition for later interprocedural resolution/exploration of the target
//! (propagate a caller's constant code address into the param, then
//! `discover_code`), which this pass does **not** yet do.

use qcode::{
    types::TypeRepr,
    value::{FunctionBody, ValueId, insn::Mnemonic},
};

use crate::{Pass, PipelineEnv};

pub struct InferCodePointers;

impl Default for InferCodePointers {
    fn default() -> Self {
        Self
    }
}

impl Pass for InferCodePointers {
    const NAME: &'static str = "infer_code_pointers";

    fn description(&self) -> &'static str {
        "Types the target of an indirect call as a code pointer"
    }

    fn run(
        &self,
        cone: &mut crate::ConeMut,
        _env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        let targets = cone.cone_functions();
        // Collect every indirect-call target, qualified to a context ValueId.
        let mut targets_ptrs: Vec<ValueId> = Vec::new();
        for &fid in &targets {
            if FunctionBody::from_id(cone.ctx(), fid).is_external() {
                continue;
            }
            for block in FunctionBody::from_id(cone.ctx(), fid).blocks() {
                for insn in block.iter() {
                    if let Mnemonic::CallInd(ci) = insn.mnemonic() {
                        targets_ptrs.push(ci.ptr.qualify(fid));
                    }
                }
            }
        }

        let mut changed = false;
        for ptr in targets_ptrs {
            // Already a code pointer? Idempotent no-op.
            if matches!(
                cone.ctx().shared.types.get(cone.ctx().type_of(ptr)).repr(),
                TypeRepr::CodePointer { .. }
            ) {
                continue;
            }
            let size = cone.ctx().shared.types.size_of(cone.ctx().type_of(ptr));
            if size == 0 {
                continue;
            }
            let code_ptr = cone.ctx().shared.types.get_or_make_code_pointer(size);
            match ptr {
                ValueId::BlockParam(pid) => {
                    let block_func = pid.func;
                    cone.ctx_for(block_func).block_param_mut(pid).type_id = code_ptr;
                    changed = true;
                }
                ValueId::Varnode(vn) => {
                    // Varnode types are program-global — a cone-free shared write,
                    // not a per-function body edit, so it goes through the dedicated
                    // accessor rather than borrowing some function's `ctx_for`.
                    cone.set_varnode_type(vn, code_ptr);
                    changed = true;
                }
                // Instruction results / literals carry derived or intrinsic
                // types; not annotated here.
                _ => {}
            }
        }

        Ok(crate::ModulePassOutcome::module_if(changed))
    }
}

crate::register_module_pass!(InferCodePointers);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::context::Context;
    use qcode::value::BasicBlock;
    use qcode_macro::qcode;

    #[test]
    fn callback_param_typed_code_pointer() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry @cb:i64>
                    call [@cb]();
                <cont>
                    return at i64 0;
            "
        );

        let env = PipelineEnv::headless(&ctx);
        let out = InferCodePointers
            .run(&mut crate::ConeMut::full(&mut ctx), &env)
            .unwrap();
        assert!(out.changed(), "pass should retype the called param");

        let root = FunctionBody::from_id(&ctx, f).root().unwrap().id;
        let block = BasicBlock::from_id(&ctx, root);
        let ty = block.params().next().unwrap().type_id();
        assert!(
            matches!(
                ctx.shared.types.get(ty).repr(),
                TypeRepr::CodePointer { .. }
            ),
            "called param should be a code pointer, got {}",
            ctx.shared.types.type_name(ty),
        );
    }
}
