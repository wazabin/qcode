//! Identify function return targets as code pointers.

use std::borrow::Cow;

use qcode::{
    types::TypeRepr,
    value::{
        FunctionBody, Instruction, QCodeMut, Renameable, ValueId,
        insn::{Mnemonic, Return},
    },
};

use crate::{Pass, PipelineEnv};

fn rename_ra<'str, 'ctx, T>(value: &mut T) -> Result<bool, String>
where
    T: Renameable<'str, 'ctx>,
{
    if value.name().is_some_and(|name| name.starts_with("ra")) {
        return Ok(false);
    }
    for suffix in 0usize.. {
        let name = if suffix == 0 {
            Cow::Borrowed("ra")
        } else {
            Cow::Owned(format!("ra{suffix}"))
        };
        match value.rename(name) {
            Ok(()) => return Ok(true),
            Err(error)
                if matches!(
                    error.ty,
                    qcode::error::ErrorTy::DuplicateName(_)
                        | qcode::error::ErrorTy::NameAlreadyExists(_)
                ) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    unreachable!("the return-address suffix space is unbounded")
}

#[derive(Default)]
pub struct IdentifyReturnAddress;

impl Pass for IdentifyReturnAddress {
    const NAME: &'static str = "identify_return_address";

    fn description(&self) -> &'static str {
        "Type return targets as code pointers and name them ra"
    }

    fn run(
        &self,
        cone: &mut crate::ConeMut,
        _env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        let mut targets = Vec::new();
        for fid in cone.cone_functions() {
            for block in FunctionBody::from_id(cone.ctx(), fid).blocks() {
                for insn in block.iter() {
                    if let Mnemonic::Return(ret) = insn.mnemonic() {
                        targets.push((insn.id, ret.ptr.qualify(fid)));
                    }
                }
            }
        }

        let mut changed = false;
        for (return_id, target) in targets {
            let size = cone.ctx().type_of(target);
            let size = cone.ctx().shared.types.size_of(size);
            if size == 0 {
                continue;
            }
            let code_ptr = cone.ctx().shared.types.get_or_make_code_pointer(size);
            let already_typed = matches!(
                cone.ctx()
                    .shared
                    .types
                    .get(cone.ctx().type_of(target))
                    .repr(),
                TypeRepr::CodePointer { .. }
            );

            match target {
                ValueId::BlockParam(id) => {
                    if !already_typed {
                        cone.ctx_for(id.func).block_param_mut(id).type_id = code_ptr;
                        changed = true;
                    }
                    let mut value =
                        qcode::value::BlockParam::from_id_mut(cone.ctx_for(id.func), id);
                    changed |= rename_ra(&mut value)?;
                }
                ValueId::Instruction(id) => {
                    let mut value = Instruction::from_id_mut(cone.ctx_for(id.func), id);
                    if !already_typed {
                        value.set_type(code_ptr);
                        changed = true;
                    }
                    changed |= rename_ra(&mut value)?;
                }
                ValueId::Varnode(id) => {
                    if !already_typed {
                        cone.set_varnode_type(id, code_ptr);
                        changed = true;
                    }
                    changed |= rename_ra(&mut cone.varnode_mut(id))?;
                }
                ValueId::Literal(id) if !already_typed => {
                    let literal = cone.ctx().shared.values.literals[id].clone();
                    let typed = cone.push_literal(qcode::value::literal::Literal {
                        type_id: code_ptr,
                        ..literal
                    });
                    let Mnemonic::Return(Return { value, .. }) =
                        cone.ctx().get_insn(return_id).mnemonic().clone()
                    else {
                        continue;
                    };
                    cone.ctx_for(return_id.func).replace_instruction_mnemonic(
                        return_id,
                        Mnemonic::Return(Return {
                            ptr: typed.localize(return_id.func),
                            value,
                        }),
                    );
                    changed = true;
                }
                _ => {}
            }
        }

        Ok(crate::ModulePassOutcome::module_if(changed))
    }
}

crate::register_module_pass!(IdentifyReturnAddress);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        context::Context,
        types::TypeRepr,
        value::{FunctionBody, Instruction, insn::Mnemonic},
    };
    use wazabin_qcode_macro::qcode;

    fn run(ctx: &mut Context) -> bool {
        let env = PipelineEnv::headless(ctx);
        IdentifyReturnAddress
            .run(&mut crate::ConeMut::full(ctx), &env)
            .unwrap()
            .changed()
    }

    #[test]
    fn types_and_renames_param_return_target_idempotently() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <entry @RSP_val_0:i64>
                return at @RSP_val_0;
            "
        );

        assert!(run(&mut ctx));
        let param = FunctionBody::from_id(&ctx, f)
            .root()
            .unwrap()
            .params()
            .next()
            .unwrap();
        assert_eq!(param.name(), Some("ra"));
        assert!(matches!(
            ctx.shared.types.get(param.type_id()).repr(),
            TypeRepr::CodePointer { .. }
        ));
        assert!(!run(&mut ctx), "a second run must be unchanged");
    }

    #[test]
    fn recognizes_instruction_return_target() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <entry>
                %target = i64 0x4000 + i64 0x0;
                return at %target;
            "
        );

        assert!(run(&mut ctx));
        let target = Instruction::from_id(&ctx, target);
        assert_eq!(target.name(), Some("ra"));
        assert!(matches!(
            ctx.shared.types.get(target.type_id()).repr(),
            TypeRepr::CodePointer { .. }
        ));
    }

    #[test]
    fn multiple_return_targets_get_stable_ra_names() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <first>
                %first_target = i64 0x4000 + i64 0x0;
                return at %first_target;
            <second>
                %second_target = i64 0x5000 + i64 0x0;
                return at %second_target;
            "
        );

        assert!(run(&mut ctx));
        let names: Vec<_> = FunctionBody::from_id(&ctx, f)
            .iter()
            .flat_map(|block| block.iter())
            .filter(|insn| matches!(insn.mnemonic(), Mnemonic::Binop(_)))
            .map(|insn| insn.name().unwrap().to_owned())
            .collect();
        assert_eq!(names, ["ra", "ra1"]);
        assert!(!run(&mut ctx), "allocated ra names must be stable");
    }

    #[test]
    fn existing_ra_prefix_is_preserved() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <entry @ra_saved:i64>
                return at @ra_saved;
            "
        );

        assert!(run(&mut ctx));
        let param = FunctionBody::from_id(&ctx, f)
            .root()
            .unwrap()
            .params()
            .next()
            .unwrap();
        assert_eq!(param.name(), Some("ra_saved"));
        assert!(!run(&mut ctx));
    }

    #[test]
    fn recognizes_literal_return_target() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <entry>
                return at i64 0x4000;
            "
        );

        assert!(run(&mut ctx));
        let ret = FunctionBody::from_id(&ctx, f)
            .iter()
            .flat_map(|block| block.iter())
            .find(|insn| matches!(insn.mnemonic(), Mnemonic::Return(_)))
            .unwrap();
        let Mnemonic::Return(ret) = ret.mnemonic() else {
            unreachable!()
        };
        let target = ret.ptr.qualify(f);
        assert!(matches!(
            ctx.shared.types.get(ctx.type_of(target)).repr(),
            TypeRepr::CodePointer { .. }
        ));
        assert!(!run(&mut ctx));
    }
}
