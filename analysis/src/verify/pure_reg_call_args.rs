//! Verify the `pure_reg` call-interface lockstep invariant.
//!
//! A `pure_reg` function is called by value: its root block params are the
//! canonical input interface and every direct `Call.args` list must be aligned
//! with those params index-for-index. Several call passes rely on this, and the
//! emulator binds pure-reg entries positionally from the call's arguments.

use qcode::{
    context::Context,
    value::{
        BasicBlock, Function, FunctionId, Instruction, Value, ValueId, ValueRef,
        insn::InstructionId,
    },
};

/// A direct call to a `pure_reg` function does not match the callee's root params.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PureRegCallArgsViolation {
    pub callee: FunctionId,
    pub call: InstructionId,
    pub expected_args: usize,
    pub actual_args: usize,
    pub size_mismatch: Option<ArgSizeMismatch>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgSizeMismatch {
    pub index: usize,
    pub param_size: usize,
    pub arg_size: usize,
}

impl PureRegCallArgsViolation {
    fn to_string_with_ctx(&self, ctx: &Context<'_>) -> String {
        let callee = Function::from_id(ctx, self.callee).name().to_owned();
        let call = Instruction::from_id(ctx, self.call);
        let call_addr = call
            .address()
            .map(|addr| format!(" at {addr:#x}"))
            .unwrap_or_default();
        let mut msg = format!(
            "pure_reg call interface mismatch: call{call_addr} to `{callee}` has {} args, callee root has {} params",
            self.actual_args, self.expected_args
        );
        if let Some(size) = &self.size_mismatch {
            msg.push_str(&format!(
                "; arg {} is {} bytes but param is {} bytes",
                size.index, size.arg_size, size.param_size
            ));
        }
        msg
    }
}

impl PureRegCallArgsViolation {
    pub fn diagnostic(&self, ctx: &Context<'_>) -> String {
        self.to_string_with_ctx(ctx)
    }
}

/// Return every direct-call interface violation for `pure_reg` functions.
pub fn verify_pure_reg_call_args(ctx: &Context<'_>) -> Vec<PureRegCallArgsViolation> {
    let mut violations = Vec::new();

    for callee in ctx.function_ids() {
        let function = Function::from_id(ctx, callee);
        if !function.is_pure_reg() {
            continue;
        }
        let Some(root) = function.root().map(|b| b.id) else {
            continue;
        };
        let param_sizes: Vec<usize> = BasicBlock::from_id(ctx, root)
            .params()
            .map(|param| param.size())
            .collect();

        for insn in ctx.instructions() {
            let qcode::value::insn::Mnemonic::Call(call) = insn.mnemonic() else {
                continue;
            };
            if call.target != callee {
                continue;
            }

            let size_mismatch = if call.args.len() == param_sizes.len() {
                first_size_mismatch(ctx, &insn.operands(), &param_sizes)
            } else {
                None
            };

            if call.args.len() != param_sizes.len() || size_mismatch.is_some() {
                violations.push(PureRegCallArgsViolation {
                    callee,
                    call: insn.id,
                    expected_args: param_sizes.len(),
                    actual_args: call.args.len(),
                    size_mismatch,
                });
            }
        }
    }

    violations
}

fn first_size_mismatch(
    ctx: &Context<'_>,
    args: &[ValueId],
    param_sizes: &[usize],
) -> Option<ArgSizeMismatch> {
    args.iter()
        .zip(param_sizes)
        .enumerate()
        .find_map(|(index, (&arg, &param_size))| {
            let arg_size = ValueRef::new(arg, ctx).size();
            (arg_size != param_size).then_some(ArgSizeMismatch {
                index,
                param_size,
                arg_size,
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{
            Function,
            insn::{Call, Mnemonic},
        },
    };

    fn pure_callee_with_params(tc: &mut TestContext, sizes: &[usize]) -> FunctionId {
        let callee = Function::make(&mut tc.ctx, "callee".into()).unwrap().id;
        let root = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x1000, __f)
        };
        Function::from_id_mut(&mut tc.ctx, callee)
            .set_root(root)
            .unwrap();
        for &size in sizes {
            BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(size);
        }
        Function::from_id_mut(&mut tc.ctx, callee).set_pure_reg(true);
        callee
    }

    fn caller_calling(
        tc: &mut TestContext,
        callee: FunctionId,
        args: Vec<ValueId>,
    ) -> InstructionId {
        let caller = Function::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let block = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x2000, __f)
        };
        Function::from_id_mut(&mut tc.ctx, caller)
            .set_root(block)
            .unwrap();
        let call_id = {
            let mut b = Builder::from_context(&mut tc.ctx, 0x2000);
            let id = b.push_call(callee).id;
            unsafe { b.dont_finalize() };
            id
        };
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: callee,
                args: args
                    .into_iter()
                    .map(|arg| arg.localize(call_id.func))
                    .collect(),
                clobbers: vec![],
            }),
        );
        call_id
    }

    #[test]
    fn accepts_matching_pure_reg_call_args() {
        let mut tc = TestContext::new();
        let callee = pure_callee_with_params(&mut tc, &[8, 4]);
        let a0 = tc.ctx.get_const(0x11, 8).id();
        let a1 = tc.ctx.get_const(0x22, 4).id();
        caller_calling(&mut tc, callee, vec![a0, a1]);

        assert!(verify_pure_reg_call_args(&tc.ctx).is_empty());
    }

    #[test]
    fn reports_argument_count_mismatch() {
        let mut tc = TestContext::new();
        let callee = pure_callee_with_params(&mut tc, &[8, 4]);
        let a0 = tc.ctx.get_const(0x11, 8).id();
        let call = caller_calling(&mut tc, callee, vec![a0]);

        let violations = verify_pure_reg_call_args(&tc.ctx);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].call, call);
        assert_eq!(violations[0].expected_args, 2);
        assert_eq!(violations[0].actual_args, 1);
    }

    #[test]
    fn reports_argument_size_mismatch() {
        let mut tc = TestContext::new();
        let callee = pure_callee_with_params(&mut tc, &[8]);
        let a0 = tc.ctx.get_const(0x11, 4).id();
        caller_calling(&mut tc, callee, vec![a0]);

        let violations = verify_pure_reg_call_args(&tc.ctx);
        assert_eq!(violations.len(), 1);
        assert_eq!(
            violations[0].size_mismatch,
            Some(ArgSizeMismatch {
                index: 0,
                param_size: 8,
                arg_size: 4,
            })
        );
    }

    #[test]
    fn ignores_non_pure_reg_callees() {
        let mut tc = TestContext::new();
        let callee = pure_callee_with_params(&mut tc, &[8, 4]);
        Function::from_id_mut(&mut tc.ctx, callee).set_pure_reg(false);
        let a0 = tc.ctx.get_const(0x11, 8).id();
        caller_calling(&mut tc, callee, vec![a0]);

        assert!(verify_pure_reg_call_args(&tc.ctx).is_empty());
    }
}
