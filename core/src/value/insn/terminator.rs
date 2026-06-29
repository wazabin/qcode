use std::fmt::Formatter;

use crate::{
    context::Context,
    value::{BasicBlock, Function, ValueId, ValueRef, block::BlockId, function::FunctionId},
};

use super::mnemonic::{Args, MnemonicKind};
use smallvec::{SmallVec, smallvec};

fn fmt_branch_target(
    f: &mut Formatter<'_>,
    ctx: &Context<'_>,
    target: BlockId,
    args: &[ValueId],
) -> std::fmt::Result {
    let block = BasicBlock::from_id(ctx, target);
    let name = block.name().unwrap_or("unnamed");
    write!(f, "<{name}")?;

    let params = block.params().collect::<Vec<_>>();
    for (i, &arg) in args.iter().enumerate() {
        write!(f, " ")?;
        if let Some(param) = params.get(i) {
            write!(f, "{param}")?;
        } else {
            write!(f, "@arg{i}")?;
        }
        write!(f, "={}", ValueRef::new(arg, ctx))?;
    }

    write!(f, ">")
}

fn fmt_call_arg_name(
    f: &mut Formatter<'_>,
    ctx: &Context<'_>,
    target: FunctionId,
    index: usize,
) -> std::fmt::Result {
    if let Some(name) = Function::from_id(ctx, target).input_arg_name(index) {
        return write!(f, "@{name}=");
    }

    write!(f, "@arg{index}=")
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Branch {
    pub target: BlockId,
    /// Arguments passed to the target block's parameters.
    pub args: Vec<ValueId>,
}

impl MnemonicKind for Branch {
    fn opcode(&self) -> &'static str {
        "branch"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "goto ")?;
        fmt_branch_target(f, ctx, self.target, &self.args)?;
        write!(f, ";")
    }

    fn args(&self) -> Args {
        SmallVec::from_vec(self.args.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BranchInd {
    pub ptr: ValueId,
}

impl MnemonicKind for BranchInd {
    fn opcode(&self) -> &'static str {
        "branchind"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "goto [{}];", ValueRef::new(self.ptr, ctx))
    }

    fn args(&self) -> Args {
        smallvec![self.ptr]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Apply {
    pub target: FunctionId,
    /// Values passed to the lambda, one per root block param, in order.
    pub args: Vec<ValueId>,
}

impl MnemonicKind for Apply {
    fn opcode(&self) -> &'static str {
        "apply"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "apply @{}(", Function::from_id(ctx, self.target).name())?;
        for (i, &arg) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            fmt_call_arg_name(f, ctx, self.target, i)?;
            write!(f, "{}", ValueRef::new(arg, ctx))?;
        }
        write!(f, ");")
    }

    fn args(&self) -> Vec<ValueId> {
        self.args.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Call {
    pub target: FunctionId,
    /// Values passed to the callee, one per inferred callee input, in order.
    pub args: Vec<ValueId>,
    /// Register / memory locations the call may write or alias (the callee's
    /// clobbered set plus escaping pointer arguments). These are *defs*, not
    /// reads: they are intentionally excluded from [`MnemonicKind::args`] so
    /// they do not participate in use-def bookkeeping.
    pub clobbers: Vec<ValueId>,
}

impl MnemonicKind for Call {
    fn opcode(&self) -> &'static str {
        "call"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        // The clobber set is intentionally not printed: it is large and
        // repetitive at every call site. Only the argument list is shown.
        write!(f, "call fn {}(", Function::from_id(ctx, self.target).name())?;
        for (i, &arg) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            fmt_call_arg_name(f, ctx, self.target, i)?;
            write!(f, "{}", ValueRef::new(arg, ctx))?;
        }
        write!(f, ");")
    }

    fn args(&self) -> Args {
        SmallVec::from_vec(self.args.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CallInd {
    pub ptr: ValueId,
    pub args: Vec<ValueId>,
}

impl MnemonicKind for CallInd {
    fn opcode(&self) -> &'static str {
        "callind"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "call [{}];", ValueRef::new(self.ptr, ctx))
    }

    fn args(&self) -> Args {
        let mut args = smallvec![self.ptr];
        args.extend(self.args.clone());
        args
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CBranch {
    pub condition: ValueId,
    pub success_block: BlockId,
    /// Arguments passed to `success_block`'s parameters when the branch is taken.
    pub success_args: Vec<ValueId>,
    pub failure_block: BlockId,
    /// Arguments passed to `failure_block`'s parameters when the branch falls through.
    pub failure_args: Vec<ValueId>,
}

impl MnemonicKind for CBranch {
    fn opcode(&self) -> &'static str {
        "cbranch"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "if {} goto ", ValueRef::new(self.condition, ctx))?;
        fmt_branch_target(f, ctx, self.success_block, &self.success_args)?;
        write!(f, " else goto ")?;
        fmt_branch_target(f, ctx, self.failure_block, &self.failure_args)?;
        write!(f, ";")
    }

    fn args(&self) -> Args {
        let mut args = smallvec![self.condition];
        args.extend_from_slice(&self.success_args);
        args.extend_from_slice(&self.failure_args);
        args
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Return {
    pub ptr: ValueId,
    pub value: Option<ValueId>,
}

impl MnemonicKind for Return {
    fn opcode(&self) -> &'static str {
        "return"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        match self.value {
            Some(value) => write!(
                f,
                "return {} at {};",
                ValueRef::new(value, ctx),
                ValueRef::new(self.ptr, ctx)
            ),
            None => write!(f, "return at {};", ValueRef::new(self.ptr, ctx)),
        }
    }

    fn args(&self) -> Args {
        let mut args = smallvec![self.ptr];
        if let Some(value) = self.value {
            args.push(value);
        }
        args
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ReturnValue {
    pub value: ValueId,
}

impl MnemonicKind for ReturnValue {
    fn opcode(&self) -> &'static str {
        "returnvalue"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "return {};", ValueRef::new(self.value, ctx))
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.value]
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use crate::{
        builder::Builder,
        context::Context,
        testing::TestContext,
        value::{BasicBlock, Function, Instruction, insn::Mnemonic},
    };

    #[test]
    fn qcode_emits_branch() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                goto <done>;
            <done>
                goto <0x1001>;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::Branch(_)));
        assert_eq!(last.as_statement().to_string(), "goto <done>;");
    }

    #[test]
    fn qcode_emits_branchind() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                local i64 ptr;
                goto [ptr];
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::BranchInd(_)));
    }

    #[test]
    fn qcode_emits_cbranch() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;

            <block>
                %c = load(cond:1, &cond);
                if %c goto <then_lbl> else goto <else_lbl>;

            <then_lbl>
                goto <0x1001>;

            <else_lbl>
                goto <0x1002>;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::CBranch(_)));
    }

    #[test]
    fn qcode_emits_call() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                call <target>;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::Call(_)));
    }

    #[test]
    fn call_display_shows_named_args_with_fallbacks() {
        let mut tc = TestContext::new();
        let callee = Function::make(&mut tc.ctx, "callee".into()).unwrap().id;
        Function::from_id_mut(&mut tc.ctx, callee).set_input_regs(vec![tc.r0]);

        let block = BasicBlock::make(&mut tc.ctx).id;
        let call_id = {
            let mut builder = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, block));
            builder.push_call(callee).id
        };

        let first = tc.ctx.get_const(1u64, 8).id();
        let second = tc.ctx.get_const(2u64, 8).id();
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(super::Call {
                target: callee,
                args: vec![first, second],
                clobbers: vec![],
            }),
        );

        let rendered = Instruction::from_id(&tc.ctx, call_id)
            .as_statement()
            .to_string();
        assert_eq!(rendered, "call fn callee(@r0=0x1, @arg1=0x2);");
    }

    #[test]
    fn call_display_names_stack_passed_arg() {
        use crate::{space::Space, value::Varnode};

        let mut tc = TestContext::new();

        // A "stack" space (addr_size = pointer width 4). A stack-passed parameter
        // is a nameless varnode in this space at the slot offset.
        let stack_space = tc.ctx.add_space(Space::new(Some("stack"), 1, 4));
        let stack_input = Varnode::make(&mut tc.ctx, 4, 4, stack_space).id;

        let callee = Function::make(&mut tc.ctx, "callee".into()).unwrap().id;
        Function::from_id_mut(&mut tc.ctx, callee).set_input_regs(vec![stack_input]);

        let block = BasicBlock::make(&mut tc.ctx).id;
        let call_id = {
            let mut builder = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, block));
            builder.push_call(callee).id
        };

        let arg = tc.ctx.get_const(7u64, 4).id();
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(super::Call {
                target: callee,
                args: vec![arg],
                clobbers: vec![],
            }),
        );

        let rendered = Instruction::from_id(&tc.ctx, call_id)
            .as_statement()
            .to_string();
        // The stack-passed input is named after its slot offset (varnode address 4).
        assert_eq!(rendered, "call fn callee(@stack_4=0x7);");
    }

    #[test]
    fn qcode_emits_callind() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                local i64 ptr;
                call [ptr];
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::CallInd(_)));
    }

    #[test]
    fn qcode_emits_return() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                local i64 ptr;
                return at ptr;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::Return(_)));
    }

    #[test]
    fn qcode_emits_lambda_apply_and_value_return() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda rec:
            <entry @s:i64>
                %next = @s + 1;
                %out = apply rec(%next);
                return %out;
            "
        );

        let rec = Function::from_name(&ctx, "rec").expect("lambda exists");
        assert!(rec.is_lambda());
        let entry = rec.root().expect("lambda has root");
        let insns = entry.instruction_ids();
        let apply = ctx.get_insn(insns[1]);
        assert!(!apply.is_terminator(), "apply is a value instruction");
        assert!(matches!(apply.mnemonic(), Mnemonic::Apply(_)));
        assert!(matches!(
            ctx.get_insn(*insns.last().unwrap()).mnemonic(),
            Mnemonic::ReturnValue(_)
        ));
        assert!(apply.as_statement().to_string().contains("apply @rec("));
        assert_eq!(
            ctx.get_insn(*insns.last().unwrap())
                .as_statement()
                .to_string(),
            "return i64 %out;"
        );
    }

    #[test]
    fn qcode_multi_block_with_label() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 V;

            <block>
                goto <body>;

            <body>
                %sum = i64 &V + i64 0x1;
                goto <0x1001>;
            "
        );

        // Entry block ends with a branch to "body".
        let entry = BasicBlock::from_id(&ctx, block);
        let entry_last = entry.iter().last().expect("entry has instructions");
        assert!(matches!(entry_last.mnemonic(), Mnemonic::Branch(_)));

        // "body" block contains the add instruction.
        let Mnemonic::Branch(branch) = entry_last.mnemonic() else {
            panic!("expected branch");
        };
        let body = BasicBlock::from_id(&ctx, branch.target);
        assert!(!body.is_empty());
    }

    #[test]
    fn qcode_cbranch_target_and_fallthrough_are_distinct_blocks() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;

            <block>
                %c = load(cond:1, &cond);
                if %c goto <then_lbl> else goto <else_lbl>;

            <then_lbl>
                goto <0x1001>;

            <else_lbl>
                goto <0x1002>;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        let Mnemonic::CBranch(cbranch) = last.mnemonic() else {
            panic!("expected cbranch");
        };
        assert_ne!(
            cbranch.success_block, cbranch.failure_block,
            "target and fallthrough must be distinct"
        );
    }

    #[test]
    fn branch_with_args_stores_args() {
        use crate::value::ValueId;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <src @a>
                goto <dst @x=@a>;
            <dst @x>
                goto <0x1001>;
            "
        );

        let src_block = BasicBlock::from_id(&ctx, src);
        let last = src_block.iter().last().expect("block has instructions");
        let Mnemonic::Branch(branch) = last.mnemonic() else {
            panic!("expected branch");
        };
        assert_eq!(branch.target, dst);
        assert_eq!(branch.args.len(), 1);
        assert_eq!(branch.args[0], ValueId::BlockParam(a));
    }

    #[test]
    fn cbranch_with_per_target_args_are_independent() {
        use crate::value::ValueId;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <src @cond:i8 @then_arg:i64 @else_arg:i64>
                if @cond goto <then_lbl @x=@then_arg> else goto <else_lbl @y=@else_arg>;
            <then_lbl @x:i64>
                goto <0x1001>;
            <else_lbl @y:i64>
                goto <0x1002>;
            "
        );

        let src_block = BasicBlock::from_id(&ctx, src);
        let insn = src_block.iter().last().expect("src has cbranch");
        let Mnemonic::CBranch(cbranch) = insn.mnemonic() else {
            panic!("expected cbranch");
        };
        assert_eq!(cbranch.success_args, [ValueId::BlockParam(then_arg)]);
        assert_eq!(cbranch.failure_args, [ValueId::BlockParam(else_arg)]);
        assert_ne!(cbranch.success_block, cbranch.failure_block);
    }

    #[test]
    fn branch_args_display() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <src @a>
                goto <done @x=@a>;
            <done @x>
                goto <0x1001>;
            "
        );

        let src_block = BasicBlock::from_id(&ctx, src);
        let last = src_block.iter().last().expect("block has instructions");
        assert_eq!(last.as_statement().to_string(), "goto <done @x=@a>;");
    }

    #[test]
    fn branch_args_are_ordered_by_target_params() {
        use crate::value::ValueId;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <src @a @b>
                goto <done @y=@a @x=@b>;
            <done @x @y>
                goto <0x1001>;
            "
        );

        let src_block = BasicBlock::from_id(&ctx, src);
        let last = src_block.iter().last().expect("block has instructions");
        let Mnemonic::Branch(branch) = last.mnemonic() else {
            panic!("expected branch");
        };
        assert_eq!(
            branch.args,
            [ValueId::BlockParam(b), ValueId::BlockParam(a)]
        );
        assert_eq!(last.as_statement().to_string(), "goto <done @x=@b @y=@a>;");
    }
}
