use std::fmt::Formatter;

use crate::{
    context::Context,
    value::{BasicBlock, Function, ValueId, ValueRef, block::BlockId, function::FunctionId},
};

use super::mnemonic::MnemonicKind;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Branch {
    pub target: BlockId,
}

impl MnemonicKind for Branch {
    fn opcode(&self) -> &'static str {
        "branch"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(
            f,
            "goto <{}>;",
            BasicBlock::from_id(ctx, self.target)
                .name()
                .unwrap_or("unnamed")
        )
    }

    fn args(&self) -> Vec<ValueId> {
        vec![]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

    fn args(&self) -> Vec<ValueId> {
        vec![self.ptr]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Call {
    pub target: FunctionId,
    pub args: Vec<ValueId>,
}

impl MnemonicKind for Call {
    fn opcode(&self) -> &'static str {
        "call"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "call fn {};", Function::from_id(ctx, self.target).name())
    }

    fn args(&self) -> Vec<ValueId> {
        self.args.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

    fn args(&self) -> Vec<ValueId> {
        let mut args = vec![self.ptr];
        args.extend(self.args.clone());
        args
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CBranch {
    pub condition: ValueId,
    pub success_block: BlockId,
    pub failure_block: BlockId,
}

impl MnemonicKind for CBranch {
    fn opcode(&self) -> &'static str {
        "cbranch"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(
            f,
            "if {} goto <{}>; else goto <{}>;",
            ValueRef::new(self.condition, ctx),
            BasicBlock::from_id(ctx, self.success_block)
                .name()
                .unwrap_or("unnamed"),
            BasicBlock::from_id(ctx, self.failure_block)
                .name()
                .unwrap_or("unnamed")
        )
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.condition]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
        write!(f, "return [{}];", ValueRef::new(self.ptr, ctx))
    }

    fn args(&self) -> Vec<ValueId> {
        let mut args = vec![self.ptr];
        if let Some(value) = self.value {
            args.push(value);
        }
        args
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use crate::{
        context::Context,
        value::{BasicBlock, insn::Mnemonic},
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
                goto [%ptr];
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
                %c = load(i8, &cond);
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
    fn qcode_emits_callind() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                local i64 ptr;
                call [%ptr];
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
                return [%ptr];
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::Return(_)));
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
                %c = load(i8, &cond);
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
}
