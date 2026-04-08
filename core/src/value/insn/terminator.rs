use std::borrow::Cow;
use std::fmt::Formatter;

use crate::{
    context::Context,
    value::{BasicBlock, Function, ValueId, block::BlockId, function::FunctionId},
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
                .unwrap_or(&Cow::Borrowed("unnamed"))
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
        write!(f, "goto [{}];", ctx.get_value(self.ptr))
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
        write!(f, "call [{}];", ctx.get_value(self.ptr))
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
    pub target: BlockId,
    pub fallthrough: BlockId,
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
            ctx.get_value(self.condition),
            BasicBlock::from_id(ctx, self.target)
                .name()
                .unwrap_or(&Cow::Borrowed("unnamed")),
            BasicBlock::from_id(ctx, self.fallthrough)
                .name()
                .unwrap_or(&Cow::Borrowed("unnamed"))
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
        write!(f, "return [{}];", ctx.get_value(self.ptr))
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
        builder::Builder,
        context::Context,
        value::{BasicBlock, insn::Mnemonic},
    };

    #[test]
    fn qcode_emits_branch() {
        let mut ctx = Context::new();
        {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            qcode!(builder, "goto <done>");
            // No finalize needed — goto terminates the block.
        }

        let block = BasicBlock::from_addr(&ctx, 0x1000).expect("block not found");
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::Branch(_)));
        assert_eq!(last.as_statement().to_string(), "goto <done>;");
    }

    #[test]
    fn qcode_emits_branchind() {
        let mut ctx = Context::new();
        {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            qcode!(builder, "local i64 ptr");
            qcode!(builder, "goto [{ptr}]");
            // No finalize needed — goto terminates the block.
        }

        let block = BasicBlock::from_addr(&ctx, 0x1000).expect("block not found");
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::BranchInd(_)));
    }

    #[test]
    fn qcode_emits_cbranch() {
        let mut ctx = Context::new();
        {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            qcode!(builder, "local i8 cond");
            qcode!(builder, "if {cond} goto <then_lbl> else goto <else_lbl>");
            // cbranch switches the builder to the (unterminated) else_lbl block.
            builder.finalize(0x1001);
        }

        let block = BasicBlock::from_addr(&ctx, 0x1000).expect("block not found");
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::CBranch(_)));
    }

    #[test]
    fn qcode_emits_call() {
        let mut ctx = Context::new();
        {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            qcode!(builder, "call <target>");
            // No finalize needed — call terminates the block.
        }

        let block = BasicBlock::from_addr(&ctx, 0x1000).expect("block not found");
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::Call(_)));
    }

    #[test]
    fn qcode_emits_callind() {
        let mut ctx = Context::new();
        {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            qcode!(builder, "local i64 ptr");
            qcode!(builder, "call [{ptr}]");
            // No finalize needed — call terminates the block.
        }

        let block = BasicBlock::from_addr(&ctx, 0x1000).expect("block not found");
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::CallInd(_)));
    }

    #[test]
    fn qcode_emits_return() {
        let mut ctx = Context::new();
        {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            qcode!(builder, "local i64 ptr");
            qcode!(builder, "return [{ptr}]");
            // No finalize needed — return terminates the block.
        }

        let block = BasicBlock::from_addr(&ctx, 0x1000).expect("block not found");
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::Return(_)));
    }

    #[test]
    fn qcode_multi_block_with_label() {
        let mut ctx = Context::new();
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        qcode!(builder, "local i32 v");
        // Emit into entry block, then declare a label and emit more there.
        qcode!(builder, "goto <body>; <body> {v} + 1");
        builder.finalize(0x1001);

        // Entry block ends with a branch to "body".
        let entry = BasicBlock::from_addr(&ctx, 0x1000).expect("block not found");
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
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        qcode!(builder, "local i8 cond");
        qcode!(builder, "if {cond} goto <then_lbl> else goto <else_lbl>");
        builder.finalize(0x1001);

        let block = BasicBlock::from_addr(&ctx, 0x1000).expect("block not found");
        let last = block.iter().last().expect("block has instructions");
        let Mnemonic::CBranch(cbranch) = last.mnemonic() else {
            panic!("expected cbranch");
        };
        assert_ne!(
            cbranch.target, cbranch.fallthrough,
            "target and fallthrough must be distinct"
        );
    }
}
