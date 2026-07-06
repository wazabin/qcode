use crate::value::ValueId;

use super::mnemonic::{Args, MnemonicKind};
use smallvec::smallvec;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Assert {
    pub condition: ValueId,
}

impl MnemonicKind for Assert {
    fn opcode(&self) -> &'static str {
        "assert"
    }

    fn args(&self) -> Args {
        smallvec![self.condition]
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use crate::{context::Context, value::insn::Mnemonic};

    #[test]
    fn test_assert_macro_and_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i8 COND;
                %cond = load(COND:1, COND);
                assert %cond;
                goto <0x1001>;
            "
        );

        let block = crate::value::BasicBlock::from_id(&ctx, block);
        let assert_insn = block
            .iter()
            .find(|i| matches!(i.mnemonic(), Mnemonic::Assert(_)))
            .expect("assert instruction should be present");

        assert_eq!(assert_insn.size(), 0);
        assert_eq!(assert_insn.as_statement().to_string(), "assert i8 %cond;");
    }
}
