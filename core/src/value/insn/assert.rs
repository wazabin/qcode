use std::fmt::Formatter;

use crate::{
    context::Context,
    value::{ValueId, ValueRef},
};

use super::mnemonic::MnemonicKind;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Assert {
    pub condition: ValueId,
}

impl MnemonicKind for Assert {
    fn opcode(&self) -> &'static str {
        "assert"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "assert {};", ValueRef::new(self.condition, ctx))
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.condition]
    }
}
