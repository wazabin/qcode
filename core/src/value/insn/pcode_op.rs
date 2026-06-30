use super::mnemonic::MnemonicKind;
use crate::value::ValueId;
use jstd::Identifier;
use serde::{Deserialize, Serialize};

#[derive(Identifier)]
pub struct PCodeOpId(usize);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PCodeOp {
    pub id: PCodeOpId,
    pub args: Vec<ValueId>,
    pub dst: Option<ValueId>,
}

impl MnemonicKind for PCodeOp {
    fn opcode(&self) -> &'static str {
        "pcode_op"
    }

    fn args(&self) -> Vec<ValueId> {
        self.args.clone()
    }
}
