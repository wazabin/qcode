use super::mnemonic::{Args, MnemonicKind};
use crate::value::ValueId;
use jstd::Identifier;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

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

    fn args(&self) -> Args {
        SmallVec::from_vec(self.args.clone())
    }
}
