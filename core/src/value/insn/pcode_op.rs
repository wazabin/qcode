use super::mnemonic::{Args, MnemonicKind};
use smallvec::SmallVec;
use crate::{
    context::Context,
    value::{ValueId, ValueRef},
};
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

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        let op = &ctx.pcode_ops[self.id];
        let args = self
            .args
            .iter()
            .map(|&arg| ValueRef::new(arg, ctx).to_string())
            .collect::<Vec<_>>()
            .join(", ");

        if let Some(dst) = self.dst {
            write!(f, "{} = {}({});", ValueRef::new(dst, ctx), op, args)
        } else {
            write!(f, "{}({});", op, args)
        }
    }

    fn args(&self) -> Args {
        SmallVec::from_vec(self.args.clone())
    }
}
