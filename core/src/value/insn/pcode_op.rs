use super::mnemonic::MnemonicKind;
use crate::value::LocalValueId;
use serde::{Deserialize, Serialize};

pub use pcode_types::PCodeOpId;

/// The reserved user op that stops the machine so the environment can act.
///
/// `vm.interrupt(code, args...) -> value?` retires everything before it,
/// runs nothing after it, and surfaces as a typed exit from the VM. The
/// environment supplies the op's result, if it declares one, and resumes.
/// It is how an injected hook hands control to the host from inside a block,
/// whether that block is interpreted or compiled. See `qcode_vm`.
pub const VM_INTERRUPT: &str = "vm.interrupt";

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PCodeOp {
    pub id: PCodeOpId,
    pub args: Vec<LocalValueId>,
    pub dst: Option<LocalValueId>,
}

impl MnemonicKind for PCodeOp {
    fn opcode(&self) -> &'static str {
        "pcode_op"
    }
}
