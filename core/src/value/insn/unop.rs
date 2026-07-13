use std::fmt::{Display, Formatter};

use crate::value::{LocalValueId, insn::mnemonic::MnemonicKind};
use smallvec::smallvec;

use super::mnemonic::Args;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Unop {
    IntNegate,
    IntNot,
    FloatNegate,
    FloatAbs,
    FloatSqrt,
    FloatCeil,
    FloatFloor,
    FloatRound,
}

impl Unop {
    /// Evaluates the integer variants of this operation on a raw bit pattern.
    ///
    /// Returns `None` for float variants, which are not pure integer operations.
    /// `size` is the operand width in bytes; the result is masked to `size` bytes.
    pub fn eval_int(&self, value: u128, size: usize) -> Option<u128> {
        use super::bits::mask_for_size;
        let mask = mask_for_size(size);
        let v = value & mask;
        match self {
            Unop::IntNegate => Some(v.wrapping_neg() & mask),
            Unop::IntNot => Some(!v & mask),
            _ => None,
        }
    }
}

impl Display for Unop {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Unop::IntNegate => "-",
            Unop::IntNot => "~",
            Unop::FloatNegate => "f-",
            Unop::FloatAbs => "abs",
            Unop::FloatSqrt => "sqrt",
            Unop::FloatCeil => "ceil",
            Unop::FloatFloor => "floor",
            Unop::FloatRound => "round",
        };

        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Unary {
    pub op: Unop,
    pub src: LocalValueId,
}

impl MnemonicKind for Unary {
    fn opcode(&self) -> &'static str {
        "unop"
    }

    fn args(&self) -> Args {
        smallvec![self.src]
    }
}

// TODO: add tests for all the different unop variants similar to the ones in flags.rs and casting.rs
