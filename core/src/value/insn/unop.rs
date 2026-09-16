use std::fmt::{Display, Formatter};

use crate::LangRef;
use crate::value::{LocalValueId, insn::mnemonic::MnemonicKind};

/// The unary operators of a [`Unary`] instruction.
///
/// Prefix operators (`-`, `~`, `f-`) are written before the operand; the
/// float functions are written as calls. The result has the operand's type.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, LangRef)]
#[langref(category = "Unary operators")]
pub enum Unop {
    /// Two's-complement negation, wrapping: `-v` is `0 - v` modulo `2^bits`.
    #[langref(syntax = "T %r = - v", example = "i32 %r = - i32 @x;")]
    IntNegate,
    /// Bitwise complement: every bit of `v` inverted.
    #[langref(syntax = "T %r = ~ v", example = "i32 %r = ~ i32 @x;")]
    IntNot,
    /// Floating-point negation: the sign bit flipped. `f- NaN` is a NaN.
    #[langref(syntax = "T %r = f- v", example = "i64 %r = f- i64 %f;")]
    FloatNegate,
    /// Floating-point absolute value: the sign bit cleared.
    #[langref(syntax = "T %r = abs(v)", example = "i64 %r = abs(i64 %f);")]
    FloatAbs,
    /// Floating-point square root, rounded to nearest even. Negative inputs
    /// produce NaN.
    #[langref(syntax = "T %r = sqrt(v)", example = "i64 %r = sqrt(i64 %f);")]
    FloatSqrt,
    /// Round toward positive infinity, as a float.
    #[langref(syntax = "T %r = ceil(v)", example = "i64 %r = ceil(i64 %f);")]
    FloatCeil,
    /// Round toward negative infinity, as a float.
    #[langref(syntax = "T %r = floor(v)", example = "i64 %r = floor(i64 %f);")]
    FloatFloor,
    /// Round to the nearest integer, as a float; ties round away from zero.
    #[langref(syntax = "T %r = round(v)", example = "i64 %r = round(i64 %f);")]
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
}

// TODO: add tests for all the different unop variants similar to the ones in flags.rs and casting.rs
