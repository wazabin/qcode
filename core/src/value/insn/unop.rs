use std::fmt::{Display, Formatter};

use crate::{
    context::Context,
    value::{ValueId, insn::mnemonic::MnemonicKind},
};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Unop {
    IntNegate,
    IntNot,
    BoolNot,
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
            Unop::BoolNot => Some(u128::from(v == 0)),
            _ => None,
        }
    }
}

impl Display for Unop {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Unop::IntNegate => "-",
            Unop::IntNot => "~",
            Unop::BoolNot => "!",
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Unary {
    pub op: Unop,
    pub src: ValueId,
}

impl MnemonicKind for Unary {
    fn opcode(&self) -> &'static str {
        "unop"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        match self.op {
            Unop::IntNegate | Unop::IntNot | Unop::BoolNot => {
                write!(f, "{} {};", self.op, ctx.get_value(self.src))
            }
            Unop::FloatNegate
            | Unop::FloatAbs
            | Unop::FloatSqrt
            | Unop::FloatCeil
            | Unop::FloatFloor
            | Unop::FloatRound => {
                write!(f, "{}({});", self.op, ctx.get_value(self.src))
            }
        }
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.src]
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use crate::{
        value::Value,
        value::insn::{Instruction, Mnemonic},
    };

    use super::*;

    // TODO: add tests for all the different unop variants similar to the ones in flags.rs and casting.rs

    #[test]
    fn test_bool_not_from_qcode() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 V0;

            <block>
                %v0 = load(i32, &V0);
                %v = !%v0;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Unop(Unary { op, src }) => {
                assert_eq!(*op, Unop::BoolNot);
                assert_eq!(ctx.get_value(*src).size(), 4);
            }
            _ => panic!("expected boolean unop instruction"),
        }
    }
}
