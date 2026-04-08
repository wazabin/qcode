use std::fmt::{Display, Formatter};

use crate::{context::Context, value::ValueId};

use super::mnemonic::MnemonicKind;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Binary {
    pub op: Binop,
    pub lhs: ValueId,
    pub rhs: ValueId,
}

impl MnemonicKind for Binary {
    fn opcode(&self) -> &'static str {
        "binop"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} {};",
            ctx.get_value(self.lhs),
            self.op,
            ctx.get_value(self.rhs)
        )
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.lhs, self.rhs]
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Binop {
    Int(IntBinop),
    Bool(BoolBinop),
    Float(FloatBinop),
}

impl Display for Binop {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Binop::Int(op) => write!(f, "{}", op),
            Binop::Bool(op) => write!(f, "{}", op),
            Binop::Float(op) => write!(f, "{}", op),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoolBinop {
    And,
    Or,
    Xor,
}

impl BoolBinop {
    /// Evaluates this boolean operation on raw bit patterns.
    /// Any non-zero value is treated as `true`; the result is `0` or `1`.
    pub fn eval(&self, lhs: u128, rhs: u128) -> u128 {
        let a = lhs != 0;
        let b = rhs != 0;
        u128::from(match self {
            BoolBinop::And => a && b,
            BoolBinop::Or => a || b,
            BoolBinop::Xor => a ^ b,
        })
    }
}

impl Display for BoolBinop {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BoolBinop::And => "&&",
            BoolBinop::Or => "||",
            BoolBinop::Xor => "xor",
        };

        write!(f, "{}", s)
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntBinop {
    Equal,
    NotEqual,
    Less,
    SLess,
    LessEqual,
    SLessEqual,
    Add,
    Sub,
    Xor,
    And,
    Or,
    ShiftLeft,
    ShiftRight,
    SShiftRight,
    Mul,
    Div,
    Rem,
    Sdiv,
    Srem,
}

impl IntBinop {
    /// Evaluates this operation on raw bit patterns.
    ///
    /// `size` is the operand width in bytes. Arithmetic results are masked to `size` bytes.
    /// Comparison results are always `0` or `1`.
    pub fn eval(&self, lhs: u128, rhs: u128, size: usize) -> u128 {
        use super::bits::{mask_for_size, signed_value};
        let mask = mask_for_size(size);
        let a = lhs & mask;
        let b = rhs & mask;
        match self {
            IntBinop::Equal => u128::from(a == b),
            IntBinop::NotEqual => u128::from(a != b),
            IntBinop::Less => u128::from(a < b),
            IntBinop::SLess => u128::from(signed_value(a, size) < signed_value(b, size)),
            IntBinop::LessEqual => u128::from(a <= b),
            IntBinop::SLessEqual => u128::from(signed_value(a, size) <= signed_value(b, size)),
            IntBinop::Add => a.wrapping_add(b) & mask,
            IntBinop::Sub => a.wrapping_sub(b) & mask,
            IntBinop::Mul => a.wrapping_mul(b) & mask,
            IntBinop::Div => {
                if b == 0 {
                    0
                } else {
                    a / b
                }
            }
            IntBinop::Rem => {
                if b == 0 {
                    0
                } else {
                    a % b
                }
            }
            IntBinop::Sdiv => {
                let l = signed_value(a, size);
                let r = signed_value(b, size);
                if r == 0 {
                    0
                } else {
                    l.overflowing_div(r).0 as u128 & mask
                }
            }
            IntBinop::Srem => {
                let l = signed_value(a, size);
                let r = signed_value(b, size);
                if r == 0 {
                    0
                } else {
                    l.overflowing_rem(r).0 as u128 & mask
                }
            }
            IntBinop::And => a & b,
            IntBinop::Or => a | b,
            IntBinop::Xor => a ^ b,
            IntBinop::ShiftLeft => a.wrapping_shl(b as u32) & mask,
            IntBinop::ShiftRight => a.wrapping_shr(b as u32) & mask,
            IntBinop::SShiftRight => (signed_value(a, size).wrapping_shr(b as u32) as u128) & mask,
        }
    }
}

impl Display for IntBinop {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            IntBinop::Equal => "==",
            IntBinop::NotEqual => "!=",
            IntBinop::Less => "<",
            IntBinop::SLess => "s<",
            IntBinop::LessEqual => "<=",
            IntBinop::SLessEqual => "s<=",
            IntBinop::Add => "+",
            IntBinop::Sub => "-",
            IntBinop::Xor => "^",
            IntBinop::And => "&",
            IntBinop::Or => "|",
            IntBinop::ShiftLeft => "<<",
            IntBinop::ShiftRight => ">>",
            IntBinop::SShiftRight => "s>>",
            IntBinop::Mul => "*",
            IntBinop::Div => "/",
            IntBinop::Rem => "%",
            IntBinop::Sdiv => "s/",
            IntBinop::Srem => "s%",
        };

        write!(f, "{}", s)
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatBinop {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Add,
    Sub,
    Mul,
    Div,
}

impl Display for FloatBinop {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            FloatBinop::Equal => "f==",
            FloatBinop::NotEqual => "f!=",
            FloatBinop::Less => "f<",
            FloatBinop::LessEqual => "f<=",
            FloatBinop::Add => "f+",
            FloatBinop::Sub => "f-",
            FloatBinop::Mul => "f*",
            FloatBinop::Div => "f/",
        };

        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use crate::{
        builder::Builder,
        value::insn::{Instruction, InstructionId, Mnemonic},
    };

    use super::*;

    macro_rules! assert_int_binop {
        ($expr:literal, $expected_op:expr, $size: literal, $expected_stmt:literal $(,)?) => {{
            let mut ctx = Context::new();
            let mut builder = Builder::from_context(&mut ctx, 0x1000);

            qcode!(builder, "local i32 v0 as V0");
            let v1: InstructionId = qcode!(builder, $expr);
            builder.finalize(0x1001);

            match ctx.values.instructions[v1].clone() {
                Instruction {
                    mnemonic:
                        Mnemonic::Binop(Binary {
                            op: Binop::Int(op), ..
                        }),
                    size: $size,
                    ..
                } => assert_eq!(op, $expected_op),

                _ => panic!(
                    "expected i32 integer binop instruction, found {}",
                    ctx.get_insn(v1).as_statement()
                ),
            }

            assert_eq!(ctx.get_insn(v1).as_statement().to_string(), $expected_stmt);
        }};
    }

    macro_rules! assert_bool_binop {
        ($expr:literal, $expected_op:expr $(,)?) => {{
            let mut ctx = Context::new();
            let mut builder = Builder::from_context(&mut ctx, 0x1000);

            qcode!(builder, "local i8 v0 as V0; local i8 v1 as V1");
            let v2: InstructionId = qcode!(builder, $expr);
            builder.finalize(0x1001);

            match ctx.values.instructions[v2].clone() {
                Instruction {
                    mnemonic:
                        Mnemonic::Binop(Binary {
                            op: Binop::Bool(op),
                            ..
                        }),
                    ..
                } => assert_eq!(op, $expected_op),

                _ => panic!("expected boolean binop instruction"),
            }
        }};
    }

    #[test]
    fn test_add_display() {
        assert_int_binop!(
            "i32 {v0} + i32 0x2",
            IntBinop::Add,
            4,
            "i32 %tmp1 = i32 %v0 + 0x2;"
        );
    }

    #[test]
    fn test_sub_display() {
        assert_int_binop!(
            "i32 {v0} - i32 0x2",
            IntBinop::Sub,
            4,
            "i32 %tmp1 = i32 %v0 - 0x2;"
        );
    }

    #[test]
    fn test_mul_display() {
        assert_int_binop!(
            "i32 {v0} * i32 0x2",
            IntBinop::Mul,
            4,
            "i32 %tmp1 = i32 %v0 * 0x2;"
        );
    }

    #[test]
    fn test_div_display() {
        assert_int_binop!(
            "i32 {v0} / i32 0x2",
            IntBinop::Div,
            4,
            "i32 %tmp1 = i32 %v0 / 0x2;"
        );
    }

    #[test]
    fn test_bit_and_display() {
        assert_int_binop!(
            "i32 {v0} & i32 0x2",
            IntBinop::And,
            4,
            "i32 %tmp1 = i32 %v0 & 0x2;"
        );
    }

    #[test]
    fn test_bit_or_display() {
        assert_int_binop!(
            "i32 {v0} | i32 0x2",
            IntBinop::Or,
            4,
            "i32 %tmp1 = i32 %v0 | 0x2;"
        );
    }

    #[test]
    fn test_bit_xor_display() {
        assert_int_binop!(
            "i32 {v0} ^ i32 0x2",
            IntBinop::Xor,
            4,
            "i32 %tmp1 = i32 %v0 ^ 0x2;"
        );
    }

    #[test]
    fn test_shl_display() {
        assert_int_binop!(
            "i32 {v0} << i32 0x2",
            IntBinop::ShiftLeft,
            4,
            "i32 %tmp1 = i32 %v0 << 0x2;",
        );
    }

    #[test]
    fn test_shr_display() {
        assert_int_binop!(
            "i32 {v0} >> i32 0x2",
            IntBinop::ShiftRight,
            4,
            "i32 %tmp1 = i32 %v0 >> 0x2;",
        );
    }

    #[test]
    fn test_eq_display() {
        assert_int_binop!(
            "i32 {v0} == i32 0x2",
            IntBinop::Equal,
            1,
            "i8 %tmp1 = i32 %v0 == 0x2;"
        );
    }

    #[test]
    fn test_ne_display() {
        assert_int_binop!(
            "i32 {v0} != i32 0x2",
            IntBinop::NotEqual,
            1,
            "i8 %tmp1 = i32 %v0 != 0x2;"
        );
    }

    #[test]
    fn test_lt_display() {
        assert_int_binop!(
            "i32 {v0} < i32 0x2",
            IntBinop::Less,
            1,
            "i8 %tmp1 = i32 %v0 < 0x2;"
        );
    }

    #[test]
    fn test_le_display() {
        assert_int_binop!(
            "i32 {v0} <= i32 0x2",
            IntBinop::LessEqual,
            1,
            "i8 %tmp1 = i32 %v0 <= 0x2;"
        );
    }

    #[test]
    fn test_gt_display() {
        assert_int_binop!(
            "i32 {v0} > i32 0x2",
            IntBinop::Less,
            1,
            "i8 %tmp1 = 0x2 < i32 %v0;",
        );
    }

    #[test]
    fn test_ge_display() {
        assert_int_binop!(
            "i32 {v0} >= i32 0x2",
            IntBinop::LessEqual,
            1,
            "i8 %tmp1 = 0x2 <= i32 %v0;",
        );
    }

    #[test]
    fn test_bool_xor_from_qcode() {
        assert_bool_binop!("{v0} ^^ {v1}", BoolBinop::Xor);
    }

    #[test]
    fn test_bool_and_from_qcode() {
        assert_bool_binop!("{v0} && {v1}", BoolBinop::And);
    }

    #[test]
    fn test_bool_or_from_qcode() {
        assert_bool_binop!("{v0} || {v1}", BoolBinop::Or);
    }

    #[test]
    #[should_panic(expected = "qcode size mismatch for value")]
    fn test_explicit_capture_size_mismatch_panics() {
        let mut ctx = Context::new();
        let mut builder = Builder::from_context(&mut ctx, 0x1000);

        qcode!(builder, "local i32 v0 as V0");
        let _ = qcode!(builder, "i16 {v0} + i16 0x2");
    }

    #[test]
    #[should_panic(expected = "qcode size mismatch in binary expression: lhs=4 rhs=2")]
    fn test_binary_operand_size_mismatch_panics() {
        let mut ctx = Context::new();
        let mut builder = Builder::from_context(&mut ctx, 0x1000);

        qcode!(builder, "local i32 v0 as V0");
        let _ = qcode!(builder, "i32 {v0} + i16 0x2");
    }
}
