use std::fmt::{Display, Formatter};

use crate::{
    context::Context,
    value::{ValueId, ValueRef},
};

use super::mnemonic::MnemonicKind;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
            ValueRef::new(self.lhs, ctx),
            self.op,
            ValueRef::new(self.rhs, ctx)
        )
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.lhs, self.rhs]
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
            IntBinop::Div => a.checked_div(b).unwrap_or(0),
            IntBinop::Rem => a.checked_rem(b).unwrap_or(0),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

    use crate::value::insn::{Instruction, Mnemonic};

    use super::*;

    #[test]
    fn test_add_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 + i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::Add),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 4);
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 + 0x2;");
    }

    #[test]
    fn test_sub_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 - i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::Sub),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 4);
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 - 0x2;");
    }

    #[test]
    fn test_mul_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
               local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 * i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::Mul),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 4);
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 * 0x2;");
    }

    #[test]
    fn test_div_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 / i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::Div),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 4);
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 / 0x2;");
    }

    #[test]
    fn test_bit_and_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 & i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::And),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 4);
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 & 0x2;");
    }

    #[test]
    fn test_bit_or_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 | i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::Or),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 4);
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 | 0x2;");
    }

    #[test]
    fn test_bit_xor_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 ^ i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::Xor),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 4);
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 ^ 0x2;");
    }

    #[test]
    fn test_shl_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 << i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::ShiftLeft),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 4);
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 << 0x2;");
    }

    #[test]
    fn test_shr_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 >> i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::ShiftRight),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 4);
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 >> 0x2;");
    }

    #[test]
    fn test_eq_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 == i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::Equal),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 1);
        assert_eq!(v.as_statement().to_string(), "i8 %v = i32 %v0 == 0x2;");
    }

    #[test]
    fn test_ne_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 != i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::NotEqual),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 1);
        assert_eq!(v.as_statement().to_string(), "i8 %v = i32 %v0 != 0x2;");
    }

    #[test]
    fn test_lt_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 < i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::Less),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 1);
        assert_eq!(v.as_statement().to_string(), "i8 %v = i32 %v0 < 0x2;");
    }

    #[test]
    fn test_le_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 <= i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::LessEqual),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 1);
        assert_eq!(v.as_statement().to_string(), "i8 %v = i32 %v0 <= 0x2;");
    }

    #[test]
    fn test_gt_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 > i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::Less),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 1);
        assert_eq!(v.as_statement().to_string(), "i8 %v = 0x2 < i32 %v0;");
    }

    #[test]
    fn test_ge_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i32 %v0 >= i32 0x2;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op), ..
            }) => assert_eq!(*op, IntBinop::LessEqual),
            _ => panic!("expected i32 integer binop instruction, found {}", v),
        }

        assert_eq!(v.size(), 1);
        assert_eq!(v.as_statement().to_string(), "i8 %v = 0x2 <= i32 %v0;");
    }

    #[test]
    fn test_bool_xor_from_qcode() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i8 V0;
                local i8 V1;
                %v0 = load(i8, V0);
                %v1 = load(i8, V1);
                %v = %v0 ^^ %v1;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Bool(op),
                ..
            }) => assert_eq!(*op, BoolBinop::Xor),
            _ => panic!("expected boolean binop instruction"),
        }

        assert_eq!(v.size(), 1);
    }

    #[test]
    fn test_bool_and_from_qcode() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i8 V0;
                local i8 V1;
                %v0 = load(i8, V0);
                %v1 = load(i8, V1);
                %v = %v0 && %v1;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Bool(op),
                ..
            }) => assert_eq!(*op, BoolBinop::And),
            _ => panic!("expected boolean binop instruction"),
        }

        assert_eq!(v.size(), 1);
    }

    #[test]
    fn test_bool_or_from_qcode() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i8 V0;
                local i8 V1;
                %v0 = load(i8, V0);
                %v1 = load(i8, V1);
                %v = %v0 || %v1;
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        match v.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Bool(op),
                ..
            }) => assert_eq!(*op, BoolBinop::Or),
            _ => panic!("expected boolean binop instruction"),
        }

        assert_eq!(v.size(), 1);
    }

    #[test]
    #[should_panic(expected = "qcode size mismatch")]
    fn test_explicit_capture_size_mismatch_panics() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %v = i16 %v0 + i16 0x2;
                goto <0x1001>;
            "
        );
    }
}
