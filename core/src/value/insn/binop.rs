use std::fmt::{Display, Formatter};

use crate::value::LocalValueId;

use super::mnemonic::{Args, MnemonicKind};
use smallvec::smallvec;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Binary {
    pub op: Binop,
    pub lhs: LocalValueId,
    pub rhs: LocalValueId,
}

impl MnemonicKind for Binary {
    fn opcode(&self) -> &'static str {
        "binop"
    }

    fn args(&self) -> Args {
        smallvec![self.lhs, self.rhs]
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Binop {
    Int(IntBinop),
    Float(FloatBinop),
}

impl Display for Binop {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Binop::Int(op) => write!(f, "{}", op),
            Binop::Float(op) => write!(f, "{}", op),
        }
    }
}

impl Binop {
    pub fn is_comparison(self) -> bool {
        match self {
            Binop::Int(op) => op.is_comparison(),
            Binop::Float(op) => op.is_comparison(),
        }
    }

    pub fn is_shift(self) -> bool {
        matches!(
            self,
            Binop::Int(IntBinop::ShiftLeft | IntBinop::ShiftRight | IntBinop::SShiftRight)
        )
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
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            IntBinop::Equal
                | IntBinop::NotEqual
                | IntBinop::Less
                | IntBinop::SLess
                | IntBinop::LessEqual
                | IntBinop::SLessEqual
        )
    }

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
            // The p-code shift operations take the shift amount as the full
            // unsigned value of input1, which may be wider than input0 and is
            // therefore not masked to `size`. A count at or beyond input0's
            // width shifts every bit out rather than wrapping around.
            IntBinop::ShiftLeft => shift_amount(rhs, size).map_or(0, |n| (a << n) & mask),
            IntBinop::ShiftRight => shift_amount(rhs, size).map_or(0, |n| (a >> n) & mask),
            IntBinop::SShiftRight => {
                let signed = signed_value(a, size);
                match shift_amount(rhs, size) {
                    Some(n) => (signed >> n) as u128 & mask,
                    // Shifting past the width leaves a copy of the sign bit.
                    None if signed < 0 => mask,
                    None => 0,
                }
            }
        }
    }
}

/// Returns the p-code shift amount for an `size`-byte input0, or `None` when
/// the count is at or beyond input0's width and every bit is shifted out.
fn shift_amount(rhs: u128, size: usize) -> Option<u32> {
    let bits = u128::from(u32::try_from(size).ok()?.saturating_mul(8));
    (rhs < bits).then_some(rhs as u32)
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

impl FloatBinop {
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            FloatBinop::Equal | FloatBinop::NotEqual | FloatBinop::Less | FloatBinop::LessEqual
        )
    }
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
    use wazabin_qcode_macro::qcode;

    use crate::context::Context;
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
                %v0 = load(V0:4, V0);
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
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 + i32 0x2;");
    }

    #[test]
    fn test_sub_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
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
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 - i32 0x2;");
    }

    #[test]
    fn test_mul_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
               local i32 V0;
                %v0 = load(V0:4, V0);
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
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 * i32 0x2;");
    }

    #[test]
    fn test_div_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
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
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 / i32 0x2;");
    }

    #[test]
    fn test_bit_and_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
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
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 & i32 0x2;");
    }

    #[test]
    fn test_bit_or_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
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
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 | i32 0x2;");
    }

    #[test]
    fn test_bit_xor_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
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
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 ^ i32 0x2;");
    }

    /// p-code takes a shift amount as the full unsigned value of input1,
    /// which may be wider than input0. A count at or beyond input0's width
    /// shifts every bit out instead of wrapping, and an arithmetic right
    /// shift leaves a copy of the sign bit.
    #[test]
    fn shift_amount_is_not_masked_to_the_operand_width() {
        // A count that only looks small once truncated to input0's width.
        // MMX shift-by-register relies on this: PSRLW's count is the whole
        // 64-bit source, so 0xc000_0000_0000_0001 must empty each 16-bit lane
        // rather than shift it by one.
        let wide = 0xc000_0000_0000_0001u128;
        assert_eq!(IntBinop::ShiftRight.eval(0x8000, wide, 2), 0);
        assert_eq!(IntBinop::ShiftLeft.eval(0x8000, wide, 2), 0);
        assert_eq!(IntBinop::SShiftRight.eval(0x8000, wide, 2), 0xffff);
        assert_eq!(IntBinop::SShiftRight.eval(0x7fff, wide, 2), 0);

        // Exactly at the width every bit is still shifted out.
        assert_eq!(IntBinop::ShiftRight.eval(0xffff, 16, 2), 0);
        assert_eq!(IntBinop::ShiftLeft.eval(0xffff, 16, 2), 0);
        assert_eq!(IntBinop::SShiftRight.eval(0x8000, 16, 2), 0xffff);

        // Ordinary in-range counts are unaffected.
        assert_eq!(IntBinop::ShiftRight.eval(0x8000, 1, 2), 0x4000);
        assert_eq!(IntBinop::ShiftLeft.eval(0x0001, 15, 2), 0x8000);
        assert_eq!(IntBinop::SShiftRight.eval(0x8000, 1, 2), 0xc000);
        assert_eq!(IntBinop::ShiftRight.eval(0x8000, 0, 2), 0x8000);

        // A full-width 128-bit operand has no count that fits in u32 alone.
        assert_eq!(IntBinop::ShiftLeft.eval(1, 127, 16), 1u128 << 127);
        assert_eq!(IntBinop::ShiftLeft.eval(1, 128, 16), 0);
    }

    #[test]
    fn test_shl_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
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
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 << i32 0x2;");
    }

    #[test]
    fn test_shr_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
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
        assert_eq!(v.as_statement().to_string(), "i32 %v = i32 %v0 >> i32 0x2;");
    }

    #[test]
    fn test_eq_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
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
        assert_eq!(
            v.as_statement().to_string(),
            "bool %v = i32 %v0 == i32 0x2;"
        );
    }

    #[test]
    fn test_ne_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
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
        assert_eq!(
            v.as_statement().to_string(),
            "bool %v = i32 %v0 != i32 0x2;"
        );
    }

    #[test]
    fn test_lt_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
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
        assert_eq!(v.as_statement().to_string(), "bool %v = i32 %v0 < i32 0x2;");
    }

    #[test]
    fn test_le_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
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
        assert_eq!(
            v.as_statement().to_string(),
            "bool %v = i32 %v0 <= i32 0x2;"
        );
    }

    #[test]
    fn test_gt_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
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
        assert_eq!(v.as_statement().to_string(), "bool %v = i32 0x2 < i32 %v0;");
    }

    #[test]
    fn test_ge_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
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
        assert_eq!(
            v.as_statement().to_string(),
            "bool %v = i32 0x2 <= i32 %v0;"
        );
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
                %v0 = load(V0:4, V0);
                %v = i16 %v0 + i16 0x2;
                goto <0x1001>;
            "
        );
    }
}
