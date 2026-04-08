use crate::{context::Context, value::ValueId};
use std::fmt::Formatter;

use super::mnemonic::MnemonicKind;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IsFloatNaN {
    pub src: ValueId,
}

impl MnemonicKind for IsFloatNaN {
    fn opcode(&self) -> &'static str {
        "is_float_nan"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "nan({});", ctx.get_value(self.src))
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.src]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LzCount {
    pub src: ValueId,
}

impl LzCount {
    /// Counts the leading zero bits of `value` interpreted as a `size`-byte integer.
    pub fn eval(value: u128, size: usize) -> u128 {
        use super::bits::mask_for_size;
        let bits = size.saturating_mul(8);
        let masked = value & mask_for_size(size);
        let count = if bits == 0 {
            0u64
        } else if bits >= u128::BITS as usize {
            masked.leading_zeros() as u64
        } else {
            (masked << (u128::BITS as usize - bits)).leading_zeros() as u64
        };
        u128::from(count)
    }
}

impl MnemonicKind for LzCount {
    fn opcode(&self) -> &'static str {
        "lz_count"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "lzcount({});", ctx.get_value(self.src))
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.src]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PopCount {
    pub src: ValueId,
}

impl PopCount {
    /// Counts the number of set bits in `value` interpreted as a `size`-byte integer.
    pub fn eval(value: u128, size: usize) -> u128 {
        use super::bits::mask_for_size;
        u128::from((value & mask_for_size(size)).count_ones())
    }
}

impl MnemonicKind for PopCount {
    fn opcode(&self) -> &'static str {
        "pop_count"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "popcount({});", ctx.get_value(self.src))
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.src]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Carry {
    pub lhs: ValueId,
    pub rhs: ValueId,
}

impl Carry {
    /// Returns `1` if the unsigned addition of `lhs` and `rhs` carries out of `size` bytes.
    pub fn eval(lhs: u128, rhs: u128, size: usize) -> u128 {
        use super::bits::mask_for_size;
        let mask = mask_for_size(size);
        let a = lhs & mask;
        let b = rhs & mask;
        let carry = if size >= 16 {
            a.overflowing_add(b).1
        } else {
            a + b > mask
        };
        u128::from(carry)
    }
}

impl MnemonicKind for Carry {
    fn opcode(&self) -> &'static str {
        "carry"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(
            f,
            "carry({}, {});",
            ctx.get_value(self.lhs),
            ctx.get_value(self.rhs)
        )
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.lhs, self.rhs]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SCarry {
    pub lhs: ValueId,
    pub rhs: ValueId,
}

impl SCarry {
    /// Returns `1` if the signed addition of `lhs` and `rhs` overflows for `size`-byte integers.
    pub fn eval(lhs: u128, rhs: u128, size: usize) -> u128 {
        use super::bits::mask_for_size;
        let bits = size.saturating_mul(8);
        let mask = mask_for_size(size);
        let a = lhs & mask;
        let b = rhs & mask;
        let result = a.wrapping_add(b) & mask;
        let sign_bit = 1u128 << (bits.saturating_sub(1));
        let overflow = (a ^ result) & (b ^ result) & sign_bit != 0;
        u128::from(overflow)
    }
}

impl MnemonicKind for SCarry {
    fn opcode(&self) -> &'static str {
        "scarry"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(
            f,
            "scarry({}, {});",
            ctx.get_value(self.lhs),
            ctx.get_value(self.rhs)
        )
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.lhs, self.rhs]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SBorrow {
    pub lhs: ValueId,
    pub rhs: ValueId,
}

impl SBorrow {
    /// Returns `1` if the signed subtraction `lhs - rhs` overflows for `size`-byte integers.
    pub fn eval(lhs: u128, rhs: u128, size: usize) -> u128 {
        use super::bits::mask_for_size;
        let bits = size.saturating_mul(8);
        let mask = mask_for_size(size);
        let a = lhs & mask;
        let b = rhs & mask;
        let result = a.wrapping_sub(b) & mask;
        let sign_bit = 1u128 << (bits.saturating_sub(1));
        let overflow = (a ^ b) & (a ^ result) & sign_bit != 0;
        u128::from(overflow)
    }
}

impl MnemonicKind for SBorrow {
    fn opcode(&self) -> &'static str {
        "sborrow"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(
            f,
            "sborrow({}, {});",
            ctx.get_value(self.lhs),
            ctx.get_value(self.rhs)
        )
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.lhs, self.rhs]
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

    macro_rules! assert_unary_flag {
        ($expr:literal, $match_pat:pat, $expected_stmt:literal, $size:expr) => {{
            let mut ctx = Context::new();
            let mut builder = Builder::from_context(&mut ctx, 0x1000);

            qcode!(builder, "local i32 v0 as V0");
            let v1: InstructionId = qcode!(builder, $expr);
            builder.finalize(0x1001);

            match ctx.values.instructions[v1].clone() {
                Instruction {
                    mnemonic: $match_pat,
                    size: $size,
                    ..
                } => {}

                _ => panic!("expected unary flag instruction"),
            }

            assert_eq!(ctx.get_insn(v1).as_statement().to_string(), $expected_stmt);
        }};
    }

    macro_rules! assert_binary_flag {
        ($expr:literal, $match_pat:pat, $expected_stmt:literal, $size:expr) => {{
            let mut ctx = Context::new();
            let mut builder = Builder::from_context(&mut ctx, 0x1000);

            qcode!(builder, "local i32 v0 as V0; local i32 v1 as V1");
            let v2: InstructionId = qcode!(builder, $expr);
            builder.finalize(0x1001);

            match ctx.values.instructions[v2].clone() {
                Instruction {
                    mnemonic: $match_pat,
                    size: $size,
                    ..
                } => {}

                _ => panic!("expected binary flag instruction"),
            }

            assert_eq!(ctx.get_insn(v2).as_statement().to_string(), $expected_stmt);
        }};
    }

    #[test]
    fn test_nan_display() {
        assert_unary_flag!(
            "nan({v0})",
            Mnemonic::IsFloatNaN(IsFloatNaN { .. }),
            "i8 %tmp1 = nan(i32 %v0);",
            1
        );
    }

    #[test]
    fn test_popcount_display() {
        assert_unary_flag!(
            "popcount({v0})",
            Mnemonic::PopCount(PopCount { .. }),
            "i8 %tmp1 = popcount(i32 %v0);",
            1
        );
    }

    #[test]
    fn test_lzcount_display() {
        assert_unary_flag!(
            "lzcount({v0})",
            Mnemonic::LzCount(LzCount { .. }),
            "i8 %tmp1 = lzcount(i32 %v0);",
            1
        );
    }

    #[test]
    fn test_carry_display() {
        assert_binary_flag!(
            "carry({v0}, {v1})",
            Mnemonic::Carry(Carry { .. }),
            "i8 %tmp2 = carry(i32 %v0, i32 %v1);",
            1
        );
    }

    #[test]
    fn test_scarry_display() {
        assert_binary_flag!(
            "scarry({v0}, {v1})",
            Mnemonic::SCarry(SCarry { .. }),
            "i8 %tmp2 = scarry(i32 %v0, i32 %v1);",
            1
        );
    }

    #[test]
    fn test_sborrow_display() {
        assert_binary_flag!(
            "sborrow({v0}, {v1})",
            Mnemonic::SBorrow(SBorrow { .. }),
            "i8 %tmp2 = sborrow(i32 %v0, i32 %v1);",
            1
        );
    }
}
