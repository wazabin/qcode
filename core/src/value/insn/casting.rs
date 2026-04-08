use crate::{
    context::Context,
    value::{
        ValueId,
        insn::bits::{mask_for_size, signed_value},
    },
};
use std::fmt::Formatter;

use super::mnemonic::MnemonicKind;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Zext {
    pub src: ValueId,
    pub size: usize,
}

impl Zext {
    /// Zero-extends `value` to `dst_size` bytes by masking off any higher bits.
    pub fn eval(value: u128, dst_size: usize) -> u128 {
        value & mask_for_size(dst_size)
    }
}

impl MnemonicKind for Zext {
    fn opcode(&self) -> &'static str {
        "zext"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "zext(i{}, {});", self.size * 8, ctx.get_value(self.src))
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.src]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Sext {
    pub src: ValueId,
    pub size: usize,
}

impl Sext {
    /// Sign-extends `value` from `src_size` bytes to `dst_size` bytes.
    pub fn eval(value: u128, src_size: usize, dst_size: usize) -> u128 {
        if src_size == 0 {
            return 0;
        }
        signed_value(value & mask_for_size(src_size), src_size) as u128 & mask_for_size(dst_size)
    }
}

impl MnemonicKind for Sext {
    fn opcode(&self) -> &'static str {
        "sext"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "sext(i{}, {});", self.size * 8, ctx.get_value(self.src))
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.src]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Range {
    pub src: ValueId,
    pub start: usize,
    pub size: usize,
}

impl Range {
    /// Extracts `size` bytes starting at byte offset `start` from `value`.
    pub fn eval(value: u128, start: usize, size: usize) -> u128 {
        use super::bits::mask_for_size;
        let shift = start.saturating_mul(8);
        if shift >= u128::BITS as usize {
            0
        } else {
            (value >> shift) & mask_for_size(size)
        }
    }
}

impl MnemonicKind for Range {
    fn opcode(&self) -> &'static str {
        "range"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}[{}:{}];",
            ctx.get_value(self.src),
            self.start,
            self.start + self.size
        )
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.src]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntToFloat {
    pub src: ValueId,
    pub size: usize,
}

impl MnemonicKind for IntToFloat {
    fn opcode(&self) -> &'static str {
        "int_to_float"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(
            f,
            "int2float(f{}, {});",
            self.size * 8,
            ctx.get_value(self.src)
        )
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.src]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FloatToFloat {
    pub src: ValueId,
    pub size: usize,
}

impl MnemonicKind for FloatToFloat {
    fn opcode(&self) -> &'static str {
        "float_to_float"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(
            f,
            "float2float(f{}, {});",
            self.size * 8,
            ctx.get_value(self.src)
        )
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.src]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FloatToInt {
    pub src: ValueId,
    pub size: usize,
}

impl MnemonicKind for FloatToInt {
    fn opcode(&self) -> &'static str {
        "float_to_int"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "trunc(i{}, {});", self.size * 8, ctx.get_value(self.src))
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.src]
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

    macro_rules! assert_cast {
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

                _ => panic!("expected cast instruction"),
            }

            assert_eq!(ctx.get_insn(v1).as_statement().to_string(), $expected_stmt);
        }};
    }

    #[test]
    fn test_zext_display() {
        assert_cast!(
            "zext(i64, i32 {v0})",
            Mnemonic::Zext(Zext { size: 8, .. }),
            "i64 %tmp1 = zext(i64, i32 %v0);",
            8
        );
    }

    #[test]
    fn test_sext_display() {
        assert_cast!(
            "sext(i64, i32 {v0})",
            Mnemonic::Sext(Sext { size: 8, .. }),
            "i64 %tmp1 = sext(i64, i32 %v0);",
            8
        );
    }

    #[test]
    fn test_int2float_display() {
        assert_cast!(
            "int2float(f32, i32 {v0})",
            Mnemonic::IntToFloat(IntToFloat { size: 4, .. }),
            "i32 %tmp1 = int2float(f32, i32 %v0);",
            4
        );
    }

    #[test]
    fn test_float2float_display() {
        assert_cast!(
            "float2float(f64, i32 {v0})",
            Mnemonic::FloatToFloat(FloatToFloat { size: 8, .. }),
            "i64 %tmp1 = float2float(f64, i32 %v0);",
            8
        );
    }

    #[test]
    fn test_trunc_display() {
        assert_cast!(
            "trunc(i16, i32 {v0})",
            Mnemonic::FloatToInt(FloatToInt { size: 2, .. }),
            "i16 %tmp1 = trunc(i16, i32 %v0);",
            2
        );
    }

    #[test]
    fn test_zext_eval() {
        // zero-extend preserves low bytes, masks off anything above dst_size
        assert_eq!(Zext::eval(0xFF, 4), 0xFF);
        assert_eq!(Zext::eval(0xDEAD_BEEF_FF_FF_FF_FF, 4), 0xFFFF_FFFF);
        assert_eq!(Zext::eval(0, 8), 0);
    }

    #[test]
    fn test_sext_eval() {
        // positive value (sign bit clear) → unchanged
        assert_eq!(Sext::eval(0x7F, 1, 4), 0x7F);
        // negative value (sign bit set) → sign-extended
        assert_eq!(Sext::eval(0xFF, 1, 4), 0xFFFF_FFFF);
        assert_eq!(Sext::eval(0x80, 1, 4), 0xFFFF_FF80);
        // zero → zero
        assert_eq!(Sext::eval(0, 1, 4), 0);
    }

    #[test]
    fn test_range_eval() {
        // extract 2 bytes starting at byte offset 1 from 0xDEAD_BEEF
        // bytes: EF BE AD DE → bytes 1..3 = BE AD → value 0xADBE
        assert_eq!(Range::eval(0xDEAD_BEEF, 1, 2), 0xADBE);
        // extract first byte
        assert_eq!(Range::eval(0xDEAD_BEEF, 0, 1), 0xEF);
        // extract from offset beyond value → 0
        assert_eq!(Range::eval(0xFF, 16, 1), 0);
    }
}
