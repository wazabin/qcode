use crate::value::{
    ValueId,
    insn::bits::{mask_for_size, signed_value},
};

use super::mnemonic::{Args, MnemonicKind};
use smallvec::smallvec;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

    fn args(&self) -> Args {
        smallvec![self.src]}
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

    fn args(&self) -> Args {
        smallvec![self.src]}
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

    fn args(&self) -> Args {
        smallvec![self.src]}
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct IntToFloat {
    pub src: ValueId,
    pub size: usize,
}

impl MnemonicKind for IntToFloat {
    fn opcode(&self) -> &'static str {
        "int_to_float"
    }

    fn args(&self) -> Args {
        smallvec![self.src]}
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct FloatToFloat {
    pub src: ValueId,
    pub size: usize,
}

impl MnemonicKind for FloatToFloat {
    fn opcode(&self) -> &'static str {
        "float_to_float"
    }

    fn args(&self) -> Args {
        smallvec![self.src]}
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct FloatToInt {
    pub src: ValueId,
    pub size: usize,
}

impl MnemonicKind for FloatToInt {
    fn opcode(&self) -> &'static str {
        "float_to_int"
    }

    fn args(&self) -> Args {
        smallvec![self.src]}
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use crate::context::Context;
    use crate::value::insn::{Instruction, Mnemonic};

    use super::*;

    #[test]
    fn test_zext_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
                i64 %v = zext(i64, i32 %v0);
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        if !matches!(v.mnemonic(), Mnemonic::Zext(Zext { size: 8, .. })) {
            panic!("expected zext instruction");
        }

        assert_eq!(v.as_statement().to_string(), "i64 %v = zext(i64, i32 %v0);");
    }

    #[test]
    fn test_sext_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
                i64 %v = sext(i64, i32 %v0);
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        if !matches!(v.mnemonic(), Mnemonic::Sext(Sext { size: 8, .. })) {
            panic!("expected sext instruction");
        }

        assert_eq!(v.as_statement().to_string(), "i64 %v = sext(i64, i32 %v0);");
    }

    #[test]
    fn test_int2float_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
                i64 %v = int2float(f32, i32 %v0);
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        if !matches!(
            v.mnemonic(),
            Mnemonic::IntToFloat(IntToFloat { size: 4, .. })
        ) {
            panic!("expected int2float instruction");
        }

        assert_eq!(
            v.as_statement().to_string(),
            "i32 %v = int2float(f32, i32 %v0);"
        );
    }

    #[test]
    fn test_float2float_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
                i64 %v = float2float(f64, i32 %v0);
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        if !matches!(
            v.mnemonic(),
            Mnemonic::FloatToFloat(FloatToFloat { size: 8, .. })
        ) {
            panic!("expected float2float instruction");
        }

        assert_eq!(
            v.as_statement().to_string(),
            "i64 %v = float2float(f64, i32 %v0);"
        );
    }

    #[test]
    fn test_trunc_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
                i16 %v = trunc(i16, i32 %v0);
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        if !matches!(
            v.mnemonic(),
            Mnemonic::FloatToInt(FloatToInt { size: 2, .. })
        ) {
            panic!("expected float2int instruction");
        }

        assert_eq!(
            v.as_statement().to_string(),
            "i16 %v = trunc(i16, i32 %v0);"
        );
    }

    #[test]
    fn test_zext_eval() {
        // zero-extend preserves low bytes, masks off anything above dst_size
        assert_eq!(Zext::eval(0xFF, 4), 0xFF);
        assert_eq!(Zext::eval(0xDEAD_BEEF_FFFF_FFFF, 4), 0xFFFF_FFFF);
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
