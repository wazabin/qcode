use crate::value::ValueId;

use super::mnemonic::MnemonicKind;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct IsFloatNaN {
    pub src: ValueId,
}

impl MnemonicKind for IsFloatNaN {
    fn opcode(&self) -> &'static str {
        "is_float_nan"
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.src]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
        // A zero value has no set bit inside the window, so `leading_zeros` runs
        // past the operand and reports the full u128 width. Clamp to the operand
        // bit-width (a no-op for any non-zero value).
        u128::from(count.min(bits as u64))
    }
}

impl MnemonicKind for LzCount {
    fn opcode(&self) -> &'static str {
        "lz_count"
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.src]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

    fn args(&self) -> Vec<ValueId> {
        vec![self.src]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

    fn args(&self) -> Vec<ValueId> {
        vec![self.lhs, self.rhs]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

    fn args(&self) -> Vec<ValueId> {
        vec![self.lhs, self.rhs]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

    fn args(&self) -> Vec<ValueId> {
        vec![self.lhs, self.rhs]
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use crate::context::Context;
    use crate::value::insn::{Instruction, Mnemonic};

    use super::*;

    #[test]
    fn test_nan_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
                %v = nan(%v0);
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        assert!(matches!(
            v.mnemonic(),
            Mnemonic::IsFloatNaN(IsFloatNaN { .. })
        ));
        assert_eq!(v.size(), 1);
        assert_eq!(v.as_statement().to_string(), "i8 %v = nan(i32 %v0);");
    }

    #[test]
    fn test_popcount_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
                %v = popcount(%v0);
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        assert!(matches!(v.mnemonic(), Mnemonic::PopCount(PopCount { .. })));
        assert_eq!(v.size(), 1);
        assert_eq!(v.as_statement().to_string(), "i8 %v = popcount(i32 %v0);");
    }

    #[test]
    fn test_lzcount_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(V0:4, V0);
                %v = lzcount(%v0);
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        assert!(matches!(v.mnemonic(), Mnemonic::LzCount(LzCount { .. })));
        assert_eq!(v.size(), 1);
        assert_eq!(v.as_statement().to_string(), "i8 %v = lzcount(i32 %v0);");
    }

    #[test]
    fn test_carry_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                local i32 V1;
                %v0 = load(V0:4, V0);
                %v1 = load(V1:4, V1);
                %v = carry(%v0, %v1);
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        assert!(matches!(v.mnemonic(), Mnemonic::Carry(Carry { .. })));
        assert_eq!(v.size(), 1);
        assert_eq!(
            v.as_statement().to_string(),
            "i8 %v = carry(i32 %v0, i32 %v1);"
        );
    }

    #[test]
    fn test_scarry_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                local i32 V1;
                %v0 = load(V0:4, V0);
                %v1 = load(V1:4, V1);
                %v = scarry(%v0, %v1);
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        assert!(matches!(v.mnemonic(), Mnemonic::SCarry(SCarry { .. })));
        assert_eq!(v.size(), 1);
        assert_eq!(
            v.as_statement().to_string(),
            "i8 %v = scarry(i32 %v0, i32 %v1);"
        );
    }

    #[test]
    fn test_sborrow_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                local i32 V1;
                %v0 = load(V0:4, V0);
                %v1 = load(V1:4, V1);
                %v = sborrow(%v0, %v1);
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        assert!(matches!(v.mnemonic(), Mnemonic::SBorrow(SBorrow { .. })));
        assert_eq!(v.size(), 1);
        assert_eq!(
            v.as_statement().to_string(),
            "i8 %v = sborrow(i32 %v0, i32 %v1);"
        );
    }
}
