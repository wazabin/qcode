use crate::{DomainMemory, EmulatorError, EmulatorErrorKind, Interpreter};
use qcode::{
    context::Context,
    space::SpaceId,
    value::{
        BasicBlock, BlockId, BlockParamId, BlockRef, Function, FunctionId, Instruction, Value,
        ValueId, ValueRef, Varnode,
        insn::{
            BoolBinop, Branch, BranchInd, CBranch, Call, CallInd, Carry, InstructionId,
            InstructionRef, IntBinop, LzCount, Mnemonic, PopCount, Range, Return, SBorrow, SCarry,
            Sext, Unop, Zext,
        },
        varnode::{VarnodeId, register::RegisterId},
    },
};
use std::{cmp, collections::HashMap};

use super::DomainValue;

#[derive(Debug, Default, Clone)]
pub struct EmulatedSpace(HashMap<u64, u8>);

impl EmulatedSpace {
    pub fn read_byte(&self, addr: u64) -> Result<u8, EmulatorErrorKind> {
        self.0
            .get(&addr)
            .copied()
            .ok_or(EmulatorErrorKind::MemoryReadError(addr))
    }

    pub fn read(&self, addr: u64, size: usize) -> Result<Vec<u8>, EmulatorErrorKind> {
        (0..size).map(|i| self.read_byte(addr + i as u64)).collect()
    }

    pub fn write_byte(&mut self, addr: u64, value: u8) {
        self.0.insert(addr, value);
    }

    /// Returns an editable region of this space for the given address and size.
    /// If `addr + size` would overflow, a zero-size region is returned (reads yield 0, writes are no-ops).
    pub fn get_mut_region(
        &mut self,
        addr: u64,
        size: usize,
    ) -> Result<EmulatedSpaceRegion<'_>, EmulatorErrorKind> {
        let end = addr
            .checked_add(size as u64)
            .ok_or(EmulatorErrorKind::AddressOverflow(addr, size))?;
        Ok(EmulatedSpaceRegion::new(self, addr, end))
    }

    /// Reads a little-endian unsigned integer from the region
    pub fn read_u128(&self, addr: u64, size: u64) -> Result<u128, EmulatorErrorKind> {
        let mut res = 0u128;

        for cur in addr..addr + cmp::min(size, 16) {
            let byte = self.read_byte(cur)?;
            res |= u128::from(byte) << ((cur - addr) * 8);
        }

        Ok(res)
    }
}

/// An exclusive region of an emulated space, used for reading/writing a contiguous range of addresses in a space.
pub struct EmulatedSpaceRegion<'space> {
    space: &'space mut EmulatedSpace,
    start: u64,
    end: u64,
}

impl<'space> EmulatedSpaceRegion<'space> {
    pub fn new(space: &'space mut EmulatedSpace, start: u64, end: u64) -> Self {
        Self { space, start, end }
    }

    /// The size of the region in bytes
    pub fn size(&self) -> usize {
        (self.end - self.start) as usize
    }

    /// Reads a region as a byte vector
    pub fn read_bytes(&self, addr: u64, size: usize) -> Result<Vec<u8>, EmulatorErrorKind> {
        assert!(addr >= self.start && addr + size as u64 <= self.end);
        self.space.read(addr, size)
    }

    /// Writes a little-endian unsigned integer to the region
    pub fn write_u128(&mut self, value: u128) {
        let end = self.start + cmp::min(16, self.size()) as u64;
        for addr in self.start..end {
            let byte = u8::try_from((value >> ((addr - self.start) * 8)) & 0xffu128).unwrap();
            self.space.write_byte(addr, byte);
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct EmulatedMemory {
    spaces: HashMap<SpaceId, EmulatedSpace>,
}

impl EmulatedMemory {
    fn read_raw(
        &self,
        space: SpaceId,
        addr: u64,
        size: usize,
    ) -> Result<Vec<u8>, EmulatorErrorKind> {
        self.spaces
            .get(&space)
            .ok_or(EmulatorErrorKind::UnknownSpace(space))?
            .read(addr, size)
    }
}

fn bool_to_u64(value: bool) -> u64 {
    if value { 1 } else { 0 }
}

fn mask_for_size(size: usize) -> u128 {
    let bits = size.saturating_mul(8);
    if bits >= u128::BITS as usize {
        u128::MAX
    } else if bits == 0 {
        0u128
    } else {
        (1u128 << bits) - 1
    }
}

fn u128_to_u64(value: u128) -> u64 {
    u64::try_from(value & u128::from(u64::MAX)).unwrap()
}

#[derive(Debug, Clone, Copy)]
pub struct SizedValue {
    /// Raw bits for this value
    value: u128,

    /// Number of bytes that are actually used in this value.
    size: u8,
}

impl SizedValue {
    pub fn new(value: u64, size: usize) -> Self {
        let size = cmp::min(size, 16) as u8;
        let value = u128::from(value) & mask_for_size(size as usize);
        Self { value, size }
    }

    fn from_bits(value: u128, size: usize) -> Self {
        let size = cmp::min(size, 16) as u8;
        let value = value & mask_for_size(size as usize);
        Self { value, size }
    }

    fn as_u64(&self) -> u64 {
        u128_to_u64(self.value & mask_for_size(self.size as usize))
    }

    fn as_bits(&self) -> u128 {
        self.value & mask_for_size(self.size as usize)
    }

    fn signed_value(&self) -> i128 {
        let bits = (self.size as usize).saturating_mul(8);
        if bits == 0 {
            return i128::from(0i8);
        }
        if bits >= u128::BITS as usize {
            return self.as_bits() as i128;
        }

        let value = self.as_bits();
        let sign_bit = u128::from(1u8) << (bits - 1);
        let extended = if (value & sign_bit) != u128::from(0u8) {
            value | !mask_for_size(self.size as usize)
        } else {
            value
        };
        extended as i128
    }

    fn widen_size(&self, other: &Self) -> usize {
        if self.size != other.size {
            return self.size as usize;
        }

        self.size as usize
    }

    fn f64_from_self(&self) -> f64 {
        match self.size as usize {
            0..=4 => f32::from_bits(self.as_u64() as u32) as f64,
            _ => f64::from_bits(self.as_u64()),
        }
    }

    fn from_f64(value: f64, size: usize) -> Self {
        match size {
            0..=4 => Self::new((value as f32).to_bits() as u64, 4),
            _ => Self::new(value.to_bits(), 8),
        }
    }
}

impl DomainValue for SizedValue {
    fn size(&self) -> Result<usize, EmulatorErrorKind> {
        Ok(self.size as usize)
    }

    fn value(&self) -> Result<u64, EmulatorErrorKind> {
        let value = self.as_bits();
        if value > u128::from(u64::MAX) {
            Err(EmulatorErrorKind::ValueError(value)) // Replace with appropriate error
        } else {
            Ok(u64::try_from(value).unwrap())
        }
    }

    fn from_u64(value: u64) -> Self {
        Self::new(value, 8)
    }

    fn is_float_nan(&self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::new(bool_to_u64(self.f64_from_self().is_nan()), 1))
    }

    fn int_to_float(&self, size: usize) -> Result<Self, EmulatorErrorKind> {
        let signed = self.signed_value();
        match size {
            4 => Ok(Self::new((signed as f32).to_bits() as u64, 4)),
            8 => Ok(Self::new((signed as f64).to_bits(), 8)),
            _ => Ok(Self::new(0, size)),
        }
    }

    fn float_to_float(&self, size: usize) -> Result<Self, EmulatorErrorKind> {
        match size {
            4 => Ok(Self::new((self.f64_from_self() as f32).to_bits() as u64, 4)),
            8 => Ok(Self::new(self.f64_from_self().to_bits(), 8)),
            _ => Ok(Self::new(0, size)),
        }
    }

    fn float_to_int(&self, size: usize) -> Result<Self, EmulatorErrorKind> {
        let value = self.f64_from_self() as i64 as u64;
        Ok(Self::new(value, size))
    }

    fn zext(&self, size: usize) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_bits(Zext::eval(self.as_bits(), size), size))
    }

    fn sext(&self, size: usize) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_bits(
            Sext::eval(self.as_bits(), self.size as usize, size),
            size,
        ))
    }

    fn range(&self, start: usize, size: usize) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_bits(
            Range::eval(self.as_bits(), start, size),
            size,
        ))
    }

    fn pop_count(&self) -> Result<Self, EmulatorErrorKind> {
        let size = self.size as usize;
        Ok(Self::from_bits(PopCount::eval(self.as_bits(), size), size))
    }

    fn lz_count(&self) -> Result<Self, EmulatorErrorKind> {
        let size = self.size as usize;
        Ok(Self::from_bits(LzCount::eval(self.as_bits(), size), size))
    }

    fn carry(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_bits(
            Carry::eval(self.as_bits(), other.as_bits(), size),
            1,
        ))
    }

    fn scarry(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_bits(
            SCarry::eval(self.as_bits(), other.as_bits(), size),
            1,
        ))
    }

    fn sborrow(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_bits(
            SBorrow::eval(self.as_bits(), other.as_bits(), size),
            1,
        ))
    }

    fn int_not(&self) -> Result<Self, EmulatorErrorKind> {
        let size = self.size as usize;
        Ok(Self::from_bits(
            Unop::IntNot.eval_int(self.as_bits(), size).unwrap(),
            size,
        ))
    }

    fn bool_not(&self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_bits(
            Unop::BoolNot
                .eval_int(self.as_bits(), self.size as usize)
                .unwrap(),
            1,
        ))
    }

    fn int_negate(&self) -> Result<Self, EmulatorErrorKind> {
        let size = self.size as usize;
        Ok(Self::from_bits(
            Unop::IntNegate.eval_int(self.as_bits(), size).unwrap(),
            size,
        ))
    }

    fn float_negate(&self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_f64(-self.f64_from_self(), self.size as usize))
    }

    fn float_abs(&self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_f64(
            self.f64_from_self().abs(),
            self.size as usize,
        ))
    }

    fn float_sqrt(&self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_f64(
            self.f64_from_self().sqrt(),
            self.size as usize,
        ))
    }

    fn float_ceil(&self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_f64(
            self.f64_from_self().ceil(),
            self.size as usize,
        ))
    }

    fn float_floor(&self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_f64(
            self.f64_from_self().floor(),
            self.size as usize,
        ))
    }

    fn float_round(&self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_f64(
            self.f64_from_self().round(),
            self.size as usize,
        ))
    }

    fn int_equal(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_bits(
            IntBinop::Equal.eval(self.as_bits(), other.as_bits(), self.size as usize),
            1,
        ))
    }

    fn int_not_equal(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_bits(
            IntBinop::NotEqual.eval(self.as_bits(), other.as_bits(), self.size as usize),
            1,
        ))
    }

    fn int_less(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_bits(
            IntBinop::Less.eval(self.as_bits(), other.as_bits(), self.size as usize),
            1,
        ))
    }

    fn int_sless(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_bits(
            IntBinop::SLess.eval(self.as_bits(), other.as_bits(), self.size as usize),
            1,
        ))
    }

    fn int_less_equal(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_bits(
            IntBinop::LessEqual.eval(self.as_bits(), other.as_bits(), self.size as usize),
            1,
        ))
    }

    fn int_sless_equal(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_bits(
            IntBinop::SLessEqual.eval(self.as_bits(), other.as_bits(), self.size as usize),
            1,
        ))
    }

    fn int_add(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_bits(
            IntBinop::Add.eval(self.as_bits(), other.as_bits(), size),
            size,
        ))
    }

    fn int_sub(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_bits(
            IntBinop::Sub.eval(self.as_bits(), other.as_bits(), size),
            size,
        ))
    }

    fn int_xor(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_bits(
            IntBinop::Xor.eval(self.as_bits(), other.as_bits(), size),
            size,
        ))
    }

    fn int_and(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_bits(
            IntBinop::And.eval(self.as_bits(), other.as_bits(), size),
            size,
        ))
    }

    fn int_or(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_bits(
            IntBinop::Or.eval(self.as_bits(), other.as_bits(), size),
            size,
        ))
    }

    fn int_shift_left(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.size as usize;
        Ok(Self::from_bits(
            IntBinop::ShiftLeft.eval(self.as_bits(), other.as_bits(), size),
            size,
        ))
    }

    fn int_shift_right(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.size as usize;
        Ok(Self::from_bits(
            IntBinop::ShiftRight.eval(self.as_bits(), other.as_bits(), size),
            size,
        ))
    }

    fn int_sshift_right(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.size as usize;
        Ok(Self::from_bits(
            IntBinop::SShiftRight.eval(self.as_bits(), other.as_bits(), size),
            size,
        ))
    }

    fn int_mul(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_bits(
            IntBinop::Mul.eval(self.as_bits(), other.as_bits(), size),
            size,
        ))
    }

    fn int_div(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_bits(
            IntBinop::Div.eval(self.as_bits(), other.as_bits(), size),
            size,
        ))
    }

    fn int_rem(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_bits(
            IntBinop::Rem.eval(self.as_bits(), other.as_bits(), size),
            size,
        ))
    }

    fn int_sdiv(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_bits(
            IntBinop::Sdiv.eval(self.as_bits(), other.as_bits(), size),
            size,
        ))
    }

    fn int_srem(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_bits(
            IntBinop::Srem.eval(self.as_bits(), other.as_bits(), size),
            size,
        ))
    }

    fn bool_and(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_bits(
            BoolBinop::And.eval(self.as_bits(), other.as_bits()),
            1,
        ))
    }

    fn bool_or(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_bits(
            BoolBinop::Or.eval(self.as_bits(), other.as_bits()),
            1,
        ))
    }

    fn bool_xor(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::from_bits(
            BoolBinop::Xor.eval(self.as_bits(), other.as_bits()),
            1,
        ))
    }

    fn float_add(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_f64(
            self.f64_from_self() + other.f64_from_self(),
            size,
        ))
    }

    fn float_sub(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_f64(
            self.f64_from_self() - other.f64_from_self(),
            size,
        ))
    }

    fn float_mul(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_f64(
            self.f64_from_self() * other.f64_from_self(),
            size,
        ))
    }

    fn float_div(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        Ok(Self::from_f64(
            self.f64_from_self() / other.f64_from_self(),
            size,
        ))
    }

    fn float_equal(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::new(
            bool_to_u64(self.f64_from_self() == other.f64_from_self()),
            1,
        ))
    }

    fn float_not_equal(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::new(
            bool_to_u64(self.f64_from_self() != other.f64_from_self()),
            1,
        ))
    }

    fn float_less(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::new(
            bool_to_u64(self.f64_from_self() < other.f64_from_self()),
            1,
        ))
    }

    fn float_less_equal(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        Ok(Self::new(
            bool_to_u64(self.f64_from_self() <= other.f64_from_self()),
            1,
        ))
    }
}

impl DomainMemory for EmulatedMemory {
    type V = SizedValue;

    fn read(
        &self,
        space: SpaceId,
        addr: Self::V,
        size: usize,
    ) -> Result<Self::V, EmulatorErrorKind> {
        let addr = addr.value()?;
        let bits = match self.spaces.get(&space) {
            Some(s) => s.read_u128(addr, size as u64)?,
            // Temp spaces (SpaceId >= 2) are per-varnode; uninitialized reads return zero.
            None if usize::from(space) >= 2 => 0,
            None => return Err(EmulatorErrorKind::UnknownSpace(space)),
        };
        Ok(SizedValue::from_bits(bits, size))
    }

    fn write(
        &mut self,
        space: SpaceId,
        addr: Self::V,
        size: usize,
        value: Self::V,
    ) -> Result<(), EmulatorErrorKind> {
        let addr = addr.value()?;

        self.spaces
            .entry(space)
            .or_default()
            .get_mut_region(addr, size)?
            .write_u128(value.as_bits());
        Ok(())
    }
}

/// Type alias for an instruction hook function, which is called with the current instruction and emulator state after each instruction is executed.
type InstructionHook = Box<dyn Fn(&InstructionRef<'_, '_>, &StandaloneEmulator) + Send + Sync>;

/// A lifetime-free emulator that takes `&Context<'_>` explicitly on each call.
/// Use this when you need to store an emulator without a lifetime (e.g., across an FFI boundary).
pub struct StandaloneEmulator {
    pub memory: EmulatedMemory,
    pub insn_values: HashMap<InstructionId, SizedValue>,
    pub block_param_values: HashMap<BlockParamId, SizedValue>,
    pub block: BlockId,
    pub idx: usize,
    /// Call stack maintained by `run_function` (outermost function first).
    pub call_stack: Vec<FunctionId>,

    pub instruction_hook: Option<InstructionHook>,
}

impl StandaloneEmulator {
    pub fn new(entry: BlockId) -> Self {
        Self {
            memory: EmulatedMemory::default(),
            insn_values: HashMap::new(),
            block_param_values: HashMap::new(),
            block: entry,
            idx: 0,
            call_stack: Vec::new(),
            instruction_hook: None,
        }
    }

    fn make_error(&self, ctx: &Context<'_>, kind: EmulatorErrorKind) -> EmulatorError {
        let block = BasicBlock::from_id(ctx, self.block);
        let instruction = block
            .instruction_ids()
            .get(self.idx)
            .copied()
            .or_else(|| block.instruction_ids().last().copied())
            .expect("cannot construct EmulatorError for empty block");

        let instruction = Instruction::from_id(ctx, instruction);
        EmulatorError {
            kind,
            ctx: format!(
                "instruction: {}\nblock: {:?}\nfunction: {:?}",
                instruction.as_statement(),
                instruction.parent().map(|b| b.name()),
                instruction.function().map(|f| f.name())
            ),
        }
    }

    pub fn from_address(ctx: &Context<'_>, addr: u64) -> Self {
        let entry = BasicBlock::from_addr(ctx, addr).expect("Invalid block address");
        Self::new(entry.id)
    }

    pub fn set_varnode(
        &mut self,
        ctx: &Context<'_>,
        id: VarnodeId,
        value: u64,
    ) -> Result<(), EmulatorErrorKind> {
        self.set_varnode_u128(ctx, id, u128::from(value))
    }

    pub fn set_varnode_u128(
        &mut self,
        ctx: &Context<'_>,
        id: VarnodeId,
        value: u128,
    ) -> Result<(), EmulatorErrorKind> {
        let varnode = Varnode::from_id(ctx, id);
        let space = varnode.space().id;
        let addr = varnode.address() as u64;
        let size = varnode.size();
        self.memory.write(
            space,
            SizedValue::from_u64(addr),
            size,
            SizedValue::from_bits(value, size),
        )
    }

    pub fn read_varnode(&self, ctx: &Context<'_>, id: VarnodeId) -> Option<u64> {
        let value = self.read_varnode_u128(ctx, id)?;
        u64::try_from(value).ok()
    }

    pub fn read_varnode_u128(&self, ctx: &Context<'_>, id: VarnodeId) -> Option<u128> {
        let varnode = Varnode::from_id(ctx, id);
        let space = varnode.space().id;
        let addr = varnode.address() as u64;
        let size = varnode.size();
        self.memory
            .read(space, SizedValue::from_u64(addr), size)
            .ok()
            .map(|v| v.as_bits())
    }

    pub fn get_value(&mut self, ctx: &Context<'_>, id: ValueId) -> Option<u64> {
        let mut tmp = TempInterpreter {
            memory: &mut self.memory,
            insn_values: &mut self.insn_values,
            block_param_values: &mut self.block_param_values,
            ctx,
        };
        tmp.get_value(id).ok().and_then(|v| v.value().ok())
    }

    pub fn set_varnode_bytes(
        &mut self,
        ctx: &Context<'_>,
        id: VarnodeId,
        bytes: &[u8],
    ) -> Result<(), EmulatorErrorKind> {
        let varnode = Varnode::from_id(ctx, id);
        let space = varnode.space().id;
        let base_addr = varnode.address() as u64;
        for (i, chunk) in bytes.chunks(8).enumerate() {
            let addr = base_addr + (i * 8) as u64;
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            let value = u64::from_le_bytes(buf);
            self.memory.write(
                space,
                SizedValue::from_u64(addr),
                chunk.len(),
                SizedValue::new(value, chunk.len()),
            )?;
        }
        Ok(())
    }

    pub fn set_varnode_by_name(
        &mut self,
        ctx: &Context<'_>,
        name: &str,
        value: u64,
    ) -> Result<bool, EmulatorErrorKind> {
        self.set_varnode_by_name_u128(ctx, name, u128::from(value))
    }

    pub fn set_varnode_by_name_u128(
        &mut self,
        ctx: &Context<'_>,
        name: &str,
        value: u128,
    ) -> Result<bool, EmulatorErrorKind> {
        match ctx.get_named(name) {
            Some(ValueId::Varnode(id)) => {
                self.set_varnode_u128(ctx, id, value)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub fn set_varnode_by_name_bytes(
        &mut self,
        ctx: &Context<'_>,
        name: &str,
        bytes: &[u8],
    ) -> Result<bool, EmulatorErrorKind> {
        match ctx.get_named(name) {
            Some(ValueId::Varnode(id)) => {
                self.set_varnode_bytes(ctx, id, bytes)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub fn read_varnode_by_name(&mut self, ctx: &Context<'_>, name: &str) -> Option<u64> {
        let value = self.read_varnode_by_name_u128(ctx, name)?;
        u64::try_from(value).ok()
    }

    pub fn read_varnode_by_name_u128(&mut self, ctx: &Context<'_>, name: &str) -> Option<u128> {
        match ctx.get_named(name)? {
            ValueId::Varnode(id) => self.read_varnode_u128(ctx, id),
            _ => None,
        }
    }

    pub fn read_varnode_bytes(&mut self, ctx: &Context<'_>, id: VarnodeId) -> Vec<u8> {
        let varnode = Varnode::from_id(ctx, id);
        let space = varnode.space().id;
        let addr = varnode.address() as u64;
        let size = varnode.size();
        self.memory.read_raw(space, addr, size).unwrap_or_default()
    }

    pub fn read_varnode_by_name_bytes(&mut self, ctx: &Context<'_>, name: &str) -> Option<Vec<u8>> {
        match ctx.get_named(name)? {
            ValueId::Varnode(id) => Some(self.read_varnode_bytes(ctx, id)),
            _ => None,
        }
    }

    pub fn get_value_bytes(&mut self, ctx: &Context<'_>, id: ValueId) -> Option<Vec<u8>> {
        let mut tmp = TempInterpreter {
            memory: &mut self.memory,
            insn_values: &mut self.insn_values,
            block_param_values: &mut self.block_param_values,
            ctx,
        };
        let sv = tmp.get_value(id).ok()?;
        let size = sv.size().ok()?;
        let bits = sv.as_bits();
        let mut bytes = vec![0u8; size];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::try_from((bits >> (i * 8)) & u128::from(0xffu8)).unwrap();
        }
        Some(bytes)
    }

    pub fn current_block(&self) -> BlockId {
        self.block
    }

    fn collect_block_args(
        &mut self,
        ctx: &Context<'_>,
        args: &[ValueId],
    ) -> Result<Vec<SizedValue>, EmulatorErrorKind> {
        let mut tmp = TempInterpreter {
            memory: &mut self.memory,
            insn_values: &mut self.insn_values,
            block_param_values: &mut self.block_param_values,
            ctx,
        };
        args.iter().map(|&arg| tmp.get_value(arg)).collect()
    }

    fn bind_block_args(
        &mut self,
        ctx: &Context<'_>,
        target: BlockId,
        args: &[ValueId],
    ) -> Result<(), EmulatorErrorKind> {
        let values = self.collect_block_args(ctx, args)?;
        let params = BasicBlock::from_id(ctx, target)
            .params()
            .map(|param| param.id)
            .collect::<Vec<_>>();
        if values.len() != params.len() {
            return Err(EmulatorErrorKind::ValueError(values.len() as u128));
        }
        for (param, value) in params.into_iter().zip(values) {
            self.block_param_values.insert(param, value);
        }
        Ok(())
    }

    pub fn step(&mut self, ctx: &Context<'_>) -> crate::Result<()> {
        let block = BasicBlock::from_id(ctx, self.block);
        let insn_ids = block.instruction_ids();
        assert!(self.idx < insn_ids.len(), "Reached end of block");
        let insn_id = insn_ids[self.idx];
        let insn = InstructionRef::new(ctx, insn_id);
        let id = insn.id;

        if let Some(hook) = self.instruction_hook.as_ref() {
            hook(&insn, self)
        }

        match insn.mnemonic() {
            Mnemonic::Branch(Branch { target, args }) => {
                self.bind_block_args(ctx, *target, args)
                    .map_err(|kind| self.make_error(ctx, kind))?;
                self.block = *target;
                self.idx = 0;
            }

            &Mnemonic::Call(Call { target, .. }) => {
                self.block = Function::from_id(ctx, target)
                    .root()
                    .ok_or_else(|| {
                        self.make_error(ctx, EmulatorErrorKind::EmptyFunctionRoot(target))
                    })?
                    .id;
                self.idx = 0;
            }

            Mnemonic::CBranch(CBranch {
                condition,
                success_block: target,
                success_args,
                failure_block: fallthrough,
                failure_args,
            }) => {
                let cond_val = self.get_value(ctx, *condition).unwrap();
                if cond_val != 0 {
                    self.bind_block_args(ctx, *target, success_args)
                        .map_err(|kind| self.make_error(ctx, kind))?;
                    self.block = *target;
                } else {
                    self.bind_block_args(ctx, *fallthrough, failure_args)
                        .map_err(|kind| self.make_error(ctx, kind))?;
                    self.block = *fallthrough;
                }
                self.idx = 0;
            }

            Mnemonic::BranchInd(BranchInd { ptr })
            | Mnemonic::CallInd(CallInd { ptr, .. })
            | Mnemonic::Return(Return { ptr, .. }) => {
                let addr = self.get_value(ctx, *ptr).unwrap();
                let target = BasicBlock::from_addr(ctx, addr)
                    .ok_or_else(|| {
                        self.make_error(ctx, EmulatorErrorKind::InvalidBlockAddress(addr))
                    })?
                    .id;
                self.block = target;
                self.idx = 0;
            }

            _ => {
                let mut tmp = TempInterpreter {
                    memory: &mut self.memory,
                    insn_values: &mut self.insn_values,
                    block_param_values: &mut self.block_param_values,
                    ctx,
                };
                if let Some(value) = tmp.interpret(insn)? {
                    self.insn_values.insert(id, value);
                }
                self.idx += 1;
            }
        }

        Ok(())
    }

    pub fn run_block(&mut self, ctx: &Context<'_>) -> crate::Result<()> {
        loop {
            self.step(ctx)?;
            if self.idx == 0 {
                break;
            }
        }
        Ok(())
    }

    /// Runs blocks until the current block starts at `addr`.
    pub fn run_until(&mut self, ctx: &Context<'_>, addr: u64) -> crate::Result<()> {
        let target = BasicBlock::from_addr(ctx, addr)
            .ok_or_else(|| self.make_error(ctx, EmulatorErrorKind::UnknownAddress(addr)))?
            .id;
        while self.block != target {
            self.run_block(ctx)?;
        }
        Ok(())
    }

    /// Runs the given function from its root block, stopping when the
    /// outermost `Return` is reached (without executing it).
    /// Nested calls are tracked via `call_depth` so inner returns are handled normally.
    /// The `call_stack` field is updated throughout execution.
    pub fn run_function(&mut self, ctx: &Context<'_>, func: FunctionId) -> crate::Result<()> {
        let root = Function::from_id(ctx, func)
            .root()
            .ok_or_else(|| self.make_error(ctx, EmulatorErrorKind::EmptyFunctionRoot(func)))?
            .id;
        self.block = root;
        self.idx = 0;
        self.call_stack.push(func);

        let mut call_depth: i32 = 0;

        let result = loop {
            let insn_ids = BasicBlock::from_id(ctx, self.block)
                .instruction_ids()
                .to_vec();
            let insn = InstructionRef::new(ctx, insn_ids[self.idx]);

            match insn.mnemonic() {
                Mnemonic::Return(_) if call_depth == 0 => break Ok(()),
                Mnemonic::Call(Call { target, .. }) => {
                    call_depth += 1;
                    self.call_stack.push(*target);
                    self.step(ctx)?;
                }
                Mnemonic::CallInd(_) => {
                    call_depth += 1;
                    self.step(ctx)?;
                    // Infer the callee from the block we landed in
                    if let Some(parent) = BasicBlock::from_id(ctx, self.block).parent() {
                        self.call_stack.push(parent.id);
                    }
                }
                Mnemonic::Return(_) => {
                    self.call_stack.pop();
                    call_depth -= 1;
                    self.step(ctx)?;
                }
                _ => {
                    self.step(ctx)?;
                }
            }
        };

        self.call_stack.pop(); // pop the outermost function
        result
    }
}

/// Private helper that pairs `&mut StandaloneEmulator` fields with `&Context<'_>`
/// so the default `Interpreter::interpret()` impl can be reused.
struct TempInterpreter<'a, 'ctx> {
    memory: &'a mut EmulatedMemory,
    insn_values: &'a mut HashMap<InstructionId, SizedValue>,
    block_param_values: &'a mut HashMap<BlockParamId, SizedValue>,
    ctx: &'ctx Context<'ctx>,
}

impl<'ctx> Interpreter for TempInterpreter<'_, 'ctx> {
    type V = SizedValue;
    type M = EmulatedMemory;

    fn memory(&mut self) -> &mut Self::M {
        self.memory
    }

    fn ctx(&self) -> &Context<'_> {
        self.ctx
    }

    fn get_value(&mut self, id: ValueId) -> Result<Self::V, EmulatorErrorKind> {
        match ValueRef::new(id, self.ctx) {
            ValueRef::Literal(literal) => Ok(SizedValue::new(literal.value(), literal.size())),
            ValueRef::Instruction(insn) => self
                .insn_values
                .get(&insn.id)
                .copied()
                .ok_or(EmulatorErrorKind::ValueError(0)),
            ValueRef::Varnode(varnode) => Ok(SizedValue::new(varnode.address() as u64, 8)),
            ValueRef::BasicBlock(_) => panic!("Cannot get value of a block"),
            ValueRef::BlockParam(param) => self
                .block_param_values
                .get(&param.id)
                .copied()
                .ok_or(EmulatorErrorKind::ValueError(0)),
            ValueRef::Function(f) => f
                .address()
                .map(SizedValue::from_u64)
                .ok_or(EmulatorErrorKind::EmptyFunctionRoot(f.id)),
        }
    }
}

pub struct Emulator<'ctx> {
    inner: StandaloneEmulator,
    ctx: &'ctx Context<'ctx>,
}

impl<'ctx> Emulator<'ctx> {
    pub fn new(ctx: &'ctx Context<'ctx>, entry: BlockId) -> Self {
        Self {
            inner: StandaloneEmulator::new(entry),
            ctx,
        }
    }

    pub fn set_instruction_hook(
        &mut self,
        hook: impl Fn(&InstructionRef<'_, '_>, &StandaloneEmulator) + Send + Sync + 'static,
    ) {
        self.inner.instruction_hook = Some(Box::new(hook));
    }

    pub fn from_function(ctx: &'ctx Context<'ctx>, func: FunctionId) -> Self {
        let entry = Function::from_id(ctx, func)
            .root()
            .expect("Cannot create emulator for function with empty root block")
            .id;
        Self::new(ctx, entry)
    }

    pub fn from_block(ctx: &'ctx Context<'ctx>, block: BlockId) -> Self {
        Self::new(ctx, block)
    }

    pub fn from_address(ctx: &'ctx Context<'ctx>, addr: u64) -> Self {
        Self {
            inner: StandaloneEmulator::from_address(ctx, addr),
            ctx,
        }
    }

    /// Debugging method to view a value at a given address
    pub fn inspect_memory(&mut self, space: SpaceId, addr: u64, size: usize) -> Option<Vec<u8>> {
        self.inner.memory.spaces.get_mut(&space).and_then(|s| {
            let region = s.get_mut_region(addr, size).ok()?;
            region.read_bytes(addr, size).ok()
        })
    }

    /// Sets the value of a varnode
    pub fn set_varnode(&mut self, id: VarnodeId, value: u64) -> Result<(), EmulatorErrorKind> {
        self.inner.set_varnode(self.ctx, id, value)
    }

    /// Sets the value of a varnode using full 128-bit precision.
    pub fn set_varnode_u128(
        &mut self,
        id: VarnodeId,
        value: u128,
    ) -> Result<(), EmulatorErrorKind> {
        self.inner.set_varnode_u128(self.ctx, id, value)
    }

    /// Sets the value of a register
    pub fn set_register(&mut self, id: RegisterId, value: u64) -> Result<(), EmulatorErrorKind> {
        let id = self.ctx.get_register(id).id;
        self.set_varnode(id, value)
    }

    /// Writes a value to memory
    pub fn write_memory(
        &mut self,
        space: SpaceId,
        addr: u64,
        value: &[u8],
    ) -> Result<(), EmulatorErrorKind> {
        let space = self.inner.memory.spaces.entry(space).or_default();
        for (i, byte) in value.iter().enumerate() {
            space.write_byte(addr + i as u64, *byte);
        }
        Ok(())
    }

    /// Sets the value of a register using full 128-bit precision.
    pub fn set_register_u128(
        &mut self,
        id: RegisterId,
        value: u128,
    ) -> Result<(), EmulatorErrorKind> {
        let id = self.ctx.get_register(id).id;
        self.set_varnode_u128(id, value)
    }

    pub fn read_varnode(&mut self, id: VarnodeId) -> Option<u64> {
        self.inner.read_varnode(self.ctx, id)
    }

    pub fn read_varnode_u128(&mut self, id: VarnodeId) -> Option<u128> {
        self.inner.read_varnode_u128(self.ctx, id)
    }

    pub fn read_register(&mut self, id: RegisterId) -> Option<u64> {
        let id = self.ctx.get_register(id).id;
        self.read_varnode(id)
    }

    pub fn read_register_u128(&mut self, id: RegisterId) -> Option<u128> {
        let id = self.ctx.get_register(id).id;
        self.read_varnode_u128(id)
    }

    /// Writes a single 64-bit lane of a wide register.
    /// Lane `n` covers bytes `[n*8 .. n*8+8]` relative to the register's base address.
    pub fn set_register_lane(&mut self, id: RegisterId, lane: usize, value: u64) {
        let (space_id, base_addr) = {
            let vn = self.ctx.get_register(id);
            (vn.space().id, vn.address() as u64)
        };
        let base = base_addr + (lane as u64) * 8;
        let space = self.inner.memory.spaces.entry(space_id).or_default();
        for i in 0..8u64 {
            space.write_byte(base + i, (value >> (i * 8)) as u8);
        }
    }

    /// Reads a single 64-bit lane of a wide register (little-endian).
    /// Lane `n` covers bytes `[n*8 .. n*8+8]` relative to the register's base address.
    pub fn read_register_lane(&mut self, id: RegisterId, lane: usize) -> u64 {
        let (space_id, base_addr) = {
            let vn = self.ctx.get_register(id);
            (vn.space().id, vn.address() as u64)
        };
        let base = base_addr + (lane as u64) * 8;
        let space = self.inner.memory.spaces.entry(space_id).or_default();
        (0..8u64).fold(0u64, |acc, i| {
            acc | u64::from(space.read_byte(base + i).unwrap_or(0)) << (i * 8)
        })
    }

    /// Gets the current block
    pub fn block(&self) -> BlockRef<'ctx, 'ctx> {
        BasicBlock::from_id(self.ctx, self.inner.block)
    }

    /// Gets the current instruction
    pub fn insn(&self) -> Option<InstructionRef<'ctx, 'ctx>> {
        let block = self.block();
        if self.inner.idx >= block.instruction_ids().len() {
            None
        } else {
            let id = block.instruction_ids()[self.inner.idx];
            Some(InstructionRef::new(self.ctx, id))
        }
    }

    /// Executes a single pcode instruction
    pub fn step(&mut self) -> crate::Result<()> {
        self.inner.step(self.ctx)
    }

    /// Executes instructions until the end of the current block
    pub fn run_block(&mut self) -> crate::Result<()> {
        self.inner.run_block(self.ctx)
    }

    /// Runs blocks until the current block starts at `addr`.
    pub fn run_until(&mut self, addr: u64) -> crate::Result<()> {
        self.inner.run_until(self.ctx, addr)
    }

    /// Runs the given function from its root block, stopping before the outermost `Return`.
    pub fn run_function(&mut self, func: FunctionId) -> crate::Result<()> {
        self.inner.run_function(self.ctx, func)
    }

    /// Returns the current emulator call stack (outermost function first).
    /// Only populated during `run_function` execution.
    pub fn call_stack(&self) -> &[FunctionId] {
        &self.inner.call_stack
    }
}

impl<'ctx> Interpreter for Emulator<'ctx> {
    type V = SizedValue;
    type M = EmulatedMemory;

    fn memory(&mut self) -> &mut Self::M {
        &mut self.inner.memory
    }

    fn ctx(&self) -> &Context<'ctx> {
        self.ctx
    }

    fn get_value(&mut self, id: ValueId) -> Result<Self::V, EmulatorErrorKind> {
        match ValueRef::new(id, self.ctx) {
            ValueRef::Literal(literal) => Ok(SizedValue::new(literal.value(), literal.size())),
            ValueRef::Instruction(insn) => self
                .inner
                .insn_values
                .get(&insn.id)
                .copied()
                .ok_or(EmulatorErrorKind::ValueError(0)),
            ValueRef::Varnode(varnode) => Ok(SizedValue::new(varnode.address() as u64, 8)),
            ValueRef::BasicBlock(_) => panic!("Cannot get value of a block"),
            ValueRef::BlockParam(param) => self
                .inner
                .block_param_values
                .get(&param.id)
                .copied()
                .ok_or(EmulatorErrorKind::ValueError(0)),
            ValueRef::Function(f) => f
                .address()
                .map(SizedValue::from_u64)
                .ok_or(EmulatorErrorKind::EmptyFunctionRoot(f.id)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::context::Context;
    use qcode_macro::qcode;

    #[test]
    fn sized_value_masks_to_declared_width() {
        let value = SizedValue::new(0x1234, 1);
        assert_eq!(value.size().unwrap(), 1);
        assert_eq!(value.value().unwrap(), 0x34);
    }

    #[test]
    fn int_add_wraps_by_width_and_sets_carry() {
        let lhs = SizedValue::new(0xff, 1);
        let rhs = SizedValue::new(0x01, 1);

        let sum = lhs.int_add(&rhs).unwrap();
        let carry = lhs.carry(&rhs).unwrap();

        assert_eq!(sum.value().unwrap(), 0x00);
        assert_eq!(sum.size().unwrap(), 1);
        assert_eq!(carry.value().unwrap(), 1);
        assert_eq!(carry.size().unwrap(), 1);
    }

    #[test]
    fn branch_args_bind_block_params() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <src>
                goto <dst @x=0x2>;
            <dst @x>
                %sum = i64 @x + 0x3;
                goto <0x1001>;
            "
        );

        let mut emu = StandaloneEmulator::new(src);
        emu.step(&ctx).expect("branch binds block params");
        emu.step(&ctx).expect("destination uses block param");

        assert_eq!(emu.get_value(&ctx, sum.into()), Some(5));
    }

    #[test]
    fn int_mul_wraps_for_64_bit_values() {
        let lhs = SizedValue::new(u64::MAX, 8);
        let rhs = SizedValue::new(2, 8);

        let product = lhs.int_mul(&rhs).unwrap();

        assert_eq!(product.value().unwrap(), u64::MAX.wrapping_mul(2));
        assert_eq!(product.size().unwrap(), 8);
    }

    #[test]
    fn int_mul_wraps_for_128_bit_values() {
        let lhs = SizedValue::from_bits(u128::MAX, 16);
        let rhs = SizedValue::from_bits(u128::from(2u8), 16);

        let product = lhs.int_mul(&rhs).unwrap();

        assert_eq!(product.as_bits(), u128::MAX.wrapping_mul(u128::from(2u8)));
        assert_eq!(product.size().unwrap(), 16);
        assert!(matches!(
            product.value(),
            Err(EmulatorErrorKind::ValueError(_))
        ));
    }

    #[test]
    fn int_div_and_rem_work_for_128_bit_values() {
        let lhs = SizedValue::from_bits(u128::MAX, 16);
        let rhs = SizedValue::from_bits(u128::from(3u8), 16);

        let q = lhs.int_div(&rhs).unwrap();
        let r = lhs.int_rem(&rhs).unwrap();

        assert_eq!(q.as_bits(), u128::MAX / u128::from(3u8));
        assert_eq!(r.as_bits(), u128::MAX % u128::from(3u8));
        assert_eq!(q.size().unwrap(), 16);
        assert_eq!(r.size().unwrap(), 16);
    }

    #[test]
    fn int_sdiv_and_srem_work_for_128_bit_values() {
        let lhs = SizedValue::from_bits(u128::from(0xffff_ffff_ffff_ffffu64), 16);
        let rhs = SizedValue::from_bits(u128::from(2u8), 16);

        let q = lhs.int_sdiv(&rhs).unwrap();
        let r = lhs.int_srem(&rhs).unwrap();

        assert_eq!(q.as_bits(), u128::from(0x7fff_ffff_ffff_ffffu64));
        assert_eq!(r.as_bits(), u128::from(1u8));
        assert_eq!(q.size().unwrap(), 16);
        assert_eq!(r.size().unwrap(), 16);
    }

    #[test]
    fn signed_extension_and_shift_behave_as_expected() {
        let negative_byte = SizedValue::new(0x80, 1);
        let extended = negative_byte.sext(8).unwrap();
        let shifted = negative_byte
            .int_sshift_right(&SizedValue::new(1, 1))
            .unwrap();

        assert_eq!(extended.value().unwrap(), 0xffff_ffff_ffff_ff80);
        assert_eq!(extended.size().unwrap(), 8);
        assert_eq!(shifted.value().unwrap(), 0xc0);
        assert_eq!(shifted.size().unwrap(), 1);
    }

    #[test]
    fn signed_comparisons_use_value_width() {
        let lhs = SizedValue::new(0xff, 1);
        let rhs = SizedValue::new(0x01, 1);

        assert_eq!(lhs.int_sless(&rhs).and_then(|v| v.value()).unwrap(), 1);
        assert_eq!(rhs.int_sless(&lhs).and_then(|v| v.value()).unwrap(), 0);
    }

    #[test]
    fn int_sub_uses_lhs_width_with_default_u64_immediate() {
        let lhs = SizedValue::new(0, 4);
        let rhs = SizedValue::from_u64(1);

        let diff = lhs.int_sub(&rhs).unwrap();

        assert_eq!(diff.size().unwrap(), 4);
        assert_eq!(diff.value().unwrap(), 0xffff_ffff);
    }

    #[test]
    fn sborrow_uses_lhs_width_with_default_u64_immediate() {
        let lhs = SizedValue::new(0x80, 1);
        let rhs = SizedValue::new(1, 1);

        // 0x80 - 1 overflows in signed 8-bit arithmetic.
        assert_eq!(lhs.sborrow(&rhs).and_then(|v| v.value()).unwrap(), 1);
    }

    #[test]
    fn scarry_uses_lhs_width_with_default_u64_immediate() {
        let lhs = SizedValue::new(0x7f, 1);
        let rhs = SizedValue::new(1, 1);

        // 0x7f + 1 overflows in signed 8-bit arithmetic.
        assert_eq!(lhs.scarry(&rhs).and_then(|v| v.value()).unwrap(), 1);
    }

    #[test]
    fn sborrow_neg() {
        let lhs = SizedValue::new(0x0, 1);
        let rhs = SizedValue::new(0x80, 1);

        // 0 - 0x80  overflows in signed 8-bit arithmetic.
        assert_eq!(lhs.sborrow(&rhs).and_then(|v| v.value()).unwrap(), 1);
    }

    #[test]
    fn lz_count_respects_width() {
        let value = SizedValue::new(0x01, 1);
        let lz = value.lz_count().unwrap();

        assert_eq!(lz.value().unwrap(), 7);
        assert_eq!(lz.size().unwrap(), 1);
    }

    #[test]
    fn float_conversion_handles_f32_and_f64() {
        let minus_one = SizedValue::new(0xff, 1);
        let as_f32 = minus_one.int_to_float(4).unwrap();
        assert_eq!(as_f32.value().unwrap(), (-1.0f32).to_bits() as u64);
        assert_eq!(as_f32.size().unwrap(), 4);

        let f32_value = SizedValue::new((1.5f32).to_bits() as u64, 4);
        let promoted = f32_value.float_to_float(8).unwrap();
        let promoted_bits = promoted.value().unwrap();
        assert_eq!(f64::from_bits(promoted_bits), 1.5f64);
        assert_eq!(promoted.size().unwrap(), 8);

        let demoted = promoted.float_to_float(4).unwrap();
        assert_eq!(demoted.value().unwrap(), (1.5f32).to_bits() as u64);
        assert_eq!(demoted.size().unwrap(), 4);
    }

    #[test]
    fn bool_operations_return_single_byte_results() {
        let truthy = SizedValue::new(2, 8);
        let falsy = SizedValue::new(0, 8);

        let and_result = truthy.bool_and(&falsy).unwrap();
        let or_result = truthy.bool_or(&falsy).unwrap();
        let xor_result = truthy.bool_xor(&truthy).unwrap();
        let not_result = falsy.bool_not().unwrap();

        assert_eq!(and_result.value().unwrap(), 0);
        assert_eq!(or_result.value().unwrap(), 1);
        assert_eq!(xor_result.value().unwrap(), 0);
        assert_eq!(not_result.value().unwrap(), 1);

        assert_eq!(and_result.size().unwrap(), 1);
        assert_eq!(or_result.size().unwrap(), 1);
        assert_eq!(xor_result.size().unwrap(), 1);
        assert_eq!(not_result.size().unwrap(), 1);
    }

    #[test]
    fn simple_addition() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 V0;
            varnode i64 V1;

        <block>
            %v0 = load(i64, &V0);
            %v1 = load(i64, &V1);
            %res = %v0 + %v1;
            goto <0x1001>;
        "
        );

        let mut emu = Emulator::from_block(&ctx, block);
        emu.set_varnode(V0, 2).unwrap();
        emu.set_varnode(V1, 3).unwrap();
        emu.run_block().unwrap();

        assert_eq!(
            emu.get_value(res.into()).and_then(|v| v.value()).unwrap(),
            5
        );
    }

    #[test]
    fn emulator_int_div_works_with_128_bit_operands() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i128 V0;
            varnode i128 V1;

        <block>
            %v0 = load(i128, &V0);
            %v1 = load(i128, &V1);

            %res = v0 / v1;
            goto <0x1001>;
        "
        );

        let mut emu = Emulator::from_block(&ctx, block);
        let v0_bits = u128::from(1u8) << 100;
        let v1_bits = u128::from(1u8) << 99;

        emu.set_varnode_u128(V0, v0_bits).unwrap();
        emu.set_varnode_u128(V1, v1_bits).unwrap();
        emu.run_block().unwrap();

        // Quotient is small enough to also be visible through legacy u64 extraction.
        assert_eq!(
            emu.get_value(res.into()).and_then(|v| v.value()).unwrap(),
            2
        );
    }

    // -----------------------------------------------------------------------
    // Memory error-handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn uninitialized_memory_reads_error() {
        let space = EmulatedSpace::default();
        assert!(matches!(
            space.read_byte(0xdead_beef),
            Err(EmulatorErrorKind::MemoryReadError(0xdead_beef))
        ));
        assert!(matches!(
            space.read(0x1000, 4),
            Err(EmulatorErrorKind::MemoryReadError(0x1000))
        ));
    }

    #[test]
    fn get_region_overflow_does_not_panic() {
        let mut space = EmulatedSpace::default();
        // addr + size would overflow u64 without checked_add
        assert!(matches!(
            space.get_mut_region(u64::MAX - 2, 8),
            Err(EmulatorErrorKind::AddressOverflow(_, _))
        ));
    }

    // -----------------------------------------------------------------------
    // run_function happy-path tests
    // -----------------------------------------------------------------------

    #[test]
    fn run_function_returns_ok_for_trivial_function() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn function:
            <entry>
                return [i64 0];
            "
        );

        let mut emu = Emulator::from_function(&ctx, function);
        assert!(emu.run_function(function).is_ok());
    }

    #[test]
    fn run_function_executes_instructions_before_return() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;

            fn function:
            <entry>
                %a = load(i64, &A);
                %b = load(i64, &B);
                %sum = %a + %b;
                return [i64 0];
            "
        );

        let mut emu = Emulator::from_function(&ctx, function);
        emu.set_varnode(A, 7).unwrap();
        emu.set_varnode(B, 5).unwrap();
        emu.run_function(function).unwrap();

        assert_eq!(
            emu.get_value(sum.into()).and_then(|v| v.value()).unwrap(),
            12
        );
    }

    #[test]
    fn run_function_call_stack_empty_after_successful_return() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;

            fn function:
            <entry>
                %a = load(i64, &A);
                %b = load(i64, &B);
                %sum = %a + %b;
                return [i64 0];
            "
        );

        let mut emu = Emulator::from_function(&ctx, function);
        emu.run_function(function).unwrap();

        assert!(emu.call_stack().is_empty());
    }

    // -----------------------------------------------------------------------
    // run_function error tests
    // -----------------------------------------------------------------------

    #[test]
    fn branchind_to_unknown_address_returns_error() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn function:
            <entry>
                # Branching to literal 0 — no block lives at address 0
                goto [i64 0];
            "
        );

        let mut emu = Emulator::from_function(&ctx, function);
        let err = emu.run_function(function).unwrap_err();
        assert!(matches!(
            err.kind,
            EmulatorErrorKind::InvalidBlockAddress(0)
        ));
    }

    #[test]
    fn error_includes_faulting_instruction_id() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn function:
            <entry>
                # Null pointer dereference
                %bad_load = load(i64, i64 0);
                return [i64 0];
            "
        );

        let mut emu = Emulator::from_function(&ctx, function);
        let err = emu.run_function(function).unwrap_err();

        assert!(
            err.ctx.contains(
                &Instruction::from_id(&ctx, bad_load)
                    .as_statement()
                    .to_string()
            )
        );
    }

    #[test]
    fn error_call_stack_reflects_active_frames_at_fault() {
        // Build callee: immediately does a BranchInd to address 0 (always fails)
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn callee:
            <entry1>
                # Branching to literal 0 — no block lives at address 0
                goto [i64 0];

            fn caller:
            <entry2>
                call <callee>;
            "
        );

        // Build caller: calls callee
        let mut emu = Emulator::from_function(&ctx, caller);
        let err = emu.run_function(caller).unwrap_err();

        assert!(matches!(
            err.kind,
            EmulatorErrorKind::InvalidBlockAddress(0)
        ));
        assert_eq!(emu.call_stack(), &[caller, callee]);
    }
}
