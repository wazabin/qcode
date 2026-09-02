use crate::{
    CallContinuation, CallInterception, CallSite, DomainMemory, EmulatorError, EmulatorErrorKind,
    Interpreter,
};
use qcode::{
    address_index::{AddressIndex, AddressTarget},
    context::Context,
    space::{MemorySpaceId, Space, SpaceId, SpaceType},
    value::{
        BasicBlock, BlockId, BlockParamId, BlockRef, FunctionBody, FunctionId, Instruction,
        LocalValueId, Value, ValueId, ValueRef, Varnode,
        insn::{
            Binary, Binop, Branch, BranchInd, CBranch, Call, CallInd, Callee, Carry, Extract,
            FloatBinop, FloatToFloat, FloatToInt, InstructionId, InstructionRef, IntBinop,
            IntToFloat, LzCount, Mnemonic, PopCount, Range, Return, SBorrow, SCarry, Scan, Sext,
            Store, Tuple, Unary, Unop, Zext,
        },
        varnode::{VarnodeId, register::RegisterId},
    },
};
use std::cmp;

mod float80;

use rustc_apfloat::Status;
use rustc_hash::{FxHashMap, FxHashSet};

use super::DomainValue;

fn require_real_callee(callee: Callee) -> Result<FunctionId, EmulatorErrorKind> {
    match callee {
        Callee::Real(target) => Ok(target),
        Callee::Minted(slot) => Err(EmulatorErrorKind::UnresolvedMintedCallee(slot)),
    }
}

/// Whether the call instruction `call_id` carries the `regpure` binding
/// convention (argpromote v2): its register interface is explicit at the site,
/// so inputs are bound positionally from `Call.args` and outputs are replayed by
/// the caller rather than by the emulator's implicit writeback.
fn call_is_regpure(ctx: &Context<'_>, call_id: InstructionId) -> bool {
    matches!(
        ctx.get_insn(call_id).mnemonic(),
        Mnemonic::Call(call) if call.tag.is_regpure()
    )
}

#[derive(Debug, Default, Clone)]
pub struct EmulatedSpace(FxHashMap<u64, u8>);

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

    pub fn read_zero_filled(&self, addr: u64, size: usize) -> Vec<u8> {
        (0..size)
            .map(|i| self.0.get(&(addr + i as u64)).copied().unwrap_or(0))
            .collect()
    }

    pub fn write_byte(&mut self, addr: u64, value: u8) {
        self.0.insert(addr, value);
    }

    /// Reserves capacity for at least `additional` more bytes, so a bulk write
    /// allocates once instead of rehashing the table on the way up.
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
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

    /// Reads a little-endian unsigned integer, treating missing bytes as zero.
    pub fn read_u128_zero_filled(&self, addr: u64, size: u64) -> u128 {
        let mut res = 0u128;

        for cur in addr..addr + cmp::min(size, 16) {
            let byte = self.0.get(&cur).copied().unwrap_or(0);
            res |= u128::from(byte) << ((cur - addr) * 8);
        }

        res
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
    spaces: FxHashMap<MemorySpaceId, EmulatedSpace>,
    zero_filled_spaces: FxHashSet<MemorySpaceId>,
    /// Space count the zero-fill set was last built for. Spaces are append-only
    /// and their type is fixed at creation, so an unchanged count means the set
    /// is still valid — this keeps the per-step call O(1) instead of rescanning.
    configured_space_count: Option<usize>,
}

impl EmulatedMemory {
    fn is_zero_filled(&self, space: MemorySpaceId) -> bool {
        matches!(space, MemorySpaceId::Temp(_)) || self.zero_filled_spaces.contains(&space)
    }

    fn configure_spaces(&mut self, ctx: &Context<'_>) {
        let space_count = ctx.space_count();
        if self.configured_space_count == Some(space_count) {
            return;
        }
        self.zero_filled_spaces.clear();
        for index in 0..space_count {
            let id = SpaceId::from(index);
            let space = Space::from_id(ctx, id);
            // x86's private x87 RAM file is architectural state like the
            // register space, not process memory. A fresh CPU state has zero
            // payload bytes there, so instructions such as FXSAVE can read it
            // before a harness explicitly seeds an f80 slot.
            if matches!(space.ty, SpaceType::Register) || space.name.as_deref() == Some("x87") {
                self.zero_filled_spaces.insert(id.into());
            }
        }
        self.configured_space_count = Some(space_count);
    }

    fn read_raw(
        &self,
        space: MemorySpaceId,
        addr: u64,
        size: usize,
    ) -> Result<Vec<u8>, EmulatorErrorKind> {
        match self.spaces.get(&space) {
            Some(value) if self.is_zero_filled(space) => Ok(value.read_zero_filled(addr, size)),
            Some(value) => value.read(addr, size),
            None if self.is_zero_filled(space) => Ok(vec![0; size]),
            None => Err(EmulatorErrorKind::UnknownSpace(space)),
        }
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

    pub fn from_bits(value: u128, size: usize) -> Self {
        let size = cmp::min(size, 16) as u8;
        let value = value & mask_for_size(size as usize);
        Self { value, size }
    }

    fn as_u64(&self) -> u64 {
        u128_to_u64(self.value & mask_for_size(self.size as usize))
    }

    pub fn as_bits(&self) -> u128 {
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

    /// Size, in bytes, at which a two-operand integer op (add/sub/and/.../carry/
    /// scarry/sborrow) is evaluated.
    ///
    /// These p-code ops require both operands to share a size, so well-formed IR
    /// always has `self.size == other.size`. When the lifter leaves them
    /// mismatched it is the *left* operand that carries the operative width: it
    /// is the destination/base that the right operand is being combined into
    /// (e.g. address arithmetic `int_add(base:8, disp:4)` must stay 8 bytes, not
    /// truncate to the displacement). The narrower side here is an immediate
    /// whose value already fits, so taking the left width is correct.
    ///
    /// Cases where the *immediate* is on the left and wider than the real
    /// operand — NEG's `OF = sborrow(0, AL)` — are instead fixed upstream, by
    /// sizing the immediate to its sibling in the lifter, so this function never
    /// sees that mismatch. See `emit_function_call` in `harbinger::emit`.
    fn widen_size(&self, _other: &Self) -> usize {
        self.size as usize
    }

    fn from_f80_bits(value: u128) -> Self {
        Self::from_bits(value, 10)
    }

    fn f64_from_self(&self) -> f64 {
        match self.size as usize {
            0..=4 => f32::from_bits(self.as_u64() as u32) as f64,
            8 => f64::from_bits(self.as_u64()),
            10 => float80::to_f64(self.as_bits()),
            _ => 0.0,
        }
    }

    fn from_f64(value: f64, size: usize) -> Self {
        match size {
            0..=4 => Self::new((value as f32).to_bits() as u64, 4),
            8 => Self::new(value.to_bits(), 8),
            10 => Self::from_f80_bits(float80::from_f64(value)),
            _ => Self::new(0, size),
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

    fn zero(size: usize) -> Self {
        Self::new(0, size)
    }

    fn is_float_nan(&self) -> Result<Self, EmulatorErrorKind> {
        let is_nan = if self.size == 10 {
            float80::is_nan(self.as_bits())
        } else {
            self.f64_from_self().is_nan()
        };
        Ok(Self::new(bool_to_u64(is_nan), 1))
    }

    fn int_to_float(&self, size: usize) -> Result<Self, EmulatorErrorKind> {
        let signed = self.signed_value();
        match size {
            4 => Ok(Self::new((signed as f32).to_bits() as u64, 4)),
            8 => Ok(Self::new((signed as f64).to_bits(), 8)),
            10 => Ok(Self::from_f80_bits(float80::from_i128(signed))),
            _ => Ok(Self::new(0, size)),
        }
    }

    fn float_to_float(&self, size: usize) -> Result<Self, EmulatorErrorKind> {
        if size == 10 {
            let value = match self.size as usize {
                0..=4 => float80::from_f32_bits(self.as_u64() as u32),
                8 => float80::from_f64(f64::from_bits(self.as_u64())),
                10 => self.as_bits(),
                _ => 0,
            };
            return Ok(Self::from_f80_bits(value));
        }
        match size {
            4 => Ok(Self::new((self.f64_from_self() as f32).to_bits() as u64, 4)),
            8 => Ok(Self::new(self.f64_from_self().to_bits(), 8)),
            _ => Ok(Self::new(0, size)),
        }
    }

    fn float_to_int(&self, size: usize) -> Result<Self, EmulatorErrorKind> {
        let value = if self.size == 10 {
            float80::to_i128(self.as_bits(), size * 8) as u64
        } else {
            self.f64_from_self() as i64 as u64
        };
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

    fn byte_swap(&self) -> Result<Self, EmulatorErrorKind> {
        let size = self.size as usize;
        let mut value = 0u128;
        for index in 0..size {
            let byte = (self.as_bits() >> (index * 8)) & 0xff;
            value |= byte << ((size - index - 1) * 8);
        }
        Ok(Self::from_bits(value, size))
    }

    fn intrinsic(
        id: qcode::value::insn::IntrinsicId,
        args: &[Self],
        out_size: usize,
    ) -> Result<Self, EmulatorErrorKind> {
        let operands: Vec<(u128, usize)> = args
            .iter()
            .map(|a| (a.as_bits(), a.size as usize))
            .collect();
        let value = id
            .desc()
            .eval(&operands, out_size)
            .ok_or_else(|| EmulatorErrorKind::UnsupportedIntrinsic(Box::from(id.name())))?;
        Ok(Self::from_bits(value, out_size))
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

    fn int_negate(&self) -> Result<Self, EmulatorErrorKind> {
        let size = self.size as usize;
        Ok(Self::from_bits(
            Unop::IntNegate.eval_int(self.as_bits(), size).unwrap(),
            size,
        ))
    }

    fn float_negate(&self) -> Result<Self, EmulatorErrorKind> {
        if self.size == 10 {
            return Ok(Self::from_f80_bits(float80::negate(self.as_bits())));
        }
        Ok(Self::from_f64(-self.f64_from_self(), self.size as usize))
    }

    fn float_abs(&self) -> Result<Self, EmulatorErrorKind> {
        if self.size == 10 {
            return Ok(Self::from_f80_bits(float80::abs(self.as_bits())));
        }
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

    fn float_add(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        if size == 10 {
            return Ok(Self::from_f80_bits(float80::add(
                self.as_bits(),
                other.as_bits(),
            )));
        }
        Ok(Self::from_f64(
            self.f64_from_self() + other.f64_from_self(),
            size,
        ))
    }

    fn float_sub(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        if size == 10 {
            return Ok(Self::from_f80_bits(float80::sub(
                self.as_bits(),
                other.as_bits(),
            )));
        }
        Ok(Self::from_f64(
            self.f64_from_self() - other.f64_from_self(),
            size,
        ))
    }

    fn float_mul(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        if size == 10 {
            return Ok(Self::from_f80_bits(float80::mul(
                self.as_bits(),
                other.as_bits(),
            )));
        }
        Ok(Self::from_f64(
            self.f64_from_self() * other.f64_from_self(),
            size,
        ))
    }

    fn float_div(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let size = self.widen_size(other);
        if size == 10 {
            return Ok(Self::from_f80_bits(float80::div(
                self.as_bits(),
                other.as_bits(),
            )));
        }
        Ok(Self::from_f64(
            self.f64_from_self() / other.f64_from_self(),
            size,
        ))
    }

    fn float_equal(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let equal = if self.size == 10 {
            float80::equal(self.as_bits(), other.as_bits())
        } else {
            self.f64_from_self() == other.f64_from_self()
        };
        Ok(Self::new(bool_to_u64(equal), 1))
    }

    fn float_not_equal(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let unequal = if self.size == 10 {
            !float80::equal(self.as_bits(), other.as_bits())
        } else {
            self.f64_from_self() != other.f64_from_self()
        };
        Ok(Self::new(bool_to_u64(unequal), 1))
    }

    fn float_less(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let less = if self.size == 10 {
            float80::less(self.as_bits(), other.as_bits())
        } else {
            self.f64_from_self() < other.f64_from_self()
        };
        Ok(Self::new(bool_to_u64(less), 1))
    }

    fn float_less_equal(&self, other: &Self) -> Result<Self, EmulatorErrorKind> {
        let less_equal = if self.size == 10 {
            float80::less_equal(self.as_bits(), other.as_bits())
        } else {
            self.f64_from_self() <= other.f64_from_self()
        };
        Ok(Self::new(bool_to_u64(less_equal), 1))
    }
}

/// The byte-addressable memory an emulator run needs, over and above the
/// value-domain reads and writes of [`DomainMemory`].
///
/// [`DomainMemory`] speaks in whole values of a domain, which is all the
/// interpreter itself needs. A *harness* needs more: to seed a fixture, dump a
/// region, or poke a single register lane, it has to talk in bytes. Keeping
/// those on a separate trait is what lets [`StandaloneEmulator`] be generic over
/// its memory, so a richer backend — one with mapped pages and permissions —
/// can be substituted without the interpreter knowing.
pub trait EmulatorMemory: DomainMemory<V = SizedValue> {
    /// Prepares per-space bookkeeping for `ctx`. Called before any access, and
    /// cheap to call repeatedly: spaces are append-only, so an implementation
    /// can skip the work when nothing has been added.
    fn configure_spaces(&mut self, ctx: &Context<'_>);

    /// Reads `size` raw bytes, without the value-domain's width handling.
    fn read_bytes(
        &self,
        space: MemorySpaceId,
        addr: u64,
        size: usize,
    ) -> Result<Vec<u8>, EmulatorErrorKind>;

    /// Writes raw bytes, creating the space if it does not exist yet.
    fn write_bytes(
        &mut self,
        space: MemorySpaceId,
        addr: u64,
        bytes: &[u8],
    ) -> Result<(), EmulatorErrorKind>;
}

impl EmulatorMemory for EmulatedMemory {
    fn configure_spaces(&mut self, ctx: &Context<'_>) {
        EmulatedMemory::configure_spaces(self, ctx)
    }

    fn read_bytes(
        &self,
        space: MemorySpaceId,
        addr: u64,
        size: usize,
    ) -> Result<Vec<u8>, EmulatorErrorKind> {
        self.read_raw(space, addr, size)
    }

    fn write_bytes(
        &mut self,
        space: MemorySpaceId,
        addr: u64,
        bytes: &[u8],
    ) -> Result<(), EmulatorErrorKind> {
        let space = self.spaces.entry(space).or_default();
        space.reserve(bytes.len());
        for (index, byte) in bytes.iter().enumerate() {
            space.write_byte(addr + index as u64, *byte);
        }
        Ok(())
    }
}

impl DomainMemory for EmulatedMemory {
    type V = SizedValue;

    fn read(
        &self,
        space: MemorySpaceId,
        addr: Self::V,
        size: usize,
    ) -> Result<Self::V, EmulatorErrorKind> {
        let addr = addr.value()?;
        let zero_filled = self.is_zero_filled(space);
        let bits = match self.spaces.get(&space) {
            Some(s) if zero_filled => s.read_u128_zero_filled(addr, size as u64),
            Some(s) => s.read_u128(addr, size as u64)?,
            None if zero_filled => 0,
            None => return Err(EmulatorErrorKind::UnknownSpace(space)),
        };
        Ok(SizedValue::from_bits(bits, size))
    }

    fn write(
        &mut self,
        space: MemorySpaceId,
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

/// Values of literals, resolved once and kept by id.
///
/// Reading a literal goes through the module's interner, which takes an
/// `RwLock` read guard per access so that constants can be minted through a
/// shared reference. That is two atomic operations for what is morally a
/// constant, and it measured at 14% of run time in a profile of an interpreter
/// loop. A literal is immutable once interned — attaching a symbolic reference
/// later changes neither its value nor its width — so caching it by id is
/// sound, and turns the access into an array index.
#[derive(Debug, Default, Clone)]
pub struct LiteralCache(Vec<Option<SizedValue>>);

impl LiteralCache {
    fn get(&mut self, ctx: &Context<'_>, id: qcode::value::LiteralId) -> SizedValue {
        let index: usize = id.into();
        if index >= self.0.len() {
            self.0.resize(index + 1, None);
        }
        match self.0[index] {
            Some(value) => value,
            None => {
                // The miss pays the interner's lock, once per distinct literal.
                let ValueRef::Literal(literal) = ValueRef::new(ValueId::Literal(id), ctx) else {
                    unreachable!("a literal id resolves to a literal")
                };
                let value = SizedValue::new(literal.value(), literal.size());
                self.0[index] = Some(value);
                value
            }
        }
    }
}

/// Values produced by instructions, stored densely.
///
/// Every p-code operation that yields a value writes one here and its consumers
/// read it back, so this is one of the hottest structures in the interpreter. A
/// `FxHashMap<InstructionId, _>` hashes a `(function, local)` pair on every
/// access; instruction ids are dense and small, so a slot array indexed by
/// those two components removes the hashing entirely.
///
/// Sparse ids cost only an unused slot, which is what makes this safe to use
/// after a pass has removed instructions from a block.
#[derive(Debug, Default, Clone)]
pub struct InsnValues(Vec<Vec<Option<SizedValue>>>);

impl InsnValues {
    pub fn get(&self, id: &InstructionId) -> Option<&SizedValue> {
        let func: usize = id.func.into();
        let local: usize = id.local.into();
        self.0.get(func)?.get(local)?.as_ref()
    }

    pub fn insert(&mut self, id: InstructionId, value: SizedValue) {
        let func: usize = id.func.into();
        let local: usize = id.local.into();
        if func >= self.0.len() {
            self.0.resize_with(func + 1, Vec::new);
        }
        let slots = &mut self.0[func];
        if local >= slots.len() {
            slots.resize(local + 1, None);
        }
        slots[local] = Some(value);
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}

/// Type alias for an instruction hook function, which is called with the current instruction and emulator state after each instruction is executed.
type InstructionHook<M> = Box<dyn Fn(&InstructionRef<'_, '_>, &StandaloneEmulator<M>) + Send + Sync>;
type CallInterceptor<M> = Box<
    dyn FnMut(
            &Context<'_>,
            &mut StandaloneEmulator<M>,
            &CallSite,
        ) -> Result<CallInterception, Box<str>>
        + Send
        + Sync,
>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepEvent {
    Normal,
    DirectCallEntered(FunctionId),
    IndirectCallEntered,
    Return,
    ReturnValue,
    InterceptedCall,
}

/// A lifetime-free emulator that takes `&Context<'_>` explicitly on each call.
/// Use this when you need to store an emulator without a lifetime (e.g., across an FFI boundary).
pub struct StandaloneEmulator<M = EmulatedMemory> {
    pub memory: M,
    /// Literal values resolved once instead of per access.
    literal_cache: LiteralCache,
    /// Whether the module contains any array or list type, and the published
    /// type count that answer was valid for.
    ///
    /// Every `Load`, `Store` and `Range` is guarded by "is this operand
    /// sequence-typed", which otherwise costs an arena resolution plus two
    /// virtual calls *per operation* — about 9% of run time. Almost every module
    /// has no sequence types at all, and that is answerable once.
    sequence_types: bool,
    sequence_types_checked_at: Option<usize>,
    /// The current block's instruction list, and which block it belongs to.
    ///
    /// Resolving a block means two registry indexes (`bodies[func].blocks[local]`)
    /// and the interpreter did it on every step, though a block is entered once
    /// and then walked. Refreshed whenever the block changes *or* execution is
    /// at a block's first instruction — a block's contents can only change while
    /// nothing is part-way through it, which is exactly the case a VM that lifts
    /// on demand creates when it fills a placeholder block and re-enters it.
    cached_block: Option<BlockId>,
    cached_insns: Vec<qcode::value::LocalInsnId>,
    pub insn_values: InsnValues,
    pub block_param_values: FxHashMap<BlockParamId, SizedValue>,
    /// Block params bound to **poison** (argpromote v2): a symbolic pure-call
    /// argument whose bits are undefined. Reading one during emulation is a hard
    /// error (`PoisonRead`), so a pure-call fold whose result actually depends on
    /// a symbolic argument bails instead of computing on a bogus concrete value.
    pub poison_params: FxHashSet<BlockParamId>,
    /// Field values of aggregate-typed instruction results (`Tuple` results and,
    /// on return, the call instruction that produced them). `Extract` projects a
    /// field back out. Keeps the scalar `SizedValue` domain unchanged — the
    /// functional `argpromote` write-set is the only producer/consumer.
    pub aggregate_values: FxHashMap<InstructionId, Vec<SizedValue>>,
    /// Field values of aggregate-typed **block params** seeded by
    /// [`run_map_body`](Self::run_map_body) (the `enumerate` `(index, elem)` lane
    /// fed to a `map` body). `Extract` on such a param projects a field back out.
    pub block_param_aggregates: FxHashMap<BlockParamId, Vec<SizedValue>>,
    /// Little-endian byte buffers of array-typed instruction results — the value
    /// domain for the sequence intrinsics (`iota`/`singleton`/`insert`/`concat`),
    /// `Scan`, and array-typed `Store`. Kept out of the scalar `SizedValue`
    /// domain the same way [`aggregate_values`](Self::aggregate_values) keeps
    /// tuples out of it; a scalar `at(arr, i)` reads one lane back into
    /// `insn_values`.
    pub array_values: FxHashMap<InstructionId, Vec<u8>>,
    pub block: BlockId,
    pub idx: usize,
    /// Call stack maintained by `run_function` (outermost function first).
    pub call_stack: Vec<FunctionId>,
    /// The call instruction id for each active nested call, so a `Return` can
    /// deposit the callee's `Return.value` as that call's result.
    call_site_stack: Vec<InstructionId>,

    /// Disposable address lookup for the immutable module snapshot supplied by
    /// the caller. The lifetime-free emulator initializes this lazily because
    /// [`StandaloneEmulator::new`] intentionally takes no `Context`.
    address_index: Option<AddressIndex>,

    pub instruction_hook: Option<InstructionHook<M>>,
    call_interceptor: Option<CallInterceptor<M>>,
}

impl StandaloneEmulator<EmulatedMemory> {
    /// Builds an emulator over the default flat memory.
    pub fn new(entry: BlockId) -> Self {
        Self::new_in(entry)
    }

    /// Builds an emulator positioned at `addr`, over the default flat memory.
    pub fn from_address(ctx: &Context<'_>, addr: u64) -> Self {
        Self::from_address_in(ctx, addr)
    }
}

impl<M: EmulatorMemory + Default> StandaloneEmulator<M> {
    /// Builds an emulator over an explicit memory backend.
    ///
    /// [`new`](StandaloneEmulator::new) is the one to reach for with the default
    /// flat memory: Rust's default type parameters do not participate in
    /// inference, so a generic `new` would force every call site to name its
    /// backend.
    pub fn new_in(entry: BlockId) -> Self {
        Self {
            memory: M::default(),
            literal_cache: LiteralCache::default(),
            sequence_types: false,
            sequence_types_checked_at: None,
            cached_block: None,
            cached_insns: Vec::new(),
            insn_values: InsnValues::default(),
            block_param_values: FxHashMap::default(),
            poison_params: FxHashSet::default(),
            aggregate_values: FxHashMap::default(),
            block_param_aggregates: FxHashMap::default(),
            array_values: FxHashMap::default(),
            block: entry,
            idx: 0,
            call_stack: Vec::new(),
            call_site_stack: Vec::new(),
            address_index: None,
            instruction_hook: None,
            call_interceptor: None,
        }
    }

    /// Takes the cached address lookup, leaving the emulator without one.
    ///
    /// The index is derived state, built once from what was assumed to be an
    /// immutable module. A VM that lifts code on demand makes the module
    /// *mutable*, so it has to keep the index current instead. Moving the index
    /// out, updating it in place as blocks are added, and moving it back with
    /// [`set_address_index`](Self::set_address_index) keeps discovery O(1) —
    /// rebuilding it per lift is quadratic in the size of the module.
    pub fn take_address_index(&mut self) -> Option<AddressIndex> {
        self.address_index.take()
    }

    /// Installs an address lookup, replacing any cached one.
    pub fn set_address_index(&mut self, address_index: AddressIndex) {
        self.address_index = Some(address_index);
    }

    /// Resolves a guest address to the block that covers it, building the
    /// cached index if there is not one yet.
    pub fn block_at_address(&mut self, ctx: &Context<'_>, address: u64) -> Option<BlockId> {
        self.block_at(ctx, address)
    }

    fn with_address_index(entry: BlockId, address_index: AddressIndex) -> Self {
        let mut emulator = Self::new_in(entry);
        emulator.address_index = Some(address_index);
        emulator
    }

    fn resolve_block_at(ctx: &Context<'_>, index: &AddressIndex, address: u64) -> Option<BlockId> {
        match index.get(address) {
            Some(AddressTarget::Block(block)) => Some(block),
            Some(AddressTarget::Function(function)) => FunctionBody::from_id(ctx, function)
                .root()
                .map(|root| root.id),
            None => None,
        }
    }

    fn block_at(&mut self, ctx: &Context<'_>, address: u64) -> Option<BlockId> {
        let index = self
            .address_index
            .get_or_insert_with(|| AddressIndex::analyze(ctx));
        Self::resolve_block_at(ctx, index, address)
    }

    fn make_error(&self, ctx: &Context<'_>, kind: EmulatorErrorKind) -> EmulatorError {
        let block = BasicBlock::from_id(ctx, self.block);
        let instruction = block
            .instruction_ids()
            .get(self.idx)
            .copied()
            .or_else(|| block.instruction_ids().last().copied())
            .expect("cannot construct EmulatorError for empty block");

        EmulatorError::new(kind, &Instruction::from_id(ctx, instruction))
    }

    /// Construct an [`EmulatorErrorKind::EmptyBlock`] error for the current block
    /// without indexing into it. [`make_error`](Self::make_error) can't serve this
    /// case — it panics when the block has no instructions to attach context to.
    fn make_empty_block_error(&self, ctx: &Context<'_>) -> EmulatorError {
        let block = BasicBlock::from_id(ctx, self.block);
        EmulatorError {
            kind: EmulatorErrorKind::EmptyBlock(self.block),
            ctx: format!(
                "Block: {:?}\nFunction: {:?}",
                block.name(),
                block.function().map(|f| f.name())
            ),
            address: block.address(),
        }
    }

    fn make_error_at(
        &self,
        ctx: &Context<'_>,
        instruction: InstructionId,
        kind: EmulatorErrorKind,
    ) -> EmulatorError {
        EmulatorError::new(kind, &Instruction::from_id(ctx, instruction))
    }

    /// Builds an emulator positioned at `addr`, over an explicit backend.
    pub fn from_address_in(ctx: &Context<'_>, addr: u64) -> Self {
        let address_index = AddressIndex::analyze(ctx);
        let entry = Self::resolve_block_at(ctx, &address_index, addr)
            .expect("Invalid block or function address");
        let mut emulator = Self::with_address_index(entry, address_index);
        emulator.memory.configure_spaces(ctx);
        emulator
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
        self.memory.configure_spaces(ctx);
        let varnode = Varnode::from_id(ctx, id);
        let space = varnode.space().id;
        let addr = varnode.address() as u64;
        let size = varnode.size();
        self.memory.write(
            space.into(),
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
            .read(space.into(), SizedValue::from_u64(addr), size)
            .ok()
            .map(|v| v.as_bits())
    }

    pub fn get_value(&mut self, ctx: &Context<'_>, id: ValueId) -> Option<u64> {
        let mut tmp = TempInterpreter {
            memory: &mut self.memory,
            literals: &mut self.literal_cache,
            insn_values: &mut self.insn_values,
            block_param_values: &mut self.block_param_values,
            poison_params: &self.poison_params,
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
                space.into(),
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
        self.memory
            .read_bytes(space.into(), addr, size)
            .unwrap_or_default()
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
            literals: &mut self.literal_cache,
            insn_values: &mut self.insn_values,
            block_param_values: &mut self.block_param_values,
            poison_params: &self.poison_params,
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

    pub fn set_call_interceptor(
        &mut self,
        interceptor: impl FnMut(
            &Context<'_>,
            &mut StandaloneEmulator<M>,
            &CallSite,
        ) -> Result<CallInterception, Box<str>>
        + Send
        + Sync
        + 'static,
    ) {
        self.call_interceptor = Some(Box::new(interceptor));
    }

    pub fn clear_call_interceptor(&mut self) {
        self.call_interceptor = None;
    }

    pub fn read_memory(
        &mut self,
        ctx: &Context<'_>,
        space: impl Into<MemorySpaceId>,
        addr: u64,
        size: usize,
    ) -> Result<Vec<u8>, EmulatorErrorKind> {
        self.memory.configure_spaces(ctx);
        self.memory.read_bytes(space.into(), addr, size)
    }

    pub fn write_memory(
        &mut self,
        ctx: &Context<'_>,
        space: impl Into<MemorySpaceId>,
        addr: u64,
        value: &[u8],
    ) -> Result<(), EmulatorErrorKind> {
        self.memory.configure_spaces(ctx);
        self.memory.write_bytes(space.into(), addr, value)
    }

    /// Resolve a terminator's bare-local argument list; `func` is the owning
    /// function of the instruction the args came from (strict IR locality).
    fn collect_block_args(
        &mut self,
        ctx: &Context<'_>,
        func: FunctionId,
        args: &[LocalValueId],
    ) -> Result<Vec<SizedValue>, EmulatorErrorKind> {
        let mut tmp = TempInterpreter {
            memory: &mut self.memory,
            literals: &mut self.literal_cache,
            insn_values: &mut self.insn_values,
            block_param_values: &mut self.block_param_values,
            poison_params: &self.poison_params,
            ctx,
        };
        args.iter()
            .map(|&arg| tmp.get_value(arg.qualify(func)))
            .collect()
    }

    fn bind_block_args(
        &mut self,
        ctx: &Context<'_>,
        func: FunctionId,
        target: BlockId,
        args: &[LocalValueId],
    ) -> Result<(), EmulatorErrorKind> {
        let values = self.collect_block_args(ctx, func, args)?;
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

    /// Resolve a range of a register varnode used as a store destination.
    ///
    /// The flat SLEIGH emitter represents `ST1[8:10] = value` with a `range`
    /// value so it retains the destination byte offset. As a source, `range`
    /// means extracted bits; as a register-space store pointer, it is an
    /// lvalue and must instead mean the base register address plus that offset.
    /// Treating it as extracted bits writes to an address derived from the old
    /// high word, leaving the high 16 bits of an i80 register stale.
    fn register_range_store_address(
        &self,
        ctx: &Context<'_>,
        func: FunctionId,
        store: &Store,
    ) -> Option<u64> {
        let space = store.space.qualify(func);
        let space_id = space.shared()?;
        if !matches!(Space::from_id(ctx, space_id).ty, SpaceType::Register) {
            return None;
        }
        let ValueRef::Instruction(range) = ValueRef::new(store.ptr.qualify(func), ctx) else {
            return None;
        };
        let Mnemonic::Range(Range { src, start, .. }) = range.mnemonic() else {
            return None;
        };
        let ValueRef::Varnode(varnode) = ValueRef::new(src.qualify(func), ctx) else {
            return None;
        };
        let varnode = Varnode::from_id(ctx, varnode.id);
        (varnode.space().id == space_id).then_some(varnode.address() as u64 + *start as u64)
    }

    fn apply_call_continuation(
        &mut self,
        ctx: &Context<'_>,
        continuation: CallContinuation,
    ) -> Result<(), EmulatorErrorKind> {
        let target = match continuation {
            CallContinuation::Block(block) => block,
            CallContinuation::Address(addr) => self
                .block_at(ctx, addr)
                .ok_or(EmulatorErrorKind::InvalidBlockAddress(addr))?,
        };
        self.block = target;
        self.idx = 0;
        Ok(())
    }

    fn intercept_call(
        &mut self,
        ctx: &Context<'_>,
        block: BlockId,
        instruction: InstructionId,
        call: &Call,
    ) -> crate::Result<Option<StepEvent>> {
        let Some(mut interceptor) = self.call_interceptor.take() else {
            return Ok(None);
        };
        let target = require_real_callee(call.target)
            .map_err(|kind| self.make_error_at(ctx, instruction, kind))?;

        let site = CallSite {
            instruction,
            block,
            target,
            // Operands are stored bare-local; the CallSite is a boundary object,
            // so qualify with the call instruction's own function.
            args: call
                .args
                .iter()
                .map(|a| a.qualify(instruction.func))
                .collect(),
        };
        let result = interceptor(ctx, self, &site);
        self.call_interceptor = Some(interceptor);

        match result {
            Ok(CallInterception::PassThrough) => Ok(None),
            Ok(CallInterception::Handled(continuation)) => {
                self.apply_call_continuation(ctx, continuation)
                    .map_err(|kind| self.make_error_at(ctx, instruction, kind))?;
                Ok(Some(StepEvent::InterceptedCall))
            }
            Err(message) => Err(self.make_error_at(
                ctx,
                instruction,
                EmulatorErrorKind::InterceptError(message),
            )),
        }
    }

    /// Evaluate an operand through the scalar interpreter while retaining its
    /// full f80 payload (the public convenience getter intentionally returns
    /// only u64 values).
    fn scalar_value(
        &mut self,
        ctx: &Context<'_>,
        id: ValueId,
    ) -> Result<SizedValue, EmulatorErrorKind> {
        let mut interpreter = TempInterpreter {
            memory: &mut self.memory,
            literals: &mut self.literal_cache,
            insn_values: &mut self.insn_values,
            block_param_values: &mut self.block_param_values,
            poison_params: &self.poison_params,
            ctx,
        };
        interpreter.get_value(id)
    }

    /// Interpret the packed-integer SLEIGH user-ops that x86's MMX/SSE
    /// constructors leave as `pcodeop` applications. Returns `None` for every
    /// other user-op so the generic interpreter stays the implementation.
    ///
    /// `pavgb`/`pavgw` are applied by the spec to one lane at a time, so they
    /// are scalar here. `pmaddwd`/`pmulhuw` receive a whole vector and are
    /// width-generic, covering the 8-byte MMX and 16-byte XMM forms alike.
    fn interpret_packed_pcode_op(
        &mut self,
        ctx: &Context<'_>,
        insn: &InstructionRef<'_, '_>,
        mnemonic: &Mnemonic,
    ) -> Result<Option<SizedValue>, EmulatorErrorKind> {
        let Mnemonic::PCodeOp(op) = mnemonic else {
            return Ok(None);
        };
        let name = ctx.shared.pcode_ops[op.id].clone();
        let func = insn.id.func;

        // The packed-decimal conversions are the only unary user-ops handled
        // here, and both need the x87 context for rounding and status.
        if let [src] = op.args.as_slice() {
            let value = self.scalar_value(ctx, src.qualify(func))?;
            if value.size != 10 {
                return Ok(None);
            }
            let Some((control_varnode, status_register)) = Self::x87_context(ctx) else {
                return Ok(None);
            };
            let control = self
                .read_varnode_u128(ctx, control_varnode)
                .unwrap_or(0x037f) as u16;
            return Ok(match name.as_ref() {
                "from_bcd" => Some(SizedValue::from_f80_bits(float80::from_bcd(value.as_bits()))),
                "extract_significand" => Some(SizedValue::from_f80_bits(
                    float80::extract_significand(value.as_bits()),
                )),
                "extract_exponent" => {
                    let result = float80::extract_exponent(value.as_bits());
                    self.record_x87_status(
                        ctx,
                        control,
                        status_register,
                        result.status,
                        None,
                        Self::f80_is_denormal(value.as_bits()),
                    )?;
                    Some(SizedValue::from_f80_bits(result.bits))
                }
                "to_bcd" => {
                    let result = float80::to_bcd(value.as_bits(), control);
                    self.record_x87_status(
                        ctx,
                        control,
                        status_register,
                        result.status,
                        None,
                        false,
                    )?;
                    Some(SizedValue::from_bits(result.bits, 10))
                }
                _ => None,
            });
        }

        let [lhs, rhs] = op.args.as_slice() else {
            return Ok(None);
        };
        let lhs = self.scalar_value(ctx, lhs.qualify(func))?;
        let rhs = self.scalar_value(ctx, rhs.qualify(func))?;

        // Unsigned rounded average of one lane: (a + b + 1) >> 1, computed
        // wide enough that the carry out of the lane is kept.
        let average = |width: usize| -> Option<SizedValue> {
            (lhs.size as usize == width && rhs.size as usize == width).then(|| {
                let sum = lhs.as_bits() + rhs.as_bits() + 1;
                SizedValue::from_bits(sum >> 1, width)
            })
        };

        let value = match name.as_ref() {
            // FPREM and FPREM1 report the quotient's low three bits in
            // C0/C3/C1 and an incomplete reduction in C2, so the condition
            // codes are written here alongside the sticky exceptions.
            "fprem" | "fprem1" if lhs.size == 10 && rhs.size == 10 => {
                let Some((control_varnode, status_register)) = Self::x87_context(ctx) else {
                    return Ok(None);
                };
                let control = self
                    .read_varnode_u128(ctx, control_varnode)
                    .unwrap_or(0x037f) as u16;
                let result =
                    float80::remainder(lhs.as_bits(), rhs.as_bits(), name.as_ref() == "fprem1");
                self.record_x87_status(
                    ctx,
                    control,
                    status_register,
                    result.status,
                    None,
                    Self::f80_denormal_operands(lhs.as_bits(), rhs.as_bits()),
                )?;
                let old = self.read_varnode_u128(ctx, status_register).unwrap_or(0) as u16;
                // C0, C1, C2 and C3 are all operation results here.
                let mut new = old & !0x4700;
                if result.incomplete {
                    new |= 1 << 10;
                } else {
                    new |= u16::from(result.quotient as u8 & 1) << 9;
                    new |= u16::from((result.quotient >> 1) as u8 & 1) << 14;
                    new |= u16::from((result.quotient >> 2) as u8 & 1) << 8;
                }
                self.set_varnode(ctx, status_register, u64::from(new))?;
                return Ok(Some(SizedValue::from_f80_bits(result.bits)));
            }
            // FSCALE reads its scale from ST(1) and rounds under the x87
            // control word, so it is interpreted here rather than as a
            // generic packed lane operation.
            "fscale" if lhs.size == 10 && rhs.size == 10 => {
                let (control, status_register) = match Self::x87_context(ctx) {
                    Some((control, status)) => (
                        self.read_varnode_u128(ctx, control).unwrap_or(0x037f) as u16,
                        status,
                    ),
                    None => return Ok(None),
                };
                let result = float80::scale_contextual(lhs.as_bits(), rhs.as_bits(), control);
                self.record_x87_status(
                    ctx,
                    control,
                    status_register,
                    result.status,
                    None,
                    Self::f80_denormal_operands(lhs.as_bits(), rhs.as_bits()),
                )?;
                return Ok(Some(SizedValue::from_f80_bits(result.bits)));
            }
            "pavgb" => average(1),
            "pavgw" => average(2),
            // Unsigned 16x16 multiply per word lane, keeping the high half.
            "pmulhuw" => Self::packed_lanes(&lhs, &rhs, 2, |a, b| ((a * b) >> 16) & 0xffff),
            // Saturating packed add/subtract. A signed lane clamps to its
            // width's bounds; an unsigned lane clamps to zero and its maximum.
            "paddsb" => Self::saturating(&lhs, &rhs, 1, true, false),
            "paddsw" => Self::saturating(&lhs, &rhs, 2, true, false),
            "psubsb" => Self::saturating(&lhs, &rhs, 1, true, true),
            "psubsw" => Self::saturating(&lhs, &rhs, 2, true, true),
            "paddusb" => Self::saturating(&lhs, &rhs, 1, false, false),
            "paddusw" => Self::saturating(&lhs, &rhs, 2, false, false),
            "psubusb" => Self::saturating(&lhs, &rhs, 1, false, true),
            "psubusw" => Self::saturating(&lhs, &rhs, 2, false, true),
            // Signed 16x16 multiplies summed in pairs into each dword lane.
            "pmaddwd" => Self::packed_lanes(&lhs, &rhs, 4, |a, b| {
                let word = |v: u128, half: u32| i64::from(((v >> (half * 16)) & 0xffff) as u16 as i16);
                let product = word(a, 0) * word(b, 0) + word(a, 1) * word(b, 1);
                u128::from(product as u32)
            }),
            _ => None,
        };
        Ok(value)
    }

    /// Saturating packed add (`subtract` false) or subtract, per `width`-byte
    /// lane. `signed` selects signed bounds over unsigned ones.
    fn saturating(
        lhs: &SizedValue,
        rhs: &SizedValue,
        width: usize,
        signed: bool,
        subtract: bool,
    ) -> Option<SizedValue> {
        let bits = width * 8;
        Self::packed_lanes(lhs, rhs, width, |a, b| {
            if signed {
                let sign = |v: u128| (v as i128) - (((v >> (bits - 1)) & 1) as i128) * (1i128 << bits);
                let (a, b) = (sign(a), sign(b));
                let value = if subtract { a - b } else { a + b };
                let max = (1i128 << (bits - 1)) - 1;
                let min = -(1i128 << (bits - 1));
                (value.clamp(min, max) as u128) & ((1u128 << bits) - 1)
            } else if subtract {
                a.saturating_sub(b)
            } else {
                (a + b).min((1u128 << bits) - 1)
            }
        })
    }

    /// Apply `lane` to each `width`-byte lane of two equally sized vectors.
    /// Returns `None` unless both operands share a width that divides evenly
    /// into lanes.
    fn packed_lanes(
        lhs: &SizedValue,
        rhs: &SizedValue,
        width: usize,
        lane: impl Fn(u128, u128) -> u128,
    ) -> Option<SizedValue> {
        let size = lhs.size as usize;
        if size != rhs.size as usize || size == 0 || size % width != 0 {
            return None;
        }
        let bits = width * 8;
        let mask = (1u128 << bits) - 1;
        let mut out = 0u128;
        for index in 0..size / width {
            let shift = index * bits;
            let a = (lhs.as_bits() >> shift) & mask;
            let b = (rhs.as_bits() >> shift) & mask;
            out |= (lane(a, b) & mask) << shift;
        }
        Some(SizedValue::from_bits(out, size))
    }

    fn x87_context(ctx: &Context<'_>) -> Option<(VarnodeId, VarnodeId)> {
        let ValueId::Varnode(control) = ctx.get_named("FPUControlWord")? else {
            return None;
        };
        let ValueId::Varnode(status) = ctx.get_named("FPUStatusWord")? else {
            return None;
        };
        Some((control, status))
    }

    /// x87 defaults to masked exceptions.  We always produce APFloat's default
    /// result and make every exception sticky.  An unmasked exception sets ES,
    /// but does not yet transfer control to a hardware exception handler; that
    /// deliberately non-trapping policy keeps the generic emulator API intact
    /// until architectural trap delivery is modelled.
    ///
    /// Not trapping would also let an unmasked instruction commit a result
    /// hardware discards, but that part is handled a level up: the constructors
    /// guard their own commit with `fpu_raised_unmasked`, so an aborted FMULP
    /// never reaches its pop and an aborted FRNDINT never reaches the `round`
    /// whose precision flag this function would record.  Only IE, DE and ZE
    /// abort there - #P stores the rounded result before trapping and #O/#U
    /// store an exponent-scaled one, so those still commit and still come
    /// through here.
    ///
    /// What remains unmodelled is delivery, not the abort: hardware defers the
    /// trap to the next floating-point instruction.  Both agree that the
    /// faulting instruction leaves x87 state unmodified, which is all the
    /// register and status comparisons observe.
    fn record_x87_status(
        &mut self,
        ctx: &Context<'_>,
        control: u16,
        status_register: VarnodeId,
        ap_status: Status,
        rounded_up: Option<bool>,
        denormal_operand: bool,
    ) -> Result<(), EmulatorErrorKind> {
        // A divide by zero is likewise decided by the zero divisor, not by the
        // dividend, so a denormal dividend goes unreported: hardware raises ZE
        // alone.
        let denormal_operand = denormal_operand && !ap_status.contains(Status::DIV_BY_ZERO);
        let mut exceptions = u16::from(denormal_operand) << 1;
        if ap_status.contains(Status::INVALID_OP) {
            exceptions |= 1 << 0;
        }
        if ap_status.contains(Status::DIV_BY_ZERO) {
            exceptions |= 1 << 2;
        }
        if ap_status.contains(Status::OVERFLOW) {
            exceptions |= 1 << 3;
        }
        if ap_status.contains(Status::UNDERFLOW) {
            exceptions |= 1 << 4;
        }
        if ap_status.contains(Status::INEXACT) {
            exceptions |= 1 << 5;
        }
        if exceptions == 0 && rounded_up.is_none() {
            return Ok(());
        }
        let old = self.read_varnode_u128(ctx, status_register).unwrap_or(0) as u16;
        let mut new = old | exceptions;
        // C1 records whether an inexact result was rounded away from zero. It
        // is an operation result, not a sticky exception bit, so an operation
        // that reports it writes it every time: a result that did not round up
        // - including an exact one - clears C1 rather than leaving the previous
        // instruction's answer standing.
        if let Some(rounded_up) = rounded_up {
            new = (new & !(1 << 9)) | (u16::from(rounded_up && ap_status.contains(Status::INEXACT)) << 9);
        }
        if exceptions & !control & 0x003f != 0 {
            // ES: one or more unmasked exceptions are pending. B mirrors ES on
            // every processor since the 387 - it reported the 8087's BUSY line,
            // and is now recomputed from the same condition.
            new |= (1 << 7) | (1 << 15);
        }
        self.set_varnode(ctx, status_register, u64::from(new))
    }

    fn toward_zero_control(control: u16) -> u16 {
        (control & !0x0c00) | 0x0c00
    }

    fn rounded_away_from_zero(value: u128, toward_zero: u128, size: usize) -> bool {
        let magnitude_mask = (1u128 << (size * 8 - 1)) - 1;
        (value & magnitude_mask) > (toward_zero & magnitude_mask)
    }

    /// x87 reports a denormal operand only when the operation actually works
    /// on it. A NaN operand determines the result on its own, so hardware
    /// raises invalid or nothing at all and leaves DE clear - the same
    /// precedence `fpu_signal_denormal2` applies in the specification.
    fn f80_denormal_operands(lhs: u128, rhs: u128) -> bool {
        if float80::is_nan(lhs) || float80::is_nan(rhs) {
            return false;
        }
        Self::f80_is_denormal(lhs) || Self::f80_is_denormal(rhs)
    }

    fn f80_is_denormal(value: u128) -> bool {
        let exponent = (value >> 64) & 0x7fff;
        exponent == 0 && value & ((1u128 << 64) - 1) != 0
    }

    fn scalar_is_denormal(value: SizedValue) -> bool {
        match value.size as usize {
            4 => value.as_u64() & 0x7f80_0000 == 0 && value.as_u64() & 0x007f_ffff != 0,
            8 => {
                value.as_u64() & 0x7ff0_0000_0000_0000 == 0
                    && value.as_u64() & 0x000f_ffff_ffff_ffff != 0
            }
            10 => Self::f80_is_denormal(value.as_bits()),
            _ => false,
        }
    }

    /// Interpret f80 operations with the x87 control/status words in scope.
    /// Returns `None` for all non-x87 operations so the generic DomainValue
    /// interpreter remains the implementation for every other architecture.
    fn interpret_x87_float(
        &mut self,
        ctx: &Context<'_>,
        insn: &InstructionRef<'_, '_>,
        mnemonic: &Mnemonic,
    ) -> Result<Option<SizedValue>, EmulatorErrorKind> {
        // Every instruction on the generic path reaches this function, so the
        // structural test comes first: resolving "FPUControlWord" and
        // "FPUStatusWord" by name per p-code operation cost about 8% of run
        // time in a profile of a loop containing no floating point at all.
        // These are the only shapes the match below handles.
        if !matches!(
            mnemonic,
            Mnemonic::Binop(Binary {
                op: Binop::Float(_),
                ..
            }) | Mnemonic::Unop(Unary {
                op: Unop::FloatSqrt | Unop::FloatRound,
                ..
            }) | Mnemonic::IntToFloat(_)
                | Mnemonic::FloatToInt(_)
                | Mnemonic::FloatToFloat(_)
        ) {
            return Ok(None);
        }
        let Some((control_register, status_register)) = Self::x87_context(ctx) else {
            return Ok(None);
        };
        let control = self
            .read_varnode_u128(ctx, control_register)
            .unwrap_or(0x037f) as u16;
        let func = insn.id.func;
        match mnemonic {
            Mnemonic::Binop(Binary {
                op: Binop::Float(operation),
                lhs,
                rhs,
            }) => {
                let lhs = self.scalar_value(ctx, lhs.qualify(func))?;
                let rhs = self.scalar_value(ctx, rhs.qualify(func))?;
                if lhs.size != 10 || rhs.size != 10 {
                    return Ok(None);
                }
                let toward_zero = Self::toward_zero_control(control);
                let (result, zero_result) = match operation {
                    FloatBinop::Add => (
                        float80::add_contextual(lhs.as_bits(), rhs.as_bits(), control),
                        float80::add_contextual(lhs.as_bits(), rhs.as_bits(), toward_zero),
                    ),
                    FloatBinop::Sub => (
                        float80::sub_contextual(lhs.as_bits(), rhs.as_bits(), control),
                        float80::sub_contextual(lhs.as_bits(), rhs.as_bits(), toward_zero),
                    ),
                    FloatBinop::Mul => (
                        float80::mul_contextual(lhs.as_bits(), rhs.as_bits(), control),
                        float80::mul_contextual(lhs.as_bits(), rhs.as_bits(), toward_zero),
                    ),
                    FloatBinop::Div => (
                        float80::div_contextual(lhs.as_bits(), rhs.as_bits(), control),
                        float80::div_contextual(lhs.as_bits(), rhs.as_bits(), toward_zero),
                    ),
                    // A comparison is quiet: only a signalling NaN raises
                    // invalid here. x87's *ordered* compares also raise it for
                    // a quiet NaN, which the FCOM constructors add explicitly,
                    // because the p-code is identical for FCOM and FUCOM and
                    // cannot distinguish them. Comparisons keep their generic
                    // boolean result and only update the sticky status word.
                    FloatBinop::Equal
                    | FloatBinop::NotEqual
                    | FloatBinop::Less
                    | FloatBinop::LessEqual => {
                        if float80::is_signaling_nan(lhs.as_bits())
                            || float80::is_signaling_nan(rhs.as_bits())
                        {
                            self.record_x87_status(
                                ctx,
                                control,
                                status_register,
                                Status::INVALID_OP,
                                None,
                                false,
                            )?;
                        }
                        return Ok(None);
                    }
                    _ => return Ok(None),
                };
                self.record_x87_status(
                    ctx,
                    control,
                    status_register,
                    result.status,
                    Some(Self::rounded_away_from_zero(
                        result.bits,
                        zero_result.bits,
                        10,
                    )),
                    Self::f80_denormal_operands(lhs.as_bits(), rhs.as_bits()),
                )?;
                Ok(Some(SizedValue::from_f80_bits(result.bits)))
            }
            // FSQRT rounds under the control word and reports inexact and the
            // C1 rounding indicator, so it takes the contextual path rather
            // than the generic sqrt, which narrows f80 through f64.
            Mnemonic::Unop(Unary {
                op: Unop::FloatSqrt,
                src,
            }) => {
                let value = self.scalar_value(ctx, src.qualify(func))?;
                if value.size != 10 {
                    return Ok(None);
                }
                let result = float80::sqrt_contextual(value.as_bits(), control);
                let zero_result = float80::sqrt_contextual(
                    value.as_bits(),
                    Self::toward_zero_control(control),
                );
                self.record_x87_status(
                    ctx,
                    control,
                    status_register,
                    result.status,
                    Some(Self::rounded_away_from_zero(
                        result.bits,
                        zero_result.bits,
                        10,
                    )),
                    Self::f80_is_denormal(value.as_bits()),
                )?;
                Ok(Some(SizedValue::from_f80_bits(result.bits)))
            }
            Mnemonic::Unop(Unary {
                op: Unop::FloatRound,
                src,
            }) => {
                let value = self.scalar_value(ctx, src.qualify(func))?;
                if value.size != 10 {
                    return Ok(None);
                }
                let result = float80::round_to_integral_contextual(value.as_bits(), control);
                let zero_result = float80::round_to_integral_contextual(
                    value.as_bits(),
                    Self::toward_zero_control(control),
                );
                self.record_x87_status(
                    ctx,
                    control,
                    status_register,
                    result.status,
                    Some(Self::rounded_away_from_zero(
                        result.bits,
                        zero_result.bits,
                        10,
                    )),
                    Self::f80_is_denormal(value.as_bits()),
                )?;
                Ok(Some(SizedValue::from_f80_bits(result.bits)))
            }
            // FILD converts exactly to the extended format.  PC applies to
            // arithmetic results, not this integer load.
            Mnemonic::IntToFloat(IntToFloat { size: 10, .. }) => Ok(None),
            Mnemonic::FloatToInt(FloatToInt { src, size }) => {
                let value = self.scalar_value(ctx, src.qualify(func))?;
                if value.size != 10 {
                    return Ok(None);
                }
                let result = float80::truncate_to_i128(value.as_bits(), size * 8);
                self.record_x87_status(
                    ctx,
                    control,
                    status_register,
                    result.status,
                    None,
                    // FIST/FISTP/FISTTP convert ST(0), always a register, so
                    // there is no denormal *memory* operand to report. A
                    // denormal source is converted like any other value and
                    // reports precision, not DE. Only invalid and precision
                    // are architecturally raised here.
                    false,
                )?;
                Ok(Some(SizedValue::from_bits(result.value as u128, *size)))
            }
            Mnemonic::FloatToFloat(FloatToFloat { src, size }) => {
                let value = self.scalar_value(ctx, src.qualify(func))?;
                if value.size == 10 && matches!(*size, 4 | 8) {
                    let result = float80::to_float_contextual(value.as_bits(), *size, control);
                    let zero_result = float80::to_float_contextual(
                        value.as_bits(),
                        *size,
                        Self::toward_zero_control(control),
                    );
                    self.record_x87_status(
                        ctx,
                        control,
                        status_register,
                        result.status,
                        Some(Self::rounded_away_from_zero(
                            result.bits,
                            zero_result.bits,
                            *size,
                        )),
                        // Narrowing an extended-precision register to f32/f64
                        // does not raise the denormal-operand exception. A
                        // denormal source too small for the destination is
                        // reported as underflow and inexact instead. DE is for
                        // a denormal *memory* operand, handled by the widening
                        // arm below.
                        false,
                    )?;
                    return Ok(Some(SizedValue::from_bits(result.bits, *size)));
                }
                if *size == 10 && Self::scalar_is_denormal(value) {
                    self.record_x87_status(ctx, control, status_register, Status::OK, None, true)?;
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    fn step_with_event(&mut self, ctx: &Context<'_>) -> crate::Result<StepEvent> {
        self.memory.configure_spaces(ctx);
        let block_id = self.block;
        if self.cached_block != Some(block_id) || self.idx == 0 {
            // The module only gains types while nothing is part-way through a
            // block — lifting happens on block entry — so this rides the same
            // refresh as the instruction list rather than paying per step.
            self.refresh_sequence_types(ctx);
            self.cached_insns.clear();
            self.cached_insns
                .extend_from_slice(ctx.block(block_id).instruction_ids());
            self.cached_block = Some(block_id);
        }
        // A degenerate block — empty, or exhausted without a terminator — is
        // malformed lifter output, not an emulator bug. Report it so a bounded
        // consumer (and a VM running lifted-on-demand code) can stop with a
        // reason instead of aborting the process.
        let Some(&local) = self.cached_insns.get(self.idx) else {
            return Err(self.make_empty_block_error(ctx));
        };
        let insn_id = InstructionId::new(block_id.func, local);
        let insn = InstructionRef::from_id(ctx, insn_id);
        let id = insn.id;

        if let Some(hook) = self.instruction_hook.as_ref() {
            hook(&insn, self)
        }

        // Resolved once and threaded through the step. Each `insn.mnemonic()`
        // re-walks `view.instruction(id)`, two registry lookups deep, and the
        // step path asked for the same instruction's mnemonic several times.
        let mnemonic = insn.mnemonic();

        match mnemonic {
            Mnemonic::Branch(Branch { target, args }) => {
                // Terminator targets are bare body-local indices in the terminator's
                // own arena (`id.func`); qualify to the current block's function.
                let target = BlockId::new(id.func, *target);
                self.bind_block_args(ctx, id.func, target, args)
                    .map_err(|kind| self.make_error(ctx, kind))?;
                self.block = target;
                self.idx = 0;
            }

            Mnemonic::Call(call) => {
                if let Some(event) = self.intercept_call(ctx, block_id, insn_id, call)? {
                    return Ok(event);
                }
                let target =
                    require_real_callee(call.target).map_err(|kind| self.make_error(ctx, kind))?;
                self.block = FunctionBody::from_id(ctx, target)
                    .root()
                    .ok_or_else(|| {
                        self.make_error(ctx, EmulatorErrorKind::EmptyFunctionRoot(target))
                    })?
                    .id;
                self.idx = 0;
                // Remember this call so the matching `Return` can deposit the
                // callee's `Return.value` as this call's (aggregate) result.
                self.call_site_stack.push(insn_id);
                return Ok(StepEvent::DirectCallEntered(target));
            }

            Mnemonic::TailCall(tc) => {
                // A tail call pops our frame and transfers to the callee's entry;
                // the callee's `Return` returns directly to *our* caller. Mirror the
                // old tail-`Branch`-into-entry behavior: jump to the callee root
                // without pushing a call frame.
                let target =
                    require_real_callee(tc.target).map_err(|kind| self.make_error(ctx, kind))?;
                self.block = FunctionBody::from_id(ctx, target)
                    .root()
                    .ok_or_else(|| {
                        self.make_error(ctx, EmulatorErrorKind::EmptyFunctionRoot(target))
                    })?
                    .id;
                self.idx = 0;
            }

            Mnemonic::Apply(apply) => {
                const APPLY_STEP_BUDGET: usize = 100_000;
                let target =
                    require_real_callee(apply.target).map_err(|kind| self.make_error(ctx, kind))?;
                let args = self
                    .collect_block_args(ctx, id.func, &apply.args)
                    .map_err(|kind| self.make_error(ctx, kind))?;
                let root = FunctionBody::from_id(ctx, target)
                    .root()
                    .ok_or_else(|| {
                        self.make_error(ctx, EmulatorErrorKind::EmptyFunctionRoot(target))
                    })?
                    .id;
                let mut nested = StandaloneEmulator::<M>::new_in(root);
                nested
                    .run_pure(ctx, target, &args, APPLY_STEP_BUDGET)
                    .map_err(|e| self.make_error(ctx, e.kind))?;
                let ret_value = lambda_return_value(ctx, nested.current_block())
                    .ok_or_else(|| self.make_error(ctx, EmulatorErrorKind::ValueError(0)))?;
                if let Some(value) = nested.get_value(ctx, ret_value) {
                    let size = ctx
                        .stored_type_of(ret_value)
                        .map(|ty| ctx.shared.types.size_of(ty))
                        .unwrap_or(8);
                    self.insn_values
                        .insert(insn_id, SizedValue::new(value, size));
                } else if let ValueId::Instruction(ret_id) = ret_value
                    && let Some(agg) = nested.aggregate_values.get(&ret_id).cloned()
                {
                    // The lambda returns an aggregate (e.g. the result tuple produced
                    // by accumulator elimination); propagate it field-wise so the
                    // caller's `extract(apply, i)` resolves — mirroring the scalar
                    // case above and the `Return` arm's call-result handling.
                    self.aggregate_values.insert(insn_id, agg);
                }
                self.idx += 1;
            }

            Mnemonic::CBranch(CBranch {
                condition,
                success_block: target,
                success_args,
                failure_block: fallthrough,
                failure_args,
            }) => {
                let cond_val = self.get_value(ctx, condition.qualify(id.func)).unwrap();
                let target = BlockId::new(id.func, *target);
                let fallthrough = BlockId::new(id.func, *fallthrough);
                if cond_val != 0 {
                    self.bind_block_args(ctx, id.func, target, success_args)
                        .map_err(|kind| self.make_error(ctx, kind))?;
                    self.block = target;
                } else {
                    self.bind_block_args(ctx, id.func, fallthrough, failure_args)
                        .map_err(|kind| self.make_error(ctx, kind))?;
                    self.block = fallthrough;
                }
                self.idx = 0;
            }

            Mnemonic::Switch(switch) => {
                let value = self
                    .get_value(ctx, switch.scrutinee.qualify(id.func))
                    .unwrap();
                let arm = switch
                    .cases
                    .iter()
                    .find(|case| case.value == value)
                    .map(|case| (case.target, &case.args));
                let (target, args) = match arm
                    .or_else(|| switch.default.map(|target| (target, &switch.default_args)))
                {
                    Some(arm) => arm,
                    // No arm matches and there is no default. A table behind a
                    // bounds check is total over its listed values, so arriving
                    // here means the guard that guaranteed that was wrong.
                    None => {
                        return Err(
                            self.make_error(ctx, EmulatorErrorKind::InvalidBlockAddress(value))
                        );
                    }
                };
                let target = BlockId::new(id.func, target);
                self.bind_block_args(ctx, id.func, target, args)
                    .map_err(|kind| self.make_error(ctx, kind))?;
                self.block = target;
                self.idx = 0;
            }

            Mnemonic::BranchInd(BranchInd { ptr }) => {
                let addr = self.get_value(ctx, ptr.qualify(id.func)).unwrap();
                let target = self.block_at(ctx, addr).ok_or_else(|| {
                    self.make_error(ctx, EmulatorErrorKind::InvalidBlockAddress(addr))
                })?;
                self.block = target;
                self.idx = 0;
            }

            Mnemonic::CallInd(CallInd { ptr, .. }) => {
                let addr = self.get_value(ctx, ptr.qualify(id.func)).unwrap();
                let target = self.block_at(ctx, addr).ok_or_else(|| {
                    self.make_error(ctx, EmulatorErrorKind::InvalidBlockAddress(addr))
                })?;
                self.block = target;
                self.idx = 0;
                self.call_site_stack.push(insn_id);
                return Ok(StepEvent::IndirectCallEntered);
            }

            Mnemonic::Return(Return { ptr, value, .. }) => {
                // Deposit the callee's return value as the result of the call that
                // entered it: an aggregate (the functional write-set) is copied
                // field-wise; a scalar return is copied through. This is what makes
                // a caller's `extract(call, i)` see the callee's effects.
                if let Some(call_id) = self.call_site_stack.pop() {
                    if let Some(LocalValueId::Instruction(src_local)) = value {
                        let src = InstructionId::new(id.func, *src_local);
                        if let Some(agg) = self.aggregate_values.get(&src).cloned() {
                            self.aggregate_values.insert(call_id, agg);
                        } else if let Some(scalar) = self.insn_values.get(&src).copied() {
                            self.insn_values.insert(call_id, scalar);
                        }
                    }
                    // v2 implicit convention: an `Opaque` (non-regpure) call to a
                    // *materialized* callee binds outputs by storing each return-pack
                    // slot back to its mapped register (post-mem2reg the callee body
                    // may no longer write those registers directly). A regpure site
                    // replays the pack itself, so it is skipped.
                    self.writeback_materialized_outputs(ctx, call_id, id.func);
                }

                let addr = self.get_value(ctx, ptr.qualify(id.func)).unwrap();
                let target = self.block_at(ctx, addr).ok_or_else(|| {
                    self.make_error(ctx, EmulatorErrorKind::InvalidBlockAddress(addr))
                })?;
                self.block = target;
                self.idx = 0;
                return Ok(StepEvent::Return);
            }

            Mnemonic::ReturnValue(_) => {
                return Ok(StepEvent::ReturnValue);
            }

            // Aggregate construction: evaluate each field and stash the field
            // vector. Fields are scalar for the register write-set (nested
            // aggregates, e.g. the RAM channel, are not modelled here yet).
            Mnemonic::Tuple(Tuple { fields }) => {
                let fields = fields.clone();
                let mut vals: Vec<SizedValue> = Vec::with_capacity(fields.len());
                for f in fields {
                    let val = {
                        let mut tmp = TempInterpreter {
                            memory: &mut self.memory,
            literals: &mut self.literal_cache,
                            insn_values: &mut self.insn_values,
                            block_param_values: &mut self.block_param_values,
                            poison_params: &self.poison_params,
                            ctx,
                        };
                        tmp.get_value(f.qualify(id.func))
                    };
                    vals.push(val.map_err(|kind| self.make_error(ctx, kind))?);
                }
                self.aggregate_values.insert(insn_id, vals);
                self.idx += 1;
            }

            // Aggregate projection: pull field `index` out of the stashed vector.
            Mnemonic::Extract(Extract { agg, index }) => {
                let field = match agg {
                    LocalValueId::Instruction(agg_local) => self
                        .aggregate_values
                        .get(&InstructionId::new(id.func, *agg_local))
                        .and_then(|v| v.get(*index))
                        .copied(),
                    // A `map` body's `enumerate` lane arrives as an aggregate
                    // block param seeded by `run_map_body`.
                    LocalValueId::BlockParam(pid_local) => self
                        .block_param_aggregates
                        .get(&BlockParamId::new(id.func, *pid_local))
                        .and_then(|v| v.get(*index))
                        .copied(),
                    _ => None,
                };
                if let Some(field) = field {
                    self.insn_values.insert(insn_id, field);
                }
                self.idx += 1;
            }

            // Whole-array load: the region snapshot `%l0 = load(ram:{N*esz}, base)`
            // that `array_promote` reads once before an original-array scan/map.
            // Its result is array-typed, so it lives in `array_values`; a scalar
            // load falls through to the generic interpreter below.
            Mnemonic::Load(load) if self.is_array_operand(ctx, ValueId::Instruction(insn_id)) => {
                let (space, ptr, size) = (load.space, load.ptr.qualify(id.func), load.size);
                let addr = self
                    .get_value(ctx, ptr)
                    .ok_or_else(|| self.make_error(ctx, EmulatorErrorKind::ValueError(0)))?;
                let buf = self
                    .read_memory(ctx, space.qualify(id.func), addr, size)
                    .map_err(|kind| self.make_error(ctx, kind))?;
                self.array_values.insert(insn_id, buf);
                self.idx += 1;
            }

            // Whole-array store: the promoted buffer written back to memory in one
            // shot (`store(ram, base <- arr)` at loop exit). A scalar store falls
            // through to the generic interpreter below.
            Mnemonic::Store(store)
                if self
                    .register_range_store_address(ctx, id.func, store)
                    .is_some() =>
            {
                let address = self
                    .register_range_store_address(ctx, id.func, store)
                    .expect("guard checked register range store address");
                let mut tmp = TempInterpreter {
                    memory: &mut self.memory,
            literals: &mut self.literal_cache,
                    insn_values: &mut self.insn_values,
                    block_param_values: &mut self.block_param_values,
                    poison_params: &self.poison_params,
                    ctx,
                };
                let value = tmp.get_value(store.src.qualify(id.func));
                drop(tmp);
                let value = value.map_err(|kind| self.make_error(ctx, kind))?;
                self.memory
                    .write(
                        store.space.qualify(id.func),
                        SizedValue::from_u64(address),
                        store.size,
                        value,
                    )
                    .map_err(|kind| self.make_error(ctx, kind))?;
                self.idx += 1;
            }

            Mnemonic::Store(store) if self.is_array_operand(ctx, store.src.qualify(id.func)) => {
                let (space, ptr, src) = (
                    store.space,
                    store.ptr.qualify(id.func),
                    store.src.qualify(id.func),
                );
                let buf = self
                    .resolve_array(ctx, src)
                    .ok_or_else(|| self.make_error(ctx, EmulatorErrorKind::ValueError(0)))?;
                let addr = self
                    .get_value(ctx, ptr)
                    .ok_or_else(|| self.make_error(ctx, EmulatorErrorKind::ValueError(0)))?;
                self.write_memory(ctx, space.qualify(id.func), addr, &buf)
                    .map_err(|kind| self.make_error(ctx, kind))?;
                self.idx += 1;
            }

            // Total left-scan: thread the accumulator through every lane, running
            // the (pure) binary body once per element, and materialize the result
            // as an array buffer.
            Mnemonic::Scan(scan) => {
                let scan = scan.clone();
                self.eval_scan(ctx, insn_id, &scan)
                    .map_err(|kind| self.make_error(ctx, kind))?;
                self.idx += 1;
            }

            // Total map: apply the pure unary body to every source element.
            Mnemonic::Map(map) => {
                let map = map.clone();
                self.eval_map(ctx, insn_id, &map)
                    .map_err(|kind| self.make_error(ctx, kind))?;
                self.idx += 1;
            }

            // Array slice: `arr[start:start+size]` on an array-typed source is a
            // sub-buffer (e.g. `l0[1..]`, the original-array scan source), kept in
            // the `array_values` domain. A scalar `Range` (bit-field extract) falls
            // through to the generic interpreter below.
            Mnemonic::Range(range) if self.is_array_operand(ctx, range.src.qualify(id.func)) => {
                let (src, start, size) = (range.src.qualify(id.func), range.start, range.size);
                let buf = self
                    .resolve_array(ctx, src)
                    .ok_or_else(|| self.make_error(ctx, EmulatorErrorKind::ValueError(0)))?;
                let end = (start + size).min(buf.len());
                let slice = buf.get(start..end).unwrap_or(&[]).to_vec();
                self.array_values.insert(insn_id, slice);
                self.idx += 1;
            }

            // The sequence intrinsics whose value is an *array* (or a lane read out
            // of one) live in the `array_values` domain rather than scalar
            // `insn_values`; every other (scalar) intrinsic falls through to the
            // generic interpreter.
            Mnemonic::Intrinsic(app) if is_array_intrinsic(app.id.name()) => {
                let name = app.id.name();
                let args: Vec<ValueId> = app.args.iter().map(|a| a.qualify(id.func)).collect();
                self.eval_array_intrinsic(ctx, insn_id, name, &args)
                    .map_err(|kind| self.make_error(ctx, kind))?;
                self.idx += 1;
            }

            _ => {
                if let Some(value) = self
                    .interpret_packed_pcode_op(ctx, &insn, mnemonic)
                    .map_err(|kind| self.make_error(ctx, kind))?
                {
                    self.insn_values.insert(id, value);
                    self.idx += 1;
                    return Ok(StepEvent::Normal);
                }
                if let Some(value) = self
                    .interpret_x87_float(ctx, &insn, mnemonic)
                    .map_err(|kind| self.make_error(ctx, kind))?
                {
                    self.insn_values.insert(id, value);
                    self.idx += 1;
                    return Ok(StepEvent::Normal);
                }
                let mut tmp = TempInterpreter {
                    memory: &mut self.memory,
            literals: &mut self.literal_cache,
                    insn_values: &mut self.insn_values,
                    block_param_values: &mut self.block_param_values,
                    poison_params: &self.poison_params,
                    ctx,
                };
                if let Some(value) = tmp.interpret(insn)? {
                    self.insn_values.insert(id, value);
                }
                self.idx += 1;
            }
        }

        Ok(StepEvent::Normal)
    }

    pub fn step(&mut self, ctx: &Context<'_>) -> crate::Result<()> {
        self.step_with_event(ctx).map(|_| ())
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
        let target = self
            .block_at(ctx, addr)
            .ok_or_else(|| self.make_error(ctx, EmulatorErrorKind::UnknownAddress(addr)))?;
        while self.block != target {
            self.run_block(ctx)?;
        }
        Ok(())
    }

    /// Runs the given function from its root block, stopping when the
    /// outermost `Return` is reached (without executing it).
    /// Nested calls are tracked via `call_depth` so inner returns are handled normally.
    /// The `call_stack` field is updated throughout execution.
    /// Seed a function's entry-block params with the values of the registers
    /// they were promoted from.
    ///
    /// `mem2reg` turns each register that is live-in to a function into a
    /// root-block parameter named after that register — these are the function's
    /// arguments. Entering the function (at the top level or via a call) binds
    /// those params from the current register file, so the callee receives the
    /// caller's register state through the calling convention. Params with no
    /// matching register (e.g. promoted stack slots) are left unbound.
    /// v2 implicit binding convention: store a materialized callee's return-pack
    /// slots back into their mapped registers on return from an `Opaque` call
    /// site. A `regpure` site is skipped (it replays the pack in its own body),
    /// as is any callee that is not materialized (no mapping to write back).
    fn writeback_materialized_outputs(
        &mut self,
        ctx: &Context<'_>,
        call_id: InstructionId,
        callee: FunctionId,
    ) {
        if call_is_regpure(ctx, call_id) {
            return;
        }
        let outputs = match &FunctionBody::from_id(ctx, callee).effects().register {
            qcode::value::RegisterChannelState::Materialized(map) => map.outputs.clone(),
            _ => return,
        };
        let Some(agg) = self.aggregate_values.get(&call_id).cloned() else {
            return;
        };
        for (field, &reg) in agg.iter().zip(&outputs) {
            let _ = self.set_varnode_u128(ctx, reg, field.as_bits());
        }
    }

    fn seed_entry_params(&mut self, ctx: &Context<'_>, func: FunctionId) {
        let Some(root) = FunctionBody::from_id(ctx, func).root() else {
            return;
        };
        let root_id = root.id;
        enum Seed {
            Reg(VarnodeId),
            /// A materialized global value input: the param's origin is its
            /// address literal, and its value is the *contents* at that address
            /// (the RAM channel threads `mem[addr]` by value — see
            /// `argpromote::ram`'s global materialization). Implicit binding
            /// therefore reads memory at the literal, not the literal itself.
            Lit(u64),
        }
        let params: Vec<(BlockParamId, Option<Seed>, usize)> = BasicBlock::from_id(ctx, root_id)
            .params()
            .map(|param| {
                let src = param
                    .name()
                    .and_then(|name| ctx.get_named(name))
                    .and_then(|value| match value {
                        ValueId::Varnode(id) => Some(Seed::Reg(id)),
                        _ => None,
                    })
                    .or_else(|| match param.origin() {
                        Some(ValueId::Literal(_)) => {
                            let ValueRef::Literal(lit) = ValueRef::new(param.origin()?, ctx) else {
                                return None;
                            };
                            Some(Seed::Lit(lit.value()))
                        }
                        _ => None,
                    });
                (param.id, src, param.size())
            })
            .collect();
        for (param_id, src, size) in params {
            let value = match src {
                Some(Seed::Reg(varnode_id)) => self.read_varnode(ctx, varnode_id),
                Some(Seed::Lit(addr)) => {
                    // The literal is the global's *address*; the param carries the
                    // value stored there. Seed from memory (little-endian, param
                    // width) rather than the raw literal.
                    let space = ctx.shared.default_space;
                    self.read_memory(ctx, space, addr, size).ok().map(|bytes| {
                        let mut buf = [0u8; 8];
                        let n = bytes.len().min(8);
                        buf[..n].copy_from_slice(&bytes[..n]);
                        u64::from_le_bytes(buf)
                    })
                }
                None => None,
            };
            if let Some(value) = value {
                self.block_param_values
                    .insert(param_id, SizedValue::new(value, size));
            }
        }
    }

    /// Bind a functionalized (`pure_reg`) callee's entry params positionally from
    /// the call's arguments, evaluated in the *caller's* frame.
    ///
    /// `argpromote_registers` makes such a callee a pure value function whose
    /// inputs flow through `Call.args` (not ambient register state), so the args
    /// are the source of truth — this replaces the register-file seeding
    /// [`seed_entry_params`](Self::seed_entry_params) does for conventional
    /// callees. The arg/param alignment invariant (`arg[i] ↔ param[i]`, built in
    /// lockstep by argpromote and preserved by mem2reg + `remove_entry_param`)
    /// makes the positional binding sound. Every arg is read before any param is
    /// written, so a self-recursive call still sees the caller's values.
    fn bind_entry_params_from_args(
        &mut self,
        ctx: &Context<'_>,
        call_id: InstructionId,
        target: FunctionId,
    ) {
        let args = match ctx.get_insn(call_id).mnemonic() {
            Mnemonic::Call(call) => call.args.clone(),
            _ => return,
        };
        let Some(root) = FunctionBody::from_id(ctx, target).root() else {
            return;
        };
        let params: Vec<(BlockParamId, usize)> = BasicBlock::from_id(ctx, root.id)
            .params()
            .map(|p| (p.id, p.size()))
            .collect();
        if args.len() != params.len() {
            // The alignment invariant is violated; fall back to register seeding
            // rather than mis-bind by position.
            self.seed_entry_params(ctx, target);
            return;
        }
        // Evaluate every arg in the caller frame first (recursion-safe), then bind.
        let values: Vec<SizedValue> = args
            .iter()
            .zip(&params)
            .map(|(&arg, &(_, size))| {
                let raw = self.get_value(ctx, arg.qualify(call_id.func)).unwrap_or(0);
                SizedValue::new(raw, size)
            })
            .collect();
        for ((param_id, _), value) in params.into_iter().zip(values) {
            self.block_param_values.insert(param_id, value);
        }
    }

    pub fn run_function(&mut self, ctx: &Context<'_>, func: FunctionId) -> crate::Result<()> {
        let root = FunctionBody::from_id(ctx, func)
            .root()
            .ok_or_else(|| self.make_error(ctx, EmulatorErrorKind::EmptyFunctionRoot(func)))?
            .id;
        self.block = root;
        self.idx = 0;
        self.call_stack.push(func);
        self.seed_entry_params(ctx, func);

        let mut call_depth: i32 = 0;

        let result = loop {
            let insn_ids = BasicBlock::from_id(ctx, self.block)
                .instruction_ids()
                .to_vec();
            let insn = InstructionRef::from_id(ctx, insn_ids[self.idx]);

            if matches!(insn.mnemonic(), Mnemonic::Return(_)) && call_depth == 0 {
                break Ok(());
            }

            match self.step_with_event(ctx)? {
                StepEvent::DirectCallEntered(target) => {
                    call_depth += 1;
                    self.call_stack.push(target);
                    // Dual binding convention (argpromote v2): a `regpure`-tagged
                    // call site passes its inputs explicitly through `Call.args`
                    // (bound positionally); an `Opaque` (implicit) call — and a
                    // conventional callee — reads them from the register file the
                    // calling convention set up. The legacy `pure_reg` flag is
                    // still honored during the migration.
                    let regpure_site = self
                        .call_site_stack
                        .last()
                        .copied()
                        .is_some_and(|call_id| call_is_regpure(ctx, call_id));
                    if regpure_site || FunctionBody::from_id(ctx, target).is_reg_materialized() {
                        if let Some(&call_id) = self.call_site_stack.last() {
                            self.bind_entry_params_from_args(ctx, call_id, target);
                        }
                    } else {
                        self.seed_entry_params(ctx, target);
                    }
                }
                StepEvent::IndirectCallEntered => {
                    call_depth += 1;
                    // Infer the callee from the block we landed in.
                    if let Some(parent) = BasicBlock::from_id(ctx, self.block).parent() {
                        let callee = parent.id;
                        self.call_stack.push(callee);
                        self.seed_entry_params(ctx, callee);
                    }
                }
                StepEvent::Return | StepEvent::ReturnValue => {
                    self.call_stack.pop();
                    call_depth -= 1;
                }
                StepEvent::Normal | StepEvent::InterceptedCall => {}
            }
        };

        self.call_stack.pop(); // pop the outermost function
        result
    }

    /// Emulate a **pure** function in isolation: bind its root params positionally
    /// from `args` (so symbolic caller inputs can be passed an arbitrary poison
    /// value) and run to the first top-level `Return` *without executing it*,
    /// leaving the body's computed values readable via [`get_value`](Self::get_value)
    /// at the block returned by [`current_block`](Self::current_block).
    ///
    /// `args` must align with the root params index-for-index (the `pure_reg`
    /// call interface). The run is bounded by `max_steps`; exceeding it yields
    /// [`EmulatorErrorKind::StepBudgetExceeded`]. Intended for v1 **leaf** pure
    /// functions (no nested calls), so call bookkeeping is intentionally minimal.
    pub fn run_pure(
        &mut self,
        ctx: &Context<'_>,
        func: FunctionId,
        args: &[SizedValue],
        max_steps: usize,
    ) -> crate::Result<()> {
        let opts: Vec<Option<SizedValue>> = args.iter().map(|&v| Some(v)).collect();
        self.run_pure_partial(ctx, func, &opts, max_steps)
    }

    /// Like [`run_pure`](Self::run_pure), but each positional argument may be
    /// [`None`] to bind that root param to **poison** (a symbolic value with
    /// undefined bits). Reading a poison param during emulation is a hard error
    /// (`PoisonRead`), so a consumer such as pure-call folding bails when the
    /// result actually depends on a symbolic argument, rather than computing on a
    /// bogus concrete value (argpromote v2, `ARGPROMOTE_REGISTERS_V2.md`).
    pub fn run_pure_partial(
        &mut self,
        ctx: &Context<'_>,
        func: FunctionId,
        args: &[Option<SizedValue>],
        max_steps: usize,
    ) -> crate::Result<()> {
        let root = FunctionBody::from_id(ctx, func)
            .root()
            .ok_or_else(|| self.make_error(ctx, EmulatorErrorKind::EmptyFunctionRoot(func)))?
            .id;
        self.block = root;
        self.idx = 0;
        self.call_stack.push(func);

        // Bind root params positionally from `args`: a concrete `Some(v)` seeds
        // the param value, a `None` marks it poison (read ⇒ hard error).
        let param_ids: Vec<BlockParamId> = BasicBlock::from_id(ctx, root)
            .params()
            .map(|p| p.id)
            .collect();
        for (param_id, arg) in param_ids.into_iter().zip(args) {
            match arg {
                Some(value) => {
                    self.block_param_values.insert(param_id, *value);
                }
                None => {
                    self.poison_params.insert(param_id);
                }
            }
        }

        self.drive_to_return(ctx, root, func, max_steps)
    }

    /// Like [`run_pure`](Self::run_pure), but for a `map` body — its leading
    /// element param may be an **aggregate** (the `enumerate` `(index, elem)`
    /// lane), seeded so the body's `Extract`s on it resolve. `args` align with
    /// the root params index-for-index: a [`BodyArg::Scalar`] seeds a scalar
    /// param, a [`BodyArg::Aggregate`] seeds an `Extract`-able tuple param.
    /// Whether `id` is an array/list-typed value (routed through
    /// [`array_values`](Self::array_values) rather than scalar `insn_values`).
    /// Refreshes [`sequence_types`](Self::sequence_types) when the module has
    /// gained types since it was last answered. The probe is lock-free; only a
    /// genuine change pays for the locked question behind it.
    fn refresh_sequence_types(&mut self, ctx: &Context<'_>) {
        let published = ctx.shared.types.published_len();
        if self.sequence_types_checked_at == Some(published) {
            return;
        }
        self.sequence_types = ctx.shared.types.has_sequence_types();
        self.sequence_types_checked_at = Some(published);
    }

    fn is_array_operand(&self, ctx: &Context<'_>, id: ValueId) -> bool {
        // Nothing in this module is sequence-typed, so no operand can be.
        if !self.sequence_types {
            return false;
        }
        match ctx.stored_type_of(id) {
            Some(ty) => {
                ctx.shared.types.array_of(ty).is_some() || ctx.shared.types.list_of(ty).is_some()
            }
            None => false,
        }
    }

    /// Resolve an array-typed operand to its little-endian byte buffer: a `Bytes`
    /// blob's data, a previously-computed `array_values` entry, or a short array
    /// materialized as a scalar literal/result.
    fn resolve_array(&mut self, ctx: &Context<'_>, id: ValueId) -> Option<Vec<u8>> {
        match id {
            ValueId::Bytes(b) => Some(ctx.shared.values.bytes[b].data.clone()),
            ValueId::Instruction(i) => self
                .array_values
                .get(&i)
                .cloned()
                .or_else(|| self.get_value_bytes(ctx, id)),
            ValueId::Literal(_) => self.get_value_bytes(ctx, id),
            // A root/block param bound by `run_pure` (e.g. an argpromote-minted
            // `[i8;N]` array param): its little-endian bytes come from the bound
            // `SizedValue`. Defensive width check — a short buffer must fail
            // resolution rather than silently produce clamped `Range` slices for
            // callers that lack the projection guarantee.
            ValueId::BlockParam(_) => {
                let bytes = self.get_value_bytes(ctx, id)?;
                let ty_size = ctx
                    .stored_type_of(id)
                    .map(|ty| ctx.shared.types.size_of(ty))?;
                (bytes.len() == ty_size).then_some(bytes)
            }
            _ => None,
        }
    }

    /// Evaluate one array-valued (or lane-reading) sequence intrinsic, depositing
    /// its result in `array_values` (arrays) or `insn_values` (`at`).
    fn eval_array_intrinsic(
        &mut self,
        ctx: &Context<'_>,
        insn_id: InstructionId,
        name: &str,
        args: &[ValueId],
    ) -> Result<(), EmulatorErrorKind> {
        match name {
            "iota" => {
                let n = self
                    .get_value(ctx, args[0])
                    .ok_or(EmulatorErrorKind::ValueError(0))?;
                let mut buf = Vec::with_capacity(n as usize * 8);
                for i in 0..n {
                    buf.extend_from_slice(&i.to_le_bytes());
                }
                self.array_values.insert(insn_id, buf);
            }
            "singleton" => {
                let buf = self
                    .get_value_bytes(ctx, args[0])
                    .ok_or(EmulatorErrorKind::ValueError(0))?;
                self.array_values.insert(insn_id, buf);
            }
            "concat" => {
                let mut a = self
                    .resolve_array(ctx, args[0])
                    .ok_or(EmulatorErrorKind::ValueError(0))?;
                let b = self
                    .resolve_array(ctx, args[1])
                    .ok_or(EmulatorErrorKind::ValueError(0))?;
                a.extend_from_slice(&b);
                self.array_values.insert(insn_id, a);
            }
            "insert" => {
                let mut buf = self
                    .resolve_array(ctx, args[0])
                    .ok_or(EmulatorErrorKind::ValueError(0))?;
                let i = self
                    .get_value(ctx, args[1])
                    .ok_or(EmulatorErrorKind::ValueError(0))? as usize;
                let vbytes = self
                    .get_value_bytes(ctx, args[2])
                    .ok_or(EmulatorErrorKind::ValueError(0))?;
                let esz = vbytes.len();
                let off = i * esz;
                if off + esz <= buf.len() {
                    buf[off..off + esz].copy_from_slice(&vbytes);
                }
                self.array_values.insert(insn_id, buf);
            }
            "enumerate" => {
                // `enumerate(arr) = [(index: i64, elem: T); N]`, materialized only
                // over a fixed array (or bounded list). A length-erased unbounded
                // list has no concrete count, so bail recoverably rather than
                // fabricate one — matching `enumerate`'s deferred `eval`.
                let src_ty = ctx
                    .stored_type_of(args[0])
                    .ok_or(EmulatorErrorKind::ValueError(0))?;
                if matches!(ctx.shared.types.list_of(src_ty), Some((_, None))) {
                    return Err(EmulatorErrorKind::UnsupportedIntrinsic(Box::from(
                        "enumerate",
                    )));
                }
                let in_elem = ctx
                    .shared
                    .types
                    .seq_elem_of(src_ty)
                    .ok_or(EmulatorErrorKind::ValueError(0))?;
                let isz = ctx.shared.types.size_of(in_elem).max(1);
                let buf = self
                    .resolve_array(ctx, args[0])
                    .ok_or(EmulatorErrorKind::ValueError(0))?;
                // The `(index, elem)` result tuple is a *structural* aggregate:
                // its fields are addressed by index, not byte offset, so lay them
                // out sequentially by field size (field 0 = i64 index, field 1 =
                // elem). This is the same layout `eval_scan` splits back out.
                let tuple_ty = ctx
                    .stored_type_of(ValueId::Instruction(insn_id))
                    .and_then(|ty| ctx.shared.types.seq_elem_of(ty))
                    .ok_or(EmulatorErrorKind::ValueError(0))?;
                let (idx_sz, elem_off) = {
                    let fields = ctx
                        .shared
                        .types
                        .aggregate_fields(tuple_ty)
                        .ok_or(EmulatorErrorKind::ValueError(0))?;
                    let [idx_f, _elem_f] = fields else {
                        return Err(EmulatorErrorKind::ValueError(0));
                    };
                    let idx_sz = ctx.shared.types.size_of(idx_f.type_id).min(8);
                    (idx_sz, idx_sz)
                };
                let tsz = idx_sz + isz;
                let count = buf.len() / isz;
                let mut out = vec![0u8; count * tsz];
                for i in 0..count {
                    let base = i * tsz;
                    let idx_bytes = (i as u64).to_le_bytes();
                    out[base..base + idx_sz].copy_from_slice(&idx_bytes[..idx_sz]);
                    out[base + elem_off..base + elem_off + isz]
                        .copy_from_slice(&buf[i * isz..i * isz + isz]);
                }
                self.array_values.insert(insn_id, out);
            }
            "at" => {
                let buf = self
                    .resolve_array(ctx, args[0])
                    .ok_or(EmulatorErrorKind::ValueError(0))?;
                let i = self
                    .get_value(ctx, args[1])
                    .ok_or(EmulatorErrorKind::ValueError(0))? as usize;
                let esz = ctx
                    .stored_type_of(ValueId::Instruction(insn_id))
                    .map(|ty| ctx.shared.types.size_of(ty))
                    .unwrap_or(8);
                let off = i * esz;
                let lane = buf
                    .get(off..off + esz)
                    .ok_or(EmulatorErrorKind::ValueError(0))?;
                self.insn_values
                    .insert(insn_id, SizedValue::from_bits(le_bits(lane), esz));
            }
            other => panic!("eval_array_intrinsic called on non-array intrinsic `{other}`"),
        }
        Ok(())
    }

    /// Thread a scan's accumulator across every lane of its source array, running
    /// the pure binary body `(acc, elem) -> acc'` once per element in a fresh
    /// nested emulator, and store the concatenated per-step accumulators as the
    /// result array buffer.
    fn eval_scan(
        &mut self,
        ctx: &Context<'_>,
        insn_id: InstructionId,
        scan: &Scan,
    ) -> Result<(), EmulatorErrorKind> {
        const SCAN_STEP_BUDGET: usize = 100_000;

        let src = self
            .resolve_array(ctx, scan.src.qualify(insn_id.func))
            .ok_or(EmulatorErrorKind::ValueError(0))?;
        // Element sizes come from the operand/result element *types* (which are
        // known even for a length-erased `[T;*]` result); the lane count is the
        // source buffer's length in input elements. This handles both a folded
        // fixed-array source and a symbolic-length `iota`.
        let in_elem = ctx
            .stored_type_of(scan.src.qualify(insn_id.func))
            .and_then(|ty| ctx.shared.types.seq_elem_of(ty))
            .ok_or(EmulatorErrorKind::ValueError(0))?;
        let isz = ctx.shared.types.size_of(in_elem).max(1);
        let out_elem = ctx
            .stored_type_of(ValueId::Instruction(insn_id))
            .and_then(|ty| ctx.shared.types.seq_elem_of(ty))
            .ok_or(EmulatorErrorKind::ValueError(0))?;
        let osz = ctx.shared.types.size_of(out_elem);
        let count = src.len() / isz;
        let body = require_real_callee(scan.body)?;
        if count == 0 {
            self.array_values.insert(insn_id, Vec::new());
            return Ok(());
        }

        // Loop-invariant captures, resolved once as scalars.
        let capture_args: Vec<BodyArg> = scan
            .captures
            .iter()
            .map(|&c| {
                let v = self
                    .get_value(ctx, c.qualify(insn_id.func))
                    .ok_or(EmulatorErrorKind::ValueError(0))?;
                let sz = ctx
                    .stored_type_of(c.qualify(insn_id.func))
                    .map(|ty| ctx.shared.types.size_of(ty))
                    .unwrap_or(8);
                Ok(BodyArg::Scalar(SizedValue::new(v, sz)))
            })
            .collect::<Result<_, EmulatorErrorKind>>()?;

        let init = self
            .get_value(ctx, scan.init.qualify(insn_id.func))
            .ok_or(EmulatorErrorKind::ValueError(0))?;
        let mut acc = SizedValue::new(init, osz);

        let root = FunctionBody::from_id(ctx, body)
            .root()
            .ok_or(EmulatorErrorKind::EmptyFunctionRoot(body))?
            .id;

        // When the source element is a tuple (the `enumerate` `(index, elem)`
        // lane), each lane is passed as an aggregate so the body's `Extract`s
        // resolve; a plain scalar element is passed as-is. The tuple is a
        // structural aggregate (fields addressed by index), so its bytes are laid
        // out sequentially by field size — the same layout `enumerate` writes.
        let elem_fields: Option<Vec<(usize, usize)>> =
            ctx.shared.types.aggregate_fields(in_elem).map(|fs| {
                let mut off = 0;
                fs.iter()
                    .map(|f| {
                        let sz = ctx.shared.types.size_of(f.type_id);
                        let field = (off, sz);
                        off += sz;
                        field
                    })
                    .collect()
            });

        let mut out = Vec::with_capacity(count * osz);
        for k in 0..count {
            let elem = &src[k * isz..k * isz + isz];
            let elem_arg = match &elem_fields {
                Some(fields) => BodyArg::Aggregate(
                    fields
                        .iter()
                        .map(|&(off, sz)| SizedValue::from_bits(le_bits(&elem[off..off + sz]), sz))
                        .collect(),
                ),
                None => BodyArg::Scalar(SizedValue::from_bits(le_bits(elem), isz)),
            };
            let mut body_args = Vec::with_capacity(2 + capture_args.len());
            body_args.push(BodyArg::Scalar(acc));
            body_args.push(elem_arg);
            body_args.extend(capture_args.iter().cloned());

            let mut emu = StandaloneEmulator::new(root);
            emu.run_map_body(ctx, body, &body_args, SCAN_STEP_BUDGET)
                .map_err(|e| e.kind)?;
            let ret = body_return_value(ctx, emu.current_block())
                .ok_or(EmulatorErrorKind::ValueError(0))?;
            let mut lane = emu
                .get_value_bytes(ctx, ret)
                .ok_or(EmulatorErrorKind::ValueError(0))?;
            lane.resize(osz, 0);
            acc = SizedValue::from_bits(le_bits(&lane), osz);
            out.extend_from_slice(&lane);
        }
        self.array_values.insert(insn_id, out);
        Ok(())
    }

    /// Total map: run the (pure) unary body once per source element and
    /// materialize the results as an array buffer. Mirrors [`Self::eval_scan`]
    /// without the threaded accumulator.
    fn eval_map(
        &mut self,
        ctx: &Context<'_>,
        insn_id: InstructionId,
        map: &qcode::value::insn::Map,
    ) -> Result<(), EmulatorErrorKind> {
        const MAP_STEP_BUDGET: usize = 100_000;

        let src = self
            .resolve_array(ctx, map.src.qualify(insn_id.func))
            .ok_or(EmulatorErrorKind::ValueError(0))?;
        let in_elem = ctx
            .stored_type_of(map.src.qualify(insn_id.func))
            .and_then(|ty| ctx.shared.types.seq_elem_of(ty))
            .ok_or(EmulatorErrorKind::ValueError(0))?;
        let isz = ctx.shared.types.size_of(in_elem).max(1);
        let out_elem = ctx
            .stored_type_of(ValueId::Instruction(insn_id))
            .and_then(|ty| ctx.shared.types.seq_elem_of(ty))
            .ok_or(EmulatorErrorKind::ValueError(0))?;
        let osz = ctx.shared.types.size_of(out_elem);
        let count = src.len() / isz;
        let body = require_real_callee(map.body)?;

        let capture_args: Vec<BodyArg> = map
            .captures
            .iter()
            .map(|&c| {
                let v = self
                    .get_value(ctx, c.qualify(insn_id.func))
                    .ok_or(EmulatorErrorKind::ValueError(0))?;
                let sz = ctx
                    .stored_type_of(c.qualify(insn_id.func))
                    .map(|ty| ctx.shared.types.size_of(ty))
                    .unwrap_or(8);
                Ok(BodyArg::Scalar(SizedValue::new(v, sz)))
            })
            .collect::<Result<_, EmulatorErrorKind>>()?;

        // The element may be an `enumerate` tuple `(index, elem)`; pass it as an
        // aggregate so the body's `Extract`s resolve (same layout as `eval_scan`).
        let elem_fields: Option<Vec<(usize, usize)>> =
            ctx.shared.types.aggregate_fields(in_elem).map(|fs| {
                let mut off = 0;
                fs.iter()
                    .map(|f| {
                        let sz = ctx.shared.types.size_of(f.type_id);
                        let field = (off, sz);
                        off += sz;
                        field
                    })
                    .collect()
            });

        let mut out = Vec::with_capacity(count * osz);
        for k in 0..count {
            let elem = &src[k * isz..k * isz + isz];
            let elem_arg = match &elem_fields {
                Some(fields) => BodyArg::Aggregate(
                    fields
                        .iter()
                        .map(|&(off, sz)| SizedValue::from_bits(le_bits(&elem[off..off + sz]), sz))
                        .collect(),
                ),
                None => BodyArg::Scalar(SizedValue::from_bits(le_bits(elem), isz)),
            };
            let mut body_args = Vec::with_capacity(1 + capture_args.len());
            body_args.push(elem_arg);
            body_args.extend(capture_args.iter().cloned());

            let mut emu = StandaloneEmulator::new(
                FunctionBody::from_id(ctx, body)
                    .root()
                    .ok_or(EmulatorErrorKind::EmptyFunctionRoot(body))?
                    .id,
            );
            emu.run_map_body(ctx, body, &body_args, MAP_STEP_BUDGET)
                .map_err(|e| e.kind)?;
            let ret = body_return_value(ctx, emu.current_block())
                .ok_or(EmulatorErrorKind::ValueError(0))?;
            let mut lane = emu
                .get_value_bytes(ctx, ret)
                .ok_or(EmulatorErrorKind::ValueError(0))?;
            lane.resize(osz, 0);
            out.extend_from_slice(&lane);
        }
        self.array_values.insert(insn_id, out);
        Ok(())
    }

    pub fn run_map_body(
        &mut self,
        ctx: &Context<'_>,
        func: FunctionId,
        args: &[BodyArg],
        max_steps: usize,
    ) -> crate::Result<()> {
        let root = FunctionBody::from_id(ctx, func)
            .root()
            .ok_or_else(|| self.make_error(ctx, EmulatorErrorKind::EmptyFunctionRoot(func)))?
            .id;
        self.block = root;
        self.idx = 0;
        self.call_stack.push(func);

        let param_ids: Vec<BlockParamId> = BasicBlock::from_id(ctx, root)
            .params()
            .map(|p| p.id)
            .collect();
        for (param_id, arg) in param_ids.into_iter().zip(args) {
            match arg {
                BodyArg::Scalar(v) => {
                    self.block_param_values.insert(param_id, *v);
                }
                BodyArg::Aggregate(fields) => {
                    self.block_param_aggregates.insert(param_id, fields.clone());
                }
            }
        }

        self.drive_to_return(ctx, root, func, max_steps)
    }

    /// Shared drive loop for the bounded `run_pure`/`run_map_body` entry points:
    /// run from the current position to the first top-level value or machine
    /// `Return` (without
    /// executing it), popping the call frame. Params must already be seeded and
    /// `func` pushed onto the call stack.
    fn drive_to_return(
        &mut self,
        ctx: &Context<'_>,
        _root: BlockId,
        _func: FunctionId,
        max_steps: usize,
    ) -> crate::Result<()> {
        let mut steps = 0usize;
        let result = loop {
            let insn_ids = BasicBlock::from_id(ctx, self.block)
                .instruction_ids()
                .to_vec();
            // A well-formed block ends in a terminator, so `self.idx` should always
            // point at a real instruction. Lifting can leave degenerate empty blocks
            // behind, though; bail with a recoverable error instead of indexing out
            // of bounds (which would crash the whole analysis via GVN pure-call
            // folding). `make_error` can't be used here — it also indexes the block.
            if self.idx >= insn_ids.len() {
                break Err(self.make_empty_block_error(ctx));
            }
            let insn = InstructionRef::from_id(ctx, insn_ids[self.idx]);
            if matches!(
                insn.mnemonic(),
                Mnemonic::Return(_) | Mnemonic::ReturnValue(_)
            ) {
                break Ok(());
            }
            steps += 1;
            if steps > max_steps {
                break Err(self.make_error(ctx, EmulatorErrorKind::StepBudgetExceeded(max_steps)));
            }
            if let Err(e) = self.step(ctx) {
                break Err(e);
            }
        };

        self.call_stack.pop();
        result
    }
}

fn lambda_return_value(ctx: &Context<'_>, block: BlockId) -> Option<ValueId> {
    let last = BasicBlock::from_id(ctx, block).iter().last()?;
    match last.mnemonic() {
        Mnemonic::ReturnValue(ret) => Some(ret.value.qualify(last.id.func)),
        _ => None,
    }
}

/// The value a scan/map body block returns — via either a machine `Return` (the
/// outlined-body form) or a lambda `ReturnValue`. `None` if the block does not
/// end in a value-carrying return.
fn body_return_value(ctx: &Context<'_>, block: BlockId) -> Option<ValueId> {
    let last = BasicBlock::from_id(ctx, block).iter().last()?;
    match last.mnemonic() {
        Mnemonic::Return(ret) => ret.value.map(|v| v.qualify(last.id.func)),
        Mnemonic::ReturnValue(ret) => Some(ret.value.qualify(last.id.func)),
        _ => None,
    }
}

/// The array-valued (or lane-reading) sequence intrinsics the emulator evaluates
/// over its [`array_values`](StandaloneEmulator::array_values) domain rather than
/// the scalar interpreter.
fn is_array_intrinsic(name: &str) -> bool {
    matches!(
        name,
        "iota" | "singleton" | "concat" | "insert" | "at" | "enumerate"
    )
}

/// Fold a little-endian byte slice (≤ 16 bytes) into a `u128`.
fn le_bits(bytes: &[u8]) -> u128 {
    let mut buf = [0u8; 16];
    let n = bytes.len().min(16);
    buf[..n].copy_from_slice(&bytes[..n]);
    u128::from_le_bytes(buf)
}

/// A positional argument to a `map` body for [`run_map_body`](StandaloneEmulator::run_map_body):
/// a scalar param value, or the field vector of an aggregate (tuple) param.
#[derive(Debug, Clone)]
pub enum BodyArg {
    Scalar(SizedValue),
    Aggregate(Vec<SizedValue>),
}

/// Private helper that pairs `&mut StandaloneEmulator` fields with `&Context<'_>`
/// so the default `Interpreter::interpret()` impl can be reused.
struct TempInterpreter<'a, 'ctx, M> {
    memory: &'a mut M,
    literals: &'a mut LiteralCache,
    insn_values: &'a mut InsnValues,
    block_param_values: &'a mut FxHashMap<BlockParamId, SizedValue>,
    poison_params: &'a FxHashSet<BlockParamId>,
    ctx: &'ctx Context<'ctx>,
}

impl<'ctx, M: EmulatorMemory> Interpreter for TempInterpreter<'_, 'ctx, M> {
    type V = SizedValue;
    type M = M;

    fn memory(&mut self) -> &mut Self::M {
        self.memory
    }

    fn ctx(&self) -> &Context<'_> {
        self.ctx
    }

    fn get_value(&mut self, id: ValueId) -> Result<Self::V, EmulatorErrorKind> {
        // Taken before `ValueRef::new`, which would resolve the literal through
        // the interner's lock.
        if let ValueId::Literal(literal) = id {
            return Ok(self.literals.get(self.ctx, literal));
        }
        match ValueRef::new(id, self.ctx) {
            ValueRef::Literal(literal) => Ok(SizedValue::new(literal.value(), literal.size())),
            // Byte blobs are wider than the emulator's scalar SizedValue.
            ValueRef::Bytes(_) => Err(EmulatorErrorKind::ValueError(0)),
            ValueRef::Instruction(insn) => self
                .insn_values
                .get(&insn.id)
                .copied()
                .ok_or(EmulatorErrorKind::ValueError(0)),
            ValueRef::Varnode(varnode) => Ok(SizedValue::new(varnode.address() as u64, 8)),
            ValueRef::Temp(temp) => Ok(SizedValue::new(temp.address() as u64, 8)),
            ValueRef::BasicBlock(_) => panic!("Cannot get value of a block"),
            ValueRef::BlockParam(param) => {
                if self.poison_params.contains(&param.id) {
                    return Err(EmulatorErrorKind::PoisonRead);
                }
                self.block_param_values
                    .get(&param.id)
                    .copied()
                    .ok_or(EmulatorErrorKind::ValueError(0))
            }
            ValueRef::Function(f) => f
                .address()
                .map(SizedValue::from_u64)
                .ok_or(EmulatorErrorKind::EmptyFunctionRoot(f.id)),
            // Poison has undefined bits: demanding its concrete value is a hard
            // error (propagating it as an unread operand never reaches here).
            ValueRef::Poison(_) => Err(EmulatorErrorKind::PoisonRead),
        }
    }
}

pub struct Emulator<'ctx, M = EmulatedMemory> {
    inner: StandaloneEmulator<M>,
    ctx: &'ctx Context<'ctx>,
}

impl<'ctx> Emulator<'ctx, EmulatedMemory> {
    /// Builds an emulator over the default flat memory.
    pub fn new(ctx: &'ctx Context<'ctx>, entry: BlockId) -> Self {
        Self::new_in(ctx, entry)
    }

    pub fn from_function(ctx: &'ctx Context<'ctx>, func: FunctionId) -> Self {
        Self::from_function_in(ctx, func)
    }

    pub fn from_block(ctx: &'ctx Context<'ctx>, block: BlockId) -> Self {
        Self::new_in(ctx, block)
    }

    pub fn from_address(ctx: &'ctx Context<'ctx>, addr: u64) -> Self {
        Self::from_address_in(ctx, addr)
    }
}

impl<'ctx, M: EmulatorMemory + Default> Emulator<'ctx, M> {
    /// Builds an emulator over an explicit memory backend.
    pub fn new_in(ctx: &'ctx Context<'ctx>, entry: BlockId) -> Self {
        let mut inner = StandaloneEmulator::<M>::with_address_index(entry, AddressIndex::analyze(ctx));
        inner.memory.configure_spaces(ctx);
        Self { inner, ctx }
    }

    pub fn set_instruction_hook(
        &mut self,
        hook: impl Fn(&InstructionRef<'_, '_>, &StandaloneEmulator<M>) + Send + Sync + 'static,
    ) {
        self.inner.instruction_hook = Some(Box::new(hook));
    }

    pub fn set_call_interceptor(
        &mut self,
        interceptor: impl FnMut(
            &Context<'_>,
            &mut StandaloneEmulator<M>,
            &CallSite,
        ) -> Result<CallInterception, Box<str>>
        + Send
        + Sync
        + 'static,
    ) {
        self.inner.set_call_interceptor(interceptor);
    }

    pub fn clear_call_interceptor(&mut self) {
        self.inner.clear_call_interceptor();
    }

    /// Builds an emulator at a function's root block, over an explicit backend.
    pub fn from_function_in(ctx: &'ctx Context<'ctx>, func: FunctionId) -> Self {
        let entry = FunctionBody::from_id(ctx, func)
            .root()
            .expect("Cannot create emulator for function with empty root block")
            .id;
        Self::new_in(ctx, entry)
    }

    /// Builds an emulator positioned at `addr`, over an explicit backend.
    pub fn from_address_in(ctx: &'ctx Context<'ctx>, addr: u64) -> Self {
        Self {
            inner: StandaloneEmulator::<M>::from_address_in(ctx, addr),
            ctx,
        }
    }

    /// Debugging method to view a value at a given address
    pub fn inspect_memory(&mut self, space: SpaceId, addr: u64, size: usize) -> Option<Vec<u8>> {
        self.inner
            .memory
            .read_bytes(MemorySpaceId::Shared(space), addr, size)
            .ok()
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
        self.inner.write_memory(self.ctx, space, addr, value)
    }

    pub fn read_memory(
        &mut self,
        space: SpaceId,
        addr: u64,
        size: usize,
    ) -> Result<Vec<u8>, EmulatorErrorKind> {
        self.inner.read_memory(self.ctx, space, addr, size)
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
        let _ = self
            .inner
            .memory
            .write_bytes(space_id.into(), base, &value.to_le_bytes());
    }

    /// Reads a single 64-bit lane of a wide register (little-endian).
    /// Lane `n` covers bytes `[n*8 .. n*8+8]` relative to the register's base address.
    pub fn read_register_lane(&mut self, id: RegisterId, lane: usize) -> u64 {
        let (space_id, base_addr) = {
            let vn = self.ctx.get_register(id);
            (vn.space().id, vn.address() as u64)
        };
        let base = base_addr + (lane as u64) * 8;
        // An unwritten lane reads as zero: register space is architectural
        // state that exists whether or not a harness has seeded it.
        let bytes = self
            .inner
            .memory
            .read_bytes(space_id.into(), base, 8)
            .unwrap_or_else(|_| vec![0; 8]);
        u64::from_le_bytes(bytes.try_into().expect("read_bytes returns 8 bytes"))
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
            Some(InstructionRef::from_id(self.ctx, id))
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

impl<'ctx, M: EmulatorMemory> Interpreter for Emulator<'ctx, M> {
    type V = SizedValue;
    type M = M;

    fn memory(&mut self) -> &mut Self::M {
        &mut self.inner.memory
    }

    fn ctx(&self) -> &Context<'ctx> {
        self.ctx
    }

    fn get_value(&mut self, id: ValueId) -> Result<Self::V, EmulatorErrorKind> {
        if let ValueId::Literal(literal) = id {
            let value = self.inner.literal_cache.get(self.ctx, literal);
            return Ok(value);
        }
        match ValueRef::new(id, self.ctx) {
            ValueRef::Literal(literal) => Ok(SizedValue::new(literal.value(), literal.size())),
            // Byte blobs are wider than the emulator's scalar SizedValue.
            ValueRef::Bytes(_) => Err(EmulatorErrorKind::ValueError(0)),
            ValueRef::Instruction(insn) => self
                .inner
                .insn_values
                .get(&insn.id)
                .copied()
                .ok_or(EmulatorErrorKind::ValueError(0)),
            ValueRef::Varnode(varnode) => Ok(SizedValue::new(varnode.address() as u64, 8)),
            ValueRef::Temp(temp) => Ok(SizedValue::new(temp.address() as u64, 8)),
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
            // Poison has undefined bits: demanding its concrete value is a hard
            // error (propagating it as an unread operand never reaches here).
            ValueRef::Poison(_) => Err(EmulatorErrorKind::PoisonRead),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::context::Context;
    use qcode::space::{Space, SpaceType};
    use qcode::value::QCodeMut;
    use qcode::value::TempSpace;
    use std::sync::{Arc, Mutex};
    use wazabin_qcode_macro::qcode;

    /// Reading a poison value is a hard error (`PoisonRead`); propagating it as
    /// an unread operand never reaches `get_value`.
    #[test]
    fn reading_poison_is_a_hard_error() {
        let mut ctx = Context::new();
        let func = ctx.anon_function();
        let block = BasicBlock::make(&mut ctx, func).with_address(0x1000).id;
        let i32_ty = ctx.shared.types.get_or_make_int(4);
        let poison = ctx.get_poison(i32_ty);
        let mut emu = StandaloneEmulator::new(block);
        let mut tmp = TempInterpreter {
            memory: &mut emu.memory,
            literals: &mut emu.literal_cache,
            insn_values: &mut emu.insn_values,
            block_param_values: &mut emu.block_param_values,
            poison_params: &emu.poison_params,
            ctx: &ctx,
        };
        assert!(matches!(
            tmp.get_value(poison),
            Err(EmulatorErrorKind::PoisonRead)
        ));
    }

    #[test]
    fn minted_callee_is_not_executable() {
        assert!(matches!(
            require_real_callee(Callee::Minted(7)),
            Err(EmulatorErrorKind::UnresolvedMintedCallee(7))
        ));
    }

    #[test]
    fn sized_value_masks_to_declared_width() {
        let value = SizedValue::new(0x1234, 1);
        assert_eq!(value.size().unwrap(), 1);
        assert_eq!(value.value().unwrap(), 0x34);
    }

    #[test]
    fn from_address_resolves_function_entry_to_root() {
        let mut ctx = Context::new();
        let function = FunctionBody::make_at_addr(&mut ctx, 0x1000, None).id;
        let root = BasicBlock::make(&mut ctx, function).with_address(0x1000).id;

        let emulator = StandaloneEmulator::from_address(&ctx, 0x1000);

        assert_eq!(emulator.current_block(), root);
        assert!(emulator.address_index.is_some());
    }

    #[test]
    fn standalone_address_lookup_builds_one_lazy_snapshot() {
        let mut ctx = Context::new();
        let function = ctx.anon_function();
        let block = BasicBlock::make(&mut ctx, function).with_address(0x2000).id;
        ctx.block_mut(block).extra_addresses.push(0x2001);
        let mut emulator = StandaloneEmulator::new(block);

        assert!(emulator.address_index.is_none());
        assert_eq!(emulator.block_at(&ctx, 0x2001), Some(block));
        assert!(emulator.address_index.is_some());
        assert_eq!(emulator.block_at(&ctx, 0x2000), Some(block));
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
    fn apply_evaluates_recursive_lambda_value_return() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda dec:
            <entry @n:i64>
                %is_zero = @n == 0;
                if %is_zero goto <done @r=@n> else goto <step @m=@n>;

            <step @m:i64>
                %next = @m - 1;
                %out = apply dec(%next);
                return %out;

            <done @r:i64>
                return @r;
            "
        );

        let dec = qcode::value::FunctionBody::from_name(&ctx, "dec")
            .expect("lambda exists")
            .id;
        let root = qcode::value::FunctionBody::from_id(&ctx, dec)
            .root()
            .expect("lambda has root")
            .id;
        let mut emu = StandaloneEmulator::new(root);
        emu.run_pure(&ctx, dec, &[SizedValue::new(3, 8)], 1000)
            .expect("recursive lambda evaluates");
        let ret = lambda_return_value(&ctx, emu.current_block()).expect("lambda returned a value");
        assert_eq!(emu.get_value(&ctx, ret), Some(0));
    }

    /// Test setup is the publication barrier: `result_type` only reads, so the
    /// types an intrinsic resolves to must exist before it is pushed.
    fn publish_iota_result(ctx: &mut Context) {
        use qcode::types::TypeRequest;
        let i64_ty = ctx.shared.types.get_or_make_int(8);
        ctx.shared
            .types
            .create_requested_types(&[TypeRequest::list(i64_ty, None)]);
    }

    /// `map @f arr` runs the pure unary body over every element and materializes
    /// the result buffer. `f(x) = x * 3`, `arr = [1, 2, 3, 4]` ⇒ `[3, 6, 9, 12]`.
    #[test]
    fn map_over_array_is_emulated() {
        let mut ctx = Context::new();
        publish_iota_result(&mut ctx);
        qcode!(
            ctx,
            "
            lambda triple:
            <tb @x:i64>
                %r = @x * 3;
                return %r;
            fn main:
            <me>
                %src = $iota(i64 0x4);
                %m = triple <$> %src;
                goto <0x1001>;
            "
        );

        let mut emu = StandaloneEmulator::new(me);
        emu.step(&ctx).expect("iota");
        emu.step(&ctx).expect("map");

        let buf = emu.array_values.get(&m).expect("map produced an array");
        let words: Vec<u64> = buf
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // iota(4) = [0,1,2,3]; triple ⇒ [0, 3, 6, 9].
        assert_eq!(words, vec![0, 3, 6, 9]);
    }

    /// End-to-end array emulation: `scanl @step init (iota n)` threads the
    /// accumulator through the driver array and materializes the result buffer.
    /// `step(acc, x) = acc + x`, `init = 10`, `iota(3) = [0, 1, 2]` ⇒
    /// `[10, 11, 13]` (out[i] = acc after adding x_i, prefix-fold style).
    #[test]
    fn scan_over_iota_is_emulated() {
        let mut ctx = Context::new();
        publish_iota_result(&mut ctx);
        qcode!(
            ctx,
            "
            lambda step:
            <sb @acc:i64 @x:i64>
                %r = @acc + @x;
                return %r;
            fn main:
            <me>
                %src = $iota(i64 0x3);
                %s = scanl @step i64 0xa %src;
                goto <0x1001>;
            "
        );

        let mut emu = StandaloneEmulator::new(me);
        // Step: iota, then scan (do not execute the terminating goto).
        emu.step(&ctx).expect("iota");
        emu.step(&ctx).expect("scan");

        let buf = emu.array_values.get(&s).expect("scan produced an array");
        let words: Vec<u64> = buf
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(words, vec![10, 11, 13]);
    }

    #[test]
    fn enumerate_over_array_is_emulated() {
        use qcode::value::{FunctionBody, ValueId, insn::IntrinsicId};

        let mut ctx = Context::new();
        let f = FunctionBody::make(&mut ctx, "f".into()).unwrap().id;
        let entry = ctx.get_or_make_block(0x1000, f);
        {
            let mut fm = FunctionBody::from_id_mut(&mut ctx, f);
            fm.set_root(entry).unwrap();
            fm.add_block(entry);
        }
        // A fixed `[i64; 4]` source `[10, 20, 30, 40]`.
        let i64_ty = ctx.shared.types.get_or_make_int(8);
        let arr_ty = ctx.shared.types.get_or_make_array(i64_ty, 4);
        let data: Vec<u8> = [10u64, 20, 30, 40]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let src = ctx.get_bytes(data).id();
        if let ValueId::Bytes(bid) = src {
            ctx.shared.values.bytes[bid].type_id = arr_ty;
        }
        // Publish the `(index, elem)` tuple and its array before pushing the
        // intrinsic: `result_type` only reads.
        {
            use qcode::types::{AggregateField, TypeRequest};
            let fields = vec![
                AggregateField::new("index", i64_ty),
                AggregateField::new("elem", i64_ty),
            ];
            let tuple = ctx
                .shared
                .types
                .create_requested_types(&[TypeRequest::aggregate(fields)])[0];
            ctx.shared
                .types
                .create_requested_types(&[TypeRequest::array(tuple, 4)]);
        }
        let enum_id = IntrinsicId::from_name("enumerate").unwrap();
        let e = {
            let mut b = ctx.builder(entry);
            let e = b.push_intrinsic(enum_id, vec![src]).id();
            let ptr = b.shr().get_const(0, 8);
            b.push_return(ptr);
            e
        };
        let ValueId::Instruction(eid) = e else {
            unreachable!()
        };

        let mut emu = StandaloneEmulator::new(entry);
        emu.step(&ctx).expect("enumerate");

        // `enumerate([10,20,30,40]) = [(0,10),(1,20),(2,30),(3,40)]`: each lane is
        // an `(index: i64, elem: i64)` tuple.
        let buf = emu
            .array_values
            .get(&eid)
            .expect("enumerate produced an array");
        let words: Vec<u64> = buf
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(words, vec![0, 10, 1, 20, 2, 30, 3, 40]);
    }

    /// Emulating a function that returns `enumerate` over an *unbounded* list
    /// bails with a recoverable `UnsupportedIntrinsic` rather than fabricating a
    /// length — a length-erased list has no concrete count to materialize. (A
    /// fixed array *is* materialized; see the `mt_scan` differential test.)
    #[test]
    fn enumerate_of_unbounded_list_bails_recoverably() {
        use qcode::value::{
            BasicBlock, FunctionBody, InstructionRef, ValueId,
            insn::{IntrinsicApp, IntrinsicId, Return},
        };

        let mut ctx = Context::new();
        let f = FunctionBody::make(&mut ctx, "f".into()).unwrap().id;
        let entry = ctx.get_or_make_block(0x1000, f);
        {
            let mut fm = FunctionBody::from_id_mut(&mut ctx, f);
            fm.set_root(entry).unwrap();
            fm.add_block(entry);
        }
        let i8 = ctx.shared.types.get_or_make_int(1);
        let list_ty = ctx.shared.types.get_or_make_unbounded_list(i8);
        let src = {
            let mut b = ctx.builder(entry);
            b.push_param(8).id()
        };
        if let ValueId::BlockParam(pid) = src {
            ctx.block_param_mut(pid).type_id = list_ty;
        }
        // Build `enumerate` over the unbounded list with an explicit result type:
        // its `result_type` declines an unbounded operand (no static length), so
        // the intrinsic is only ever constructed this way, never via inference.
        let enum_id = IntrinsicId::from_name("enumerate").unwrap();
        // Create the intrinsic in the block's own storage arena so the pushed
        // instruction stays strict-local (its parent block lives in the same
        // function) — a foreign instruction placement is a locality violation the
        // in-body-id localization (ruling 2) forbids.
        let env = {
            let insn = InstructionRef::from_mnemonic_with_type(
                &mut ctx,
                entry.func,
                Mnemonic::Intrinsic(IntrinsicApp {
                    id: enum_id,
                    args: vec![src.localize(entry.func)],
                }),
                list_ty,
            )
            .id;
            BasicBlock::from_id_mut(&mut ctx, entry).push_insn(insn);
            ValueId::Instruction(insn)
        };
        let ptr = ctx.get_const(0, 8).id();
        {
            let mut b = ctx.builder(entry);
            b.push_return(ptr);
        }
        let rid = BasicBlock::from_id(&ctx, entry).iter().last().unwrap().id;
        ctx.replace_instruction_mnemonic(
            rid,
            Mnemonic::Return(Return {
                ptr: ptr.localize(rid.func),
                value: Some(env.localize(rid.func)),
            }),
        );

        let mut emu = StandaloneEmulator::new(entry);
        let err = emu
            .run_pure(&ctx, f, &[SizedValue::new(0, 4)], 1000)
            .expect_err("enumerate must not be emulated");
        assert!(
            matches!(err.kind, EmulatorErrorKind::UnsupportedIntrinsic(ref n) if &**n == "enumerate"),
            "expected recoverable UnsupportedIntrinsic, got {:?}",
            err.kind
        );
    }

    /// Build pure `f(arr: [i8; n])` returning `at(arr, idx)` and run it with the
    /// array param bound to `bound`. Returns the emulated scalar lane, or `None`
    /// if `resolve_array` refuses the binding (e.g. an oversize param).
    fn run_at_over_array_param(n: usize, idx: u64, bound: SizedValue) -> Option<u64> {
        use qcode::value::{
            BasicBlock, FunctionBody, ValueId,
            insn::{IntrinsicId, Return},
        };

        let mut ctx = Context::new();
        let arr_ty = {
            let i8 = ctx.shared.types.get_or_make_int(1);
            ctx.shared.types.get_or_make_array(i8, n)
        };
        let f = FunctionBody::make(&mut ctx, "f".into()).unwrap().id;
        let entry = ctx.get_or_make_block(0x1000, f);
        {
            let mut fm = FunctionBody::from_id_mut(&mut ctx, f);
            fm.set_root(entry).unwrap();
            fm.add_block(entry);
        }
        let arr_pid = BasicBlock::from_id_mut(&mut ctx, entry).push_param(n).id;
        ctx.block_param_mut(arr_pid).type_id = arr_ty;

        let at_id = IntrinsicId::from_name("at").unwrap();
        let (ret, ptr, lane);
        {
            let mut b = ctx.builder(entry);
            let arr = ValueId::BlockParam(arr_pid);
            let i = b.shr().get_const(idx, 8);
            lane = b.push_intrinsic(at_id, vec![arr, i]).id();
            ptr = b.shr().get_const(0, 8);
            ret = b.push_return(ptr).id();
        }
        let ValueId::Instruction(rid) = ret else {
            unreachable!()
        };
        ctx.replace_instruction_mnemonic(
            rid,
            Mnemonic::Return(Return {
                ptr: ptr.localize(rid.func),
                value: Some(lane.localize(rid.func)),
            }),
        );

        let mut emu = StandaloneEmulator::new(entry);
        emu.run_pure(&ctx, f, &[bound], 1000).ok()?;
        emu.get_value(&ctx, lane)
    }

    /// Dispatch through a `switch`: the arm whose case matches the scrutinee is
    /// taken, an unmatched value falls to the default, and an arm's block
    /// arguments are bound on the way through.
    fn run_switch(scrutinee: u64) -> Option<u64> {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda sw:
            <entry @i:i64>
                switch @i { 0x0 => <a>, 0x3 => <b @v=0x63>, default => <d> };
            <a>
                return 0x11;
            <b @v:i64>
                return @v;
            <d>
                return 0x99;
            "
        );
        let root = FunctionBody::from_id(&ctx, sw).root().expect("root").id;
        let mut emu = StandaloneEmulator::new(root);
        emu.run_pure(&ctx, sw, &[SizedValue::new(scrutinee, 8)], 1000)
            .ok()?;
        let term = BasicBlock::from_id(&ctx, emu.block)
            .instruction_ids()
            .last()
            .copied()?;
        let Mnemonic::ReturnValue(r) = Instruction::from_id(&ctx, term).mnemonic() else {
            return None;
        };
        emu.get_value(&ctx, r.value.qualify(term.func))
    }

    #[test]
    fn switch_selects_the_matching_arm() {
        assert_eq!(run_switch(0), Some(0x11));
        // The matching arm binds its target's block parameter.
        assert_eq!(run_switch(3), Some(0x63));
        // No case matches 7, so control reaches the default.
        assert_eq!(run_switch(7), Some(0x99));
    }

    /// A literal-bound `[i8;4]` array param resolves through the `BlockParam` arm
    /// of `resolve_array`: each little-endian byte of `0x2f76bfc2` is readable via
    /// `at`, exactly the read the pure-call folder needs.
    #[test]
    fn run_pure_reads_array_param_bytes_little_endian() {
        let arg = SizedValue::new(0x2f76bfc2, 4);
        // LE layout of 0x2f76bfc2 = [0xc2, 0xbf, 0x76, 0x2f].
        assert_eq!(run_at_over_array_param(4, 0, arg), Some(0xc2));
        assert_eq!(run_at_over_array_param(4, 1, arg), Some(0xbf));
        assert_eq!(run_at_over_array_param(4, 2, arg), Some(0x76));
        assert_eq!(run_at_over_array_param(4, 3, arg), Some(0x2f));
    }

    /// An oversize array param (declared width beyond `SizedValue`'s 16-byte cap)
    /// can't be materialized from a scalar binding, so the defensive width check
    /// fails resolution rather than hand back a clamped, misaligned buffer.
    #[test]
    fn run_pure_rejects_oversize_array_param() {
        // A 20-byte `[i8;20]` param: the bound `SizedValue` clamps to 16 bytes, so
        // `bytes.len() (16) != ty_size (20)` and resolution must fail.
        assert_eq!(
            run_at_over_array_param(20, 0, SizedValue::new(0xff, 20)),
            None
        );
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
    fn x87_context_uses_rounding_control_and_makes_apfloat_exceptions_sticky() {
        // Half an f80 ULP at 1.0 rounds back to 1.0 under RC=nearest, but
        // upward rounding produces the next representable extended value.
        let one = 0x3fff_8000_0000_0000_0000;
        let half_ulp = 0x3fbf_8000_0000_0000_0000;
        assert_eq!(float80::add_contextual(one, half_ulp, 0x037f).bits, one);
        assert_eq!(float80::add_contextual(one, half_ulp, 0x0b7f).bits, one + 1);
        // PC=single rounds the f80 significand to 24 bits without narrowing
        // the exponent.  RC=up selects the next single-precision quantum.
        let single_half_ulp = 0x3fe7_8000_0000_0000_0000;
        assert_eq!(
            float80::add_contextual(one, single_half_ulp, 0x007f).bits,
            one
        );
        assert_eq!(
            float80::add_contextual(one, single_half_ulp, 0x087f).bits,
            one + (1u128 << 40)
        );
        // f80 stores narrow under RC too: the exact halfway value stores as
        // 1.0 under nearest-even and as the next f32 under round-up.
        let one_plus_half_single_ulp = one + (1u128 << 39);
        assert_eq!(
            float80::to_float_contextual(one_plus_half_single_ulp, 4, 0x037f).bits,
            u128::from(1.0f32.to_bits())
        );
        assert_eq!(
            float80::to_float_contextual(one_plus_half_single_ulp, 4, 0x0b7f).bits,
            u128::from((1.0f32).to_bits() + 1)
        );

        let divide_by_zero = float80::div_contextual(one, 0, 0x037f);
        assert!(divide_by_zero.status.contains(Status::DIV_BY_ZERO));

        // The concrete interpreter's non-trapping policy keeps a result for
        // unmasked exceptions, but marks ES in addition to the sticky flag.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i16 FPUControlWord;
            varnode i16 FPUStatusWord;
            varnode f80 A;
            varnode f80 B;

        <block>
            %a = load(A:10, &A);
            %b = load(B:10, &B);
            %result = %a f/ %b;
            goto <0x1001>;
        "
        );
        let mut emu = Emulator::from_block(&ctx, block);
        emu.set_varnode(FPUControlWord, 0x037b).unwrap(); // ZE unmasked
        emu.set_varnode_u128(A, one).unwrap();
        emu.set_varnode_u128(B, 0).unwrap();
        emu.run_block().unwrap();
        assert_eq!(emu.get_value(result.into()).unwrap().size().unwrap(), 10);
        assert_eq!(emu.read_varnode(FPUStatusWord), Some((1 << 2) | (1 << 7)));
    }

    /// The 80-bit square root is computed on the integer significand, so it
    /// keeps all 64 bits. Routing it through f64 would lose eleven of them.
    #[test]
    fn float80_sqrt_is_correctly_rounded_at_extended_precision() {
        let nearest = 0x037f;
        let two = 0x4000_8000_0000_0000_0000;
        let four = 0x4001_8000_0000_0000_0000;
        let one = 0x3fff_8000_0000_0000_0000;

        // sqrt(2) rounds up into the last significand bit.
        let root_two = float80::sqrt_contextual(two, nearest);
        assert_eq!(root_two.bits, 0x3fff_b504_f333_f9de_6484);
        assert!(root_two.status.contains(Status::INEXACT));

        // Exact roots stay exact and report nothing.
        for (input, expect) in [(four, two), (one, one), (0, 0)] {
            let result = float80::sqrt_contextual(input, nearest);
            assert_eq!(result.bits, expect);
            assert_eq!(result.status, Status::OK);
        }

        // sqrt(3) is a case where the integer root does round up, so nearest
        // and truncation land on different significands.
        let three = 0x4000_c000_0000_0000_0000;
        assert_eq!(
            float80::sqrt_contextual(three, nearest).bits,
            0x3fff_ddb3_d742_c265_539e
        );
        assert_eq!(
            float80::sqrt_contextual(three, 0x0f7f).bits,
            0x3fff_ddb3_d742_c265_539d
        );

        // A negative operand is invalid and yields the indefinite QNaN.
        let negative = float80::sqrt_contextual(0xbfff_8000_0000_0000_0000, nearest);
        assert_eq!(negative.bits, 0xffff_c000_0000_0000_0000);
        assert!(negative.status.contains(Status::INVALID_OP));
    }

    #[test]
    fn float80_arithmetic_preserves_extended_precision_bits() {
        // x87 80-bit encodings: significand in bits 0..64, exponent/sign in
        // bits 64..80. 3.0 is not representable by merely treating f80 as an
        // f64 bit-pattern, which was the former behavior.
        let one = SizedValue::from_bits(0x3fff_8000_0000_0000_0000, 10);
        let two = SizedValue::from_bits(0x4000_8000_0000_0000_0000, 10);
        let three = one.float_add(&two).unwrap();

        assert_eq!(three.as_bits(), 0x4000_c000_0000_0000_0000);
        assert_eq!(three.size().unwrap(), 10);
        assert_eq!(two.float_to_float(10).unwrap().as_bits(), two.as_bits());
        assert_eq!(
            SizedValue::new(3, 1).int_to_float(10).unwrap().as_bits(),
            three.as_bits()
        );
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
            %v0 = load(V0:8, &V0);
            %v1 = load(V1:8, &V1);
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
    fn gep_emulates_as_base_plus_offset() {
        let mut ctx = Context::new();

        // `Inner { val: i32 @ 0x08 }` (0x08 via leading padding), `%p : Inner*`.
        qcode!(
            ctx,
            "
            type Inner { _: 8, val: 4 };
            varnode i64 V0;

        <block>
            Inner* %p = load(V0:8, &V0);
            %fld = gep(%p.val);
            goto <0x1001>;
        "
        );

        let mut emu = Emulator::from_block(&ctx, block);
        emu.set_varnode(V0, 0x1000).unwrap();
        emu.run_block().unwrap();

        let fld = emu.get_value(fld.into()).unwrap();
        assert_eq!(fld.value().unwrap(), 0x1008);
        // Width follows the pointer base, not the immediate's default u64.
        assert_eq!(fld.size().unwrap(), 8);
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
            %v0 = load(V0:16, &V0);
            %v1 = load(V1:16, &V1);

            %res = %v0 / %v1;
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
    fn configured_register_and_body_temporary_spaces_zero_fill_missing_bytes() {
        let mut ctx = Context::new();
        let mut register = Space::new(Some("register"), 1, 8);
        register.ty = SpaceType::Register;
        let register = ctx.add_space(register);
        let function = ctx.anon_function();
        let temporary = MemorySpaceId::Temp(ctx.bodies[function].push_temp_space(TempSpace::new(
            Some("scratch"),
            1,
            8,
        )));
        let mut memory = EmulatedMemory::default();
        memory.configure_spaces(&ctx);

        for space in [register.into(), temporary] {
            assert_eq!(
                memory
                    .read(space, SizedValue::from_u64(0x1000), 4)
                    .unwrap()
                    .value()
                    .unwrap(),
                0
            );
        }

        memory
            .write(
                ctx.shared.default_space.into(),
                SizedValue::from_u64(0x1000),
                1,
                SizedValue::new(0xaa, 1),
            )
            .unwrap();
        assert!(matches!(
            memory.read(
                ctx.shared.default_space.into(),
                SizedValue::from_u64(0x1001),
                1
            ),
            Err(EmulatorErrorKind::MemoryReadError(0x1001))
        ));
    }

    #[test]
    fn temporary_spaces_with_the_same_address_are_isolated() {
        use qcode::value::TempSpace;

        let mut ctx = Context::new();
        let first_fn = FunctionBody::make(&mut ctx, "first".into()).unwrap().id;
        let second_fn = FunctionBody::make(&mut ctx, "second".into()).unwrap().id;
        let first = ctx.bodies[first_fn].push_temp_space(TempSpace::new(None, 1, 8));
        let second = ctx.bodies[second_fn].push_temp_space(TempSpace::new(None, 1, 8));
        assert_eq!(first.local, second.local, "fixture must collide local IDs");
        let first = MemorySpaceId::Temp(first);
        let second = MemorySpaceId::Temp(second);
        let mut memory = EmulatedMemory::default();
        memory.configure_spaces(&ctx);

        let address = SizedValue::from_u64(0x20);
        memory
            .write(first, address, 1, SizedValue::new(0xaa, 1))
            .unwrap();
        memory
            .write(second, address, 1, SizedValue::new(0x55, 1))
            .unwrap();

        assert_eq!(
            memory.read(first, address, 1).unwrap().value().unwrap(),
            0xaa
        );
        assert_eq!(
            memory.read(second, address, 1).unwrap().value().unwrap(),
            0x55
        );
    }

    #[test]
    fn interpreter_qualifies_colliding_local_spaces_by_function() {
        use qcode::value::TempSpace;

        fn make_writer(
            ctx: &mut Context<'static>,
            name: &'static str,
            byte: u64,
        ) -> (FunctionId, qcode::value::TempSpaceId) {
            let fid = FunctionBody::make(ctx, name.into()).unwrap().id;
            let root = BasicBlock::make(ctx, fid).id;
            FunctionBody::from_id_mut(ctx, fid).set_root(root).unwrap();
            let space = ctx.bodies[fid].push_temp_space(TempSpace::new(None, 1, 8));
            let mut b = (ctx).builder(root);
            let ptr = b.shr().get_const(0x20, 8);
            let value = b.shr().get_const(byte, 1);
            b.push_store(
                value,
                ptr,
                qcode::space::LocalMemorySpaceId::Temp(space.local),
            );
            b.push_return(ptr);
            (fid, space)
        }

        let mut ctx = Context::new();
        let (first, first_space) = make_writer(&mut ctx, "first", 0xaa);
        let (second, second_space) = make_writer(&mut ctx, "second", 0x55);
        assert_eq!(first_space.local, second_space.local);

        let root = FunctionBody::from_id(&ctx, first).root().unwrap().id;
        let mut emulator = StandaloneEmulator::new(root);
        emulator.run_function(&ctx, first).unwrap();
        emulator.run_function(&ctx, second).unwrap();

        let address = SizedValue::from_u64(0x20);
        assert_eq!(
            emulator
                .memory
                .read(MemorySpaceId::Temp(first_space), address, 1)
                .unwrap()
                .value()
                .unwrap(),
            0xaa
        );
        assert_eq!(
            emulator
                .memory
                .read(MemorySpaceId::Temp(second_space), address, 1)
                .unwrap()
                .value()
                .unwrap(),
            0x55
        );
    }

    #[test]
    fn sized_value_byte_swap_preserves_width() {
        let value = SizedValue::new(0x1234, 2).byte_swap().unwrap();
        assert_eq!(value.value().unwrap(), 0x3412);
        assert_eq!(value.size().unwrap(), 2);
    }

    #[test]
    fn swap_bytes_pcode_op_is_emulated() {
        let mut ctx = Context::new();
        let op = ctx.shared.pcode_ops.push(Box::from("swap_bytes"));
        let block_id = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let target = ctx.get_or_make_block(0x1001, block_id.func);
        let result = {
            let src = ctx.get_const(0x1234, 2).id();
            let mut builder = ctx.builder(block_id);
            let result = builder.push_pcode_op(op, vec![src], None, 2).id;
            builder.finalize(target);
            result
        };
        let mut emulator = Emulator::from_block(&ctx, block_id);

        emulator.step().unwrap();

        assert_eq!(
            emulator
                .get_value(result.into())
                .and_then(|value| value.value())
                .unwrap(),
            0x3412
        );
    }

    #[test]
    fn undef_pcode_op_is_zero_at_its_declared_width() {
        let mut ctx = Context::new();
        let op = ctx.shared.pcode_ops.push(Box::from("undef"));
        let block_id = {
            let function = ctx.anon_function();
            ctx.get_or_make_block(0x1000, function)
        };
        let target = ctx.get_or_make_block(0x1001, block_id.func);
        let result = {
            let mut builder = ctx.builder(block_id);
            let result = builder.push_pcode_op(op, vec![], None, 1).id;
            builder.finalize(target);
            result
        };
        let mut emulator = Emulator::from_block(&ctx, block_id);

        emulator.step().unwrap();

        let value = emulator.get_value(result.into()).unwrap();
        assert_eq!(value.value().unwrap(), 0);
        assert_eq!(value.size().unwrap(), 1);
    }

    #[test]
    fn rol_intrinsic_is_emulated() {
        use qcode::value::insn::IntrinsicId;
        let mut ctx = Context::new();
        let rol = IntrinsicId::from_name("rol").unwrap();
        let block_id = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let target = ctx.get_or_make_block(0x1001, block_id.func);
        let result = {
            let x = ctx.get_const(0x1234_5678, 4).id();
            let k = ctx.get_const(8, 4).id();
            let mut builder = ctx.builder(block_id);
            let result = builder.push_intrinsic(rol, vec![x, k]).id;
            builder.finalize(target);
            result
        };
        let mut emulator = Emulator::from_block(&ctx, block_id);

        emulator.step().unwrap();

        assert_eq!(
            emulator
                .get_value(result.into())
                .and_then(|value| value.value())
                .unwrap(),
            0x1234_5678u32.rotate_left(8) as u64,
        );
    }

    #[test]
    fn unknown_pcode_op_returns_typed_error() {
        let mut ctx = Context::new();
        let op = ctx.shared.pcode_ops.push(Box::from("rdpmc"));
        let block_id = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let target = ctx.get_or_make_block(0x1001, block_id.func);
        {
            let mut builder = ctx.builder(block_id);
            builder.push_pcode_op(op, vec![], None, 0);
            builder.finalize(target);
        }
        let mut emulator = Emulator::from_block(&ctx, block_id);

        let error = emulator.step().unwrap_err();

        assert!(matches!(
            error.kind,
            EmulatorErrorKind::UnsupportedPCodeOp(operation) if operation.as_ref() == "rdpmc"
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
                return at i64 0;
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
                %a = load(A:8, &A);
                %b = load(B:8, &B);
                %sum = %a + %b;
                return at i64 0;
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
                %a = load(A:8, &A);
                %b = load(B:8, &B);
                %sum = %a + %b;
                return at i64 0;
            "
        );

        let mut emu = Emulator::from_function(&ctx, function);
        emu.set_varnode(A, 0).unwrap();
        emu.set_varnode(B, 0).unwrap();
        emu.run_function(function).unwrap();

        assert!(emu.call_stack().is_empty());
    }

    #[test]
    fn unhandled_direct_call_still_enters_callee() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn callee:
            <callee_entry>
                return at i64 0;

            <caller>
                call <callee>;
            "
        );

        let mut emu = Emulator::from_block(&ctx, caller);
        emu.step().unwrap();

        assert_eq!(emu.block().id, callee_entry);
    }

    #[test]
    fn handled_direct_call_resumes_at_selected_block() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 RET;

            fn library:
            <library_entry>
                return at i64 0;

            fn function:
            <entry>
                call <library>;
            <after_call>
                %ret = load(RET:8, &RET);
                return at i64 0;
            "
        );

        let mut emu = Emulator::from_function(&ctx, function);
        emu.set_call_interceptor(move |ctx, emu, site| {
            if site.target == library {
                emu.set_varnode(ctx, RET, 42)
                    .map_err(|err| err.to_string().into_boxed_str())?;
                Ok(CallInterception::Handled(CallContinuation::Block(
                    after_call,
                )))
            } else {
                Ok(CallInterception::PassThrough)
            }
        });

        emu.run_function(function).unwrap();

        assert_eq!(
            emu.get_value(ret.into()).and_then(|v| v.value()).unwrap(),
            42
        );
        assert!(emu.call_stack().is_empty());
    }

    #[test]
    fn handled_direct_call_can_resume_by_address() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn library:
            <library_entry>
                return at i64 0;

            <entry>
                call <library>;
            <0x2000>
                return at i64 0;
            "
        );

        let mut emu = Emulator::from_block(&ctx, entry);
        emu.set_call_interceptor(move |_, _, site| {
            if site.target == library {
                Ok(CallInterception::Handled(CallContinuation::Address(0x2000)))
            } else {
                Ok(CallInterception::PassThrough)
            }
        });

        emu.step().unwrap();

        assert_eq!(emu.block().address(), Some(0x2000));
    }

    #[test]
    fn handled_direct_call_reports_unknown_continuation_address() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn library:
            <library_entry>
                return at i64 0;

            <entry>
                call <library>;
            "
        );

        let mut emu = Emulator::from_block(&ctx, entry);
        emu.set_call_interceptor(move |_, _, site| {
            if site.target == library {
                Ok(CallInterception::Handled(CallContinuation::Address(0xdead)))
            } else {
                Ok(CallInterception::PassThrough)
            }
        });

        let err = emu.step().unwrap_err();

        assert!(matches!(
            err.kind,
            EmulatorErrorKind::InvalidBlockAddress(0xdead)
        ));
    }

    #[test]
    fn call_interceptor_errors_are_reported_at_call_site() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn library:
            <library_entry>
                return at i64 0;

            <entry>
                call <library>;
            "
        );

        let mut emu = Emulator::from_block(&ctx, entry);
        emu.set_call_interceptor(|_, _, _| Err("model failed".into()));

        let err = emu.step().unwrap_err();

        assert!(matches!(
            err.kind,
            EmulatorErrorKind::InterceptError(message) if message.as_ref() == "model failed"
        ));
        assert!(err.ctx.contains("call fn library();"));
    }

    #[test]
    fn interceptor_can_model_state_across_calls() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn make_object:
            <make_object_entry>
                return at i64 0;

            fn append_byte:
            <append_byte_entry>
                return at i64 0;

            fn function:
            <entry>
                call <make_object>;
            <append>
                call <append_byte>;
            <done>
                return at i64 0;
            "
        );

        let modeled = Arc::new(Mutex::new(Vec::<u8>::new()));
        let modeled_for_hook = Arc::clone(&modeled);
        let mut emu = Emulator::from_function(&ctx, function);
        emu.set_call_interceptor(move |_, _, site| {
            let mut model = modeled_for_hook.lock().unwrap();
            if site.target == make_object {
                model.clear();
                Ok(CallInterception::Handled(CallContinuation::Block(append)))
            } else if site.target == append_byte {
                model.push(0x41);
                Ok(CallInterception::Handled(CallContinuation::Block(done)))
            } else {
                Ok(CallInterception::PassThrough)
            }
        });

        emu.run_function(function).unwrap();

        assert_eq!(*modeled.lock().unwrap(), vec![0x41]);
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
                %bad_load = load(ram:8, i64 0);
                return at i64 0;
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
