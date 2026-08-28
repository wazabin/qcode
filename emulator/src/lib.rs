use qcode::{
    context::Context,
    space::MemorySpaceId,
    value::{
        BlockId, FunctionId, InstructionId, ValueId, Varnode,
        insn::{
            Binary, Binop, Carry, FloatBinop, FloatToFloat, FloatToInt, Gep, InstructionRef,
            IntBinop, IntToFloat, IsFloatNaN, Load, LzCount, Mnemonic, PopCount, Range, SBorrow,
            SCarry, Sext, Store, Unary, Unop, Zext,
        },
        varnode::{VarnodeId, register::RegisterId},
    },
};

mod concrete;

pub use concrete::{BodyArg, Emulator, SizedValue, StandaloneEmulator};

#[derive(Debug, Clone)]
pub struct CallSite {
    pub instruction: InstructionId,
    pub block: BlockId,
    pub target: FunctionId,
    pub args: Vec<ValueId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallContinuation {
    Block(BlockId),
    Address(u64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallInterception {
    PassThrough,
    Handled(CallContinuation),
}

#[derive(Debug)]
pub struct EmulatorError {
    /// What went wrong
    pub kind: EmulatorErrorKind,
    /// The context in which the error occurred
    /// Usually the instruction, it's address and function it belongs to
    pub ctx: String,

    /// The address at which the error occurred, if applicable.
    pub address: Option<u64>,
}

impl EmulatorError {
    pub fn new(kind: EmulatorErrorKind, insn: &InstructionRef<'_, '_>) -> Self {
        Self {
            kind,
            ctx: format!(
                "Instruction: {}\nBlock: {:?}\nFunction: {:?}",
                insn.as_statement(),
                insn.parent().map(|b| b.name()),
                insn.function().map(|f| f.name())
            ),
            address: insn.parent().and_then(|b| b.address()),
        }
    }
}

#[derive(Debug)]
pub enum EmulatorErrorKind {
    /// Branch/call/return resolved to an address with no known block
    InvalidBlockAddress(u64),
    /// Called a function that has no root block
    EmptyFunctionRoot(FunctionId),
    /// A pass-local minted callee escaped its installation barrier.
    UnresolvedMintedCallee(u32),
    /// `run_until` was given an address with no corresponding block
    UnknownAddress(u64),
    /// Attempted to construct a memory region that would overflow the address space
    AddressOverflow(u64, usize),
    /// Attempted to read memory at an address that hasn't been written to
    MemoryReadError(u64),
    /// Attempted to write to an address that can't be read back (e.g. MMIO)
    MemoryWriteError(u64),
    /// This value is too large to be represented in the target type (e.g. trying to interpret a 128-bit value as a 64-bit value)
    ValueError(u128),
    /// Attempted to access a register that is not present in the context
    UnknownRegister(RegisterId),
    /// Attempted to read from a memory space that has not been initialised
    UnknownSpace(MemorySpaceId),
    /// Encountered an architecture-specific p-code operation without an emulator implementation
    UnsupportedPCodeOp(Box<str>),
    /// An intrinsic's evaluator could not produce a result (e.g. a trap or
    /// unsupported operand width)
    UnsupportedIntrinsic(Box<str>),
    /// A user-provided call interceptor failed while modeling a call
    InterceptError(Box<str>),
    /// A bounded emulation run (e.g. `run_pure`) exceeded its step budget.
    StepBudgetExceeded(usize),
    /// A mnemonic the interpreter does not model (e.g. `map`, whose whole-array
    /// emulation is deferred). Recoverable: a best-effort consumer such as
    /// pure-call folding simply declines to harvest, rather than crashing.
    UnsupportedMnemonic(&'static str),
    /// Execution reached a block with no instructions (a malformed/degenerate
    /// block left behind by lifting). Recoverable: bounded consumers decline to
    /// harvest rather than indexing out of bounds.
    EmptyBlock(BlockId),
    /// A [`poison`](qcode::value::poison) value was demanded as a concrete datum.
    /// Poison has undefined bits, so reading it is a hard error (argpromote v2);
    /// propagating it as an unread operand is fine. Bounded consumers (e.g.
    /// pure-call folding) treat this as a bail signal.
    PoisonRead,
}

impl std::fmt::Display for EmulatorErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBlockAddress(addr) => write!(f, "invalid block address {addr:#x}"),
            Self::EmptyFunctionRoot(func) => write!(f, "function {func:?} has no root block"),
            Self::UnresolvedMintedCallee(slot) => {
                write!(f, "minted callee placeholder #{slot} is not executable")
            }
            Self::UnknownAddress(addr) => write!(f, "unknown address {addr:#x}"),
            Self::AddressOverflow(addr, size) => {
                write!(f, "address overflow at {addr:#x} with size {size}")
            }
            Self::MemoryReadError(addr) => write!(f, "memory read error at address {addr:#x}"),
            Self::MemoryWriteError(addr) => write!(f, "memory write error at address {addr:#x}"),
            Self::ValueError(value) => write!(f, "value {value} is too large to represent"),
            Self::UnknownRegister(reg) => write!(f, "register {reg:?} not found in context"),
            Self::UnknownSpace(space) => write!(f, "memory space {space:?} not initialised"),
            Self::UnsupportedPCodeOp(op) => write!(f, "unsupported p-code operation `{op}`"),
            Self::UnsupportedIntrinsic(op) => write!(f, "unsupported intrinsic `{op}`"),
            Self::InterceptError(message) => write!(f, "call interceptor failed: {message}"),
            Self::StepBudgetExceeded(budget) => {
                write!(f, "emulation exceeded step budget of {budget}")
            }
            Self::UnsupportedMnemonic(op) => write!(f, "unsupported mnemonic `{op}`"),
            Self::EmptyBlock(block) => write!(f, "block {block:?} has no instructions"),
            Self::PoisonRead => write!(f, "read of a poison value (undefined bits)"),
        }
    }
}

impl std::fmt::Display for EmulatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "emulator error: {}", self.kind)?;
        write!(f, " (context: {})", self.ctx)?;
        Ok(())
    }
}

impl std::error::Error for EmulatorError {}

pub type Result<T> = std::result::Result<T, EmulatorError>;

/// A trait to describe a value
/// This is used to abstract over different types of interpretations (symbolic, concrete, abstract, etc.)
pub trait DomainValue: Clone + Copy {
    /// Returns the size of this value in bytes
    fn size(&self) -> std::result::Result<usize, EmulatorErrorKind>;

    /// Attempt to read this value as a little-endian unsigned integer.
    fn value(&self) -> std::result::Result<u64, EmulatorErrorKind>;

    fn from_u64(value: u64) -> Self;

    /// Creates an all-zero value at `size` bytes. Domains that track width
    /// should override this; the default preserves compatibility for domains
    /// that only have a machine-word zero representation.
    fn zero(_size: usize) -> Self {
        Self::from_u64(0)
    }

    fn is_float_nan(&self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_to_float(&self, size: usize) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_to_float(&self, size: usize) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_to_int(&self, size: usize) -> std::result::Result<Self, EmulatorErrorKind>;
    fn zext(&self, size: usize) -> std::result::Result<Self, EmulatorErrorKind>;
    fn sext(&self, size: usize) -> std::result::Result<Self, EmulatorErrorKind>;
    fn range(&self, start: usize, size: usize) -> std::result::Result<Self, EmulatorErrorKind>;
    fn byte_swap(&self) -> std::result::Result<Self, EmulatorErrorKind>;

    /// Evaluate a pure intrinsic on its concrete operands, producing an
    /// `out_size`-byte result via the intrinsic's shared evaluator.
    fn intrinsic(
        id: qcode::value::insn::IntrinsicId,
        args: &[Self],
        out_size: usize,
    ) -> std::result::Result<Self, EmulatorErrorKind>;

    fn pop_count(&self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn lz_count(&self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn carry(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn scarry(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn sborrow(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;

    fn int_not(&self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_negate(&self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_negate(&self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_abs(&self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_sqrt(&self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_ceil(&self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_floor(&self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_round(&self) -> std::result::Result<Self, EmulatorErrorKind>;

    fn int_equal(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_not_equal(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_less(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_sless(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_less_equal(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_sless_equal(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_add(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_sub(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_xor(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_and(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_or(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_shift_left(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_shift_right(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_sshift_right(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_mul(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_div(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_rem(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_sdiv(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn int_srem(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;

    fn float_add(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_sub(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_mul(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_div(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_equal(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_not_equal(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_less(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
    fn float_less_equal(&self, other: &Self) -> std::result::Result<Self, EmulatorErrorKind>;
}

pub trait DomainMemory {
    type V: DomainValue;

    /// Reads a value from the given address.
    /// The size of the value must be less than or equal to the size of the region being read from.
    fn read(
        &self,
        space: MemorySpaceId,
        addr: Self::V,
        size: usize,
    ) -> std::result::Result<Self::V, EmulatorErrorKind>;

    /// Writes a value to the given address.
    /// The size of the value must be less than or equal to the size of the region being written to.
    fn write(
        &mut self,
        space: MemorySpaceId,
        addr: Self::V,
        size: usize,
        data: Self::V,
    ) -> std::result::Result<(), EmulatorErrorKind>;
}

pub trait Interpreter {
    type V: DomainValue;
    type M: DomainMemory<V = Self::V>;

    fn ctx(&self) -> &Context<'_>;

    fn memory(&mut self) -> &mut Self::M;

    /// Gets the value of the given value ID, if it exists in the context.
    fn get_value(&mut self, id: ValueId) -> std::result::Result<Self::V, EmulatorErrorKind>;

    /// Reads the value of a varnode from the emulated memory space.
    fn get_varnode_value(
        &mut self,
        id: VarnodeId,
    ) -> std::result::Result<Self::V, EmulatorErrorKind> {
        let varnode = Varnode::from_id(self.ctx(), id);
        let space = varnode.space().id;
        let addr = Self::V::from_u64(varnode.address() as u64);
        let size = varnode.size();
        self.memory().read(space.into(), addr, size)
    }

    /// Writes a value to a varnode in the emulated memory space.
    fn set_varnode_value(
        &mut self,
        id: VarnodeId,
        value: Self::V,
    ) -> std::result::Result<(), EmulatorErrorKind> {
        let varnode = Varnode::from_id(self.ctx(), id);
        let space = varnode.space().id;
        let addr = Self::V::from_u64(varnode.address() as u64);
        let size = varnode.size();
        self.memory().write(space.into(), addr, size, value)?;
        Ok(())
    }

    /// Returns the value of the given register, if it exists in the context.
    fn get_register_value(
        &mut self,
        reg_id: RegisterId,
    ) -> std::result::Result<Self::V, EmulatorErrorKind> {
        let id = *self
            .ctx()
            .shared
            .registers
            .get(&reg_id)
            .ok_or(EmulatorErrorKind::UnknownRegister(reg_id))?;
        self.get_varnode_value(id)
    }

    /// Sets the value of the given register, if it exists in the context.
    fn set_register_value(
        &mut self,
        reg_id: RegisterId,
        value: Self::V,
    ) -> std::result::Result<(), EmulatorErrorKind> {
        let id = *self
            .ctx()
            .shared
            .registers
            .get(&reg_id)
            .ok_or(EmulatorErrorKind::UnknownRegister(reg_id))?;
        self.set_varnode_value(id, value)
    }

    /// Gets the value of the given instruction, if it has one.
    /// This is used for instructions that produce a value, such as copy, load, and binary operations.
    fn interpret_(
        &mut self,
        insn: &InstructionRef<'_, '_>,
    ) -> std::result::Result<Option<Self::V>, EmulatorErrorKind> {
        // Operands are stored bare-local; qualify with the instruction's own
        // function (strict IR locality: operands live in the same arena).
        let func = insn.id.func;
        let v = match insn.mnemonic() {
            // ===== Memory operations =====
            &Mnemonic::Load(Load { space, ptr, size }) => {
                let addr = self.get_value(ptr.qualify(func))?;
                Some(self.memory().read(space.qualify(func), addr, size)?)
            }

            &Mnemonic::Store(Store {
                space,
                ptr,
                size,
                src,
            }) => {
                let addr = self.get_value(ptr.qualify(func))?;
                let value = self.get_value(src.qualify(func))?;
                self.memory()
                    .write(space.qualify(func), addr, size, value)?;
                None
            }

            // ===== Control flow operations =====
            Mnemonic::Branch(_)
            | Mnemonic::CBranch(_)
            | Mnemonic::BranchInd(_)
            | Mnemonic::Call(_)
            | Mnemonic::CallInd(_)
            | Mnemonic::Return(_)
            | Mnemonic::ReturnValue(_)
            | Mnemonic::Apply(_) => None,

            // ===== Unary operations =====
            Mnemonic::Unop(Unary { op, src }) => {
                let value = self.get_value(src.qualify(func))?;
                let v = match op {
                    Unop::IntNegate => value.int_negate(),
                    Unop::IntNot => value.int_not(),
                    Unop::FloatNegate => value.float_negate(),
                    Unop::FloatAbs => value.float_abs(),
                    Unop::FloatSqrt => value.float_sqrt(),
                    Unop::FloatCeil => value.float_ceil(),
                    Unop::FloatFloor => value.float_floor(),
                    Unop::FloatRound => value.float_round(),
                    _ => todo!("unimplemented unary operation: {:?}", op),
                }?;
                Some(v)
            }

            // ===== Binary operations =====
            Mnemonic::Binop(Binary { op, lhs, rhs }) => {
                let value1 = self.get_value(lhs.qualify(func))?;
                let value2 = self.get_value(rhs.qualify(func))?;
                let v = match *op {
                    Binop::Int(IntBinop::Equal) => value1.int_equal(&value2),
                    Binop::Int(IntBinop::NotEqual) => value1.int_not_equal(&value2),
                    Binop::Int(IntBinop::Less) => value1.int_less(&value2),
                    Binop::Int(IntBinop::SLess) => value1.int_sless(&value2),
                    Binop::Int(IntBinop::LessEqual) => value1.int_less_equal(&value2),
                    Binop::Int(IntBinop::SLessEqual) => value1.int_sless_equal(&value2),
                    Binop::Int(IntBinop::Add) => value1.int_add(&value2),
                    Binop::Int(IntBinop::Sub) => value1.int_sub(&value2),
                    Binop::Int(IntBinop::Xor) => value1.int_xor(&value2),
                    Binop::Int(IntBinop::And) => value1.int_and(&value2),
                    Binop::Int(IntBinop::Or) => value1.int_or(&value2),
                    Binop::Int(IntBinop::ShiftLeft) => value1.int_shift_left(&value2),
                    Binop::Int(IntBinop::ShiftRight) => value1.int_shift_right(&value2),
                    Binop::Int(IntBinop::SShiftRight) => value1.int_sshift_right(&value2),
                    Binop::Int(IntBinop::Mul) => value1.int_mul(&value2),
                    Binop::Int(IntBinop::Div) => value1.int_div(&value2),
                    Binop::Int(IntBinop::Rem) => value1.int_rem(&value2),
                    Binop::Int(IntBinop::Sdiv) => value1.int_sdiv(&value2),
                    Binop::Int(IntBinop::Srem) => value1.int_srem(&value2),

                    Binop::Float(FloatBinop::Add) => value1.float_add(&value2),
                    Binop::Float(FloatBinop::Sub) => value1.float_sub(&value2),
                    Binop::Float(FloatBinop::Mul) => value1.float_mul(&value2),
                    Binop::Float(FloatBinop::Div) => value1.float_div(&value2),
                    Binop::Float(FloatBinop::Equal) => value1.float_equal(&value2),
                    Binop::Float(FloatBinop::NotEqual) => value1.float_not_equal(&value2),
                    Binop::Float(FloatBinop::Less) => value1.float_less(&value2),
                    Binop::Float(FloatBinop::LessEqual) => value1.float_less_equal(&value2),
                    _ => todo!("unimplemented binary operation: {:?}", op),
                }?;
                Some(v)
            }

            // ===== Bit manipulation operations =====
            &Mnemonic::PopCount(PopCount { src }) => {
                let value = self.get_value(src.qualify(func))?;
                Some(value.pop_count()?)
            }

            &Mnemonic::LzCount(LzCount { src }) => {
                let value = self.get_value(src.qualify(func))?;
                Some(value.lz_count()?)
            }

            &Mnemonic::Carry(Carry { lhs, rhs }) => {
                let value1 = self.get_value(lhs.qualify(func))?;
                let value2 = self.get_value(rhs.qualify(func))?;
                Some(value1.carry(&value2)?)
            }

            &Mnemonic::SCarry(SCarry { lhs, rhs }) => {
                let value1 = self.get_value(lhs.qualify(func))?;
                let value2 = self.get_value(rhs.qualify(func))?;
                Some(value1.scarry(&value2)?)
            }

            &Mnemonic::SBorrow(SBorrow { lhs, rhs }) => {
                let value1 = self.get_value(lhs.qualify(func))?;
                let value2 = self.get_value(rhs.qualify(func))?;
                Some(value1.sborrow(&value2)?)
            }

            // ===== Casting operations =====
            &Mnemonic::IsFloatNaN(IsFloatNaN { src }) => {
                let value = self.get_value(src.qualify(func))?;
                Some(value.is_float_nan()?)
            }
            &Mnemonic::IntToFloat(IntToFloat { src, size }) => {
                let value = self.get_value(src.qualify(func))?;
                Some(value.int_to_float(size)?)
            }
            &Mnemonic::FloatToFloat(FloatToFloat { src, size }) => {
                let value = self.get_value(src.qualify(func))?;
                Some(value.float_to_float(size)?)
            }
            &Mnemonic::FloatToInt(FloatToInt { src, size }) => {
                let value = self.get_value(src.qualify(func))?;
                Some(value.float_to_int(size)?)
            }
            &Mnemonic::Zext(Zext { src, size }) => {
                let value = self.get_value(src.qualify(func))?;
                Some(value.zext(size)?)
            }
            &Mnemonic::Sext(Sext { src, size }) => {
                let value = self.get_value(src.qualify(func))?;
                Some(value.sext(size)?)
            }
            &Mnemonic::Range(Range { src, start, size }) => {
                let value = self.get_value(src.qualify(func))?;
                Some(value.range(start, size)?)
            }

            // ===== Aggregate operations =====
            // `Gep` is pure pointer arithmetic: base pointer + constant byte
            // offset. The width follows the base (int_add uses the lhs width),
            // so the immediate's default u64 width is harmless.
            &Mnemonic::Gep(Gep { base, offset }) => {
                let base = self.get_value(base.qualify(func))?;
                let offset = Self::V::from_u64(offset as u64);
                Some(base.int_add(&offset)?)
            }

            // ===== Other operations =====
            Mnemonic::PCodeOp(op) => {
                let name = self.ctx().shared.pcode_ops[op.id].clone();
                match (name.as_ref(), op.args.as_slice()) {
                    ("swap_bytes", [src]) => Some(self.get_value(src.qualify(func))?.byte_swap()?),
                    // SLEIGH uses this zero-argument user-op as an explicit
                    // write of an architecturally undefined value. Concrete
                    // emulation deliberately chooses zero, while retaining
                    // the p-code op and its destination in the IR for
                    // analysis consumers.
                    ("undef", []) => Some(Self::V::zero(insn.size())),
                    // The LOCK prefix's bus semantics are not observable in a
                    // single-threaded replay: it orders an access against other
                    // agents, and constrains nothing about the resulting state.
                    // The paired markers stay in the IR for analysis consumers
                    // that care which region is atomic; they produce no value.
                    ("LOCK" | "UNLOCK", []) => None,
                    _ => return Err(EmulatorErrorKind::UnsupportedPCodeOp(name)),
                }
            }

            Mnemonic::Intrinsic(intr) => {
                let out_size = insn.size();
                let mut args = Vec::with_capacity(intr.args.len());
                for &arg in &intr.args {
                    args.push(self.get_value(arg.qualify(func))?);
                }
                Some(Self::V::intrinsic(intr.id, &args, out_size)?)
            }

            // `map` has no interpreter (whole-array emulation is deferred). Bail
            // recoverably so a best-effort consumer — pure-call folding emulating a
            // function whose return depends on a `map` — declines to harvest the
            // field instead of crashing the whole analysis. (Element projection
            // does not go through emulation; it inlines the body via `ArrayProject`.)
            Mnemonic::Map(_) => return Err(EmulatorErrorKind::UnsupportedMnemonic("map")),

            _ => todo!("unimplemented mnemonic: {:?}", insn.mnemonic()),
        };

        Ok(v)
    }

    fn interpret(&mut self, insn: InstructionRef<'_, '_>) -> Result<Option<Self::V>> {
        self.interpret_(&insn)
            .map_err(|kind| EmulatorError::new(kind, &insn))
    }
}
