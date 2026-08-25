//! Lift decoded [`sleigh::Instruction`]s into QCode.
//!
//! This crate is the integration boundary between `wazabin-sleigh` and
//! `qcode`. It consumes SLEIGH's flat [`sleigh::InstructionPcode`] API rather
//! than its source-shaped AST, keeping both core crates independent.
//!
//! Construct a [`SleighLifter`] once per compiled specification. Its
//! [`SleighLifter::new_context`] method clones a prebuilt QCode context with
//! spaces, registers, and user p-code operations already installed. Reuse an
//! [`AddressIndex`] with [`SleighLifter::lift_instruction_indexed`] when
//! lifting multiple instructions.

use std::borrow::Cow;

use jstd::registry::Registry;
use qcode::{
    address_index::AddressIndex,
    builder::Builder,
    context::Context,
    space::SpaceType,
    value::{
        BasicBlock, BlockId, FunctionBody, FunctionId, QCodeView, Renameable, Value, ValueId,
        insn::PCodeOpId,
        varnode::{Varnode as QcodeVarnode, VarnodeId},
    },
};
use rustc_hash::FxHashMap as HashMap;
use sleigh::{
    CompiledSpec, Decoder, Instruction, InstructionPcode, Opcode, PcodeOp, SPACE_CONST, SpaceId,
    Varnode,
};

/// Failure while converting flat p-code to QCode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiftError {
    /// An operation did not have the number of operands required by its opcode.
    InvalidArity {
        opcode: Opcode,
        expected: usize,
        actual: usize,
    },
    /// A flat varnode is neither a constant, a SLEIGH unique temporary, nor a
    /// register installed in this lifter's base context.
    UnknownVarnode(Varnode),
    /// A raw operation has no QCode equivalent yet.
    UnsupportedOpcode(Opcode),
    /// A relative local branch pointed outside its instruction's op sequence.
    InvalidLocalBranch { op: usize, relative: i64 },
    /// A direct control-flow target did not name an address-space varnode.
    InvalidDirectTarget(Opcode),
    /// A `SUBPIECE` range was outside its input value.
    InvalidSubPiece { offset: usize, size: usize },
    /// SLEIGH could not expand or flatten an instruction's semantics.
    Sleigh(String),
}

impl std::fmt::Display for LiftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArity {
                opcode,
                expected,
                actual,
            } => {
                write!(f, "{opcode:?} needs {expected} inputs, got {actual}")
            }
            Self::UnknownVarnode(varnode) => write!(f, "unknown flat varnode {varnode:?}"),
            Self::UnsupportedOpcode(opcode) => {
                write!(f, "unsupported flat p-code opcode {opcode:?}")
            }
            Self::InvalidLocalBranch { op, relative } => {
                write!(
                    f,
                    "local branch at op {op} has invalid relative target {relative}"
                )
            }
            Self::InvalidDirectTarget(opcode) => write!(f, "invalid direct target for {opcode:?}"),
            Self::InvalidSubPiece { offset, size } => {
                write!(
                    f,
                    "SUBPIECE [{offset}..{}) is outside its input",
                    offset + size
                )
            }
            Self::Sleigh(error) => f.write_str(error),
        }
    }
}

impl std::error::Error for LiftError {}

/// Specification-specific state reused across all lifting sessions.
///
/// Building the base context interns every SLEIGH register and installs all
/// spaces and `define pcodeop` names once. A session obtains a prebuilt clone via
/// [`new_context`](Self::new_context), avoiding per-instruction registry setup
/// and name hashing.
pub struct SleighLifter<'spec> {
    spec: &'spec CompiledSpec,
    base: Context<'static>,
    storage: HashMap<Varnode, VarnodeId>,
    unique_space: SpaceId,
}

impl<'spec> SleighLifter<'spec> {
    /// Creates a lifter for `spec` and prebuilds its immutable QCode state.
    pub fn new(spec: &'spec CompiledSpec) -> Self {
        let mut base = Context::default();
        base.shared.default_space = spec.default_space();
        let unique_space = spec
            .spaces()
            .find(|space| matches!(&space.space().ty, SpaceType::Unique))
            .expect("compiled SLEIGH specifications always have a unique space")
            .id;

        // Both registries retain SLEIGH's stable IDs, so flat p-code space and
        // user-op IDs can be used directly on the hot path.
        let spaces: Registry<SpaceId, qcode::space::Space> =
            spec.spaces().map(|space| space.space().clone()).collect();
        base.load_spaces(spaces);
        base.shared.pcode_ops = spec
            .pcode_ops()
            .map(Box::<str>::from)
            .collect::<Registry<PCodeOpId, Box<str>>>();
        for space in spec.spaces() {
            if let Some(name) = space.name() {
                base.shared.named_spaces.insert(Box::from(name), space.id);
            }
        }

        let mut storage = HashMap::default();
        for register in spec.registers() {
            let id = QcodeVarnode::make(
                &mut base,
                register.offset() as i64,
                register.size(),
                register.space(),
            )
            .with_name(Cow::Owned(register.name().to_owned()))
            .expect("SLEIGH register names are unique")
            .id;
            base.shared.registers.insert(register.id, id);
            storage.insert(
                Varnode::new(register.space(), register.offset() as u64, register.size()),
                id,
            );
        }

        Self {
            spec,
            base,
            storage,
            unique_space,
        }
    }

    /// Returns the compiled SLEIGH specification used by this lifter.
    pub fn spec(&self) -> &'spec CompiledSpec {
        self.spec
    }

    /// Creates a QCode module initialized for this specification.
    pub fn new_context(&self) -> Context<'static> {
        self.base.clone()
    }

    /// Decodes one instruction from `bytes` and lowers its flat p-code into
    /// `ctx`.
    ///
    /// This is the byte-oriented entry point for clients that need only the
    /// SLEIGH-to-QCode boundary, not a binary container or recursive discovery
    /// engine. Bulk lifters should retain an address index and use
    /// [`decode_and_lift_indexed`](Self::decode_and_lift_indexed).
    pub fn decode_and_lift(
        &self,
        ctx: &mut Context<'static>,
        address: u64,
        bytes: &[u8],
        function: Option<FunctionId>,
    ) -> Result<BlockId, LiftError> {
        let mut addresses = AddressIndex::analyze(ctx);
        self.decode_and_lift_indexed(ctx, &mut addresses, address, bytes, function)
    }

    /// Decodes one instruction from `bytes` and lowers it while reusing
    /// `addresses` across a lifting session.
    pub fn decode_and_lift_indexed(
        &self,
        ctx: &mut Context<'static>,
        addresses: &mut AddressIndex,
        address: u64,
        bytes: &[u8],
        function: Option<FunctionId>,
    ) -> Result<BlockId, LiftError> {
        let decode_context = self.spec.new_context();
        let instruction = Decoder::new(self.spec)
            .decode_one(address, bytes, &decode_context)
            .map_err(|error| LiftError::Sleigh(error.to_string()))?;
        self.lift_instruction_indexed(ctx, addresses, &instruction, function)
    }

    /// Lowers a decoded instruction's flat p-code into `ctx`.
    ///
    /// This convenience method builds an address index. Bulk lifters should use
    /// [`lift_instruction_indexed`](Self::lift_instruction_indexed) instead.
    pub fn lift_instruction(
        &self,
        ctx: &mut Context<'static>,
        instruction: &Instruction<'_, '_>,
        function: Option<FunctionId>,
    ) -> Result<BlockId, LiftError> {
        let mut addresses = AddressIndex::analyze(ctx);
        self.lift_instruction_indexed(ctx, &mut addresses, instruction, function)
    }

    /// Lowers a decoded instruction while reusing `addresses` across a lifting
    /// session.
    pub fn lift_instruction_indexed(
        &self,
        ctx: &mut Context<'static>,
        addresses: &mut AddressIndex,
        instruction: &Instruction<'_, '_>,
        function: Option<FunctionId>,
    ) -> Result<BlockId, LiftError> {
        let pcode = instruction
            .pcode_ops()
            .map_err(|error| LiftError::Sleigh(error.to_string()))?;
        self.lift_pcode_indexed(
            ctx,
            addresses,
            instruction.address(),
            instruction.len(),
            &pcode,
            function,
        )
    }

    /// Lowers already-flattened p-code. This is useful for cached or
    /// differentially-tested instruction semantics.
    pub fn lift_pcode_indexed(
        &self,
        ctx: &mut Context<'static>,
        addresses: &mut AddressIndex,
        address: u64,
        length: usize,
        pcode: &InstructionPcode,
        function: Option<FunctionId>,
    ) -> Result<BlockId, LiftError> {
        let function = match function {
            Some(function) => function,
            None => FunctionBody::from_addr_or_create_indexed(ctx, addresses, address).id,
        };
        let entry = ctx.get_or_make_block_indexed(addresses, address, function);
        BasicBlock::from_id_mut(ctx, entry).in_function(function);

        // Resolve all module-owned targets before borrowing the function body
        // through Builder. This is also the only pass over the flat operations
        // needed for direct inter-instruction control flow.
        let mut branches = HashMap::default();
        let mut calls = HashMap::default();
        for op in &pcode.ops {
            match op.opcode {
                Opcode::Branch | Opcode::CBranch => {
                    if let Some(target) = op
                        .inputs
                        .first()
                        .filter(|target| target.space != SPACE_CONST)
                    {
                        let block =
                            ctx.get_or_make_block_indexed(addresses, target.offset, function);
                        branches.insert(target.offset, block);
                    }
                }
                Opcode::Call => {
                    if let Some(target) = op
                        .inputs
                        .first()
                        .filter(|target| target.space != SPACE_CONST)
                    {
                        let callee = FunctionBody::from_addr_or_create_indexed(
                            ctx,
                            addresses,
                            target.offset,
                        )
                        .id;
                        calls.insert(target.offset, callee);
                    }
                }
                _ => {}
            }
        }
        let next = ctx.get_or_make_block_indexed(addresses, address + length as u64, function);

        let mut emitter = FlatEmitter::new(
            ctx.builder(entry),
            &self.storage,
            self.unique_space,
            branches,
            calls,
            address,
        );
        emitter.emit(pcode, next)?;
        Ok(entry)
    }
}

struct FlatEmitter<'storage, 'str, 'ctx> {
    builder: Builder<'str, 'ctx>,
    storage: &'storage HashMap<Varnode, VarnodeId>,
    unique_space: SpaceId,
    unique: HashMap<u64, ValueId>,
    branches: HashMap<u64, BlockId>,
    calls: HashMap<u64, FunctionId>,
    address: u64,
    fallthrough: usize,
}

impl<'storage, 'str, 'ctx> FlatEmitter<'storage, 'str, 'ctx> {
    fn new(
        builder: Builder<'str, 'ctx>,
        storage: &'storage HashMap<Varnode, VarnodeId>,
        unique_space: SpaceId,
        branches: HashMap<u64, BlockId>,
        calls: HashMap<u64, FunctionId>,
        address: u64,
    ) -> Self {
        Self {
            builder,
            storage,
            unique_space,
            unique: HashMap::default(),
            branches,
            calls,
            address,
            fallthrough: 0,
        }
    }

    fn emit(&mut self, pcode: &InstructionPcode, next: BlockId) -> Result<(), LiftError> {
        // A raw op can introduce at most one unique result. Reserving once
        // keeps temporary tracking allocation-free for normal instructions.
        self.unique.reserve(pcode.ops.len());
        let mut labels = HashMap::default();
        labels.reserve(pcode.ops.len());
        for (index, op) in pcode.ops.iter().enumerate() {
            if matches!(op.opcode, Opcode::Branch | Opcode::CBranch)
                && let Some(target) = op
                    .inputs
                    .first()
                    .filter(|target| target.space == SPACE_CONST)
            {
                let target = Self::local_target(index, target.offset, pcode.ops.len())?;
                // A label at the end of instruction p-code is the machine
                // instruction's fall-through, not an empty local block.
                if target < pcode.ops.len() {
                    labels.entry(target).or_insert_with(|| {
                        self.builder.get_or_make_local_label(Cow::Owned(format!(
                            "pcode_{:x}_{target}",
                            self.address
                        )))
                    });
                }
            }
        }

        for (index, op) in pcode.ops.iter().enumerate() {
            if let Some(&target) = labels.get(&index)
                && self.builder.current_block() != target
            {
                if !self.builder.is_terminated() {
                    self.builder.push_branch(target);
                }
                self.builder.switch_to_block(target);
            }
            self.open_continuation();
            self.builder.set_address(self.address);
            self.emit_op(index, op, &labels, pcode.ops.len(), next)?;
            self.builder.clear_address();
        }
        if !self.builder.is_terminated() {
            self.builder.push_branch(next);
        }
        Ok(())
    }

    fn open_continuation(&mut self) {
        if !self.builder.is_terminated() {
            return;
        }
        let label = self.builder.get_or_make_local_label(Cow::Owned(format!(
            "pcode_fallthrough_{:x}_{}",
            self.address, self.fallthrough
        )));
        self.fallthrough += 1;
        self.builder.switch_to_block(label);
    }

    fn local_target(op: usize, raw: u64, len: usize) -> Result<usize, LiftError> {
        let relative = raw as i64;
        let target = isize::try_from(relative)
            .ok()
            .and_then(|relative| op.checked_add_signed(relative))
            .filter(|target| *target <= len);
        target.ok_or(LiftError::InvalidLocalBranch { op, relative })
    }

    fn input(&mut self, op: &PcodeOp, index: usize) -> Result<ValueId, LiftError> {
        let value = *op.inputs.get(index).ok_or(LiftError::InvalidArity {
            opcode: op.opcode,
            expected: index + 1,
            actual: op.inputs.len(),
        })?;
        self.value(value)
    }

    fn value(&mut self, varnode: Varnode) -> Result<ValueId, LiftError> {
        if varnode.space == SPACE_CONST {
            return Ok(self.builder.shr().get_const(varnode.offset, varnode.size));
        }
        if varnode.space == self.unique_space {
            return self
                .unique
                .get(&varnode.offset)
                .copied()
                .ok_or(LiftError::UnknownVarnode(varnode));
        }
        let id = self
            .storage
            .get(&varnode)
            .copied()
            .ok_or(LiftError::UnknownVarnode(varnode))?;
        Ok(self.builder.ensure_local(ValueId::Varnode(id)))
    }

    fn write(&mut self, output: Option<Varnode>, value: ValueId) -> Result<(), LiftError> {
        let Some(output) = output else {
            return Ok(());
        };
        if output.space == SPACE_CONST {
            return Err(LiftError::UnknownVarnode(output));
        }
        if let Some(&id) = self.storage.get(&output) {
            self.builder.push_copy(value, ValueId::Varnode(id));
        } else if output.space == self.unique_space {
            self.unique.insert(output.offset, value);
        } else {
            return Err(LiftError::UnknownVarnode(output));
        }
        Ok(())
    }

    fn emit_op(
        &mut self,
        index: usize,
        op: &PcodeOp,
        labels: &HashMap<usize, BlockId>,
        len: usize,
        next: BlockId,
    ) -> Result<(), LiftError> {
        use Opcode::*;
        let unary = |this: &mut Self, f: fn(&mut Builder<'str, 'ctx>, ValueId) -> ValueId| {
            let input = this.input(op, 0)?;
            let value = f(&mut this.builder, input);
            this.write(op.output, value)
        };
        let binary =
            |this: &mut Self, f: fn(&mut Builder<'str, 'ctx>, ValueId, ValueId) -> ValueId| {
                let lhs = this.input(op, 0)?;
                let rhs = this.input(op, 1)?;
                let value = f(&mut this.builder, lhs, rhs);
                this.write(op.output, value)
            };
        match op.opcode {
            Copy => {
                let value = self.input(op, 0)?;
                self.write(op.output, value)
            }
            IntAdd => binary(self, |b, a, c| b.push_add(a, c).id()),
            IntSub => binary(self, |b, a, c| b.push_sub(a, c).id()),
            IntMult => binary(self, |b, a, c| b.push_mul(a, c).id()),
            IntDiv => binary(self, |b, a, c| b.push_div(a, c).id()),
            IntSDiv => binary(self, |b, a, c| b.push_sdiv(a, c).id()),
            IntRem => binary(self, |b, a, c| b.push_mod(a, c).id()),
            IntSRem => binary(self, |b, a, c| b.push_smod(a, c).id()),
            IntAnd => binary(self, |b, a, c| b.push_bit_and(a, c).id()),
            IntOr => binary(self, |b, a, c| b.push_bit_or(a, c).id()),
            IntXor => binary(self, |b, a, c| b.push_bit_xor(a, c).id()),
            IntLeft => binary(self, |b, a, c| b.push_shl(a, c).id()),
            IntRight => binary(self, |b, a, c| b.push_shr(a, c).id()),
            IntSRight => binary(self, |b, a, c| b.push_sshr(a, c).id()),
            IntEqual => binary(self, |b, a, c| b.push_eq(a, c).id()),
            IntNotEqual => binary(self, |b, a, c| b.push_ne(a, c).id()),
            IntSLess => binary(self, |b, a, c| b.push_slt(a, c).id()),
            IntSLessEqual => binary(self, |b, a, c| b.push_sle(a, c).id()),
            IntLess => binary(self, |b, a, c| b.push_lt(a, c).id()),
            IntLessEqual => binary(self, |b, a, c| b.push_le(a, c).id()),
            IntCarry => binary(self, |b, a, c| b.push_carry(a, c).id()),
            IntSCarry => binary(self, |b, a, c| b.push_scarry(a, c).id()),
            IntSBorrow => binary(self, |b, a, c| b.push_sborrow(a, c).id()),
            IntNegate => unary(self, |b, a| b.push_bit_negate(a).id()),
            Int2Comp => unary(self, |b, a| b.push_neg(a).id()),
            BoolNegate => {
                let input = self.input(op, 0)?;
                let input = self.ensure_bool(input);
                let value = self.builder.push_bool_not(input).id();
                self.write(op.output, value)
            }
            BoolXor => self.bool_binop(op, |b, a, c| b.push_bool_xor(a, c).id()),
            BoolAnd => self.bool_binop(op, |b, a, c| b.push_bool_and(a, c).id()),
            BoolOr => self.bool_binop(op, |b, a, c| b.push_bool_or(a, c).id()),
            FloatAdd => binary(self, |b, a, c| b.push_fadd(a, c).id()),
            FloatSub => binary(self, |b, a, c| b.push_fsub(a, c).id()),
            FloatMult => binary(self, |b, a, c| b.push_fmul(a, c).id()),
            FloatDiv => binary(self, |b, a, c| b.push_fdiv(a, c).id()),
            FloatEqual => binary(self, |b, a, c| b.push_feq(a, c).id()),
            FloatNotEqual => binary(self, |b, a, c| b.push_fne(a, c).id()),
            FloatLess => binary(self, |b, a, c| b.push_flt(a, c).id()),
            FloatLessEqual => binary(self, |b, a, c| b.push_fle(a, c).id()),
            FloatNeg => unary(self, |b, a| b.push_fneg(a).id()),
            FloatNan => unary(self, |b, a| b.push_is_nan(a).id()),
            FloatAbs => unary(self, |b, a| b.push_abs(a).id()),
            FloatSqrt => unary(self, |b, a| b.push_sqrt(a).id()),
            FloatCeil => unary(self, |b, a| b.push_ceil(a).id()),
            FloatFloor => unary(self, |b, a| b.push_floor(a).id()),
            FloatRound => unary(self, |b, a| b.push_round(a).id()),
            IntZext => self.convert(op, |b, value, size| b.push_zext(value, size).id()),
            IntSext => self.convert(op, |b, value, size| b.push_sext(value, size).id()),
            FloatInt2Float => {
                self.convert(op, |b, value, size| b.push_int_to_float(value, size).id())
            }
            FloatFloat2Float => {
                self.convert(op, |b, value, size| b.push_float_to_float(value, size).id())
            }
            FloatTrunc => self.convert(op, |b, value, size| b.push_trunc(value, size).id()),
            PopCount => self.convert(op, |b, value, size| b.push_popcount(value, size).id()),
            LzCount => self.convert(op, |b, value, size| b.push_lzcount(value, size).id()),
            Load => {
                let output = op.output.ok_or(LiftError::InvalidArity {
                    opcode: op.opcode,
                    expected: 2,
                    actual: op.inputs.len(),
                })?;
                let space = self.space(op, 0)?;
                let ptr = self.input(op, 1)?;
                let value = self.builder.push_load::<true>(ptr, output.size, space).id();
                self.write(Some(output), value)
            }
            Store => {
                let space = self.space(op, 0)?;
                let ptr = self.input(op, 1)?;
                let value = self.input(op, 2)?;
                self.builder.push_store(value, ptr, space);
                Ok(())
            }
            SubPiece => {
                let src = self.input(op, 0)?;
                let offset = self.input_raw(op, 1)? as usize;
                let output = op.output.ok_or(LiftError::InvalidArity {
                    opcode: op.opcode,
                    expected: 2,
                    actual: op.inputs.len(),
                })?;
                let value = self
                    .builder
                    .get_range(src, offset..offset + output.size)
                    .ok_or(LiftError::InvalidSubPiece {
                        offset,
                        size: output.size,
                    })?
                    .id();
                self.write(Some(output), value)
            }
            CallOther => {
                let id = PCodeOpId::new(self.input_raw(op, 0)? as usize);
                let mut args = Vec::with_capacity(op.inputs.len().saturating_sub(1));
                for input in op.inputs.iter().skip(1).copied() {
                    args.push(self.value(input)?);
                }
                let size = op.output.map_or(0, |output| output.size);
                let value = self.builder.push_pcode_op(id, args, None, size).id();
                self.write(op.output, value)
            }
            Branch => self.branch(index, op, labels, len, next),
            CBranch => self.cbranch(index, op, labels, len, next),
            BranchInd => {
                let target = self.input(op, 0)?;
                self.builder.push_branchind(target);
                Ok(())
            }
            Call => {
                let target = op.inputs.first().ok_or(LiftError::InvalidArity {
                    opcode: op.opcode,
                    expected: 1,
                    actual: 0,
                })?;
                let callee = self
                    .calls
                    .get(&target.offset)
                    .copied()
                    .ok_or(LiftError::InvalidDirectTarget(op.opcode))?;
                self.builder.push_call(callee);
                Ok(())
            }
            CallInd => {
                let target = self.input(op, 0)?;
                self.builder.push_call_ind(target);
                Ok(())
            }
            Return => {
                let target = self.input(op, 0)?;
                self.builder.push_return(target);
                Ok(())
            }
            _ => Err(LiftError::UnsupportedOpcode(op.opcode)),
        }
    }

    fn ensure_bool(&mut self, value: ValueId) -> ValueId {
        let ty = self.builder.view().type_of(value);
        if self.builder.shr().types.is_bool(ty) {
            return value;
        }
        let size = self.builder.shr().types.size_of(ty).max(1);
        let zero = self.builder.shr().get_const(0, size);
        self.builder.push_ne(value, zero).id()
    }

    fn bool_binop(
        &mut self,
        op: &PcodeOp,
        f: fn(&mut Builder<'str, 'ctx>, ValueId, ValueId) -> ValueId,
    ) -> Result<(), LiftError> {
        let lhs = self.input(op, 0)?;
        let lhs = self.ensure_bool(lhs);
        let rhs = self.input(op, 1)?;
        let rhs = self.ensure_bool(rhs);
        let value = f(&mut self.builder, lhs, rhs);
        self.write(op.output, value)
    }

    fn convert(
        &mut self,
        op: &PcodeOp,
        f: fn(&mut Builder<'str, 'ctx>, ValueId, usize) -> ValueId,
    ) -> Result<(), LiftError> {
        let output = op.output.ok_or(LiftError::InvalidArity {
            opcode: op.opcode,
            expected: 1,
            actual: op.inputs.len(),
        })?;
        let input = self.input(op, 0)?;
        let value = f(&mut self.builder, input, output.size);
        self.write(Some(output), value)
    }

    fn input_raw(&self, op: &PcodeOp, index: usize) -> Result<u64, LiftError> {
        let value = op.inputs.get(index).ok_or(LiftError::InvalidArity {
            opcode: op.opcode,
            expected: index + 1,
            actual: op.inputs.len(),
        })?;
        if value.space != SPACE_CONST {
            return Err(LiftError::InvalidDirectTarget(op.opcode));
        }
        Ok(value.offset)
    }

    fn space(&self, op: &PcodeOp, index: usize) -> Result<SpaceId, LiftError> {
        let raw = self.input_raw(op, index)?;
        usize::try_from(raw)
            .ok()
            .map(SpaceId::new)
            .ok_or(LiftError::InvalidDirectTarget(op.opcode))
    }

    fn branch(
        &mut self,
        index: usize,
        op: &PcodeOp,
        labels: &HashMap<usize, BlockId>,
        len: usize,
        next: BlockId,
    ) -> Result<(), LiftError> {
        let target = op.inputs.first().ok_or(LiftError::InvalidArity {
            opcode: op.opcode,
            expected: 1,
            actual: 0,
        })?;
        let block = if target.space == SPACE_CONST {
            let target = Self::local_target(index, target.offset, len)?;
            if target == len { next } else { labels[&target] }
        } else {
            *self
                .branches
                .get(&target.offset)
                .ok_or(LiftError::InvalidDirectTarget(op.opcode))?
        };
        self.builder.push_branch(block);
        Ok(())
    }

    fn cbranch(
        &mut self,
        index: usize,
        op: &PcodeOp,
        labels: &HashMap<usize, BlockId>,
        len: usize,
        next: BlockId,
    ) -> Result<(), LiftError> {
        let target = op.inputs.first().ok_or(LiftError::InvalidArity {
            opcode: op.opcode,
            expected: 2,
            actual: 0,
        })?;
        let condition = self.input(op, 1)?;
        let condition = self.ensure_bool(condition);
        let target = if target.space == SPACE_CONST {
            let target = Self::local_target(index, target.offset, len)?;
            if target == len { next } else { labels[&target] }
        } else {
            *self
                .branches
                .get(&target.offset)
                .ok_or(LiftError::InvalidDirectTarget(op.opcode))?
        };
        let fallthrough = self.builder.get_or_make_local_label(Cow::Owned(format!(
            "pcode_fallthrough_{:x}_{}",
            self.address, self.fallthrough
        )));
        self.fallthrough += 1;
        self.builder.push_cbranch(condition, target, fallthrough);
        self.builder.switch_to_block(fallthrough);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::SleighLifter;
    use sleigh::{Compiler, Decoder, SourceDb};

    #[test]
    fn lifts_flat_register_copy_from_a_decoded_instruction() {
        let mut sources = SourceDb::new();
        let root = sources.add_file(
            "tiny.slaspec",
            "define endian=little;
             define space ram type=ram_space size=4 default;
             define space register type=register_space size=4;
             define register offset=0 size=4 [ r0 ];
             define token instr(8) op=(0,7);
             :set is op=1 { r0 = 0x12345678:4; }
             :add is op=2 { r0 = r0 + 1:4; }",
        );
        let spec = Compiler::new(&mut sources).compile(root).unwrap();
        let instruction = Decoder::new(&spec)
            .decode_one(0x1000, &[1], &spec.new_context())
            .unwrap();
        assert_eq!(instruction.pcode_ops().unwrap().ops.len(), 1);

        let lifter = SleighLifter::new(&spec);
        let mut ctx = lifter.new_context();
        lifter
            .lift_instruction(&mut ctx, &instruction, None)
            .unwrap();
        // COPY to r0 plus the explicit machine-instruction fall-through.
        assert_eq!(ctx.instructions().count(), 2);
    }

    #[test]
    fn decodes_and_lifts_bytes_without_a_product_disassembler() {
        let spec = sleigh_precompile::x64::spec();
        let lifter = SleighLifter::new(spec);
        let mut ctx = lifter.new_context();

        lifter
            .decode_and_lift(&mut ctx, 0x1000, b"\x48\x89\xd8", None)
            .unwrap();

        assert!(ctx.instructions().count() > 1);
    }

    #[test]
    fn lifts_terminal_local_branch_to_instruction_fallthrough() {
        let spec = sleigh_precompile::x64::spec();
        let instruction = Decoder::new(spec)
            .decode_one(0x1000, b"\x0f\xb0\x1d\x00\xf1\x9a\xff", &spec.new_context())
            .unwrap();
        let lifter = SleighLifter::new(spec);
        let mut ctx = lifter.new_context();
        lifter
            .lift_instruction(&mut ctx, &instruction, None)
            .unwrap();
    }

    #[test]
    fn lifts_x64_non_rep_string_forms_from_the_binit_corpus() {
        let spec = sleigh_precompile::x64::spec();
        let lifter = SleighLifter::new(spec);
        for bytes in [
            b"\xa4".as_slice(),
            b"\x66\xa5",
            b"\xa5",
            b"\x48\xa5",
            b"\xa6",
            b"\x66\xa7",
            b"\xa7",
            b"\x48\xa7",
            b"\xac",
            b"\x66\xad",
            b"\xad",
            b"\x48\xad",
            b"\xae",
            b"\x66\xaf",
            b"\xaf",
            b"\x48\xaf",
            b"\xaa",
            b"\x66\xab",
            b"\xab",
            b"\x48\xab",
        ] {
            let instruction = Decoder::new(spec)
                .decode_one(0x1000, bytes, &spec.new_context())
                .unwrap();
            let mut ctx = lifter.new_context();
            lifter
                .lift_instruction(&mut ctx, &instruction, None)
                .unwrap();
        }
    }

    #[test]
    fn lifts_flat_arithmetic_without_rewalking_the_sleigh_ast() {
        let mut sources = SourceDb::new();
        let root = sources.add_file(
            "tiny.slaspec",
            "define endian=little;
             define space ram type=ram_space size=4 default;
             define space register type=register_space size=4;
             define register offset=0 size=4 [ r0 ];
             define token instr(8) op=(0,7);
             :add is op=2 { r0 = r0 + 1:4; }",
        );
        let spec = Compiler::new(&mut sources).compile(root).unwrap();
        let instruction = Decoder::new(&spec)
            .decode_one(0x1000, &[2], &spec.new_context())
            .unwrap();
        assert_eq!(
            instruction.pcode_ops().unwrap().ops[0].opcode,
            sleigh::Opcode::IntAdd
        );

        let lifter = SleighLifter::new(&spec);
        let mut ctx = lifter.new_context();
        lifter
            .lift_instruction(&mut ctx, &instruction, None)
            .unwrap();
        // Register load, add, register copy, and machine fall-through.
        assert_eq!(ctx.instructions().count(), 4);
    }
}
