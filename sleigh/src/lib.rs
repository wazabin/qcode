//! Lift decoded [`sleigh::Instruction`]s into QCode.
//!
//! This crate is the integration boundary between `wazabin-sleigh` and
//! `qcode`. It consumes SLEIGH's flat p-code API rather
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
    CompiledSpec, Decoder, Instruction, InstructionPcode, LabelId, Opcode, PcodeOp, PcodePlan,
    PcodeSink, SPACE_CONST, SpaceId, Varnode,
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
        let address = instruction.address();
        let length = instruction.len();
        let function = self.function_for(ctx, addresses, address, function);
        // The plan carries every fact needed before the builder borrows the
        // function body, so no flat p-code vector is built or re-scanned.
        instruction
            .pcode_ops_streamed(|plan| {
                self.emitter(ctx, addresses, address, length, function, plan)
            })
            .map_err(|error| LiftError::Sleigh(error.to_string()))?
            .finish()
    }

    /// Resolves the function an instruction at `address` belongs to.
    fn function_for(
        &self,
        ctx: &mut Context<'static>,
        addresses: &mut AddressIndex,
        address: u64,
        function: Option<FunctionId>,
    ) -> FunctionId {
        match function {
            Some(function) => function,
            None => FunctionBody::from_addr_or_create_indexed(ctx, addresses, address).id,
        }
    }

    /// Creates the emitter for one instruction, resolving all module-owned
    /// blocks and callees from `plan` before borrowing the function body.
    fn emitter<'ctx>(
        &self,
        ctx: &'ctx mut Context<'static>,
        addresses: &mut AddressIndex,
        address: u64,
        length: usize,
        function: FunctionId,
        plan: &PcodePlan,
    ) -> FlatEmitter<'_, 'static, 'ctx> {
        let entry = ctx.get_or_make_block_indexed(addresses, address, function);
        BasicBlock::from_id_mut(ctx, entry).in_function(function);

        let mut branches = HashMap::default();
        for &target in plan.direct_branches() {
            let block = ctx.get_or_make_block_indexed(addresses, target, function);
            branches.insert(target, block);
        }
        let mut calls = HashMap::default();
        for &target in plan.direct_calls() {
            let callee = FunctionBody::from_addr_or_create_indexed(ctx, addresses, target).id;
            calls.insert(target, callee);
        }
        let next = ctx.get_or_make_block_indexed(addresses, address + length as u64, function);

        FlatEmitter::new(
            entry,
            next,
            ctx.builder(entry),
            &self.storage,
            self.unique_space,
            branches,
            calls,
            address,
            plan,
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
        self.lift_pcode_ops_indexed(ctx, addresses, address, length, &pcode.ops, function)
    }

    /// Lowers a borrowed flat p-code operation sequence. This is kept private
    /// because callers needing an inspectable intermediate should use
    /// [`lift_pcode_indexed`](Self::lift_pcode_indexed).
    ///
    /// Unlike the streamed path this has no plan, so it recovers the same facts
    /// by scanning the operations: their direct targets, and the operation
    /// indices local branches resolve to.
    fn lift_pcode_ops_indexed(
        &self,
        ctx: &mut Context<'static>,
        addresses: &mut AddressIndex,
        address: u64,
        length: usize,
        pcode: &[PcodeOp],
        function: Option<FunctionId>,
    ) -> Result<BlockId, LiftError> {
        let function = self.function_for(ctx, addresses, address, function);
        let plan = Self::plan_from_ops(pcode)?;
        let mut emitter = self.emitter(ctx, addresses, address, length, function, &plan.plan);
        for (index, op) in pcode.iter().enumerate() {
            if let Some(&label) = plan.labels.get(&index) {
                emitter.label(label);
            }
            match (op.opcode, op.inputs.first()) {
                (Opcode::Branch | Opcode::CBranch, Some(target)) if target.space == SPACE_CONST => {
                    let target = Self::local_target(index, target.offset, pcode.len())?;
                    let condition = (op.opcode == Opcode::CBranch)
                        .then(|| op.inputs.get(1).copied())
                        .flatten();
                    match plan.labels.get(&target) {
                        Some(&label) => emitter.branch_label(op.opcode, label, condition),
                        // A branch past the last operation is the machine
                        // instruction's fall-through, not a local block.
                        None => emitter.branch_next(op.opcode, condition),
                    }
                }
                _ => emitter.op(op.opcode, op.output, &op.inputs),
            }
        }
        emitter.finish()
    }

    /// Rebuilds the plan facts of an already-flattened instruction.
    fn plan_from_ops(pcode: &[PcodeOp]) -> Result<VectorPlan, LiftError> {
        let mut plan = PcodePlan::default();
        let mut labels = HashMap::default();
        for (index, op) in pcode.iter().enumerate() {
            let Some(target) = op.inputs.first() else {
                continue;
            };
            match op.opcode {
                Opcode::Branch | Opcode::CBranch if target.space == SPACE_CONST => {
                    let target = Self::local_target(index, target.offset, pcode.len())?;
                    if target < pcode.len() && !labels.contains_key(&target) {
                        labels.insert(target, plan.declare_label(&format!("pcode_{target}")));
                    }
                }
                Opcode::Branch | Opcode::CBranch => plan.declare_direct_branch(target.offset),
                Opcode::Call if target.space != SPACE_CONST => {
                    plan.declare_direct_call(target.offset);
                }
                _ => {}
            }
        }
        Ok(VectorPlan { plan, labels })
    }

    fn local_target(op: usize, raw: u64, len: usize) -> Result<usize, LiftError> {
        let relative = raw as i64;
        let target = isize::try_from(relative)
            .ok()
            .and_then(|relative| op.checked_add_signed(relative))
            .filter(|target| *target <= len);
        target.ok_or(LiftError::InvalidLocalBranch { op, relative })
    }
}

/// A plan recovered from already-flattened p-code, with the operation index
/// each of its labels stands at.
struct VectorPlan {
    plan: PcodePlan,
    labels: HashMap<usize, LabelId>,
}

/// One operation as it reaches the emitter, borrowed from either a streaming
/// p-code sink or an already-flattened operation vector.
struct OpRef<'a> {
    opcode: Opcode,
    output: Option<Varnode>,
    inputs: &'a [Varnode],
}

/// Emits QCode for one instruction's flat p-code.
///
/// The emitter is a [`PcodeSink`]: it never sees a flat p-code vector, and it
/// takes its whole-instruction facts — the blocks its direct branches and
/// calls reach — from the plan its owner resolved before borrowing the body.
struct FlatEmitter<'spec, 'str, 'ctx> {
    builder: Builder<'str, 'ctx>,
    /// Immutable architectural register locations, shared by every instruction.
    base_storage: &'spec HashMap<Varnode, VarnodeId>,
    /// SLEIGH's unique space, whose varnodes are instruction-local.
    unique_space: SpaceId,
    /// Per-instruction unique-space locations. Unlike register locations these
    /// must not be shared, because SLEIGH's unique space is instruction-local.
    unique_storage: HashMap<Varnode, ValueId>,
    branches: HashMap<u64, BlockId>,
    calls: HashMap<u64, FunctionId>,
    /// Blocks for the plan's instruction-local labels, made on first mention.
    labels: Vec<Option<BlockId>>,
    entry: BlockId,
    next: BlockId,
    address: u64,
    fallthrough: usize,
    /// A sink cannot fail, so the first failure is latched and the rest of the
    /// instruction is ignored; its caller discards a partial instruction.
    error: Option<LiftError>,
}

impl<'spec, 'str, 'ctx> FlatEmitter<'spec, 'str, 'ctx> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        entry: BlockId,
        next: BlockId,
        builder: Builder<'str, 'ctx>,
        base_storage: &'spec HashMap<Varnode, VarnodeId>,
        unique_space: SpaceId,
        branches: HashMap<u64, BlockId>,
        calls: HashMap<u64, FunctionId>,
        address: u64,
        plan: &PcodePlan,
    ) -> Self {
        Self {
            builder,
            base_storage,
            unique_space,
            unique_storage: HashMap::default(),
            branches,
            calls,
            // A terminal label is the instruction's fall-through, not a block.
            labels: (0..plan.labels().len())
                .map(|index| plan.is_terminal(LabelId::from_index(index)).then_some(next))
                .collect(),
            entry,
            next,
            address,
            fallthrough: 0,
            error: None,
        }
    }

    /// Closes the instruction and returns its entry block.
    fn finish(mut self) -> Result<BlockId, LiftError> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        if !self.builder.is_terminated() {
            self.builder.push_branch(self.next);
        }
        Ok(self.entry)
    }

    /// Branches to the instruction's fall-through, the target of a local
    /// branch past the last operation.
    fn branch_next(&mut self, opcode: Opcode, condition: Option<Varnode>) {
        let next = self.next;
        self.branch_to(opcode, next, condition);
    }

    fn block_for(&mut self, label: LabelId) -> BlockId {
        if let Some(block) = self.labels[label.index()] {
            return block;
        }
        let block = self.builder.get_or_make_local_label(Cow::Owned(format!(
            "pcode_{:x}_{}",
            self.address,
            label.index()
        )));
        self.labels[label.index()] = Some(block);
        block
    }

    fn branch_to(&mut self, opcode: Opcode, target: BlockId, condition: Option<Varnode>) {
        let condition = match (opcode, condition) {
            (Opcode::CBranch, Some(condition)) => match self.value(condition) {
                Ok(condition) => Some(self.ensure_bool(condition)),
                Err(error) => return self.fail(error),
            },
            (Opcode::CBranch, None) => {
                return self.fail(LiftError::InvalidArity {
                    opcode,
                    expected: 2,
                    actual: 1,
                });
            }
            _ => None,
        };
        match condition {
            Some(condition) => {
                let fallthrough = self.open_fallthrough();
                self.builder.push_cbranch(condition, target, fallthrough);
                self.builder.switch_to_block(fallthrough);
            }
            None => {
                self.builder.push_branch(target);
            }
        }
    }

    fn fail(&mut self, error: LiftError) {
        self.error.get_or_insert(error);
    }

    fn open_fallthrough(&mut self) -> BlockId {
        let label = self.builder.get_or_make_local_label(Cow::Owned(format!(
            "pcode_fallthrough_{:x}_{}",
            self.address, self.fallthrough
        )));
        self.fallthrough += 1;
        label
    }

    fn open_continuation(&mut self) {
        if !self.builder.is_terminated() {
            return;
        }
        let label = self.open_fallthrough();
        self.builder.switch_to_block(label);
    }

    fn input(&mut self, op: &OpRef<'_>, index: usize) -> Result<ValueId, LiftError> {
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
        let value = self.storage(varnode)?;
        Ok(self.builder.ensure_local(value))
    }

    /// Resolves a varnode's QCode location, giving each instruction-local
    /// unique varnode its own body-local temporary on first use. Flat p-code
    /// varnodes are mutable locations rather than SSA values, so this keeps
    /// even overlapping unique varnodes isolated as their storage identity
    /// requires.
    fn storage(&mut self, varnode: Varnode) -> Result<ValueId, LiftError> {
        if varnode.space == self.unique_space {
            let temp = *self
                .unique_storage
                .entry(varnode)
                .or_insert_with(|| ValueId::Temp(self.builder.make_temp(varnode.size)));
            return Ok(temp);
        }
        self.base_storage
            .get(&varnode)
            .copied()
            .map(ValueId::Varnode)
            .ok_or(LiftError::UnknownVarnode(varnode))
    }

    fn write(&mut self, output: Option<Varnode>, value: ValueId) -> Result<(), LiftError> {
        let Some(output) = output else {
            return Ok(());
        };
        if output.space == SPACE_CONST {
            return Err(LiftError::UnknownVarnode(output));
        }
        let destination = self.storage(output)?;
        self.builder.push_copy(value, destination);
        Ok(())
    }

    fn emit_op(&mut self, op: &OpRef<'_>) -> Result<(), LiftError> {
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
            Branch | CBranch => self.direct_branch(op),
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
        op: &OpRef<'_>,
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
        op: &OpRef<'_>,
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

    fn input_raw(&self, op: &OpRef<'_>, index: usize) -> Result<u64, LiftError> {
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

    fn space(&self, op: &OpRef<'_>, index: usize) -> Result<SpaceId, LiftError> {
        let raw = self.input_raw(op, index)?;
        usize::try_from(raw)
            .ok()
            .map(SpaceId::new)
            .ok_or(LiftError::InvalidDirectTarget(op.opcode))
    }

    /// Emits a branch out of this instruction. Instruction-local targets
    /// never reach here: they arrive as [`PcodeSink::branch_label`].
    fn direct_branch(&mut self, op: &OpRef<'_>) -> Result<(), LiftError> {
        let target = op.inputs.first().ok_or(LiftError::InvalidArity {
            opcode: op.opcode,
            expected: 1 + usize::from(op.opcode == Opcode::CBranch),
            actual: 0,
        })?;
        if target.space == SPACE_CONST {
            return Err(LiftError::InvalidDirectTarget(op.opcode));
        }
        let block = *self
            .branches
            .get(&target.offset)
            .ok_or(LiftError::InvalidDirectTarget(op.opcode))?;
        let condition = op.inputs.get(1).copied();
        self.branch_to(op.opcode, block, condition);
        Ok(())
    }
}

impl PcodeSink for FlatEmitter<'_, '_, '_> {
    fn op(&mut self, opcode: Opcode, output: Option<Varnode>, inputs: &[Varnode]) {
        if self.error.is_some() {
            return;
        }
        self.open_continuation();
        self.builder.set_address(self.address);
        let op = OpRef {
            opcode,
            output,
            inputs,
        };
        if let Err(error) = self.emit_op(&op) {
            self.fail(error);
        }
        self.builder.clear_address();
    }

    fn label(&mut self, label: LabelId) {
        if self.error.is_some() {
            return;
        }
        let block = self.block_for(label);
        // A terminal label is the instruction's fall-through: execution
        // reaches it by falling out of the instruction, not through a block.
        if block == self.next || self.builder.current_block() == block {
            return;
        }
        if !self.builder.is_terminated() {
            self.builder.push_branch(block);
        }
        self.builder.switch_to_block(block);
    }

    fn branch_label(&mut self, opcode: Opcode, label: LabelId, condition: Option<Varnode>) {
        if self.error.is_some() {
            return;
        }
        self.open_continuation();
        self.builder.set_address(self.address);
        let target = self.block_for(label);
        self.branch_to(opcode, target, condition);
        self.builder.clear_address();
    }
}

#[cfg(test)]
mod tests {
    use super::SleighLifter;
    use qcode::address_index::AddressIndex;
    use qcode_emulator::Emulator;
    use sleigh::{Compiler, Decoder, SourceDb};

    /// The streamed path and the already-flattened path must lower the same
    /// instruction to the same QCode.
    #[test]
    fn streamed_and_flat_pcode_lift_identically() {
        let spec = sleigh_precompile::x64::spec();
        let lifter = SleighLifter::new(spec);
        for bytes in [
            b"\x48\x89\xd8".as_slice(),
            b"\x48\x01\xd8",
            b"\x74\x05",
            b"\xe8\x10\x00\x00\x00",
            b"\xff\xe0",
            b"\xc3",
            b"\x0f\xb0\x1d\x00\xf1\x9a\xff",
            b"\xf7\xf1",
            b"\x48\x0f\xaf\xc3",
        ] {
            let instruction = Decoder::new(spec)
                .decode_one(0x1000, bytes, &spec.new_context())
                .unwrap();

            let mut streamed = lifter.new_context();
            lifter
                .lift_instruction(&mut streamed, &instruction, None)
                .unwrap();

            let pcode = instruction.pcode_ops().unwrap();
            let mut flat = lifter.new_context();
            let mut addresses = AddressIndex::analyze(&flat);
            lifter
                .lift_pcode_indexed(
                    &mut flat,
                    &mut addresses,
                    instruction.address(),
                    instruction.len(),
                    &pcode,
                    None,
                )
                .unwrap();

            assert_eq!(
                streamed.to_string(),
                flat.to_string(),
                "{bytes:02x?} lifts differently"
            );
        }
    }

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
    fn lifts_x64_unsized_unary_and_carry_literals() {
        let spec = sleigh_precompile::x64::spec();
        let lifter = SleighLifter::new(spec);
        for bytes in [
            b"\x48\x0f\xb3\xd8".as_slice(), // BTR RAX,RBX
            b"\x0f\x06",                    // CLTS
            b"\x48\xf7\xd8",                // NEG RAX
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
    fn emulates_loop_carried_unique_pcode_temporaries() {
        let spec = sleigh_precompile::x64::spec();
        let lifter = SleighLifter::new(spec);
        for (bytes, inputs, output_name, expected) in [
            (
                b"\x66\xf3\x0f\xbc\xc3".as_slice(),
                &[("BX", 0)][..],
                "AX",
                16,
            ),
            (b"\x48\x0f\xbd\xc3", &[("RBX", 1)], "RAX", 0),
            (
                b"\xc4\xe2\x63\xf5\xc1",
                &[("EBX", 1), ("ECX", 0x8000_0000)],
                "EAX",
                0x8000_0000,
            ),
            (
                b"\xc4\xe2\x62\xf5\xc1",
                &[("EBX", 0x8000_0000), ("ECX", 0x8000_0000)],
                "EAX",
                1,
            ),
        ] {
            let instruction = Decoder::new(spec)
                .decode_one(0x1000, bytes, &spec.new_context())
                .unwrap();
            let output_id = spec
                .registers()
                .find(|register| register.name() == output_name)
                .unwrap()
                .id;
            let mut ctx = lifter.new_context();
            lifter
                .lift_instruction(&mut ctx, &instruction, None)
                .unwrap();

            let mut emulator = Emulator::from_address(&ctx, 0x1000);
            for &(name, value) in inputs {
                let id = spec
                    .registers()
                    .find(|register| register.name() == name)
                    .unwrap()
                    .id;
                emulator.set_register(id, value).unwrap();
            }
            for _ in 0..1000 {
                if emulator.block().address() == Some(0x1000 + bytes.len() as u64) {
                    break;
                }
                emulator.step().unwrap();
            }
            assert_eq!(
                emulator.block().address(),
                Some(0x1000 + bytes.len() as u64),
                "{instruction} did not reach its fall-through block"
            );
            assert_eq!(
                emulator.read_register(output_id),
                Some(expected),
                "{instruction}"
            );
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
