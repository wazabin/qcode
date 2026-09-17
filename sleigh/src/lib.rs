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
//!
//! # Example
//!
//! Decode one x86-64 instruction and lift it into a QCode context:
//!
//! ```no_run
//! use sleigh::Decoder;
//! use sleigh_precompile::x64;
//! use qcode::address_index::AddressIndex;
//! use wazabin_qcode_sleigh::{FlatPcode, SleighLifter};
//!
//! let spec = x64::spec();
//!
//! // `48 89 d8` is `MOV RAX, RBX`.
//! let instruction = Decoder::new(spec)
//!     .decode_one(0x1000, &[0x48, 0x89, 0xd8], &spec.new_context())
//!     .expect("the bytes decode");
//! let flat = FlatPcode::lower(&instruction).expect("the semantics emit");
//!
//! let lifter = SleighLifter::new(spec);
//! let mut ctx = lifter.new_context();
//! let mut addresses = AddressIndex::analyze(&ctx);
//! lifter
//!     .lift_pcode_indexed(&mut ctx, &mut addresses, &flat, None)
//!     .expect("the p-code lowers");
//!
//! println!("{ctx}");
//! ```
//!
//! The `qcode-dump` example prints every stage — disassembly text, SLEIGH AST,
//! flat p-code, and the resulting QCode:
//!
//! ```sh
//! cargo run --example qcode-dump -- 4889d8
//! ```

pub mod decode;
pub mod session;
pub mod vm_source;

use std::{borrow::Cow, cell::RefCell};

use jstd::registry::Registry;
use qcode::{
    address_index::AddressIndex,
    builder::Builder,
    context::{ArchitectureId, Context},
    lift::{
        CallTarget, Construction, Continuation, Emitter, ExitArm, ExitKind, LiftTarget, Lifted,
        TargetError,
    },
    space::SpaceType,
    value::{
        BlockId, FunctionBody, FunctionId, InstructionId, QCodeView, Renameable, TempId, Value,
        ValueId,
        insn::{Callee, PCodeOpId},
        varnode::{Varnode as QcodeVarnode, VarnodeId},
    },
};
use rustc_hash::FxHashMap as HashMap;
use sleigh::{
    CompiledSpec, Decoder, EmitError, Instruction, InstructionPcode, LabelId, Opcode, PcodeOp,
    PcodePlan, PcodeSink, SPACE_CONST, SpaceId, SpecFingerprint, Varnode,
};

/// Already-flattened p-code of one instruction, with the identity of the
/// specification that produced it.
///
/// Flat p-code names registers by space and offset, which mean what a lifter's
/// specification says they mean; p-code of another specification would be
/// lowered against the wrong registers, silently. A decoded [`Instruction`]
/// carries its specification, so lowering it is checked; this type carries the
/// same provenance for p-code that has left the instruction behind — kept for
/// inspection, cached, or deserialized — so the flat entry points can check it
/// too. [`lower`](Self::lower) takes it from the instruction; [`from_parts`](Self::from_parts)
/// is for p-code stored with its fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatPcode {
    fingerprint: SpecFingerprint,
    address: u64,
    length: usize,
    pcode: InstructionPcode,
}

impl FlatPcode {
    /// Flattens `instruction`'s semantics and records which specification
    /// they came from.
    pub fn lower(instruction: &Instruction<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            fingerprint: instruction.spec().fingerprint(),
            address: instruction.address(),
            length: instruction.len(),
            pcode: instruction.pcode_ops()?,
        })
    }

    /// Wraps p-code the caller kept, with the fingerprint of the specification
    /// that produced it — which the caller must have stored alongside it, as
    /// the p-code itself does not say. A wrong fingerprint is the caller's
    /// bug: the lifter trusts it.
    pub fn from_parts(
        fingerprint: SpecFingerprint,
        address: u64,
        length: usize,
        pcode: InstructionPcode,
    ) -> Self {
        Self {
            fingerprint,
            address,
            length,
            pcode,
        }
    }

    /// The specification the p-code was produced by.
    pub fn fingerprint(&self) -> SpecFingerprint {
        self.fingerprint
    }

    pub fn address(&self) -> u64 {
        self.address
    }

    pub fn length(&self) -> usize {
        self.length
    }

    /// The operations, for inspection.
    pub fn pcode(&self) -> &InstructionPcode {
        &self.pcode
    }

    pub fn ops(&self) -> &[PcodeOp] {
        &self.pcode.ops
    }

    /// Takes the operations back out.
    pub fn into_pcode(self) -> InstructionPcode {
        self.pcode
    }
}

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
    /// The destination refused the instruction: its address is taken, a
    /// target belongs to another function, or the index is stale.
    Target(TargetError),
    /// The context was not built for this lifter's specification: it carries
    /// no architecture stamp, or another specification's.
    IncompatibleContext,
    /// The decoded instruction or flat p-code was produced by a different
    /// specification than this lifter's, so its varnodes name registers this
    /// lifter cannot map.
    IncompatibleSpec,
    /// The bytes did not decode.
    Decode(sleigh::DecodeError),
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
            Self::Target(error) => error.fmt(f),
            Self::IncompatibleContext => {
                f.write_str("the context was not built for this specification")
            }
            Self::IncompatibleSpec => {
                f.write_str("the instruction was decoded by another specification")
            }
            Self::Decode(error) => error.fmt(f),
        }
    }
}

impl From<sleigh::DecodeError> for LiftError {
    fn from(error: sleigh::DecodeError) -> Self {
        Self::Decode(error)
    }
}

impl std::error::Error for LiftError {}

impl From<TargetError> for LiftError {
    fn from(error: TargetError) -> Self {
        Self::Target(error)
    }
}

impl From<sleigh::EmitError> for LiftError {
    fn from(error: sleigh::EmitError) -> Self {
        Self::Sleigh(error.to_string())
    }
}

/// Specification-specific state reused across all lifting sessions.
///
/// Building the base context interns every SLEIGH register and installs all
/// spaces and `define pcodeop` names once. A session obtains a prebuilt clone via
/// [`new_context`](Self::new_context), avoiding per-instruction registry setup
/// and name hashing.
pub struct SleighLifter<'spec> {
    spec: &'spec CompiledSpec,
    /// The specification's identity, as stamped on every context this lifter
    /// builds and required of every one it lowers into.
    architecture: ArchitectureId,
    base: Context<'static>,
    storage: HashMap<Varnode, VarnodeId>,
    unique_space: SpaceId,
    flat_control_flow: bool,
}

impl<'spec> SleighLifter<'spec> {
    /// Lowers the guest's calls and returns as plain jumps.
    ///
    /// A guest `call` is a push and a jump, and a `ret` is a pop and an
    /// indirect jump. SLEIGH already emits both halves: the stack write and the
    /// stack read are ordinary p-code, and the `CALL`/`RETURN` operations on
    /// top of them are nothing but the transfer of control. So lowering those
    /// to [`Branch`](qcode::value::insn::Branch) and
    /// [`BranchInd`](qcode::value::insn::BranchInd) loses no semantics.
    ///
    /// What it avoids is the *function structure* that a real `Call` implies —
    /// a callee `FunctionId`, entered through that function's root block. An
    /// emulator has no use for it: the guest keeps its own stack, and its code
    /// is flat, with branch targets rather than a call graph. Worse, imposing
    /// it is not merely useless but unsound for code discovered on demand,
    /// because guest code branches across function boundaries freely, and a
    /// block belongs to exactly one function's arena.
    ///
    /// A decompiler wants the opposite and gets it by default. This is for
    /// running code, not for understanding it.
    pub fn with_flat_control_flow(mut self) -> Self {
        self.flat_control_flow = true;
        self
    }

    /// Creates a lifter for `spec` and prebuilds its immutable QCode state.
    pub fn new(spec: &'spec CompiledSpec) -> Self {
        let architecture = ArchitectureId::new(spec.fingerprint().as_u128());
        let mut base = Context::default();
        base.set_architecture(architecture);
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
            architecture,
            base,
            storage,
            unique_space,
            flat_control_flow: false,
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
    ) -> Result<Lifted, LiftError> {
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
    ) -> Result<Lifted, LiftError> {
        let decode_context = self.spec.new_context();
        let instruction = Decoder::new(self.spec).decode_one(address, bytes, &decode_context)?;
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
    ) -> Result<Lifted, LiftError> {
        let mut addresses = AddressIndex::analyze(ctx);
        self.lift_instruction_indexed(ctx, &mut addresses, instruction, function)
    }

    /// Lowers a decoded instruction while reusing `addresses` across a lifting
    /// session.
    ///
    /// This is the indexed entry point: `addresses` must be current for `ctx`,
    /// which is the caller's to keep. Without `function`, the instruction is
    /// lowered into the function at its address, made if there is none — a
    /// function per isolated instruction, which bulk lifters avoid by naming
    /// their host. See [`lift_into`](Self::lift_into) for the checked form.
    pub fn lift_instruction_indexed(
        &self,
        ctx: &mut Context<'static>,
        addresses: &mut AddressIndex,
        instruction: &Instruction<'_, '_>,
        function: Option<FunctionId>,
    ) -> Result<Lifted, LiftError> {
        // Both checks precede `function_for`, so a refused instruction never
        // leaves even a host function behind.
        self.check_instruction(instruction)?;
        self.check_compatible(ctx)?;
        let function = self.function_for(ctx, addresses, instruction.address(), function);
        let mut target = LiftTarget::bind_indexed(ctx, addresses, function)?;
        self.lift_into(&mut target, instruction)
    }

    /// Lowers a decoded instruction into a bound target.
    ///
    /// This is the one lowering of decoded input: every byte-oriented and
    /// indexed entry point decodes or binds and then comes here. The
    /// instruction is constructed transactionally — see
    /// [`qcode::lift::target`] — so an error leaves the target as it was.
    pub fn lift_into(
        &self,
        target: &mut LiftTarget<'_, 'static>,
        instruction: &Instruction<'_, '_>,
    ) -> Result<Lifted, LiftError> {
        self.lower(target, instruction, self.flat_control_flow)
    }

    /// [`lift_into`](Self::lift_into) with the control-flow lowering chosen
    /// by the caller rather than the lifter: a scratch session lowers calls
    /// as jumps whatever the lifter would, since a callee function cannot be
    /// discarded with the instruction that made it.
    pub(crate) fn lower(
        &self,
        target: &mut LiftTarget<'_, 'static>,
        instruction: &Instruction<'_, '_>,
        flat: bool,
    ) -> Result<Lifted, LiftError> {
        self.check_instruction(instruction)?;
        self.check_compatible(target.context())?;
        let mut construction = target.begin(instruction.address(), instruction.len())?;
        // The plan carries every fact needed before the builder borrows the
        // function body, so no flat p-code vector is built or re-scanned.
        instruction
            .try_pcode_ops_streamed(|plan| self.emitter(&mut construction, plan, flat))?
            .finish()?;
        Ok(construction.commit()?)
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

    /// Refuses a context that was not built for this lifter's specification.
    ///
    /// Compatibility is identity, not resemblance: the context must carry the
    /// [architecture stamp](Context::architecture) of this specification's
    /// [fingerprint](CompiledSpec::fingerprint), which
    /// [`new_context`](Self::new_context) — of this lifter or of any lifter for
    /// the same specification — put there. A context of another specification
    /// is refused even when its spaces and registers happen to coincide with
    /// this one's, and so is a context built by hand or reloaded from before
    /// stamps existed: this lifter's register map names varnodes by the ids
    /// its own base assigned, and only the stamp says the destination assigned
    /// them the same way.
    fn check_compatible(&self, ctx: &Context<'static>) -> Result<(), LiftError> {
        if ctx.architecture() != Some(self.architecture) {
            return Err(LiftError::IncompatibleContext);
        }
        Ok(())
    }

    /// Refuses an instruction decoded by another specification.
    ///
    /// The instruction's varnodes only mean what this lifter's register map
    /// says if it was decoded by this specification, so the fingerprints must
    /// agree. A foreign-spec instruction would otherwise be lowered against
    /// the wrong registers, silently.
    fn check_instruction(&self, instruction: &Instruction<'_, '_>) -> Result<(), LiftError> {
        self.check_fingerprint(instruction.spec().fingerprint())
    }

    fn check_fingerprint(&self, fingerprint: SpecFingerprint) -> Result<(), LiftError> {
        if fingerprint != self.spec.fingerprint() {
            return Err(LiftError::IncompatibleSpec);
        }
        Ok(())
    }

    /// Creates the emitter for one instruction, resolving all module-owned
    /// blocks and callees from `plan` before borrowing the function body.
    fn emitter<'c, 'a>(
        &self,
        construction: &'c mut Construction<'_, 'a, 'static>,
        plan: &PcodePlan,
        flat: bool,
    ) -> Result<FlatEmitter<'_, 'static, 'c>, LiftError> {
        let address = construction.address();
        let length = construction.length();

        let mut workspace = take_workspace();
        let filled = (|| {
            for &target in plan.direct_branches() {
                workspace
                    .branches
                    .insert(target, construction.block_at(target)?);
            }
            for &target in plan.direct_calls() {
                if flat {
                    // Just another branch target, resolved in this same function.
                    workspace
                        .branches
                        .insert(target, construction.block_at(target)?);
                } else {
                    workspace
                        .calls
                        .insert(target, construction.callee_at(target)?);
                }
            }
            construction.block_at(address + length as u64)
        })();
        let next = match filled {
            Ok(next) => next,
            Err(error) => {
                return_workspace(workspace);
                return Err(error.into());
            }
        };

        Ok(FlatEmitter::new(
            next,
            construction.emitter(),
            &self.storage,
            self.unique_space,
            workspace,
            address,
            plan,
            flat,
        ))
    }

    /// Lowers already-flattened p-code, which must be this specification's
    /// (see [`FlatPcode`]). This is useful for cached or differentially-tested
    /// instruction semantics.
    pub fn lift_pcode_indexed(
        &self,
        ctx: &mut Context<'static>,
        addresses: &mut AddressIndex,
        flat: &FlatPcode,
        function: Option<FunctionId>,
    ) -> Result<Lifted, LiftError> {
        // Both checks precede `function_for`, so refused p-code never leaves
        // even a host function behind.
        self.check_fingerprint(flat.fingerprint())?;
        self.check_compatible(ctx)?;
        let function = self.function_for(ctx, addresses, flat.address(), function);
        let mut target = LiftTarget::bind_indexed(ctx, addresses, function)?;
        self.lift_pcode_into(&mut target, flat)
    }

    /// Lowers already-flattened p-code into a bound target. See
    /// [`lift_pcode_indexed`](Self::lift_pcode_indexed).
    ///
    /// Unlike the streamed path this has no plan, so it recovers the same facts
    /// by scanning the operations: their direct targets, and the operation
    /// indices local branches resolve to.
    pub fn lift_pcode_into(
        &self,
        target: &mut LiftTarget<'_, 'static>,
        flat: &FlatPcode,
    ) -> Result<Lifted, LiftError> {
        self.check_fingerprint(flat.fingerprint())?;
        self.check_compatible(target.context())?;
        let address = flat.address();
        let length = flat.length();
        let pcode = flat.ops();
        let plan = Self::plan_from_ops(pcode, self.flat_control_flow)?;
        let mut construction = target.begin(address, length)?;
        let mut emitter = self.emitter(&mut construction, &plan.plan, self.flat_control_flow)?;
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
        emitter.finish()?;
        Ok(construction.commit()?)
    }

    /// Rebuilds the plan facts of an already-flattened instruction.
    fn plan_from_ops(pcode: &[PcodeOp], flat_control_flow: bool) -> Result<VectorPlan, LiftError> {
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
                    if flat_control_flow {
                        plan.declare_direct_branch(target.offset);
                    } else {
                        plan.declare_direct_call(target.offset);
                    }
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

/// The per-instruction tables of a [`FlatEmitter`], kept between
/// instructions so their capacity is: an instruction fills them, the next
/// one clears them, and neither allocates once the tables have seen the
/// busiest lowering of the session.
#[derive(Default)]
struct Workspace {
    unique_storage: HashMap<Varnode, UniqueSlot>,
    dirty: Vec<Varnode>,
    branches: HashMap<u64, BlockId>,
    calls: HashMap<u64, Callee>,
    labels: Vec<Option<BlockId>>,
}

impl Workspace {
    fn clear(&mut self) {
        self.unique_storage.clear();
        self.dirty.clear();
        self.branches.clear();
        self.calls.clear();
        self.labels.clear();
    }
}

thread_local! {
    /// The workspace of the last emitter on this thread, empty. A nested
    /// lowering — there is none — would find it taken and start from empty.
    static WORKSPACE: RefCell<Workspace> = RefCell::default();
}

fn take_workspace() -> Workspace {
    WORKSPACE.with(|slot| std::mem::take(&mut *slot.borrow_mut()))
}

fn return_workspace(mut workspace: Workspace) {
    workspace.clear();
    WORKSPACE.with(|slot| *slot.borrow_mut() = workspace);
}

/// Where one instruction-local unique varnode lives.
///
/// Flat p-code writes its uniques as mutable locations, but nearly all of them
/// are written once and read in the same straight-line run of operations, so
/// the emitter keeps a unique as the SSA value last written to it and gives it
/// memory only when that value has to cross a block boundary — or when it is
/// read before it is written, which is the p-code's bug to have and the
/// temporary's to absorb.
#[derive(Default, Clone, Copy)]
struct UniqueSlot {
    /// The SSA value holding the unique's contents, valid within the current
    /// block; a value from an earlier block is dropped when a new one opens.
    value: Option<ValueId>,
    /// The body-local temporary the unique spills to, made on first need.
    temp: Option<TempId>,
    /// Whether `value` has been written since it was last stored to `temp`.
    dirty: bool,
}

/// Emits QCode for one instruction's flat p-code.
///
/// The emitter is a [`PcodeSink`]: it never sees a flat p-code vector, and it
/// takes its whole-instruction facts — the blocks its direct branches and
/// calls reach — from the plan its owner resolved before borrowing the body.
struct FlatEmitter<'spec, 'str, 'ctx> {
    /// The builder into the construction, which also takes the record of the
    /// instruction's blocks and exits at the operations that open and take
    /// them — before `flat` decides what they lower to.
    builder: Emitter<'ctx, 'str>,
    /// Immutable architectural register locations, shared by every instruction.
    base_storage: &'spec HashMap<Varnode, VarnodeId>,
    /// SLEIGH's unique space, whose varnodes are instruction-local.
    unique_space: SpaceId,
    /// Per-instruction unique-space locations. Unlike register locations these
    /// must not be shared, because SLEIGH's unique space is instruction-local.
    unique_storage: HashMap<Varnode, UniqueSlot>,
    /// The uniques written since their last spill, in write order, so a spill
    /// stores them deterministically.
    dirty: Vec<Varnode>,
    /// Whether the plan has a label that opens a block of this instruction.
    /// Without one, no p-code after an unconditional transfer can run, so
    /// nothing needs to survive it.
    has_local_blocks: bool,
    branches: HashMap<u64, BlockId>,
    calls: HashMap<u64, Callee>,
    /// Blocks for the plan's instruction-local labels, made on first mention.
    labels: Vec<Option<BlockId>>,
    next: BlockId,
    address: u64,
    fallthrough: usize,
    /// Lower the guest's calls and returns as jumps. See
    /// [`SleighLifter::with_flat_control_flow`].
    flat: bool,
    /// A sink cannot fail, so the first failure is latched and the rest of the
    /// instruction is ignored; its caller discards a partial instruction.
    error: Option<LiftError>,
}

impl<'spec, 'str, 'ctx> FlatEmitter<'spec, 'str, 'ctx> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        next: BlockId,
        builder: Emitter<'ctx, 'str>,
        base_storage: &'spec HashMap<Varnode, VarnodeId>,
        unique_space: SpaceId,
        mut workspace: Workspace,
        address: u64,
        plan: &PcodePlan,
        flat: bool,
    ) -> Self {
        // A terminal label is the instruction's fall-through, not a block.
        workspace.labels.extend(
            (0..plan.labels().len())
                .map(|index| plan.is_terminal(LabelId::from_index(index)).then_some(next)),
        );
        Self {
            builder,
            base_storage,
            unique_space,
            unique_storage: workspace.unique_storage,
            dirty: workspace.dirty,
            has_local_blocks: (0..plan.labels().len())
                .any(|index| !plan.is_terminal(LabelId::from_index(index))),
            branches: workspace.branches,
            calls: workspace.calls,
            labels: workspace.labels,
            next,
            address,
            fallthrough: 0,
            flat,
            error: None,
        }
    }

    /// Closes the instruction. Running off the end of its p-code is a
    /// fall-through.
    fn finish(mut self) -> Result<(), LiftError> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        if !self.builder.is_terminated() {
            let site = self.builder.push_branch(self.next).id;
            self.builder
                .exit(site, ExitArm::Unconditional, ExitKind::Fallthrough);
        }
        Ok(())
    }

    /// Branches to the instruction's fall-through, the target of a local
    /// branch past the last operation.
    fn branch_next(&mut self, opcode: Opcode, condition: Option<Varnode>) {
        let next = self.next;
        if let Some((site, arm)) = self.branch_to(opcode, next, condition) {
            self.builder.exit(site, arm, ExitKind::Fallthrough);
        }
    }

    fn block_for(&mut self, label: LabelId) -> BlockId {
        if let Some(block) = self.labels[label.index()] {
            return block;
        }
        let (address, index) = (self.address, label.index());
        let block = self.fresh_block(|| format!("pcode_{address:x}_{index}"));
        self.labels[label.index()] = Some(block);
        self.builder.block(block);
        block
    }

    /// Terminates the current block with a branch to `target`, conditionally
    /// if `opcode` says so, and continues in the not-taken block. Returns the
    /// branch and which of its arms reaches `target`, or nothing once the
    /// instruction has failed.
    fn branch_to(
        &mut self,
        opcode: Opcode,
        target: BlockId,
        condition: Option<Varnode>,
    ) -> Option<(InstructionId, ExitArm)> {
        let condition = match (opcode, condition) {
            (Opcode::CBranch, Some(condition)) => match self.raw_value(condition) {
                Ok(condition) => Some(self.ensure_bool(condition)),
                Err(error) => {
                    self.fail(error);
                    return None;
                }
            },
            (Opcode::CBranch, None) => {
                self.fail(LiftError::InvalidArity {
                    opcode,
                    expected: 2,
                    actual: 1,
                });
                return None;
            }
            _ => None,
        };
        Some(match condition {
            Some(condition) => {
                self.leave_block(true);
                let fallthrough = self.open_fallthrough();
                let site = self.builder.push_cbranch(condition, target, fallthrough).id;
                self.builder.switch_to_block(fallthrough);
                self.enter_block();
                (site, ExitArm::Taken)
            }
            None => {
                self.leave_block(false);
                (self.builder.push_branch(target).id, ExitArm::Unconditional)
            }
        })
    }

    fn fail(&mut self, error: LiftError) {
        self.error.get_or_insert(error);
    }

    fn open_fallthrough(&mut self) -> BlockId {
        let (address, index) = (self.address, self.fallthrough);
        let label = self.fresh_block(|| format!("pcode_fallthrough_{address:x}_{index}"));
        self.fallthrough += 1;
        self.builder.block(label);
        label
    }

    /// A new block of the host for this instruction's own control flow,
    /// named by `name` when the builder names things. The emitter keeps its
    /// own handle on every block it makes, so it never looks one up by name.
    fn fresh_block(&mut self, name: impl FnOnce() -> String) -> BlockId {
        if self.builder.naming() {
            self.builder.get_or_make_local_label(Cow::Owned(name()))
        } else {
            self.builder.push_anonymous_block()
        }
    }

    /// Opens a block for p-code that follows a terminator: the continuation
    /// of a call, or code only local branches reach.
    fn open_continuation(&mut self) {
        if !self.builder.is_terminated() {
            return;
        }
        let label = self.open_fallthrough();
        self.builder.switch_to_block(label);
        self.builder.continue_in(label);
        self.enter_block();
    }

    fn input(&mut self, op: &OpRef<'_>, index: usize) -> Result<ValueId, LiftError> {
        let value = self.input_varnode(op, index)?;
        self.value(value)
    }

    /// An input read as held, for the consumers that want a `bool`.
    fn raw_input(&mut self, op: &OpRef<'_>, index: usize) -> Result<ValueId, LiftError> {
        let value = self.input_varnode(op, index)?;
        self.raw_value(value)
    }

    fn input_varnode(&self, op: &OpRef<'_>, index: usize) -> Result<Varnode, LiftError> {
        op.inputs
            .get(index)
            .copied()
            .ok_or(LiftError::InvalidArity {
                opcode: op.opcode,
                expected: index + 1,
                actual: op.inputs.len(),
            })
    }

    /// Reads a varnode as an integer operand. A unique that holds a `bool`
    /// value — a comparison's result — is widened to its byte here, which is
    /// what its memory temporary would have held.
    fn value(&mut self, varnode: Varnode) -> Result<ValueId, LiftError> {
        let value = self.raw_value(varnode)?;
        let ty = self.builder.view().type_of(value);
        if self.builder.shr().types.is_bool(ty) {
            return Ok(self.builder.push_zext(value, 1).id());
        }
        Ok(value)
    }

    /// Reads a varnode as it is held: a `bool` stays a `bool`, for the
    /// consumers that want one.
    fn raw_value(&mut self, varnode: Varnode) -> Result<ValueId, LiftError> {
        if varnode.space == SPACE_CONST {
            return Ok(self.builder.shr().get_const(varnode.offset, varnode.size));
        }
        if varnode.space == self.unique_space {
            return Ok(self.read_unique(varnode));
        }
        let value = self.base_storage(varnode)?;
        Ok(self.builder.ensure_local(value))
    }

    /// Resolves an architectural varnode's QCode location.
    fn base_storage(&self, varnode: Varnode) -> Result<ValueId, LiftError> {
        self.base_storage
            .get(&varnode)
            .copied()
            .map(ValueId::Varnode)
            .ok_or(LiftError::UnknownVarnode(varnode))
    }

    /// The SSA value of a unique: the one last written in this block, else a
    /// load of its temporary, which is made here if the unique is read before
    /// it is ever written. Each unique varnode — by space, offset and size —
    /// has its own slot, so overlapping uniques stay isolated as their storage
    /// identity requires.
    fn read_unique(&mut self, varnode: Varnode) -> ValueId {
        let slot = self.unique_storage.entry(varnode).or_default();
        if let Some(value) = slot.value {
            return value;
        }
        let temp = *slot
            .temp
            .get_or_insert_with(|| self.builder.make_temp(varnode.size));
        let value = self.builder.ensure_local(ValueId::Temp(temp));
        let slot = self
            .unique_storage
            .get_mut(&varnode)
            .expect("the slot was just made");
        slot.value = Some(value);
        slot.dirty = false;
        value
    }

    fn write(&mut self, output: Option<Varnode>, value: ValueId) -> Result<(), LiftError> {
        let Some(output) = output else {
            return Ok(());
        };
        if output.space == SPACE_CONST {
            return Err(LiftError::UnknownVarnode(output));
        }
        if output.space == self.unique_space {
            let slot = self.unique_storage.entry(output).or_default();
            slot.value = Some(value);
            if !slot.dirty {
                slot.dirty = true;
                self.dirty.push(output);
            }
            return Ok(());
        }
        let destination = self.base_storage(output)?;
        self.builder.push_copy(value, destination);
        Ok(())
    }

    /// Stores every unique written since the last spill to its temporary, so
    /// a block the control flow reaches next can load it.
    fn spill(&mut self) {
        for varnode in std::mem::take(&mut self.dirty) {
            let slot = self
                .unique_storage
                .get_mut(&varnode)
                .expect("a dirty unique has a slot");
            let value = slot.value.expect("a dirty unique holds a value");
            slot.dirty = false;
            let temp = *slot
                .temp
                .get_or_insert_with(|| self.builder.make_temp(varnode.size));
            self.builder.push_copy(value, ValueId::Temp(temp));
        }
    }

    /// Prepares to end the current block with a transfer. Uniques must be in
    /// memory if p-code of this instruction can run afterwards: always when
    /// the transfer `continues` in a block of its own (a conditional branch's
    /// fall-through, a call's continuation), otherwise only when a label can
    /// bring control back.
    fn leave_block(&mut self, continues: bool) {
        if continues || self.has_local_blocks {
            self.spill();
        }
    }

    /// Notes that a new block is current: values held from the previous one
    /// are not usable in it, so the next read of each unique loads it.
    fn enter_block(&mut self) {
        debug_assert!(self.dirty.is_empty(), "a block was left without a spill");
        for slot in self.unique_storage.values_mut() {
            slot.value = None;
        }
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
                let input = self.raw_input(op, 0)?;
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
                self.leave_block(false);
                let site = self.builder.push_branchind(target).id;
                self.builder
                    .exit(site, ExitArm::Unconditional, ExitKind::BranchInd);
                Ok(())
            }
            Call => {
                let target = op.inputs.first().ok_or(LiftError::InvalidArity {
                    opcode: op.opcode,
                    expected: 1,
                    actual: 0,
                })?;
                // The return address is already on the guest's stack by now —
                // SLEIGH wrote it with ordinary p-code — so in flat mode all
                // that is left of the call is the jump.
                self.leave_block(!self.flat);
                let site = if self.flat {
                    let block = self
                        .branches
                        .get(&target.offset)
                        .copied()
                        .ok_or(LiftError::InvalidDirectTarget(op.opcode))?;
                    self.builder.push_branch(block).id
                } else {
                    let callee = self
                        .calls
                        .get(&target.offset)
                        .copied()
                        .ok_or(LiftError::InvalidDirectTarget(op.opcode))?;
                    self.builder.push_call(callee).id
                };
                self.builder.exit(
                    site,
                    ExitArm::Unconditional,
                    ExitKind::Call {
                        callee: CallTarget::Address(target.offset),
                        continuation: Continuation::Next,
                    },
                );
                Ok(())
            }
            CallInd => {
                let target = self.input(op, 0)?;
                self.leave_block(!self.flat);
                let site = if self.flat {
                    self.builder.push_branchind(target).id
                } else {
                    self.builder.push_call_ind(target).id
                };
                self.builder.exit(
                    site,
                    ExitArm::Unconditional,
                    ExitKind::CallInd {
                        continuation: Continuation::Next,
                    },
                );
                Ok(())
            }
            Return => {
                let target = self.input(op, 0)?;
                // The stack has already been popped into this value; returning
                // is a jump to it.
                self.leave_block(false);
                let site = if self.flat {
                    self.builder.push_branchind(target).id
                } else {
                    self.builder.push_return(target).id
                };
                self.builder
                    .exit(site, ExitArm::Unconditional, ExitKind::Return);
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
        let lhs = self.raw_input(op, 0)?;
        let lhs = self.ensure_bool(lhs);
        let rhs = self.raw_input(op, 1)?;
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
        if let Some((site, arm)) = self.branch_to(op.opcode, block, condition) {
            self.builder.exit(
                site,
                arm,
                ExitKind::Branch {
                    target: target.offset,
                },
            );
        }
        Ok(())
    }
}

impl Drop for FlatEmitter<'_, '_, '_> {
    fn drop(&mut self) {
        return_workspace(Workspace {
            unique_storage: std::mem::take(&mut self.unique_storage),
            dirty: std::mem::take(&mut self.dirty),
            branches: std::mem::take(&mut self.branches),
            calls: std::mem::take(&mut self.calls),
            labels: std::mem::take(&mut self.labels),
        });
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
            self.spill();
            self.builder.push_branch(block);
        }
        self.builder.switch_to_block(block);
        self.builder.continue_in(block);
        self.enter_block();
    }

    fn branch_label(&mut self, opcode: Opcode, label: LabelId, condition: Option<Varnode>) {
        if self.error.is_some() {
            return;
        }
        self.open_continuation();
        self.builder.set_address(self.address);
        let target = self.block_for(label);
        if target == self.next {
            // A branch to the terminal label leaves the instruction.
            self.branch_next(opcode, condition);
        } else {
            self.branch_to(opcode, target, condition);
        }
        self.builder.clear_address();
    }
}

#[cfg(test)]
mod tests {
    use super::{FlatPcode, SleighLifter};
    use qcode::address_index::AddressIndex;
    use qcode::lift::{CallTarget, Continuation, Exit, ExitArm, ExitKind, Lifted};
    use qcode::value::{BasicBlock, Instruction};
    use qcode_emulator::Emulator;
    use sleigh::{CompiledSpec, Compiler, Decoder, SourceDb};

    /// Lifts one x86-64 instruction at 0x1000 in both lowering modes.
    fn lift_x64_both_modes(bytes: &[u8]) -> [(qcode::context::Context<'static>, Lifted); 2] {
        let spec = sleigh_precompile::x64::spec();
        [
            SleighLifter::new(spec),
            SleighLifter::new(spec).with_flat_control_flow(),
        ]
        .map(|lifter| {
            let instruction = Decoder::new(spec)
                .decode_one(0x1000, bytes, &spec.new_context())
                .unwrap();
            let mut ctx = lifter.new_context();
            let lifted = lifter
                .lift_instruction(&mut ctx, &instruction, None)
                .unwrap();
            (ctx, lifted)
        })
    }

    fn kinds(lifted: &Lifted) -> Vec<(ExitArm, ExitKind)> {
        lifted
            .exits()
            .iter()
            .map(|exit| (exit.arm(), exit.kind().clone()))
            .collect()
    }

    /// The opcode of the instruction an exit is recorded at.
    fn site_opcode(ctx: &qcode::context::Context<'static>, exit: &Exit) -> &'static str {
        Instruction::from_id(ctx, exit.site()).mnemonic().opcode()
    }

    fn tiny_spec(constructors: &str) -> CompiledSpec {
        let mut sources = SourceDb::new();
        let root = sources.add_file(
            "tiny.slaspec",
            format!(
                "define endian=little;
                 define space ram type=ram_space size=4 default;
                 define space register type=register_space size=4;
                 define register offset=0 size=4 [ r0 ];
                 define token instr(8) op=(0,7);
                 {constructors}"
            ),
        );
        Compiler::new(&mut sources).compile(root).unwrap()
    }

    fn lift_tiny(
        spec: &CompiledSpec,
        byte: u8,
        flat: bool,
    ) -> (qcode::context::Context<'static>, Lifted) {
        let bytes = [byte];
        let instruction = Decoder::new(spec)
            .decode_one(0x1000, &bytes, &spec.new_context())
            .unwrap();
        let mut lifter = SleighLifter::new(spec);
        if flat {
            lifter = lifter.with_flat_control_flow();
        }
        let mut ctx = lifter.new_context();
        let lifted = lifter
            .lift_instruction(&mut ctx, &instruction, None)
            .unwrap();
        (ctx, lifted)
    }

    #[test]
    fn a_jump_to_its_own_address_is_a_machine_transfer() {
        for (bytes, name) in [(b"\xeb\xfe".as_slice(), "jmp $"), (b"\xf4", "hlt")] {
            for (_, lifted) in lift_x64_both_modes(bytes) {
                assert_eq!(
                    kinds(&lifted),
                    [(ExitArm::Unconditional, ExitKind::Branch { target: 0x1000 })],
                    "{name}"
                );
                assert_eq!(lifted.blocks(), &[lifted.entry()], "{name}");
                assert!(!lifted.falls_through(), "{name}");
            }
        }
    }

    #[test]
    fn a_conditional_jump_reports_its_taken_arm_and_the_fallthrough() {
        // `jz +5`, then the same with a branch-hint prefix, one byte longer.
        for (bytes, target, next) in [
            (b"\x74\x05".as_slice(), 0x1007, 0x1002),
            (b"\x2e\x74\x05", 0x1008, 0x1003),
        ] {
            for (ctx, lifted) in lift_x64_both_modes(bytes) {
                assert_eq!(lifted.next_address(), next);
                assert_eq!(
                    kinds(&lifted),
                    [
                        (ExitArm::Taken, ExitKind::Branch { target }),
                        (ExitArm::Unconditional, ExitKind::Fallthrough),
                    ],
                    "{bytes:02x?}"
                );
                assert_eq!(site_opcode(&ctx, &lifted.exits()[0]), "cbranch");
                assert_eq!(site_opcode(&ctx, &lifted.exits()[1]), "branch");
                // The entry and the not-taken block; the target and the next
                // instruction's placeholders are not the instruction's.
                assert_eq!(lifted.blocks().len(), 2, "{bytes:02x?}");
                assert!(lifted.falls_through());
            }
        }
    }

    #[test]
    fn calls_and_returns_keep_their_kind_under_flat_control_flow() {
        let cases: [(&[u8], ExitKind); 4] = [
            (
                b"\xe8\x10\x00\x00\x00",
                ExitKind::Call {
                    callee: CallTarget::Address(0x1015),
                    continuation: Continuation::Next,
                },
            ),
            (
                b"\xff\xd0",
                ExitKind::CallInd {
                    continuation: Continuation::Next,
                },
            ),
            (b"\xc3", ExitKind::Return),
            (b"\xff\xe0", ExitKind::BranchInd),
        ];
        for (bytes, kind) in cases {
            let [(structured, lifted), (flat, flat_lifted)] = lift_x64_both_modes(bytes);
            assert_eq!(kinds(&lifted), [(ExitArm::Unconditional, kind.clone())]);
            assert_eq!(kinds(&flat_lifted), kinds(&lifted), "{bytes:02x?}");
            assert!(!lifted.falls_through());

            let expected = match kind {
                ExitKind::Call { .. } => ("call", "branch"),
                ExitKind::CallInd { .. } => ("callind", "branchind"),
                ExitKind::Return => ("return", "branchind"),
                _ => ("branchind", "branchind"),
            };
            assert_eq!(site_opcode(&structured, &lifted.exits()[0]), expected.0);
            assert_eq!(site_opcode(&flat, &flat_lifted.exits()[0]), expected.1);
        }
    }

    #[test]
    fn rep_movsb_leaves_by_its_count_check_and_returns_to_itself() {
        for (_, lifted) in lift_x64_both_modes(b"\xf3\xa4") {
            assert_eq!(
                kinds(&lifted),
                [
                    (ExitArm::Taken, ExitKind::Branch { target: 0x1002 }),
                    (ExitArm::Unconditional, ExitKind::Branch { target: 0x1000 }),
                ]
            );
            assert!(lifted.falls_through(), "an exhausted count continues");
        }
    }

    #[test]
    fn a_call_followed_by_pcode_continues_in_the_instruction() {
        let spec = tiny_spec(":callx is op=3 { call 0x2000; r0 = 1:4; }");
        for flat in [false, true] {
            let (ctx, lifted) = lift_tiny(&spec, 3, flat);
            let [call, fallthrough] = lifted.exits() else {
                panic!("{:?}", lifted.exits());
            };
            let ExitKind::Call {
                callee: CallTarget::Address(0x2000),
                continuation: Continuation::Block(block),
            } = call.kind().clone()
            else {
                panic!("{call:?}");
            };
            assert_eq!(fallthrough.kind(), &ExitKind::Fallthrough);
            assert_eq!(lifted.blocks(), &[lifted.entry(), block]);
            // The continuation holds the rest of the p-code and leaves by the
            // fall-through.
            let continuation = BasicBlock::from_id(&ctx, block);
            assert_eq!(
                Instruction::from_id(&ctx, fallthrough.site())
                    .block()
                    .map(|b| b.id),
                Some(continuation.id)
            );
        }
    }

    #[test]
    fn a_conditional_branch_to_the_terminal_label_is_a_taken_fallthrough() {
        let spec = tiny_spec(":cset is op=4 { if (r0 == 0:4) goto <skip>; r0 = 1:4; <skip> }");
        let (ctx, lifted) = lift_tiny(&spec, 4, false);
        assert_eq!(
            kinds(&lifted),
            [
                (ExitArm::Taken, ExitKind::Fallthrough),
                (ExitArm::Unconditional, ExitKind::Fallthrough),
            ]
        );
        assert_eq!(site_opcode(&ctx, &lifted.exits()[0]), "cbranch");
        assert_eq!(lifted.blocks().len(), 2);
    }

    #[test]
    fn a_failed_instruction_leaves_the_context_as_it_was() {
        let spec = tiny_spec(
            ":bad is op=6 { r0 = 1:4; r0 = newobject(r0); }
             :good is op=7 { r0 = 2:4; }",
        );
        let instruction = Decoder::new(&spec)
            .decode_one(0x1000, &[6], &spec.new_context())
            .unwrap();
        assert!(
            instruction
                .pcode_ops()
                .unwrap()
                .ops
                .iter()
                .any(|op| op.opcode == sleigh::Opcode::New),
            "the fixture relies on NEW being unsupported"
        );
        let lifter = SleighLifter::new(&spec);
        let mut ctx = lifter.new_context();
        let function = ctx.anon_function();
        let mut addresses = AddressIndex::analyze(&ctx);
        let before = ctx.to_string();

        let error = lifter
            .lift_instruction_indexed(&mut ctx, &mut addresses, &instruction, Some(function))
            .unwrap_err();
        assert_eq!(
            error,
            super::LiftError::UnsupportedOpcode(sleigh::Opcode::New)
        );
        assert_eq!(
            ctx.to_string(),
            before,
            "the partial instruction was undone"
        );
        assert!(addresses.is_empty(), "no placeholder survived");

        // The address lifts afterwards, and so does another instruction.
        let good = Decoder::new(&spec)
            .decode_one(0x1000, &[7], &spec.new_context())
            .unwrap();
        let lifted = lifter
            .lift_instruction_indexed(&mut ctx, &mut addresses, &good, Some(function))
            .unwrap();
        assert!(lifted.falls_through());
        assert_eq!(addresses.block_at(0x1000), Some(lifted.entry()));
    }

    #[test]
    fn an_address_is_lifted_once() {
        let spec = sleigh_precompile::x64::spec();
        let lifter = SleighLifter::new(spec);
        let instruction = Decoder::new(spec)
            .decode_one(0x1000, b"\x48\x89\xd8", &spec.new_context())
            .unwrap();
        let mut ctx = lifter.new_context();
        let function = ctx.anon_function();
        let mut addresses = AddressIndex::analyze(&ctx);
        let lifted = lifter
            .lift_instruction_indexed(&mut ctx, &mut addresses, &instruction, Some(function))
            .unwrap();
        let again = lifter
            .lift_instruction_indexed(&mut ctx, &mut addresses, &instruction, Some(function))
            .unwrap_err();
        assert_eq!(
            again,
            super::LiftError::Target(qcode::lift::TargetError::AlreadyLifted {
                address: 0x1000,
                block: lifted.entry()
            })
        );
        // The fall-through placeholder is empty and takes its instruction.
        let next = Decoder::new(spec)
            .decode_one(0x1003, b"\x48\x89\xd8", &spec.new_context())
            .unwrap();
        lifter
            .lift_instruction_indexed(&mut ctx, &mut addresses, &next, Some(function))
            .unwrap();
    }

    #[test]
    fn a_context_of_another_specification_is_refused() {
        let tiny = tiny_spec(":set is op=1 { r0 = 1:4; }");
        let x64 = sleigh_precompile::x64::spec();
        let instruction = Decoder::new(x64)
            .decode_one(0x1000, b"\x48\x89\xd8", &x64.new_context())
            .unwrap();
        let mut ctx = SleighLifter::new(&tiny).new_context();
        assert_eq!(
            SleighLifter::new(x64)
                .lift_instruction(&mut ctx, &instruction, None)
                .unwrap_err(),
            super::LiftError::IncompatibleContext
        );
        assert_eq!(ctx.functions().count(), 0, "nothing was created first");
    }

    #[test]
    fn an_instruction_of_another_specification_is_refused() {
        // A compatible destination, but the instruction was decoded by x86, so
        // its varnodes do not name this x64 lifter's registers.
        let x64 = sleigh_precompile::x64::spec();
        let x86 = sleigh_precompile::x86::spec();
        let instruction = Decoder::new(x86)
            .decode_one(0x1000, b"\x89\xd8", &x86.new_context())
            .unwrap();
        let lifter = SleighLifter::new(x64);
        let mut ctx = lifter.new_context();
        assert_eq!(
            lifter
                .lift_instruction(&mut ctx, &instruction, None)
                .unwrap_err(),
            super::LiftError::IncompatibleSpec
        );
        assert_eq!(ctx.functions().count(), 0, "nothing was created first");
    }

    #[test]
    fn a_look_alike_specification_is_refused_before_anything_is_mutated() {
        // Two specifications with the same spaces and the same register at the
        // same place, one constructor's semantics apart. Structurally they are
        // indistinguishable; by identity they are not the same specification,
        // and neither's lifter accepts the other's context, instruction or
        // flat p-code.
        let first = tiny_spec(":set is op=1 { r0 = 1:4; }");
        let alike = tiny_spec(":set is op=1 { r0 = 2:4; }");
        assert_ne!(first.fingerprint(), alike.fingerprint());
        let lifter = SleighLifter::new(&first);
        let other = SleighLifter::new(&alike);

        let mut ctx = other.new_context();
        let before = (ctx.to_string(), ctx.revision());
        let instruction = Decoder::new(&first)
            .decode_one(0x1000, &[1], &first.new_context())
            .unwrap();
        assert_eq!(
            lifter.lift_instruction(&mut ctx, &instruction, None).err(),
            Some(super::LiftError::IncompatibleContext)
        );
        assert_eq!(
            (ctx.to_string(), ctx.revision()),
            before,
            "refused untouched"
        );

        let mut ctx = lifter.new_context();
        let before = (ctx.to_string(), ctx.revision());
        let foreign = Decoder::new(&alike)
            .decode_one(0x1000, &[1], &alike.new_context())
            .unwrap();
        assert_eq!(
            lifter.lift_instruction(&mut ctx, &foreign, None).err(),
            Some(super::LiftError::IncompatibleSpec)
        );
        let flat = FlatPcode::lower(&foreign).unwrap();
        let mut addresses = AddressIndex::analyze(&ctx);
        assert_eq!(
            lifter
                .lift_pcode_indexed(&mut ctx, &mut addresses, &flat, None)
                .err(),
            Some(super::LiftError::IncompatibleSpec)
        );
        assert_eq!(
            (ctx.to_string(), ctx.revision()),
            before,
            "refused untouched"
        );
        // And through a bound target.
        let function = ctx.anon_function();
        let before = ctx.to_string();
        let mut addresses = AddressIndex::analyze(&ctx);
        let mut target =
            qcode::lift::LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
        assert_eq!(
            lifter.lift_pcode_into(&mut target, &flat).err(),
            Some(super::LiftError::IncompatibleSpec)
        );
        let _ = target;
        assert_eq!(ctx.to_string(), before, "refused untouched");
    }

    #[test]
    fn cached_pcode_is_trusted_on_the_fingerprint_it_claims() {
        // The provenance of cached p-code is the caller's claim: the one
        // honest way to lift cached p-code is to store the fingerprint with
        // it, and p-code of a look-alike specification passes under a
        // forged one.
        let first = tiny_spec(":set is op=1 { r0 = 1:4; }");
        let alike = tiny_spec(":set is op=1 { r0 = 2:4; }");
        let lifter = SleighLifter::new(&first);
        let foreign = Decoder::new(&alike)
            .decode_one(0x1000, &[1], &alike.new_context())
            .unwrap();
        let flat = FlatPcode::lower(&foreign).unwrap();
        let claimed = FlatPcode::from_parts(
            first.fingerprint(),
            flat.address(),
            flat.length(),
            flat.pcode().clone(),
        );
        let mut ctx = lifter.new_context();
        let mut addresses = AddressIndex::analyze(&ctx);
        assert_eq!(
            lifter
                .lift_pcode_indexed(&mut ctx, &mut addresses, &flat, None)
                .err(),
            Some(super::LiftError::IncompatibleSpec)
        );
        lifter
            .lift_pcode_indexed(&mut ctx, &mut addresses, &claimed, None)
            .unwrap();
    }

    #[test]
    fn the_stamp_is_the_specifications_identity_and_survives_a_clone_and_a_reload() {
        let first = tiny_spec(":set is op=1 { r0 = 1:4; }");
        let lifter = SleighLifter::new(&first);
        let instruction = Decoder::new(&first)
            .decode_one(0x1000, &[1], &first.new_context())
            .unwrap();

        // A hand-built module has no stamp, and no lifter accepts it.
        let mut bare = qcode::context::Context::new();
        assert_eq!(
            lifter.lift_instruction(&mut bare, &instruction, None).err(),
            Some(super::LiftError::IncompatibleContext)
        );

        // Identity, not instance: another lifter for the same specification
        // builds an acceptable context.
        let twin = SleighLifter::new(&first);
        let mut ctx = twin.new_context();
        lifter
            .lift_instruction(&mut ctx, &instruction, None)
            .unwrap();
        let mut cloned = ctx.clone();
        let next = Decoder::new(&first)
            .decode_one(0x1001, &[1], &first.new_context())
            .unwrap();
        lifter.lift_instruction(&mut cloned, &next, None).unwrap();
        let bytes = bincode::serde::encode_to_vec(&ctx, bincode::config::standard()).unwrap();
        let (mut reloaded, _): (qcode::context::Context<'static>, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(reloaded.architecture(), ctx.architecture());
        lifter.lift_instruction(&mut reloaded, &next, None).unwrap();
    }

    #[test]
    fn a_function_of_another_context_is_refused() {
        let spec = sleigh_precompile::x64::spec();
        let lifter = SleighLifter::new(spec);
        let instruction = Decoder::new(spec)
            .decode_one(0x1000, b"\x48\x89\xd8", &spec.new_context())
            .unwrap();
        let mut other = lifter.new_context();
        other.anon_function();
        let foreign = other.anon_function();
        let mut ctx = lifter.new_context();
        assert_eq!(
            lifter
                .lift_instruction(&mut ctx, &instruction, Some(foreign))
                .unwrap_err(),
            super::LiftError::Target(qcode::lift::TargetError::UnknownFunction(foreign))
        );
    }

    #[test]
    fn an_internal_loop_has_no_exits() {
        let spec = tiny_spec(":spin is op=5 { <again> r0 = r0 + 1:4; goto <again>; }");
        let (_, lifted) = lift_tiny(&spec, 5, false);
        assert!(lifted.exits().is_empty(), "{:?}", lifted.exits());
        assert!(!lifted.falls_through());
        assert_eq!(lifted.blocks().len(), 2, "the entry and the loop block");
    }

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
            let streamed_lifted = lifter
                .lift_instruction(&mut streamed, &instruction, None)
                .unwrap();

            let pcode = FlatPcode::lower(&instruction).unwrap();
            let mut flat = lifter.new_context();
            let mut addresses = AddressIndex::analyze(&flat);
            let flat_lifted = lifter
                .lift_pcode_indexed(&mut flat, &mut addresses, &pcode, None)
                .unwrap();

            assert_eq!(
                streamed.to_string(),
                flat.to_string(),
                "{bytes:02x?} lifts differently"
            );
            assert_eq!(
                streamed_lifted, flat_lifted,
                "{bytes:02x?} reports different results"
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

    /// A specification whose `r0` and `r1` are 4-byte registers, with one
    /// instruction per test of the unique-space lowering.
    fn unique_spec() -> CompiledSpec {
        let mut sources = SourceDb::new();
        let root = sources.add_file(
            "uniques.slaspec",
            "define endian=little;
             define space ram type=ram_space size=4 default;
             define space register type=register_space size=4;
             define register offset=0 size=4 [ r0 r1 ];
             define token instr(8) op=(0,7);
             :chain is op=2 { local t:4 = r0 + 1:4; r0 = t * 2:4; }
             :cross is op=3 {
                 local t:4 = r0 + 1:4;
                 if (r0 == 0:4) goto <done>;
                 r0 = t;
                 <done>
                 r1 = t;
             }",
        );
        Compiler::new(&mut sources).compile(root).unwrap()
    }

    fn register(spec: &CompiledSpec, name: &str) -> sleigh::RegisterId {
        spec.registers()
            .find(|register| register.name() == name)
            .unwrap()
            .id
    }

    #[test]
    fn uniques_read_in_their_own_block_are_values_not_memory() {
        let spec = unique_spec();
        let instruction = Decoder::new(&spec)
            .decode_one(0x1000, &[2], &spec.new_context())
            .unwrap();
        let lifter = SleighLifter::new(&spec);
        let mut ctx = lifter.new_context();
        lifter
            .lift_instruction(&mut ctx, &instruction, None)
            .unwrap();
        // Register load, add, multiply, register copy, and machine
        // fall-through: the unique `t` is the add's result, never stored.
        assert_eq!(ctx.instructions().count(), 5, "{ctx}");

        let mut emulator = Emulator::from_address(&ctx, 0x1000);
        emulator.set_register(register(&spec, "r0"), 5).unwrap();
        while emulator.block().address() != Some(0x1001) {
            emulator.step().unwrap();
        }
        assert_eq!(emulator.read_register(register(&spec, "r0")), Some(12));
    }

    #[test]
    fn uniques_read_across_a_block_boundary_go_through_memory() {
        let spec = unique_spec();
        let instruction = Decoder::new(&spec)
            .decode_one(0x1000, &[3], &spec.new_context())
            .unwrap();
        let lifter = SleighLifter::new(&spec);
        let mut ctx = lifter.new_context();
        lifter
            .lift_instruction(&mut ctx, &instruction, None)
            .unwrap();
        // The entry block: two register loads, the add, the compare, a store
        // of `t` and of the condition to their temporaries, and the branch.
        // Then the not-taken block and the label's block each load `t`, store
        // it to a register, and jump.
        assert_eq!(ctx.instructions().count(), 13, "{ctx}");

        for (r0, expected_r0, expected_r1) in [(5, 6, 6), (0, 0, 1)] {
            let mut emulator = Emulator::from_address(&ctx, 0x1000);
            emulator.set_register(register(&spec, "r0"), r0).unwrap();
            while emulator.block().address() != Some(0x1001) {
                emulator.step().unwrap();
            }
            assert_eq!(
                emulator.read_register(register(&spec, "r0")),
                Some(expected_r0)
            );
            assert_eq!(
                emulator.read_register(register(&spec, "r1")),
                Some(expected_r1)
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
