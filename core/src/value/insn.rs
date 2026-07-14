//! Core SSA value type: instructions. An instruction is a value that is defined by an operation and can be used by other instructions.
//! Each instruction has a mnemonic, which is the operation that it performs, and a size
//! in bytes of the value it defines.
//! Instructions that do not define a value (e.g. terminators) have a size of 0.
use crate::{
    context::Context,
    error::Result,
    space::{Space, SpaceId, SpaceRef, SpaceType},
    types::TypeId,
    value::{
        BlockId, BlockRef, FunctionId, FunctionRef, LocalBlockId, ModuleView, QCodeView, Value,
        ValueId,
        util::{
            base_ref::{BaseRef, HostRef, WithCtx, WithCtxMut, WithHost},
            named::{Named, Renameable, update_context_name},
        },
    },
};
use jstd::Identifier;
use std::{
    borrow::Cow,
    fmt::{Display, Formatter},
    marker::PhantomData,
};

mod aggregate;
mod assert;
mod binop;
mod bits;
mod casting;
mod flags;
pub(crate) mod intrinsic;
mod map;
mod memory;
mod mnemonic;
mod pcode_op;
mod scan;
pub mod segment;
mod terminator;
mod unop;

pub use aggregate::{Extract, Gep, Tuple};
pub use assert::Assert;
pub use binop::{Binary, Binop, FloatBinop, IntBinop};
pub use casting::{FloatToFloat, FloatToInt, IntToFloat, Range, Sext, Zext};
pub use flags::{Carry, IsFloatNaN, LzCount, PopCount, SBorrow, SCarry};
pub use intrinsic::{
    Intrinsic, IntrinsicApp, IntrinsicId, IntrinsicRegistration, RootOp, Simplified,
    recognizers_for,
};
pub use map::Map;
pub use memory::{Load, Store};
pub use mnemonic::Mnemonic;
pub use pcode_op::{PCodeOp, PCodeOpId};
pub use scan::Scan;
pub use terminator::{
    Apply, Branch, BranchInd, CBranch, Call, CallInd, Callee, Return, ReturnValue, TailCall,
};
pub use unop::{Unary, Unop};

/// Function-local instruction index. Storage detail: indexes the owning
/// [`FunctionBody`](crate::value::FunctionBody)'s instruction arena. Pass composite
/// [`InstructionId`]s around in pass code, not these.
#[derive(Identifier)]
pub struct LocalInsnId(u32);

crate::composite_id!(InstructionId, LocalInsnId);

/// A local SSA value, which is a value that is defined by an instruction and can be used by other instructions.
/// Local values are not associated with any particular memory location.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Instruction<'str> {
    /// The name of this instruction
    pub(crate) name: Option<Cow<'str, str>>,

    /// The type of this instruction's result value (encodes size and semantic kind).
    pub(crate) type_id: TypeId,

    /// The instruction which defines this value.
    mnemonic: Mnemonic,

    /// The block that this instruction belongs to, if any (bare body-local index;
    /// strict IR locality means the parent block lives in the same arena as the
    /// instruction, so its owning `FunctionId` is the instruction's own `id.func`).
    /// Instructions that are not part of any block (e.g. lifted from data sections) have `None` here.
    pub(crate) parent: Option<LocalBlockId>,

    // Address of the binary instruction
    address: Option<u64>,

    _marker: std::marker::PhantomData<&'str ()>,
}

impl<'str> Instruction<'str> {
    pub(crate) fn new(type_id: TypeId, mnemonic: Mnemonic) -> Self {
        Self {
            name: None,
            parent: None,
            type_id,
            mnemonic,
            address: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn mnemonic(&self) -> &Mnemonic {
        &self.mnemonic
    }

    /// Mutable access to this instruction's mnemonic (crate-internal; used by the
    /// generic mutation host to rewrite operands).
    pub(crate) fn mnemonic_mut(&mut self) -> &mut Mnemonic {
        &mut self.mnemonic
    }

    /// Sets this instruction's machine address (crate-internal; used by the
    /// generic builder, which routes through the mutation host).
    pub(crate) fn set_address(&mut self, address: u64) {
        self.address = Some(address);
    }

    pub fn from_id<'ctx>(
        ctx: &'ctx Context<'str>,
        id: InstructionId,
    ) -> InstructionRef<'str, 'ctx> {
        InstructionRef::new(ModuleView::new(ctx), id)
    }

    pub fn from_id_mut<'ctx>(
        ctx: &'ctx mut Context<'str>,
        id: InstructionId,
    ) -> InstructionMutRef<'str, 'ctx> {
        InstructionMutRef::from_id(ctx, id)
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx, R> InstructionRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx Instruction<'str> {
        self.view.instruction(self.id)
    }

    /// The name of this instruction's output value
    pub fn name(&'s self) -> Option<&'ctx str> {
        self.inner().name.as_deref()
    }

    /// The [`TypeId`] of this instruction's result value.
    pub fn type_id(&'s self) -> TypeId {
        self.inner().type_id
    }

    /// The size in bytes of the instruction's output value
    pub fn size(&'s self) -> usize {
        self.view.shared().types.size_of(self.inner().type_id)
    }

    /// The basic block that this instruction belongs to, if any.
    pub fn parent(&'s self) -> Option<BlockRef<'str, 'ctx, R>> {
        self.inner()
            .parent
            .map(|local| BlockRef::new(self.view, BlockId::new(self.id.func, local)))
    }

    pub fn block(&'s self) -> Option<BlockRef<'str, 'ctx, R>> {
        self.parent()
    }

    /// The function that this instruction belongs to, if any.
    pub fn function(&'s self) -> Option<FunctionRef<'str, 'ctx, R>> {
        self.parent().and_then(|block| block.parent())
    }

    /// The mnemonic of the instruction
    pub fn mnemonic(&'s self) -> &'ctx Mnemonic {
        &self.inner().mnemonic
    }

    /// The operands consumed by this instruction, as qualified [`ValueId`]s.
    ///
    /// This is the func-qualifying, pass-facing operand accessor (stage 6a §11,
    /// option b): it is the routing target for the `insn.mnemonic().args()`
    /// call sites. Today it forwards `MnemonicKind::args`
    /// verbatim; once in-body operand storage flips to `LocalValueId`, only this
    /// body changes — it qualifies each local operand with the owning function
    /// (`self.id.func`), which a bare `&Mnemonic` cannot do — so every caller
    /// keeps seeing qualified `ValueId`s unchanged.
    pub fn operands(&'s self) -> smallvec::SmallVec<[ValueId; 2]> {
        let func = self.id.func;
        self.mnemonic()
            .args()
            .into_iter()
            .map(|v| v.qualify(func))
            .collect()
    }

    /// The address of the corresponding instruction
    pub fn address(&'s self) -> Option<u64> {
        self.inner().address
    }

    /// The address-space provenance for this instruction's result, if any.
    ///
    /// Returns `Some` only for instructions whose result type is a pointer to a
    /// known memory space (e.g. [`StackAddress`](crate::types::StackAddress)).
    pub fn space(&'s self) -> Option<SpaceRef<'ctx>> {
        self.view
            .shared()
            .types
            .space_of(self.inner().type_id)
            .map(|id| Space::from_id(self.view.shared(), id))
    }

    /// The opcode for this instruction
    pub fn opcode(&'s self) -> &'static str {
        self.mnemonic().opcode()
    }

    /// Is this instruction a terminator (i.e. does it end a basic block)?
    pub fn is_terminator(&'s self) -> bool {
        self.mnemonic().is_terminator()
    }

    fn fmt(&'s self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ty = self.view.shared().types.type_name(self.type_id());

        if let Some(name) = self.name() {
            write!(f, "{ty} %{name}")
        } else {
            // Function-local index: `%tmp{local}` is unique within a function,
            // which is the scope the parser resolves names in.
            let id: usize = self.id.local.into();
            write!(f, "{ty} %tmp{id:x}")
        }
    }
}

// Own-instruction mutations, emitted for each concrete mutation backing —
// `&mut Context` (module) and `PassBacking` (checked-out function pass) — so a
// `FunctionPass` can retype and rename the instructions it owns whether the
// function lives in the module registry or has been checked out. Mirror the
// `&mut Context`-only [`InstructionMutRef::set_type`] / `Renameable` impls.
macro_rules! impl_insn_mut_verbs {
    (<$($l:lifetime),*> $ctx:ty) => {
        impl<$($l),*> BaseRef<$ctx, InstructionId> {
    /// Sets this instruction's result type (own-instruction edit, host-routed).
    /// Panics on an incompatible same-nonzero-size change, exactly like
    /// [`InstructionMutRef::set_type`].
    pub fn set_result_type(&mut self, new_type: TypeId) {
        let current = self.ctx.read_host().instruction(self.id).type_id;
        let (current_size, new_size) = {
            let types = &self.ctx.shr().types;
            (types.size_of(current), types.size_of(new_type))
        };
        assert!(
            current_size == 0 || current_size == new_size,
            "cannot change instruction result type: size {current_size} → {new_size}",
        );
        self.ctx.instruction_mut(self.id).type_id = new_type;
    }

    /// Renames this instruction in its owning function's local name table
    /// (own-instruction edit, host-routed). Mirrors the `Renameable` impl for
    /// [`InstructionMutRef`]. Errors only on a duplicate name.
    pub fn rename_local(&mut self, name: Cow<'str, str>) -> Result<()> {
        let old_name = self
            .ctx
            .read_host()
            .instruction(self.id)
            .name
            .as_deref()
            .map(str::to_owned);
        self.ctx
            .register_local_name(self.id.into(), name.clone(), old_name.as_deref())?;
        self.ctx.instruction_mut(self.id).name = Some(name);
        Ok(())
    }
        }
    };
}

impl_insn_mut_verbs!(<'c, 'str> &'c mut Context<'str>);
impl_insn_mut_verbs!(<'a, 'str> crate::value::util::host_mut::PassBacking<'a, 'str>);

#[derive(Clone, Copy)]
pub struct InstructionRef<'str, 'ctx, R = ModuleView<'ctx, 'str>> {
    pub id: InstructionId,
    pub(in crate::value) view: R,
    marker: PhantomData<&'ctx &'str ()>,
}

impl<'str, 'ctx, R> InstructionRef<'str, 'ctx, R> {
    pub fn new(view: R, id: InstructionId) -> Self {
        Self {
            id,
            view,
            marker: PhantomData,
        }
    }

    pub fn id(&self) -> ValueId {
        self.id.into()
    }

    /// Format this instruction as a string, with the mnemonic and operands.
    pub fn as_statement(&self) -> InstructionStatement<'_, 'str, 'ctx, R> {
        InstructionStatement(self)
    }
}

impl<'str, 'ctx> InstructionRef<'str, 'ctx> {
    pub fn from_id(ctx: &'ctx Context<'str>, id: InstructionId) -> Self {
        Self::new(ModuleView::new(ctx), id)
    }

    /// Creates an instruction with a plain `Int(size)` result type, born into
    /// `func`'s instruction arena. The result is detached (`parent == None`)
    /// until a block appends it.
    pub fn from_mnemonic(
        ctx: &'ctx mut Context<'str>,
        func: FunctionId,
        mnemonic: Mnemonic,
        size: usize,
    ) -> Self {
        let type_id = ctx.shared.types.get_or_make_int(size);
        let insn = Instruction::new(type_id, mnemonic);
        let id = ctx.push_insn(func, insn);
        InstructionRef::new(ModuleView::new(ctx), id)
    }

    /// Creates an instruction with an explicit [`TypeId`], born into `func`.
    ///
    /// Pass a [`StackAddress`](crate::types::StackAddress) type id when the
    /// result is a stack-space pointer. Register-space provenance is silently
    /// demoted to `Int` (pointer arithmetic on registers is not meaningful).
    pub fn from_mnemonic_with_type(
        ctx: &'ctx mut Context<'str>,
        func: FunctionId,
        mnemonic: Mnemonic,
        type_id: TypeId,
    ) -> Self {
        let insn = Instruction::new(type_id, mnemonic);
        let id = ctx.push_insn(func, insn);
        InstructionRef::new(ModuleView::new(ctx), id)
    }

    /// Creates an instruction, deriving the result type from an optional space tag.
    ///
    /// This is a migration shim that types the result as `Int(size)` regardless of
    /// the `space` tag. New code should use [`from_mnemonic_with_type`] directly.
    pub fn from_mnemonic_with_space(
        ctx: &'ctx mut Context<'str>,
        func: FunctionId,
        mnemonic: Mnemonic,
        size: usize,
        _space: Option<SpaceId>,
    ) -> Self {
        let type_id = ctx.shared.types.get_or_make_int(size);
        let insn = Instruction::new(type_id, mnemonic);
        let id = ctx.push_insn(func, insn);
        InstructionRef::new(ModuleView::new(ctx), id)
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 'ctx, 'str> for InstructionRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'ctx Context<'str> {
        // Module-scope-only escape hatch: shared-only reads go through
        // `host().shr()`; only whole-module walks (callees/callers) reach here,
        // and those panic on a checked-out host by design (context-split Pin B).
        self.view.context()
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithHost<'s, 'ctx, 'str> for InstructionRef<'str, 'ctx> {
    fn host(&'s self) -> HostRef<'ctx, 'str> {
        HostRef::Module(self.view.context())
    }
}

impl<'str: 'ctx, 'ctx, R> Named for InstructionRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn name(&self) -> Option<&str> {
        self.view.instruction(self.id).name.as_deref()
    }
}

impl<'str: 'ctx, 'ctx, R> Display for InstructionRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        InstructionRef::fmt(self, f)
    }
}

impl<'str: 'ctx, 'ctx, R> Value<'str, 'ctx> for InstructionRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        InstructionRef::size(self)
    }
}

pub type InstructionMutRef<'str, 'ctx> = BaseRef<&'ctx mut Context<'str>, InstructionId>;

impl<'str, 'ctx> InstructionMutRef<'str, 'ctx> {
    pub fn as_ref(&self) -> InstructionRef<'str, '_> {
        InstructionRef::new(ModuleView::new(self.ctx), self.id)
    }

    fn inner(&self) -> &Instruction<'str> {
        self.ctx.instruction(self.id)
    }

    pub fn inner_mut(&mut self) -> &mut Instruction<'str> {
        self.ctx.instruction_mut(self.id)
    }

    pub fn mnemonic_mut(&mut self) -> &mut Mnemonic {
        &mut self.inner_mut().mnemonic
    }

    /// Replace this instruction's mnemonic while keeping the reverse use-def
    /// map in sync.
    pub fn set_mnemonic(&mut self, mnemonic: Mnemonic) {
        let old_args = self.inner().mnemonic.args();
        let new_args = mnemonic.args();

        // Operand uses are recorded in this instruction's own function map.
        let func = self.id.func;
        for arg in old_args {
            if let Some(users) = self.ctx.bodies[func].users.get_mut(&arg) {
                users.retain(|&local| local != self.id.localize(func));
            }
        }

        for arg in new_args {
            self.ctx.bodies[func]
                .users
                .entry(arg)
                .or_default()
                .push(self.id.localize(func));
        }

        self.inner_mut().mnemonic = mnemonic;
    }

    pub fn address_mut(&mut self) -> &mut Option<u64> {
        &mut self.inner_mut().address
    }

    pub fn set_address(&mut self, address: u64) {
        *self.address_mut() = Some(address);
    }

    /// Sets the type of this instruction's result.
    ///
    /// Panics if the instruction already has a type that is incompatible with
    /// `new_type` (same size but different kind).
    pub fn set_type(&mut self, new_type: TypeId) {
        let current = self.inner().type_id;
        let current_size = self.ctx.shared.types.size_of(current);
        let new_size = self.ctx.shared.types.size_of(new_type);
        if current_size != 0 && current_size != new_size {
            panic!(
                "Cannot change type of instruction {}: size {} → {}",
                self, current_size, new_size
            );
        }
        self.inner_mut().type_id = new_type;
    }

    /// Sets the result type, allowing the byte size to change.
    ///
    /// Unlike [`set_type`](Self::set_type), this does not enforce size
    /// invariance: it exists for transforms that legitimately resize an
    /// aggregate result, such as trimming dead fields from a returned write-set
    /// (`dead_signature`). Prefer `set_type` for any same-size retype.
    pub fn set_type_resized(&mut self, new_type: TypeId) {
        self.inner_mut().type_id = new_type;
    }

    /// Sets the address-space provenance of this instruction's result.
    ///
    /// A non-register space promotes the result to a
    /// [`SpaceAddress`](crate::types::SpaceAddress) of the same byte width.
    /// Register spaces are ignored (pointer arithmetic is not allowed in the
    /// register space).
    pub fn set_space(&mut self, space: SpaceId) {
        // Pointer arithmetic is not allowed in the register space.
        if matches!(
            Space::from_id(&self.ctx.shared, space).ty,
            SpaceType::Register
        ) {
            return;
        }
        let size = self.ctx.shared.types.size_of(self.inner().type_id);
        let type_id = self.ctx.shared.types.get_or_make_space_address(size, space);
        self.inner_mut().type_id = type_id;
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 's, 'str> for InstructionMutRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'s Context<'str> {
        self.ctx
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtxMut<'s, 'str> for InstructionMutRef<'str, 'ctx> {
    fn ctx_mut(&'s mut self) -> &'s mut Context<'str> {
        self.ctx
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithHost<'s, 's, 'str> for InstructionMutRef<'str, 'ctx> {
    fn host(&'s self) -> HostRef<'s, 'str> {
        HostRef::Module(self.ctx)
    }
}

impl Named for InstructionMutRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        self.ctx.instruction(self.id).name.as_deref()
    }
}

impl Display for InstructionMutRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for InstructionMutRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        self.as_ref().size()
    }
}

impl<'str, 'ctx> Renameable<'str, 'ctx> for InstructionMutRef<'str, 'ctx> {
    fn rename(&mut self, name: Cow<'str, str>) -> Result<()> {
        let id = self.id();
        let old_name = self.inner_mut().name.take();
        update_context_name(id, self.ctx_mut(), name.clone(), old_name.as_deref())?;
        self.inner_mut().name = Some(name);
        Ok(())
    }
}

/// A formattable wrapper around an instruction reference, which formats the instruction as a string with its mnemonic and operands.
pub struct InstructionStatement<'a, 'str, 'ctx, R = ModuleView<'ctx, 'str>>(
    &'a InstructionRef<'str, 'ctx, R>,
);

impl<'str: 'ctx, 'ctx, R> Display for InstructionStatement<'_, 'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // The statement's rendering (result binding + mnemonic) is defined once,
        // as tokens, in `segment`; the `Display` form is those tokens joined.
        for token in segment::instruction_segments(self.0) {
            write!(f, "{}", token.text)?;
        }
        Ok(())
    }
}
