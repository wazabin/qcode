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
        BasicBlock, BlockId, BlockRef, FunctionRef, Value, ValueId,
        util::{
            base_ref::{BaseRef, WithCtx, WithCtxMut},
            named::{Named, Renameable, update_context_name},
        },
    },
};
use jstd::Identifier;
use std::{
    borrow::Cow,
    fmt::{Display, Formatter},
};

mod aggregate;
mod assert;
mod binop;
mod bits;
mod casting;
mod flags;
pub(crate) mod intrinsic;
mod memory;
mod mnemonic;
mod pcode_op;
mod terminator;
mod unop;

pub use aggregate::{Extract, Tuple};
pub use assert::Assert;
pub use binop::{Binary, Binop, BoolBinop, FloatBinop, IntBinop};
pub use casting::{FloatToFloat, FloatToInt, IntToFloat, Range, Sext, Zext};
pub use flags::{Carry, IsFloatNaN, LzCount, PopCount, SBorrow, SCarry};
pub use intrinsic::{
    Intrinsic, IntrinsicDesc, IntrinsicId, IntrinsicRegistration, RootOp, Simplified,
    recognizers_for,
};
pub use memory::{Load, Store};
pub use mnemonic::Mnemonic;
pub use pcode_op::{PCodeOp, PCodeOpId};
pub use terminator::{Branch, BranchInd, CBranch, Call, CallInd, Return};
pub use unop::{Unary, Unop};

#[derive(Identifier)]
pub struct InstructionId(usize);

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

    /// The block that this instruction belongs to, if any.
    /// Instructions that are not part of any block (e.g. lifted from data sections) have `None` here.
    pub(crate) parent: Option<BlockId>,

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

    pub fn from_id<'ctx>(
        ctx: &'ctx Context<'str>,
        id: InstructionId,
    ) -> InstructionRef<'str, 'ctx> {
        InstructionRef::from_id(ctx, id)
    }

    pub fn from_id_mut<'ctx>(
        ctx: &'ctx mut Context<'str>,
        id: InstructionId,
    ) -> InstructionMutRef<'str, 'ctx> {
        InstructionMutRef::from_id(ctx, id)
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx, Ctx> BaseRef<Ctx, InstructionId>
where
    Self: WithCtx<'s, 'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx Instruction<'str> {
        &self.ctx().values.instructions[self.id]
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
        self.ctx().types.size_of(self.inner().type_id)
    }

    /// The basic block that this instruction belongs to, if any.
    pub fn parent(&'s self) -> Option<BlockRef<'str, 'ctx>> {
        self.inner()
            .parent
            .map(|id| BasicBlock::from_id(self.ctx(), id))
    }

    pub fn block(&'s self) -> Option<BlockRef<'str, 'ctx>> {
        self.parent()
    }

    /// The function that this instruction belongs to, if any.
    pub fn function(&'s self) -> Option<FunctionRef<'str, 'ctx>> {
        self.parent().and_then(|block| block.parent())
    }

    /// The mnemonic of the instruction
    pub fn mnemonic(&'s self) -> &'ctx Mnemonic {
        &self.inner().mnemonic
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
        self.ctx()
            .types
            .space_of(self.inner().type_id)
            .map(|id| Space::from_id(self.ctx(), id))
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
        let bit_size = self.size() * 8;

        if let Some(name) = self.name() {
            write!(f, "i{} %{}", bit_size, name)
        } else {
            let id: usize = self.id.into();
            write!(f, "i{} %tmp{:x}", bit_size, id)
        }
    }
}

pub type InstructionRef<'str, 'ctx> = BaseRef<&'ctx Context<'str>, InstructionId>;

impl<'str, 'ctx> InstructionRef<'str, 'ctx> {
    /// Creates an instruction with a plain `Int(size)` result type.
    pub fn from_mnemonic(ctx: &'ctx mut Context<'str>, mnemonic: Mnemonic, size: usize) -> Self {
        let type_id = ctx.types.get_or_make_int(size);
        let insn = Instruction::new(type_id, mnemonic);
        let id = ctx.values.push_insn(insn);
        Self::from_id(ctx, id)
    }

    /// Creates an instruction with an explicit [`TypeId`].
    ///
    /// Pass a [`StackAddress`](crate::types::StackAddress) type id when the
    /// result is a stack-space pointer. Register-space provenance is silently
    /// demoted to `Int` (pointer arithmetic on registers is not meaningful).
    pub fn from_mnemonic_with_type(
        ctx: &'ctx mut Context<'str>,
        mnemonic: Mnemonic,
        type_id: TypeId,
    ) -> Self {
        let insn = Instruction::new(type_id, mnemonic);
        let id = ctx.values.push_insn(insn);
        Self::from_id(ctx, id)
    }

    /// Creates an instruction, deriving the result type from an optional space tag.
    ///
    /// This is a migration shim: callers that still pass a `SpaceId` to indicate
    /// provenance get `StackAddress` for the stack space and `Int(size)` for
    /// everything else. New code should use [`from_mnemonic_with_type`] directly.
    pub fn from_mnemonic_with_space(
        ctx: &'ctx mut Context<'str>,
        mnemonic: Mnemonic,
        size: usize,
        space: Option<SpaceId>,
    ) -> Self {
        // Pointer arithmetic is not allowed in the register space; treat as Int.
        let effective_space =
            space.filter(|&s| !matches!(Space::from_id(ctx, s).ty, SpaceType::Register));

        let type_id = match effective_space {
            Some(sid)
                if ctx
                    .types
                    .stack_address_id()
                    .and_then(|sa| ctx.types.space_of(sa))
                    == Some(sid) =>
            {
                // The space matches the registered stack space → StackAddress
                ctx.types.stack_address_id().unwrap()
            }
            _ => ctx.types.get_or_make_int(size),
        };

        let insn = Instruction::new(type_id, mnemonic);
        let id = ctx.values.push_insn(insn);
        Self::from_id(ctx, id)
    }

    /// Format this instruction as a string, with the mnemonic and operands
    pub fn as_statement(&self) -> InstructionStatement<'_> {
        InstructionStatement(self)
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 'ctx, 'str> for InstructionRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'ctx Context<'str> {
        self.ctx
    }
}

impl Named for InstructionRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        self.ctx.values.instructions[self.id].name.as_deref()
    }
}

impl Display for InstructionRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt(f)
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for InstructionRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        self.size()
    }
}

pub type InstructionMutRef<'str, 'ctx> = BaseRef<&'ctx mut Context<'str>, InstructionId>;

impl<'str, 'ctx> InstructionMutRef<'str, 'ctx> {
    pub fn inner_mut(&mut self) -> &mut Instruction<'str> {
        &mut self.ctx.values.instructions[self.id]
    }

    pub fn mnemonic_mut(&mut self) -> &mut Mnemonic {
        &mut self.inner_mut().mnemonic
    }

    /// Replace this instruction's mnemonic while keeping the reverse use-def
    /// map in sync.
    pub fn set_mnemonic(&mut self, mnemonic: Mnemonic) {
        let old_args = self.inner().mnemonic.args();
        let new_args = mnemonic.args();

        for arg in old_args {
            if let Some(users) = self.ctx.values.users.get_mut(&arg) {
                users.retain(|&user| user != self.id);
            }
        }

        for arg in new_args {
            self.ctx.values.users.entry(arg).or_default().push(self.id);
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
        let current_size = self.ctx.types.size_of(current);
        let new_size = self.ctx.types.size_of(new_type);
        if current_size != 0 && current_size != new_size {
            panic!(
                "Cannot change type of instruction {}: size {} → {}",
                self, current_size, new_size
            );
        }
        self.inner_mut().type_id = new_type;
    }

    /// Sets the address-space provenance of this instruction's result.
    ///
    /// The stack space promotes the result to
    /// [`StackAddress`](crate::types::StackAddress); any other non-register
    /// space promotes it to a [`SpaceAddress`](crate::types::SpaceAddress) of
    /// the same byte width. Register spaces are ignored (pointer arithmetic is
    /// not allowed in the register space).
    pub fn set_space(&mut self, space: SpaceId) {
        // Pointer arithmetic is not allowed in the register space.
        if matches!(Space::from_id(self.ctx, space).ty, SpaceType::Register) {
            return;
        }
        // The stack space has dedicated `StackAddress` semantics.
        if let Some(sa_id) = self.ctx.types.stack_address_id()
            && self.ctx.types.space_of(sa_id) == Some(space)
        {
            self.inner_mut().type_id = sa_id;
            return;
        }
        let size = self.ctx.types.size_of(self.inner().type_id);
        let type_id = self.ctx.types.get_or_make_space_address(size, space);
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

impl Named for InstructionMutRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        self.ctx.values.instructions[self.id].name.as_deref()
    }
}

impl Display for InstructionMutRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt(f)
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for InstructionMutRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        self.size()
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
pub struct InstructionStatement<'a>(&'a InstructionRef<'a, 'a>);

impl Display for InstructionStatement<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.0.size() != 0 {
            write!(f, "{} = ", self.0)?;
        }

        self.0.mnemonic().fmt(f, self.0.ctx)
    }
}
