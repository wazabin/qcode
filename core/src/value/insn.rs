//! Core SSA value type: instructions. An instruction is a value that is defined by an operation and can be used by other instructions.
//! Each instruction has a mnemonic, which is the operation that it performs, and a size
//! in bytes of the value it defines.
//! Instructions that do not define a value (e.g. terminators) have a size of 0.
use crate::{
    context::Context,
    error::Result,
    space::{Space, SpaceId, SpaceRef, SpaceType},
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

mod binop;
mod bits;
mod casting;
mod flags;
mod memory;
mod mnemonic;
mod pcode_op;
mod terminator;
mod unop;

pub use binop::{Binary, Binop, BoolBinop, FloatBinop, IntBinop};
pub use casting::{FloatToFloat, FloatToInt, IntToFloat, Range, Sext, Zext};
pub use flags::{Carry, IsFloatNaN, LzCount, PopCount, SBorrow, SCarry};
pub use memory::{Load, Store};
pub use mnemonic::Mnemonic;
pub use pcode_op::{PCodeOp, PCodeOpId};
pub use terminator::{Branch, BranchInd, CBranch, Call, CallInd, Return};
pub use unop::{Unary, Unop};

#[derive(Identifier)]
pub struct InstructionId(usize);

/// A local SSA value, which is a value that is defined by an instruction and can be used by other instructions.
/// Local values are not associated with any particular memory location.
#[derive(Clone)]
pub struct Instruction<'str> {
    /// The name of this instruction
    pub(crate) name: Option<Cow<'str, str>>,

    /// The size of this value in bytes
    size: usize,

    /// The instruction which defines this value.
    mnemonic: Mnemonic,

    /// Optional address-space provenance for this instruction's result.
    space: Option<SpaceId>,

    /// The block that this instruction belongs to, if any.
    /// Instructions that are not part of any block (e.g. lifted from data sections) have `None` here.
    pub(crate) parent: Option<BlockId>,

    // Address of the binary instruction
    address: Option<u64>,

    _marker: std::marker::PhantomData<&'str ()>,
}

impl<'str> Instruction<'str> {
    fn new(size: usize, mnemonic: Mnemonic, space: Option<SpaceId>) -> Self {
        Self {
            name: None,
            parent: None,
            size,
            mnemonic,
            address: None,
            space,
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

    /// The size in bytes of the instruction's output value
    pub fn size(&'s self) -> usize {
        self.inner().size
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

    /// The address-space provenance for this instruction's result, if known.
    pub fn space(&'s self) -> Option<SpaceRef<'ctx>> {
        self.inner().space.map(|id| Space::from_id(self.ctx(), id))
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
    pub fn from_mnemonic(ctx: &'ctx mut Context<'str>, mnemonic: Mnemonic, size: usize) -> Self {
        Self::from_mnemonic_with_space(ctx, mnemonic, size, None)
    }

    // Pointer arithmetic is not allowed in the register space
    pub fn from_mnemonic_with_space(
        ctx: &'ctx mut Context<'str>,
        mnemonic: Mnemonic,
        size: usize,
        space: Option<SpaceId>,
    ) -> Self {
        // Pointer arithmetic is not allowed in the register space
        let space =
            space.filter(|&space| !matches!(Space::from_id(ctx, space).ty, SpaceType::Register));
        let insn = Instruction::new(size, mnemonic, space);
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

    pub fn address_mut(&mut self) -> &mut Option<u64> {
        &mut self.inner_mut().address
    }

    pub fn set_address(&mut self, address: u64) {
        *self.address_mut() = Some(address);
    }

    pub fn set_space(&mut self, space: SpaceId) {
        if self.inner().space.is_some_and(|s| s != space) {
            panic!(
                "Cannot change space of instruction {} from {:?} to {:?}",
                self,
                self.inner().space.map(|s| Space::from_id(self.ctx, s)),
                Space::from_id(self.ctx, space)
            );
        }

        self.inner_mut().space = Some(space);
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
