//! IR value types and the central [`ValueId`] discriminant.
//!
//! Every piece of IR state is a *value* — a constant, an instruction result,
//! a memory location, a basic block, or a function. Values are stored in a
//! [`Context`] arena and addressed through cheap, `Copy` ID types. To inspect
//! a value you convert its ID into a *reference* type (e.g. [`InstructionRef`],
//! [`BlockRef`]) that borrows the context.
//!
//! # Type Map
//!
//! | ID type            | Reference type     | What it represents          |
//! |--------------------|--------------------|-----------------------------|
//! | [`LiteralId`]      | [`LiteralRef`]     | Integer constant            |
//! | [`InstructionId`]  | [`InstructionRef`] | SSA value                   |
//! | [`VarnodeId`]      | [`VarnodeRef`]     | Named memory location       |
//! | [`BlockId`]        | [`BlockRef`]       | Basic block                 |
//! | [`FunctionId`]     | [`FunctionRef`]    | Lifted or external function |
//!
//! The [`ValueId`] enum unifies all five ID types so that code that works with
//! arbitrary values (e.g. use-def chains, operand lists) can do so without
//! generics.

use crate::context::Context;
use std::fmt::{Debug, Display, Formatter};

pub use block::{BasicBlock, BlockId, BlockMutRef, BlockRef};
pub use function::{Function, FunctionId, FunctionMutRef, FunctionRef};
pub use insn::{Instruction, InstructionId, InstructionRef};
pub use literal::{LiteralId, LiteralRef};
pub use util::named::{Named, Renameable};
pub use varnode::{Varnode, VarnodeId, VarnodeRef, register::Register, register::RegisterId};

pub mod block;
pub mod function;
pub mod insn;
pub mod literal;
pub mod registry;
pub mod util;
pub mod varnode;

/// A type-erased handle to any IR value stored in a [`Context`].
///
/// `ValueId` is the "universal pointer" used wherever code must refer to a
/// value without knowing its concrete type at compile time — for example in
/// instruction operand lists, use-def chains, and the address/name maps.
///
/// It is `Copy`, cheap to compare, and contains no borrow of the context.
/// To inspect the value, call [`Context::get_value`] which returns a
/// [`ValueRef`] tied to the context's lifetime.
///
/// # Exhaustiveness
///
/// `ValueId` is `#[non_exhaustive]`; new variants may be added in future
/// versions without a major semver bump.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ValueId {
    /// A compile-time integer constant, optionally carrying a symbolic label.
    Literal(LiteralId),
    /// An SSA value produced by an [`Instruction`].
    Instruction(InstructionId),
    /// A control-flow node ([`BasicBlock`]).
    BasicBlock(BlockId),
    /// A named memory location ([`Varnode`]) such as a register or global.
    Varnode(VarnodeId),
    /// A lifted or external [`Function`].
    Function(FunctionId),
}

impl ValueId {
    pub fn ty(&self) -> &'static str {
        match self {
            ValueId::Literal(_) => "Literal",
            ValueId::Instruction(_) => "Instruction",
            ValueId::BasicBlock(_) => "BasicBlock",
            ValueId::Varnode(_) => "Varnode",
            ValueId::Function(_) => "Function",
        }
    }

    pub fn as_literal(self) -> Option<LiteralId> {
        if let ValueId::Literal(id) = self {
            Some(id)
        } else {
            None
        }
    }

    pub fn as_instruction(self) -> Option<InstructionId> {
        if let ValueId::Instruction(id) = self {
            Some(id)
        } else {
            None
        }
    }

    pub fn as_block(self) -> Option<BlockId> {
        if let ValueId::BasicBlock(id) = self {
            Some(id)
        } else {
            None
        }
    }

    pub fn is_varnode(self) -> bool {
        matches!(self, ValueId::Varnode(_))
    }

    pub fn as_varnode(self) -> Option<VarnodeId> {
        if let ValueId::Varnode(id) = self {
            Some(id)
        } else {
            None
        }
    }

    pub fn as_function(self) -> Option<FunctionId> {
        if let ValueId::Function(id) = self {
            Some(id)
        } else {
            None
        }
    }
}

impl From<LiteralId> for ValueId {
    fn from(id: LiteralId) -> Self {
        ValueId::Literal(id)
    }
}

impl From<InstructionId> for ValueId {
    fn from(id: InstructionId) -> Self {
        ValueId::Instruction(id)
    }
}

impl From<BlockId> for ValueId {
    fn from(id: BlockId) -> Self {
        ValueId::BasicBlock(id)
    }
}

impl From<VarnodeId> for ValueId {
    fn from(id: VarnodeId) -> Self {
        ValueId::Varnode(id)
    }
}

impl From<FunctionId> for ValueId {
    fn from(id: FunctionId) -> Self {
        ValueId::Function(id)
    }
}

impl From<ValueId> for usize {
    fn from(id: ValueId) -> Self {
        match id {
            ValueId::Literal(lit_id) => lit_id.into(),
            ValueId::Instruction(insn_id) => insn_id.into(),
            ValueId::BasicBlock(bb_id) => bb_id.into(),
            ValueId::Varnode(var_id) => var_id.into(),
            ValueId::Function(fn_id) => fn_id.into(),
        }
    }
}

impl Display for ValueId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.ty(), usize::from(*self))
    }
}

/// Trait implemented by all typed value reference types.
///
/// Provides a uniform interface to a value's [`ValueId`] and its size in bytes.
/// Terminators and non-data values (blocks, functions) report `size() == 0`.
pub trait Value<'str, 'ctx>: Display {
    /// The context-unique identifier for this value.
    fn id(&self) -> ValueId;

    /// The size of this value's output in bytes, or `0` for non-data values
    /// (terminators, blocks, functions).
    fn size(&self) -> usize;
}

/// A borrowed, type-erased view of any value in a [`Context`].
///
/// `ValueRef` is the runtime-typed counterpart to [`ValueId`]. It is
/// produced by [`Context::get_value`] and borrows the context for `'ctx`.
/// Use pattern matching to downcast to a concrete reference type.
pub enum ValueRef<'str, 'ctx> {
    Literal(LiteralRef<'str, 'ctx>),
    Instruction(InstructionRef<'str, 'ctx>),
    BasicBlock(BlockRef<'str, 'ctx>),
    Varnode(VarnodeRef<'str, 'ctx>),
    Function(FunctionRef<'str, 'ctx>),
}

impl Debug for ValueRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueRef::Literal(_) => f.write_str("Literal"),
            ValueRef::Instruction(_) => f.write_str("Instruction"),
            ValueRef::BasicBlock(_) => f.write_str("BasicBlock"),
            ValueRef::Varnode(_) => f.write_str("Varnode"),
            ValueRef::Function(_) => f.write_str("Function"),
        }
    }
}

impl<'str, 'ctx> From<LiteralRef<'str, 'ctx>> for ValueRef<'str, 'ctx> {
    fn from(lit_ref: LiteralRef<'str, 'ctx>) -> Self {
        ValueRef::Literal(lit_ref)
    }
}

impl<'str, 'ctx> From<InstructionRef<'str, 'ctx>> for ValueRef<'str, 'ctx> {
    fn from(insn_ref: InstructionRef<'str, 'ctx>) -> Self {
        ValueRef::Instruction(insn_ref)
    }
}

impl<'str, 'ctx> From<BlockRef<'str, 'ctx>> for ValueRef<'str, 'ctx> {
    fn from(bb_ref: BlockRef<'str, 'ctx>) -> Self {
        ValueRef::BasicBlock(bb_ref)
    }
}

impl<'str, 'ctx> From<VarnodeRef<'str, 'ctx>> for ValueRef<'str, 'ctx> {
    fn from(var_ref: VarnodeRef<'str, 'ctx>) -> Self {
        ValueRef::Varnode(var_ref)
    }
}

impl<'str, 'ctx> From<FunctionRef<'str, 'ctx>> for ValueRef<'str, 'ctx> {
    fn from(fn_ref: FunctionRef<'str, 'ctx>) -> Self {
        ValueRef::Function(fn_ref)
    }
}

impl<'str, 'ctx> ValueRef<'str, 'ctx> {
    fn inner(&self) -> &dyn Value<'str, 'ctx> {
        match self {
            ValueRef::Literal(lit_ref) => lit_ref,
            ValueRef::Instruction(insn_ref) => insn_ref,
            ValueRef::BasicBlock(bb_ref) => bb_ref,
            ValueRef::Varnode(var_ref) => var_ref,
            ValueRef::Function(fn_ref) => fn_ref,
        }
    }

    pub fn new(id: ValueId, ctx: &'ctx Context<'str>) -> Self {
        match id {
            ValueId::Literal(lit_id) => ValueRef::Literal(LiteralRef::new(ctx, lit_id)),
            ValueId::Instruction(insn_id) => {
                ValueRef::Instruction(InstructionRef::new(ctx, insn_id))
            }
            ValueId::BasicBlock(bb_id) => ValueRef::BasicBlock(BasicBlock::from_id(ctx, bb_id)),
            ValueId::Varnode(var_id) => ValueRef::Varnode(Varnode::from_id(ctx, var_id)),
            ValueId::Function(fn_id) => ValueRef::Function(Function::from_id(ctx, fn_id)),
        }
    }

    pub fn from_id(ctx: &'ctx Context<'str>, id: ValueId) -> Self {
        Self::new(id, ctx)
    }
}

impl Display for ValueRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            // Display function as compact reference when used as a value operand.
            ValueRef::Function(fn_ref) => write!(f, "<{}>", fn_ref.name()),
            _ => self.inner().fmt(f),
        }
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for ValueRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.inner().id()
    }

    fn size(&self) -> usize {
        self.inner().size()
    }
}
