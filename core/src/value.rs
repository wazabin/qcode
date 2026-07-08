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

use crate::{context::Context, space::SpaceRef};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display, Formatter};

/// Declares a composite IR ID: a `{ func: FunctionId, local: LocalX }` pair.
///
/// SSA is intra-function, so every instruction/block/param/edge handle carries
/// the owning function plus a function-local index. Unlike the `Identifier`
/// newtypes these do **not** index a global `Registry` — they route through the
/// owning [`Function`]'s arena. Derived `Ord` compares `func` then `local`.
#[macro_export]
macro_rules! composite_id {
    ($name:ident, $local:ty) => {
        #[derive(
            Clone,
            Copy,
            PartialEq,
            Eq,
            Hash,
            PartialOrd,
            Ord,
            Default,
            ::serde::Serialize,
            ::serde::Deserialize,
        )]
        pub struct $name {
            pub func: $crate::value::FunctionId,
            pub local: $local,
        }

        impl $name {
            pub const fn new(func: $crate::value::FunctionId, local: $local) -> Self {
                Self { func, local }
            }
        }

        impl ::core::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                write!(
                    f,
                    concat!(stringify!($name), "({}:{})"),
                    self.func, self.local
                )
            }
        }

        impl ::core::fmt::Display for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                write!(f, "{}:{}", self.func, self.local)
            }
        }
    };
}

pub use block::cfg::{LocalBlockId, LocalEdgeId};
pub use block::{BasicBlock, BlockId, BlockMutRef, BlockRef};
pub use block_param::{BlockParam, BlockParamId, BlockParamMutRef, BlockParamRef, LocalParamId};
pub use bytes::{
    Bytes, BytesDisplay, BytesId, BytesRef, StringEncoding, decode_string, escape_decoded,
    render_bytes_literal,
};
pub use function::{Function, FunctionId, FunctionKind, FunctionMutRef, FunctionRef, ParamAttrs};
pub use insn::LocalInsnId;
pub use insn::{Instruction, InstructionId, InstructionRef};
pub use literal::{LiteralId, LiteralRef};
pub use util::named::{Named, Renameable};
pub use varnode::{Varnode, VarnodeId, VarnodeRef, register::Register, register::RegisterId};

pub mod block;
pub mod block_param;
pub mod bytes;
pub mod function;
pub mod insn;
pub mod interner;
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
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueId {
    /// A compile-time integer constant, optionally carrying a symbolic label.
    Literal(LiteralId),
    /// A compile-time opaque byte blob wider than a [`Literal`] can hold.
    Bytes(BytesId),
    /// An SSA value produced by an [`Instruction`].
    Instruction(InstructionId),
    /// A control-flow node ([`BasicBlock`]).
    BasicBlock(BlockId),
    /// A typed parameter declared at the entry of a basic block.
    BlockParam(BlockParamId),
    /// A named memory location ([`Varnode`]) such as a register or global.
    Varnode(VarnodeId),
    /// A lifted or external [`Function`].
    Function(FunctionId),
}

impl ValueId {
    pub fn ty(&self) -> &'static str {
        match self {
            ValueId::Literal(_) => "Literal",
            ValueId::Bytes(_) => "Bytes",
            ValueId::Instruction(_) => "Instruction",
            ValueId::BasicBlock(_) => "BasicBlock",
            ValueId::BlockParam(_) => "BlockParam",
            ValueId::Varnode(_) => "Varnode",
            ValueId::Function(_) => "Function",
        }
    }

    /// The function that owns this value's definition, if it is an SSA def
    /// (an [`Instruction`] result or a [`BlockParam`]). Shared values (literals,
    /// bytes, varnodes) and functions/blocks return `None` — they have no single
    /// owning function and their per-function use-lists live in each using
    /// function's [`users`](crate::value::function::Function::users) map.
    pub fn owning_function(self) -> Option<FunctionId> {
        match self {
            ValueId::Instruction(id) => Some(id.func),
            ValueId::BlockParam(id) => Some(id.func),
            _ => None,
        }
    }

    /// The function whose **name table** owns this value's name, if any. Block,
    /// instruction, and block-param names are function-scoped, so those return
    /// their function; module-scoped values (functions, varnodes, spaces,
    /// literals, bytes) return `None` and use the global name map. Unlike
    /// [`owning_function`](Self::owning_function) this includes blocks (by their
    /// storage function).
    pub fn name_scope_function(self) -> Option<FunctionId> {
        match self {
            ValueId::Instruction(id) => Some(id.func),
            ValueId::BlockParam(id) => Some(id.func),
            ValueId::BasicBlock(id) => Some(id.func),
            _ => None,
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

    pub fn as_block_param(self) -> Option<BlockParamId> {
        if let ValueId::BlockParam(id) = self {
            Some(id)
        } else {
            None
        }
    }

    pub fn as_bytes(self) -> Option<BytesId> {
        if let ValueId::Bytes(id) = self {
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

impl From<BytesId> for ValueId {
    fn from(id: BytesId) -> Self {
        ValueId::Bytes(id)
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

impl From<BlockParamId> for ValueId {
    fn from(id: BlockParamId) -> Self {
        ValueId::BlockParam(id)
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

impl ValueId {
    /// A total-order sort key that does not require `Into<usize>` (which the
    /// composite instruction/block/param IDs deliberately lack). The tuple is
    /// `(variant_tag, function-or-0, local-or-global-index)`; global values put
    /// their index in the third slot with function 0.
    pub fn order_key(&self) -> (u8, u32, u32) {
        match *self {
            ValueId::Literal(id) => (0, 0, u32::try_from(usize::from(id)).unwrap_or(u32::MAX)),
            ValueId::Bytes(id) => (1, 0, u32::try_from(usize::from(id)).unwrap_or(u32::MAX)),
            ValueId::Varnode(id) => (2, 0, u32::try_from(usize::from(id)).unwrap_or(u32::MAX)),
            ValueId::Function(id) => (3, 0, u32::try_from(usize::from(id)).unwrap_or(u32::MAX)),
            ValueId::Instruction(id) => {
                (4, usize::from(id.func) as u32, usize::from(id.local) as u32)
            }
            ValueId::BasicBlock(id) => {
                (5, usize::from(id.func) as u32, usize::from(id.local) as u32)
            }
            ValueId::BlockParam(id) => {
                (6, usize::from(id.func) as u32, usize::from(id.local) as u32)
            }
        }
    }
}

impl Display for ValueId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match *self {
            ValueId::Literal(id) => write!(f, "Literal({})", usize::from(id)),
            ValueId::Bytes(id) => write!(f, "Bytes({})", usize::from(id)),
            ValueId::Varnode(id) => write!(f, "Varnode({})", usize::from(id)),
            ValueId::Function(id) => write!(f, "Function({})", usize::from(id)),
            ValueId::Instruction(id) => write!(f, "Instruction({id})"),
            ValueId::BasicBlock(id) => write!(f, "BasicBlock({id})"),
            ValueId::BlockParam(id) => write!(f, "BlockParam({id})"),
        }
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
    Bytes(BytesRef<'str, 'ctx>),
    Instruction(InstructionRef<'str, 'ctx>),
    BasicBlock(BlockRef<'str, 'ctx>),
    BlockParam(BlockParamRef<'str, 'ctx>),
    Varnode(VarnodeRef<'str, 'ctx>),
    Function(FunctionRef<'str, 'ctx>),
}

impl Debug for ValueRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueRef::Literal(_) => f.write_str("Literal"),
            ValueRef::Bytes(_) => f.write_str("Bytes"),
            ValueRef::Instruction(_) => f.write_str("Instruction"),
            ValueRef::BasicBlock(_) => f.write_str("BasicBlock"),
            ValueRef::BlockParam(_) => f.write_str("BlockParam"),
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

impl<'str, 'ctx> From<BytesRef<'str, 'ctx>> for ValueRef<'str, 'ctx> {
    fn from(bytes_ref: BytesRef<'str, 'ctx>) -> Self {
        ValueRef::Bytes(bytes_ref)
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

impl<'str, 'ctx> From<BlockParamRef<'str, 'ctx>> for ValueRef<'str, 'ctx> {
    fn from(param_ref: BlockParamRef<'str, 'ctx>) -> Self {
        ValueRef::BlockParam(param_ref)
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
            ValueRef::Bytes(bytes_ref) => bytes_ref,
            ValueRef::Instruction(insn_ref) => insn_ref,
            ValueRef::BasicBlock(bb_ref) => bb_ref,
            ValueRef::BlockParam(param_ref) => param_ref,
            ValueRef::Varnode(var_ref) => var_ref,
            ValueRef::Function(fn_ref) => fn_ref,
        }
    }

    pub fn new(id: ValueId, ctx: &'ctx Context<'str>) -> Self {
        match id {
            ValueId::Literal(lit_id) => ValueRef::Literal(LiteralRef::new(ctx, lit_id)),
            ValueId::Bytes(bytes_id) => ValueRef::Bytes(BytesRef::new(ctx, bytes_id)),
            ValueId::Instruction(insn_id) => {
                ValueRef::Instruction(InstructionRef::from_id(ctx, insn_id))
            }
            ValueId::BasicBlock(bb_id) => ValueRef::BasicBlock(BasicBlock::from_id(ctx, bb_id)),
            ValueId::BlockParam(param_id) => {
                ValueRef::BlockParam(BlockParam::from_id(ctx, param_id))
            }
            ValueId::Varnode(var_id) => ValueRef::Varnode(Varnode::from_id(ctx, var_id)),
            ValueId::Function(fn_id) => ValueRef::Function(Function::from_id(ctx, fn_id)),
        }
    }

    pub fn from_id(ctx: &'ctx Context<'str>, id: ValueId) -> Self {
        Self::new(id, ctx)
    }

    pub fn space(&self) -> Option<SpaceRef<'ctx>> {
        match self {
            ValueRef::Varnode(v) => Some(v.space()),
            ValueRef::Instruction(i) => i.space(),
            ValueRef::Literal(_)
            | ValueRef::Bytes(_)
            | ValueRef::BasicBlock(_)
            | ValueRef::BlockParam(_)
            | ValueRef::Function(_) => None,
        }
    }
}

impl Display for ValueRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // A value operand's rendering — `<ty> <atom>` uniformly, bare for value
        // references with no scalar type — is defined once, as tokens, in the
        // instruction `segment` module; `Display` is those tokens concatenated.
        let ctx = match self {
            ValueRef::Literal(r) => r.ctx,
            ValueRef::Bytes(r) => r.ctx,
            ValueRef::Instruction(r) => r.ctx.shared(),
            ValueRef::BasicBlock(r) => r.ctx.shared(),
            ValueRef::BlockParam(r) => r.ctx.shared(),
            ValueRef::Varnode(r) => r.ctx,
            ValueRef::Function(r) => r.ctx.shared(),
        };
        for token in insn::segment::value_tokens(ctx, self.id()) {
            write!(f, "{}", token.text)?;
        }
        Ok(())
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
