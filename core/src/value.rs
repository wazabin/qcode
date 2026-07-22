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
/// owning [`FunctionBody`]'s arena. Derived `Ord` compares `func` then `local`.
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

            /// Drop the qualifying function and yield the bare body-local index for
            /// in-body storage, asserting (debug builds) that the id belongs to
            /// `func` — the strict-locality tripwire. Use at every storage-flip
            /// write site where the ambient owning function is known.
            #[inline]
            pub fn localize(self, func: $crate::value::FunctionId) -> $local {
                debug_assert_eq!(
                    self.func, func,
                    concat!(stringify!($name), "::localize: foreign id"),
                );
                self.local
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

pub use block::cfg::LocalBlockId;
pub use block::{BasicBlock, BlockId, BlockMutRef, BlockRef};
pub use block_param::{BlockParam, BlockParamId, BlockParamMutRef, BlockParamRef, LocalParamId};
pub use bytes::{
    Bytes, BytesDisplay, BytesId, BytesRef, StringEncoding, decode_string, escape_decoded,
    render_bytes_literal,
};
pub use function::{
    ArgMemKind, BodyArenaKindStats, BodyArenaStats, ExternArg, ExternArgmem, ExternInterface,
    ExternSlot, FunctionBody, FunctionEffects, FunctionId, FunctionKind, FunctionMutRef,
    FunctionRef, MemoryChannelState, ParamAttrs, RegisterChannelState, RegisterEffectSets,
    RegisterInterfaceMap, WrittenSpaces, WrittenSpacesState,
};
pub use insn::LocalInsnId;
pub use insn::{Instruction, InstructionId, InstructionRef};
pub use literal::{LiteralId, LiteralRef};
pub use poison::{Poison, PoisonId, PoisonRef};
pub use temp::{
    LocalTempId, LocalTempSpaceId, Temp, TempId, TempRef, TempSpace, TempSpaceId, TempSpaceRef,
};
pub use util::named::{Named, Renameable};
pub use varnode::{Varnode, VarnodeId, VarnodeRef, register::Register, register::RegisterId};

pub mod block;
pub mod block_param;
pub mod bytes;
pub mod function;
pub mod insn;
pub mod interner;
pub mod literal;
pub mod poison;
pub mod registry;
pub mod temp;
pub mod util;
pub mod varnode;
pub mod view;
pub mod view_mut;

pub use view::{BodyView, ModuleView, QCodeView};
pub use view_mut::QCodeMut;

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
    /// A function-local temporary memory value.
    Temp(TempId),
    /// A lifted or external [`FunctionBody`].
    Function(FunctionId),
    /// A typed **poison** value: a placeholder with undefined bits (argpromote
    /// v2 clobber slots). Never folded by GVN; reading it in the emulator is a
    /// hard error. See [`poison`].
    Poison(PoisonId),
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
            ValueId::Temp(_) => "Temp",
            ValueId::Function(_) => "Function",
            ValueId::Poison(_) => "Poison",
        }
    }

    /// The function that owns this value's definition, if it is an SSA def
    /// (an [`Instruction`] result or a [`BlockParam`]). Shared values (literals,
    /// bytes, varnodes) and functions/blocks return `None` — they have no single
    /// owning function and their per-function use-lists live in each using
    /// function's reverse-use map (exposed through [`FunctionBody::users_of`]).
    pub fn owning_function(self) -> Option<FunctionId> {
        match self {
            ValueId::Instruction(id) => Some(id.func),
            ValueId::BlockParam(id) => Some(id.func),
            ValueId::Temp(id) => Some(id.func),
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
            ValueId::Temp(id) => Some(id.func),
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

    pub fn as_temp(self) -> Option<TempId> {
        if let ValueId::Temp(id) = self {
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

    pub fn as_poison(self) -> Option<PoisonId> {
        if let ValueId::Poison(id) = self {
            Some(id)
        } else {
            None
        }
    }

    pub fn is_poison(self) -> bool {
        matches!(self, ValueId::Poison(_))
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

impl From<TempId> for ValueId {
    fn from(id: TempId) -> Self {
        ValueId::Temp(id)
    }
}

impl From<FunctionId> for ValueId {
    fn from(id: FunctionId) -> Self {
        ValueId::Function(id)
    }
}

impl From<PoisonId> for ValueId {
    fn from(id: PoisonId) -> Self {
        ValueId::Poison(id)
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
            ValueId::Temp(id) => (7, usize::from(id.func) as u32, usize::from(id.local) as u32),
            ValueId::Poison(id) => (8, 0, u32::try_from(usize::from(id)).unwrap_or(u32::MAX)),
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
            ValueId::Temp(id) => write!(f, "Temp({id})"),
            ValueId::Poison(id) => write!(f, "Poison({})", usize::from(id)),
        }
    }
}

/// The **body-local** twin of [`ValueId`]: the form an in-body operand holds once
/// storage is localized (stage 6a, ruling 2).
///
/// It mirrors `ValueId` variant-for-variant, but its arena arms
/// (`Instruction`/`BasicBlock`/`BlockParam`) carry a **bare function-local index**
/// (`LocalInsnId`/`LocalBlockId`/`LocalParamId`) with **no** owning `FunctionId` —
/// SSA is intra-function, so the owning function is the ambient body and does not
/// need re-storing on every operand. The shared/module arms
/// (`Literal`/`Bytes`/`Varnode`/`Function`) carry the exact same globally-interned
/// or module ids as `ValueId` (they are already the boundary currency).
///
/// The two forms convert through [`LocalValueId::qualify`] (stamp the ambient
/// `func`) and [`ValueId::localize`] (drop it, asserting it matched). `ValueId`
/// stays the qualified boundary currency (module tables, refs, consumer crates);
/// `LocalValueId` is confined to in-body storage.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocalValueId {
    /// A compile-time integer constant (module-interned; same id as `ValueId`).
    Literal(LiteralId),
    /// A compile-time opaque byte blob (module-interned; same id as `ValueId`).
    Bytes(BytesId),
    /// An SSA value produced by an [`Instruction`] — bare body-local index.
    Instruction(LocalInsnId),
    /// A control-flow node ([`BasicBlock`]) — bare body-local index.
    BasicBlock(LocalBlockId),
    /// A typed block-entry parameter — bare body-local index.
    BlockParam(LocalParamId),
    /// A named memory location ([`Varnode`]) (module id; same as `ValueId`).
    Varnode(VarnodeId),
    /// A body-owned temporary value — bare body-local index.
    Temp(LocalTempId),
    /// A lifted or external [`FunctionBody`] (module id; same as `ValueId`).
    Function(FunctionId),
    /// A typed poison value (module-interned; same id as `ValueId`).
    Poison(PoisonId),
}

impl LocalValueId {
    /// Qualify a body-local id back into the boundary [`ValueId`] by stamping the
    /// owning function `func` onto the arena arms. Shared/module arms pass through
    /// unchanged.
    pub fn qualify(self, func: FunctionId) -> ValueId {
        match self {
            LocalValueId::Literal(id) => ValueId::Literal(id),
            LocalValueId::Bytes(id) => ValueId::Bytes(id),
            LocalValueId::Varnode(id) => ValueId::Varnode(id),
            LocalValueId::Function(id) => ValueId::Function(id),
            LocalValueId::Poison(id) => ValueId::Poison(id),
            LocalValueId::Instruction(local) => {
                ValueId::Instruction(InstructionId::new(func, local))
            }
            LocalValueId::BasicBlock(local) => ValueId::BasicBlock(BlockId::new(func, local)),
            LocalValueId::BlockParam(local) => ValueId::BlockParam(BlockParamId::new(func, local)),
            LocalValueId::Temp(local) => ValueId::Temp(TempId::new(func, local)),
        }
    }
}

impl ValueId {
    /// Localize a qualified id for storage inside `func`'s body, dropping the
    /// owning `FunctionId` from the arena arms. In debug builds this asserts the
    /// id's `func` equals `func` — the strict-locality tripwire (ruling 2): a
    /// foreign operand is a bug and fires loudly here rather than misrouting.
    /// Shared/module arms pass through unchanged.
    pub fn localize(self, func: FunctionId) -> LocalValueId {
        match self {
            ValueId::Literal(id) => LocalValueId::Literal(id),
            ValueId::Bytes(id) => LocalValueId::Bytes(id),
            ValueId::Varnode(id) => LocalValueId::Varnode(id),
            ValueId::Function(id) => LocalValueId::Function(id),
            ValueId::Poison(id) => LocalValueId::Poison(id),
            ValueId::Instruction(id) => {
                debug_assert_eq!(
                    id.func, func,
                    "localize: foreign instruction operand {id:?} in function {func} \
                     (strict IR locality, ruling 2)"
                );
                LocalValueId::Instruction(id.local)
            }
            ValueId::BasicBlock(id) => {
                debug_assert_eq!(
                    id.func, func,
                    "localize: foreign block operand {id:?} in function {func} \
                     (strict IR locality, ruling 2)"
                );
                LocalValueId::BasicBlock(id.local)
            }
            ValueId::BlockParam(id) => {
                debug_assert_eq!(
                    id.func, func,
                    "localize: foreign block-param operand {id:?} in function {func} \
                     (strict IR locality, ruling 2)"
                );
                LocalValueId::BlockParam(id.local)
            }
            ValueId::Temp(id) => LocalValueId::Temp(id.localize(func)),
        }
    }

    /// Drop the owning `FunctionId` from the arena arms **without** a locality
    /// check, using the id's *own* embedded func. Unlike [`localize`](Self::localize)
    /// there is no ambient function to assert against: this is for keying a
    /// per-function body map (e.g. `FunctionBody.users`) by a value that already
    /// carries its own func. Within one body's map every arena key has that
    /// body's func, so stripping it is injective and lookup-stable; shared/module
    /// arms pass through unchanged.
    pub fn strip_func(self) -> LocalValueId {
        match self {
            ValueId::Literal(id) => LocalValueId::Literal(id),
            ValueId::Bytes(id) => LocalValueId::Bytes(id),
            ValueId::Varnode(id) => LocalValueId::Varnode(id),
            ValueId::Function(id) => LocalValueId::Function(id),
            ValueId::Poison(id) => LocalValueId::Poison(id),
            ValueId::Instruction(id) => LocalValueId::Instruction(id.local),
            ValueId::BasicBlock(id) => LocalValueId::BasicBlock(id.local),
            ValueId::BlockParam(id) => LocalValueId::BlockParam(id.local),
            ValueId::Temp(id) => LocalValueId::Temp(id.local),
        }
    }

    /// Localize this id **only if it is function-agnostic** — a shared/module arm
    /// (`Literal`/`Bytes`/`Varnode`/`Function`) that carries no owning function and
    /// so needs no ambient body to convert. Returns `None` for the function-scoped
    /// arena arms (`Instruction`/`BasicBlock`/`BlockParam`/`Temp`), which cannot be
    /// dropped into a *different* body's local space without misrouting. This is the
    /// safe localizer for a value that may legitimately be a constant flowing into a
    /// minted body but must otherwise be remapped through a value map.
    pub fn as_function_agnostic(self) -> Option<LocalValueId> {
        match self {
            ValueId::Literal(id) => Some(LocalValueId::Literal(id)),
            ValueId::Bytes(id) => Some(LocalValueId::Bytes(id)),
            ValueId::Varnode(id) => Some(LocalValueId::Varnode(id)),
            ValueId::Function(id) => Some(LocalValueId::Function(id)),
            ValueId::Poison(id) => Some(LocalValueId::Poison(id)),
            ValueId::Instruction(_)
            | ValueId::BasicBlock(_)
            | ValueId::BlockParam(_)
            | ValueId::Temp(_) => None,
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
pub enum ValueRef<'str, 'ctx, R = ModuleView<'ctx, 'str>> {
    Literal(LiteralRef<'str, 'ctx>),
    Bytes(BytesRef<'str, 'ctx>),
    Instruction(InstructionRef<'str, 'ctx, R>),
    BasicBlock(BlockRef<'str, 'ctx, R>),
    BlockParam(BlockParamRef<'str, 'ctx, R>),
    Varnode(VarnodeRef<'str, 'ctx>),
    Temp(TempRef<'str, 'ctx, R>),
    Function(FunctionRef<'str, 'ctx, R>),
    Poison(PoisonRef<'str, 'ctx>),
}

impl<R> Debug for ValueRef<'_, '_, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueRef::Literal(_) => f.write_str("Literal"),
            ValueRef::Bytes(_) => f.write_str("Bytes"),
            ValueRef::Instruction(_) => f.write_str("Instruction"),
            ValueRef::BasicBlock(_) => f.write_str("BasicBlock"),
            ValueRef::BlockParam(_) => f.write_str("BlockParam"),
            ValueRef::Varnode(_) => f.write_str("Varnode"),
            ValueRef::Temp(_) => f.write_str("Temp"),
            ValueRef::Function(_) => f.write_str("Function"),
            ValueRef::Poison(_) => f.write_str("Poison"),
        }
    }
}

impl<'str, 'ctx, R> From<LiteralRef<'str, 'ctx>> for ValueRef<'str, 'ctx, R> {
    fn from(lit_ref: LiteralRef<'str, 'ctx>) -> Self {
        ValueRef::Literal(lit_ref)
    }
}

impl<'str, 'ctx, R> From<BytesRef<'str, 'ctx>> for ValueRef<'str, 'ctx, R> {
    fn from(bytes_ref: BytesRef<'str, 'ctx>) -> Self {
        ValueRef::Bytes(bytes_ref)
    }
}

impl<'str, 'ctx, R> From<InstructionRef<'str, 'ctx, R>> for ValueRef<'str, 'ctx, R> {
    fn from(insn_ref: InstructionRef<'str, 'ctx, R>) -> Self {
        ValueRef::Instruction(insn_ref)
    }
}

impl<'str, 'ctx, R> From<BlockRef<'str, 'ctx, R>> for ValueRef<'str, 'ctx, R> {
    fn from(bb_ref: BlockRef<'str, 'ctx, R>) -> Self {
        ValueRef::BasicBlock(bb_ref)
    }
}

impl<'str, 'ctx, R> From<BlockParamRef<'str, 'ctx, R>> for ValueRef<'str, 'ctx, R> {
    fn from(param_ref: BlockParamRef<'str, 'ctx, R>) -> Self {
        ValueRef::BlockParam(param_ref)
    }
}

impl<'str, 'ctx, R> From<VarnodeRef<'str, 'ctx>> for ValueRef<'str, 'ctx, R> {
    fn from(var_ref: VarnodeRef<'str, 'ctx>) -> Self {
        ValueRef::Varnode(var_ref)
    }
}

impl<'str, 'ctx, R> From<TempRef<'str, 'ctx, R>> for ValueRef<'str, 'ctx, R> {
    fn from(temp_ref: TempRef<'str, 'ctx, R>) -> Self {
        ValueRef::Temp(temp_ref)
    }
}

impl<'str, 'ctx, R> From<FunctionRef<'str, 'ctx, R>> for ValueRef<'str, 'ctx, R> {
    fn from(fn_ref: FunctionRef<'str, 'ctx, R>) -> Self {
        ValueRef::Function(fn_ref)
    }
}

impl<'str, 'ctx, R> From<PoisonRef<'str, 'ctx>> for ValueRef<'str, 'ctx, R> {
    fn from(poison_ref: PoisonRef<'str, 'ctx>) -> Self {
        ValueRef::Poison(poison_ref)
    }
}

impl<'str, 'ctx> ValueRef<'str, 'ctx> {
    pub fn new(id: ValueId, ctx: &'ctx Context<'str>) -> Self {
        ValueRef::from_view(ModuleView::new(ctx), id)
    }

    pub fn from_id(ctx: &'ctx Context<'str>, id: ValueId) -> Self {
        Self::new(id, ctx)
    }
}

impl<'str: 'ctx, 'ctx, R> ValueRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    pub fn from_view(view: R, id: ValueId) -> Self {
        match id {
            ValueId::Literal(id) => ValueRef::Literal(LiteralRef::from_id(view.shared(), id)),
            ValueId::Bytes(id) => ValueRef::Bytes(BytesRef::from_id(view.shared(), id)),
            ValueId::Varnode(id) => ValueRef::Varnode(Varnode::from_id(view.shared(), id)),
            ValueId::Temp(id) => ValueRef::Temp(TempRef::new(view, id)),
            ValueId::Instruction(id) => ValueRef::Instruction(InstructionRef::new(view, id)),
            ValueId::BasicBlock(id) => ValueRef::BasicBlock(BlockRef::new(view, id)),
            ValueId::BlockParam(id) => ValueRef::BlockParam(BlockParamRef::new(view, id)),
            ValueId::Function(id) => ValueRef::Function(FunctionRef::new(view, id)),
            ValueId::Poison(id) => ValueRef::Poison(PoisonRef::from_id(view.shared(), id)),
        }
    }

    fn inner(&self) -> &dyn Value<'str, 'ctx> {
        match self {
            ValueRef::Literal(r) => r,
            ValueRef::Bytes(r) => r,
            ValueRef::Instruction(r) => r,
            ValueRef::BasicBlock(r) => r,
            ValueRef::BlockParam(r) => r,
            ValueRef::Varnode(r) => r,
            ValueRef::Temp(r) => r,
            ValueRef::Function(r) => r,
            ValueRef::Poison(r) => r,
        }
    }

    pub fn space(&self) -> Option<SpaceRef<'ctx>> {
        match self {
            ValueRef::Varnode(v) => Some(v.space()),
            ValueRef::Instruction(i) => i.space(),
            ValueRef::Temp(_) => None,
            ValueRef::Literal(_)
            | ValueRef::Bytes(_)
            | ValueRef::BasicBlock(_)
            | ValueRef::BlockParam(_)
            | ValueRef::Function(_)
            | ValueRef::Poison(_) => None,
        }
    }

    /// Qualified memory-space provenance. Unlike [`space`](Self::space), this
    /// represents body-local temporary spaces without pretending they are
    /// shared [`Space`] values.
    pub fn memory_space(&self) -> Option<crate::space::MemorySpaceId> {
        match self {
            ValueRef::Varnode(v) => Some(crate::space::MemorySpaceId::Shared(v.space().id)),
            ValueRef::Instruction(i) => i.memory_space(),
            ValueRef::Temp(t) => Some(t.memory_space()),
            ValueRef::Literal(_)
            | ValueRef::Bytes(_)
            | ValueRef::BasicBlock(_)
            | ValueRef::BlockParam(_)
            | ValueRef::Function(_)
            | ValueRef::Poison(_) => None,
        }
    }
}

impl<'str: 'ctx, 'ctx, R> Display for ValueRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // A value operand's rendering — `<ty> <atom>` uniformly, bare for value
        // references with no scalar type — is defined once, as tokens, in the
        // instruction `segment` module; `Display` is those tokens concatenated.
        //
        // Shared-leaf refs (literal, bytes, varnode) carry only a `&Shared`, so
        // they render through the `&Shared` token path; the arena-cluster refs
        // route reads through their static `QCodeView` provider (context-split
        // Pin B).
        let tokens = match self {
            ValueRef::Literal(r) => insn::segment::value_tokens_shared(r.ctx, self.id()),
            ValueRef::Bytes(r) => insn::segment::value_tokens_shared(r.ctx, self.id()),
            ValueRef::Varnode(r) => insn::segment::value_tokens_shared(r.ctx, self.id()),
            ValueRef::Poison(r) => insn::segment::value_tokens_shared(r.ctx, self.id()),
            ValueRef::Temp(r) => insn::segment::value_tokens_view(r.view, self.id()),
            ValueRef::Instruction(r) => insn::segment::value_tokens_view(r.view, self.id()),
            ValueRef::BasicBlock(r) => insn::segment::value_tokens_view(r.view, self.id()),
            ValueRef::BlockParam(r) => insn::segment::value_tokens_view(r.view, self.id()),
            ValueRef::Function(r) => insn::segment::value_tokens_view(r.view, self.id()),
        };
        for token in tokens {
            write!(f, "{}", token.text)?;
        }
        Ok(())
    }
}

impl<'str: 'ctx, 'ctx, R> Value<'str, 'ctx> for ValueRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn id(&self) -> ValueId {
        self.inner().id()
    }

    fn size(&self) -> usize {
        self.inner().size()
    }
}

#[cfg(test)]
mod local_value_id_tests {
    use super::*;

    /// Every `ValueId` variant round-trips losslessly through
    /// `localize(func).qualify(func)`, with the arena arms carrying the given
    /// `func` and the shared/module arms ignoring it.
    #[test]
    fn qualify_localize_round_trips_every_variant() {
        let func = FunctionId::from(7usize);
        let other = FunctionId::from(3usize);

        let arena: [ValueId; 4] = [
            ValueId::Instruction(InstructionId::new(func, LocalInsnId::from(2usize))),
            ValueId::BasicBlock(BlockId::new(func, LocalBlockId::from(5usize))),
            ValueId::BlockParam(BlockParamId::new(func, LocalParamId::from(1usize))),
            ValueId::Temp(TempId::new(func, LocalTempId::from(4usize))),
        ];
        for id in arena {
            assert_eq!(id.localize(func).qualify(func), id, "{id:?}");
        }

        // Shared/module arms pass through regardless of the ambient func.
        let shared: [ValueId; 4] = [
            ValueId::Literal(LiteralId::from(0usize)),
            ValueId::Bytes(BytesId::from(0usize)),
            ValueId::Varnode(VarnodeId::from(0usize)),
            ValueId::Function(other),
        ];
        for id in shared {
            assert_eq!(id.localize(func).qualify(func), id, "{id:?}");
            // Any func requalifies a shared arm to itself — it carries no func.
            assert_eq!(id.localize(func).qualify(other), id, "{id:?}");
        }
    }

    /// `localize` asserts the operand's func matches the ambient body (the
    /// strict-locality tripwire). A foreign arena id panics in debug builds.
    #[test]
    #[should_panic(expected = "strict IR locality")]
    #[cfg(debug_assertions)]
    fn localize_rejects_foreign_arena_id() {
        let func = FunctionId::from(7usize);
        let foreign = FunctionId::from(9usize);
        let id = ValueId::Instruction(InstructionId::new(foreign, LocalInsnId::from(0usize)));
        let _ = id.localize(func);
    }

    #[test]
    #[should_panic(expected = "TempId::localize: foreign id")]
    #[cfg(debug_assertions)]
    fn localize_rejects_foreign_temporary_id() {
        let owner = FunctionId::from(7usize);
        let foreign = FunctionId::from(9usize);
        let id = ValueId::Temp(TempId::new(foreign, LocalTempId::from(0usize)));
        let _ = id.localize(owner);
    }
}
