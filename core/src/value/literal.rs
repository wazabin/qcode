//! Compile-time integer constants, optionally with symbolic labels.
//!
//! A [`Literal`] stores a raw `u64` value together with an optional
//! [`SymbolicRef`] that gives it meaning beyond its numeric value — for example
//! the address of a known block or function. When a literal has a symbolic
//! reference it is displayed as `&<name>` rather than `0x…`.
//!
//! Every literal carries a [`TypeId`] that encodes its byte width (and, for
//! pointer literals, its space provenance). Type is preserved through constant
//! folding.

use crate::{
    context::Shared,
    types::TypeId,
    value::{
        Value, ValueId,
        block::BlockId,
        function::FunctionId,
        util::base_ref::{BaseRef, WithShared},
    },
};
use jstd::Identifier;

#[derive(Identifier)]
pub struct LiteralId(usize);

/// An optional symbolic meaning attached to a [`Literal`].
///
/// When the assembler/lifter knows that a numeric constant is actually the
/// address of a block, a function, or a string, it stores a `SymbolicRef` so
/// that the literal can be displayed and reasoned about symbolically.
///
#[derive(Debug, Clone, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SymbolicRef {
    /// The literal is the address of this basic block.
    Block(BlockId),
    /// The literal is the entry address of this function.
    Function(FunctionId),
    /// The literal is a pointer to this string constant.
    String(String),
}

/// A compile-time integer constant stored in a [`Context`](crate::context::Context).
///
/// The raw value is a `u64`; [`LiteralRef::value`] masks it to the width
/// described by the literal's [`TypeId`].
#[derive(Debug, Clone, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Literal {
    /// Raw integer value (may be wider than the type's size before masking).
    pub value: u64,
    /// The type of this constant (encodes size and semantic kind).
    pub type_id: TypeId,
    /// Optional symbolic annotation (block address, function address, string).
    pub symbolic: Option<SymbolicRef>,
}

pub type LiteralRef<'str, 'ctx> = BaseRef<&'ctx Shared<'str>, LiteralId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithShared<'s, 'ctx, 'str> for LiteralRef<'str, 'ctx> {
    fn shared(&'s self) -> &'ctx Shared<'str> {
        self.ctx
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx, Ctx> BaseRef<Ctx, LiteralId>
where
    Self: WithShared<'s, 'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx Literal {
        &self.shared().values.literals[self.id]
    }

    pub fn mask(&'s self) -> u64 {
        let size = self.shared().types.size_of(self.inner().type_id);
        if size >= 8 {
            u64::MAX
        } else {
            (1u64 << (size * 8)) - 1
        }
    }

    pub fn value(&'s self) -> u64 {
        self.inner().value & self.mask()
    }

    /// Returns the [`TypeId`] of this literal.
    pub fn type_id(&'s self) -> TypeId {
        self.inner().type_id
    }
}

impl std::fmt::Display for LiteralRef<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let literal = &self.ctx.values.literals[self.id];
        match &literal.symbolic {
            // Symbolic block/function names live in bodies/interfaces, which a
            // `&Shared`-backed leaf ref cannot reach; the block/function *name*
            // rendering is done by the full-context path (`segment::literal_atom`,
            // used by the instruction renderer and the dataflow graph). Here we
            // fall back to the numeric form. See context-split 5b-ii item #1.
            Some(SymbolicRef::Block(_)) | Some(SymbolicRef::Function(_)) => {
                write!(f, "&<0x{:x}>", literal.value)
            }
            Some(SymbolicRef::String(s)) => write!(f, "&{:?}", s),
            // A `bool` literal prints as `true`/`false`; the `bool` type token is
            // emitted by the operand's type prefix, so the round-trip is `bool true`.
            None if self.ctx.types.is_bool(literal.type_id) => {
                write!(f, "{}", if literal.value != 0 { "true" } else { "false" })
            }
            None => write!(f, "0x{:x}", literal.value),
        }
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for LiteralRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        ValueId::Literal(self.id)
    }

    fn size(&self) -> usize {
        self.ctx
            .types
            .size_of(self.ctx.values.literals[self.id].type_id)
    }
}
