//! Compile-time integer constants, optionally with symbolic labels.
//!
//! A [`Literal`] stores a raw `u64` value together with an optional
//! [`SymbolicRef`] that gives it meaning beyond its numeric value - for example
//! the address of a known block or function. When a literal has a symbolic
//! reference it is displayed as `&<name>` rather than `0x…`.

use crate::{
    context::Context,
    value::{
        Function, Value, ValueId,
        block::{BlockId, BlockRef},
        function::FunctionId,
        util::base_ref::{BaseRef, WithCtx},
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
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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
/// The raw value is a `u64`; [`LiteralRef::value`] masks it to `size` bytes.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Literal {
    /// Raw integer value (may be wider than `size` before masking).
    pub value: u64,
    /// Size of this constant in bytes.
    pub size: usize,
    /// Optional symbolic annotation (block address, function address, string).
    pub symbolic: Option<SymbolicRef>,
}

pub type LiteralRef<'str, 'ctx> = BaseRef<&'ctx Context<'str>, LiteralId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 'ctx, 'str> for LiteralRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'ctx Context<'str> {
        self.ctx
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx, Ctx> BaseRef<Ctx, LiteralId>
where
    Self: WithCtx<'s, 'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx Literal {
        &self.ctx().values.literals[self.id]
    }

    pub fn mask(&'s self) -> u64 {
        let size = self.inner().size;
        if size >= 8 {
            u64::MAX
        } else {
            (1u64 << (size * 8)) - 1
        }
    }

    pub fn value(&'s self) -> u64 {
        self.inner().value & self.mask()
    }
}

impl std::fmt::Display for LiteralRef<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let literal = &self.ctx.values.literals[self.id];
        match &literal.symbolic {
            Some(SymbolicRef::Block(bid)) => {
                let block = BlockRef::new(self.ctx, *bid);
                match block.name() {
                    Some(name) => write!(f, "&<{}>", name),
                    None => write!(f, "&<0x{:x}>", literal.value),
                }
            }
            Some(SymbolicRef::Function(fid)) => {
                let fn_ref = Function::from_id(self.ctx, *fid);
                write!(f, "&<{}>", fn_ref.name())
            }
            Some(SymbolicRef::String(s)) => write!(f, "&{:?}", s),
            None => write!(f, "0x{:x}", literal.value),
        }
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for LiteralRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        ValueId::Literal(self.id)
    }

    fn size(&self) -> usize {
        self.ctx.values.literals[self.id].size
    }
}
