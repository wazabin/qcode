// TODO: maybe try to reuse some of num_ref's code here?

use crate::{
    context::{Context, Shared},
    value::ValueId,
};

/// A generic wrapper struct for referencing an ID with a context.
/// This is the basis for all `*Ref` and `*MutRef`.
///
/// `Id` is any cheap `Copy` handle — either a `Registry` `Identifier` (varnode,
/// literal, function, …) or a composite IR id (`InstructionId`, `BlockId`, …).
/// The bound is intentionally just `Copy`; routing to storage is the concern of
/// the concrete ref impls, not this wrapper.
pub struct BaseRef<Ctx, Id: Copy> {
    pub id: Id,
    pub(in crate::value) ctx: Ctx,
}

impl<Ctx, Id: Copy> BaseRef<Ctx, Id> {
    pub fn new(ctx: Ctx, id: Id) -> Self {
        BaseRef { id, ctx }
    }
}

impl<Ctx, Id: Copy + Into<ValueId>> BaseRef<Ctx, Id> {
    pub fn id(&self) -> ValueId {
        self.id.into()
    }
}

impl<'str, 'ctx, Id: Copy> BaseRef<&'ctx Context<'str>, Id> {
    pub fn from_id(ctx: &'ctx Context<'str>, id: Id) -> Self {
        BaseRef { id, ctx }
    }
}

/// Construct a shared-leaf wrapper ref (literal, bytes, varnode) over the
/// module's [`Shared`] IR state, taking **either** a `&Context` or a `&Shared`
/// (via [`AsShared`]) so module- and pass-scope callers share one spelling
/// (context-split stage 5b-ii item #1).
impl<'a, 'str, Id: Copy> BaseRef<&'a Shared<'str>, Id> {
    pub fn from_id(src: impl AsShared<'a, 'str>, id: Id) -> Self {
        BaseRef {
            id,
            ctx: src.as_shared(),
        }
    }
}

impl<'str, 'ctx, Id: Copy> BaseRef<&'ctx mut Context<'str>, Id> {
    pub fn from_id(ctx: &'ctx mut Context<'str>, id: Id) -> Self {
        BaseRef { id, ctx }
    }
}

/// Convert mutable refs to immutable refs by cloning the ID and sharing the context reference.
impl<'str, 'ctx, Id: Copy> From<BaseRef<&'ctx mut Context<'str>, Id>>
    for BaseRef<&'ctx Context<'str>, Id>
{
    fn from(r: BaseRef<&'ctx mut Context<'str>, Id>) -> Self {
        BaseRef {
            id: r.id,
            ctx: r.ctx,
        }
    }
}

pub trait WithCtx<'s, 'ctx: 's, 'str: 'ctx> {
    fn ctx(&'s self) -> &'ctx Context<'str>;
}

/// Narrows a module handle to its [`Shared`] IR state, so the shared-leaf
/// wrapper-ref constructors (`Varnode::from_id`, `Space::from_id`,
/// `LiteralRef`/`BytesRef`) accept **either** a whole `&Context` (module scope)
/// **or** a bare `&Shared` (pass scope, via `QCodeView::shared()` /
/// `PassBacking::shr()`) with no call-site churn (context-split stage 5b-ii item
/// #1). The leaf refs themselves store only a `&Shared`.
pub trait AsShared<'a, 'str> {
    // Implemented only for `Copy` reference types (`&Context`, `&Shared`), so
    // taking `self` by value is a cheap pointer copy, not an owning move.
    #[allow(clippy::wrong_self_convention)]
    fn as_shared(self) -> &'a Shared<'str>;
}

impl<'a, 'str> AsShared<'a, 'str> for &'a Shared<'str> {
    fn as_shared(self) -> &'a Shared<'str> {
        self
    }
}

impl<'a, 'str> AsShared<'a, 'str> for &'a Context<'str> {
    fn as_shared(self) -> &'a Shared<'str> {
        &self.shared
    }
}

/// The shared-leaf twin of [`WithCtx`]: a ref that can yield the module's
/// [`Shared`] IR state (interners, spaces, registers, name map) for its read
/// methods. Implemented by the shared-leaf wrapper refs (literal, bytes,
/// varnode) — which now carry only a `&Shared` — and by their `&mut Context`
/// mutation twins (which narrow through `self.ctx.shared`). This replaces the
/// leaf refs' old `WithCtx` dependency: a shared-leaf value never needs the
/// whole `&Context`, only its shared part (context-split stage 5b-ii item #1).
pub trait WithShared<'s, 'sh: 's, 'str: 'sh> {
    fn shared(&'s self) -> &'sh Shared<'str>;
}

pub trait WithCtxMut<'s, 'str: 's>: WithCtx<'s, 's, 'str> {
    fn ctx_mut(&'s mut self) -> &'s mut Context<'str>;
}
