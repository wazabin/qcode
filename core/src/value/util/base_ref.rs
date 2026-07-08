// TODO: maybe try to reuse some of num_ref's code here?

use crate::{context::Context, value::ValueId};

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

pub trait WithCtxMut<'s, 'str: 's>: WithCtx<'s, 's, 'str> {
    fn ctx_mut(&'s mut self) -> &'s mut Context<'str>;
}
