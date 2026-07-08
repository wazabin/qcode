use crate::{
    context::Context,
    error::Result,
    types::TypeId,
    value::{
        Value, ValueId,
        block::{BasicBlock, BlockId, BlockRef},
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

/// Function-local block-parameter index (indexes the owning [`Function`]'s
/// param arena).
#[derive(Identifier)]
pub struct LocalParamId(u32);

crate::composite_id!(BlockParamId, LocalParamId);

/// A typed parameter declared at the entry of a basic block.
///
/// Block parameters are the receiving side of block arguments: when a
/// [`Branch`](crate::value::insn::Branch) or
/// [`CBranch`](crate::value::insn::CBranch) passes arguments to a target
/// block, the i-th argument binds to the i-th `BlockParam` of that block.
///
/// Unlike [`Instruction`](crate::value::Instruction) results, block params are
/// not produced by any operation — they are value sources at block entry,
/// analogous to function arguments in MLIR block-argument style.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlockParam<'str> {
    /// Position of this param in the owning block's param list.
    pub index: usize,

    /// The type of this parameter's value.
    pub type_id: TypeId,

    /// The block this parameter belongs to.
    pub parent: Option<BlockId>,

    /// Optional debug name (displayed as `%name`).
    pub name: Option<Cow<'str, str>>,

    /// Optional source value this param was created to promote (the varnode or
    /// stack-slot literal). Not displayed; it is a stable cross-run identity that
    /// lets passes like mem2reg reuse an existing param instead of duplicating it,
    /// even for varnodes that have no `name`.
    pub origin: Option<ValueId>,

    /// When `true`, the dead-param sweeps must not collect this param even while
    /// it has no users. It marks a real input caught mid-transformation — most
    /// notably the incoming stack pointer between `brighten` (which relabels its
    /// uses to `@stack_base`) and `lower_stack` (which relabels them back) — so a
    /// transient zero-user window is not mistaken for a dead argument.
    #[serde(default)]
    pub protected: bool,
}

impl<'str> BlockParam<'str> {
    /// Allocates a new block parameter in `ctx`, attaches it to `block_id`, and
    /// returns a mutable reference. The caller is responsible for appending the
    /// returned `BlockParamId` to the block's `params` list.
    pub fn make<'ctx>(
        ctx: &'ctx mut Context<'str>,
        block_id: BlockId,
        size: usize,
    ) -> BlockParamMutRef<'str, 'ctx> {
        let type_id = ctx.types.get_or_make_int(size);
        let index = ctx.values.block(block_id).params.len();
        let id = ctx.values.push_block_param(
            block_id.func,
            BlockParam {
                index,
                type_id,
                parent: Some(block_id),
                name: None,
                origin: None,
                protected: false,
            },
        );
        BlockParamMutRef::from_id(ctx, id)
    }

    pub fn from_id<'ctx>(ctx: &'ctx Context<'str>, id: BlockParamId) -> BlockParamRef<'str, 'ctx> {
        BlockParamRef::from_id(ctx, id)
    }

    pub fn from_id_mut<'ctx>(
        ctx: &'ctx mut Context<'str>,
        id: BlockParamId,
    ) -> BlockParamMutRef<'str, 'ctx> {
        BlockParamMutRef::from_id(ctx, id)
    }
}

// Shared read-only methods available on both BlockParamRef and BlockParamMutRef
impl<'s, 'ctx: 's, 'str: 'ctx, Ctx> BaseRef<Ctx, BlockParamId>
where
    Self: WithCtx<'s, 'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx BlockParam<'str> {
        self.ctx().values.block_param(self.id)
    }

    /// Position of this parameter in the owning block's param list.
    pub fn index(&'s self) -> usize {
        self.inner().index
    }

    /// The [`TypeId`] of this parameter's value.
    pub fn type_id(&'s self) -> TypeId {
        self.inner().type_id
    }

    /// Size of this parameter's value in bytes.
    pub fn size(&'s self) -> usize {
        self.ctx().types.size_of(self.inner().type_id)
    }

    /// The block this parameter belongs to, if any.
    pub fn parent(&'s self) -> Option<BlockRef<'str, 'ctx>> {
        self.inner()
            .parent
            .map(|id| BasicBlock::from_id(self.ctx(), id))
    }

    pub fn name(&'s self) -> Option<&'ctx str> {
        self.inner().name.as_deref()
    }

    /// The source value this param was created to promote, if recorded.
    pub fn origin(&'s self) -> Option<ValueId> {
        self.inner().origin
    }

    fn fmt(&'s self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Surface a richer-than-integer type (e.g. a seeded `TEB*` segment base)
        // as a `Type ` prefix. Plain `Int` params stay bare `@name` so the many
        // existing signature assertions (`<f @ESP @EDI>`) are unaffected.
        let types = &self.ctx().types;
        let ty = types.type_name(self.type_id());
        if types.pointee_of(self.type_id()).is_some()
            || types.struct_name_of(self.type_id()).is_some()
        {
            write!(f, "{ty} ")?;
        }
        if let Some(name) = self.name() {
            write!(f, "@{name}")
        } else {
            let id: usize = self.id.local.into();
            write!(f, "@param{id:x}")
        }
    }

    /// Formats this parameter in *declaration* position — the form that appears
    /// in a block header, `@name:iN`. Unlike the operand [`fmt`](Self::fmt), a
    /// scalar param surfaces its type as a `:iN`/`:fN` suffix so the header
    /// round-trips through the parser's `block_param_decl` rule. Pointer/struct
    /// params (no parser syntax) and unnamed/untyped params fall back to the
    /// operand rendering.
    pub(crate) fn fmt_decl(&'s self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let types = &self.ctx().types;
        let tid = self.type_id();
        let is_scalar = types.pointee_of(tid).is_none() && types.struct_name_of(tid).is_none();
        match (self.name(), is_scalar && self.size() > 0) {
            (Some(name), true) => write!(f, "@{name}:{}", types.type_name(tid)),
            _ => self.fmt(f),
        }
    }
}

pub type BlockParamRef<'str, 'ctx> = BaseRef<&'ctx Context<'str>, BlockParamId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 'ctx, 'str> for BlockParamRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'ctx Context<'str> {
        self.ctx
    }
}

impl Named for BlockParamRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        self.ctx.values.block_param(self.id).name.as_deref()
    }
}

impl Display for BlockParamRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt(f)
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for BlockParamRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        self.size()
    }
}

pub type BlockParamMutRef<'str, 'ctx> = BaseRef<&'ctx mut Context<'str>, BlockParamId>;

impl<'str, 'ctx> BlockParamMutRef<'str, 'ctx> {
    fn inner_mut(&mut self) -> &mut BlockParam<'str> {
        self.ctx.values.block_param_mut(self.id)
    }

    pub fn set_size(&mut self, size: usize) {
        let type_id = self.ctx.types.get_or_make_int(size);
        self.inner_mut().type_id = type_id;
    }

    /// Record the source value this param promotes (see [`BlockParam::origin`]).
    pub fn set_origin(&mut self, origin: ValueId) {
        self.inner_mut().origin = Some(origin);
    }

    pub fn constrain_size(&mut self, size: usize) {
        let current = self.size();
        if current == 0 {
            self.set_size(size);
        } else {
            assert_eq!(
                current, size,
                "block parameter size mismatch for {}: existing {} bytes, new {} bytes",
                self, current, size
            );
        }
    }

    pub fn as_ref(&self) -> BlockParamRef<'str, '_> {
        BlockParamRef::new(self.ctx, self.id)
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 's, 'str> for BlockParamMutRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'s Context<'str> {
        self.ctx
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtxMut<'s, 'str> for BlockParamMutRef<'str, 'ctx> {
    fn ctx_mut(&'s mut self) -> &'s mut Context<'str> {
        self.ctx
    }
}

impl Named for BlockParamMutRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        self.ctx.values.block_param(self.id).name.as_deref()
    }
}

impl Display for BlockParamMutRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt(f)
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for BlockParamMutRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        self.size()
    }
}

impl<'str, 'ctx> Renameable<'str, 'ctx> for BlockParamMutRef<'str, 'ctx> {
    fn rename(&mut self, name: Cow<'str, str>) -> Result<()> {
        let id = self.id.into();
        let old_name = self.inner_mut().name.take();
        update_context_name(id, self.ctx, name.clone(), old_name.as_deref())?;
        self.ctx.values.block_param_mut(self.id).name = Some(name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context::Context, value::BasicBlock};

    #[test]
    fn make_block_param_sets_index_and_size() {
        let mut ctx = Context::new();
        let block_id = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .id;

        let p0_id = BasicBlock::from_id_mut(&mut ctx, block_id).push_param(8).id;
        let p1_id = BasicBlock::from_id_mut(&mut ctx, block_id).push_param(4).id;

        let p0 = BlockParam::from_id(&ctx, p0_id);
        let p1 = BlockParam::from_id(&ctx, p1_id);
        assert_eq!(p0.index(), 0);
        assert_eq!(p0.size(), 8);
        assert_eq!(p1.index(), 1);
        assert_eq!(p1.size(), 4);
    }

    #[test]
    fn block_param_display_uses_name_when_set() {
        let mut ctx = Context::new();
        let block_id = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .id;
        let p_id = BasicBlock::from_id_mut(&mut ctx, block_id).push_param(8).id;

        let mut p = BlockParam::from_id_mut(&mut ctx, p_id);
        p.rename("myval".into()).expect("rename ok");
        assert_eq!(p.to_string(), "@myval");
    }

    #[test]
    fn block_param_display_fallback_when_unnamed() {
        let mut ctx = Context::new();
        let block_id = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .id;
        let p_id = BasicBlock::from_id_mut(&mut ctx, block_id).push_param(4).id;
        let p = BlockParam::from_id(&ctx, p_id);
        let s = p.to_string();
        assert!(s.starts_with("@param"), "expected @param<hex>, got {s}");
    }
}
