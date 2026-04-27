use crate::{
    context::Context,
    error::Result,
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

#[derive(Identifier)]
pub struct BlockParamId(usize);

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
#[derive(Debug, Clone)]
pub struct BlockParam<'str> {
    /// Position of this param in the owning block's param list.
    pub index: usize,

    /// Size of the value in bytes.
    pub size: usize,

    /// The block this parameter belongs to.
    pub parent: Option<BlockId>,

    /// Optional debug name (displayed as `%name`).
    pub name: Option<Cow<'str, str>>,
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
        let index = ctx.values.basic_blocks[block_id].params.len();
        let id = ctx.values.block_params.push(BlockParam {
            index,
            size,
            parent: Some(block_id),
            name: None,
        });
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
        &self.ctx().values.block_params[self.id]
    }

    /// Position of this parameter in the owning block's param list.
    pub fn index(&'s self) -> usize {
        self.inner().index
    }

    /// Size of this parameter's value in bytes.
    pub fn size(&'s self) -> usize {
        self.inner().size
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

    fn fmt(&'s self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = self.name() {
            write!(f, "@{name}")
        } else {
            let id: usize = self.id.into();
            write!(f, "@param{id:x}")
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
        self.ctx.values.block_params[self.id].name.as_deref()
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
        &mut self.ctx.values.block_params[self.id]
    }

    pub fn set_size(&mut self, size: usize) {
        self.inner_mut().size = size;
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
        self.ctx.values.block_params[self.id].name.as_deref()
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
    fn rename(&mut self, name: Cow<'str, str>) -> Result<'str, ()> {
        let id = self.id.into();
        let old_name = self.inner_mut().name.take();
        update_context_name(id, self.ctx, name.clone(), old_name.as_deref())?;
        self.ctx.values.block_params[self.id].name = Some(name);
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
        let block_id = BasicBlock::make(&mut ctx).id;

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
        let block_id = BasicBlock::make(&mut ctx).id;
        let p_id = BasicBlock::from_id_mut(&mut ctx, block_id).push_param(8).id;

        let mut p = BlockParam::from_id_mut(&mut ctx, p_id);
        p.rename("myval".into()).expect("rename ok");
        assert_eq!(p.to_string(), "@myval");
    }

    #[test]
    fn block_param_display_fallback_when_unnamed() {
        let mut ctx = Context::new();
        let block_id = BasicBlock::make(&mut ctx).id;
        let p_id = BasicBlock::from_id_mut(&mut ctx, block_id).push_param(4).id;
        let p = BlockParam::from_id(&ctx, p_id);
        let s = p.to_string();
        assert!(s.starts_with("@param"), "expected @param<hex>, got {s}");
    }
}
