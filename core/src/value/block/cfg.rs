// TODO: use a modified baseRef

use jstd::{
    Identifier,
    graph::{Cfg, FxBuildHasher},
};

use crate::{
    context::Context,
    value::{
        BasicBlock, BlockRef,
        function::FunctionRef,
        util::base_ref::{BaseRef, WithCtx, WithHost},
    },
};

/// Function-local block index (indexes the owning [`Function`]'s block arena).
#[derive(Identifier)]
pub struct LocalBlockId(u32);

/// Function-local CFG-edge index (indexes the owning [`Function`]'s edge arena).
#[derive(Identifier)]
pub struct LocalEdgeId(u32);

crate::composite_id!(BlockId, LocalBlockId);
crate::composite_id!(EdgeId, LocalEdgeId);

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct EdgeData {
    pub from: BlockId,
    pub to: BlockId,
}

impl<'s, 'ctx: 's, 'str: 'ctx, Ctx> BaseRef<Ctx, EdgeId>
where
    Self: WithCtx<'s, 'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx EdgeData {
        self.ctx().values.edge(self.id)
    }

    pub fn from(&'s self) -> BlockRef<'str, 'ctx> {
        let from = self.inner().from;
        BasicBlock::from_id(self.ctx(), from)
    }

    pub fn to(&'s self) -> BlockRef<'str, 'ctx> {
        let to = self.inner().to;
        BasicBlock::from_id(self.ctx(), to)
    }
}

pub type EdgeRef<'str, 'ctx> = BaseRef<&'ctx Context<'str>, EdgeId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 's, 'str> for EdgeRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'s Context<'str> {
        self.ctx
    }
}

pub type EdgeMutRef<'str, 'ctx> = BaseRef<&'ctx mut Context<'str>, EdgeId>;

impl<'str, 'ctx> EdgeMutRef<'str, 'ctx> {
    pub fn inner_mut(&mut self) -> &mut EdgeData {
        self.ctx.values.edge_mut(self.id)
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 's, 'str> for EdgeMutRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'s Context<'str> {
        self.ctx
    }
}

// ---------------------------------------------------------------------------
// A CFG rooted at a single function.
//
// The CFG is inherently per-function: its nodes are the function's own blocks.
// Dominator analysis needs only the successor relation (see jstd's `Cfg`), which
// [`BlockRef::successors`] already routes through the function's [`HostRef`], so
// it reads a *checked-out* function correctly inside a `FunctionPass`.
// ---------------------------------------------------------------------------

impl<'str, 'ctx> Cfg for FunctionRef<'str, 'ctx> {
    type NodeId = BlockId;

    // Fixed-seed hasher (not std's `RandomState`), so a block's incident-edge set
    // — and thus `successors()` iteration — is deterministic across runs; per-run
    // nondeterminism would otherwise leak into borderline mem2reg promotions and
    // hence into the lifted IR.
    type Hasher = FxBuildHasher;

    fn successors(&self, b: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        let succs: Vec<BlockId> = BlockRef::new(self.host(), b)
            .successors()
            .map(|(_, s)| s)
            .collect();
        succs.into_iter()
    }
}
