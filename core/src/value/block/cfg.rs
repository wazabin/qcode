// TODO: use a modified baseRef

use std::collections::HashSet;

use jstd::{
    Identifier,
    graph::{Edge, FxBuildHasher, Graph, Node},
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
// A CFG graph rooted at a single function.
//
// The CFG is inherently per-function: its nodes are the function's own blocks
// and its edges the function's own edge arena. Implementing `Graph` on
// [`FunctionRef`] (rather than on `Context`, whose `nodes()` spanned every block
// in the module) both scopes the graph correctly and — because `FunctionRef`
// carries a [`HostRef`](crate::value::util::base_ref::HostRef) — reads a
// *checked-out* function through its host, so dominator analysis works inside a
// `FunctionPassV2`.
//
// jstd's `Node`/`Edge` hand `&'graph Self::Graph` around to build sibling views,
// so the node/edge handles borrow the `FunctionRef`; every arena access routes
// through `graph.host()`.
// ---------------------------------------------------------------------------

/// Node view for the per-function CFG [`Graph`].
pub struct CfgNode<'g, 'str, 'ctx> {
    graph: &'g FunctionRef<'str, 'ctx>,
    id: BlockId,
}

/// Edge view for the per-function CFG [`Graph`].
pub struct CfgEdge<'g, 'str, 'ctx> {
    graph: &'g FunctionRef<'str, 'ctx>,
    id: EdgeId,
}

impl<'g, 'str: 'g, 'ctx: 'g> Node<'g> for CfgNode<'g, 'str, 'ctx> {
    type Graph = FunctionRef<'str, 'ctx>;

    fn new(id: BlockId, graph: &'g FunctionRef<'str, 'ctx>) -> Self {
        Self { graph, id }
    }

    fn id(&self) -> BlockId {
        self.id
    }

    fn graph(&self) -> &'g FunctionRef<'str, 'ctx> {
        self.graph
    }

    fn edge_ids(&self) -> &'g HashSet<EdgeId, FxBuildHasher> {
        &self.graph.host().block(self.id).edges
    }

    fn edge_count(&self) -> usize {
        self.graph.host().block(self.id).edges.len()
    }
}

impl<'g, 'str: 'g, 'ctx: 'g> Edge<'g> for CfgEdge<'g, 'str, 'ctx> {
    type Graph = FunctionRef<'str, 'ctx>;

    fn new(id: EdgeId, graph: &'g FunctionRef<'str, 'ctx>) -> Self {
        Self { graph, id }
    }

    fn id(&self) -> EdgeId {
        self.id
    }

    fn graph(&self) -> &'g FunctionRef<'str, 'ctx> {
        self.graph
    }

    fn from_id(&self) -> BlockId {
        self.graph.host().edge(self.id).from
    }

    fn to_id(&self) -> BlockId {
        self.graph.host().edge(self.id).to
    }
}

impl<'str, 'ctx> Graph for FunctionRef<'str, 'ctx> {
    type NodeId = BlockId;
    type EdgeId = EdgeId;

    // Same fixed-seed hasher as the module `Graph for Context` impl, so a block's
    // incident-edge set iterates deterministically (see that impl's note).
    type Hasher = FxBuildHasher;

    type Node<'g>
        = CfgNode<'g, 'str, 'ctx>
    where
        Self: 'g;

    type Edge<'g>
        = CfgEdge<'g, 'str, 'ctx>
    where
        Self: 'g;

    fn get_node(&self, id: BlockId) -> Option<CfgNode<'_, 'str, 'ctx>> {
        Some(CfgNode { graph: self, id })
    }

    fn get_edge(&self, id: EdgeId) -> Option<CfgEdge<'_, 'str, 'ctx>> {
        Some(CfgEdge { graph: self, id })
    }

    fn nodes(&self) -> impl Iterator<Item = CfgNode<'_, 'str, 'ctx>> + '_ {
        self.block_ids()
            .into_iter()
            .map(move |id| CfgNode { graph: self, id })
    }

    fn edges(&self) -> impl Iterator<Item = CfgEdge<'_, 'str, 'ctx>> + '_ {
        self.edge_ids()
            .into_iter()
            .map(move |id| CfgEdge { graph: self, id })
    }
}
