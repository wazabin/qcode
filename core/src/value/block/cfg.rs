// TODO: use a modified baseRef

use std::collections::HashSet;

use jstd::{
    Identifier,
    graph::{Edge, EdgeMut, Node, NodeMut},
};

use crate::{
    context::Context,
    value::{
        BlockMutRef, BlockRef,
        util::base_ref::{BaseRef, WithCtx},
    },
};

#[derive(Identifier)]
pub struct BlockId(usize);

#[derive(Identifier)]
pub struct EdgeId(usize);

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
        &self.ctx().values.edges[self.id]
    }

    pub fn from(&'s self) -> BlockRef<'str, 'ctx> {
        let from = self.inner().from;
        BlockRef::new(self.ctx(), from)
    }

    pub fn to(&'s self) -> BlockRef<'str, 'ctx> {
        let to = self.inner().to;
        BlockRef::new(self.ctx(), to)
    }
}

pub type EdgeRef<'str, 'ctx> = BaseRef<&'ctx Context<'str>, EdgeId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 's, 'str> for EdgeRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'s Context<'str> {
        self.ctx
    }
}

impl<'str, 'ctx> Edge<'ctx> for EdgeRef<'str, 'ctx> {
    type Graph = Context<'str>;

    fn id(&self) -> EdgeId {
        self.id
    }

    fn new(id: EdgeId, graph: &'ctx Context<'str>) -> Self {
        Self { id, ctx: graph }
    }

    fn graph(&self) -> &'ctx Context<'str> {
        self.ctx
    }

    fn from_id(&self) -> <Self::Graph as jstd::graph::Graph>::NodeId {
        self.inner().from
    }

    fn to_id(&self) -> <Self::Graph as jstd::graph::Graph>::NodeId {
        self.inner().to
    }
}

pub type EdgeMutRef<'str, 'ctx> = BaseRef<&'ctx mut Context<'str>, EdgeId>;

impl<'str, 'ctx> EdgeMutRef<'str, 'ctx> {
    pub fn inner_mut(&mut self) -> &mut EdgeData {
        &mut self.ctx.values.edges[self.id]
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 's, 'str> for EdgeMutRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'s Context<'str> {
        self.ctx
    }
}

impl<'str, 'ctx> EdgeMut<'ctx> for EdgeMutRef<'str, 'ctx> {
    type Graph = Context<'str>;

    fn id(&self) -> EdgeId {
        self.id
    }

    fn new(id: EdgeId, graph: &'ctx mut Context<'str>) -> Self {
        Self { id, ctx: graph }
    }

    fn graph(&mut self) -> &mut Context<'str> {
        self.ctx
    }

    fn from_id(&self) -> BlockId {
        self.inner().from
    }

    fn to_id(&self) -> BlockId {
        self.inner().to
    }

    fn set_from(&mut self, node: BlockId) {
        self.inner_mut().from = node;
    }
}

impl<'str, 'ctx> Node<'ctx> for BlockRef<'str, 'ctx> {
    type Graph = Context<'str>;

    fn id(&self) -> BlockId {
        self.id
    }

    fn new(id: BlockId, graph: &'ctx Context<'str>) -> Self {
        Self::from_id(graph, id)
    }

    fn graph(&self) -> &'ctx Context<'str> {
        self.ctx
    }

    fn edge_ids(&self) -> &'ctx HashSet<EdgeId> {
        &self.ctx.values.basic_blocks[self.id].edges
    }

    fn edge_count(&self) -> usize {
        self.ctx.values.basic_blocks[self.id].edges.len()
    }
}

impl<'str, 'ctx> NodeMut<'ctx> for BlockMutRef<'str, 'ctx> {
    type Graph = Context<'str>;

    fn new(id: BlockId, graph: &'ctx mut Context<'str>) -> Self {
        Self::from_id(graph, id)
    }

    fn id(&self) -> BlockId {
        self.id
    }

    fn graph(&mut self) -> &mut Context<'str> {
        self.ctx
    }

    fn edge_ids(&self) -> &HashSet<EdgeId> {
        &self.ctx.values.basic_blocks[self.id].edges
    }

    fn edge_count(&self) -> usize {
        self.ctx.values.basic_blocks[self.id].edges.len()
    }

    fn add_edge_id(&mut self, edge: EdgeId) {
        self.ctx.values.basic_blocks[self.id].edges.insert(edge);
    }

    fn remove_edge_id(&mut self, edge: EdgeId) {
        self.ctx.values.basic_blocks[self.id].edges.remove(&edge);
    }
}
