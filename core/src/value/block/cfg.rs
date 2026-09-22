// TODO: use a modified baseRef

use jstd::{
    Identifier,
    graph::{Cfg, FxBuildHasher},
};

use crate::value::{BlockRef, QCodeView, function::FunctionRef};

/// Function-local block index (indexes the owning [`FunctionBody`](crate::value::FunctionBody)'s block arena).
#[derive(Identifier)]
pub struct LocalBlockId(u32);

crate::composite_id!(BlockId, LocalBlockId);

/// Function-local CFG-edge index. A plain body-local id (stage 4): it indexes
/// the owning [`FunctionBody`](crate::value::FunctionBody)'s edge arena directly and carries **no** function
/// qualifier. Global addressing of an edge is the explicit pair
/// `(FunctionId, EdgeId)`; every edge is stored in its `from` block's function,
/// so the owning function is recoverable from either incident block.
#[derive(Identifier)]
pub struct EdgeId(u32);

/// The edges incident to one block, as a small set: a block has one or two
/// successors and a few predecessors, so the ids live inline and a lookup
/// is a scan — no allocation for the block a lift makes per instruction,
/// and no hashing. A block with hundreds of edges (a resolved jump table)
/// pays a linear insert each; it is rare, and built once.
///
/// Iteration is in insertion order, which is deterministic.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EdgeSet(smallvec::SmallVec<[EdgeId; 3]>);

impl serde::Serialize for EdgeSet {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.as_slice().serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for EdgeSet {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Vec::<EdgeId>::deserialize(deserializer).map(|edges| Self(edges.into_iter().collect()))
    }
}

impl EdgeSet {
    /// Adds `edge`; whether it was not there already.
    pub fn insert(&mut self, edge: EdgeId) -> bool {
        if self.0.contains(&edge) {
            return false;
        }
        self.0.push(edge);
        true
    }

    /// Removes `edge`; whether it was there.
    pub fn remove(&mut self, edge: &EdgeId) -> bool {
        match self.0.iter().position(|e| e == edge) {
            Some(at) => {
                self.0.remove(at);
                true
            }
            None => false,
        }
    }

    pub fn contains(&self, edge: &EdgeId) -> bool {
        self.0.contains(edge)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, EdgeId> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit();
    }
}

impl<'a> IntoIterator for &'a EdgeSet {
    type Item = &'a EdgeId;
    type IntoIter = std::slice::Iter<'a, EdgeId>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct EdgeData {
    pub from: LocalBlockId,
    pub to: LocalBlockId,
}

// A plain body-local [`EdgeId`] no longer self-describes its owning function, so
// the old whole-context `EdgeRef`/`EdgeMutRef` wrappers (which resolved
// `values.edge(id)` without a function) are gone. Edges are read through
// `FunctionBody::edge(id)` / `QCodeView::edge(func, id)` with the owning function named
// explicitly. Strict locality (context-split ruling 2) guarantees both endpoints
// live in the edge's own function, so `from`/`to` are bare `LocalBlockId`s;
// qualify with the owning function (which every reader names) to get a composite
// `BlockId`.

// ---------------------------------------------------------------------------
// A CFG rooted at a single function.
//
// The CFG is inherently per-function: its nodes are the function's own blocks.
// Dominator analysis needs only the successor relation (see jstd's `Cfg`), which
// [`BlockRef::successors`] already routes through the function's [`QCodeView`], so
// it reads a *checked-out* function correctly inside a `FunctionPass`.
// ---------------------------------------------------------------------------

impl<'str: 'ctx, 'ctx, R> Cfg for FunctionRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    type NodeId = BlockId;

    // Fixed-seed hasher (not std's `RandomState`), so a block's incident-edge set
    // — and thus `successors()` iteration — is deterministic across runs; per-run
    // nondeterminism would otherwise leak into borderline mem2reg promotions and
    // hence into the lifted IR.
    type Hasher = FxBuildHasher;

    fn successors(&self, b: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        let succs: Vec<BlockId> = BlockRef::new(self.view, b)
            .successors()
            .map(|(_, s)| s)
            .collect();
        succs.into_iter()
    }
}
