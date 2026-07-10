// TODO: use a modified baseRef

use jstd::{
    Identifier,
    graph::{Cfg, FxBuildHasher},
};

use crate::value::{BlockRef, function::FunctionRef, util::base_ref::WithHost};

/// Function-local block index (indexes the owning [`Function`]'s block arena).
#[derive(Identifier)]
pub struct LocalBlockId(u32);

crate::composite_id!(BlockId, LocalBlockId);

/// Function-local CFG-edge index. A plain body-local id (stage 4): it indexes
/// the owning [`Function`]'s edge arena directly and carries **no** function
/// qualifier. Global addressing of an edge is the explicit pair
/// `(FunctionId, EdgeId)`; every edge is stored in its `from` block's function,
/// so the owning function is recoverable from either incident block.
#[derive(Identifier)]
pub struct EdgeId(u32);

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct EdgeData {
    pub from: BlockId,
    pub to: BlockId,
}

// A plain body-local [`EdgeId`] no longer self-describes its owning function, so
// the old whole-context `EdgeRef`/`EdgeMutRef` wrappers (which resolved
// `values.edge(id)` without a function) are gone. Edges are read through
// `Function::edge(id)` / `HostRef::edge(func, id)` with the owning function named
// explicitly. `EdgeData`'s `from`/`to` are still `BlockId`s, so an edge's
// endpoints resolve as blocks directly.

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
