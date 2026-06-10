//! Heuristic *assumptions* made during analysis, and a [`AssumptionKnowledge`]
//! base that survives checkpoint+replay.
//!
//! An analysis pass may make a *reasonable assumption* about a value before it
//! has been proven — for example, that a called function returns normally to the
//! instruction after the `call`. Each such guess is recorded in a side ledger
//! (an arena on [`ValueRegistry`](crate::value::registry::ValueRegistry)), keyed
//! by the entity it predicts (a callee [`FunctionId`]). When that entity is
//! later analyzed, a verification pass flips the assumption to
//! [`Confirmed`](AssumptionStatus::Confirmed) or
//! [`Violated`](AssumptionStatus::Violated).
//!
//! Because analysis passes mutate the [`Context`](crate::context::Context) arena
//! in place, a *wrong* assumption leaves behind IR that is now incorrect. The
//! invalidation strategy is **checkpoint + replay**: a freshly-lifted baseline
//! `Context` is cloned before any speculative analysis, and on a violation the
//! whole pipeline is replayed from that baseline with the newly-learned fact
//! pinned in an [`AssumptionKnowledge`]. Knowledge only ever grows, so replay
//! terminates.

use std::collections::HashSet;

use jstd::Identifier;

use crate::value::{block::BlockId, block::EdgeId, function::FunctionId, insn::InstructionId};

/// Identifies an [`Assumption`] stored in the ledger.
#[derive(Identifier)]
pub struct AssumptionId(usize);

/// Verification state of an [`Assumption`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AssumptionStatus {
    /// Recorded but not yet checked against the predicted entity.
    Unverified,
    /// Checked and found to hold.
    Confirmed,
    /// Checked and found to be wrong; whatever was derived from it must be
    /// discarded (see the module-level checkpoint+replay note).
    Violated,
}

/// The heuristic kinds we can assume.
///
/// `#[non_exhaustive]` so adding future heuristics does not break exhaustive
/// matches in downstream crates.
#[non_exhaustive]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AssumptionKind {
    /// A called function returns normally to the fall-through after the call.
    CallReturns(CallReturnsAssumption),
}

/// "Function `callee` returns to the instruction after the call at `call_site`."
///
/// Materialized as a real CFG edge `call_block -> continuation` so downstream
/// dataflow can reason across the call.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CallReturnsAssumption {
    /// The function we assume returns — the primary verification key.
    pub callee: FunctionId,
    /// The `Call` instruction the assumption is about.
    pub call_site: InstructionId,
    /// The block that ends in the call.
    pub call_block: BlockId,
    /// The fall-through block we assume control returns to.
    pub continuation: BlockId,
    /// The synthetic CFG edge `call_block -> continuation` we added.
    pub continuation_edge: EdgeId,
}

/// A recorded heuristic guess together with its verification [`AssumptionStatus`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Assumption {
    pub kind: AssumptionKind,
    pub status: AssumptionStatus,
}

impl Assumption {
    /// Creates an [`Unverified`](AssumptionStatus::Unverified) assumption.
    pub fn new(kind: AssumptionKind) -> Self {
        Self {
            kind,
            status: AssumptionStatus::Unverified,
        }
    }

    /// The callee this assumption predicts, used to index it for verification.
    pub fn callee(&self) -> Option<FunctionId> {
        match &self.kind {
            AssumptionKind::CallReturns(a) => Some(a.callee),
        }
    }
}

/// Facts proven across checkpoint+replay rounds, kept *outside* the snapshotted
/// [`Context`](crate::context::Context) so they persist when the working copy is
/// discarded.
///
/// Keyed by [`FunctionId`], which is stable across `Context::clone` (registry
/// IDs are indices into the immutable baseline). The set only grows, which is
/// what guarantees the replay loop converges.
#[derive(Debug, Clone, Default)]
pub struct AssumptionKnowledge {
    /// Callees proven not to return (noreturn): `exit`/`abort`, infinite loops,
    /// or any function whose body contains no `Return`.
    pub noreturn: HashSet<FunctionId>,
}

impl AssumptionKnowledge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `f` has been proven not to return.
    pub fn is_noreturn(&self, f: FunctionId) -> bool {
        self.noreturn.contains(&f)
    }
}
