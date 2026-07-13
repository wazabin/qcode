//! Typed queue of code addresses discovered during lifting/analysis but not yet
//! lifted into the IR.
//!
//! Durable discovery records use stable addresses, not context-local `FunctionId`
//! or `BlockId` values. The queue is shared between a raw clean context and
//! disposable optimized clones, so IDs from one context must not leak into the
//! other.

use std::collections::BTreeMap;

pub type Address = u64;

/// Why an address is believed to start a function.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum FunctionDiscoveryReason {
    Entry,
    CrtMain,
    CallTarget,
    TailCall,
    UserSeed,
}

/// The control-flow edge that exposed a block target.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum EdgeKind {
    Entry,
    DirectBranch,
    ConditionalBranch,
    Fallthrough,
    CallTarget,
    CallFallthrough,
    JumpTableTarget,
    TailCall,
    Speculative,
}

/// Whether a discovered address starts a new function or extends an existing one.
#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum DiscoveryKind {
    Function {
        reason: FunctionDiscoveryReason,
    },
    Block {
        /// Entry address of the owning function.
        function: Address,
        edge_kind: EdgeKind,
    },
}

/// Stable identity of a discovery item.
#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct DiscoveryKey {
    pub target: Address,
    pub kind: DiscoveryKind,
    /// Source block for edge-bearing discoveries. Two transfers to the same
    /// target are distinct work items because the lifter must attach both
    /// source edges after the target exists.
    pub source_block: Option<Address>,
}

/// Why this discovery exists. Kept as debugging/UI metadata that records how a
/// discovered address came to be queued.
#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DiscoveryProvenance {
    LoaderEntry,

    DirectLift {
        source_addr: Address,
    },

    Optimization {
        pass: String,
        assumption: Option<String>,
    },

    UserSeed,

    #[default]
    Unknown,
}
/// A code address discovered but not yet lifted.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Discovery {
    pub target: Address,
    pub kind: DiscoveryKind,
    /// Symbol-name hint for the discovered code, when known.
    pub name: Option<String>,
    /// Stable source metadata. These are addresses so the item can move between
    /// clean and optimized contexts.
    pub source_function: Option<Address>,
    pub source_block: Option<Address>,
    pub source_addr: Option<Address>,
    pub provenance: DiscoveryProvenance,
}

impl Discovery {
    pub fn function(target: Address) -> Self {
        Self {
            target,
            kind: DiscoveryKind::Function {
                reason: FunctionDiscoveryReason::CallTarget,
            },
            name: None,
            source_function: None,
            source_block: None,
            source_addr: None,
            provenance: DiscoveryProvenance::Unknown,
        }
    }

    pub fn entry(target: Address) -> Self {
        Self::function(target)
            .with_function_reason(FunctionDiscoveryReason::Entry)
            .with_provenance(DiscoveryProvenance::LoaderEntry)
    }

    pub fn block(target: Address, function: Address) -> Self {
        Self {
            target,
            kind: DiscoveryKind::Block {
                function,
                edge_kind: EdgeKind::DirectBranch,
            },
            name: None,
            source_function: Some(function),
            source_block: None,
            source_addr: None,
            provenance: DiscoveryProvenance::Unknown,
        }
    }

    pub fn key(&self) -> DiscoveryKey {
        DiscoveryKey {
            target: self.target,
            kind: self.kind.clone(),
            source_block: self.source_block,
        }
    }

    pub fn with_name(mut self, name: Option<String>) -> Self {
        self.name = name;
        self
    }

    pub fn from_addr(mut self, addr: Address) -> Self {
        self.source_addr = Some(addr);
        if matches!(self.provenance, DiscoveryProvenance::Unknown) {
            self.provenance = DiscoveryProvenance::DirectLift { source_addr: addr };
        }
        self
    }

    pub fn from_block_addr(mut self, addr: Address) -> Self {
        self.source_block = Some(addr);
        self
    }

    pub fn with_edge_kind(mut self, edge_kind: EdgeKind) -> Self {
        if let DiscoveryKind::Block {
            edge_kind: existing,
            ..
        } = &mut self.kind
        {
            *existing = edge_kind;
        }
        self
    }

    pub fn with_function_reason(mut self, reason: FunctionDiscoveryReason) -> Self {
        if let DiscoveryKind::Function {
            reason: existing, ..
        } = &mut self.kind
        {
            *existing = reason;
        }
        self
    }

    pub fn with_provenance(mut self, provenance: DiscoveryProvenance) -> Self {
        self.provenance = provenance;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DiscoveryState {
    Pending,
    Lifted,
    Failed { reason: String },
    Skipped { reason: String },
}

/// The pending-discovery queue and durable outcome state.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DiscoveryQueue {
    pending: BTreeMap<DiscoveryKey, Discovery>,
    states: BTreeMap<DiscoveryKey, DiscoveryState>,
}

impl DiscoveryQueue {
    /// Record a discovery unless this exact key already has a terminal outcome.
    pub fn insert(&mut self, discovery: Discovery) -> bool {
        let key = discovery.key();
        if matches!(
            self.states.get(&key),
            Some(
                DiscoveryState::Lifted
                    | DiscoveryState::Failed { .. }
                    | DiscoveryState::Skipped { .. }
            )
        ) {
            return false;
        }
        let inserted = !self.pending.contains_key(&key);
        self.pending.entry(key.clone()).or_insert(discovery);
        self.states.entry(key).or_insert(DiscoveryState::Pending);
        inserted
    }

    pub fn drain(&mut self) -> Vec<Discovery> {
        std::mem::take(&mut self.pending).into_values().collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Discovery> + '_ {
        self.pending.values()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn mark_lifted(&mut self, key: DiscoveryKey) {
        self.states.insert(key, DiscoveryState::Lifted);
    }

    pub fn mark_failed(&mut self, key: DiscoveryKey, reason: impl Into<String>) {
        self.states.insert(
            key,
            DiscoveryState::Failed {
                reason: reason.into(),
            },
        );
    }

    pub fn mark_skipped(&mut self, key: DiscoveryKey, reason: impl Into<String>) {
        self.states.insert(
            key,
            DiscoveryState::Skipped {
                reason: reason.into(),
            },
        );
    }

    pub fn state(&self, key: &DiscoveryKey) -> Option<&DiscoveryState> {
        self.states.get(key)
    }

    pub fn states(&self) -> impl Iterator<Item = (&DiscoveryKey, &DiscoveryState)> + '_ {
        self.states.iter()
    }

    /// Every code address this queue lifted successfully, as a portable
    /// [`CodeSeed`]. Exported from one analysis run and replayed into the next
    /// (via [`Context::seed_code`](crate::context::Context::seed_code)) so the
    /// lifter reaches jump-table targets in its first pass instead of waiting for
    /// the analysis fixpoint to discover them round by round.
    pub fn lifted_seeds(&self) -> Vec<CodeSeed> {
        self.states
            .iter()
            .filter(|(_, state)| matches!(state, DiscoveryState::Lifted))
            .map(|(key, _)| CodeSeed::from_key(key))
            .collect()
    }
}

/// A code address known to lift, captured from one analysis run to pre-seed the
/// next. Mirrors a [`DiscoveryKey`] but drops the run-specific bits the lifter
/// reconstructs on its own (the precise CFG edge is re-attached by the pass that
/// originally found the target).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CodeSeed {
    /// A discovered function entry.
    Function { target: Address },
    /// A block at `target` inside the function entered at `function`.
    Block {
        function: Address,
        target: Address,
        edge_kind: EdgeKind,
    },
}

impl CodeSeed {
    fn from_key(key: &DiscoveryKey) -> Self {
        match key.kind {
            DiscoveryKind::Function { .. } => CodeSeed::Function { target: key.target },
            DiscoveryKind::Block {
                function,
                edge_kind,
            } => CodeSeed::Block {
                function,
                target: key.target,
                edge_kind,
            },
        }
    }

    /// Rebuild the [`Discovery`] to enqueue, attributed to [`UserSeed`] so the
    /// origin stays honest rather than impersonating the pass that first found it.
    ///
    /// [`UserSeed`]: DiscoveryProvenance::UserSeed
    pub fn into_discovery(self) -> Discovery {
        match self {
            CodeSeed::Function { target } => {
                Discovery::function(target).with_provenance(DiscoveryProvenance::UserSeed)
            }
            CodeSeed::Block {
                function,
                target,
                edge_kind,
            } => Discovery::block(target, function)
                .with_edge_kind(edge_kind)
                .with_provenance(DiscoveryProvenance::UserSeed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedups_exact_key_not_target() {
        let mut q = DiscoveryQueue::default();
        assert!(q.insert(Discovery::block(0x1000, 0x900)));
        assert!(!q.insert(Discovery::block(0x1000, 0x900)));
        assert!(q.insert(Discovery::block(0x1000, 0x900).from_block_addr(0x910)));
        assert!(q.insert(Discovery::block(0x1000, 0x900).from_block_addr(0x920)));
        assert!(q.insert(Discovery::block(0x1000, 0xa00)));
        assert!(q.insert(Discovery::function(0x1000)));

        let all: Vec<_> = q.iter().collect();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn terminal_state_blocks_requeue_of_same_key() {
        let mut q = DiscoveryQueue::default();
        let d = Discovery::function(0x10);
        let key = d.key();
        assert!(q.insert(d.clone()));
        q.drain();
        q.mark_failed(key, "decode");
        assert!(!q.insert(d));
        assert!(q.is_empty());
    }

    #[test]
    fn terminal_target_from_one_source_does_not_suppress_another_source() {
        let mut q = DiscoveryQueue::default();
        let first = Discovery::block(0x1000, 0x900).from_block_addr(0x910);
        let second = Discovery::block(0x1000, 0x900).from_block_addr(0x920);
        let first_key = first.key();

        assert!(q.insert(first));
        q.drain();
        q.mark_lifted(first_key);

        assert!(q.insert(second));
        assert_eq!(q.pending_len(), 1);
    }

    #[test]
    fn drain_empties_pending_but_keeps_state() {
        let mut q = DiscoveryQueue::default();
        let d = Discovery::function(0x10);
        let key = d.key();
        q.insert(d);
        let drained = q.drain();
        assert_eq!(drained.len(), 1);
        assert!(q.is_empty());
        assert_eq!(q.state(&key), Some(&DiscoveryState::Pending));
    }
}
