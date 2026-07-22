//! Stable identity for a reconstruction obligation: an unresolved control
//! transfer that reconstruction still owes an answer for.
//!
//! This module holds *only* the identity. The obligation database, its states,
//! and its evidence live in `qcode_analysis` — they are derived analysis state
//! and must not become part of serialized qcode bodies (see the roadmap's
//! "Reconstruction facts and hypotheses" section).
//!
//! Like [`DiscoveryKey`](crate::discovery::DiscoveryKey), the key is built from
//! stable binary addresses rather than context-local `FunctionId` / `BlockId` /
//! `InstructionId` values. Obligations are produced while analyzing a disposable
//! optimized clone and consumed by the scheduler against the persistent clean
//! context, so an arena ID from one context would be meaningless — or worse,
//! silently valid — in the other.

use crate::discovery::Address;

/// The kind of control transfer an obligation stands for.
///
/// Part of the key, not just a payload: the same address cannot host both, but
/// keeping the kind in the identity means a record carries what it is without a
/// lookup, and it matches [`DiscoveryKind`](crate::discovery::DiscoveryKind)'s
/// precedent of typing the work item.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum TransferKind {
    /// An indirect branch — `Mnemonic::BranchInd`. Resolvable today by
    /// `handle_jump_tables`.
    IndirectBranch,
    /// An indirect call — `Mnemonic::CallInd`. No resolver exists yet; these
    /// obligations stay pending until candidate-target analysis lands.
    IndirectCall,
}

impl TransferKind {
    /// Whether any pass can currently attempt to resolve this kind of transfer.
    ///
    /// `IndirectCall` obligations are recorded for completeness and reporting
    /// but are inert: nothing attempts them, so a permanently-pending indirect
    /// call is expected, not a bug. Reporting uses this to avoid presenting
    /// inert obligations as reconstruction failures.
    pub fn has_resolver(self) -> bool {
        match self {
            TransferKind::IndirectBranch => true,
            TransferKind::IndirectCall => false,
        }
    }
}

/// Stable identity of a reconstruction obligation.
///
/// Keyed on the address of the transfer *instruction*, not its block. A block's
/// start address is not stable under the straight-line merging the optimized
/// clone performs (the jump-table pass already works around this — see its
/// `dispatch_source_addr` helper), whereas the instruction address is fixed by
/// the binary.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct ObligationKey {
    /// Address of the indirect transfer instruction itself.
    pub site: Address,
    pub kind: TransferKind,
}

impl ObligationKey {
    pub fn branch(site: Address) -> Self {
        Self {
            site,
            kind: TransferKind::IndirectBranch,
        }
    }

    pub fn call(site: Address) -> Self {
        Self {
            site,
            kind: TransferKind::IndirectCall,
        }
    }
}

impl std::fmt::Display for ObligationKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.kind {
            TransferKind::IndirectBranch => "branch",
            TransferKind::IndirectCall => "call",
        };
        write!(f, "{kind}@{:#x}", self.site)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_participates_in_identity() {
        assert_ne!(ObligationKey::branch(0x1000), ObligationKey::call(0x1000));
    }

    #[test]
    fn same_site_and_kind_is_the_same_obligation() {
        assert_eq!(ObligationKey::branch(0x1000), ObligationKey::branch(0x1000));
    }

    #[test]
    fn only_indirect_branches_have_a_resolver_today() {
        assert!(TransferKind::IndirectBranch.has_resolver());
        assert!(!TransferKind::IndirectCall.has_resolver());
    }

    #[test]
    fn ordering_groups_by_site_then_kind() {
        let mut keys = vec![
            ObligationKey::call(0x2000),
            ObligationKey::branch(0x2000),
            ObligationKey::branch(0x1000),
        ];
        keys.sort();
        assert_eq!(
            keys,
            vec![
                ObligationKey::branch(0x1000),
                ObligationKey::branch(0x2000),
                ObligationKey::call(0x2000),
            ]
        );
    }

    #[test]
    fn display_is_stable_and_readable() {
        assert_eq!(
            ObligationKey::branch(0x401a60).to_string(),
            "branch@0x401a60"
        );
        assert_eq!(ObligationKey::call(0x401a60).to_string(), "call@0x401a60");
    }
}
