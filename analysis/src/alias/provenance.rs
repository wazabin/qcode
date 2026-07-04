//! A small provenance lattice for frame-freshness disjointness.
//!
//! Every pointer value the frame-freshness rules reason about is classified,
//! once and memoized, into a union of provenance flags describing *where the
//! address could have come from*: this function's own frame, the caller's
//! frame, the static image, an incoming parameter, or a value loaded from
//! memory. [`AliasResult::provably_disjoint`](super::AliasResult::provably_disjoint)
//! then answers disjointness by inspecting the two operands' provenance,
//! replacing the earlier zoo of ad-hoc recursive predicates.
//!
//! The classifier lives on [`FrameInfo`](super::FrameInfo); this module only
//! defines the lattice element.

/// A union of provenance flags. `Provenance::default()` is the empty set (no
/// bits) — used as the identity for [`Provenance::union`] and, transiently, as
/// the placeholder for an in-progress value while a peel cycle is being broken.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct Provenance(u8);

impl Provenance {
    /// `@SP`-rooted slot below entry SP ([`FrameClass::Local`]), or a
    /// realigned-frame slot.
    ///
    /// [`FrameClass::Local`]: crate::stack::frame::FrameClass::Local
    pub(crate) const OWN_FRAME: Self = Self(1 << 0);
    /// `@SP`-rooted slot at or above entry SP ([`FrameClass::CallerFrame`]),
    /// including the bare `@SP` param itself.
    ///
    /// [`FrameClass::CallerFrame`]: crate::stack::frame::FrameClass::CallerFrame
    pub(crate) const CALLER_FRAME: Self = Self(1 << 1);
    /// A fixed absolute address: a literal, a globalized-global param
    /// (`@glob_<addr>`), or affine arithmetic over either.
    pub(crate) const GLOBAL_STATIC: Self = Self(1 << 2);
    /// Peels to a non-`@SP` root-block parameter — a caller-supplied pointer.
    pub(crate) const INPUT: Self = Self(1 << 3);
    /// An actual `Load` on the peel path, or a partial-promotion *snapshot*
    /// param (`origin` = a global-static slot param).
    pub(crate) const LOADED: Self = Self(1 << 4);
    /// Anything unclassifiable (calls, non-affine pcode, multiplies, …).
    pub(crate) const OPAQUE: Self = Self(1 << 5);

    /// Set union (bit-or).
    pub(crate) fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether every bit in `flag` is set in `self`.
    pub(crate) fn contains(self, flag: Self) -> bool {
        self.0 & flag.0 == flag.0
    }

    /// Whether `self` is exactly `flag` — the flag and nothing else.
    pub(crate) fn is_pure(self, flag: Self) -> bool {
        self.0 == flag.0
    }

    /// Whether `self` is nonempty and every set bit is within `mask`.
    pub(crate) fn is_nonempty_subset_of(self, mask: Self) -> bool {
        self.0 != 0 && self.0 & !mask.0 == 0
    }

    /// The empty provenance (no bits).
    pub(crate) fn is_empty(self) -> bool {
        self.0 == 0
    }
}
