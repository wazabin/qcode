//! The precise memory **footprint** lattice value: the persistable half of a
//! function's memory effect channel.
//!
//! These types are produced by `qcode_analysis`'s RAM effect channel
//! (`calls::argpromote::ram_summary`) and consumed by anything that needs to
//! know *which addresses* a function may touch rather than merely which spaces.
//! They live in core — not in the analysis crate that derives them — because
//! they are persisted on [`MemoryChannelState`](super::MemoryChannelState),
//! and core cannot depend upward on `qcode_analysis`.
//!
//! Ordering is deliberate: the sets are [`BTreeSet`]s, not `FxHashSet`s, so the
//! serialized form and every iteration over a footprint are deterministic. A
//! footprint is compared across analysis runs (the effect delta, and the
//! sliced-vs-full differential harness), and a hash-ordered set would make two
//! equal footprints render differently.

use std::collections::BTreeSet;

/// What an effect entry's offsets are relative to.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum RamBase {
    /// The pointer passed at this positional argument index of the summary
    /// owner's `pure_reg` interface (`param[i] ↔ Call.args[i]` lockstep).
    Param(u32),
    /// A slot in the summary owner's **own frame**: a constant offset (`< 0`)
    /// from its incoming `@SP`. Minted only by the effect channel's `transfer`
    /// (a callee effect rebased through an own-frame-local argument), always a
    /// *write*, and dropped again when transferred one level further up — the
    /// frame dies at return, so the effect is contained.
    Frame(i64),
    /// An absolute (literal) address in real ram.
    Global(u64),
    /// A **function-private space** landing: a callee effect rebased through a
    /// call argument that is a pointer into the summary owner's own private
    /// (shadow/temp) space. Outward-invisible: a private-space object can
    /// neither be observed nor aliased by any caller, so like a `Frame` write it
    /// is dropped from the outward footprint.
    Private,
}

/// One scalar effect: `size` bytes at `base + offset`, read or written.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct RamField {
    pub base: RamBase,
    pub offset: i64,
    pub size: usize,
    pub write: bool,
}

/// One bounded dynamic-index effect: the half-open byte span
/// `[base + lo, base + hi)`, read (and written iff `write`).
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct RamRegion {
    pub base: RamBase,
    pub lo: i64,
    pub hi: i64,
    pub write: bool,
}

/// One **whole-object** effect: the callee touches the entire (extent-unknown)
/// object addressed by `base`, read (and written iff `write`). Minted **only**
/// from an external prototype's pointer parameters; bodied-function scans never
/// mint object entries (their footprint is exhaustively classified into
/// fields/regions). `write == true` models a read+write (possibly in-out)
/// access — the object is both potentially read and clobbered.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct RamObject {
    pub base: RamBase,
    pub write: bool,
}

/// The precise half of a memory summary: the exhaustively classified footprint.
///
/// The sets are ordered ([`BTreeSet`]) rather than hashed: this value is
/// persisted and compared across runs, so its iteration and wire order must not
/// depend on hash seeding or insertion order.
#[derive(
    Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct Footprint {
    #[serde(default)]
    pub fields: BTreeSet<RamField>,
    #[serde(default)]
    pub regions: BTreeSet<RamRegion>,
    #[serde(default)]
    pub objects: BTreeSet<RamObject>,
}

impl Footprint {
    /// Total number of entries across all three components.
    pub fn len(&self) -> usize {
        self.fields.len() + self.regions.len() + self.objects.len()
    }

    /// Whether the footprint holds no entries at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether nothing in this footprint is observable by any caller: empty, or
    /// `Frame`/`Private`-contained (writes into the owner's own frame, dead at
    /// return, or into a function-private space). This is the argpromote
    /// blocking-call gate's admission predicate.
    pub fn invisible(&self) -> bool {
        self.fields
            .iter()
            .all(|f| matches!(f.base, RamBase::Frame(_) | RamBase::Private))
            && self
                .regions
                .iter()
                .all(|r| matches!(r.base, RamBase::Frame(_) | RamBase::Private))
            && self.objects.iter().all(|o| {
                if matches!(o.base, RamBase::Private) {
                    // A private-space landing is fully self-contained (read or
                    // write): the shadow can neither be observed nor aliased.
                    return true;
                }
                // A `Frame`-based whole-object entry is invisible: a `Frame`
                // write dies with the owner's frame (contained under the
                // confinement assumption). A `Frame`-based object *read* must
                // never exist — the channel's `transfer` sends a frame-landing
                // read to ⊤ (freshness), same as a field — so it should never
                // reach this predicate; treat it defensively as not-invisible if
                // one somehow does.
                debug_assert!(
                    o.write || !matches!(o.base, RamBase::Frame(_)),
                    "a Frame-based object read must have been rejected by transfer"
                );
                matches!(o.base, RamBase::Frame(_)) && o.write
            })
    }
}
