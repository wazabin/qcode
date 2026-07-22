//! Effect *delta* classification: how one function's solved effect summary
//! changed between two rounds (roadmap Milestone 3, § "Effect lattice and
//! deltas").
//!
//! # Why this is not the engine's `join`
//!
//! [`effect_engine`](super::effect_engine) already carries an order: its
//! `join(&mut into, &from) -> bool` reports whether `into` *grew*. That order is
//! the right one *within* a single solve, where the worklist is monotone by
//! construction and effects only ever widen until the fixpoint settles.
//!
//! This module answers a different question, across *successive* solves. When
//! the pipeline reoptimizes a body and re-solves, a summary can genuinely
//! **narrow** — an indirect call gets resolved, a callee stops being ⊤, a dead
//! store is eliminated — and narrowing is invisible to a grow-only predicate.
//! Scheduling wants the direction, not just "something moved":
//!
//! | delta | what a scheduler may conclude |
//! |---|---|
//! | [`Equal`](EffectDelta::Equal) | no caller can observe a difference — **stop propagating** |
//! | [`Narrowed`](EffectDelta::Narrowed) | callers may now optimize *better* — queue them (an opportunity) |
//! | [`Widened`](EffectDelta::Widened) | facts callers derived may be **invalid** — invalidate them (an obligation) |
//! | [`Incomparable`](EffectDelta::Incomparable) | neither contains the other — treat as widened (conservative) |
//!
//! Only `Equal` licenses *stopping*. Everything else must reach callers, which
//! is why `Incomparable` exists as its own verdict rather than being folded into
//! `Widened`: the distinction is worth keeping in logs even though today's
//! conservative action for both is the same.
//!
//! # What this covers
//!
//! [`effects_delta`] compares a whole [`FunctionEffects`] — **both** channels:
//!
//! - the register channel ([`register_channel_delta`]), and
//! - the memory channel ([`memory_channel_delta`]): the coarse written-space
//!   tri-state and the precise RAM [`Footprint`], as two independent components.
//!
//! The two memory components are composed independently on purpose: `coarse` is
//! documented as deliberately laxer than `precise` (see `ram_summary.rs`), so
//! they can legitimately disagree, and [`EffectDelta::composite`] already
//! handles mixed directions correctly.
//!
//! ## Accepted imprecision: no subsumption between entry kinds
//!
//! A [`RamObject`](qcode::value::RamObject) (a whole, extent-unknown object)
//! semantically *subsumes* a [`RamField`](qcode::value::RamField) at the same
//! base, so plain set inclusion reports
//! [`Incomparable`](EffectDelta::Incomparable) where the true relation is
//! containment. That is imprecise but **sound**: `Incomparable` and `Widened`
//! drive the same conservative action, so the only cost is a missed
//! optimization opportunity, never a missed invalidation. Building a
//! subsumption order over entry kinds would be invented precision with no
//! consumer today.
//!
//! ## The polarity of a defaulted channel
//!
//! `MemoryChannelState::default()` is ⊤ in *both* components:
//! [`WrittenSpacesState::Unstamped`] means "no verdict recorded", and
//! `precise: None` means "inexpressible". This is the opposite of the analysis
//! crate's `RamEffects::default()`, where an empty `Footprint` is ⊥ ("a fresh
//! solve has found nothing yet"). Both are right in context; a delta reading a
//! *persisted* channel is reading the former, so a defaulted channel is ⊤.
//!
//! # Nothing consumes this yet
//!
//! This is step 3 of the milestone: the lattice compares both channels, but
//! still with no behavioural change. Adoption is gated on step 4 (the
//! type-level module-pass write handle) and step 5 (the incremental driver);
//! see `docs/plans/milestone-3-effect-deltas/02-incremental-invalidation.md`.

// Steps 1-3 of the milestone land the lattice with no consumer, so every item
// here is legitimately unused until the incremental driver adopts it (see the
// adoption gate in the module docs). The unit tests below exercise all of it;
// this allow exists so `clippy -D warnings` stays green in the interim, and
// should be removed by the commit that wires the delta into scheduling.
#![allow(dead_code)]

use qcode::value::{
    Footprint, FunctionEffects, MemoryChannelState, RegisterChannelState, RegisterEffectSets,
    WrittenSpacesState,
};
use rustc_hash::FxHashSet;
use std::collections::BTreeSet;

/// How a summary changed between two solves, as a position in the effect
/// lattice rather than a mere changed/unchanged bit.
///
/// "Narrower" always means *more precise / less conservative* — a smaller set of
/// permitted effects, which licenses **more** optimization in callers. ⊤ (the
/// most conservative value: "may do anything") is the top of this order, so
/// moving toward ⊤ is [`Widened`](Self::Widened).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EffectDelta {
    /// Semantically identical. The only verdict that licenses stopping
    /// propagation to callers.
    Equal,
    /// Strictly more precise than before: every component is equal-or-narrower
    /// and at least one is strictly narrower.
    Narrowed,
    /// Strictly more conservative than before: the dual of
    /// [`Narrowed`](Self::Narrowed).
    Widened,
    /// Neither summary contains the other — some component narrowed while
    /// another widened, or the two are different kinds of fact entirely.
    /// Consumers must treat this as at least as severe as
    /// [`Widened`](Self::Widened).
    Incomparable,
}

impl EffectDelta {
    /// Whether this delta lets a scheduler stop propagating to callers.
    ///
    /// Deliberately phrased as the *positive* license rather than
    /// `is_changed()`: stopping is the dangerous direction, so it should read as
    /// an explicit permission at the call site.
    pub(crate) fn allows_stopping(self) -> bool {
        matches!(self, EffectDelta::Equal)
    }

    /// Combine independent component deltas into a composite verdict.
    ///
    /// The roadmap's rule: narrowed iff every component is equal-or-narrower and
    /// at least one is strictly narrower; widened is the dual; any mix of
    /// directions — or any [`Incomparable`](Self::Incomparable) component — is
    /// incomparable. An empty iterator is [`Equal`](Self::Equal) (a summary with
    /// no components cannot have changed).
    fn composite(parts: impl IntoIterator<Item = EffectDelta>) -> EffectDelta {
        let mut saw_narrower = false;
        let mut saw_wider = false;
        for part in parts {
            match part {
                EffectDelta::Equal => {}
                EffectDelta::Narrowed => saw_narrower = true,
                EffectDelta::Widened => saw_wider = true,
                EffectDelta::Incomparable => return EffectDelta::Incomparable,
            }
        }
        match (saw_narrower, saw_wider) {
            (false, false) => EffectDelta::Equal,
            (true, false) => EffectDelta::Narrowed,
            (false, true) => EffectDelta::Widened,
            (true, true) => EffectDelta::Incomparable,
        }
    }
}

/// Classify two sets by inclusion.
///
/// A *smaller* set of permitted effects is narrower (more precise), so
/// `new ⊂ old` is [`Narrowed`](EffectDelta::Narrowed). Neither-contains-the-other
/// is [`Incomparable`](EffectDelta::Incomparable).
///
/// Set semantics, not sequence semantics: the persisted
/// [`RegisterEffectSets`] vectors are documented as sorted and deduplicated, but
/// nothing in the type enforces it, and a delta that silently depended on that
/// invariant would misreport rather than fail loudly if a producer ever emitted
/// an unsorted vector. Hashing into sets costs nothing at scheduling frequency.
fn delta_of_sets<T: std::hash::Hash + Eq>(old: &[T], new: &[T]) -> EffectDelta {
    let old: FxHashSet<&T> = old.iter().collect();
    let new: FxHashSet<&T> = new.iter().collect();
    match (new.is_subset(&old), new.is_superset(&old)) {
        (true, true) => EffectDelta::Equal,
        (true, false) => EffectDelta::Narrowed,
        (false, true) => EffectDelta::Widened,
        (false, false) => EffectDelta::Incomparable,
    }
}

/// Classify two solved register effect sets componentwise (`loads`, `stores`).
fn delta_of_register_sets(old: &RegisterEffectSets, new: &RegisterEffectSets) -> EffectDelta {
    EffectDelta::composite([
        delta_of_sets(&old.loads, &new.loads),
        delta_of_sets(&old.stores, &new.stores),
    ])
}

/// Whether this summary is semantically ⊤ — "may do anything", the most
/// conservative value.
///
/// [`Unsolved`](RegisterChannelState::Unsolved) and [`Top`](RegisterChannelState::Top) are
/// distinct *representations* that every consumer already collapses into the
/// same conservative arm (`gvn/mem_forward.rs` maps both to
/// `CallClobbers::AllRegisters`; `mem/mem2reg.rs` gates them together). This
/// lattice is ordered by **meaning**, so the two are one point: an
/// `Unsolved → Top` transition classifies as [`Equal`](EffectDelta::Equal) and
/// correctly propagates nothing, rather than spending a caller invalidation on a
/// transition no consumer can observe.
fn is_top(effects: &RegisterChannelState) -> bool {
    matches!(
        effects,
        RegisterChannelState::Unsolved | RegisterChannelState::Top
    )
}

/// Classify how a function's register-channel effect summary changed.
///
/// See the module docs for the ordering rationale and the adoption gate. The
/// interesting cases:
///
/// - ⊤ (`Unsolved`/`Top`) is the top of the order: leaving it is a narrowing,
///   falling back to it is a widening.
/// - Two [`Solved`](RegisterChannelState::Solved) summaries compare componentwise by
///   set inclusion on `loads` and `stores`.
/// - [`Materialized`](RegisterChannelState::Materialized) is **not** ordered against
///   `Solved`. Materialization does not shrink an effect set; it changes the
///   *binding convention* callers use (by-value params and a return pack). It is
///   a different kind of fact, so it is `Incomparable` — which forces callers to
///   be revisited, the outcome a convention change requires.
/// - Two `Materialized` summaries are `Equal` when their interface maps match
///   and `Incomparable` otherwise. This is a deliberate coarsening: an ordering
///   over binding maps would be invented precision, and since `Widened` and
///   `Incomparable` drive the same conservative action there is nothing to gain
///   from it today.
pub(crate) fn register_channel_delta(
    old: &RegisterChannelState,
    new: &RegisterChannelState,
) -> EffectDelta {
    match (is_top(old), is_top(new)) {
        (true, true) => return EffectDelta::Equal,
        // Leaving ⊤ for any known summary is a narrowing; falling back to ⊤ from
        // one is a widening.
        (true, false) => return EffectDelta::Narrowed,
        (false, true) => return EffectDelta::Widened,
        (false, false) => {}
    }

    match (old, new) {
        (RegisterChannelState::Solved(old), RegisterChannelState::Solved(new)) => {
            delta_of_register_sets(old, new)
        }
        (RegisterChannelState::Materialized(old), RegisterChannelState::Materialized(new)) => {
            if old == new {
                EffectDelta::Equal
            } else {
                EffectDelta::Incomparable
            }
        }
        // Solved vs Materialized, either direction: different kinds of fact.
        _ => EffectDelta::Incomparable,
    }
}

/// Classify two ordered sets by inclusion, with the same polarity as
/// [`delta_of_sets`]: a *smaller* set of permitted effects is narrower.
///
/// A separate entry point from [`delta_of_sets`] only because the persisted
/// [`Footprint`] components are already [`BTreeSet`]s — there is nothing to
/// re-hash, and their ordering is a deliberate determinism guarantee.
fn delta_of_btree_sets<T: Ord>(old: &BTreeSet<T>, new: &BTreeSet<T>) -> EffectDelta {
    match (new.is_subset(old), new.is_superset(old)) {
        (true, true) => EffectDelta::Equal,
        (true, false) => EffectDelta::Narrowed,
        (false, true) => EffectDelta::Widened,
        (false, false) => EffectDelta::Incomparable,
    }
}

/// Classify two precise footprints componentwise (`fields`, `regions`,
/// `objects`), each by plain set inclusion.
///
/// All three components are compared identically even though, as measured on
/// real binaries, `objects` is the only one carrying production signal today
/// (argpromote erases bodied functions' outward memory effects, so every
/// non-empty footprint observed came from an external prototype's argmem). The
/// ordering must not encode that skew — a bodied footprint is not a different
/// kind of fact, merely a rarer one.
///
/// See the module docs for the accepted subsumption imprecision between entry
/// kinds.
fn delta_of_footprints(old: &Footprint, new: &Footprint) -> EffectDelta {
    EffectDelta::composite([
        delta_of_btree_sets(&old.fields, &new.fields),
        delta_of_btree_sets(&old.regions, &new.regions),
        delta_of_btree_sets(&old.objects, &new.objects),
    ])
}

/// Classify the coarse written-space tri-state.
///
/// [`Unstamped`](WrittenSpacesState::Unstamped) and
/// [`Unbounded`](WrittenSpacesState::Unbounded) are **one lattice point** (⊤),
/// for exactly the reason `Unsolved`/`Top` are on the register channel: they are
/// distinct *representations* — "no verdict yet" and "verdict: anything" — that
/// every consumer collapses into the same conservative arm. Ordering by meaning
/// means an `Unstamped → Unbounded` transition is [`Equal`](EffectDelta::Equal)
/// and correctly spends no caller invalidation.
///
/// [`Bounded`](WrittenSpacesState::Bounded) sets compare by inclusion: a smaller
/// witnessed set of written spaces is narrower.
fn coarse_delta(old: &WrittenSpacesState, new: &WrittenSpacesState) -> EffectDelta {
    let is_top = |s: &WrittenSpacesState| {
        matches!(
            s,
            WrittenSpacesState::Unstamped | WrittenSpacesState::Unbounded
        )
    };
    match (old, new) {
        (o, n) if is_top(o) && is_top(n) => EffectDelta::Equal,
        // ⊤ is the most conservative value: leaving it narrows, returning widens.
        (o, _) if is_top(o) => EffectDelta::Narrowed,
        (_, n) if is_top(n) => EffectDelta::Widened,
        (WrittenSpacesState::Bounded(old), WrittenSpacesState::Bounded(new)) => {
            delta_of_sets(old, new)
        }
        // Unreachable: every non-`Bounded` state is ⊤ and handled above.
        _ => EffectDelta::Incomparable,
    }
}

/// Classify the precise footprint component, where `None` is ⊤.
///
/// The polarity that matters (see the module docs): a persisted `None` means
/// *inexpressible*, not *empty*. `Some(empty)` is ⊥ — a function proven to touch
/// nothing — so `None → Some(_)` narrows and `Some(_) → None` widens.
fn precise_delta(old: &Option<Footprint>, new: &Option<Footprint>) -> EffectDelta {
    match (old, new) {
        (None, None) => EffectDelta::Equal,
        (None, Some(_)) => EffectDelta::Narrowed,
        (Some(_), None) => EffectDelta::Widened,
        (Some(old), Some(new)) => delta_of_footprints(old, new),
    }
}

/// Classify how a function's memory-channel effect summary changed: the coarse
/// written-space verdict and the precise footprint, as two independent
/// components.
///
/// They are independent by design — `coarse` is deliberately laxer than
/// `precise`, so a function may have a bounded space set alongside a ⊤
/// footprint, and the two may move in opposite directions between solves.
/// [`EffectDelta::composite`] reports that mix as
/// [`Incomparable`](EffectDelta::Incomparable), which is the correct
/// conservative verdict.
pub(crate) fn memory_channel_delta(
    old: &MemoryChannelState,
    new: &MemoryChannelState,
) -> EffectDelta {
    EffectDelta::composite([
        coarse_delta(&old.coarse, &new.coarse),
        precise_delta(&old.precise, &new.precise),
    ])
}

/// Classify how a function's **whole** effect summary changed, across every
/// channel.
///
/// This is the entry point a scheduler must use: a delta over a single channel
/// can report [`Equal`](EffectDelta::Equal) while the other channel changed, and
/// `Equal` is the one verdict that licenses stopping propagation to callers —
/// so a single-channel comparison under-invalidates, the unsound direction.
pub(crate) fn effects_delta(old: &FunctionEffects, new: &FunctionEffects) -> EffectDelta {
    EffectDelta::composite([
        register_channel_delta(&old.register, &new.register),
        memory_channel_delta(&old.memory, &new.memory),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::space::SpaceId;
    use qcode::value::{RamBase, RamField, RamObject, RamRegion, RegisterInterfaceMap, VarnodeId};

    /// `VarnodeId` is an opaque typed id; build distinct ones by index.
    fn vn(i: usize) -> VarnodeId {
        VarnodeId::from(i)
    }

    fn solved(loads: &[usize], stores: &[usize]) -> RegisterChannelState {
        RegisterChannelState::Solved(RegisterEffectSets {
            loads: loads.iter().copied().map(vn).collect(),
            stores: stores.iter().copied().map(vn).collect(),
        })
    }

    fn bounded(spaces: &[usize]) -> WrittenSpacesState {
        WrittenSpacesState::Bounded(spaces.iter().copied().map(SpaceId::from).collect())
    }

    /// A footprint holding only whole-object entries — the shape that carries
    /// real production signal (external prototypes' argmem).
    fn fp_objects(bases: &[u32]) -> Footprint {
        Footprint {
            objects: bases.iter().copied().map(object).collect(),
            ..Default::default()
        }
    }

    fn object(base: u32) -> RamObject {
        RamObject {
            base: RamBase::Param(base),
            write: true,
        }
    }

    fn field(base: u32, offset: i64) -> RamField {
        RamField {
            base: RamBase::Param(base),
            offset,
            size: 8,
            write: true,
        }
    }

    fn region(base: u32) -> RamRegion {
        RamRegion {
            base: RamBase::Param(base),
            lo: 0,
            hi: 32,
            write: true,
        }
    }

    fn mem(coarse: WrittenSpacesState, precise: Option<Footprint>) -> MemoryChannelState {
        MemoryChannelState { coarse, precise }
    }

    fn effects(register: RegisterChannelState, memory: MemoryChannelState) -> FunctionEffects {
        FunctionEffects { register, memory }
    }

    /// A cross product spanning every constructor of both channels, used by the
    /// reflexivity and antisymmetry properties.
    fn all_effects() -> Vec<FunctionEffects> {
        let registers = [
            RegisterChannelState::Unsolved,
            RegisterChannelState::Top,
            solved(&[], &[]),
            solved(&[1], &[]),
            solved(&[1, 2], &[3]),
            solved(&[2, 5], &[3]),
            materialized(&[1]),
        ];
        let memories = [
            mem(WrittenSpacesState::Unstamped, None),
            mem(WrittenSpacesState::Unbounded, None),
            mem(bounded(&[]), None),
            mem(bounded(&[1]), Some(Footprint::default())),
            mem(bounded(&[1, 2]), Some(fp_objects(&[0]))),
            mem(bounded(&[2, 3]), Some(fp_objects(&[0, 1]))),
            mem(
                bounded(&[1]),
                Some(Footprint {
                    fields: [field(0, 0)].into_iter().collect(),
                    regions: [region(0)].into_iter().collect(),
                    objects: [object(0)].into_iter().collect(),
                }),
            ),
        ];
        registers
            .iter()
            .flat_map(|r| memories.iter().map(|m| effects(r.clone(), m.clone())))
            .collect()
    }

    fn materialized(inputs: &[usize]) -> RegisterChannelState {
        RegisterChannelState::Materialized(RegisterInterfaceMap {
            inputs: inputs.iter().copied().map(vn).collect(),
            ..Default::default()
        })
    }

    // ---- the ⊤ points -----------------------------------------------------

    /// `Unsolved` and `Top` are one lattice point, so moving between them is
    /// `Equal` in both directions and propagates nothing.
    #[test]
    fn unsolved_and_top_are_the_same_point() {
        let cases = [
            (
                RegisterChannelState::Unsolved,
                RegisterChannelState::Unsolved,
            ),
            (RegisterChannelState::Top, RegisterChannelState::Top),
            (RegisterChannelState::Unsolved, RegisterChannelState::Top),
            (RegisterChannelState::Top, RegisterChannelState::Unsolved),
        ];
        for (old, new) in cases {
            assert_eq!(
                register_channel_delta(&old, &new),
                EffectDelta::Equal,
                "{old:?} -> {new:?}"
            );
        }
    }

    /// ⊤ is the *most conservative* value, so leaving it narrows and returning
    /// to it widens. Getting this backwards is the unsound direction: it would
    /// let a function fall back to "may clobber anything" while callers kept
    /// facts derived from the precise summary.
    #[test]
    fn top_is_the_top_of_the_order() {
        for top in [RegisterChannelState::Unsolved, RegisterChannelState::Top] {
            for known in [solved(&[1], &[2]), materialized(&[1])] {
                assert_eq!(register_channel_delta(&top, &known), EffectDelta::Narrowed);
                assert_eq!(register_channel_delta(&known, &top), EffectDelta::Widened);
            }
        }
    }

    // ---- solved vs solved -------------------------------------------------

    #[test]
    fn identical_solved_summaries_are_equal() {
        assert_eq!(
            register_channel_delta(&solved(&[1, 2], &[3]), &solved(&[1, 2], &[3])),
            EffectDelta::Equal
        );
    }

    /// Order and duplication are not part of the value: these are sets.
    #[test]
    fn solved_comparison_is_set_not_sequence() {
        assert_eq!(
            register_channel_delta(&solved(&[1, 2], &[3]), &solved(&[2, 1, 1], &[3])),
            EffectDelta::Equal
        );
    }

    #[test]
    fn shrinking_a_component_narrows() {
        assert_eq!(
            register_channel_delta(&solved(&[1, 2], &[3]), &solved(&[1], &[3])),
            EffectDelta::Narrowed
        );
    }

    #[test]
    fn growing_a_component_widens() {
        assert_eq!(
            register_channel_delta(&solved(&[1], &[3]), &solved(&[1, 2], &[3])),
            EffectDelta::Widened
        );
    }

    /// Both components moving the same way is still that way.
    #[test]
    fn both_components_narrowing_narrows() {
        assert_eq!(
            register_channel_delta(&solved(&[1, 2], &[3, 4]), &solved(&[1], &[3])),
            EffectDelta::Narrowed
        );
    }

    /// The composite rule's key case: mixed directions are incomparable, not
    /// "changed in the dominant direction".
    #[test]
    fn one_narrowing_and_one_widening_is_incomparable() {
        assert_eq!(
            register_channel_delta(&solved(&[1, 2], &[3]), &solved(&[1], &[3, 4])),
            EffectDelta::Incomparable
        );
    }

    /// A single component that overlaps without containment is incomparable on
    /// its own.
    #[test]
    fn disjoint_component_is_incomparable() {
        assert_eq!(
            register_channel_delta(&solved(&[1, 2], &[3]), &solved(&[2, 5], &[3])),
            EffectDelta::Incomparable
        );
    }

    #[test]
    fn empty_summaries_compare_equal_and_order_against_nonempty() {
        assert_eq!(
            register_channel_delta(&solved(&[], &[]), &solved(&[], &[])),
            EffectDelta::Equal
        );
        assert_eq!(
            register_channel_delta(&solved(&[1], &[]), &solved(&[], &[])),
            EffectDelta::Narrowed
        );
        assert_eq!(
            register_channel_delta(&solved(&[], &[]), &solved(&[1], &[])),
            EffectDelta::Widened
        );
    }

    // ---- materialization --------------------------------------------------

    #[test]
    fn identical_materialized_maps_are_equal() {
        assert_eq!(
            register_channel_delta(&materialized(&[1, 2]), &materialized(&[1, 2])),
            EffectDelta::Equal
        );
    }

    /// A changed binding map is a convention change: callers must be revisited,
    /// so it must never classify as `Equal`.
    #[test]
    fn differing_materialized_maps_never_stop_propagation() {
        let delta = register_channel_delta(&materialized(&[1]), &materialized(&[1, 2]));
        assert_eq!(delta, EffectDelta::Incomparable);
        assert!(!delta.allows_stopping());
    }

    /// Materialization changes how callers bind, not how much the callee may do,
    /// so it is not ordered against `Solved` in either direction.
    #[test]
    fn materialized_is_not_ordered_against_solved() {
        assert_eq!(
            register_channel_delta(&solved(&[1], &[2]), &materialized(&[1])),
            EffectDelta::Incomparable
        );
        assert_eq!(
            register_channel_delta(&materialized(&[1]), &solved(&[1], &[2])),
            EffectDelta::Incomparable
        );
    }

    // ---- the stopping license ---------------------------------------------

    /// The safety-critical property: `Equal` is the *only* verdict that lets a
    /// scheduler stop.
    #[test]
    fn only_equal_allows_stopping() {
        assert!(EffectDelta::Equal.allows_stopping());
        for delta in [
            EffectDelta::Narrowed,
            EffectDelta::Widened,
            EffectDelta::Incomparable,
        ] {
            assert!(!delta.allows_stopping(), "{delta:?}");
        }
    }

    // ---- the composite rule, exhaustively ---------------------------------

    /// Every ordered pair from the four-point lattice, checked against the
    /// roadmap's rule directly.
    #[test]
    fn composite_rule_is_exhaustive_over_pairs() {
        use EffectDelta::*;
        let all = [Equal, Narrowed, Widened, Incomparable];
        for a in all {
            for b in all {
                let expected = match (a, b) {
                    (Incomparable, _) | (_, Incomparable) => Incomparable,
                    (Narrowed, Widened) | (Widened, Narrowed) => Incomparable,
                    (Narrowed, _) | (_, Narrowed) => Narrowed,
                    (Widened, _) | (_, Widened) => Widened,
                    (Equal, Equal) => Equal,
                };
                assert_eq!(EffectDelta::composite([a, b]), expected, "{a:?} + {b:?}");
            }
        }
    }

    /// A summary with no components cannot have changed.
    #[test]
    fn composite_of_nothing_is_equal() {
        assert_eq!(EffectDelta::composite([]), EffectDelta::Equal);
    }

    /// `composite` must be order-insensitive — components are independent.
    #[test]
    fn composite_is_commutative() {
        use EffectDelta::*;
        let all = [Equal, Narrowed, Widened, Incomparable];
        for a in all {
            for b in all {
                assert_eq!(
                    EffectDelta::composite([a, b]),
                    EffectDelta::composite([b, a]),
                    "{a:?} + {b:?}"
                );
            }
        }
    }

    /// Reflexivity across every constructor: a summary compared with itself is
    /// always `Equal`, including at both ⊤ representations.
    #[test]
    fn every_summary_is_equal_to_itself() {
        let all = [
            RegisterChannelState::Unsolved,
            RegisterChannelState::Top,
            solved(&[], &[]),
            solved(&[1, 2], &[3]),
            materialized(&[1, 2]),
        ];
        for e in &all {
            assert_eq!(register_channel_delta(e, e), EffectDelta::Equal, "{e:?}");
        }
    }

    // ---- the memory channel: coarse written spaces -------------------------

    /// `Unstamped` and `Unbounded` are one lattice point (⊤), so moving between
    /// them is `Equal` in both directions and propagates nothing.
    #[test]
    fn unstamped_and_unbounded_are_the_same_point() {
        let tops = [WrittenSpacesState::Unstamped, WrittenSpacesState::Unbounded];
        for old in &tops {
            for new in &tops {
                assert_eq!(
                    coarse_delta(old, new),
                    EffectDelta::Equal,
                    "{old:?}->{new:?}"
                );
            }
        }
    }

    /// The coarse ⊤ is the *most conservative* value: leaving it narrows,
    /// falling back to it widens. The unsound inversion would let a function
    /// return to "may write any space" while callers kept facts derived from a
    /// bounded set.
    #[test]
    fn coarse_top_is_the_top_of_the_order() {
        for top in [WrittenSpacesState::Unstamped, WrittenSpacesState::Unbounded] {
            for bounded in [bounded(&[]), bounded(&[1]), bounded(&[1, 2])] {
                assert_eq!(coarse_delta(&top, &bounded), EffectDelta::Narrowed);
                assert_eq!(coarse_delta(&bounded, &top), EffectDelta::Widened);
            }
        }
    }

    #[test]
    fn bounded_space_sets_compare_by_inclusion() {
        assert_eq!(
            coarse_delta(&bounded(&[1, 2]), &bounded(&[2, 1])),
            EffectDelta::Equal
        );
        assert_eq!(
            coarse_delta(&bounded(&[1, 2]), &bounded(&[1])),
            EffectDelta::Narrowed
        );
        assert_eq!(
            coarse_delta(&bounded(&[1]), &bounded(&[1, 2])),
            EffectDelta::Widened
        );
        assert_eq!(
            coarse_delta(&bounded(&[1, 2]), &bounded(&[2, 3])),
            EffectDelta::Incomparable
        );
    }

    // ---- the memory channel: the precise footprint -------------------------

    /// `None` is ⊤ — *inexpressible*, not *empty*. This is the polarity trap:
    /// `Some(Footprint::default())` is ⊥ (proven to touch nothing), the exact
    /// opposite meaning, so `None → Some(_)` must narrow.
    #[test]
    fn precise_none_is_top_not_bottom() {
        let empty = Some(Footprint::default());
        assert_eq!(precise_delta(&None, &None), EffectDelta::Equal);
        assert_eq!(precise_delta(&None, &empty), EffectDelta::Narrowed);
        assert_eq!(precise_delta(&empty, &None), EffectDelta::Widened);
        // ...and ⊥ still orders below a non-empty footprint.
        let some = Some(fp_objects(&[0]));
        assert_eq!(precise_delta(&some, &empty), EffectDelta::Narrowed);
        assert_eq!(precise_delta(&empty, &some), EffectDelta::Widened);
    }

    /// `objects` is the component that carries real production signal (every
    /// non-empty footprint measured on real binaries came from an external
    /// prototype's argmem), so it gets the inclusion test in its own right.
    #[test]
    fn object_sets_compare_by_inclusion() {
        assert_eq!(
            precise_delta(&Some(fp_objects(&[0, 1])), &Some(fp_objects(&[1, 0]))),
            EffectDelta::Equal
        );
        assert_eq!(
            precise_delta(&Some(fp_objects(&[0, 1])), &Some(fp_objects(&[0]))),
            EffectDelta::Narrowed
        );
        assert_eq!(
            precise_delta(&Some(fp_objects(&[0])), &Some(fp_objects(&[0, 1]))),
            EffectDelta::Widened
        );
        assert_eq!(
            precise_delta(&Some(fp_objects(&[0, 1])), &Some(fp_objects(&[1, 2]))),
            EffectDelta::Incomparable
        );
    }

    /// All three footprint components are compared identically, and mixed
    /// directions across them are incomparable.
    #[test]
    fn footprint_components_compose_independently() {
        let old = Footprint {
            fields: [field(0, 0), field(0, 8)].into_iter().collect(),
            regions: Default::default(),
            objects: [object(1)].into_iter().collect(),
        };
        // fields narrow, objects unchanged.
        let narrower = Footprint {
            fields: [field(0, 0)].into_iter().collect(),
            ..old.clone()
        };
        assert_eq!(delta_of_footprints(&old, &narrower), EffectDelta::Narrowed);
        // fields narrow while objects grow: incomparable.
        let mixed = Footprint {
            fields: [field(0, 0)].into_iter().collect(),
            regions: Default::default(),
            objects: [object(1), object(2)].into_iter().collect(),
        };
        assert_eq!(delta_of_footprints(&old, &mixed), EffectDelta::Incomparable);
        // a regions-only change is visible.
        let with_region = Footprint {
            regions: [region(0)].into_iter().collect(),
            ..old.clone()
        };
        assert_eq!(
            delta_of_footprints(&old, &with_region),
            EffectDelta::Widened
        );
    }

    /// A whole-object entry semantically subsumes a field at the same base, but
    /// this lattice deliberately does *not* reason about that: plain set
    /// inclusion reports `Incomparable`. Documented as accepted imprecision —
    /// `Incomparable` and `Widened` drive the same conservative action, so the
    /// only cost is a missed opportunity, never a missed invalidation.
    #[test]
    fn entry_kind_subsumption_is_not_modelled() {
        let object_only = Footprint {
            objects: [object(0)].into_iter().collect(),
            ..Default::default()
        };
        let field_only = Footprint {
            fields: [field(0, 0)].into_iter().collect(),
            ..Default::default()
        };
        let delta = delta_of_footprints(&object_only, &field_only);
        assert_eq!(delta, EffectDelta::Incomparable);
        // The property that makes the imprecision safe:
        assert!(!delta.allows_stopping());
    }

    // ---- the memory channel, composed --------------------------------------

    /// A defaulted memory channel is ⊤ in *both* components, so anything known
    /// narrows against it.
    #[test]
    fn defaulted_memory_channel_is_top() {
        let top = MemoryChannelState::default();
        let known = mem(bounded(&[1]), Some(Footprint::default()));
        assert_eq!(memory_channel_delta(&top, &top), EffectDelta::Equal);
        assert_eq!(memory_channel_delta(&top, &known), EffectDelta::Narrowed);
        assert_eq!(memory_channel_delta(&known, &top), EffectDelta::Widened);
    }

    /// The two memory components are independent: `coarse` is deliberately
    /// laxer than `precise`, so they may legitimately disagree, and a
    /// disagreement is `Incomparable`.
    #[test]
    fn coarse_and_precise_disagreeing_is_incomparable() {
        let old = mem(bounded(&[1, 2]), Some(fp_objects(&[0])));
        let new = mem(bounded(&[1]), Some(fp_objects(&[0, 1])));
        assert_eq!(memory_channel_delta(&old, &new), EffectDelta::Incomparable);
    }

    // ---- the composed whole-effects delta ----------------------------------

    /// **Required non-vacuity property.** A change confined to the memory
    /// channel must never classify as `Equal` — `Equal` is the one verdict that
    /// licenses stopping propagation, so a composite that masked a memory change
    /// would under-invalidate, which is exactly what ruling 10 exists to
    /// prevent. Checked for a `coarse`-only change and a `precise`-only change
    /// separately, since either alone must be sufficient.
    #[test]
    fn a_memory_only_change_is_never_equal() {
        let register = solved(&[1, 2], &[3]);

        // Vary `coarse` alone, holding `precise` fixed.
        let coarse_states = [
            WrittenSpacesState::Unstamped,
            WrittenSpacesState::Unbounded,
            bounded(&[]),
            bounded(&[1]),
            bounded(&[1, 2]),
            bounded(&[2, 3]),
        ];
        let is_top = |s: &WrittenSpacesState| {
            matches!(
                s,
                WrittenSpacesState::Unstamped | WrittenSpacesState::Unbounded
            )
        };
        for old_coarse in &coarse_states {
            for new_coarse in &coarse_states {
                // `Unstamped` vs `Unbounded` is the ⊤-collapse: one lattice
                // point, legitimately `Equal`. Every other differing pair must
                // move.
                if old_coarse == new_coarse || (is_top(old_coarse) && is_top(new_coarse)) {
                    continue;
                }
                let old = effects(register.clone(), mem(old_coarse.clone(), None));
                let new = effects(register.clone(), mem(new_coarse.clone(), None));
                let delta = effects_delta(&old, &new);
                assert_ne!(
                    delta,
                    EffectDelta::Equal,
                    "{old_coarse:?} -> {new_coarse:?}"
                );
                assert!(!delta.allows_stopping());
            }
        }

        // Vary `precise` alone, holding `coarse` fixed.
        let precise_states = [
            None,
            Some(Footprint::default()),
            Some(fp_objects(&[0])),
            Some(fp_objects(&[0, 1])),
            Some(Footprint {
                fields: [field(0, 0)].into_iter().collect(),
                ..Default::default()
            }),
            Some(Footprint {
                regions: [region(0)].into_iter().collect(),
                ..Default::default()
            }),
        ];
        for old_precise in &precise_states {
            for new_precise in &precise_states {
                if old_precise == new_precise {
                    continue;
                }
                let old = effects(register.clone(), mem(bounded(&[1]), old_precise.clone()));
                let new = effects(register.clone(), mem(bounded(&[1]), new_precise.clone()));
                let delta = effects_delta(&old, &new);
                assert_ne!(
                    delta,
                    EffectDelta::Equal,
                    "{old_precise:?} -> {new_precise:?}"
                );
                assert!(!delta.allows_stopping());
            }
        }
    }

    /// The dual sanity check: a register-only change is still visible through
    /// the composite, so composing the memory channel in did not swallow it.
    #[test]
    fn a_register_only_change_is_never_equal() {
        let memory = mem(bounded(&[1]), Some(fp_objects(&[0])));
        let old = effects(solved(&[1, 2], &[3]), memory.clone());
        let new = effects(solved(&[1], &[3]), memory);
        assert_eq!(effects_delta(&old, &new), EffectDelta::Narrowed);
    }

    /// A whole-effects summary compared with itself is `Equal`, across every
    /// combination of channel states.
    #[test]
    fn every_effects_summary_is_equal_to_itself() {
        for e in &all_effects() {
            assert_eq!(effects_delta(e, e), EffectDelta::Equal, "{e:?}");
        }
    }

    /// **Required antisymmetry property, extended to both channels.** Swapping
    /// the arguments must swap narrowed/widened and fix equal/incomparable —
    /// across the register channel, the coarse verdict and the precise
    /// footprint simultaneously. This is what catches a polarity inversion
    /// anywhere in the new comparison chain.
    #[test]
    fn swapping_arguments_inverts_direction_across_both_channels() {
        let all = all_effects();
        for old in &all {
            for new in &all {
                let forward = effects_delta(old, new);
                let backward = effects_delta(new, old);
                let expected = match forward {
                    EffectDelta::Equal => EffectDelta::Equal,
                    EffectDelta::Narrowed => EffectDelta::Widened,
                    EffectDelta::Widened => EffectDelta::Narrowed,
                    EffectDelta::Incomparable => EffectDelta::Incomparable,
                };
                assert_eq!(backward, expected, "{old:?} -> {new:?}");
            }
        }
    }

    /// Antisymmetry of direction: swapping the arguments swaps
    /// narrowed/widened and fixes equal/incomparable. This is the property that
    /// catches a polarity inversion anywhere in the comparison chain.
    #[test]
    fn swapping_arguments_inverts_direction() {
        let all = [
            RegisterChannelState::Unsolved,
            RegisterChannelState::Top,
            solved(&[], &[]),
            solved(&[1], &[]),
            solved(&[1, 2], &[3]),
            solved(&[2, 5], &[3]),
            materialized(&[1]),
            materialized(&[1, 2]),
        ];
        for old in &all {
            for new in &all {
                let forward = register_channel_delta(old, new);
                let backward = register_channel_delta(new, old);
                let expected = match forward {
                    EffectDelta::Equal => EffectDelta::Equal,
                    EffectDelta::Narrowed => EffectDelta::Widened,
                    EffectDelta::Widened => EffectDelta::Narrowed,
                    EffectDelta::Incomparable => EffectDelta::Incomparable,
                };
                assert_eq!(backward, expected, "{old:?} -> {new:?}");
            }
        }
    }
}
