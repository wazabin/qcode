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
//! # Nothing consumes this yet
//!
//! This is step 1 of the milestone: the lattice and its tests, with no
//! behavioural change. **It must not be wired into scheduling as-is.** See
//! [`FunctionEffects`] below — this compares the *register* channel only,
//! because that is the only channel [`FunctionEffects`] persists on this branch.
//! A scheduler that stopped propagation on `Equal` today would miss a change in
//! the memory channel (`written_spaces`, still stored on `FunctionSignature`)
//! and under-invalidate. The gate for adoption is the effects-unification work
//! that folds the memory component into [`FunctionEffects`].

// Step 1 of the milestone lands the lattice with no consumer, so every item
// here is legitimately unused until the scheduler adopts it (see the adoption
// gate in the module docs). The unit tests below exercise all of it; this allow
// exists so `clippy -D warnings` stays green in the interim, and should be
// removed by the commit that wires the delta into scheduling.
#![allow(dead_code)]

use qcode::value::{FunctionEffects, RegisterEffectSets};
use rustc_hash::FxHashSet;

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
/// [`Unsolved`](FunctionEffects::Unsolved) and [`Top`](FunctionEffects::Top) are
/// distinct *representations* that every consumer already collapses into the
/// same conservative arm (`gvn/mem_forward.rs` maps both to
/// `CallClobbers::AllRegisters`; `mem/mem2reg.rs` gates them together). This
/// lattice is ordered by **meaning**, so the two are one point: an
/// `Unsolved → Top` transition classifies as [`Equal`](EffectDelta::Equal) and
/// correctly propagates nothing, rather than spending a caller invalidation on a
/// transition no consumer can observe.
fn is_top(effects: &FunctionEffects) -> bool {
    matches!(effects, FunctionEffects::Unsolved | FunctionEffects::Top)
}

/// Classify how a function's register-channel effect summary changed.
///
/// See the module docs for the ordering rationale and the adoption gate. The
/// interesting cases:
///
/// - ⊤ (`Unsolved`/`Top`) is the top of the order: leaving it is a narrowing,
///   falling back to it is a widening.
/// - Two [`Solved`](FunctionEffects::Solved) summaries compare componentwise by
///   set inclusion on `loads` and `stores`.
/// - [`Materialized`](FunctionEffects::Materialized) is **not** ordered against
///   `Solved`. Materialization does not shrink an effect set; it changes the
///   *binding convention* callers use (by-value params and a return pack). It is
///   a different kind of fact, so it is `Incomparable` — which forces callers to
///   be revisited, the outcome a convention change requires.
/// - Two `Materialized` summaries are `Equal` when their interface maps match
///   and `Incomparable` otherwise. This is a deliberate coarsening: an ordering
///   over binding maps would be invented precision, and since `Widened` and
///   `Incomparable` drive the same conservative action there is nothing to gain
///   from it today.
pub(crate) fn effect_delta(old: &FunctionEffects, new: &FunctionEffects) -> EffectDelta {
    match (is_top(old), is_top(new)) {
        (true, true) => return EffectDelta::Equal,
        // Leaving ⊤ for any known summary is a narrowing; falling back to ⊤ from
        // one is a widening.
        (true, false) => return EffectDelta::Narrowed,
        (false, true) => return EffectDelta::Widened,
        (false, false) => {}
    }

    match (old, new) {
        (FunctionEffects::Solved(old), FunctionEffects::Solved(new)) => {
            delta_of_register_sets(old, new)
        }
        (FunctionEffects::Materialized(old), FunctionEffects::Materialized(new)) => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::{RegisterInterfaceMap, VarnodeId};

    /// `VarnodeId` is an opaque typed id; build distinct ones by index.
    fn vn(i: usize) -> VarnodeId {
        VarnodeId::from(i)
    }

    fn solved(loads: &[usize], stores: &[usize]) -> FunctionEffects {
        FunctionEffects::Solved(RegisterEffectSets {
            loads: loads.iter().copied().map(vn).collect(),
            stores: stores.iter().copied().map(vn).collect(),
        })
    }

    fn materialized(inputs: &[usize]) -> FunctionEffects {
        FunctionEffects::Materialized(RegisterInterfaceMap {
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
            (FunctionEffects::Unsolved, FunctionEffects::Unsolved),
            (FunctionEffects::Top, FunctionEffects::Top),
            (FunctionEffects::Unsolved, FunctionEffects::Top),
            (FunctionEffects::Top, FunctionEffects::Unsolved),
        ];
        for (old, new) in cases {
            assert_eq!(
                effect_delta(&old, &new),
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
        for top in [FunctionEffects::Unsolved, FunctionEffects::Top] {
            for known in [solved(&[1], &[2]), materialized(&[1])] {
                assert_eq!(effect_delta(&top, &known), EffectDelta::Narrowed);
                assert_eq!(effect_delta(&known, &top), EffectDelta::Widened);
            }
        }
    }

    // ---- solved vs solved -------------------------------------------------

    #[test]
    fn identical_solved_summaries_are_equal() {
        assert_eq!(
            effect_delta(&solved(&[1, 2], &[3]), &solved(&[1, 2], &[3])),
            EffectDelta::Equal
        );
    }

    /// Order and duplication are not part of the value: these are sets.
    #[test]
    fn solved_comparison_is_set_not_sequence() {
        assert_eq!(
            effect_delta(&solved(&[1, 2], &[3]), &solved(&[2, 1, 1], &[3])),
            EffectDelta::Equal
        );
    }

    #[test]
    fn shrinking_a_component_narrows() {
        assert_eq!(
            effect_delta(&solved(&[1, 2], &[3]), &solved(&[1], &[3])),
            EffectDelta::Narrowed
        );
    }

    #[test]
    fn growing_a_component_widens() {
        assert_eq!(
            effect_delta(&solved(&[1], &[3]), &solved(&[1, 2], &[3])),
            EffectDelta::Widened
        );
    }

    /// Both components moving the same way is still that way.
    #[test]
    fn both_components_narrowing_narrows() {
        assert_eq!(
            effect_delta(&solved(&[1, 2], &[3, 4]), &solved(&[1], &[3])),
            EffectDelta::Narrowed
        );
    }

    /// The composite rule's key case: mixed directions are incomparable, not
    /// "changed in the dominant direction".
    #[test]
    fn one_narrowing_and_one_widening_is_incomparable() {
        assert_eq!(
            effect_delta(&solved(&[1, 2], &[3]), &solved(&[1], &[3, 4])),
            EffectDelta::Incomparable
        );
    }

    /// A single component that overlaps without containment is incomparable on
    /// its own.
    #[test]
    fn disjoint_component_is_incomparable() {
        assert_eq!(
            effect_delta(&solved(&[1, 2], &[3]), &solved(&[2, 5], &[3])),
            EffectDelta::Incomparable
        );
    }

    #[test]
    fn empty_summaries_compare_equal_and_order_against_nonempty() {
        assert_eq!(
            effect_delta(&solved(&[], &[]), &solved(&[], &[])),
            EffectDelta::Equal
        );
        assert_eq!(
            effect_delta(&solved(&[1], &[]), &solved(&[], &[])),
            EffectDelta::Narrowed
        );
        assert_eq!(
            effect_delta(&solved(&[], &[]), &solved(&[1], &[])),
            EffectDelta::Widened
        );
    }

    // ---- materialization --------------------------------------------------

    #[test]
    fn identical_materialized_maps_are_equal() {
        assert_eq!(
            effect_delta(&materialized(&[1, 2]), &materialized(&[1, 2])),
            EffectDelta::Equal
        );
    }

    /// A changed binding map is a convention change: callers must be revisited,
    /// so it must never classify as `Equal`.
    #[test]
    fn differing_materialized_maps_never_stop_propagation() {
        let delta = effect_delta(&materialized(&[1]), &materialized(&[1, 2]));
        assert_eq!(delta, EffectDelta::Incomparable);
        assert!(!delta.allows_stopping());
    }

    /// Materialization changes how callers bind, not how much the callee may do,
    /// so it is not ordered against `Solved` in either direction.
    #[test]
    fn materialized_is_not_ordered_against_solved() {
        assert_eq!(
            effect_delta(&solved(&[1], &[2]), &materialized(&[1])),
            EffectDelta::Incomparable
        );
        assert_eq!(
            effect_delta(&materialized(&[1]), &solved(&[1], &[2])),
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
            FunctionEffects::Unsolved,
            FunctionEffects::Top,
            solved(&[], &[]),
            solved(&[1, 2], &[3]),
            materialized(&[1, 2]),
        ];
        for e in &all {
            assert_eq!(effect_delta(e, e), EffectDelta::Equal, "{e:?}");
        }
    }

    /// Antisymmetry of direction: swapping the arguments swaps
    /// narrowed/widened and fixes equal/incomparable. This is the property that
    /// catches a polarity inversion anywhere in the comparison chain.
    #[test]
    fn swapping_arguments_inverts_direction() {
        let all = [
            FunctionEffects::Unsolved,
            FunctionEffects::Top,
            solved(&[], &[]),
            solved(&[1], &[]),
            solved(&[1, 2], &[3]),
            solved(&[2, 5], &[3]),
            materialized(&[1]),
            materialized(&[1, 2]),
        ];
        for old in &all {
            for new in &all {
                let forward = effect_delta(old, new);
                let backward = effect_delta(new, old);
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
