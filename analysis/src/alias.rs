use std::collections::HashMap;

use qcode::{
    context::Context,
    space::SpaceId,
    value::{ValueId, ValueRef},
};

mod anderson;
mod simple;

pub use anderson::alias_analysis;

/// Abstract node in the alias graph.
///
/// `Unknown` is the top element: any value joined to it may alias everything.
/// `Id(n)` is a concrete node allocated during analysis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeId {
    Unknown,
    Id(usize),
}

pub struct AliasResult {
    pub(crate) value_to_root: HashMap<ValueId, NodeId>,
    /// Exact byte intervals for pointer values whose location could be
    /// statically resolved: `(space_id, byte_start, byte_end)`.
    /// Populated by location-aware analyses (e.g. `simple`); empty otherwise.
    pub(crate) value_to_interval: HashMap<ValueId, (SpaceId, u64, u64)>,
}

impl AliasResult {
    /// The alias equivalence-class of `a`, or `None` if `a` was not
    /// involved in any constraint during analysis.
    pub fn alias_class(&self, a: ValueId) -> Option<NodeId> {
        self.value_to_root.get(&a).copied()
    }

    /// The exact byte interval `(space, start, end)` of `a`, when a
    /// location-aware analysis (e.g. [`AliasResult::simple`]) could statically
    /// resolve it. Returns `None` for values with no precise location.
    pub fn interval(&self, a: ValueId) -> Option<(SpaceId, u64, u64)> {
        self.value_to_interval.get(&a).copied()
    }

    /// Conservative must-alias query.
    ///
    /// Returns `true` only when `a` and `b` are guaranteed to refer to the
    /// exact same byte range (same address space, start, and end). Defaults
    /// to `false` when precise location information is unavailable.
    pub fn must_alias(&self, a: ValueId, b: ValueId) -> bool {
        self.interval(a)
            .zip(self.interval(b))
            .is_some_and(|(ia, ib)| ia == ib)
    }

    /// Returns true if `outer`'s byte interval fully contains `inner`'s (same
    /// space, `start_outer <= start_inner`, `end_outer >= end_inner`). Falls
    /// back to `false` when precise interval information is unavailable for
    /// either value.
    pub fn covers(&self, inner: ValueId, outer: ValueId) -> bool {
        self.interval(inner).zip(self.interval(outer)).is_some_and(
            |((si, start_i, end_i), (so, start_o, end_o))| {
                si == so && start_o <= start_i && end_o >= end_i
            },
        )
    }

    /// Returns all other tracked pointer values whose interval is contained
    /// within `ptr`'s interval in the same space (including values covering
    /// exactly the same range under a different `ValueId`), as
    /// `(sub_ptr, byte_offset_within_ptr, sub_size)`, sorted by offset then
    /// size for deterministic output.
    pub fn sub_intervals_of(&self, ptr: ValueId) -> Vec<(ValueId, usize, usize)> {
        let Some(&(ptr_space, ptr_start, ptr_end)) = self.value_to_interval.get(&ptr) else {
            return Vec::new();
        };
        let mut subs: Vec<_> = self
            .value_to_interval
            .iter()
            .filter_map(|(&other, &(space, start, end))| {
                if other == ptr || space != ptr_space {
                    return None;
                }
                if start >= ptr_start && end <= ptr_end {
                    let byte_offset = usize::try_from(start - ptr_start).ok()?;
                    let sub_size = usize::try_from(end - start).ok()?;
                    Some((other, byte_offset, sub_size))
                } else {
                    None
                }
            })
            .collect();
        subs.sort_unstable_by_key(|&(_, off, size)| (off, size));
        subs
    }

    /// Conservative may-alias query.
    ///
    /// Returns `true` if `a` and `b` share a class, or if either is
    /// `NodeId::Unknown`. Returns `false` if either value was never
    /// involved in any constraint (isolated - no alias relationship).
    pub fn may_alias(&self, ctx: &Context, a: ValueId, b: ValueId) -> bool {
        // If a and b don't share the same address space they can't alias.
        // Note this deliberately overrides `NodeId::Unknown` below: aliasing
        // means "same storage location", and values in different spaces can
        // never occupy the same location, however unknown their class is.
        let a_space = ValueRef::new(a, ctx).space().map(|s| s.id);
        let b_space = ValueRef::new(b, ctx).space().map(|s| s.id);
        if let (Some(sa), Some(sb)) = (a_space, b_space)
            && sa != sb
        {
            return false;
        }

        match (self.alias_class(a), self.alias_class(b)) {
            (None, _) | (_, None) => false,
            (Some(NodeId::Unknown), _) | (_, Some(NodeId::Unknown)) => true,
            (Some(ra), Some(rb)) => ra == rb,
        }
    }
}

#[cfg(test)]
mod tests {
    use qcode::{
        context::Context,
        space::{Space, SpaceType},
        value::InstructionId,
    };

    use super::*;

    fn value(n: usize) -> ValueId {
        ValueId::Instruction(InstructionId::from(n))
    }

    /// An `AliasResult` with hand-written intervals: value(0) spans [0, 8),
    /// value(1) spans [0, 4), value(2) spans [4, 8), value(3) also spans
    /// [0, 8) (equal to value(0)), and value(4) spans [0, 8) in another space.
    fn interval_fixture() -> AliasResult {
        let mut ctx = Context::new();
        let mut space_a = Space::new(Some("a"), 1, 8);
        space_a.ty = SpaceType::Register;
        let sa = ctx.add_space(space_a);
        let mut space_b = Space::new(Some("b"), 1, 8);
        space_b.ty = SpaceType::Register;
        let sb = ctx.add_space(space_b);

        let value_to_interval = HashMap::from([
            (value(0), (sa, 0, 8)),
            (value(1), (sa, 0, 4)),
            (value(2), (sa, 4, 8)),
            (value(3), (sa, 0, 8)),
            (value(4), (sb, 0, 8)),
        ]);
        AliasResult {
            value_to_root: HashMap::new(),
            value_to_interval,
        }
    }

    #[test]
    fn must_alias_requires_identical_interval_and_space() {
        let r = interval_fixture();
        assert!(r.must_alias(value(0), value(3)));
        assert!(!r.must_alias(value(0), value(1)), "containment is not must");
        assert!(!r.must_alias(value(0), value(4)), "different space");
        assert!(!r.must_alias(value(0), value(9)), "untracked value");
    }

    #[test]
    fn covers_is_outer_contains_inner() {
        let r = interval_fixture();
        assert!(
            r.covers(value(1), value(0)),
            "outer [0,8) covers inner [0,4)"
        );
        assert!(!r.covers(value(0), value(1)), "inner does not cover outer");
        assert!(r.covers(value(0), value(3)), "equal intervals cover");
        assert!(!r.covers(value(1), value(4)), "different space");
        assert!(!r.covers(value(1), value(9)), "untracked value");
    }

    #[test]
    fn sub_intervals_include_equal_ranges_and_are_sorted() {
        let r = interval_fixture();
        assert_eq!(
            r.sub_intervals_of(value(0)),
            vec![(value(1), 0, 4), (value(3), 0, 8), (value(2), 4, 4)],
            "subs sorted by (offset, size); equal-interval value(3) included; \
             other-space value(4) and value(0) itself excluded"
        );
        assert!(r.sub_intervals_of(value(9)).is_empty(), "untracked value");
    }
}
