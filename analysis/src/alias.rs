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
        match (
            self.value_to_interval.get(&a),
            self.value_to_interval.get(&b),
        ) {
            (Some(ia), Some(ib)) => ia == ib,
            _ => false,
        }
    }

    /// Returns true if `b`'s byte interval fully contains `a`'s (same space,
    /// `start_b <= start_a`, `end_b >= end_a`). Falls back to `false` when
    /// precise interval information is unavailable for either value.
    pub fn covers(&self, a: ValueId, b: ValueId) -> bool {
        match (
            self.value_to_interval.get(&a),
            self.value_to_interval.get(&b),
        ) {
            (Some(&(sa, start_a, end_a)), Some(&(sb, start_b, end_b))) => {
                sa == sb && start_b <= start_a && end_b >= end_a
            }
            _ => false,
        }
    }

    /// Returns all tracked pointer values whose interval is strictly contained
    /// within `ptr`'s interval in the same space, as
    /// `(sub_ptr, byte_offset_within_ptr, sub_size)`.
    pub fn sub_intervals_of(&self, ptr: ValueId) -> Vec<(ValueId, usize, usize)> {
        let Some(&(ptr_space, ptr_start, ptr_end)) = self.value_to_interval.get(&ptr) else {
            return Vec::new();
        };
        self.value_to_interval
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
            .collect()
    }

    /// Conservative may-alias query.
    ///
    /// Returns `true` if `a` and `b` share a class, or if either is
    /// `NodeId::Unknown`. Returns `false` if either value was never
    /// involved in any constraint (isolated - no alias relationship).
    pub fn may_alias(&self, ctx: &Context, a: ValueId, b: ValueId) -> bool {
        // If a and b don't share the same address space they can't alias.
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
