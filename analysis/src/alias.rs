use std::collections::HashMap;

use qcode::{
    context::Context,
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
}

impl AliasResult {
    /// The alias equivalence-class of `a`, or `None` if `a` was not
    /// involved in any constraint during analysis.
    pub fn alias_class(&self, a: ValueId) -> Option<NodeId> {
        self.value_to_root.get(&a).copied()
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
