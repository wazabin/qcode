//! Use edges: the operand relation read backwards.
//!
//! Every operand occurrence in a function body is one [`Use`]: instruction
//! `user`'s operand number `operand_index` names `value`. The uses of one
//! value form a singly linked list threaded through the edges (their `next`),
//! whose head lives on the value itself ([`WithUsers`]) — an instruction,
//! block parameter, block or temporary owns its head; a shared value (a
//! literal, varnode, bytes, function or poison) may be used by many bodies,
//! so each body keeps its heads for those in a sparse map. The list is
//! prepended to, so its order is unspecified.
//!
//! The edges are derived bookkeeping: they are skipped by serde and rebuilt
//! from the operands after deserialization
//! ([`FunctionBody::rebuild_uses`](crate::value::FunctionBody::rebuild_uses)),
//! and [`UseId`] never leaves the crate.

use jstd::{Identifier, recycling_arena::RecyclingArena};
use rustc_hash::FxHashMap;

use crate::value::{LocalValueId, insn::LocalInsnId, link::Link};

/// Index into a body's [`UseArena`]. Crate-private: an edge is only ever
/// reached through a value's use list.
#[derive(Identifier)]
pub(crate) struct UseId(u32);

/// One operand occurrence: `user`'s operand `operand_index` names `value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Use {
    pub(crate) value: LocalValueId,
    pub(crate) user: LocalInsnId,
    pub(crate) operand_index: u16,
    /// The next use of `value`, if any.
    pub(crate) next: Option<UseId>,
}

/// A value that owns the head of its own use list.
pub(crate) trait WithUsers {
    fn first_use(&self) -> Option<UseId>;
    fn set_first_use(&mut self, head: Option<UseId>);
}

/// The use-list heads of the shared values a body uses — literals, bytes,
/// varnodes, functions, poison — which cannot carry a head themselves: use
/// tracking is per body, and bodies are mutated independently.
///
/// Literals and varnodes, which every instruction of a lift names, are
/// dense by id: the head of a literal is a slot in a vector, not a hash
/// probe. The rest are sparse.
#[derive(Debug, Clone, Default)]
pub(crate) struct SharedHeads {
    literals: Vec<Link<UseId>>,
    varnodes: Vec<Link<UseId>>,
    other: FxHashMap<LocalValueId, UseId>,
}

impl SharedHeads {
    #[inline]
    pub(crate) fn get(&self, value: LocalValueId) -> Option<UseId> {
        match value {
            LocalValueId::Literal(id) => self.literals.get(usize::from(id))?.get(),
            LocalValueId::Varnode(id) => self.varnodes.get(usize::from(id))?.get(),
            _ => self.other.get(&value).copied(),
        }
    }

    #[inline]
    pub(crate) fn set(&mut self, value: LocalValueId, head: Option<UseId>) {
        let dense = match value {
            LocalValueId::Literal(id) => (&mut self.literals, usize::from(id)),
            LocalValueId::Varnode(id) => (&mut self.varnodes, usize::from(id)),
            _ => {
                match head {
                    Some(head) => {
                        self.other.insert(value, head);
                    }
                    None => {
                        self.other.remove(&value);
                    }
                }
                return;
            }
        };
        let (heads, at) = dense;
        if at >= heads.len() {
            if head.is_none() {
                return;
            }
            heads.resize(at + 1, Link::none());
        }
        heads[at].set(head);
    }

    /// Every shared value with a use, in no particular order.
    pub(crate) fn values(&self) -> impl Iterator<Item = LocalValueId> + '_ {
        let literals = self
            .literals
            .iter()
            .enumerate()
            .filter(|(_, head)| head.get().is_some())
            .map(|(id, _)| LocalValueId::Literal(crate::value::LiteralId::from(id)));
        let varnodes = self
            .varnodes
            .iter()
            .enumerate()
            .filter(|(_, head)| head.get().is_some())
            .map(|(id, _)| LocalValueId::Varnode(crate::value::VarnodeId::from(id)));
        literals.chain(varnodes).chain(self.other.keys().copied())
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.values().next().is_none()
    }

    pub(crate) fn clear(&mut self) {
        self.literals.clear();
        self.varnodes.clear();
        self.other.clear();
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        self.literals.shrink_to_fit();
        self.varnodes.shrink_to_fit();
        self.other.shrink_to_fit();
    }
}

/// Body-local slab storage for use edges. Removed slots are reused; `UseId`
/// remains stable only while its edge is live and never escapes the body.
pub(crate) type UseArena = RecyclingArena<UseId, Use>;

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(raw: usize) -> Use {
        Use {
            value: LocalValueId::Instruction(LocalInsnId::from(raw)),
            user: LocalInsnId::from(raw + 100),
            operand_index: 0,
            next: None,
        }
    }

    #[test]
    fn freed_slots_are_reused_most_recent_first() {
        let mut arena = UseArena::default();
        let a = arena.push(edge(0));
        let b = arena.push(edge(1));
        let c = arena.push(edge(2));
        assert_eq!(arena.len(), 3);

        assert_eq!(arena.remove(b), edge(1));
        assert_eq!(arena.remove(a), edge(0));
        assert_eq!(arena.len(), 1);
        assert!(!arena.contains(a));
        assert!(!arena.contains(b));
        assert!(arena.contains(c));

        assert_eq!(arena.push(edge(3)), a, "the last freed slot goes first");
        assert_eq!(arena.push(edge(4)), b);
        assert_eq!(arena.push(edge(5)), UseId::from(3), "then the slab grows");
        assert_eq!(arena.len(), 4);

        let live: Vec<UseId> = arena.iter().map(|edge| edge.id).collect();
        assert_eq!(live, vec![a, b, c, UseId::from(3)]);
    }

    #[test]
    fn iteration_skips_holes_and_yields_identified_edges() {
        let mut arena = UseArena::default();
        assert!(arena.is_empty());
        let ids: Vec<UseId> = (0..5).map(|raw| arena.push(edge(raw))).collect();
        arena.remove(ids[1]);
        arena.remove(ids[3]);
        assert!(!arena.is_empty());
        assert_eq!(arena.len(), 3);

        let seen: Vec<(UseId, Use)> = arena.iter().map(|e| (e.id, *e.inner)).collect();
        assert_eq!(
            seen,
            vec![(ids[0], edge(0)), (ids[2], edge(2)), (ids[4], edge(4))]
        );
        assert_eq!(arena[ids[2]], edge(2), "indexing yields the payload");
    }

    #[test]
    fn free_and_unknown_ids_do_not_resolve() {
        let mut arena = UseArena::default();
        let a = arena.push(edge(0));
        let b = arena.push(edge(1));
        arena.remove(a);

        assert!(!arena.contains(a));
        assert!(arena.get(a).is_none());
        assert!(arena.get_mut(a).is_none());
        assert!(!arena.contains(UseId::from(7)), "past the slab");
        assert!(arena.get(UseId::from(7)).is_none());

        let mut live = arena.get_mut(b).expect("b is live");
        assert_eq!(live.id, b);
        live.next = Some(b);
        assert_eq!(arena.get(b).map(|e| e.next), Some(Some(b)));

        // The freed id aliases the next edge stored: a stale `a` would now
        // read a different edge, which is why no id outlives its edge.
        assert_eq!(arena.push(edge(9)), a);
        assert_eq!(arena[a], edge(9));
    }

    #[test]
    fn clear_forgets_the_free_list_and_restarts_ids() {
        let mut arena = UseArena::default();
        let a = arena.push(edge(0));
        let b = arena.push(edge(1));
        arena.remove(a);
        arena.clear();
        assert!(arena.is_empty());
        assert!(!arena.contains(a));
        assert!(!arena.contains(b));
        assert_eq!(arena.push(edge(2)), UseId::from(0), "ids restart at zero");
        assert_eq!(
            arena.push(edge(3)),
            UseId::from(1),
            "the old free list is gone"
        );
        arena.shrink_to_fit();
        assert_eq!(arena.iter().count(), 2);
    }

    #[test]
    fn churn_never_grows_the_slab_past_its_peak() {
        let mut arena = UseArena::default();
        let mut live: Vec<UseId> = (0..8).map(|raw| arena.push(edge(raw))).collect();
        let peak = 8;
        for round in 0..100 {
            // Drop half, then refill: the slab must reuse the freed slots.
            for _ in 0..4 {
                let id = live.remove(round % live.len());
                arena.remove(id);
            }
            for raw in 0..4 {
                let id = arena.push(edge(raw + round));
                assert!(usize::from(id) < peak, "slot {id:?} beyond the peak");
                live.push(id);
            }
            assert_eq!(arena.len(), peak);
            assert_eq!(arena.iter().count(), peak);
        }
        let mut seen: Vec<usize> = arena.iter().map(|e| usize::from(e.id)).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..peak).collect::<Vec<_>>());
    }

    #[test]
    fn cloning_copies_the_slots_and_the_free_list() {
        let mut arena = UseArena::default();
        let a = arena.push(edge(0));
        let b = arena.push(edge(1));
        arena.remove(a);
        let mut copy = arena.clone();
        assert_eq!(copy.len(), 1);
        assert!(!copy.contains(a));
        assert_eq!(copy[b], edge(1));
        assert_eq!(copy.push(edge(2)), a, "the clone reuses the same free slot");
        assert!(!arena.contains(a), "and the original is untouched");
    }

    #[test]
    #[should_panic(expected = "is already free")]
    fn double_remove_panics() {
        let mut arena = UseArena::default();
        let a = arena.push(edge(0));
        arena.remove(a);
        arena.remove(a);
    }

    #[test]
    #[should_panic(expected = "is free")]
    fn indexing_a_free_slot_panics() {
        let mut arena = UseArena::default();
        let a = arena.push(edge(0));
        arena.remove(a);
        let _ = arena[a];
    }
}
