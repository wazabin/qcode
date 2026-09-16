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

use jstd::Identifier;

use crate::value::{LocalValueId, insn::LocalInsnId};

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
    fn first_use_mut(&mut self) -> &mut Option<UseId>;
}

/// One arena slot: a live edge, or a free slot on the free list.
#[derive(Debug, Clone, Copy)]
enum Slot {
    Live(Use),
    /// The next free slot after this one, if any.
    Free(Option<UseId>),
}

impl Slot {
    fn live(&self) -> Option<&Use> {
        match self {
            Slot::Live(edge) => Some(edge),
            Slot::Free(_) => None,
        }
    }

    fn live_mut(&mut self) -> Option<&mut Use> {
        match self {
            Slot::Live(edge) => Some(edge),
            Slot::Free(_) => None,
        }
    }
}

/// Slab storage for a body's use edges. A removed slot is reused by the next
/// insertion (through a free list threaded through the free slots), so a
/// body that churns — lifting a block, then deleting most of it — does not
/// grow the arena past its peak. Ids are stable while their edge is live;
/// nothing outside the body holds one, so reuse is safe.
#[derive(Debug, Clone, Default)]
pub(crate) struct UseArena {
    slots: Vec<Slot>,
    free: Option<UseId>,
    live: usize,
}

impl UseArena {
    /// Stores `edge`, reusing a freed slot when there is one.
    pub(crate) fn push(&mut self, edge: Use) -> UseId {
        self.live += 1;
        match self.free {
            Some(id) => {
                let slot = &mut self.slots[usize::from(id)];
                let Slot::Free(next) = *slot else {
                    unreachable!("use {id:?} is on the free list but live");
                };
                self.free = next;
                *slot = Slot::Live(edge);
                id
            }
            None => {
                let id = UseId::from(self.slots.len());
                self.slots.push(Slot::Live(edge));
                id
            }
        }
    }

    /// Frees `id`'s slot, returning the edge it held.
    pub(crate) fn remove(&mut self, id: UseId) -> Use {
        let slot = &mut self.slots[usize::from(id)];
        let Slot::Live(edge) = *slot else {
            panic!("use {id:?} is already free");
        };
        *slot = Slot::Free(self.free);
        self.free = Some(id);
        self.live -= 1;
        edge
    }

    /// The number of live edges.
    pub(crate) fn len(&self) -> usize {
        self.live
    }

    /// Whether `id` names a live edge.
    pub(crate) fn contains(&self, id: UseId) -> bool {
        self.slots
            .get(usize::from(id))
            .is_some_and(|slot| slot.live().is_some())
    }

    /// Every live edge, in slot order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (UseId, &Use)> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(raw, slot)| Some((UseId::from(raw), slot.live()?)))
    }

    /// Drops every edge.
    pub(crate) fn clear(&mut self) {
        self.slots.clear();
        self.free = None;
        self.live = 0;
    }

    /// Releases capacity beyond the slots in use (live or on the free list).
    pub(crate) fn shrink_to_fit(&mut self) {
        self.slots.shrink_to_fit();
    }
}

impl std::ops::Index<UseId> for UseArena {
    type Output = Use;

    fn index(&self, id: UseId) -> &Use {
        self.slots[usize::from(id)]
            .live()
            .unwrap_or_else(|| panic!("use {id:?} is free"))
    }
}

impl std::ops::IndexMut<UseId> for UseArena {
    fn index_mut(&mut self, id: UseId) -> &mut Use {
        self.slots[usize::from(id)]
            .live_mut()
            .unwrap_or_else(|| panic!("use {id:?} is free"))
    }
}

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

        let live: Vec<UseId> = arena.iter().map(|(id, _)| id).collect();
        assert_eq!(live, vec![a, b, c, UseId::from(3)]);
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
