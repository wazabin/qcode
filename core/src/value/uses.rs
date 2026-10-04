//! Use edges: the operand relation read backwards.
//!
//! Every operand occurrence in a function body is one [`Use`]: an instruction
//! `user`'s operand number `i` names a value. The edges of one instruction's
//! operands are consecutive in the body's [`UseSlab`], a *run* that the
//! instruction names ([`Instruction::use_run`](crate::value::Instruction)),
//! so the edge of operand `i` is the run's `i`th: reaching it from the user
//! costs nothing, and so does the user's operand number from the edge.
//!
//! The uses of one value form a doubly linked list threaded through the edges
//! (their `next` and `prev`), whose head lives on the value itself
//! ([`WithUsers`]) — an instruction, block parameter, block or temporary owns
//! its head; a shared value (a literal, varnode, bytes, function or poison)
//! may be used by many bodies, so each body keeps its heads for those in a
//! sparse map. The list is prepended to, so its order is unspecified, and an
//! edge comes out of it in constant time wherever it stands.
//!
//! An operand whose value this body has no storage for — one a transformation
//! is in the middle of removing — has its edge in the run but in no list: the
//! edge is *unlinked*. An edge is linked when it has a `prev`, or is the head
//! of its value's list; the caller, who knows the value, tells the two apart.
//!
//! The edges are derived bookkeeping: they are skipped by serde and rebuilt
//! from the operands after deserialization
//! ([`FunctionBody::rebuild_uses`](crate::value::FunctionBody::rebuild_uses)),
//! and [`UseId`] never leaves the crate.

use std::ops::{Index, IndexMut};

use jstd::Identifier;
use rustc_hash::FxHashMap;

use crate::value::{LocalValueId, insn::LocalInsnId, link::Link};

/// Index into a body's [`UseSlab`]. Crate-private: an edge is only ever
/// reached through a value's use list or its user's run.
#[derive(Identifier)]
pub(crate) struct UseId(u32);

/// One operand occurrence of `user`, in the lists of the value it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Use {
    pub(crate) user: LocalInsnId,
    next: Link<UseId>,
    prev: Link<UseId>,
}

impl Use {
    /// The next use of the same value, if any.
    pub(crate) fn next(&self) -> Option<UseId> {
        self.next.get()
    }

    /// The previous use of the same value; `None` at the head of the list,
    /// and for an unlinked edge.
    pub(crate) fn prev(&self) -> Option<UseId> {
        self.prev.get()
    }
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
/// Sparse, by hash: the ids of shared values are the module's, and a body
/// names a handful of the module's millions. A table indexed by id would
/// give every body the size of the module, and a module of tens of
/// thousands of functions the square of it.
#[derive(Debug, Clone, Default)]
pub(crate) struct SharedHeads {
    heads: FxHashMap<LocalValueId, UseId>,
}

impl SharedHeads {
    #[inline]
    pub(crate) fn get(&self, value: LocalValueId) -> Option<UseId> {
        self.heads.get(&value).copied()
    }

    #[inline]
    pub(crate) fn set(&mut self, value: LocalValueId, head: Option<UseId>) {
        match head {
            Some(head) => {
                self.heads.insert(value, head);
            }
            None => {
                self.heads.remove(&value);
            }
        }
    }

    /// Every shared value with a use, in no particular order.
    pub(crate) fn values(&self) -> impl Iterator<Item = LocalValueId> + '_ {
        self.heads.keys().copied()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.heads.clear();
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        self.heads.shrink_to_fit();
    }

    /// The bytes the heads take, by capacity.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.heads.capacity() * (std::mem::size_of::<(LocalValueId, UseId)>() + 1)
    }
}

/// A body's use edges, in runs: one run per instruction with operands, as
/// long as it has operands, which the body hands back on removal.
///
/// A freed run is kept on a list of the runs of its length, so the next
/// instruction with that many operands takes it over: the slab grows to the
/// peak of its live runs, not to the number ever made.
#[derive(Debug, Clone, Default)]
pub(crate) struct UseSlab {
    edges: Vec<Use>,
    /// `free[n]` heads the free runs of `n` edges, threaded through the
    /// `next` of each run's first edge.
    free: Vec<Link<UseId>>,
    /// How many edges are in a use list.
    linked: usize,
}

impl UseSlab {
    /// A run of `n >= 1` unlinked edges for `user`'s operands, reusing a free
    /// run of the same length when there is one.
    pub(crate) fn alloc_run(&mut self, user: LocalInsnId, n: usize) -> UseId {
        debug_assert!(n > 0, "an instruction without operands has no run");
        let unlinked = Use {
            user,
            next: Link::none(),
            prev: Link::none(),
        };
        if let Some(run) = self.free.get(n).and_then(|head| head.get()) {
            self.free[n] = self.edges[usize::from(run)].next;
            self.edges[usize::from(run)..usize::from(run) + n].fill(unlinked);
            return run;
        }
        let at = self.edges.len();
        let run = UseId::from(at);
        assert_eq!(usize::from(run), at, "use slab identifier space exhausted");
        self.edges.resize(at + n, unlinked);
        run
    }

    /// Frees the run of `n` edges at `run`. Every edge in it must be
    /// unlinked.
    pub(crate) fn free_run(&mut self, run: UseId, n: usize) {
        debug_assert!(
            self.edges[usize::from(run)..usize::from(run) + n]
                .iter()
                .all(|edge| edge.prev.get().is_none()),
            "a run is freed with a linked edge"
        );
        if self.free.len() <= n {
            self.free.resize(n + 1, Link::none());
        }
        self.edges[usize::from(run)].next = self.free[n];
        self.free[n] = Link::from(Some(run));
    }

    /// The edge `index` places on from `run`: the edge of operand `index`.
    pub(crate) fn at(run: UseId, index: usize) -> UseId {
        UseId::from(usize::from(run) + index)
    }

    /// The operand number of `edge` in its user's run.
    pub(crate) fn operand_index(edge: UseId, run: UseId) -> usize {
        usize::from(edge) - usize::from(run)
    }

    /// Puts `edge`, which is unlinked, at the front of the list headed by
    /// `head`. The caller makes `edge` the head.
    #[inline]
    pub(crate) fn push_front(&mut self, edge: UseId, head: Option<UseId>) {
        if let Some(head) = head {
            self.edges[usize::from(head)].prev = Link::from(Some(edge));
        }
        let at = &mut self.edges[usize::from(edge)];
        at.next = Link::from(head);
        at.prev = Link::none();
        self.linked += 1;
    }

    /// Takes the linked `edge` out of its list, joining its neighbours, and
    /// leaves it unlinked. Returns the neighbours, so that a caller finding
    /// no `prev` can move the head to the `next`.
    #[inline]
    pub(crate) fn unlink(&mut self, edge: UseId) -> (Option<UseId>, Option<UseId>) {
        let Use { next, prev, .. } = self.edges[usize::from(edge)];
        if let Some(prev) = prev.get() {
            self.edges[usize::from(prev)].next = next;
        }
        if let Some(next) = next.get() {
            self.edges[usize::from(next)].prev = prev;
        }
        let at = &mut self.edges[usize::from(edge)];
        at.next = Link::none();
        at.prev = Link::none();
        self.linked -= 1;
        (prev.get(), next.get())
    }

    /// Forgets that `edge` was in a list, without touching its neighbours:
    /// for a whole list that is being dropped at once.
    #[inline]
    pub(crate) fn detach(&mut self, edge: UseId) {
        let at = &mut self.edges[usize::from(edge)];
        at.next = Link::none();
        at.prev = Link::none();
        self.linked -= 1;
    }

    /// How many edges are in a use list.
    pub(crate) fn len(&self) -> usize {
        self.linked
    }

    /// Whether no edge is in a use list.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.linked == 0
    }

    /// How many edges the slab has issued, live runs and free ones.
    pub(crate) fn slots(&self) -> usize {
        self.edges.len()
    }

    /// Every free run, as its length and its first edge.
    pub(crate) fn free_runs(&self) -> impl Iterator<Item = (usize, UseId)> + '_ {
        self.free.iter().enumerate().flat_map(move |(n, head)| {
            let mut at = head.get();
            std::iter::from_fn(move || {
                let run = at?;
                at = self.edges[usize::from(run)].next.get();
                Some((n, run))
            })
        })
    }

    /// How many edges are in free runs.
    #[cfg(test)]
    pub(crate) fn free_slots(&self) -> usize {
        self.free_runs().map(|(n, _)| n).sum()
    }

    pub(crate) fn clear(&mut self) {
        self.edges.clear();
        self.free.clear();
        self.linked = 0;
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        self.edges.shrink_to_fit();
        self.free.shrink_to_fit();
    }

    /// The bytes the slab holds, by capacity.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.edges.capacity() * std::mem::size_of::<Use>()
            + self.free.capacity() * std::mem::size_of::<Link<UseId>>()
    }
}

impl Index<UseId> for UseSlab {
    type Output = Use;

    #[inline]
    fn index(&self, edge: UseId) -> &Use {
        &self.edges[usize::from(edge)]
    }
}

impl IndexMut<UseId> for UseSlab {
    #[inline]
    fn index_mut(&mut self, edge: UseId) -> &mut Use {
        &mut self.edges[usize::from(edge)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(raw: usize) -> LocalInsnId {
        LocalInsnId::from(raw)
    }

    #[test]
    fn a_freed_run_is_reused_by_a_run_of_its_length() {
        let mut slab = UseSlab::default();
        let a = slab.alloc_run(user(0), 2);
        let b = slab.alloc_run(user(1), 1);
        let c = slab.alloc_run(user(2), 2);
        assert_eq!((usize::from(a), usize::from(b), usize::from(c)), (0, 2, 3));
        assert_eq!(slab.slots(), 5);

        slab.free_run(a, 2);
        assert_eq!(slab.free_slots(), 2);
        assert_eq!(
            slab.alloc_run(user(3), 1),
            UseId::from(5),
            "no free run of one"
        );
        assert_eq!(slab.alloc_run(user(4), 2), a, "the free run of two");
        assert_eq!(slab[a].user, user(4), "and its edges are its new user's");
        assert_eq!(slab.free_slots(), 0);
        assert_eq!(slab.slots(), 6);
    }

    #[test]
    fn edges_link_and_unlink_in_constant_time_from_anywhere() {
        let mut slab = UseSlab::default();
        let run = slab.alloc_run(user(0), 4);
        let edge = |i| UseSlab::at(run, i);
        let mut head = None;
        for i in 0..4 {
            slab.push_front(edge(i), head);
            head = Some(edge(i));
        }
        assert_eq!(slab.len(), 4);
        // Order is 3, 2, 1, 0. Take the middle, then the head, then the tail.
        assert_eq!(slab.unlink(edge(2)), (Some(edge(3)), Some(edge(1))));
        assert_eq!(slab.unlink(edge(3)), (None, Some(edge(1))));
        head = Some(edge(1));
        assert_eq!(slab.unlink(edge(0)), (Some(edge(1)), None));
        assert_eq!(slab[edge(1)].next(), None);
        assert_eq!(slab[edge(1)].prev(), None);
        assert_eq!(head, Some(edge(1)));
        assert_eq!(slab.len(), 1);
        assert_eq!(UseSlab::operand_index(edge(3), run), 3);
    }
}
