//! The blocks a function body owns, in the order they were added.
//!
//! A roster is a list with a position index, so a block comes off it in
//! constant time wherever it stands: removal leaves a hole where the block
//! was, and the holes are squeezed out once they outnumber the blocks, which
//! makes removing `n` blocks cost `O(n)` in all. Order is that of addition,
//! holes skipped, and is what [`FunctionBody::roster`](crate::value::FunctionBody)
//! consumers see.

use crate::value::block::cfg::LocalBlockId;

/// The block ids a body's roster lists, in order, over a position index.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(from = "Vec<LocalBlockId>", into = "Vec<LocalBlockId>")]
pub(crate) struct Roster {
    /// Listed blocks and holes, in order.
    order: Vec<LocalBlockId>,
    /// The position in `order` of each block id, by id; `NOT_LISTED` for a
    /// block that is not listed. Grows with the largest id listed.
    position: Vec<u32>,
    /// How many blocks are listed: `order` less its holes.
    len: usize,
}

const NOT_LISTED: u32 = u32::MAX;

/// What `order` holds where a block was: the one id no arena issues.
fn hole() -> LocalBlockId {
    LocalBlockId::from(u32::MAX as usize)
}

impl Roster {
    /// Lists `block` after the others. A block already listed stays where it
    /// is.
    pub(crate) fn push(&mut self, block: LocalBlockId) {
        if self.contains(block) {
            return;
        }
        let id = usize::from(block);
        assert_ne!(block, hole(), "the last id is the hole");
        if id >= self.position.len() {
            self.position.resize(id + 1, NOT_LISTED);
        }
        self.position[id] = u32::try_from(self.order.len()).expect("a roster fits u32 positions");
        self.order.push(block);
        self.len += 1;
    }

    /// Whether `block` is listed.
    pub(crate) fn contains(&self, block: LocalBlockId) -> bool {
        self.position
            .get(usize::from(block))
            .is_some_and(|&at| at != NOT_LISTED)
    }

    /// Takes `block` off the list; whether it was on it.
    pub(crate) fn remove(&mut self, block: LocalBlockId) -> bool {
        let Some(slot) = self.position.get_mut(usize::from(block)) else {
            return false;
        };
        let at = std::mem::replace(slot, NOT_LISTED);
        if at == NOT_LISTED {
            return false;
        }
        self.order[at as usize] = hole();
        self.len -= 1;
        // Holes are squeezed out when they outnumber what is listed, so the
        // list is never much longer than its blocks and the work of
        // squeezing, a pass over the list, is paid for by the removals
        // that made the holes.
        if self.order.len() > 2 * self.len + 16 {
            self.squeeze();
        }
        true
    }

    fn squeeze(&mut self) {
        self.order.retain(|&block| block != hole());
        for (at, block) in self.order.iter().enumerate() {
            self.position[usize::from(*block)] = at as u32;
        }
    }

    /// The listed blocks, in order.
    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = LocalBlockId> + '_ {
        self.order.iter().copied().filter(|&block| block != hole())
    }

    /// How many blocks are listed.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn clear(&mut self) {
        self.order.clear();
        self.position.clear();
        self.len = 0;
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        self.squeeze();
        self.order.shrink_to_fit();
        self.position.truncate(
            self.order
                .iter()
                .map(|&block| usize::from(block) + 1)
                .max()
                .unwrap_or(0),
        );
        self.position.shrink_to_fit();
    }

    /// The bytes the roster holds, by capacity.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.order.capacity() * std::mem::size_of::<LocalBlockId>()
            + self.position.capacity() * std::mem::size_of::<u32>()
    }

    /// Lists `block` a second time, which a roster never does: for tests of
    /// the integrity check that reports it.
    #[cfg(test)]
    pub(crate) fn push_again(&mut self, block: LocalBlockId) {
        self.order.push(block);
        self.len += 1;
    }
}

impl PartialEq for Roster {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl Eq for Roster {}

impl From<Vec<LocalBlockId>> for Roster {
    fn from(blocks: Vec<LocalBlockId>) -> Self {
        let mut roster = Self::default();
        for block in blocks {
            roster.push(block);
        }
        roster
    }
}

impl From<Roster> for Vec<LocalBlockId> {
    fn from(roster: Roster) -> Self {
        roster.iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(raw: usize) -> LocalBlockId {
        LocalBlockId::from(raw)
    }

    fn listed(roster: &Roster) -> Vec<usize> {
        roster.iter().map(usize::from).collect()
    }

    #[test]
    fn blocks_come_off_from_anywhere_and_the_order_of_the_rest_stays() {
        let mut roster = Roster::default();
        for raw in [4, 0, 7, 2, 9] {
            roster.push(id(raw));
        }
        assert_eq!(listed(&roster), vec![4, 0, 7, 2, 9]);
        assert!(roster.remove(id(7)));
        assert!(!roster.remove(id(7)), "already off");
        assert!(!roster.remove(id(3)), "never on");
        assert!(roster.remove(id(4)));
        assert_eq!(listed(&roster), vec![0, 2, 9]);
        assert_eq!(roster.len(), 3);
        assert!(roster.contains(id(9)) && !roster.contains(id(7)));
        roster.push(id(7));
        assert_eq!(listed(&roster), vec![0, 2, 9, 7]);
    }

    #[test]
    fn a_block_listed_twice_is_listed_once() {
        let mut roster = Roster::default();
        roster.push(id(1));
        roster.push(id(1));
        assert_eq!(listed(&roster), vec![1]);
    }

    #[test]
    fn removing_many_squeezes_the_holes_out() {
        let mut roster = Roster::default();
        for raw in 0..1000 {
            roster.push(id(raw));
        }
        for raw in (0..1000).filter(|raw| raw % 10 != 0) {
            assert!(roster.remove(id(raw)));
        }
        assert_eq!(roster.len(), 100);
        assert!(roster.order.len() <= 2 * 100 + 16 + 1, "holes are squeezed");
        assert_eq!(
            listed(&roster),
            (0..1000).step_by(10).collect::<Vec<_>>(),
            "in the order added"
        );
        assert!((0..1000).all(|raw| roster.contains(id(raw)) == (raw % 10 == 0)));
    }

    #[test]
    fn it_round_trips_through_the_list_it_serializes_as() {
        let mut roster = Roster::default();
        for raw in [3, 1, 2] {
            roster.push(id(raw));
        }
        roster.remove(id(1));
        let wire: Vec<LocalBlockId> = roster.clone().into();
        assert_eq!(wire, vec![id(3), id(2)]);
        assert_eq!(Roster::from(wire), roster);
    }
}
