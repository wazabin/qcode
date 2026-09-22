//! Four-byte optional ids for the links an IR node carries.
//!
//! An [`Option`] of a `u32`-backed id is eight bytes, since the id has no
//! niche. An instruction carries four such links (its block, its
//! neighbours, its first use) and a machine address, and a body holds
//! millions of instructions, so the links are packed: `None` is the one
//! bit pattern no arena ever issues.

use std::marker::PhantomData;

use jstd::registry::Identifier;

/// An optional `u32`-backed id in four bytes. Reads and writes go through
/// [`Option`], so the packing is invisible to the code that links nodes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Link<Id>(u32, PhantomData<Id>);

impl<Id> Link<Id> {
    const NONE: u32 = u32::MAX;

    pub(crate) const fn none() -> Self {
        Self(Self::NONE, PhantomData)
    }
}

impl<Id: Identifier> Link<Id> {
    pub(crate) fn get(self) -> Option<Id> {
        (self.0 != Self::NONE).then(|| Id::from(self.0 as usize))
    }

    pub(crate) fn set(&mut self, id: Option<Id>) {
        *self = Self::from(id);
    }

    pub(crate) fn take(&mut self) -> Option<Id> {
        std::mem::take(self).get()
    }
}

impl<Id: Identifier> From<Option<Id>> for Link<Id> {
    fn from(id: Option<Id>) -> Self {
        match id {
            Some(id) => {
                let raw: usize = id.into();
                let raw = u32::try_from(raw).expect("an id fits a link");
                assert_ne!(raw, Self::NONE, "the last id is the empty link");
                Self(raw, PhantomData)
            }
            None => Self::none(),
        }
    }
}

impl<Id: Identifier> Default for Link<Id> {
    fn default() -> Self {
        Self::none()
    }
}

impl<Id: Identifier + std::fmt::Debug> std::fmt::Debug for Link<Id> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.get().fmt(f)
    }
}

impl<Id: Identifier + serde::Serialize> serde::Serialize for Link<Id> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.get().serialize(serializer)
    }
}

impl<'de, Id: Identifier + serde::Deserialize<'de>> serde::Deserialize<'de> for Link<Id> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Option::<Id>::deserialize(deserializer).map(Self::from)
    }
}

/// An optional machine address in eight bytes: `None` is the all-ones
/// address, which no instruction is at.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PackedAddress(u64);

impl PackedAddress {
    const NONE: u64 = u64::MAX;

    pub(crate) const fn none() -> Self {
        Self(Self::NONE)
    }

    pub(crate) fn get(self) -> Option<u64> {
        (self.0 != Self::NONE).then_some(self.0)
    }

    pub(crate) fn set(&mut self, address: Option<u64>) {
        *self = Self::from(address);
    }
}

impl From<Option<u64>> for PackedAddress {
    fn from(address: Option<u64>) -> Self {
        match address {
            Some(address) => {
                assert_ne!(address, Self::NONE, "the last address is the empty one");
                Self(address)
            }
            None => Self::none(),
        }
    }
}

impl std::fmt::Debug for PackedAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.get().fmt(f)
    }
}

impl serde::Serialize for PackedAddress {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.get().serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for PackedAddress {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Option::<u64>::deserialize(deserializer).map(Self::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::LocalInsnId;

    #[test]
    fn a_link_round_trips_and_is_four_bytes() {
        assert_eq!(std::mem::size_of::<Link<LocalInsnId>>(), 4);
        let mut link: Link<LocalInsnId> = Link::none();
        assert_eq!(link.get(), None);
        link.set(Some(LocalInsnId::from(7usize)));
        assert_eq!(link.get(), Some(LocalInsnId::from(7usize)));
        assert_eq!(link.take(), Some(LocalInsnId::from(7usize)));
        assert_eq!(link.get(), None);
        assert_eq!(std::mem::size_of::<PackedAddress>(), 8);
        assert_eq!(PackedAddress::from(Some(0)).get(), Some(0));
        assert_eq!(PackedAddress::none().get(), None);
    }
}
