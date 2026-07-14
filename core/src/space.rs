//! Memory spaces: uniformly-addressed regions that varnodes live in.
//!
//! Every [`Varnode`](crate::value::Varnode) belongs to exactly one [`Space`].
//! Spaces model distinct address ranges — for example RAM, ROM, and the
//! processor register file are each their own space.
//! The basic unit for a space is a byte. This can be changed by setting the
//! space's *word size* (bytes per addressable unit)
//! and *address size* (bytes needed to hold a pointer into the space).
use std::fmt::Display;

use jstd::{Identifier, registry::Identified};
use serde::{Deserialize, Serialize};

use crate::value::{FunctionId, LocalTempSpaceId, TempSpaceId, util::base_ref::AsShared};

/// A stable, context-unique identifier for a [`Space`].
#[derive(Identifier)]
pub struct SpaceId(usize);

/// A memory-space handle stored inside one function body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LocalMemorySpaceId {
    Shared(SpaceId),
    Temp(LocalTempSpaceId),
}

impl LocalMemorySpaceId {
    pub const fn qualify(self, function: FunctionId) -> MemorySpaceId {
        match self {
            Self::Shared(id) => MemorySpaceId::Shared(id),
            Self::Temp(local) => MemorySpaceId::Temp(TempSpaceId::new(function, local)),
        }
    }

    /// Returns the shared space while producers still use only module storage.
    pub const fn shared(self) -> Option<SpaceId> {
        match self {
            Self::Shared(id) => Some(id),
            Self::Temp(_) => None,
        }
    }
}

impl From<SpaceId> for LocalMemorySpaceId {
    fn from(id: SpaceId) -> Self {
        Self::Shared(id)
    }
}

impl PartialEq<SpaceId> for LocalMemorySpaceId {
    fn eq(&self, other: &SpaceId) -> bool {
        self.shared() == Some(*other)
    }
}

impl PartialEq<LocalMemorySpaceId> for SpaceId {
    fn eq(&self, other: &LocalMemorySpaceId) -> bool {
        other == self
    }
}

/// A module/API memory-space handle. Local temporary spaces retain their owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemorySpaceId {
    Shared(SpaceId),
    Temp(TempSpaceId),
}

impl MemorySpaceId {
    pub fn localize(self, function: FunctionId) -> LocalMemorySpaceId {
        match self {
            Self::Shared(id) => LocalMemorySpaceId::Shared(id),
            Self::Temp(id) => LocalMemorySpaceId::Temp(id.localize(function)),
        }
    }

    pub const fn owning_function(self) -> Option<FunctionId> {
        match self {
            Self::Shared(_) => None,
            Self::Temp(id) => Some(id.func),
        }
    }

    pub const fn shared(self) -> Option<SpaceId> {
        match self {
            Self::Shared(id) => Some(id),
            Self::Temp(_) => None,
        }
    }
}

impl From<SpaceId> for MemorySpaceId {
    fn from(id: SpaceId) -> Self {
        Self::Shared(id)
    }
}

/// The const space is used for constant values such as immediate values
pub const SPACE_CONST: SpaceId = SpaceId(0);

/// The broad category of a memory space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpaceType {
    /// Readable and writable memory (e.g. heap, stack, data segments).
    Ram,
    /// Read-only memory (e.g. flash, ROM).
    Rom,
    /// Processor registers
    Register,
    /// Builder-created local storage for intermediate values
    Temporary,
}

pub type SpaceRef<'ctx> = Identified<SpaceId, &'ctx Space>;

/// A named, uniformly-addressed memory region.
///
/// Each space has a *word size* (bytes per addressable unit) and an *address
/// size* (bytes needed to hold a pointer into the space).  For most RAM spaces
/// these are 1 and 8 respectively on a 64-bit architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Space {
    /// Optional human-readable name (e.g. `"ram"`, `"register"`).
    pub name: Option<Box<str>>,

    /// The size of a memory location with a single address in this space, in bytes.
    pub word_size: usize,

    /// The size of addresses in this space, in bytes.
    pub addr_size: usize,

    /// The kind of this space.
    pub ty: SpaceType,
}

impl Space {
    /// Creates a new RAM space with the given name (or anonymous if `None`),
    /// word size, and address size.
    pub fn new(name: Option<&str>, word_size: usize, addr_size: usize) -> Self {
        Self {
            name: name.map(Box::from),
            word_size,
            addr_size,
            ty: SpaceType::Ram,
        }
    }

    /// Builds a space from an id. Accepts either a `&Context` or a bare `&Shared`
    /// (via [`AsShared`]).
    pub fn from_id<'ctx, 'str: 'ctx>(
        src: impl AsShared<'ctx, 'str>,
        id: SpaceId,
    ) -> SpaceRef<'ctx> {
        SpaceRef::new(id, &src.as_shared().spaces[id])
    }
}

impl Display for Space {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "{}", name)
        } else {
            write!(f, "space")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporary_memory_space_qualification_preserves_owner() {
        let function = FunctionId::from(3);
        let local = LocalTempSpaceId::from(4);
        let qualified = LocalMemorySpaceId::Temp(local).qualify(function);

        assert_eq!(
            qualified,
            MemorySpaceId::Temp(TempSpaceId::new(function, local))
        );
        assert_eq!(qualified.owning_function(), Some(function));
        assert_eq!(
            qualified.localize(function),
            LocalMemorySpaceId::Temp(local)
        );
        assert_eq!(
            MemorySpaceId::Shared(SpaceId::from(2)).owning_function(),
            None
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "TempSpaceId::localize: foreign id")]
    fn temporary_memory_space_rejects_foreign_localization() {
        MemorySpaceId::Temp(TempSpaceId::new(
            FunctionId::from(1),
            LocalTempSpaceId::from(0),
        ))
        .localize(FunctionId::from(2));
    }
}
