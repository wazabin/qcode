//! Memory spaces: uniformly-addressed regions that varnodes live in.
//!
//! Every [`Varnode`](crate::value::Varnode) belongs to exactly one [`Space`].
//! The space vocabulary itself lives in `pcode-types`, shared with the SLEIGH
//! decoder; this module re-exports it and adds the qcode-specific handles that
//! distinguish shared spaces from per-function temporaries.

use serde::{Deserialize, Serialize};

pub use pcode_types::space::{SPACE_CONST, Space, SpaceId, SpaceRef, SpaceStore, SpaceType};

use crate::value::{FunctionId, LocalTempSpaceId, TempSpaceId};

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
