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

use crate::context::Context;

/// A stable, context-unique identifier for a [`Space`].
#[derive(Identifier)]
pub struct SpaceId(usize);

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

    /// Builds a space from an id
    pub fn from_id<'ctx, 'str>(ctx: &'ctx Context<'str>, id: SpaceId) -> SpaceRef<'ctx> {
        SpaceRef::new(id, &ctx.shared.spaces[id])
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
