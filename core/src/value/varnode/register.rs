use crate::space::SpaceId;
use jstd::{Identifier, registry::Identified};
use serde::{Deserialize, Serialize};

#[derive(Identifier)]
pub struct RegisterId(usize);

/// A PCode register
#[derive(Serialize, Deserialize)]
pub struct Register {
    pub name: Box<str>,

    /// The name of the space this register is a part of
    pub space: SpaceId,

    /// The offset of this register in the space
    pub offset: usize,

    /// The size of this register in bytes
    pub size: usize,
}

pub type RegisterRef<'b> = Identified<RegisterId, &'b Register>;

pub type RegisterMutRef<'b> = Identified<RegisterId, &'b mut Register>;
