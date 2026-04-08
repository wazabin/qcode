use crate::space::SpaceId;
use jstd::{Identifier, registry::Identified};

#[derive(Identifier)]
pub struct RegisterId(usize);

/// A PCode register
pub struct Register<'a> {
    pub name: &'a str,

    /// The name of the space this register is a part of
    pub space: SpaceId,

    /// The offset of this register in the space
    pub offset: usize,

    /// The size of this register in bytes
    pub size: usize,
}

pub type RegisterRef<'a, 'b> = Identified<RegisterId, &'b Register<'a>>;

pub type RegisterMutRef<'a, 'b> = Identified<RegisterId, &'b mut Register<'a>>;
