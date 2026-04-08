//! Varnodes: named, typed memory locations used as IR operands.
//!
//! A [`Varnode`] represents a specific location in a [`Space`] — it has an
//! address, a size in bytes, and belongs to exactly one space.
//! Varnodes are used to model processor registers, global variables, and
//! other named memory locations.
//! Unlike [`Instruction`](crate::value::Instruction) results, they are not
//! directly SSA form: multiple instructions can read or write the same
//! varnode's *memory location*.
//! The Varnode's value however is a constant.
//!
//! [`Space`]: crate::space::Space

use std::borrow::Cow;

use jstd::{Identifier, registry::Identified};

use crate::{
    error::Result,
    context::Context,
    space::{Space, SpaceId},
    value::{
        Value, ValueId,
        util::{
            base_ref::{BaseRef, WithCtx},
            named::{Named, Renameable, update_context_name},
        },
    },
};

pub mod register;

#[derive(Identifier)]
pub struct VarnodeId(usize);

/// A named, typed reference to a specific location in a memory [`Space`].
///
/// A varnode is identified by its `(space, address, size)` triple. Registers
/// are modelled as varnodes in the register space; memory operands are varnodes
/// in the RAM space.
///
/// [`Space`]: crate::space::Space
#[derive(Debug, Clone)]
pub struct Varnode<'str> {
    name: Option<Cow<'str, str>>,

    /// The address of this varnode, in the space it belongs to.
    address: i64,

    /// The size of this varnode in bytes.
    size: usize,

    /// The space this varnode belongs to.
    space: SpaceId,
}

impl<'str> Varnode<'str> {
    fn new(base: i64, size: usize, space: SpaceId) -> Self {
        Self {
            name: None,
            address: base,
            size,
            space,
        }
    }

    /// Creates a new varnode in the context and returns a mutable reference to it.
    pub fn make<'ctx>(
        ctx: &'ctx mut Context<'str>,
        base: i64,
        size: usize,
        space: SpaceId,
    ) -> VarnodeMutRef<'str, 'ctx> {
        let id = ctx.values.varnodes.push(Varnode::new(base, size, space));
        VarnodeMutRef::from_id(ctx, id)
    }

    /// Retrieves an existing varnode from the context by its ID and returns an immutable reference to it.
    pub fn from_id<'ctx>(ctx: &'ctx Context<'str>, id: VarnodeId) -> VarnodeRef<'str, 'ctx> {
        VarnodeRef::from_id(ctx, id)
    }

    /// Retrieves an existing varnode from the context by its ID and returns a mutable reference to it.
    pub fn from_id_mut<'ctx>(
        ctx: &'ctx mut Context<'str>,
        id: VarnodeId,
    ) -> VarnodeMutRef<'str, 'ctx> {
        VarnodeMutRef::from_id(ctx, id)
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx, Ctx> BaseRef<Ctx, VarnodeId>
where
    Self: WithCtx<'s, 'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx Varnode<'str> {
        &self.ctx().values.varnodes[self.id]
    }

    fn fmt(&'s self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = self.name() {
            write!(f, "{}", name)
        } else {
            write!(f, "[{}]:{} {}", *self.space(), self.size(), self.address())
        }
    }

    /// The space this varnode belongs to.
    pub fn space(&'s self) -> Identified<SpaceId, &'ctx Space<'str>> {
        let space_id = self.inner().space;
        Identified::new(space_id, self.ctx().get_space(space_id))
    }

    /// The address at which this varnode begins
    pub fn address(&'s self) -> i64 {
        self.inner().address
    }

    /// The number of bytes in this varnode's range
    pub fn size(&'s self) -> usize {
        self.inner().size
    }

    /// An optional name for this varnode, used for debugging and display purposes.
    /// The name of a varnode is guaranteed to be unique within the context,
    /// and renaming a varnode will update the context's name registry to maintain this invariant.
    pub fn name(&'s self) -> Option<&'ctx str> {
        self.inner().name.as_deref()
    }
}

pub type VarnodeRef<'str, 'ctx> = BaseRef<&'ctx Context<'str>, VarnodeId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 'ctx, 'str> for VarnodeRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'ctx Context<'str> {
        self.ctx
    }
}

impl Named for VarnodeRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        self.name()
    }
}

impl std::fmt::Display for VarnodeRef<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt(f)
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for VarnodeRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        self.size()
    }
}

pub type VarnodeMutRef<'str, 'ctx> = BaseRef<&'ctx mut Context<'str>, VarnodeId>;

impl<'str, 'ctx> VarnodeMutRef<'str, 'ctx> {
    fn inner_mut(&mut self) -> &mut Varnode<'str> {
        &mut self.ctx.values.varnodes[self.id]
    }

    fn rename(&mut self, name: Cow<'str, str>) -> Result<'str, ()> {
        let id = self.id.into();
        let old_name = self.inner_mut().name.take();
        update_context_name(id, self.ctx, name.clone(), old_name.as_deref())?;
        self.ctx.values.varnodes[self.id].name = Some(name);
        Ok(())
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 's, 'str> for VarnodeMutRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'s Context<'str> {
        self.ctx
    }
}

impl Named for VarnodeMutRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        self.name()
    }
}

impl std::fmt::Display for VarnodeMutRef<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt(f)
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for VarnodeMutRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        self.size()
    }
}

impl<'str, 'ctx> Renameable<'str, 'ctx> for VarnodeMutRef<'str, 'ctx> {
    fn rename(&mut self, name: Cow<'str, str>) -> Result<'str, ()> {
        self.rename(name)
    }
}
