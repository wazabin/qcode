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

use jstd::Identifier;

use crate::{
    context::{Context, Shared},
    error::Result,
    space::{Space, SpaceId, SpaceRef},
    value::{
        Value, ValueId,
        util::{
            base_ref::{BaseRef, WithCtx, WithShared},
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Varnode<'str> {
    name: Option<Cow<'str, str>>,

    /// An integer label for a generated temporary, used to derive a display name
    /// (`v{label}`) lazily without allocating a `String` or touching the
    /// context's name map. Only set when `name` is `None`.
    label: Option<u32>,

    /// The address of this varnode, in the space it belongs to.
    address: i64,

    /// The size of this varnode in bytes.
    size: usize,

    /// The space this varnode belongs to.
    space: SpaceId,
}

impl<'str> Varnode<'str> {
    /// Returns the size of this varnode in bytes, without requiring a context reference.
    pub fn size_bytes(&self) -> usize {
        self.size
    }

    fn new(base: i64, size: usize, space: SpaceId) -> Self {
        Self {
            name: None,
            label: None,
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
        let id = ctx
            .shared
            .values
            .varnodes
            .push(Varnode::new(base, size, space));
        VarnodeMutRef::from_id(ctx, id)
    }

    /// Retrieves an existing varnode by its ID and returns an immutable reference
    /// to it. Accepts either a `&Context` or a bare `&Shared` (via
    /// [`AsShared`](crate::value::util::base_ref::AsShared)).
    pub fn from_id<'ctx>(
        src: impl crate::value::util::base_ref::AsShared<'ctx, 'str>,
        id: VarnodeId,
    ) -> VarnodeRef<'str, 'ctx> {
        VarnodeRef::from_id(src, id)
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
    Self: WithShared<'s, 'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx Varnode<'str> {
        &self.shared().values.varnodes[self.id]
    }

    fn fmt(&'s self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = self.name() {
            write!(f, "{}", name)
        } else if let Some(label) = self.inner().label {
            // Generated temporary: derive its name lazily, no allocation.
            write!(f, "v{label}")
        } else {
            write!(f, "[{}]:{} {}", *self.space(), self.size(), self.address())
        }
    }

    /// The space this varnode belongs to.
    pub fn space(&'s self) -> SpaceRef<'ctx> {
        Space::from_id(self.shared(), self.inner().space)
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

    /// The integer label of a generated temporary, if any. Temporaries derive
    /// their display name (`v{label}`) from this without an allocation.
    pub fn label(&'s self) -> Option<u32> {
        self.inner().label
    }
}

pub type VarnodeRef<'str, 'ctx> = BaseRef<&'ctx Shared<'str>, VarnodeId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithShared<'s, 'ctx, 'str> for VarnodeRef<'str, 'ctx> {
    fn shared(&'s self) -> &'ctx Shared<'str> {
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
        &mut self.ctx.shared.values.varnodes[self.id]
    }

    /// Sets the integer label used to derive a generated temporary's display
    /// name. Unlike [`Renameable::rename`], this neither allocates nor touches
    /// the context name map, so it stays off the per-instruction hot path.
    pub fn set_label(&mut self, label: u32) {
        self.inner_mut().label = Some(label);
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 's, 'str> for VarnodeMutRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'s Context<'str> {
        self.ctx
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithShared<'s, 's, 'str> for VarnodeMutRef<'str, 'ctx> {
    fn shared(&'s self) -> &'s Shared<'str> {
        &self.ctx.shared
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
    fn rename(&mut self, name: Cow<'str, str>) -> Result<()> {
        let id = self.id.into();
        let old_name = self.inner_mut().name.take();
        update_context_name(id, self.ctx, name.clone(), old_name.as_deref())?;
        self.ctx.shared.values.varnodes[self.id].name = Some(name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use crate::context::Context;

    #[test]
    fn varnode_name() {
        let mut ctx = Context::new();
        qcode!(ctx, "<block> varnode i64 ptr; goto <0x1001>;");

        let varnode = Varnode::from_id(&ctx, ptr);
        assert_eq!(varnode.name(), Some("ptr"));
        assert_eq!(varnode.space().name.as_deref(), Some("ptr"));
    }
}
