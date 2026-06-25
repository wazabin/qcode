//! Opaque compile-time byte blobs — constants wider than a [`Literal`] can hold.
//!
//! A numeric [`Literal`](crate::value::Literal) is a single `u64`; constants
//! that exceed 64 bits (SSE/AVX register pools, wide stack/memory reads, the
//! result of coalescing several adjacent constant stores) cannot be represented
//! that way without breaking the u64-centric folding pipeline. A [`Bytes`]
//! value sidesteps that: it is a raw, opaque byte vector with **no arithmetic
//! meaning**, stored as a little-endian, memory-order snapshot (`data[i]` is the
//! byte at `base + i`, matching the target's fixed little-endian layout).
//!
//! Every `Bytes` carries a [`TypeId`] — typically an `Array(i8, len)` — with the
//! invariant `size_of(type_id) == data.len()`, enforced at construction. Unlike
//! numeric literals, `Bytes` values are **not interned**: each construction
//! produces a fresh [`BytesId`], so downstream equality must compare contents,
//! never IDs.

use crate::{
    context::Context,
    types::TypeId,
    value::{
        Value, ValueId,
        util::base_ref::{BaseRef, WithCtx},
    },
};
use jstd::Identifier;

#[derive(Identifier)]
pub struct BytesId(usize);

/// A compile-time opaque byte blob stored in a [`Context`](crate::context::Context).
///
/// The bytes are held in little-endian, memory-order layout. See the module
/// docs for the rationale and invariants.
#[derive(Debug, Clone, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Bytes {
    /// Raw bytes in target memory order (`data[i]` = byte at `base + i`).
    pub data: Vec<u8>,
    /// The type of this constant; `size_of(type_id) == data.len()`.
    pub type_id: TypeId,
}

pub type BytesRef<'str, 'ctx> = BaseRef<&'ctx Context<'str>, BytesId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 'ctx, 'str> for BytesRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'ctx Context<'str> {
        self.ctx
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx, Ctx> BaseRef<Ctx, BytesId>
where
    Self: WithCtx<'s, 'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx Bytes {
        &self.ctx().values.bytes[self.id]
    }

    /// The raw bytes in target memory order.
    pub fn data(&'s self) -> &'ctx [u8] {
        &self.inner().data
    }

    /// Returns the [`TypeId`] of this blob.
    pub fn type_id(&'s self) -> TypeId {
        self.inner().type_id
    }
}

impl std::fmt::Display for BytesRef<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = &self.ctx.values.bytes[self.id].data;
        write!(f, "b\"")?;
        for &b in data {
            write!(f, "\\x{:02x}", b)?;
        }
        write!(f, "\"")
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for BytesRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        ValueId::Bytes(self.id)
    }

    fn size(&self) -> usize {
        self.ctx.values.bytes[self.id].data.len()
    }
}
