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

/// The encoding under which a byte blob was successfully read as text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StringEncoding {
    /// Printable 7-bit ASCII (one byte per character).
    Ascii,
    /// Printable UTF-16, little-endian (two bytes per code unit).
    Utf16Le,
}

impl StringEncoding {
    /// Short human-readable label (e.g. for a UI column).
    pub fn label(self) -> &'static str {
        match self {
            StringEncoding::Ascii => "ascii",
            StringEncoding::Utf16Le => "utf16le",
        }
    }
}

/// Attempt to decode `data` as a printable ASCII or UTF-16LE string.
///
/// Both encodings tolerate a single trailing NUL terminator (the common C /
/// Windows-`W` convention). Returns the decoded text and the encoding it was
/// read under, but only when every character is printable; otherwise the blob
/// has no clean string reading and the caller should fall back to the `\xNN`
/// hex form.
pub fn decode_string(data: &[u8]) -> Option<(StringEncoding, String)> {
    if data.is_empty() {
        return None;
    }

    // ASCII, optionally NUL-terminated.
    let ascii = data.strip_suffix(&[0]).unwrap_or(data);
    if !ascii.is_empty() && ascii.iter().all(|&b| b.is_ascii_graphic() || b == b' ') {
        return Some((
            StringEncoding::Ascii,
            ascii.iter().map(|&b| b as char).collect(),
        ));
    }

    // UTF-16LE, optionally NUL-terminated.
    if data.len() >= 2 && data.len() % 2 == 0 {
        let units: Vec<u16> = data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let units = units.strip_suffix(&[0]).unwrap_or(&units);
        if !units.is_empty()
            && let Ok(s) = String::from_utf16(units)
            && s.chars().all(|c| !c.is_control())
        {
            return Some((StringEncoding::Utf16Le, s));
        }
    }

    None
}

/// Escape a decoded string for display inside `b"..."` quotes.
pub fn escape_decoded(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

impl std::fmt::Display for BytesRef<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = &self.ctx.values.bytes[self.id].data;
        if let Some((_, s)) = decode_string(data) {
            return write!(f, "b\"{}\"", escape_decoded(&s));
        }
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
