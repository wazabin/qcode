//! Flat storage for the spaces a guest does not address.
//!
//! Register, unique and temporary spaces are small, dense and hit constantly:
//! every operand of every p-code operation is a read or a write here. The
//! emulator's general-purpose backing for them is `FxHashMap<u64, u8>`, which
//! costs *one hash lookup per byte* — reading `RAX` is eight lookups, and a
//! `Vec` allocation for the result.
//!
//! These spaces are nothing like guest memory: their addresses are assigned by
//! the specification, start at zero, and span a few hundred bytes. A plain
//! `Vec<u8>` indexed by address is the right shape, turning each access into a
//! bounds check and a copy.
//!
//! # This is not, by itself, faster
//!
//! Measured against the hash-map backing on a hot loop, this is neutral to
//! about 5% *slower*. Profiling says why: register access is roughly 4% of run
//! time, so removing per-byte hashing from it cannot matter much. The cost is
//! interpreter dispatch and allocation churn instead.
//!
//! It is kept because a JIT needs it. Compiled code has to reach the register
//! file by address, and it cannot call into a hash map per operand and stay
//! worth compiling. This is groundwork for that, not a speedup in its own
//! right.
//!
//! Guest RAM deliberately does *not* live here — it is sparse across a 64-bit
//! space and needs permissions, which is what [`Mmu`](crate::Mmu) is for.

use qcode::{
    context::Context,
    space::{MemorySpaceId, Space, SpaceId, SpaceType},
};
use qcode_emulator::EmulatorErrorKind;
use rustc_hash::FxHashMap;

/// Upper bound on how large a flat space may grow.
///
/// These spaces hold registers and lifter scratch, so a few kilobytes is
/// typical. The cap exists so that a malformed address cannot turn into an
/// enormous allocation; it is far above any real specification's needs.
const MAX_FLAT_SPACE: usize = 1 << 24;

/// One densely-addressed space.
#[derive(Debug, Default, Clone)]
pub struct FlatSpace {
    bytes: Vec<u8>,
    /// Which bytes have been written. Only consulted for spaces that are not
    /// zero-filled, where reading an unwritten byte is a lifter error worth
    /// reporting rather than a silent zero.
    written: Vec<bool>,
    /// Whether an unwritten byte reads as zero.
    ///
    /// Register space is architectural state that exists whether or not a
    /// harness seeded it; a unique is scratch that must be written before it is
    /// read, so reading one that was not is a defect.
    zero_filled: bool,
}

impl FlatSpace {
    fn new(zero_filled: bool) -> Self {
        Self {
            bytes: Vec::new(),
            written: Vec::new(),
            zero_filled,
        }
    }

    /// Grows the space so that `end` bytes are addressable.
    fn reserve_to(&mut self, end: usize) -> Result<(), EmulatorErrorKind> {
        if end > MAX_FLAT_SPACE {
            return Err(EmulatorErrorKind::AddressOverflow(end as u64, 0));
        }
        if self.bytes.len() < end {
            self.bytes.resize(end, 0);
            // Tracked only where it is consulted. For a zero-filled space an
            // unwritten byte legitimately reads as zero, so maintaining this
            // would be a second array touched on every write for nothing.
            if !self.zero_filled {
                self.written.resize(end, false);
            }
        }
        Ok(())
    }

    /// The byte range an access covers, or `None` if it would overflow.
    fn range(addr: u64, size: usize) -> Option<(usize, usize)> {
        let start = usize::try_from(addr).ok()?;
        let end = start.checked_add(size)?;
        (end <= MAX_FLAT_SPACE).then_some((start, end))
    }

    /// Grows the space so `len` bytes are addressable, and returns the base
    /// pointer of its storage.
    ///
    /// For compiled code, which addresses this storage directly rather than
    /// through the accessors. The caller must not cause the space to grow while
    /// holding the pointer — growing may reallocate — which is why the required
    /// size is requested up front.
    pub fn base_ptr(&mut self, len: usize) -> Result<*mut u8, EmulatorErrorKind> {
        self.reserve_to(len)?;
        Ok(self.bytes.as_mut_ptr())
    }

    pub fn read_bytes(&self, addr: u64, size: usize) -> Result<Vec<u8>, EmulatorErrorKind> {
        let Some((start, end)) = Self::range(addr, size) else {
            return Err(EmulatorErrorKind::AddressOverflow(addr, size));
        };
        // Past the end of what has been written: zero-filled spaces read zero,
        // others report the first missing byte.
        if end > self.bytes.len() {
            if !self.zero_filled {
                return Err(EmulatorErrorKind::MemoryReadError(
                    self.bytes.len().max(start) as u64,
                ));
            }
            let mut out = vec![0; size];
            let available = self.bytes.len().saturating_sub(start);
            if available > 0 {
                out[..available].copy_from_slice(&self.bytes[start..self.bytes.len()]);
            }
            return Ok(out);
        }
        if !self.zero_filled
            && let Some(offset) = self.written[start..end].iter().position(|written| !written)
        {
            return Err(EmulatorErrorKind::MemoryReadError((start + offset) as u64));
        }
        Ok(self.bytes[start..end].to_vec())
    }

    /// Reads a little-endian unsigned integer, matching the emulator's value
    /// domain (which is at most 16 bytes wide).
    pub fn read_u128(&self, addr: u64, size: usize) -> Result<u128, EmulatorErrorKind> {
        let width = size.min(16);
        let Some((start, end)) = Self::range(addr, width) else {
            return Err(EmulatorErrorKind::AddressOverflow(addr, width));
        };
        if end > self.bytes.len()
            || (!self.zero_filled && self.written[start..end].contains(&false))
        {
            // Fall back to the checked path, which reports the exact byte.
            let bytes = self.read_bytes(addr, width)?;
            let mut bits = 0u128;
            for (index, byte) in bytes.iter().enumerate() {
                bits |= u128::from(*byte) << (index * 8);
            }
            return Ok(bits);
        }
        let mut bits = 0u128;
        for (index, byte) in self.bytes[start..end].iter().enumerate() {
            bits |= u128::from(*byte) << (index * 8);
        }
        Ok(bits)
    }

    pub fn write_bytes(&mut self, addr: u64, bytes: &[u8]) -> Result<(), EmulatorErrorKind> {
        let Some((start, end)) = Self::range(addr, bytes.len()) else {
            return Err(EmulatorErrorKind::AddressOverflow(addr, bytes.len()));
        };
        self.reserve_to(end)?;
        self.bytes[start..end].copy_from_slice(bytes);
        if !self.zero_filled {
            self.written[start..end].fill(true);
        }
        Ok(())
    }

    /// Writes the low `size` bytes of a little-endian integer.
    pub fn write_u128(
        &mut self,
        addr: u64,
        size: usize,
        bits: u128,
    ) -> Result<(), EmulatorErrorKind> {
        let width = size.min(16);
        let Some((start, end)) = Self::range(addr, width) else {
            return Err(EmulatorErrorKind::AddressOverflow(addr, width));
        };
        self.reserve_to(end)?;
        for index in 0..width {
            self.bytes[start + index] = (bits >> (index * 8)) as u8;
        }
        if !self.zero_filled {
            self.written[start..end].fill(true);
        }
        Ok(())
    }
}

/// The set of flat spaces for a module.
#[derive(Debug, Default, Clone)]
pub struct FlatSpaces {
    /// Storage, appended to as spaces are first touched.
    ///
    /// Held in a `Vec` rather than keyed directly by id so that an index is
    /// *stable*: a caller that resolves a space once can address it forever
    /// without hashing again. Compiled code re-enters through here on every
    /// block execution, and two map lookups per space per entry was a
    /// measurable share of the JIT's run time.
    spaces: Vec<FlatSpace>,
    /// Where each space's storage lives in `spaces`. Append-only.
    slots: FxHashMap<MemorySpaceId, usize>,
    /// Shared spaces whose unwritten bytes read as zero, learned from the
    /// context. Rebuilt only when the module gains spaces.
    zero_filled: FxHashMap<SpaceId, bool>,
    configured_space_count: Option<usize>,
}

impl FlatSpaces {
    /// Learns which spaces are zero-filled. Cheap to call repeatedly: spaces are
    /// append-only, so an unchanged count means nothing to redo.
    pub fn configure(&mut self, ctx: &Context<'_>) {
        let count = ctx.space_count();
        if self.configured_space_count == Some(count) {
            return;
        }
        self.zero_filled.clear();
        for index in 0..count {
            let id = SpaceId::from(index);
            let space = Space::from_id(ctx, id);
            // Register space is architectural state, and x86's private x87 file
            // is too — FXSAVE can read a slot before a harness seeds it.
            let zero =
                matches!(space.ty, SpaceType::Register) || space.name.as_deref() == Some("x87");
            self.zero_filled.insert(id, zero);
        }
        self.configured_space_count = Some(count);
    }

    /// Puts back the contents captured in `snapshot`, keeping this table's
    /// shape.
    ///
    /// Slots are promised stable for the life of these spaces, and a
    /// compiled block holds the ones it resolved. A snapshot taken before a
    /// space was first touched has fewer slots, so the table is not replaced
    /// wholesale: each space the snapshot holds is copied over, and any space
    /// that has appeared since is reset to untouched.
    pub fn restore(&mut self, snapshot: &FlatSpaces) {
        debug_assert!(
            snapshot
                .slots
                .iter()
                .all(|(space, &slot)| self.slots.get(space) == Some(&slot)),
            "slots are append-only, so a snapshot's are a prefix of the live table's"
        );
        for (index, space) in self.spaces.iter_mut().enumerate() {
            match snapshot.spaces.get(index) {
                Some(saved) => {
                    space.bytes.clone_from(&saved.bytes);
                    space.written.clone_from(&saved.written);
                }
                None => {
                    space.bytes.fill(0);
                    space.written.fill(false);
                }
            }
        }
    }

    /// The stable index of `space`'s storage, creating it on first use.
    ///
    /// Resolve this once and address the space with [`base_ptr_at`](Self::base_ptr_at)
    /// thereafter; the index stays valid for the life of these spaces.
    pub fn slot(&mut self, space: MemorySpaceId) -> usize {
        if let Some(&slot) = self.slots.get(&space) {
            return slot;
        }
        let zero_filled = self.is_zero_filled(space);
        self.spaces.push(FlatSpace::new(zero_filled));
        let slot = self.spaces.len() - 1;
        self.slots.insert(space, slot);
        slot
    }

    /// The base pointer of the storage at `slot`, grown to hold `len` bytes.
    ///
    /// Panics if `slot` did not come from [`slot`](Self::slot) on these spaces.
    pub fn base_ptr_at(&mut self, slot: usize, len: usize) -> Result<*mut u8, EmulatorErrorKind> {
        self.spaces[slot].base_ptr(len)
    }

    fn is_zero_filled(&self, space: MemorySpaceId) -> bool {
        match space {
            // Lifter scratch is created per function and read after writing.
            MemorySpaceId::Temp(_) => true,
            MemorySpaceId::Shared(id) => self.zero_filled.get(&id).copied().unwrap_or(false),
        }
    }

    /// The base pointer of `space`'s storage, grown to hold `len` bytes.
    pub fn base_ptr(
        &mut self,
        space: MemorySpaceId,
        len: usize,
    ) -> Result<*mut u8, EmulatorErrorKind> {
        self.entry(space).base_ptr(len)
    }

    pub fn get(&self, space: MemorySpaceId) -> Option<&FlatSpace> {
        self.slots.get(&space).map(|&slot| &self.spaces[slot])
    }

    /// The space's storage, created on first use.
    pub fn entry(&mut self, space: MemorySpaceId) -> &mut FlatSpace {
        let slot = self.slot(space);
        &mut self.spaces[slot]
    }

    pub fn read_u128(
        &self,
        space: MemorySpaceId,
        addr: u64,
        size: usize,
    ) -> Result<u128, EmulatorErrorKind> {
        match self.get(space) {
            Some(flat) => flat.read_u128(addr, size),
            None if self.is_zero_filled(space) => Ok(0),
            None => Err(EmulatorErrorKind::UnknownSpace(space)),
        }
    }

    pub fn read_bytes(
        &self,
        space: MemorySpaceId,
        addr: u64,
        size: usize,
    ) -> Result<Vec<u8>, EmulatorErrorKind> {
        match self.get(space) {
            Some(flat) => flat.read_bytes(addr, size),
            None if self.is_zero_filled(space) => Ok(vec![0; size]),
            None => Err(EmulatorErrorKind::UnknownSpace(space)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_little_endian_value() {
        let mut flat = FlatSpace::new(true);
        flat.write_u128(8, 4, 0xdead_beef).unwrap();
        assert_eq!(flat.read_u128(8, 4).unwrap(), 0xdead_beef);
        assert_eq!(flat.read_bytes(8, 4).unwrap(), vec![0xef, 0xbe, 0xad, 0xde]);
    }

    #[test]
    fn a_zero_filled_space_reads_unwritten_bytes_as_zero() {
        let flat = FlatSpace::new(true);
        assert_eq!(flat.read_u128(0, 8).unwrap(), 0);
        assert_eq!(flat.read_bytes(0, 4).unwrap(), vec![0; 4]);
    }

    #[test]
    fn a_scratch_space_reports_an_unwritten_read() {
        // Reading a unique that was never written is a lifter defect, and the
        // flat store must keep reporting it rather than inventing a zero.
        let mut flat = FlatSpace::new(false);
        flat.write_bytes(0, &[1, 2]).unwrap();
        assert!(flat.read_bytes(0, 2).is_ok());
        assert!(matches!(
            flat.read_bytes(0, 4),
            Err(EmulatorErrorKind::MemoryReadError(2))
        ));
    }

    #[test]
    fn a_partially_written_scratch_read_names_the_missing_byte() {
        let mut flat = FlatSpace::new(false);
        flat.write_bytes(0, &[0; 8]).unwrap();
        let mut flat2 = FlatSpace::new(false);
        flat2.write_bytes(4, &[1, 2, 3, 4]).unwrap();
        // Bytes 0..4 were never written even though the space is long enough.
        assert!(matches!(
            flat2.read_bytes(0, 8),
            Err(EmulatorErrorKind::MemoryReadError(0))
        ));
        assert!(flat.read_bytes(0, 8).is_ok());
    }

    #[test]
    fn writes_grow_the_space_and_preserve_neighbours() {
        let mut flat = FlatSpace::new(true);
        flat.write_bytes(0, &[9; 4]).unwrap();
        flat.write_bytes(64, &[7; 4]).unwrap();
        assert_eq!(flat.read_bytes(0, 4).unwrap(), vec![9; 4]);
        assert_eq!(flat.read_bytes(64, 4).unwrap(), vec![7; 4]);
        assert_eq!(flat.read_bytes(32, 4).unwrap(), vec![0; 4]);
    }

    #[test]
    fn an_absurd_address_is_refused_rather_than_allocated() {
        let mut flat = FlatSpace::new(true);
        assert!(matches!(
            flat.write_bytes(u64::MAX - 8, &[1; 4]),
            Err(EmulatorErrorKind::AddressOverflow(..))
        ));
    }

    #[test]
    fn a_wide_value_is_truncated_to_the_domain_width() {
        let mut flat = FlatSpace::new(true);
        flat.write_u128(0, 32, u128::MAX).unwrap();
        // Only the domain's 16 bytes are stored.
        assert_eq!(flat.read_u128(0, 16).unwrap(), u128::MAX);
        assert_eq!(flat.read_bytes(16, 4).unwrap(), vec![0; 4]);
    }
}
