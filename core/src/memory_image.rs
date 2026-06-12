//! A serializable snapshot of a binary's initialized memory.
//!
//! Analysis passes only ever receive a [`Context`](crate::context::Context); the
//! rich loader-side `BinaryFormat` (ELF/PE parsers) lives a crate up and cannot
//! be persisted. [`MemoryImage`] is the small, `Serialize`-able subset of that
//! memory-read surface that a pass needs: "give me the bytes / a little-endian
//! integer at a virtual address, and tell me whether that address is
//! executable". The lifter copies the binary's mapped regions into it once, and
//! it then rides inside the serialized `Context` (so a saved session is
//! self-describing).
//!
//! The motivating consumer is the jump-table pass
//! ([`crate::context::Context::read_uint`] reads table entries straight out of
//! `.rodata`).

/// One contiguous mapped region of the binary image.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct Segment {
    /// Virtual address of the first byte.
    start: u64,
    /// The mapped bytes (file contents plus any zero-fill the loader materialized).
    bytes: Vec<u8>,
    /// Whether the region is executable (so resolved jump targets can be sanity
    /// checked against code memory).
    executable: bool,
}

impl Segment {
    /// Inclusive-exclusive end of the region.
    fn end(&self) -> u64 {
        self.start + self.bytes.len() as u64
    }

    /// True if `addr` falls within `[start, end)`.
    fn contains(&self, addr: u64) -> bool {
        addr >= self.start && addr < self.end()
    }
}

/// The initialized memory of a loaded binary, as a set of mapped segments kept
/// sorted by start address.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryImage {
    segments: Vec<Segment>,
}

impl MemoryImage {
    /// Add a mapped region. Segments are kept sorted by start address; callers
    /// need not insert in order.
    pub fn add_segment(&mut self, start: u64, bytes: Vec<u8>, executable: bool) {
        if bytes.is_empty() {
            return;
        }
        let seg = Segment {
            start,
            bytes,
            executable,
        };
        let pos = self
            .segments
            .partition_point(|s| s.start < seg.start);
        self.segments.insert(pos, seg);
    }

    /// The segment containing `addr`, if any.
    fn segment_at(&self, addr: u64) -> Option<&Segment> {
        // Largest start <= addr, then a range check (segments are sorted and,
        // for a sane loader, non-overlapping).
        let pos = self.segments.partition_point(|s| s.start <= addr);
        self.segments.get(pos.checked_sub(1)?).filter(|s| s.contains(addr))
    }

    /// Read `n` bytes starting at `addr`, all from a single mapped segment.
    /// Returns `None` if any byte in `[addr, addr + n)` is unmapped.
    pub fn read_bytes(&self, addr: u64, n: usize) -> Option<Vec<u8>> {
        let seg = self.segment_at(addr)?;
        let off = (addr - seg.start) as usize;
        let end = off.checked_add(n)?;
        seg.bytes.get(off..end).map(|s| s.to_vec())
    }

    /// Read a little-endian unsigned integer of `size` bytes (1..=8) at `addr`.
    ///
    /// Endianness is fixed to little-endian for now (x86/x64); a big-endian /
    /// arch-driven variant is a TODO.
    pub fn read_uint(&self, addr: u64, size: usize) -> Option<u64> {
        if size == 0 || size > 8 {
            return None;
        }
        let bytes = self.read_bytes(addr, size)?;
        let mut value = 0u64;
        for (i, &b) in bytes.iter().enumerate() {
            value |= (b as u64) << (i * 8);
        }
        Some(value)
    }

    /// True if `addr` lies in an executable mapped region.
    pub fn is_executable(&self, addr: u64) -> bool {
        self.segment_at(addr).is_some_and(|s| s.executable)
    }

    /// True if `addr` is mapped by any segment.
    pub fn contains(&self, addr: u64) -> bool {
        self.segment_at(addr).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image() -> MemoryImage {
        let mut img = MemoryImage::default();
        // Insert out of order to exercise the sorted insert.
        img.add_segment(0x2000, vec![0xaa, 0xbb, 0xcc, 0xdd], true);
        img.add_segment(0x1000, vec![0x01, 0x02, 0x03, 0x04], false);
        img
    }

    #[test]
    fn read_uint_little_endian() {
        let img = image();
        assert_eq!(img.read_uint(0x1000, 4), Some(0x04030201));
        assert_eq!(img.read_uint(0x1000, 2), Some(0x0201));
        assert_eq!(img.read_uint(0x1001, 1), Some(0x02));
    }

    #[test]
    fn read_bytes_within_segment() {
        let img = image();
        assert_eq!(img.read_bytes(0x2001, 2), Some(vec![0xbb, 0xcc]));
    }

    #[test]
    fn unmapped_and_straddling_are_none() {
        let img = image();
        // Fully unmapped.
        assert_eq!(img.read_uint(0x500, 1), None);
        // Runs off the end of the segment.
        assert_eq!(img.read_bytes(0x1003, 2), None);
        // Gap between the two segments.
        assert_eq!(img.read_uint(0x1004, 1), None);
    }

    #[test]
    fn executability() {
        let img = image();
        assert!(img.is_executable(0x2002));
        assert!(!img.is_executable(0x1002));
        assert!(!img.is_executable(0x9999));
    }
}
