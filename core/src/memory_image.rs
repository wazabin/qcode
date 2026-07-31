//! A serializable snapshot of a binary's initialized memory.
//!
//! [`MemoryImage`] is the persistence/test form of the byte surface: it
//! implements [`binfmt::BinaryFormat`], so a snapshot (built from a live
//! format's `mapped_regions()` at save time) or a test-seeded image can be
//! `Arc`-wrapped and handed to the pipeline as `PipelineEnv.binary`, exactly
//! like a live ELF/PE handle. During a live lift the bytes stay in the loader's
//! format object only — nothing is copied into the `Context`.
//!
//! The motivating consumer is the jump-table pass, which reads table entries
//! straight out of `.rodata` through the shared handle.

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
    /// Whether the region is writable. A writable region's initialized bytes are
    /// not a reliable constant — the runtime (e.g. the dynamic linker populating
    /// the GOT) may overwrite them — so passes that resolve control flow or fold
    /// constants out of memory must not trust it. Defaults to `false` for images
    /// deserialized from before this field existed.
    #[serde(default)]
    writable: bool,
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
    /// Whether the per-segment `executable` flags are authoritative. Until the
    /// `memory_protections` pass establishes them, the lifter treats every mapped
    /// byte as potentially executable (default r/x); once known, it narrows to the
    /// real flags. See [`may_be_executable`](Self::may_be_executable).
    #[serde(default)]
    protections_known: bool,
}

impl MemoryImage {
    /// Add a mapped region. Segments are kept sorted by start address; callers
    /// need not insert in order.
    pub fn add_segment(&mut self, start: u64, bytes: Vec<u8>, executable: bool, writable: bool) {
        if bytes.is_empty() {
            return;
        }
        let seg = Segment {
            start,
            bytes,
            executable,
            writable,
        };
        let pos = self.segments.partition_point(|s| s.start < seg.start);
        self.segments.insert(pos, seg);
    }

    /// The segment containing `addr`, if any.
    fn segment_at(&self, addr: u64) -> Option<&Segment> {
        // Largest start <= addr, then a range check (segments are sorted and,
        // for a sane loader, non-overlapping).
        let pos = self.segments.partition_point(|s| s.start <= addr);
        self.segments
            .get(pos.checked_sub(1)?)
            .filter(|s| s.contains(addr))
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

    /// True if `addr` lies in an executable mapped region (per the raw segment
    /// flag, regardless of whether protections have been established).
    pub fn is_executable(&self, addr: u64) -> bool {
        self.segment_at(addr).is_some_and(|s| s.executable)
    }

    /// True only if `addr` is mapped in a region *known* to be writable: the
    /// per-segment protection flags must be established (see
    /// [`protections_known`](Self::protections_known)) and the containing segment
    /// writable. Before protections are known the segment flags are not
    /// authoritative, so this conservatively returns `false` ("not proven
    /// writable"). Callers use it to refuse to treat mutable memory (e.g. a GOT
    /// slot the dynamic linker rewrites) as a constant.
    pub fn is_known_writable(&self, addr: u64) -> bool {
        self.protections_known && self.segment_at(addr).is_some_and(|s| s.writable)
    }

    /// True if `addr` is mapped by any segment.
    pub fn contains(&self, addr: u64) -> bool {
        self.segment_at(addr).is_some()
    }

    /// The `[start, end)` bounds of the segment containing `addr`, if any.
    pub fn segment_bounds(&self, addr: u64) -> Option<(u64, u64)> {
        self.segment_at(addr).map(|s| (s.start, s.end()))
    }

    /// Whether the per-segment executable flags are authoritative (the
    /// `memory_protections` pass has run).
    pub fn protections_known(&self) -> bool {
        self.protections_known
    }

    /// Mark the per-segment protection flags as authoritative, so the lifter
    /// narrows from the permissive default to the real flags.
    pub fn mark_protections_known(&mut self) {
        self.protections_known = true;
    }

    /// True if no segments have been loaded yet. Used to make binary memory
    /// loading idempotent across fixpoint rounds.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

/// The persistence/test backing of the byte surface: a reloaded `.harbinger`
/// snapshot (or a `qcode!`-DSL test that seeded segments) wraps its
/// `MemoryImage` in an `Arc` and hands it to `PipelineEnv.binary`, so passes
/// read initialized memory through one trait regardless of whether a live
/// container format is behind it.
impl binfmt::BinaryFormat for MemoryImage {
    fn load_address(&self) -> u64 {
        self.segments.first().map(|s| s.start).unwrap_or(0)
    }

    fn byte_at(&self, addr: u64) -> Option<u8> {
        let seg = self.segment_at(addr)?;
        seg.bytes.get((addr - seg.start) as usize).copied()
    }

    fn bytes_at(&self, addr: u64) -> Option<&[u8]> {
        let seg = self.segment_at(addr)?;
        seg.bytes.get((addr - seg.start) as usize..)
    }

    /// An image records mapped bytes, not entry metadata.
    fn entry_points(&self) -> Vec<u64> {
        Vec::new()
    }

    /// Images do not record an architecture; x86-64 is the workspace default
    /// (mirrors [`Blob`]'s placeholder). Consumers of `PipelineEnv.binary`
    /// read bytes and permissions, never the architecture.
    ///
    /// [`Blob`]: binfmt::blob::Blob
    fn architecture(&self) -> binfmt::Arch {
        binfmt::Arch::X86_64
    }

    fn segment_bounds(&self, addr: u64) -> Option<(u64, u64)> {
        MemoryImage::segment_bounds(self, addr)
    }

    fn is_executable(&self, addr: u64) -> bool {
        MemoryImage::is_executable(self, addr)
    }

    /// Answers straight from the per-segment flag: an image is only ever built
    /// from an authoritative container format's `mapped_regions` (or a test's
    /// explicit `add_segment`), so the flags need no separate establishment
    /// step. (The inherent [`MemoryImage::is_known_writable`] keeps the legacy
    /// `protections_known` gate for its remaining callers.)
    fn is_known_writable(&self, addr: u64) -> bool {
        self.segment_at(addr).is_some_and(|s| s.writable)
    }

    /// The mirror of [`is_known_writable`](Self::is_known_writable): a mapped
    /// segment whose recorded flag says "not writable" is proven read-only. The
    /// flags come from the container format's `mapped_regions`, so this is only
    /// as authoritative as the format that filled them.
    ///
    /// [`is_known_writable`]: binfmt::BinaryFormat::is_known_writable
    fn is_known_read_only(&self, addr: u64) -> bool {
        self.segment_at(addr).is_some_and(|s| !s.writable)
    }

    fn mapped_regions(&self) -> Vec<(u64, Vec<u8>, bool, bool)> {
        self.segments
            .iter()
            .map(|s| (s.start, s.bytes.clone(), s.executable, s.writable))
            .collect()
    }

    fn read_bytes(&self, addr: u64, n: usize) -> Option<Vec<u8>> {
        MemoryImage::read_bytes(self, addr, n)
    }

    fn read_uint(&self, addr: u64, size: usize) -> Option<u64> {
        MemoryImage::read_uint(self, addr, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image() -> MemoryImage {
        let mut img = MemoryImage::default();
        // Insert out of order to exercise the sorted insert.
        img.add_segment(0x2000, vec![0xaa, 0xbb, 0xcc, 0xdd], true, false);
        img.add_segment(0x1000, vec![0x01, 0x02, 0x03, 0x04], false, false);
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

    #[test]
    fn protections_known_is_off_by_default() {
        let mut img = image();
        assert!(!img.protections_known());
        img.mark_protections_known();
        assert!(img.protections_known());
    }

    #[test]
    fn known_writable_requires_established_protections() {
        let mut img = MemoryImage::default();
        img.add_segment(0x1000, vec![0u8; 4], false, true); // writable data
        img.add_segment(0x2000, vec![0u8; 4], false, false); // read-only data

        // Until protections are established, writability is not authoritative, so
        // nothing is *known* writable.
        assert!(!img.is_known_writable(0x1000));
        assert!(!img.is_known_writable(0x2000));

        img.mark_protections_known();
        assert!(img.is_known_writable(0x1000), "writable segment now known");
        assert!(!img.is_known_writable(0x2000), "read-only stays read-only");
        assert!(!img.is_known_writable(0x9999), "unmapped is not writable");
    }
}
