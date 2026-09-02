//! Software MMU: page-granular mapping with per-byte permissions.
//!
//! This is the process-memory model the QCode emulator's flat
//! [`EmulatedSpace`](qcode_emulator::EmulatedSpace) map cannot express. A flat
//! `addr -> byte` map answers "what is here", but a VM needs to answer "may this
//! access happen at all", and to answer it *without aborting the run* — an
//! unmapped read is a fault the guest may legitimately take, not an emulator
//! bug.
//!
//! Permissions are tracked per byte rather than per page. The cost is one extra
//! byte of bookkeeping per guest byte; the benefit is that sub-page granularity
//! (a redzone between two heap chunks, a partially initialized stack frame)
//! costs nothing extra to express, which is the whole point of a
//! fuzzing-oriented memory model.
//!
//! Only the RAM space is mapped through here. Register, unique and temporary
//! spaces keep their flat representation: permission-checking a varnode write to
//! `RAX` is meaningless, and putting it on this path would add a page lookup to
//! the hottest operation in the interpreter.

use rustc_hash::FxHashMap;

/// Guest page size. Chosen to match the x86-64 base page so that guest `mmap`
/// granularity and MMU granularity agree; nothing here depends on the value.
pub const PAGE_SIZE: u64 = 0x1000;
const PAGE_MASK: u64 = PAGE_SIZE - 1;

/// A permission bitset, one per guest byte.
pub type Perm = u8;

/// Permission bits. `MAP` is what distinguishes "mapped with no access" from
/// "not mapped at all" — the two produce different faults, and a guest can
/// observe the difference (`mprotect(PROT_NONE)` succeeds on mapped memory and
/// fails on unmapped memory).
pub mod perm {
    use super::Perm;

    pub const NONE: Perm = 0;
    /// The byte holds a defined value. Cleared bytes read as a fault under
    /// [`Mmu::check_uninit`](super::Mmu::check_uninit), which is how
    /// use-of-uninitialized-memory is caught.
    pub const INIT: Perm = 1 << 0;
    pub const READ: Perm = 1 << 1;
    pub const WRITE: Perm = 1 << 2;
    pub const EXEC: Perm = 1 << 3;
    /// The byte belongs to a mapped region.
    pub const MAP: Perm = 1 << 4;
    pub const READ_WATCH: Perm = 1 << 5;
    pub const WRITE_WATCH: Perm = 1 << 6;

    pub const READ_WRITE: Perm = READ | WRITE;
    /// Conventional permissions for a freshly mapped, zero-filled region.
    pub const RW_INIT: Perm = MAP | READ | WRITE | INIT;
    /// Conventional permissions for loaded code.
    pub const RX_INIT: Perm = MAP | READ | EXEC | INIT;
}

/// Why an access could not be performed.
///
/// These are *values*, not errors in the "the emulator broke" sense: the VM
/// turns them into a guest-visible exit so a harness can map a fault to a signal,
/// a fuzzing crash, or a page-fault handler, and resume if it wants to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultKind {
    ReadUnmapped,
    ReadPerm,
    ReadUninit,
    WriteUnmapped,
    WritePerm,
    ExecUnmapped,
    ExecViolation,
    ReadWatch,
    WriteWatch,
    /// The access range wrapped past the end of the address space.
    AddressOverflow,
}

/// A failed access, with the exact byte that failed rather than the start of the
/// access. A 8-byte read straddling a mapping boundary faults at the boundary,
/// and that address is what a guest fault handler would see in `CR2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemFault {
    pub kind: FaultKind,
    pub addr: u64,
}

impl MemFault {
    fn new(kind: FaultKind, addr: u64) -> Self {
        Self { kind, addr }
    }
}

impl std::fmt::Display for MemFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = match self.kind {
            FaultKind::ReadUnmapped => "read of unmapped memory",
            FaultKind::ReadPerm => "read of unreadable memory",
            FaultKind::ReadUninit => "read of uninitialized memory",
            FaultKind::WriteUnmapped => "write to unmapped memory",
            FaultKind::WritePerm => "write to unwritable memory",
            FaultKind::ExecUnmapped => "execution of unmapped memory",
            FaultKind::ExecViolation => "execution of non-executable memory",
            FaultKind::ReadWatch => "watched read",
            FaultKind::WriteWatch => "watched write",
            FaultKind::AddressOverflow => "address space overflow",
        };
        write!(f, "{what} at {:#x}", self.addr)
    }
}

impl std::error::Error for MemFault {}

/// One guest page: its bytes, and one permission bitset per byte.
///
/// Boxed as fixed-size arrays so a page is a single allocation of a known size
/// and page lookups hand back a contiguous slice the accessors can bulk-check.
#[derive(Clone)]
struct Page {
    data: Box<[u8; PAGE_SIZE as usize]>,
    perm: Box<[Perm; PAGE_SIZE as usize]>,
}

impl Page {
    fn unmapped() -> Self {
        Self {
            data: Box::new([0; PAGE_SIZE as usize]),
            perm: Box::new([perm::NONE; PAGE_SIZE as usize]),
        }
    }
}

/// A sparse, page-granular guest address space.
///
/// Unmapped pages are simply absent, so a 64-bit address space costs only what
/// the guest actually touches.
#[derive(Clone, Default)]
pub struct Mmu {
    pages: FxHashMap<u64, Page>,
    /// When set, reading a byte without [`perm::INIT`] faults. Off by default:
    /// a plain replay harness seeds registers and memory it cares about and
    /// legitimately reads zeroes elsewhere, and would drown in false faults.
    pub check_uninit: bool,
    /// When set, [`perm::READ_WATCH`] and [`perm::WRITE_WATCH`] bytes fault.
    /// Separate from the bits themselves so watchpoints can be armed once and
    /// cheaply silenced during harness setup.
    pub watchpoints_armed: bool,
}

/// Splits an access into per-page `(page index, offset, length)` chunks.
///
/// Every accessor walks pages rather than bytes: a 4096-byte read is one hash
/// lookup and a bulk permission scan, not 4096 lookups.
fn page_chunks(addr: u64, len: u64) -> Result<impl Iterator<Item = (u64, usize, usize)>, MemFault> {
    if len != 0 && addr.checked_add(len - 1).is_none() {
        return Err(MemFault::new(FaultKind::AddressOverflow, addr));
    }
    let mut remaining = len;
    let mut cursor = addr;
    Ok(std::iter::from_fn(move || {
        if remaining == 0 {
            return None;
        }
        let offset = cursor & PAGE_MASK;
        let take = (PAGE_SIZE - offset).min(remaining);
        let chunk = (cursor >> 12, offset as usize, take as usize);
        cursor = cursor.wrapping_add(take);
        remaining -= take;
        Some(chunk)
    }))
}

impl Mmu {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of pages currently backed by memory. Mostly a test and telemetry
    /// hook — it is the honest measure of an MMU's footprint.
    pub fn resident_pages(&self) -> usize {
        self.pages.len()
    }

    /// Maps `len` bytes at `addr` with `permissions`, zero-filling the range.
    ///
    /// Follows `MAP_FIXED` semantics: an already-mapped range is replaced rather
    /// than refused. [`perm::MAP`] is added implicitly — a mapped byte is mapped
    /// regardless of what the caller asked for.
    pub fn map(&mut self, addr: u64, len: u64, permissions: Perm) -> Result<(), MemFault> {
        for (index, offset, take) in page_chunks(addr, len)? {
            let page = self.pages.entry(index).or_insert_with(Page::unmapped);
            page.data[offset..offset + take].fill(0);
            page.perm[offset..offset + take].fill(permissions | perm::MAP);
        }
        Ok(())
    }

    /// Unmaps `len` bytes at `addr`, discarding contents and permissions.
    ///
    /// A page whose every byte becomes unmapped is dropped outright, so
    /// map/unmap churn does not leak pages.
    pub fn unmap(&mut self, addr: u64, len: u64) -> Result<(), MemFault> {
        for (index, offset, take) in page_chunks(addr, len)? {
            let Some(page) = self.pages.get_mut(&index) else {
                continue;
            };
            page.data[offset..offset + take].fill(0);
            page.perm[offset..offset + take].fill(perm::NONE);
            if page.perm.iter().all(|&p| p == perm::NONE) {
                self.pages.remove(&index);
            }
        }
        Ok(())
    }

    /// Changes the permissions of an already-mapped range, preserving contents.
    ///
    /// [`perm::INIT`] is preserved rather than taken from `permissions`:
    /// initializedness is a property of the bytes, and `mprotect` does not
    /// scribble on them. Unmapped bytes in the range fault, matching `mprotect`.
    pub fn protect(&mut self, addr: u64, len: u64, permissions: Perm) -> Result<(), MemFault> {
        // Checked in a separate pass so a partially-invalid request changes
        // nothing — a half-applied mprotect would be a state the guest cannot
        // reach on real hardware.
        for (index, offset, take) in page_chunks(addr, len)? {
            let base = index << 12;
            match self.pages.get(&index) {
                Some(page) => {
                    for byte in offset..offset + take {
                        if page.perm[byte] & perm::MAP == 0 {
                            let at = base + byte as u64;
                            return Err(MemFault::new(FaultKind::WriteUnmapped, at));
                        }
                    }
                }
                None => return Err(MemFault::new(FaultKind::WriteUnmapped, base + offset as u64)),
            }
        }

        for (index, offset, take) in page_chunks(addr, len)? {
            let page = self.pages.get_mut(&index).expect("checked above");
            for byte in offset..offset + take {
                let init = page.perm[byte] & perm::INIT;
                page.perm[byte] = permissions | perm::MAP | init;
            }
        }
        Ok(())
    }

    /// Returns the permissions of a single byte, or [`perm::NONE`] if unmapped.
    pub fn permissions(&self, addr: u64) -> Perm {
        self.pages
            .get(&(addr >> 12))
            .map_or(perm::NONE, |page| page.perm[(addr & PAGE_MASK) as usize])
    }

    /// Reads `out.len()` bytes into `out`, requiring [`perm::READ`].
    pub fn read(&self, addr: u64, out: &mut [u8]) -> Result<(), MemFault> {
        self.read_with(addr, out, perm::READ)
    }

    /// Reads instruction bytes, requiring [`perm::EXEC`].
    ///
    /// The decoder calls this rather than [`read`](Self::read) so that jumping
    /// into a non-executable page faults at the fetch, the way it does on
    /// hardware, instead of silently decoding data as code.
    pub fn read_code(&self, addr: u64, out: &mut [u8]) -> Result<(), MemFault> {
        self.read_with(addr, out, perm::EXEC)
    }

    fn read_with(&self, addr: u64, out: &mut [u8], required: Perm) -> Result<(), MemFault> {
        let executing = required & perm::EXEC != 0;
        let mut written = 0;
        for (index, offset, take) in page_chunks(addr, out.len() as u64)? {
            let base = index << 12;
            let Some(page) = self.pages.get(&index) else {
                let kind = if executing {
                    FaultKind::ExecUnmapped
                } else {
                    FaultKind::ReadUnmapped
                };
                return Err(MemFault::new(kind, base + offset as u64));
            };
            for byte in offset..offset + take {
                let at = base + byte as u64;
                let held = page.perm[byte];
                if held & perm::MAP == 0 {
                    let kind = if executing {
                        FaultKind::ExecUnmapped
                    } else {
                        FaultKind::ReadUnmapped
                    };
                    return Err(MemFault::new(kind, at));
                }
                if held & required == 0 {
                    let kind = if executing {
                        FaultKind::ExecViolation
                    } else {
                        FaultKind::ReadPerm
                    };
                    return Err(MemFault::new(kind, at));
                }
                if self.check_uninit && held & perm::INIT == 0 {
                    return Err(MemFault::new(FaultKind::ReadUninit, at));
                }
                if self.watchpoints_armed && held & perm::READ_WATCH != 0 {
                    return Err(MemFault::new(FaultKind::ReadWatch, at));
                }
            }
            out[written..written + take].copy_from_slice(&page.data[offset..offset + take]);
            written += take;
        }
        Ok(())
    }

    /// Writes `bytes` at `addr`, requiring [`perm::WRITE`] and marking the
    /// written bytes initialized.
    pub fn write(&mut self, addr: u64, bytes: &[u8]) -> Result<(), MemFault> {
        // Permissions are validated across the whole range before any byte
        // lands, so a store straddling a read-only boundary leaves memory
        // untouched instead of half-written.
        let mut read = 0;
        for (index, offset, take) in page_chunks(addr, bytes.len() as u64)? {
            let base = index << 12;
            let Some(page) = self.pages.get(&index) else {
                return Err(MemFault::new(
                    FaultKind::WriteUnmapped,
                    base + offset as u64,
                ));
            };
            for byte in offset..offset + take {
                let at = base + byte as u64;
                let held = page.perm[byte];
                if held & perm::MAP == 0 {
                    return Err(MemFault::new(FaultKind::WriteUnmapped, at));
                }
                if held & perm::WRITE == 0 {
                    return Err(MemFault::new(FaultKind::WritePerm, at));
                }
                if self.watchpoints_armed && held & perm::WRITE_WATCH != 0 {
                    return Err(MemFault::new(FaultKind::WriteWatch, at));
                }
            }
            read += take;
        }
        debug_assert_eq!(read, bytes.len());

        let mut written = 0;
        for (index, offset, take) in page_chunks(addr, bytes.len() as u64)? {
            let page = self.pages.get_mut(&index).expect("checked above");
            page.data[offset..offset + take].copy_from_slice(&bytes[written..written + take]);
            for byte in offset..offset + take {
                page.perm[byte] |= perm::INIT;
            }
            written += take;
        }
        Ok(())
    }

    /// Writes `bytes` ignoring permissions, mapping any absent pages.
    ///
    /// This is the loader and harness entry point — seeding a guest image or a
    /// fixture is not a guest access and must not be refused by the permissions
    /// it is itself installing. Never reachable from emulated code.
    pub fn write_unchecked(&mut self, addr: u64, bytes: &[u8], permissions: Perm) {
        let mut written = 0;
        let chunks = page_chunks(addr, bytes.len() as u64)
            .expect("write_unchecked range must fit the address space");
        for (index, offset, take) in chunks {
            let page = self.pages.entry(index).or_insert_with(Page::unmapped);
            page.data[offset..offset + take].copy_from_slice(&bytes[written..written + take]);
            page.perm[offset..offset + take].fill(permissions | perm::MAP | perm::INIT);
            written += take;
        }
    }

    /// Captures the full contents of the address space.
    ///
    /// Deliberately a deep copy: correctness first, and a copy-on-write or
    /// dirty-page scheme is a drop-in replacement behind this same pair of
    /// methods once snapshot cost shows up in a profile.
    pub fn snapshot(&self) -> MmuSnapshot {
        MmuSnapshot {
            pages: self.pages.clone(),
        }
    }

    /// Restores a snapshot, discarding every change made since it was taken.
    pub fn restore(&mut self, snapshot: &MmuSnapshot) {
        self.pages.clone_from(&snapshot.pages);
    }
}

/// An opaque point-in-time copy of an [`Mmu`].
#[derive(Clone)]
pub struct MmuSnapshot {
    pages: FxHashMap<u64, Page>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapped() -> Mmu {
        let mut mmu = Mmu::new();
        mmu.map(0x1000, 0x2000, perm::RW_INIT).unwrap();
        mmu
    }

    #[test]
    fn maps_reads_and_writes() {
        let mut mmu = mapped();
        mmu.write(0x1004, &[1, 2, 3, 4]).unwrap();
        let mut out = [0; 4];
        mmu.read(0x1004, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3, 4]);
    }

    #[test]
    fn unmapped_access_faults_at_the_offending_byte() {
        let mmu = mapped();
        let mut out = [0; 4];
        // The read starts inside the mapping and runs off its end; the fault
        // address is the boundary, not the start of the access.
        assert_eq!(
            mmu.read(0x2ffe, &mut out).unwrap_err(),
            MemFault::new(FaultKind::ReadUnmapped, 0x3000)
        );
    }

    #[test]
    fn access_spanning_pages_is_contiguous() {
        let mut mmu = mapped();
        let bytes: Vec<u8> = (0..16).collect();
        mmu.write(0x1ff8, &bytes).unwrap();
        let mut out = [0; 16];
        mmu.read(0x1ff8, &mut out).unwrap();
        assert_eq!(out.to_vec(), bytes);
    }

    #[test]
    fn write_to_read_only_memory_faults_and_changes_nothing() {
        let mut mmu = Mmu::new();
        mmu.map(0x1000, PAGE_SIZE, perm::RX_INIT).unwrap();
        assert_eq!(
            mmu.write(0x1000, &[0xff]).unwrap_err(),
            MemFault::new(FaultKind::WritePerm, 0x1000)
        );
        let mut out = [0xaa];
        mmu.read(0x1000, &mut out).unwrap();
        assert_eq!(out, [0]);
    }

    #[test]
    fn partially_refused_write_is_not_applied() {
        let mut mmu = Mmu::new();
        mmu.map(0x1000, PAGE_SIZE, perm::RW_INIT).unwrap();
        mmu.map(0x2000, PAGE_SIZE, perm::RX_INIT).unwrap();
        // Straddles the writable/read-only boundary: nothing must land.
        assert_eq!(
            mmu.write(0x1ffc, &[0xff; 8]).unwrap_err(),
            MemFault::new(FaultKind::WritePerm, 0x2000)
        );
        let mut out = [0xaa; 4];
        mmu.read(0x1ffc, &mut out).unwrap();
        assert_eq!(out, [0; 4]);
    }

    #[test]
    fn fetching_from_non_executable_memory_faults() {
        let mut mmu = mapped();
        let mut out = [0; 4];
        assert_eq!(
            mmu.read_code(0x1000, &mut out).unwrap_err(),
            MemFault::new(FaultKind::ExecViolation, 0x1000)
        );
        mmu.protect(0x1000, PAGE_SIZE, perm::READ | perm::EXEC)
            .unwrap();
        mmu.read_code(0x1000, &mut out).unwrap();
    }

    #[test]
    fn protect_preserves_contents_and_initializedness() {
        let mut mmu = mapped();
        mmu.write(0x1000, &[7; 4]).unwrap();
        mmu.check_uninit = true;
        mmu.protect(0x1000, PAGE_SIZE, perm::READ).unwrap();
        let mut out = [0; 4];
        mmu.read(0x1000, &mut out).unwrap();
        assert_eq!(out, [7; 4]);
    }

    #[test]
    fn protect_of_unmapped_memory_is_refused_entirely() {
        let mut mmu = mapped();
        assert!(mmu.protect(0x2000, 0x2000, perm::READ).is_err());
        // The mapped prefix keeps its original permissions.
        assert_eq!(mmu.permissions(0x2000), perm::RW_INIT);
    }

    #[test]
    fn uninitialized_reads_fault_only_when_checked() {
        let mut mmu = Mmu::new();
        mmu.map(0x1000, PAGE_SIZE, perm::MAP | perm::READ_WRITE)
            .unwrap();
        let mut out = [0; 1];
        mmu.read(0x1000, &mut out).unwrap();

        mmu.check_uninit = true;
        assert_eq!(
            mmu.read(0x1000, &mut out).unwrap_err(),
            MemFault::new(FaultKind::ReadUninit, 0x1000)
        );
        // Writing the byte defines it.
        mmu.write(0x1000, &[1]).unwrap();
        mmu.read(0x1000, &mut out).unwrap();
    }

    #[test]
    fn watchpoints_fire_only_when_armed() {
        let mut mmu = Mmu::new();
        mmu.map(0x1000, PAGE_SIZE, perm::RW_INIT | perm::WRITE_WATCH)
            .unwrap();
        mmu.write(0x1000, &[1]).unwrap();

        mmu.watchpoints_armed = true;
        assert_eq!(
            mmu.write(0x1000, &[2]).unwrap_err(),
            MemFault::new(FaultKind::WriteWatch, 0x1000)
        );
        // A read of the same byte is unaffected by a write watch.
        let mut out = [0; 1];
        mmu.read(0x1000, &mut out).unwrap();
        assert_eq!(out, [1]);
    }

    #[test]
    fn unmap_releases_pages_and_faults_afterwards() {
        let mut mmu = mapped();
        assert_eq!(mmu.resident_pages(), 2);
        mmu.unmap(0x1000, 0x2000).unwrap();
        assert_eq!(mmu.resident_pages(), 0);
        let mut out = [0; 1];
        assert_eq!(
            mmu.read(0x1000, &mut out).unwrap_err(),
            MemFault::new(FaultKind::ReadUnmapped, 0x1000)
        );
    }

    #[test]
    fn partial_unmap_keeps_the_rest_of_the_page() {
        let mut mmu = mapped();
        mmu.write(0x1000, &[9; 8]).unwrap();
        mmu.unmap(0x1000, 4).unwrap();
        assert_eq!(mmu.resident_pages(), 2);
        let mut out = [0; 4];
        mmu.read(0x1004, &mut out).unwrap();
        assert_eq!(out, [9; 4]);
    }

    #[test]
    fn snapshot_and_restore_round_trips_contents_and_mappings() {
        let mut mmu = mapped();
        mmu.write(0x1000, &[1, 2, 3, 4]).unwrap();
        let snapshot = mmu.snapshot();

        mmu.write(0x1000, &[9, 9, 9, 9]).unwrap();
        mmu.map(0x8000, PAGE_SIZE, perm::RW_INIT).unwrap();
        mmu.unmap(0x2000, PAGE_SIZE).unwrap();

        mmu.restore(&snapshot);
        let mut out = [0; 4];
        mmu.read(0x1000, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3, 4]);
        // The post-snapshot mapping is gone and the unmapped page is back.
        assert_eq!(mmu.permissions(0x8000), perm::NONE);
        mmu.read(0x2000, &mut out).unwrap();
    }

    #[test]
    fn access_wrapping_the_address_space_faults() {
        let mmu = mapped();
        let mut out = [0; 8];
        assert_eq!(
            mmu.read(u64::MAX - 2, &mut out).unwrap_err(),
            MemFault::new(FaultKind::AddressOverflow, u64::MAX - 2)
        );
    }

    #[test]
    fn write_unchecked_maps_and_ignores_permissions() {
        let mut mmu = Mmu::new();
        mmu.write_unchecked(0x1000, &[1, 2, 3, 4], perm::READ | perm::EXEC);
        let mut out = [0; 4];
        mmu.read(0x1000, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3, 4]);
        assert_eq!(
            mmu.write(0x1000, &[0]).unwrap_err(),
            MemFault::new(FaultKind::WritePerm, 0x1000)
        );
    }
}
