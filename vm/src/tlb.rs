//! A software TLB: the shape compiled code needs to reach guest memory without
//! calling back into Rust.
//!
//! The [`Mmu`](crate::mmu::Mmu) answers an access with a hash lookup, a
//! per-page bounds walk and a per-byte permission scan. That is the right shape
//! for an interpreter, which pays a dispatch per operation anyway, and the wrong
//! one for compiled code, where it is the *only* thing standing between a guest
//! load and a machine load. This cache is the part a compiled block can inline:
//! an index, a tag compare, and an add.
//!
//! # What an entry promises, and what it does not
//!
//! An entry says only *where the page lives*: the host address of a resident
//! [`PageData`](crate::mmu::PageData) for one guest page. It says nothing about
//! permissions. Compiled code checks those itself, per byte, out of the same
//! allocation — which is why [`PageData`](crate::mmu::PageData) keeps the
//! permission bytes next to the data bytes at a fixed offset rather than in a
//! second allocation.
//!
//! Splitting reads from writes would let the table itself carry the coarse
//! permission, as icicle's does. It would buy nothing here: this MMU's
//! permissions are per *byte*, so the byte scan happens either way, and one
//! table means one entry to fill and invalidate.
//!
//! # Invalidation
//!
//! An entry holds a raw pointer into a page allocation, so it outlives its page
//! only if nobody drops one. Every operation that can add, drop or replace a
//! page flushes the whole table; permission edits do not, because permissions
//! are read live from the page rather than cached here.

use crate::mmu::PAGE_SIZE;

/// log2 of the number of entries. 64 entries covers the working set of the code
/// this runs — a stack page, a couple of data pages, the code page — with room
/// for a few more before it thrashes, and the whole table stays inside a few
/// cache lines.
pub const TLB_INDEX_BITS: u32 = 6;

/// Number of entries in the table. A power of two, so indexing is a mask.
pub const TLB_ENTRIES: usize = 1 << TLB_INDEX_BITS;

/// The address bits an entry's tag holds: the guest page base.
const PAGE_MASK: u64 = PAGE_SIZE - 1;

/// A tag that no address produces, because every real tag is page-aligned.
const INVALID_TAG: u64 = u64::MAX;

/// One cached translation.
///
/// `#[repr(C)]` and 16 bytes wide on purpose: compiled code computes an entry's
/// address arithmetically from the guest address, so both the field order and
/// the size are part of the interface.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TlbEntry {
    /// The guest page base this entry translates, or [`INVALID_TAG`].
    pub tag: u64,
    /// Added to a guest address to get the host address of that byte.
    pub guest_to_host_offset: u64,
}

impl TlbEntry {
    const fn invalid() -> Self {
        Self {
            tag: INVALID_TAG,
            guest_to_host_offset: 0,
        }
    }

    /// The tag an address belongs under.
    pub const fn tag_of(addr: u64) -> u64 {
        addr & !PAGE_MASK
    }
}

/// The table compiled code indexes.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TranslationCache {
    pub entries: [TlbEntry; TLB_ENTRIES],
}

impl Default for TranslationCache {
    fn default() -> Self {
        Self {
            entries: [TlbEntry::invalid(); TLB_ENTRIES],
        }
    }
}

impl TranslationCache {
    /// The slot an address maps to.
    pub const fn index(addr: u64) -> usize {
        ((addr >> PAGE_SIZE.trailing_zeros()) as usize) & (TLB_ENTRIES - 1)
    }

    /// Drops every cached translation.
    ///
    /// Called for any change to which pages exist or where they live. It is
    /// deliberately the blunt instrument: mapping is a setup-time operation and
    /// a mis-scoped invalidation here would be a use-after-free in compiled
    /// code.
    pub fn flush(&mut self) {
        self.entries.fill(TlbEntry::invalid());
    }

    /// Caches `host` as the address of the page holding `addr`.
    ///
    /// # Safety
    ///
    /// `host` must be the start of a page allocation that stays live and in
    /// place until the next [`flush`](Self::flush).
    pub fn insert(&mut self, addr: u64, host: *mut u8) {
        let tag = TlbEntry::tag_of(addr);
        self.entries[Self::index(addr)] = TlbEntry {
            tag,
            // Stored relative to the guest address so the hot path is one add
            // rather than a mask and an add.
            guest_to_host_offset: (host as u64).wrapping_sub(tag),
        };
    }

    /// The host address of `addr`, if it is cached.
    pub fn lookup(&self, addr: u64) -> Option<*mut u8> {
        let entry = self.entries[Self::index(addr)];
        (entry.tag == TlbEntry::tag_of(addr))
            .then(|| addr.wrapping_add(entry.guest_to_host_offset) as *mut u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_table_translates_nothing() {
        let tlb = TranslationCache::default();
        assert!(tlb.lookup(0).is_none());
        // The last page in the address space, whose base is the one bit
        // pattern the invalid tag could be confused with.
        assert!(tlb.lookup(!PAGE_MASK).is_none());
    }

    #[test]
    fn an_inserted_page_translates_every_byte_in_it() {
        let mut page = vec![0u8; PAGE_SIZE as usize];
        let host = page.as_mut_ptr();
        let mut tlb = TranslationCache::default();
        tlb.insert(0x1234, host);

        assert_eq!(tlb.lookup(0x1000), Some(host));
        assert_eq!(tlb.lookup(0x1fff), Some(unsafe { host.add(0xfff) }));
        // The next page is a different tag, and shares no entry with this one.
        assert!(tlb.lookup(0x2000).is_none());
    }

    #[test]
    fn a_flush_forgets_everything() {
        let mut page = vec![0u8; PAGE_SIZE as usize];
        let mut tlb = TranslationCache::default();
        tlb.insert(0x1000, page.as_mut_ptr());
        tlb.flush();
        assert!(tlb.lookup(0x1000).is_none());
    }

    #[test]
    fn pages_a_multiple_of_the_table_size_apart_share_a_slot() {
        let stride = PAGE_SIZE * TLB_ENTRIES as u64;
        assert_eq!(
            TranslationCache::index(0x1000),
            TranslationCache::index(0x1000 + stride)
        );
        let mut page = vec![0u8; PAGE_SIZE as usize];
        let mut tlb = TranslationCache::default();
        tlb.insert(0x1000, page.as_mut_ptr());
        tlb.insert(0x1000 + stride, page.as_mut_ptr());
        // The later insert evicted the earlier one rather than answering for it.
        assert!(tlb.lookup(0x1000).is_none());
        assert!(tlb.lookup(0x1000 + stride).is_some());
    }
}
