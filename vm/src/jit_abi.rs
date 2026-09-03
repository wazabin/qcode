//! The calling interface between compiled code and this VM's memory.
//!
//! Compiled code reaches guest RAM through an inlined translation and an
//! inlined permission check (see [`crate::tlb`]). Neither can answer every
//! access: the page may not be cached yet, the access may straddle a page
//! boundary, the permissions may refuse it, or a dynamic check may be in force
//! that the inline test cannot express. All of those land here, on the same
//! [`Mmu`](crate::mmu::Mmu) the interpreter uses.
//!
//! That is the point of the split. There is one implementation of what an
//! access *means* — mapping, permissions, initializedness, watchpoints, fault
//! addresses — and the inline path is a cache in front of it, never a second
//! opinion. A miss costs a call; a wrong answer would cost correctness.
//!
//! # Faults
//!
//! These return a status rather than taking a fault themselves, because
//! compiled code has to unwind on its own: it stops at the faulting access,
//! leaving the guest state its earlier stores produced, and returns the status
//! to the runtime. The fault itself is left on the [`VmMemory`] for the caller
//! to take, exactly as an interpreted fault would be.

use crate::VmMemory;

/// The access succeeded.
pub const ACCESS_OK: u32 = 0;
/// The access faulted; the fault is on the [`VmMemory`].
pub const ACCESS_FAULT: u32 = 1;

/// Performs a guest RAM read that compiled code could not do inline.
///
/// # Safety
///
/// `memory` must point to a live [`VmMemory`] that nothing else is borrowing,
/// `out` to a writable `u64`, and `size` must be at most 8. Called only from
/// code this crate's JIT backend generated.
pub unsafe extern "C" fn qcode_jit_load(
    memory: *mut VmMemory,
    addr: u64,
    size: u32,
    out: *mut u64,
) -> u32 {
    // SAFETY: the caller guarantees an exclusive, live pointer.
    let memory = unsafe { &mut *memory };
    let size = size as usize;
    debug_assert!(size <= 8);

    let mut bytes = [0u8; 8];
    if let Err(fault) = memory.mmu.read(addr, &mut bytes[..size]) {
        memory.record_read_fault(fault);
        return ACCESS_FAULT;
    }
    // SAFETY: the caller guarantees `out` is a writable `u64`.
    unsafe { out.write(u64::from_le_bytes(bytes)) };

    // The access worked, so the page is resident and reachable: cache it so
    // the next execution of this instruction stays inline. An access that
    // straddled two pages caches the first, which the inline path's same-page
    // test will decline again — that is a slow instruction, not a wrong one.
    memory.mmu.cache_translation(addr);
    ACCESS_OK
}

/// Performs a guest RAM write that compiled code could not do inline.
///
/// # Safety
///
/// As [`qcode_jit_load`], and `size` must be at most 8.
pub unsafe extern "C" fn qcode_jit_store(
    memory: *mut VmMemory,
    addr: u64,
    size: u32,
    value: u64,
) -> u32 {
    // SAFETY: the caller guarantees an exclusive, live pointer.
    let memory = unsafe { &mut *memory };
    let size = size as usize;
    debug_assert!(size <= 8);

    if let Err(fault) = memory.mmu.write(addr, &value.to_le_bytes()[..size]) {
        memory.record_write_fault(fault);
        return ACCESS_FAULT;
    }
    memory.mmu.cache_translation(addr);
    ACCESS_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmu::{FaultKind, PAGE_SIZE, perm};

    fn memory() -> VmMemory {
        let mut memory = VmMemory::new();
        memory.mmu.map(0x1000, PAGE_SIZE, perm::RW_INIT).unwrap();
        memory
    }

    #[test]
    fn a_served_access_leaves_the_page_cached() {
        let mut memory = memory();
        let mut out = 0u64;
        let status = unsafe { qcode_jit_store(&raw mut memory, 0x1008, 4, 0xdead_beef) };
        assert_eq!(status, ACCESS_OK);
        let status = unsafe { qcode_jit_load(&raw mut memory, 0x1008, 4, &raw mut out) };
        assert_eq!(status, ACCESS_OK);
        assert_eq!(out, 0xdead_beef);
        // Having served it the slow way once, the next one can be inline.
        assert!(memory.mmu.cache_translation(0x1008));
    }

    #[test]
    fn a_refused_access_reports_a_fault_and_caches_nothing() {
        let mut memory = memory();
        let mut out = 0u64;
        let status = unsafe { qcode_jit_load(&raw mut memory, 0x9000, 1, &raw mut out) };
        assert_eq!(status, ACCESS_FAULT);
        assert_eq!(
            memory.take_fault().map(|fault| fault.kind),
            Some(FaultKind::ReadUnmapped)
        );
    }

    #[test]
    fn a_write_to_read_only_memory_faults_at_the_byte() {
        let mut memory = VmMemory::new();
        memory.mmu.map(0x1000, PAGE_SIZE, perm::RX_INIT).unwrap();
        let status = unsafe { qcode_jit_store(&raw mut memory, 0x1000, 1, 0xff) };
        assert_eq!(status, ACCESS_FAULT);
        assert_eq!(memory.take_fault().map(|fault| fault.addr), Some(0x1000));
    }

    #[test]
    fn an_access_straddling_two_pages_is_served_whole() {
        let mut memory = VmMemory::new();
        memory
            .mmu
            .map(0x1000, 2 * PAGE_SIZE, perm::RW_INIT)
            .unwrap();
        let mut out = 0u64;
        // The inline path declines this shape outright; the slow path is the
        // only thing that ever sees it, so it has to be right here.
        unsafe { qcode_jit_store(&raw mut memory, 0x1ffe, 8, 0x0102_0304_0506_0708) };
        unsafe { qcode_jit_load(&raw mut memory, 0x1ffe, 8, &raw mut out) };
        assert_eq!(out, 0x0102_0304_0506_0708);
    }
}
