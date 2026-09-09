//! A virtual machine layer over the QCode interpreter.
//!
//! [`qcode_emulator`] interprets QCode over a *pre-lifted, immutable* module: it
//! answers "what does this IR compute". This crate adds what a machine needs on
//! top of that — mapped memory with permissions, faults delivered as values
//! rather than aborts, code discovered on demand as the guest reaches it, and
//! snapshots — so that a guest program can be run rather than merely evaluated.
//!
//! # Faults are values, not aborts
//!
//! A bad guest access is returned to the caller, so a harness can observe it
//! and carry on rather than dying:
//!
//! ```
//! use qcode_vm::{VmMemory, FaultKind, perm};
//!
//! let mut memory = VmMemory::new();
//! // One read-only page of initialised memory.
//! memory.mmu.map(0x1000, 0x1000, perm::MAP | perm::READ | perm::INIT).unwrap();
//!
//! let mut buffer = [0u8; 4];
//! assert!(memory.mmu.read(0x1000, &mut buffer).is_ok());
//!
//! // Writing it faults, and says why and where.
//! let fault = memory.mmu.write(0x1000, &[0xff]).unwrap_err();
//! assert_eq!(fault.kind, FaultKind::WritePerm);
//! assert_eq!(fault.addr, 0x1000);
//!
//! // So does touching an address that was never mapped.
//! let fault = memory.mmu.read(0x9000, &mut buffer).unwrap_err();
//! assert_eq!(fault.kind, FaultKind::ReadUnmapped);
//! ```
//!
//! Execution strategy is pluggable through `set_block_executor`, which is how
//! [`qcode_jit`](https://docs.rs/qcode_jit) is installed on a machine.

pub mod flat;
pub mod jit_abi;
pub mod memory;
pub mod mmu;
pub mod optimize;
pub mod stats;
pub mod tlb;
pub mod vm;

pub use jit_abi::{
    ACCESS_FAULT, ACCESS_OK, qcode_jit_load, qcode_jit_sdiv128, qcode_jit_srem128, qcode_jit_store,
    qcode_jit_udiv128, qcode_jit_urem128,
};
pub use memory::VmMemory;
pub use mmu::{
    FaultKind, MemFault, Mmu, MmuSnapshot, PAGE_PERM_OFFSET, PAGE_SIZE, PageData, Perm, perm,
};
pub use optimize::{Cleanup, forward_temp_stores};
pub use stats::Stats;
pub use tlb::{TLB_ENTRIES, TLB_INDEX_BITS, TlbEntry, TranslationCache};
pub use vm::{BlockExecutor, CodeError, CodeSource, Executed, Vm, VmExit};
