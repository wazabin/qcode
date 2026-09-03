//! A virtual machine layer over the QCode interpreter.
//!
//! [`qcode_emulator`] interprets QCode over a *pre-lifted, immutable* module: it
//! answers "what does this IR compute". This crate adds what a machine needs on
//! top of that — mapped memory with permissions, faults delivered as values
//! rather than aborts, code discovered on demand as the guest reaches it, and
//! snapshots — so that a guest program can be run rather than merely evaluated.

pub mod flat;
pub mod memory;
pub mod optimize;
pub mod stats;
pub mod mmu;
pub mod vm;

pub use memory::VmMemory;
pub use optimize::{Cleanup, forward_temp_stores};
pub use stats::Stats;
pub use vm::{BlockExecutor, CodeError, CodeSource, Vm, VmExit};
pub use mmu::{FaultKind, MemFault, Mmu, MmuSnapshot, PAGE_SIZE, Perm, perm};
