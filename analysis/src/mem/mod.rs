//! Memory-oriented passes and analyses: promoting memory to registers
//! (mem2reg) and the memory-liveness dataflow that backs dead-load/store
//! elimination.

pub mod array_promote;
pub mod mem2reg;
pub mod mem_liveness;

pub use mem_liveness::{MemLiveness, compute_memory_liveness};
pub use mem2reg::mem2reg;
