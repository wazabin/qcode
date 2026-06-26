//! Dead-code elimination: removing dead instructions and dead memory loads.

mod dead_block_args;
mod dead_insns;
pub mod dead_load;

pub use dead_block_args::remove_dead_block_args;
pub use dead_insns::{dead_insns, remove_dead_insns};
pub use dead_load::{dead_load_insns, remove_dead_load_insns, remove_dead_load_insns_block};
