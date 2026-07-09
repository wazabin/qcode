//! Dead-code elimination: removing dead instructions and dead memory loads.

mod dead_block_args;
mod dead_insns;
pub mod dead_load;

pub use dead_block_args::{remove_dead_block_args, remove_dead_block_params};
pub(crate) use dead_block_args::{remove_dead_block_args_host, remove_dead_block_params_host};
pub(crate) use dead_block_args::{remove_params_from_block, remove_params_from_block_host};
pub use dead_insns::{dead_insns, remove_dead_insns};
pub use dead_load::{dead_load_insns, remove_dead_load_insns, remove_dead_load_insns_block};
