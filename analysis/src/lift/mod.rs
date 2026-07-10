//! Recursive disassembly: discovering and lifting new code addresses.

mod discovery;
mod split;

pub use discovery::{discover_addresses_in_binary, lift_new_addresses};
pub use split::{has_cross_function_reference, split_overlapping_functions};
