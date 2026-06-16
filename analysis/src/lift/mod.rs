//! Recursive disassembly: discovering and lifting new code addresses.

mod discovery;
mod split;

pub use discovery::{discover_addresses_in_binary, lift_new_addresses};
pub use split::split_overlapping_functions;
