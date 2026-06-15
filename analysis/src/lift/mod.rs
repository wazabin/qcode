//! Recursive disassembly: discovering and lifting new code addresses.

mod discovery;

pub use discovery::{discover_addresses_in_binary, lift_new_addresses};
