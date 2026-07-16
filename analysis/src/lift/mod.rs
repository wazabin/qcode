//! Recursive disassembly: discovering and lifting new code addresses.

use qcode::{address_index::AddressIndex, context::Context};

/// Cached module-wide lookup from machine addresses to functions and blocks.
///
/// Lifting takes ownership of this analysis while it grows the clean IR and
/// updates the index alongside every address-bearing mutation. Passes that only
/// rewrite instructions, types, names, signatures, or metadata preserve it;
/// transforms that add/delete/readdress blocks or functions invalidate it unless
/// they likewise take, update, and return the index.
pub struct AddressAnalysis;

impl crate::GlobalAnalysis for AddressAnalysis {
    type Result = AddressIndex;

    fn analyze(ctx: &Context<'_>) -> Self::Result {
        AddressIndex::analyze(ctx)
    }
}

mod discovery;

pub use discovery::{discover_addresses_in_binary, lift_new_addresses};
