//! Processor registers.
//!
//! The definitions live in `pcode-types`, shared with the SLEIGH decoder so
//! that a decoded register and an IR register are the same type.

pub use pcode_types::register::{Register, RegisterId, RegisterMutRef, RegisterRef};
