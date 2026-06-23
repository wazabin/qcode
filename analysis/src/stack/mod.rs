//! Stack-frame passes: re-rooting stack accesses on the incoming stack pointer
//! `@SP` and canonicalizing per-offset slots.

pub(crate) mod brighten;
pub(crate) mod canonicalize;
pub(crate) mod frame;

pub use brighten::brighten_stack;
