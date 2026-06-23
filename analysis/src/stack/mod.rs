//! Stack-frame passes: brightening symbolic stack accesses and lowering them
//! back onto the real stack pointer.

pub(crate) mod brighten;
pub(crate) mod canonicalize;
pub(crate) mod frame;
mod lower_stack;

pub use brighten::brighten_stack;
pub use lower_stack::lower_stack;
