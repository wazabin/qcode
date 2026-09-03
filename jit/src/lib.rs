//! A Cranelift JIT backend for QCode: an execution strategy alongside the
//! interpreter, not a replacement for it.
//!
//! [`qcode_emulator`] interprets QCode one operation at a time, which means
//! every intermediate value is materialised into its value table and every
//! operand is resolved through the module. Compiled code does neither: a QCode
//! block is already SSA, so it maps onto Cranelift's SSA directly and values
//! consumed inside the block stay in machine registers.
//!
//! The backend is deliberately partial — see [`compile::Unsupported`]. A block
//! it declines is run by the interpreter instead, so coverage can grow without
//! ever being a correctness question.
pub mod compile;
pub mod jit;

pub use compile::{SpaceTable, Unsupported};
pub use jit::{Jit, JitStats};
