//! Typed SSA-style p-code IR for binary analysis.
//!
//! `qcode` models the semantics of lifted machine code. It is inspired by
//! Ghidra's p-code, with additional first-class values for instructions, basic
//! blocks, functions, and literals. The crate contains the IR, its builder,
//! the QCode text-format lowering API, and integrity checks; optimization and
//! recovery passes live separately in [`qcode_analysis`].
//!
//! # Getting started
//!
//! A [`Context`] owns a QCode module. Create one, then construct IR with a
//! [`Builder`] or parse QCode source through [`lower::lower_str`].
//!
//! ```rust
//! use qcode::context::Context;
//!
//! let _context = Context::new();
//! ```
//!
//! # Core concepts
//!
//! - A [`Space`] is a uniformly addressed memory region, such as RAM or a
//!   register file.
//! - A [`ValueId`] identifies every IR value: literals, SSA instructions,
//!   varnodes, blocks, and functions.
//! - A [`FunctionBody`] owns a function's instructions, blocks, and block
//!   parameters; module-wide values are owned by the [`Context`].
//! - A [`Builder`] emits instructions and constructs control flow in a block.
//!
//! Reference types such as [`InstructionRef`], [`BlockRef`], and
//! [`FunctionRef`] borrow their owning context, so they cannot outlive the IR
//! arena.
//!
//! # QCode source
//!
//! [`lower::lower_str`] parses QCode source at runtime. For source literals,
//! the re-exported [`qcode!`] macro performs the same lowering and binds names
//! declared in the source into the surrounding Rust scope.
//!
//! [`qcode_analysis`]: https://docs.rs/qcode_analysis
//! [`Space`]: crate::space::Space
//! [`ValueId`]: crate::value::ValueId
//! [`FunctionBody`]: crate::value::FunctionBody
//! [`Context`]: crate::context::Context
//! [`Builder`]: crate::builder::Builder
//! [`InstructionRef`]: crate::value::InstructionRef
//! [`BlockRef`]: crate::value::BlockRef
//! [`FunctionRef`]: crate::value::FunctionRef
//! [`qcode!`]: macro@qcode

pub mod address_index;
mod arena_integrity;
pub mod assumption;
pub mod builder;
pub mod context;
pub mod discovery;
pub mod error;
pub mod intrinsics;
pub mod lower;
pub mod memory_image;
pub mod obligation;
pub mod pass_scope;
pub mod space;
pub mod types;
pub mod value;

pub use arena_integrity::{verify_body_arena_integrity, verify_body_arena_integrity_scoped};
pub use wazabin_qcode_macro::qcode;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

/// Re-export for the [`pass_log!`] macro; not public API.
#[doc(hidden)]
pub use log as __log;
