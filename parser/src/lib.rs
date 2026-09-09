//! Parser and AST for the QCode text format.
//!
//! QCode has a textual surface syntax, which is how much of the toolchain's
//! test suite is written and what the `qcode!` macro accepts. This crate turns
//! that text into a syntax tree and nothing more: it performs no name
//! resolution, no typing, and no lowering, so it depends on neither the IR nor
//! the rest of the toolchain.
//!
//! **Most users want [`qcode`] instead**, and its `qcode::lower::lower_str`,
//! which parses *and* lowers into a live module. Reach for this crate when a
//! tool needs the syntax itself — a formatter, a linter, an editor, or
//! anything reporting source spans back to a human. It is also what
//! [`wazabin_qcode_macro`] uses to validate QCode literals at compile time.
//!
//! # Example
//!
//! ```
//! use wazabin_qcode_parser::{ast::ProgramKind, qcode_from_str};
//!
//! let program = qcode_from_str("%sum = i64 0x2 + 0x3;").expect("valid QCode");
//! match program.kind {
//!     ProgramKind::Statements(statements) => assert_eq!(statements.len(), 1),
//!     ProgramKind::Functions { .. } => panic!("this source declares no functions"),
//! }
//! ```
//!
//! A syntax error carries its location rather than just a message:
//!
//! ```
//! use wazabin_qcode_parser::qcode_from_str;
//!
//! assert!(qcode_from_str("%sum = i64 0x2 +;").is_err());
//! ```
//!
//! [`qcode`]: https://docs.rs/qcode
//! [`wazabin_qcode_macro`]: https://docs.rs/wazabin-qcode-macro

pub mod ast;
mod parser;

pub use parser::{ParseError, parse_program};

/// Parse QCode source into a [`ast::Program`].
///
/// An alias for [`parse_program`], named for readability at call sites.
pub fn qcode_from_str(program: &str) -> Result<ast::Program, ParseError> {
    parse_program(program)
}
