//! Control-flow structuring backend (SAILR): the decompilation passes that turn
//! optimized qcode into a high-level [`Program`] AST and pretty-print it.
//!
//! Decompilation is a sequence of [`DecompilePass`]es
//! ([`decompile_function`]), each reading the immutable qcode IR and rewriting
//! the shared AST: [`Structure`] recovers nested `if`/loop control flow from the
//! CFG (recovering per-edge branch conditions from block terminators, which the
//! bare `from -> to` CFG edges do not carry), then [`RefineLoops`] rewrites
//! endless loops into `while`/`do-while`. SAILR deopt passes join the sequence
//! as they land.
//!
//! [`DecompilePass`]: crate::DecompilePass

pub mod ast;
pub mod cast;
mod condition;
mod emit;
mod lower;
mod lower_expr;
mod refine;
mod structuring;
mod switch;
pub mod tokens;

pub use ast::{Program, Stmt};
pub use cast::{BinOp, Expr, UnOp};
pub use condition::{BlockExit, EdgeCondition, block_exit};
pub use emit::{emit_c, emit_tokens};
pub use lower::lower_function;
pub use lower_expr::lower_expr;
pub use refine::RefineLoops;
pub use structuring::{Structure, decompile_function};
pub use switch::RecoverSwitch;
pub use tokens::{Token, TokenKind, TokenLine};
