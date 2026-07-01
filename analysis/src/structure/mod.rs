//! Control-flow structuring backend (SAILR).
//!
//! See `docs/sailr-structuring.md` for the overall design. This is **phase 0**:
//! condition recovery, the enabler for everything else.
//!
//! The CFG stores edges as bare `from -> to` pairs ([`EdgeData`]); it records
//! *no* branch condition and *no* notion of which outgoing edge is the taken vs.
//! not-taken side. That information lives only in the block's terminator
//! instruction ([`CBranch`]). Structuring needs it to attach reaching
//! conditions to edges, so this module reconstructs a typed, polarity-aware view
//! of a block's control-flow exit.
//!
//! [`EdgeData`]: qcode::value::block
//! [`CBranch`]: qcode::value::insn::CBranch

pub mod ast;
pub mod cast;
mod condition;
mod emit;
mod lower;
mod lower_expr;
mod structuring;

pub use ast::{Program, Stmt};
pub use cast::{BinOp, Expr, UnOp};
pub use condition::{BlockExit, EdgeCondition, block_exit};
pub use emit::{Line, emit_c, emit_lines};
pub use lower::lower_function;
pub use lower_expr::lower_expr;
pub use structuring::structure_function;
