//! The structuring AST and its goto-count metric.
//!
//! Phase 1 only populates the "flat" subset of this AST — a straight list of
//! labels, raw IR statements, and gotos (see [`lower`](super::lower)). Later
//! phases will introduce nested `if`/`while`/`switch` nodes that replace runs of
//! gotos; the goto count is the metric that must fall monotonically as they do.

use std::collections::HashMap;

use qcode::value::{BlockId, InstructionId};

use super::cast::Expr;

/// A single structured statement.
///
/// Phase 1 emits only the flat, goto-oriented variants. Control-flow terminators
/// (`branch`/`cbranch`/`branchind`) are *not* represented as [`Stmt::Raw`]; they
/// are lowered into [`Stmt::Goto`] / [`Stmt::GotoIf`] instead. Every other
/// instruction, including calls and returns, is carried through verbatim as
/// [`Stmt::Raw`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    /// A jump target for the block `BlockId`.
    Label(BlockId),
    /// A verbatim IR instruction (arithmetic, load/store, call, return, ...).
    Raw(InstructionId),
    /// `goto <label>;`
    Goto(BlockId),
    /// `if (<cond>) goto <label>;`
    GotoIf { cond: Expr, target: BlockId },
}

/// A lowered function body: a flat statement list plus the label names its
/// gotos and labels refer to.
#[derive(Debug, Clone, Default)]
pub struct Program {
    pub stmts: Vec<Stmt>,
    pub(crate) labels: HashMap<BlockId, String>,
}

impl Program {
    /// The number of `goto` statements (conditional and unconditional) in the
    /// program. This is the SAILR quality metric: it starts high in phase 1 and
    /// must not increase as later phases structure the code.
    pub fn goto_count(&self) -> usize {
        self.stmts
            .iter()
            .filter(|s| matches!(s, Stmt::Goto(_) | Stmt::GotoIf { .. }))
            .count()
    }

    /// The label name assigned to `block`, if it is referenced in this program.
    pub fn label(&self, block: BlockId) -> Option<&str> {
        self.labels.get(&block).map(String::as_str)
    }
}
