//! The structuring AST and its goto-count metric.
//!
//! Phase 1 only populates the "flat" subset of this AST — a straight list of
//! labels, raw IR statements, and gotos (see [`lower`](super::lower)). Later
//! phases will introduce nested `if`/`while`/`switch` nodes that replace runs of
//! gotos; the goto count is the metric that must fall monotonically as they do.

use std::collections::HashMap;

use qcode::value::{BlockId, InstructionId, ValueId, function::FunctionId};

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
    /// A block-parameter (phi) copy `param = value;`, realized on a CFG edge that
    /// passes `value` to the successor block's parameter `param`. SSA join values
    /// live in block parameters; structuring materializes the incoming-argument
    /// assignment as a statement on the edge that carries it.
    Assign { param: ValueId, value: ValueId },
    /// `tmp<n> = value;` — saves a value that a cyclic phi copy on the same edge
    /// is about to clobber, so a later [`Stmt::AssignTemp`] can read the
    /// pre-transfer value. Temps are numbered per edge and used immediately.
    SaveTemp { temp: usize, value: ValueId },
    /// `param = tmp<n>;` — the phi copy that completes a cycle broken by a
    /// [`Stmt::SaveTemp`].
    AssignTemp { param: ValueId, temp: usize },
    /// `goto <label>;`
    Goto(BlockId),
    /// `if (<cond>) goto <label>;`
    GotoIf { cond: Expr, target: BlockId },
    /// A structured `if (cond) { then } else { els }`. `els` is empty for an
    /// if-then with no else arm. Introduced by phase 2 (schema matching).
    If {
        cond: Expr,
        then: Vec<Stmt>,
        els: Vec<Stmt>,
    },
    /// A pre-tested loop `while (cond) { body }`.
    While { cond: Expr, body: Vec<Stmt> },
    /// A post-tested loop `do { body } while (cond);`.
    DoWhile { cond: Expr, body: Vec<Stmt> },
    /// An endless loop `while (true) { body }` whose exits are structured as
    /// [`Stmt::Break`]. The general form when neither pre- nor post-test matches.
    Loop { body: Vec<Stmt> },
    /// `break;` — leaves the innermost enclosing loop.
    Break,
    /// `continue;` — jumps to the next iteration of the innermost enclosing loop.
    Continue,
    /// A `switch (scrutinee) { … }` recovered from an equality cascade. `default`
    /// is empty when the cascade has no fallthrough default arm.
    Switch {
        scrutinee: Expr,
        cases: Vec<SwitchCase>,
        default: Vec<Stmt>,
    },
}

/// One arm of a [`Stmt::Switch`]: the constant labels that select it (more than
/// one for fallthrough cases, e.g. `case 1: case 2:`) and the body they run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchCase {
    pub values: Vec<u64>,
    pub body: Vec<Stmt>,
    /// The comparison instructions this case folded away (the equality tests
    /// against its labels), for mapping the arm back to the low-level code.
    pub insns: Vec<InstructionId>,
}

/// Counts the `goto` statements (conditional and unconditional) in a statement
/// tree, recursing into structured nodes.
pub fn count_gotos(stmts: &[Stmt]) -> usize {
    stmts
        .iter()
        .map(|s| match s {
            Stmt::Goto(_) | Stmt::GotoIf { .. } => 1,
            Stmt::If { then, els, .. } => count_gotos(then) + count_gotos(els),
            // `break`/`continue` are structured control flow, not gotos.
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::Loop { body } => {
                count_gotos(body)
            }
            Stmt::Switch { cases, default, .. } => {
                cases.iter().map(|c| count_gotos(&c.body)).sum::<usize>() + count_gotos(default)
            }
            Stmt::Label(_)
            | Stmt::Raw(_)
            | Stmt::Assign { .. }
            | Stmt::SaveTemp { .. }
            | Stmt::AssignTemp { .. }
            | Stmt::Break
            | Stmt::Continue => 0,
        })
        .sum()
}

/// A lowered function body: a flat statement list plus the label names its
/// gotos and labels refer to.
#[derive(Debug, Clone, Default)]
pub struct Program {
    /// The function this body was lowered from, used to emit the `fn name(...)`
    /// header. `None` only for an empty default program that no pass has filled.
    pub(crate) function: Option<FunctionId>,
    pub stmts: Vec<Stmt>,
    pub(crate) labels: HashMap<BlockId, String>,
}

impl Program {
    /// The number of `goto` statements (conditional and unconditional) in the
    /// program. This is the SAILR quality metric: it starts high in phase 1 and
    /// must not increase as later phases structure the code.
    pub fn goto_count(&self) -> usize {
        count_gotos(&self.stmts)
    }

    /// The label name assigned to `block`, if it is referenced in this program.
    pub fn label(&self, block: BlockId) -> Option<&str> {
        self.labels.get(&block).map(String::as_str)
    }
}
