//! Pseudo-C emission for a lowered [`Program`].
//!
//! Deliberately dumb and total: every [`Stmt`] variant renders to a line of
//! C-like text. Instruction bodies reuse the IR's own statement formatting.

use qcode::{
    context::Context,
    value::{BlockId, Instruction},
};

use super::{ast::Program, ast::Stmt};

/// One emitted line of pseudo-C, tagged with the block it belongs to.
///
/// The `block` provenance lets interactive front-ends map source lines back to
/// CFG blocks (e.g. to highlight or navigate), which a flat string cannot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line {
    /// The rendered text of the line (no trailing newline).
    pub text: String,
    /// The block this line belongs to, if any. `Label` lines and the statements
    /// that follow them carry the enclosing block.
    pub block: Option<BlockId>,
}

/// Renders `program` as a sequence of provenance-tagged lines.
pub fn emit_lines(ctx: &Context, program: &Program) -> Vec<Line> {
    let mut lines = Vec::with_capacity(program.stmts.len());
    let mut current: Option<BlockId> = None;
    for stmt in &program.stmts {
        if let Stmt::Label(block) = stmt {
            current = Some(*block);
        }
        lines.push(Line {
            text: stmt_text(ctx, program, stmt),
            block: current,
        });
    }
    lines
}

/// Renders `program` as pseudo-C source text.
pub fn emit_c(ctx: &Context, program: &Program) -> String {
    let mut out = String::new();
    for stmt in &program.stmts {
        out.push_str(&stmt_text(ctx, program, stmt));
        out.push('\n');
    }
    out
}

/// Renders a single statement to its line of pseudo-C (no trailing newline).
fn stmt_text(ctx: &Context, program: &Program, stmt: &Stmt) -> String {
    match stmt {
        Stmt::Label(block) => format!("{}:", label_of(program, *block)),
        Stmt::Raw(insn) => {
            let insn = Instruction::from_id(ctx, *insn);
            format!("    {}", insn.as_statement())
        }
        Stmt::Goto(target) => format!("    goto {};", label_of(program, *target)),
        Stmt::GotoIf { cond, target } => {
            format!("    if ({cond}) goto {};", label_of(program, *target))
        }
    }
}

/// The label name for `block`, falling back to a synthetic name if the program
/// somehow lacks one (should not happen for well-formed programs).
fn label_of(program: &Program, block: BlockId) -> String {
    program
        .label(block)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("bb_{}", Into::<usize>::into(block)))
}
