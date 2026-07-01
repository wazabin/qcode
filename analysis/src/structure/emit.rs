//! Pseudo-C emission for a lowered [`Program`].
//!
//! Deliberately dumb and total: every [`Stmt`] variant renders to one or more
//! lines of C-like text. Instruction bodies reuse the IR's own statement
//! formatting. Structured nodes ([`Stmt::If`]) nest with brace indentation.

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
    /// The rendered text of the line, including leading indentation.
    pub text: String,
    /// The block this line belongs to, if determinable.
    pub block: Option<BlockId>,
}

/// Renders `program` as a sequence of provenance-tagged lines.
pub fn emit_lines(ctx: &Context, program: &Program) -> Vec<Line> {
    let mut lines = Vec::new();
    emit_block(ctx, program, &program.stmts, 0, &mut lines);
    lines
}

/// Renders `program` as pseudo-C source text.
pub fn emit_c(ctx: &Context, program: &Program) -> String {
    let mut out = String::new();
    for line in emit_lines(ctx, program) {
        out.push_str(&line.text);
        out.push('\n');
    }
    out
}

fn emit_block(
    ctx: &Context,
    program: &Program,
    stmts: &[Stmt],
    indent: usize,
    out: &mut Vec<Line>,
) {
    for stmt in stmts {
        emit_stmt(ctx, program, stmt, indent, out);
    }
}

fn emit_stmt(ctx: &Context, program: &Program, stmt: &Stmt, indent: usize, out: &mut Vec<Line>) {
    let pad = "    ".repeat(indent);
    match stmt {
        Stmt::Label(block) => out.push(Line {
            // Labels sit one level out from the code they head.
            text: format!(
                "{}{}:",
                "    ".repeat(indent.saturating_sub(1)),
                label_of(program, *block)
            ),
            block: Some(*block),
        }),
        Stmt::Raw(insn) => {
            let insn = Instruction::from_id(ctx, *insn);
            out.push(Line {
                text: format!("{pad}{}", insn.as_statement()),
                block: insn.block().map(|b| b.id),
            });
        }
        Stmt::Goto(target) => out.push(Line {
            text: format!("{pad}goto {};", label_of(program, *target)),
            block: None,
        }),
        Stmt::GotoIf { cond, target } => out.push(Line {
            text: format!("{pad}if ({cond}) goto {};", label_of(program, *target)),
            block: None,
        }),
        Stmt::If { cond, then, els } => {
            out.push(Line {
                text: format!("{pad}if ({cond}) {{"),
                block: None,
            });
            emit_block(ctx, program, then, indent + 1, out);
            if els.is_empty() {
                out.push(Line {
                    text: format!("{pad}}}"),
                    block: None,
                });
            } else {
                out.push(Line {
                    text: format!("{pad}}} else {{"),
                    block: None,
                });
                emit_block(ctx, program, els, indent + 1, out);
                out.push(Line {
                    text: format!("{pad}}}"),
                    block: None,
                });
            }
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
