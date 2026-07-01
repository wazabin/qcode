//! Pseudo-C emission for a lowered [`Program`].
//!
//! Deliberately dumb and total: every [`Stmt`] variant renders to a line of
//! C-like text. Instruction bodies reuse the IR's own statement formatting.

use std::fmt::Write;

use qcode::{context::Context, value::Instruction};

use super::{ast::Program, ast::Stmt};

/// Renders `program` as pseudo-C source text.
pub fn emit_c(ctx: &Context, program: &Program) -> String {
    let mut out = String::new();
    for stmt in &program.stmts {
        emit_stmt(ctx, program, stmt, &mut out);
    }
    out
}

fn emit_stmt(ctx: &Context, program: &Program, stmt: &Stmt, out: &mut String) {
    match stmt {
        Stmt::Label(block) => {
            let name = label_of(program, *block);
            let _ = writeln!(out, "{name}:");
        }
        Stmt::Raw(insn) => {
            let insn = Instruction::from_id(ctx, *insn);
            let _ = writeln!(out, "    {}", insn.as_statement());
        }
        Stmt::Goto(target) => {
            let name = label_of(program, *target);
            let _ = writeln!(out, "    goto {name};");
        }
        Stmt::GotoIf { cond, target } => {
            let name = label_of(program, *target);
            let _ = writeln!(out, "    if ({cond}) goto {name};");
        }
    }
}

/// The label name for `block`, falling back to a synthetic name if the program
/// somehow lacks one (should not happen for well-formed programs).
fn label_of(program: &Program, block: qcode::value::BlockId) -> String {
    program
        .label(block)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("bb_{}", Into::<usize>::into(block)))
}
