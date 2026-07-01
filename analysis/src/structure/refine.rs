//! Decompilation pass: refine endless loops into `while` / `do-while`.
//!
//! [`super::structuring`] emits every natural loop as an endless [`Stmt::Loop`]
//! with `break`/`continue` exits. This pass walks the finished AST and rewrites
//! each loop into the most specific pre- or post-tested form its shape permits;
//! everything it can't classify stays an endless `while (true)`.

use qcode::{
    context::Context,
    value::{FunctionId, Instruction, ValueId},
};

use crate::pipeline::DecompilePass;

use super::{
    ast::{Program, Stmt},
    emit::is_side_effecting,
};

/// Rewrites every endless `Loop` in the program into a pre- or post-tested loop
/// where its shape permits.
#[derive(Default)]
pub struct RefineLoops;

impl DecompilePass for RefineLoops {
    const NAME: &'static str = "refine_loops";

    fn description(&self) -> &'static str {
        "rewrite endless loops into while / do-while"
    }

    fn run(
        &self,
        ctx: &Context,
        _fun_id: FunctionId,
        program: &mut Program,
    ) -> Result<bool, String> {
        Ok(refine_stmts(ctx, &mut program.stmts))
    }
}

crate::register_decompile_pass!(RefineLoops);

/// Refines every `Loop` in `stmts`, recursing into nested bodies first so an
/// inner loop is refined before the loop that contains it.
fn refine_stmts(ctx: &Context, stmts: &mut [Stmt]) -> bool {
    let mut changed = false;
    for stmt in stmts.iter_mut() {
        match stmt {
            Stmt::If { then, els, .. } => {
                changed |= refine_stmts(ctx, then);
                changed |= refine_stmts(ctx, els);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::Loop { body } => {
                changed |= refine_stmts(ctx, body);
            }
            _ => {}
        }
        if let Stmt::Loop { body } = stmt {
            let refined = refine_loop(ctx, std::mem::take(body));
            changed |= !matches!(refined, Stmt::Loop { .. });
            *stmt = refined;
        }
    }
    changed
}

/// Rewrites one endless loop body into its most specific loop form: a `while`
/// when the loop tests before its body, a `do-while` when it tests after, and
/// an endless [`Stmt::Loop`] otherwise.
///
/// - **`while (c) { … }`** — the body is (a possibly-empty run of condition-only
///   statements, then) a single `if (c) { … continue } else break`. The
///   condition-only prefix is dropped: being pure and single-use, it folds into
///   `c` and is emitted nowhere else.
/// - **`do { … } while (c)`** — the body ends in `if (c) continue; else break;`
///   with the loop body preceding it.
fn refine_loop(ctx: &Context, mut body: Vec<Stmt>) -> Stmt {
    // Pre-test: an optional condition-only prefix followed by the loop's `if`,
    // one arm of which is exactly a `break`.
    let prefix = body.iter().take_while(|s| is_condition_only(ctx, s)).count();
    if prefix + 1 == body.len() {
        if let Some(Stmt::If { cond, then, els }) = body.get(prefix) {
            if is_only(els, &Stmt::Break) {
                return Stmt::While {
                    cond: cond.clone(),
                    body: without_trailing_continue(then.clone()),
                };
            }
            if is_only(then, &Stmt::Break) {
                return Stmt::While {
                    cond: cond.clone().logical_not(),
                    body: without_trailing_continue(els.clone()),
                };
            }
        }
    }

    // Post-test: the body ends in `if (c) continue; else break;` (in either
    // polarity); everything before it is the loop body, run each iteration.
    if let Some(Stmt::If { cond, then, els }) = body.last() {
        let cond = if is_only(then, &Stmt::Continue) && is_only(els, &Stmt::Break) {
            Some(cond.clone())
        } else if is_only(then, &Stmt::Break) && is_only(els, &Stmt::Continue) {
            Some(cond.clone().logical_not())
        } else {
            None
        };
        if let Some(cond) = cond {
            body.pop();
            return Stmt::DoWhile { cond, body };
        }
    }

    Stmt::Loop { body }
}

/// Whether `stmts` is exactly the single statement `stmt`.
fn is_only(stmts: &[Stmt], stmt: &Stmt) -> bool {
    stmts == std::slice::from_ref(stmt)
}

/// Drops a trailing `continue` (redundant at the end of a loop body).
fn without_trailing_continue(mut stmts: Vec<Stmt>) -> Vec<Stmt> {
    if let Some(Stmt::Continue) = stmts.last() {
        stmts.pop();
    }
    stmts
}

/// Whether a statement is a pure, single-use value that only feeds a control-flow
/// condition — so it folds into the condition expression and is emitted nowhere
/// else. Such statements are structurally present but invisible in the output,
/// so pattern matchers (loop refinement, switch recovery) skip past them.
pub(crate) fn is_condition_only(ctx: &Context, stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Raw(id) => {
            let insn = Instruction::from_id(ctx, *id);
            !is_side_effecting(&insn.mnemonic()) && ctx.users(ValueId::Instruction(*id)).len() <= 1
        }
        _ => false,
    }
}
