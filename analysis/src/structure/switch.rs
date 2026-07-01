//! Decompilation pass: recover `switch` statements from comparison trees.
//!
//! A `switch` compiles to a tree of comparisons on one scrutinee:
//!
//! - **equality leaves** — `x == c` (or gcc's `(x - c) == 0`, or `||`-fused
//!   `x == a || x == b`) select a case;
//! - **navigation nodes** — a relational test on the scrutinee (`x < k`, or
//!   gcc's `sborrow`-based signed compare) that splits the search space but
//!   selects no case;
//! - a **shared default** block that the no-match paths reach, either by falling
//!   into its label or by `goto`.
//!
//! Structuring turns that tree into nested `if`/`else`. This pass walks it, taking
//! equality leaves as cases and recursing through navigation nodes, and folds it
//! back into a [`Stmt::Switch`] — subsuming the `goto`s to the shared default.
//! Navigation nodes are recognised structurally (a scrutinee-only relational
//! condition), so the `sborrow` signed-compare idiom needs no special decoding.
//! This recovers the dense `jump_table` fixture's `dispatch` to a clean `match`.
//!
//! Not yet handled: **range cases**, where a body covers several values reached
//! by a relational bound rather than equalities (`case 1: case 2:` compiled to
//! `1 <= x <= 2`), as in the `branch_table` fixture's `classify` — that needs
//! interval tracking over the tree and is left un-folded (a safe no-op). The
//! now-dead comparison setup left before a recovered `match` is cosmetic, for a
//! later AST dead-code pass.

use std::collections::HashSet;

use qcode::{
    context::Context,
    value::{BlockId, FunctionId, Instruction},
};

use crate::pipeline::DecompilePass;

use super::{
    ast::{Program, Stmt, SwitchCase},
    cast::{BinOp, Expr},
    emit::is_side_effecting,
};

/// Fewer than this many case labels is left as an `if`/`else` chain — a `switch`
/// earns its keep only once the chain is long enough to read better as one.
const MIN_CASES: usize = 3;

/// Recovers `switch` statements from comparison trees in the AST.
#[derive(Default)]
pub struct RecoverSwitch;

impl DecompilePass for RecoverSwitch {
    const NAME: &'static str = "recover_switch";

    fn description(&self) -> &'static str {
        "recover switch statements from comparison trees"
    }

    fn run(
        &self,
        ctx: &Context,
        _fun_id: FunctionId,
        program: &mut Program,
    ) -> Result<bool, String> {
        Ok(recover_stmts(ctx, &mut program.stmts))
    }
}

crate::register_decompile_pass!(RecoverSwitch);

/// Recovers switches in `stmts`. Folds the largest switch rooted at each
/// statement first (top-down), then recurses into the result's bodies so a
/// switch nested inside a case is recovered too.
fn recover_stmts(ctx: &Context, stmts: &mut [Stmt]) -> bool {
    let mut changed = false;
    for stmt in stmts.iter_mut() {
        if let Some(switch) = try_switch(ctx, stmt) {
            *stmt = switch;
            changed = true;
        }
        changed |= recover_children(ctx, stmt);
    }
    changed
}

/// Recurses into a statement's child bodies.
fn recover_children(ctx: &Context, stmt: &mut Stmt) -> bool {
    match stmt {
        Stmt::If { then, els, .. } => recover_stmts(ctx, then) | recover_stmts(ctx, els),
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::Loop { body } => {
            recover_stmts(ctx, body)
        }
        Stmt::Switch { cases, default, .. } => {
            let mut changed = recover_stmts(ctx, default);
            for case in cases {
                changed |= recover_stmts(ctx, &mut case.body);
            }
            changed
        }
        _ => false,
    }
}

/// Folds the comparison tree rooted at `stmt` into a [`Stmt::Switch`], or returns
/// `None` if `stmt` does not head such a tree (or it is too small to be worth a
/// `switch`). The root must be an equality test, which anchors the scrutinee.
fn try_switch(ctx: &Context, stmt: &Stmt) -> Option<Stmt> {
    let Stmt::If { cond, .. } = stmt else {
        return None;
    };
    let (scrutinee, _) = match_case_test(cond)?;

    let mut collector = Collector {
        ctx,
        scrutinee: scrutinee.clone(),
        cases: Vec::new(),
        default: None,
        default_label: None,
        gotos: HashSet::new(),
        ok: true,
    };
    collector.collect(std::slice::from_ref(stmt));
    if !collector.ok {
        return None;
    }

    // Every goto encountered must route to the one captured default block; a goto
    // to anything else is an exit we can't fold away.
    match collector.default_label {
        Some(label) if !collector.gotos.iter().all(|g| *g == label) => return None,
        None if !collector.gotos.is_empty() => return None,
        _ => {}
    }

    let label_count: usize = collector.cases.iter().map(|c| c.values.len()).sum();
    if label_count < MIN_CASES {
        return None;
    }

    let mut cases = collector.cases;
    for case in &mut cases {
        case.values.sort_unstable();
    }
    cases.sort_by_key(|c| c.values.first().copied().unwrap_or(0));

    Some(Stmt::Switch {
        scrutinee,
        cases,
        default: collector.default.unwrap_or_default(),
    })
}

/// Accumulates the cases and shared default of a comparison tree as it is walked.
struct Collector<'c, 'ctx> {
    ctx: &'c Context<'ctx>,
    scrutinee: Expr,
    cases: Vec<SwitchCase>,
    /// The body of the shared default block, once found.
    default: Option<Vec<Stmt>>,
    /// The label of the shared default block (when it is a `goto` target).
    default_label: Option<BlockId>,
    /// Every `goto` target seen in the tree; all must be the default label.
    gotos: HashSet<BlockId>,
    /// Cleared when the tree contains something this pass cannot fold.
    ok: bool,
}

impl Collector<'_, '_> {
    /// Walks one arm of the tree, classifying it as a case, a navigation split, a
    /// default block, or a goto to the default.
    fn collect(&mut self, stmts: &[Stmt]) {
        if !self.ok || stmts.is_empty() {
            return;
        }
        // A comparison node leads with the (now-dead once folded) statements that
        // compute its test — often multiply-used, since the compiler shares
        // `x - k` between the equality and the relational test. Skip past them to
        // reach the node, but keep the *original* arm for a default body, so a
        // body of pure assignments is not mistaken for setup.
        match skip_setup_prefix(self.ctx, stmts) {
            [Stmt::If { cond, then, els }] => {
                if let Some((s, values)) = match_case_test(cond) {
                    if s == self.scrutinee {
                        self.cases.push(SwitchCase {
                            values,
                            body: then.clone(),
                        });
                        self.collect(els);
                        return;
                    }
                }
                if scrutinee_of(cond).as_ref() == Some(&self.scrutinee) {
                    // A relational split on the scrutinee: both sides continue.
                    self.collect(then);
                    self.collect(els);
                    return;
                }
                // A condition off the scrutinee: this whole arm is the default.
                self.set_default(None, stmts.to_vec());
            }
            // A labeled block the no-match paths `goto`: the shared default.
            [Stmt::Label(l), body @ ..] => self.set_default(Some(*l), body.to_vec()),
            // A bare jump to the (labeled) default; its body is captured elsewhere.
            [Stmt::Goto(l)] => {
                self.gotos.insert(*l);
            }
            // Anything else (including an all-setup arm that trimmed to nothing):
            // a plain default body. Keep the original, untrimmed statements.
            _ => self.set_default(None, stmts.to_vec()),
        }
    }

    /// Records the shared default block, bailing if a different one was already
    /// found (a well-formed switch has exactly one).
    fn set_default(&mut self, label: Option<BlockId>, body: Vec<Stmt>) {
        if self.default.replace(body).is_some() {
            self.ok = false;
        }
        if let Some(label) = label {
            self.default_label = Some(label);
        }
    }
}

/// The scrutinee of a condition: the sole variable it mentions, or `None` if it
/// mentions none or more than one.
fn scrutinee_of(cond: &Expr) -> Option<Expr> {
    let mut vars = Vec::new();
    collect_vars(cond, &mut vars);
    let first = vars.first()?.clone();
    vars.iter().all(|v| *v == first).then_some(Expr::Var(first))
}

/// Collects the names of every variable leaf of `expr`.
fn collect_vars(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Var(name) => out.push(name.clone()),
        Expr::Const(_) => {}
        Expr::Unary(_, e) | Expr::Deref(e) | Expr::Cast { expr: e, .. } => collect_vars(e, out),
        Expr::Binary(_, a, b) => {
            collect_vars(a, out);
            collect_vars(b, out);
        }
        Expr::Unknown { operands, .. } => {
            for operand in operands {
                collect_vars(operand, out);
            }
        }
    }
}

/// Matches a case test: a single equality or an `||`-chain of equalities against
/// a common scrutinee, returning it and the constant labels it selects. Handles
/// both `x == c` and gcc's `(x - c) == 0` encoding.
fn match_case_test(cond: &Expr) -> Option<(Expr, Vec<u64>)> {
    let mut values = Vec::new();
    let scrutinee = collect_eq_disjuncts(cond, &mut values)?;
    Some((scrutinee, values))
}

/// Collects the constant labels of an equality (or `||`-chain of equalities),
/// appending them to `values` and returning the common scrutinee.
fn collect_eq_disjuncts(cond: &Expr, values: &mut Vec<u64>) -> Option<Expr> {
    match cond {
        Expr::Binary(BinOp::LOr, a, b) => {
            let left = collect_eq_disjuncts(a, values)?;
            let right = collect_eq_disjuncts(b, values)?;
            (left == right).then_some(left)
        }
        _ => {
            let (scrutinee, value) = single_eq(cond)?;
            values.push(value);
            Some(scrutinee)
        }
    }
}

/// Matches one equality against a constant: `x == c`, `c == x`, or gcc's
/// `(x - c) == 0`. Returns the scrutinee `x` and the constant `c`.
fn single_eq(cond: &Expr) -> Option<(Expr, u64)> {
    let Expr::Binary(BinOp::Eq, a, b) = cond else {
        return None;
    };
    let (expr, k) = const_split(a, b)?;
    // `(x - c) == 0` selects `c`; the compiler emits equality this way.
    if k == 0 {
        if let Expr::Binary(BinOp::Sub, lhs, rhs) = &expr {
            if let Expr::Const(c) = rhs.as_ref() {
                return Some(((**lhs).clone(), *c));
            }
        }
    }
    Some((expr, k))
}

/// Splits a comparison's operands into its (single) constant side and the other
/// side, or `None` when neither or both are constants.
fn const_split(a: &Expr, b: &Expr) -> Option<(Expr, u64)> {
    match (a, b) {
        (Expr::Const(_), Expr::Const(_)) => None,
        (e, Expr::Const(c)) | (Expr::Const(c), e) => Some((e.clone(), *c)),
        _ => None,
    }
}

/// Drops the leading pure statements of `stmts` — the arithmetic computing the
/// next comparison, which becomes dead once the tree folds to a `switch`.
fn skip_setup_prefix<'a>(ctx: &Context, stmts: &'a [Stmt]) -> &'a [Stmt] {
    let skip = stmts.iter().take_while(|s| is_pure_setup(ctx, s)).count();
    &stmts[skip..]
}

/// Whether a statement is a pure (side-effect-free) value definition — comparison
/// setup that folds away with the switch.
fn is_pure_setup(ctx: &Context, stmt: &Stmt) -> bool {
    matches!(stmt, Stmt::Raw(id) if !is_side_effecting(&Instruction::from_id(ctx, *id).mnemonic()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{decompile_function, emit_c};
    use qcode::context::Context;
    use qcode_macro::qcode;

    fn has_switch(stmts: &[Stmt]) -> bool {
        stmts.iter().any(|s| matches!(s, Stmt::Switch { .. }))
    }

    #[test]
    fn equality_cascade_folds_into_switch() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 sel;
            varnode i32 out;

            fn f:
            <entry>
                %x = load(i32, &sel);
                %c1 = i32 %x == i32 0x1;
                if %c1 goto <case1> else goto <t2>;
            <case1>
                store(&out, i32 0x10);
                goto <done>;
            <t2>
                %c2 = i32 %x == i32 0x2;
                if %c2 goto <case2> else goto <t3>;
            <case2>
                store(&out, i32 0x20);
                goto <done>;
            <t3>
                %c3 = i32 %x == i32 0x3;
                if %c3 goto <case3> else goto <default_lbl>;
            <case3>
                store(&out, i32 0x30);
                goto <done>;
            <default_lbl>
                store(&out, i32 0x0);
                goto <done>;
            <done>
                local i64 ptr;
                return [ptr];
            "
        );

        let program = decompile_function(&ctx, f);
        let c = emit_c(&ctx, &program);

        assert!(has_switch(&program.stmts), "expected a switch node:\n{c}");
        assert_eq!(program.goto_count(), 0, "switch should erase the gotos:\n{c}");
        for needle in ["match sel {", "0x1 =>", "0x2 =>", "0x3 =>", "_ =>"] {
            assert!(c.contains(needle), "missing `{needle}` in:\n{c}");
        }
    }

    #[test]
    fn comparison_tree_with_shared_default_folds() {
        // A balanced tree: the middle case at the root, a `<` navigation split,
        // and a default reached both by falling into its label and by `goto`.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 sel;
            varnode i32 out;

            fn f:
            <entry>
                %x = load(i32, &sel);
                %c2 = i32 %x == i32 0x2;
                if %c2 goto <case2> else goto <nav>;
            <case2>
                store(&out, i32 0x20);
                goto <done>;
            <nav>
                %lt = i32 %x < i32 0x2;
                if %lt goto <lo> else goto <hi>;
            <lo>
                %c1 = i32 %x == i32 0x1;
                if %c1 goto <case1> else goto <default_lbl>;
            <case1>
                store(&out, i32 0x10);
                goto <done>;
            <hi>
                %c3 = i32 %x == i32 0x3;
                if %c3 goto <case3> else goto <default_lbl>;
            <case3>
                store(&out, i32 0x30);
                goto <done>;
            <default_lbl>
                store(&out, i32 0x0);
                goto <done>;
            <done>
                local i64 ptr;
                return [ptr];
            "
        );

        let program = decompile_function(&ctx, f);
        let c = emit_c(&ctx, &program);

        assert!(has_switch(&program.stmts), "expected a switch node:\n{c}");
        assert_eq!(
            program.goto_count(),
            0,
            "the shared default's gotos should be subsumed:\n{c}"
        );
        // Cases come out sorted, and the navigation `x < 2` is not itself a case.
        for needle in ["match sel {", "0x1 =>", "0x2 =>", "0x3 =>", "_ =>"] {
            assert!(c.contains(needle), "missing `{needle}` in:\n{c}");
        }
    }

    #[test]
    fn short_cascade_stays_if_else() {
        // Two cases is below the switch threshold: left as an if/else chain.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 sel;
            varnode i32 out;

            fn f:
            <entry>
                %x = load(i32, &sel);
                %c1 = i32 %x == i32 0x1;
                if %c1 goto <case1> else goto <t2>;
            <case1>
                store(&out, i32 0x10);
                goto <done>;
            <t2>
                %c2 = i32 %x == i32 0x2;
                if %c2 goto <case2> else goto <default_lbl>;
            <case2>
                store(&out, i32 0x20);
                goto <done>;
            <default_lbl>
                store(&out, i32 0x0);
                goto <done>;
            <done>
                local i64 ptr;
                return [ptr];
            "
        );

        let program = decompile_function(&ctx, f);
        assert!(
            !has_switch(&program.stmts),
            "two cases should stay if/else:\n{}",
            emit_c(&ctx, &program)
        );
    }
}
