//! Decompilation pass: recover `switch` statements from equality cascades.
//!
//! A `switch` compiled to a linear chain of equality tests —
//! `if (x == a) … else if (x == b) … else …` — is structured as a right-nested
//! `if`/`else` tree. This pass folds such a chain back into a [`Stmt::Switch`].
//! It recognises both the direct `x == c` form and gcc's `(x - c) == 0`
//! encoding, and `||`-fused fallthrough labels (`case a: case b:`).
//!
//! Scope: **linear** cascades only. gcc's `-O2` balanced binary-search trees
//! (relational navigation nodes with a shared `default` reached by `goto`) are
//! not folded yet — that is the next increment, and what removes the residual
//! gotos on the `branch_table` / `jump_table` fixtures.

use qcode::{context::Context, value::FunctionId};

use crate::pipeline::DecompilePass;

use super::{
    ast::{Program, Stmt, SwitchCase},
    cast::{BinOp, Expr},
    refine::is_condition_only,
};

/// Fewer than this many case labels is left as an `if`/`else` chain — a `switch`
/// earns its keep only once the chain is long enough to read better as one.
const MIN_CASES: usize = 3;

/// Recovers `switch` statements from linear equality cascades in the AST.
#[derive(Default)]
pub struct RecoverSwitch;

impl DecompilePass for RecoverSwitch {
    const NAME: &'static str = "recover_switch";

    fn description(&self) -> &'static str {
        "recover switch statements from equality cascades"
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

/// Recovers switches in `stmts`, recursing into nested bodies first so an inner
/// cascade folds before the statement that contains it.
fn recover_stmts(ctx: &Context, stmts: &mut [Stmt]) -> bool {
    let mut changed = false;
    for stmt in stmts.iter_mut() {
        changed |= recover_children(ctx, stmt);
        if let Some(switch) = try_switch(ctx, stmt) {
            *stmt = switch;
            changed = true;
        }
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

/// Folds the equality cascade headed by `stmt` into a [`Stmt::Switch`], or
/// returns `None` if `stmt` is not such a cascade (or is too short to be worth a
/// `switch`).
fn try_switch(ctx: &Context, stmt: &Stmt) -> Option<Stmt> {
    let Stmt::If { cond, then, els } = stmt else {
        return None;
    };
    let (scrutinee, values) = match_case_test(cond)?;
    let mut cases = vec![SwitchCase {
        values,
        body: then.clone(),
    }];

    // Walk the right-nested `else` chain, taking each `x == c` test on the same
    // scrutinee as another case; the first non-matching `else` is the default.
    // Each arm leads with the (emit-invisible) statements computing the next test
    // condition, which we skip past.
    let mut rest = skip_condition_prefix(ctx, els);
    while let [Stmt::If { cond, then, els }] = rest {
        match match_case_test(cond) {
            Some((s, values)) if s == scrutinee => {
                cases.push(SwitchCase {
                    values,
                    body: then.clone(),
                });
                rest = skip_condition_prefix(ctx, els);
            }
            _ => break,
        }
    }

    let label_count: usize = cases.iter().map(|c| c.values.len()).sum();
    if label_count < MIN_CASES {
        return None;
    }
    Some(Stmt::Switch {
        scrutinee,
        cases,
        default: rest.to_vec(),
    })
}

/// Drops the leading condition-only statements of `stmts` — the pure, single-use
/// values that compute the next case test and are emitted nowhere.
fn skip_condition_prefix<'a>(ctx: &Context, stmts: &'a [Stmt]) -> &'a [Stmt] {
    let skip = stmts
        .iter()
        .take_while(|s| is_condition_only(ctx, s))
        .count();
    &stmts[skip..]
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
