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
    value::{BlockId, FunctionId, Instruction, InstructionId, ValueId},
};

use crate::pipeline::DecompilePass;

use super::{
    ast::{Program, Stmt, SwitchCase},
    cast::{BinOp, Expr, ExprKind},
    emit::{is_memory_read, is_side_effecting},
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
    collector.collect(std::slice::from_ref(stmt), (0, u64::MAX));
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

    // A value that selects more than one case (an overlapping/obfuscated tree)
    // would fold into two `match` arms for the same label — reject it.
    let mut seen = HashSet::new();
    if !collector
        .cases
        .iter()
        .flat_map(|c| &c.values)
        .all(|&v| seen.insert(v))
    {
        return None;
    }

    // A `break`/`continue` that escapes to an enclosing loop reads unambiguously
    // in the if/else form but would look like a switch `break` once folded into a
    // `match` arm. Leave such a cascade unfolded rather than emit ambiguous code.
    let default_has_ctl = collector
        .default
        .as_deref()
        .is_some_and(has_unbound_loop_ctl);
    if default_has_ctl
        || collector
            .cases
            .iter()
            .any(|c| has_unbound_loop_ctl(&c.body))
    {
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

/// Whether `stmts` contains a `break`/`continue` that targets an enclosing loop
/// — i.e. one not already bound by a nested loop inside `stmts`. Nested loops
/// capture their own loop control, but `if`/`switch` (a Rust `match`) do not, so
/// the walk descends through those and stops at loops.
fn has_unbound_loop_ctl(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Break | Stmt::Continue => true,
        Stmt::If { then, els, .. } => has_unbound_loop_ctl(then) || has_unbound_loop_ctl(els),
        Stmt::Switch { cases, default, .. } => {
            has_unbound_loop_ctl(default) || cases.iter().any(|c| has_unbound_loop_ctl(&c.body))
        }
        _ => false,
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
    ///
    /// `interval` is the range of scrutinee values that can reach this arm, given
    /// the navigation splits taken above it. A case whose constant falls outside
    /// that range is a contradiction (an inconsistent or obfuscated tree) and
    /// aborts the fold, so a dead case is never merged into a live `match`.
    fn collect(&mut self, stmts: &[Stmt], interval: (u64, u64)) {
        if !self.ok || stmts.is_empty() {
            return;
        }
        let (lo, hi) = interval;
        // A comparison node leads with the (now-dead once folded) statements that
        // compute its test — often multiply-used, since the compiler shares
        // `x - k` between the equality and the relational test. Skip past them to
        // reach the node, but keep the *original* arm for a default body, so a
        // body of pure assignments is not mistaken for setup.
        match skip_setup_prefix(self.ctx, stmts) {
            [Stmt::If { cond, then, els }] => {
                if let Some((s, values)) = match_case_test(cond)
                    && same_scrutinee(&s, &self.scrutinee)
                {
                    if !values.iter().all(|&v| lo <= v && v <= hi) {
                        // A case for a value the navigation splits already ruled
                        // out — an inconsistent tree we must not fold.
                        self.ok = false;
                        return;
                    }
                    let mut insns = Vec::new();
                    collect_expr_insns(cond, &mut insns);
                    self.cases.push(SwitchCase {
                        values,
                        body: then.clone(),
                        insns,
                    });
                    self.collect(els, interval);
                    return;
                }
                if scrutinee_of(cond).is_some_and(|s| same_scrutinee(&s, &self.scrutinee)) {
                    // A relational split on the scrutinee: both sides continue,
                    // each with the sub-interval the split implies (when it is a
                    // recognizable unsigned test; otherwise the interval is kept).
                    let (then_iv, els_iv) = split_interval(cond, &self.scrutinee, interval)
                        .unwrap_or((interval, interval));
                    self.collect(then, then_iv);
                    self.collect(els, els_iv);
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

/// The sub-intervals a relational split on the scrutinee implies for its true
/// and false sides, clamped to the current `interval`. Returns `None` when the
/// condition is not a recognizable unsigned relational test against a constant on
/// the bare scrutinee (e.g. a signed, cast-wrapped compare), in which case the
/// caller keeps the interval unchanged rather than narrow it wrongly.
fn split_interval(
    cond: &Expr,
    scrutinee: &Expr,
    (lo, hi): (u64, u64),
) -> Option<((u64, u64), (u64, u64))> {
    let ExprKind::Binary(op, a, b) = &cond.kind else {
        return None;
    };
    // Orient the relation as `scrutinee <op> k`.
    let (k, op) = if same_scrutinee(a, scrutinee) {
        match b.kind {
            ExprKind::Const(k) => (k, *op),
            _ => return None,
        }
    } else if same_scrutinee(b, scrutinee) {
        match a.kind {
            ExprKind::Const(k) => (k, flip_cmp(*op)),
            _ => return None,
        }
    } else {
        return None;
    };
    const EMPTY: (u64, u64) = (1, 0);
    let clamp = |a: u64, b: u64| (a.max(lo), b.min(hi));
    Some(match op {
        // x < k → [lo, k-1] ; x >= k → [k, hi]
        BinOp::Lt => (
            k.checked_sub(1).map_or(EMPTY, |u| clamp(lo, u)),
            clamp(k, hi),
        ),
        // x <= k → [lo, k] ; x > k → [k+1, hi]
        BinOp::Le => (
            clamp(lo, k),
            k.checked_add(1).map_or(EMPTY, |l| clamp(l, hi)),
        ),
        // x > k → [k+1, hi] ; x <= k → [lo, k]
        BinOp::Gt => (
            k.checked_add(1).map_or(EMPTY, |l| clamp(l, hi)),
            clamp(lo, k),
        ),
        // x >= k → [k, hi] ; x < k → [lo, k-1]
        BinOp::Ge => (
            clamp(k, hi),
            k.checked_sub(1).map_or(EMPTY, |u| clamp(lo, u)),
        ),
        _ => return None,
    })
}

/// The comparison operator with its operands swapped (`k < x` ⟺ `x > k`).
fn flip_cmp(op: BinOp) -> BinOp {
    match op {
        BinOp::Lt => BinOp::Gt,
        BinOp::Le => BinOp::Ge,
        BinOp::Gt => BinOp::Lt,
        BinOp::Ge => BinOp::Le,
        other => other,
    }
}

/// Whether two expressions name the same scrutinee. Prefers IR identity: two
/// occurrences lowered from the same [`ValueId`] are the same scrutinee, and two
/// lowered from *different* values are not — even if they share a display name
/// (distinct SSA versions of a register can). Falls back to structural equality
/// (by name) only when a value is missing provenance.
fn same_scrutinee(a: &Expr, b: &Expr) -> bool {
    match (a.value, b.value) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

/// The scrutinee of a condition: the sole variable leaf it mentions (returned
/// with its IR provenance intact), or `None` if it mentions none or more than
/// one distinct variable.
fn scrutinee_of(cond: &Expr) -> Option<Expr> {
    let mut leaves = Vec::new();
    collect_var_leaves(cond, &mut leaves);
    let first = *leaves.first()?;
    // All leaves must be the same variable (by name); return the leaf itself so
    // its `ValueId` is preserved for `same_scrutinee`.
    leaves.iter().all(|v| *v == first).then(|| first.clone())
}

/// Collects every variable-leaf expression of `expr` (preserving provenance).
fn collect_var_leaves<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    match &expr.kind {
        ExprKind::Var(_) => out.push(expr),
        ExprKind::Const(_) => {}
        ExprKind::Unary(_, e) | ExprKind::Deref { ptr: e, .. } | ExprKind::Cast { expr: e, .. } => {
            collect_var_leaves(e, out)
        }
        ExprKind::Binary(_, a, b) => {
            collect_var_leaves(a, out);
            collect_var_leaves(b, out);
        }
        ExprKind::Unknown { operands, .. } => {
            for operand in operands {
                collect_var_leaves(operand, out);
            }
        }
    }
}

/// Collects the instruction results of every node of `expr` (its provenance),
/// so a folded-away comparison tree can be mapped back to the low-level code.
fn collect_expr_insns(expr: &Expr, out: &mut Vec<InstructionId>) {
    if let Some(ValueId::Instruction(id)) = expr.value {
        out.push(id);
    }
    match &expr.kind {
        ExprKind::Const(_) | ExprKind::Var(_) => {}
        ExprKind::Unary(_, e) | ExprKind::Deref { ptr: e, .. } | ExprKind::Cast { expr: e, .. } => {
            collect_expr_insns(e, out)
        }
        ExprKind::Binary(_, a, b) => {
            collect_expr_insns(a, out);
            collect_expr_insns(b, out);
        }
        ExprKind::Unknown { operands, .. } => {
            for operand in operands {
                collect_expr_insns(operand, out);
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
    match &cond.kind {
        ExprKind::Binary(BinOp::LOr, a, b) => {
            let left = collect_eq_disjuncts(a, values)?;
            let right = collect_eq_disjuncts(b, values)?;
            same_scrutinee(&left, &right).then_some(left)
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
    let ExprKind::Binary(BinOp::Eq, a, b) = &cond.kind else {
        return None;
    };
    let (expr, k) = const_split(a, b)?;
    // `(x - c) == 0` selects `c`; the compiler emits equality this way.
    if k == 0
        && let ExprKind::Binary(BinOp::Sub, lhs, rhs) = &expr.kind
        && let ExprKind::Const(c) = rhs.kind
    {
        return Some(((**lhs).clone(), c));
    }
    Some((expr, k))
}

/// Splits a comparison's operands into its (single) constant side and the other
/// side, or `None` when neither or both are constants.
fn const_split(a: &Expr, b: &Expr) -> Option<(Expr, u64)> {
    match (&a.kind, &b.kind) {
        (ExprKind::Const(_), ExprKind::Const(_)) => None,
        (_, ExprKind::Const(c)) => Some((a.clone(), *c)),
        (ExprKind::Const(c), _) => Some((b.clone(), *c)),
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
///
/// A load is excluded even though it is side-effect-free: its value is only valid
/// until the next aliasing write, so it is not freely foldable and must not be
/// skipped as if it were dead setup (it is emitted as its own statement).
fn is_pure_setup(ctx: &Context, stmt: &Stmt) -> bool {
    matches!(stmt, Stmt::Raw(id) if {
        let m = Instruction::from_id(ctx, *id).mnemonic();
        !is_side_effecting(m) && !is_memory_read(m)
    })
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

    /// Asserts every `goto <name>;` in the emitted C has a matching `<name>:`
    /// label line — i.e. no jump dangles to an absorbed/suppressed label.
    fn assert_no_dangling_gotos(c: &str) {
        let labels: std::collections::HashSet<&str> = c
            .lines()
            .filter_map(|l| l.trim().strip_suffix(':'))
            .collect();
        for line in c.lines() {
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("goto ") {
                let target = rest.trim_end_matches(';');
                assert!(
                    labels.contains(target),
                    "goto `{target}` has no label in:\n{c}"
                );
            }
        }
    }

    #[test]
    fn unbound_loop_control_is_detected() {
        // A bare break/continue, or one reachable through if/match, escapes to an
        // enclosing loop; one inside a nested loop is bound by that loop.
        let c = Expr::var("c");
        assert!(has_unbound_loop_ctl(&[Stmt::Break]));
        assert!(has_unbound_loop_ctl(&[Stmt::If {
            cond: c.clone(),
            then: vec![Stmt::Continue],
            els: vec![],
        }]));
        assert!(!has_unbound_loop_ctl(&[Stmt::Loop {
            body: vec![Stmt::Break],
        }]));
    }

    #[test]
    fn negative_case_label_prints_signed() {
        // A case value whose sign bit is set at the scrutinee's i32 width should
        // read as `-1`, not the sign-extended `0xffffffff`.
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
                %cm = i32 %x == i32 0xffffffff;
                if %cm goto <casem> else goto <default_lbl>;
            <casem>
                store(&out, i32 0xff0);
                goto <done>;
            <default_lbl>
                store(&out, i32 0x0);
                goto <done>;
            <done>
                local i64 ptr;
                return [ptr];
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        let c = emit_c(&ctx, &program);
        assert!(has_switch(&program.stmts), "expected a switch:\n{c}");
        assert!(
            c.contains("-1 =>") && !c.contains("0xffffffff"),
            "negative case label should print as signed:\n{c}"
        );
    }

    #[test]
    fn overlapping_cases_are_not_folded() {
        // The value 0x1 selects two different arms — an inconsistent tree that
        // must not fold into a `match` with a duplicated label.
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
                %c3 = i32 %x == i32 0x1;
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

        let program = decompile_function(&ctx, f).unwrap();
        let c = emit_c(&ctx, &program);
        assert!(
            !has_switch(&program.stmts),
            "overlapping cases must not fold:\n{c}"
        );
        assert_no_dangling_gotos(&c);
    }

    #[test]
    fn inconsistent_navigation_tree_is_not_folded() {
        // The `x < 2` navigation routes the `lo` side, yet that side tests
        // `x == 5` — impossible under `x < 2`. Such a contradictory tree
        // (patched/obfuscated) must not fold: doing so would merge a dead case
        // into the live `match`. It stays an if/else chain instead.
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
                %c5 = i32 %x == i32 0x5;
                if %c5 goto <case5> else goto <default_lbl>;
            <case5>
                store(&out, i32 0x50);
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

        let program = decompile_function(&ctx, f).unwrap();
        let c = emit_c(&ctx, &program);
        assert!(
            !has_switch(&program.stmts),
            "an inconsistent tree must not fold to a switch:\n{c}"
        );
        // No case body may be lost by the (aborted) fold.
        for needle in ["0x20", "0x50", "0x30", "0x0"] {
            assert!(c.contains(needle), "missing body `{needle}`:\n{c}");
        }
    }

    #[test]
    fn switch_default_with_external_goto_stays_consistent() {
        // The shared default is also reached from outside the comparison cascade.
        // Folding the cascade into a `match` that absorbs the default's label
        // would orphan that outside jump; validation must keep the output
        // consistent (falling back to the flat lowering) rather than dangle it.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 sel;
            varnode i8 pre;
            varnode i32 out;

            fn f:
            <entry>
                %g = load(i8, &pre);
                if %g goto <cascade> else goto <default_lbl>;
            <cascade>
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

        let program = decompile_function(&ctx, f).unwrap();
        let c = emit_c(&ctx, &program);
        // No jump may dangle, and no case body may be lost.
        assert_no_dangling_gotos(&c);
        for needle in ["0x10", "0x20", "0x30", "0x0"] {
            assert!(c.contains(needle), "missing case body `{needle}`:\n{c}");
        }
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

        let program = decompile_function(&ctx, f).unwrap();
        let c = emit_c(&ctx, &program);

        assert!(has_switch(&program.stmts), "expected a switch node:\n{c}");
        assert_eq!(
            program.goto_count(),
            0,
            "switch should erase the gotos:\n{c}"
        );
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

        let program = decompile_function(&ctx, f).unwrap();
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

        let program = decompile_function(&ctx, f).unwrap();
        assert!(
            !has_switch(&program.stmts),
            "two cases should stay if/else:\n{}",
            emit_c(&ctx, &program)
        );
    }
}
