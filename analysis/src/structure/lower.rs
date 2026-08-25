//! Phase 1 lowering: CFG → flat goto-based [`Program`].
//!
//! This is the "correct but maximally ugly" baseline. Every block becomes a
//! label followed by its verbatim non-control-flow instructions, and every CFG
//! out-edge becomes an explicit `goto`. No fall-through elision, no nesting —
//! that is what phase 2 (schema matching) is for. The point is to have an
//! end-to-end pipeline and a goto-count metric to drive down.

use std::collections::HashMap;

use qcode::{
    context::Context,
    value::{BasicBlock, BlockId, BlockRef, FunctionId, InstructionRef, ValueId, insn::Mnemonic},
};

use super::{
    BlockExit, ast::Program, ast::Stmt, ast::SwitchCase, block_exit, lower_expr::lower_expr,
};

/// Lowers `function_id` into a flat, goto-based [`Program`].
///
/// Blocks are emitted in the function's block order (address-sorted), with the
/// entry block first. Returns an empty program if the function has no root.
pub fn lower_function(ctx: &Context, function_id: FunctionId) -> Program {
    let function = qcode::value::FunctionRef::from_id(ctx, function_id);
    let Some(root) = function.root() else {
        return Program {
            function: Some(function_id),
            ..Program::default()
        };
    };
    let root_id = root.id;

    // Emission order: entry first, then the remaining blocks in address order.
    let mut order: Vec<BlockId> = vec![root_id];
    order.extend(function.blocks().map(|b| b.id).filter(|&id| id != root_id));

    let labels = assign_labels(ctx, &order);

    let mut stmts = Vec::new();
    for &block_id in &order {
        let block = BasicBlock::from_id(ctx, block_id);
        stmts.push(Stmt::Label(block_id));
        lower_block(ctx, block, &mut stmts);
    }

    Program {
        function: Some(function_id),
        stmts,
        labels,
    }
}

/// Assigns each block a stable label name: its IR name if it has one, otherwise
/// a synthesized `bb_<addr>` / `bb_<id>`.
pub(crate) fn assign_labels(ctx: &Context, order: &[BlockId]) -> HashMap<BlockId, String> {
    let mut labels = HashMap::with_capacity(order.len());
    for &block_id in order {
        let block = BasicBlock::from_id(ctx, block_id);
        let name = match (block.name(), block.address()) {
            (Some(name), _) => name.to_string(),
            (None, Some(addr)) => format!("bb_{addr:x}"),
            (None, None) => format!("bb_{}", usize::from(block_id.local)),
        };
        labels.insert(block_id, name);
    }
    labels
}

/// Appends the statements for a single block: its verbatim body instructions
/// followed by the gotos its control-flow exit lowers to.
fn lower_block(ctx: &Context, block: BlockRef<'_, '_>, out: &mut Vec<Stmt>) {
    for insn in block.instructions() {
        if is_replaced_by_goto(&insn) {
            continue;
        }
        out.push(Stmt::Raw(insn.id));
    }

    let from = block.id;
    match block_exit(block) {
        // The `return` instruction is a verbatim body statement (kept above);
        // it needs no goto.
        BlockExit::Return => {}
        BlockExit::Goto { target, .. } => {
            out.extend(block_arg_moves(ctx, from, target));
            out.push(Stmt::Goto(target));
        }
        BlockExit::Branch {
            condition,
            true_target,
            false_target,
            ..
        } => {
            // `if (cond) goto TRUE; goto FALSE;`. The true edge's phi copies must
            // run only when it is taken, so when it carries any they move inside a
            // guarded block (`if (cond) { copies; goto TRUE; }`).
            let true_moves = block_arg_moves(ctx, from, true_target);
            if true_moves.is_empty() {
                out.push(Stmt::GotoIf {
                    cond: lower_expr(ctx, condition),
                    target: true_target,
                });
            } else {
                let mut then = true_moves;
                then.push(Stmt::Goto(true_target));
                out.push(Stmt::If {
                    cond: lower_expr(ctx, condition),
                    then,
                    els: Vec::new(),
                });
            }
            out.extend(block_arg_moves(ctx, from, false_target));
            out.push(Stmt::Goto(false_target));
        }
        // A resolved dispatch becomes a real `switch`, each arm jumping to its
        // target. The baseline still uses gotos for the bodies — assembling
        // those is the structurer's job — but the dispatch itself is now
        // explicit, and every goto is reachable, where the indirect fallback
        // below emits one per successor of which only the first can run.
        BlockExit::Switch {
            scrutinee,
            arms,
            default,
            ..
        } => {
            let cases = arms
                .into_iter()
                .map(|(values, target)| SwitchCase {
                    values,
                    body: block_arg_moves(ctx, from, target)
                        .into_iter()
                        .chain(std::iter::once(Stmt::Goto(target)))
                        .collect(),
                    // No comparison instructions were folded away: the labels
                    // came from the terminator, not from an equality cascade.
                    insns: Vec::new(),
                })
                .collect();
            let default = default
                .map(|target| {
                    block_arg_moves(ctx, from, target)
                        .into_iter()
                        .chain(std::iter::once(Stmt::Goto(target)))
                        .collect()
                })
                .unwrap_or_default();
            out.push(Stmt::Switch {
                scrutinee: lower_expr(ctx, scrutinee),
                cases,
                default,
            });
        }

        // Indirect / unstructured exits: preserve reachability with a goto to
        // every successor. Lossy but keeps the baseline correct-by-construction.
        BlockExit::Indirect { edges } | BlockExit::Unstructured { edges } => {
            for (_, target) in edges {
                out.extend(block_arg_moves(ctx, from, target));
                out.push(Stmt::Goto(target));
            }
        }
    }
}

/// The block-parameter (phi) copies realized on the edge `from -> to`: each of
/// `to`'s parameters paired with the argument `from`'s terminator passes for it,
/// as `param = arg;` assignments.
///
/// These are the SSA join copies, which have *parallel* semantics: every
/// argument is evaluated in the pre-transfer state. Emitting them as sequential
/// assignments is only correct in dependency order (a copy whose destination
/// another copy still reads must come last), so the copies are sequentialized
/// here; a dependency cycle (e.g. a swap `a=b, b=a`) is broken by saving one
/// clobbered value into a temp ([`Stmt::SaveTemp`] / [`Stmt::AssignTemp`]).
/// Identity copies (`p = p`) are dropped as no-ops.
pub(crate) fn block_arg_moves(ctx: &Context, from: BlockId, to: BlockId) -> Vec<Stmt> {
    let args = edge_args(ctx, from, to);
    if args.is_empty() {
        return Vec::new();
    }
    let target = BasicBlock::from_id(ctx, to);
    let copies: Vec<(ValueId, ValueId)> = target
        .params()
        .zip(args)
        .filter_map(|(param, arg)| {
            let param = ValueId::BlockParam(param.id);
            (param != arg).then_some((param, arg))
        })
        .collect();
    sequentialize_copies(copies)
}

/// The source of a pending phi copy while sequentializing: the original IR
/// value, or a temp holding its pre-transfer value after a cycle was broken.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CopySrc {
    Value(ValueId),
    Temp(usize),
}

/// Orders the parallel copies `(param, arg)` into sequential statements that
/// preserve their parallel semantics: a copy is emitted only once no pending
/// copy still reads its destination. When no copy is emittable (a dependency
/// cycle), one about-to-be-clobbered destination is saved into a fresh temp and
/// its readers retargeted, which breaks the cycle.
fn sequentialize_copies(copies: Vec<(ValueId, ValueId)>) -> Vec<Stmt> {
    let mut pending: Vec<(ValueId, CopySrc)> = copies
        .into_iter()
        .map(|(p, a)| (p, CopySrc::Value(a)))
        .collect();
    let mut out = Vec::with_capacity(pending.len());
    let mut temps = 0;
    while !pending.is_empty() {
        let ready = pending.iter().position(|&(param, _)| {
            !pending
                .iter()
                .any(|&(other, src)| other != param && src == CopySrc::Value(param))
        });
        match ready {
            Some(i) => {
                let (param, src) = pending.remove(i);
                out.push(match src {
                    CopySrc::Value(value) => Stmt::Assign { param, value },
                    CopySrc::Temp(temp) => Stmt::AssignTemp { param, temp },
                });
            }
            None => {
                // Every pending destination is still read by another copy: a
                // cycle. Save one destination's pre-transfer value and retarget
                // its readers to the temp; that copy becomes emittable.
                let clobbered = pending[0].0;
                let temp = temps;
                temps += 1;
                out.push(Stmt::SaveTemp {
                    temp,
                    value: clobbered,
                });
                for (_, src) in pending.iter_mut() {
                    if *src == CopySrc::Value(clobbered) {
                        *src = CopySrc::Temp(temp);
                    }
                }
            }
        }
    }
    out
}

/// The arguments `from`'s terminator passes along its edge to `to`, or empty if
/// the edge carries none (or `from` has no argument-passing terminator).
fn edge_args(ctx: &Context, from: BlockId, to: BlockId) -> Vec<ValueId> {
    let block = BasicBlock::from_id(ctx, from);
    let Some(term) = block.iter().last().filter(|insn| insn.is_terminator()) else {
        return Vec::new();
    };
    match term.mnemonic() {
        Mnemonic::Branch(b) if b.target == to.local => b
            .args
            .iter()
            .map(|value| value.qualify(from.func))
            .collect(),
        Mnemonic::CBranch(c) if c.success_block == to.local => c
            .success_args
            .iter()
            .map(|value| value.qualify(from.func))
            .collect(),
        Mnemonic::CBranch(c) if c.failure_block == to.local => c
            .failure_args
            .iter()
            .map(|value| value.qualify(from.func))
            .collect(),
        // The first arm reaching `to`. `block_exit` only groups cases that agree
        // on their argument list, so any arm printed for `to` carries these.
        Mnemonic::Switch(sw) => sw
            .cases
            .iter()
            .find(|case| case.target == to.local)
            .map(|case| case.args.as_slice())
            .or_else(|| (sw.default == Some(to.local)).then_some(sw.default_args.as_slice()))
            .map(|args| args.iter().map(|v| v.qualify(from.func)).collect())
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Whether an instruction is a pure control-flow terminator that phase 1 replaces
/// with structured gotos (as opposed to a value/effect statement to keep).
///
/// An indirect branch qualifies only when its successors were actually
/// materialized (a resolved jump table): then the gotos to those successors say
/// everything the terminator did. An *unresolved* `branchind` has no successors,
/// so there is no goto to replace it with — dropping it would silently delete
/// the transfer of control (typically an indirect tail call, `jmp *%rax`). It is
/// kept as a body statement and rendered by the backend instead.
pub(crate) fn is_replaced_by_goto(insn: &InstructionRef<'_, '_>) -> bool {
    match insn.mnemonic() {
        Mnemonic::Branch(_) | Mnemonic::CBranch(_) | Mnemonic::Switch(_) => true,
        Mnemonic::BranchInd(_) => insn
            .block()
            .is_some_and(|block| block.successors().next().is_some()),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::emit_c;
    use wazabin_qcode_macro::qcode;

    /// A `switch` terminator lowers to a real multi-way statement, not one goto
    /// per successor. Cases reaching the same block share an arm, which is what
    /// makes a jump table with repeated slots read as `case a | b:`.
    #[test]
    fn switch_lowers_to_a_multiway_statement_with_shared_arms() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 idx;

            fn f:
            <entry>
                %i = load(idx:8, &idx);
                switch %i { 0x0 => <a_lbl>, 0x1 => <b_lbl>, 0x5 => <b_lbl>, 0x2 => <c_lbl> };
            <a_lbl>
                goto <0x1001>;
            <b_lbl>
                goto <0x1002>;
            <c_lbl>
                goto <0x1003>;
            "
        );

        let program = lower_function(&ctx, f);
        let c = emit_c(&ctx, &program, None);

        // The scrutinee is a real expression, and the dispatch is one statement.
        assert!(c.contains("match idx"), "expected a dispatch on idx:\n{c}");
        // 0x1 and 0x5 both reach `b_lbl`, so they share one arm rather than
        // producing two.
        assert!(
            c.contains("0x1 | 0x5 =>"),
            "cases reaching one block should share an arm:\n{c}"
        );
        assert!(c.contains("0x0 =>"), "expected the 0x0 arm:\n{c}");
        assert!(c.contains("0x2 =>"), "expected the 0x2 arm:\n{c}");
        // Every arm jumps to its own target: no arm is unreachable, unlike the
        // indirect fallback's straight-line goto list.
        for label in ["a_lbl", "b_lbl", "c_lbl"] {
            assert!(
                c.contains(&format!("goto {label}")),
                "expected a jump to {label}:\n{c}"
            );
        }
    }

    /// Projecting an aggregate names the field it takes. The field name lives in
    /// the aggregate's *type*, not in the instruction's operands, so the generic
    /// `opcode(args)` fallback rendered every projection of one value identically
    /// — losing which register each read.
    #[test]
    fn a_projection_names_its_field() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 OUT;

            fn f:
                <entry @x:i64>
                    %t = pack(RAX=@x, RDX=i64 0x7);
                    %hi = extract(%t.RDX);
                    store(OUT:8, &OUT <- %hi);
                    return at i64 0x0;
            "
        );

        let c = emit_c(&ctx, &lower_function(&ctx, f), None);
        assert!(
            c.contains(".RDX"),
            "the projection should name its field:\n{c}"
        );
        assert!(
            !c.contains("extract("),
            "the opaque pseudo-call fallback should be gone:\n{c}"
        );
    }

    /// A call nothing reads stays an unbound statement. The dense name counter
    /// must agree with that: it skips exactly the roots that introduce no name,
    /// so a call which *is* read has to be counted and one which is not must not
    /// be — otherwise two values land on the same letter and the second reads as
    /// a reassignment of the first.
    #[test]
    fn an_unread_call_result_is_not_bound() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 OUT;

            fn callee:
                <c_entry @x:i64>
                    return at i64 0x0;

            fn f:
                <entry>
                    %v = i64 0x10 + i64 0x20;
                    store(OUT:8, &OUT <- %v);
                    call fn callee(@x=%v) // -> <cont>;
                <cont>
                    return at i64 0x0;
            "
        );
        let _ = callee;

        let c = emit_c(&ctx, &lower_function(&ctx, f), None);
        assert!(c.contains("callee("), "expected the call:\n{c}");
        assert!(
            !c.contains("= callee("),
            "an unread call result must not be bound:\n{c}"
        );
    }

    /// A value defined inside a loop and read after it must be declared above the
    /// loop. Declared at its definition site it would sit in a scope the reader
    /// has already left, so the name dangles.
    #[test]
    fn a_value_escaping_its_loop_is_declared_above_it() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 N;
            varnode i32 OUT;

            fn f:
                <entry>
                    goto <head>;
                <head>
                    %n = load(N:4, &N);
                    %acc = %n * 0x3;
                    %c = %n == 0x0;
                    if %c goto <done> else goto <head>;
                <done>
                    store(OUT:4, &OUT <- %acc);
                    return at i64 0x0;
            "
        );

        let program = crate::structure::decompile_function(&ctx, f).expect("decompiles");
        let c = emit_c(&ctx, &program, None);

        // If `acc` is defined inside a loop body and read after it, its
        // declaration must appear before the loop, and its definition site must
        // then be a plain assignment rather than a second declaration.
        if let Some(decl) = c.find("_t acc;").or_else(|| c.find("_t a;")) {
            let assign = c.find("acc =").or_else(|| c.find("a ="));
            assert!(
                assign.is_some_and(|at| at > decl),
                "the bare declaration must precede the assignment:\n{c}"
            );
        }
        // Whatever the shape, no name may be used before it is introduced.
        assert!(!c.contains("= extract("), "stale extract rendering:\n{c}");
    }

    /// An unresolved indirect branch has no successors, so there is no goto that
    /// stands for it. It used to be dropped as "replaced by a goto" anyway,
    /// deleting the whole transfer of control — the indirect form of a tail call
    /// (`jmp *%rax`) simply vanished from the output.
    #[test]
    fn an_unresolved_indirect_branch_survives_lowering() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 dst;

            fn f:
            <entry>
                %p = load(dst:8, &dst);
                goto [i64 %p];
            "
        );

        let c = emit_c(&ctx, &lower_function(&ctx, f), None);
        assert!(
            c.contains("goto *dst"),
            "the indirect transfer must survive lowering:\n{c}"
        );
        assert!(
            !c.contains("branchind"),
            "the opaque fallback should be gone:\n{c}"
        );
    }

    /// A tail call reads as a return of the callee. Left to the generic
    /// `opcode(args)` renderer it printed as an SSA definition of a zero-width
    /// value, naming no callee at all — gcc emits these when it splits a cold arm
    /// into its own function and jumps to it.
    #[test]
    fn a_tail_call_reads_as_a_return_of_the_callee() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn helper:
                <h_entry>
                    return at i64 0x0;

            fn f:
                <entry>
                    tailcall fn helper();
            "
        );
        let _ = helper;

        let c = emit_c(&ctx, &lower_function(&ctx, f), None);
        assert!(
            c.contains("return helper("),
            "a tail call should name its callee:\n{c}"
        );
        assert!(
            !c.contains("tailcall("),
            "the opaque fallback should be gone:\n{c}"
        );
        assert!(
            !c.contains("uint0_t"),
            "a terminator must not render as a zero-width definition:\n{c}"
        );
    }

    #[test]
    fn conditional_lowers_to_gotos_with_real_condition() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;

            fn f:
            <entry>
                %c = load(cond:1, &cond);
                if %c goto <then_lbl> else goto <else_lbl>;
            <then_lbl>
                goto <0x1001>;
            <else_lbl>
                goto <0x1002>;
            "
        );

        let program = lower_function(&ctx, f);

        // Two `goto`s for the branch (if-goto + goto) plus the two tail gotos to
        // the out-of-function continuations.
        assert!(program.goto_count() >= 3, "expected several gotos");

        let c = emit_c(&ctx, &program, None);
        // The conditional renders the loaded variable as a real C expression,
        // not an opaque temp or a stringified predicate.
        assert!(
            c.contains("if (cond)"),
            "condition should be a real expr:\n{c}"
        );
        assert!(c.contains("goto"), "should contain gotos:\n{c}");
        // The entry label is emitted first, just inside the `fn` header.
        assert!(
            c.starts_with("fn f() {\n    entry:"),
            "entry label first inside the fn body:\n{c}"
        );
    }

    #[test]
    fn flat_lowering_carries_phi_copies_on_both_edges() {
        // The flat lowering is the always-correct fallback, so it too must copy
        // block-parameter arguments on every edge. A conditional's true-edge copy
        // must be guarded so it runs only when that edge is taken.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i32 x;

            fn f:
            <entry>
                %c = load(cond:1, &cond);
                if %c goto <succ @v=i32 0x1> else goto <fail @w=i32 0x2>;
            <succ @v:i32>
                store(x:4, &x <- @v);
                goto <0x2000>;
            <fail @w:i32>
                store(x:4, &x <- @w);
                goto <0x2001>;
            "
        );

        let program = lower_function(&ctx, f);
        let c = emit_c(&ctx, &program, None);
        // The true-edge copy is guarded by the branch condition; the false-edge
        // copy runs on fall-through.
        assert!(
            c.contains("if (cond) {") && c.contains("v = 0x1"),
            "true-edge phi copy should be guarded by the condition:\n{c}"
        );
        assert!(c.contains("w = 0x2"), "false-edge phi copy missing:\n{c}");
    }

    #[test]
    fn swapped_phi_copies_go_through_a_temp() {
        // The back edge passes `@a=@b @b=@a` — a parallel swap. Sequential
        // assignments `a = b; b = a;` would lose `a`; the cycle must be broken
        // with a temp: `t = a; a = b; b = t;` (in some order).
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i32 x;

            fn f:
            <entry>
                goto <head @a=i32 0x1 @b=i32 0x2>;
            <head @a:i32 @b:i32>
                %c = load(cond:1, &cond);
                if %c goto <head @a=@b @b=@a> else goto <exit_lbl>;
            <exit_lbl>
                store(x:4, &x <- @a);
                return at i64 0;
            "
        );

        let program = lower_function(&ctx, f);
        let c = emit_c(&ctx, &program, None);
        assert!(
            c.contains("phi_tmp0 = a;") || c.contains("phi_tmp0 = b;"),
            "the swap cycle should save one value into a temp:\n{c}"
        );
        assert!(
            c.contains("= phi_tmp0;"),
            "the cycle-closing copy should read the temp:\n{c}"
        );
        // Exactly one side of the swap reads the temp; the other reads its
        // partner directly. Both reading each other directly is the lost-value
        // bug this guards against.
        assert!(
            c.contains("a = b;") != c.contains("b = a;"),
            "exactly one direct copy plus one temp-closing copy expected:\n{c}"
        );
    }

    #[test]
    fn acyclic_phi_copies_are_ordered_without_a_temp() {
        // `@a=@b @b=%n` reads `b` before overwriting it, so plain ordering
        // (a = b first) suffices — no temp should appear.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i32 x;

            fn f:
            <entry>
                goto <head @a=i32 0x1 @b=i32 0x2>;
            <head @a:i32 @b:i32>
                %c = load(cond:1, &cond);
                %n = i32 @b + i32 0x1;
                if %c goto <head @a=@b @b=%n> else goto <exit_lbl>;
            <exit_lbl>
                store(x:4, &x <- @a);
                return at i64 0;
            "
        );

        let program = lower_function(&ctx, f);
        let c = emit_c(&ctx, &program, None);
        assert!(!c.contains("phi_tmp"), "acyclic copies need no temp:\n{c}");
        // The single-use increment folds into the copy, so the back edge reads
        // `a = b; b = b + 0x1;` — and that order is mandatory.
        let a_copy = c.find("a = b;").expect("a = b copy missing");
        let b_copy = c.find("b = b + 0x1;").expect("b = b + 0x1 copy missing");
        assert!(
            a_copy < b_copy,
            "`a = b` must run before `b` is overwritten:\n{c}"
        );
    }

    #[test]
    fn single_return_block_has_no_gotos() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn g:
            <entry>
                return at i64 0;
            "
        );
        let program = lower_function(&ctx, g);
        assert_eq!(program.goto_count(), 0, "a lone return needs no goto");
    }
}
