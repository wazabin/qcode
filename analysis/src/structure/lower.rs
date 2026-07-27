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

use super::{BlockExit, ast::Program, ast::Stmt, block_exit, lower_expr::lower_expr};

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
        _ => Vec::new(),
    }
}

/// Whether an instruction is a pure control-flow terminator that phase 1 replaces
/// with structured gotos (as opposed to a value/effect statement to keep).
pub(crate) fn is_replaced_by_goto(insn: &InstructionRef<'_, '_>) -> bool {
    matches!(
        insn.mnemonic(),
        Mnemonic::Branch(_) | Mnemonic::CBranch(_) | Mnemonic::BranchInd(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::emit_c;
    use qcode_macro::qcode;

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

        let c = emit_c(&ctx, &program);
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
        let c = emit_c(&ctx, &program);
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
        let c = emit_c(&ctx, &program);
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
        let c = emit_c(&ctx, &program);
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
