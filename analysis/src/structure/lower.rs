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
    value::{BasicBlock, BlockId, BlockRef, FunctionId, InstructionRef, insn::Mnemonic},
};

use super::{BlockExit, ast::Program, ast::Stmt, block_exit, lower_expr::lower_expr};

/// Lowers `function_id` into a flat, goto-based [`Program`].
///
/// Blocks are emitted in the function's block order (address-sorted), with the
/// entry block first. Returns an empty program if the function has no root.
pub fn lower_function(ctx: &Context, function_id: FunctionId) -> Program {
    let function = qcode::value::Function::from_id(ctx, function_id);
    let Some(root) = function.root() else {
        return Program::default();
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

    Program { stmts, labels }
}

/// Assigns each block a stable label name: its IR name if it has one, otherwise
/// a synthesized `bb_<addr>` / `bb_<id>`.
fn assign_labels(ctx: &Context, order: &[BlockId]) -> HashMap<BlockId, String> {
    let mut labels = HashMap::with_capacity(order.len());
    for &block_id in order {
        let block = BasicBlock::from_id(ctx, block_id);
        let name = match (block.name(), block.address()) {
            (Some(name), _) => name.to_string(),
            (None, Some(addr)) => format!("bb_{addr:x}"),
            (None, None) => format!("bb_{}", Into::<usize>::into(block_id)),
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

    match block_exit(block) {
        // The `return` instruction is a verbatim body statement (kept above);
        // it needs no goto.
        BlockExit::Return => {}
        BlockExit::Goto { target, .. } => out.push(Stmt::Goto(target)),
        BlockExit::Branch {
            condition,
            true_target,
            false_target,
            ..
        } => {
            // `if (cond) goto TRUE; goto FALSE;`
            out.push(Stmt::GotoIf {
                cond: lower_expr(ctx, condition),
                target: true_target,
            });
            out.push(Stmt::Goto(false_target));
        }
        // Indirect / unstructured exits: preserve reachability with a goto to
        // every successor. Lossy but keeps the baseline correct-by-construction.
        BlockExit::Indirect { edges } | BlockExit::Unstructured { edges } => {
            for (_, target) in edges {
                out.push(Stmt::Goto(target));
            }
        }
    }
}

/// Whether an instruction is a pure control-flow terminator that phase 1 replaces
/// with structured gotos (as opposed to a value/effect statement to keep).
fn is_replaced_by_goto(insn: &InstructionRef<'_, '_>) -> bool {
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
                %c = load(i8, &cond);
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
        // The conditional renders the dereferenced load as a real C expression,
        // not an opaque temp or a stringified predicate.
        assert!(c.contains("if (*"), "condition should be a real expr:\n{c}");
        assert!(c.contains("goto"), "should contain gotos:\n{c}");
        // The entry label is emitted first.
        assert!(c.starts_with("entry:"), "entry label first:\n{c}");
    }

    #[test]
    fn single_return_block_has_no_gotos() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn g:
            <entry>
                local i64 ptr;
                return [ptr];
            "
        );
        let program = lower_function(&ctx, g);
        assert_eq!(program.goto_count(), 0, "a lone return needs no goto");
    }
}
