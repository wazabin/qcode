//! Phase 2: acyclic region structuring via dominators and post-dominators.
//!
//! Turns runs of gotos into nested `if`/`if-else`/sequences. The base algorithm
//! (SAILR's, following DREAM/Phoenix): for a two-way branch, the reconvergence
//! point is its immediate post-dominator; the two sides are structured
//! recursively up to that merge, then emission continues from the merge.
//!
//! Scope guard: functions containing a loop (a CFG back edge) or that fail to
//! structure fall back to the phase-1 flat lowering ([`super::lower_function`]),
//! so output is always correct. Loops are a later phase.

use std::collections::{HashMap, HashSet};

use jstd::graph::analysis::{DominatorTree, compute_dominators, compute_postdominators};
use qcode::{
    context::Context,
    value::{BasicBlock, BlockId, function::FunctionId},
};

use super::{
    BlockExit,
    ast::{Program, Stmt},
    block_exit,
    lower::{assign_labels, is_replaced_by_goto},
    lower_expr::lower_expr,
    lower_function,
};

/// Structures `function_id` into nested control flow, falling back to flat
/// goto-based lowering for functions with loops or irreducible control flow.
pub fn structure_function(ctx: &Context, function_id: FunctionId) -> Program {
    let function = qcode::value::Function::from_id(ctx, function_id);
    let Some(root) = function.root() else {
        return Program::default();
    };
    let root_id = root.id;

    let nodes = reachable(ctx, root_id);
    let node_set: HashSet<BlockId> = nodes.iter().copied().collect();
    let doms = compute_dominators(ctx, root_id);

    // Loops are out of scope for phase 2: fall back to the flat lowering, which
    // handles any CFG correctly (if uglily).
    if has_back_edge(ctx, &nodes, &node_set, &doms) {
        return lower_function(ctx, function_id);
    }

    let exit_set = exit_blocks(ctx, &nodes, &node_set);
    let pdom = compute_postdominators(ctx, &nodes, &node_set, &exit_set);

    let labels = assign_labels(ctx, &nodes);
    let mut structurer = Structurer {
        ctx,
        node_set,
        pdom,
        visited: HashSet::new(),
    };
    let stmts = structurer.region(root_id, None);
    Program { stmts, labels }
}

struct Structurer<'ctx, 'a> {
    ctx: &'a Context<'ctx>,
    node_set: HashSet<BlockId>,
    /// Post-dominator sets: `pdom[n]` is every block that post-dominates `n`
    /// (including `n` itself).
    pdom: HashMap<BlockId, HashSet<BlockId>>,
    /// Blocks already emitted, to guarantee termination and avoid duplication.
    visited: HashSet<BlockId>,
}

impl Structurer<'_, '_> {
    /// Structures the region entered at `from`, stopping before `stop` (the
    /// enclosing merge point). Returns the statement list for that region.
    fn region(&mut self, from: BlockId, stop: Option<BlockId>) -> Vec<Stmt> {
        let mut out = Vec::new();
        let mut cur = Some(from);

        while let Some(b) = cur {
            if Some(b) == stop {
                break;
            }
            // A target outside the analyzed subgraph (e.g. an inter-function
            // continuation) or an already-emitted join: reference it by goto.
            if !self.node_set.contains(&b) || self.visited.contains(&b) {
                out.push(Stmt::Goto(b));
                break;
            }
            self.visited.insert(b);

            // A join point (>1 predecessor) may be a goto target, so label it.
            if self.predecessor_count(b) > 1 {
                out.push(Stmt::Label(b));
            }
            self.emit_body(b, &mut out);

            match block_exit(BasicBlock::from_id(self.ctx, b)) {
                BlockExit::Return => cur = None,
                BlockExit::Goto { target, .. } => cur = Some(target),
                BlockExit::Branch {
                    condition,
                    true_target,
                    false_target,
                    ..
                } => {
                    let merge = self.immediate_postdom(b);
                    let then = self.branch_arm(true_target, merge);
                    let els = self.branch_arm(false_target, merge);
                    out.push(Stmt::If {
                        cond: lower_expr(self.ctx, condition),
                        then,
                        els,
                    });
                    cur = merge;
                }
                BlockExit::Indirect { edges } | BlockExit::Unstructured { edges } => {
                    for (_, target) in edges {
                        out.push(Stmt::Goto(target));
                    }
                    cur = None;
                }
            }
        }
        out
    }

    /// Structures one arm of a branch. An arm that jumps straight to the merge
    /// contributes no statements (the empty else, or an if-then guard).
    fn branch_arm(&mut self, target: BlockId, merge: Option<BlockId>) -> Vec<Stmt> {
        if Some(target) == merge {
            Vec::new()
        } else {
            self.region(target, merge)
        }
    }

    /// Emits a block's verbatim body: every instruction except the pure
    /// control-flow terminators, which the caller turns into structured flow.
    fn emit_body(&self, block: BlockId, out: &mut Vec<Stmt>) {
        let block = BasicBlock::from_id(self.ctx, block);
        for insn in block.instructions() {
            if !is_replaced_by_goto(&insn) {
                out.push(Stmt::Raw(insn.id));
            }
        }
    }

    /// The immediate post-dominator of `b` (its reconvergence point), or `None`
    /// when its paths diverge to different exits with no common post-dominator.
    ///
    /// Derived from the post-dominator sets: among the strict post-dominators of
    /// `b`, the immediate one is post-dominated by all the others.
    fn immediate_postdom(&self, b: BlockId) -> Option<BlockId> {
        let set = self.pdom.get(&b)?;
        let strict: Vec<BlockId> = set.iter().copied().filter(|&x| x != b).collect();
        strict
            .iter()
            .copied()
            .find(|&p| strict.iter().all(|&q| self.pdom[&p].contains(&q)))
    }

    fn predecessor_count(&self, b: BlockId) -> usize {
        BasicBlock::from_id(self.ctx, b)
            .predecessors()
            .filter(|(_, p)| self.node_set.contains(p))
            .count()
    }
}

/// The blocks reachable from `root`, in DFS order.
fn reachable(ctx: &Context, root: BlockId) -> Vec<BlockId> {
    jstd::graph::analysis::reachable_from_root(ctx, root)
}

/// Blocks with no successor inside the analyzed subgraph (returns and
/// inter-function tails) — the post-dominator analysis's exit set.
fn exit_blocks(ctx: &Context, nodes: &[BlockId], node_set: &HashSet<BlockId>) -> HashSet<BlockId> {
    nodes
        .iter()
        .copied()
        .filter(|&b| {
            BasicBlock::from_id(ctx, b)
                .successors()
                .all(|(_, s)| !node_set.contains(&s))
        })
        .collect()
}

/// Whether the subgraph has a back edge: an edge `u -> v` where `v` dominates
/// `u`. Such an edge is a loop, which phase 2 does not structure.
fn has_back_edge(
    ctx: &Context,
    nodes: &[BlockId],
    node_set: &HashSet<BlockId>,
    doms: &DominatorTree<BlockId>,
) -> bool {
    nodes.iter().any(|&u| {
        BasicBlock::from_id(ctx, u)
            .successors()
            .filter(|(_, v)| node_set.contains(v))
            .any(|(_, v)| doms.dominates(v, u))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{ast::Stmt, emit_c, lower_function};
    use qcode_macro::qcode;

    #[test]
    fn if_then_else_structures_to_nested_if() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i32 x;

            fn f:
            <entry>
                %c = load(i8, &cond);
                if %c goto <then_lbl> else goto <else_lbl>;
            <then_lbl>
                store(&x, i32 0x1);
                goto <merge>;
            <else_lbl>
                store(&x, i32 0x2);
                goto <merge>;
            <merge>
                local i64 ptr;
                return [ptr];
            "
        );

        let program = structure_function(&ctx, f);
        // The branch is fully structured: no gotos remain (the merge is inlined).
        assert_eq!(
            program.goto_count(),
            0,
            "if/else should erase gotos:\n{}",
            emit_c(&ctx, &program)
        );
        // Top level contains an `If` node with both arms populated.
        let has_if = program.stmts.iter().any(
            |s| matches!(s, Stmt::If { then, els, .. } if !then.is_empty() && !els.is_empty()),
        );
        assert!(
            has_if,
            "expected a populated if/else:\n{:#?}",
            program.stmts
        );
    }

    #[test]
    fn if_then_structures_without_else() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i32 x;

            fn f:
            <entry>
                %c = load(i8, &cond);
                if %c goto <then_lbl> else goto <merge>;
            <then_lbl>
                store(&x, i32 0x1);
                goto <merge>;
            <merge>
                local i64 ptr;
                return [ptr];
            "
        );

        let program = structure_function(&ctx, f);
        assert_eq!(
            program.goto_count(),
            0,
            "if-then should erase gotos:\n{}",
            emit_c(&ctx, &program)
        );
        let if_then = program
            .stmts
            .iter()
            .any(|s| matches!(s, Stmt::If { els, .. } if els.is_empty()));
        assert!(
            if_then,
            "expected an if-then (empty else):\n{:#?}",
            program.stmts
        );
    }

    #[test]
    fn structuring_beats_flat_lowering_on_gotos() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i32 x;

            fn f:
            <entry>
                %c = load(i8, &cond);
                if %c goto <then_lbl> else goto <else_lbl>;
            <then_lbl>
                store(&x, i32 0x1);
                goto <merge>;
            <else_lbl>
                store(&x, i32 0x2);
                goto <merge>;
            <merge>
                local i64 ptr;
                return [ptr];
            "
        );

        let flat = lower_function(&ctx, f).goto_count();
        let structured = structure_function(&ctx, f).goto_count();
        assert!(
            structured < flat,
            "structuring should reduce gotos ({structured} !< {flat})"
        );
    }

    #[test]
    fn loop_falls_back_to_flat_lowering() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;

            fn f:
            <entry>
                goto <head>;
            <head>
                %c = load(i8, &cond);
                if %c goto <body> else goto <exit_lbl>;
            <body>
                goto <head>;
            <exit_lbl>
                local i64 ptr;
                return [ptr];
            "
        );

        // Back edge body -> head is detected; output equals the flat lowering.
        let structured = structure_function(&ctx, f);
        let flat = lower_function(&ctx, f);
        assert_eq!(emit_c(&ctx, &structured), emit_c(&ctx, &flat));
    }
}
