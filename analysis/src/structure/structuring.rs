//! Phase 2: region structuring via dominators and post-dominators.
//!
//! Turns runs of gotos into nested `if`/`if-else`/sequences and `while`/`do`
//! loops. The base algorithm (SAILR's, following DREAM/Phoenix): for a two-way
//! branch, the reconvergence point is its immediate post-dominator; the two
//! sides are structured recursively up to that merge. Natural loops (identified
//! by their back edges) are structured as an endless [`Stmt::Loop`] whose exit
//! and back edges become [`Stmt::Break`] / [`Stmt::Continue`]; a later
//! refinement rewrites the endless loop into a `while`/`do-while` when its shape
//! permits.
//!
//! Scope guard: functions with irreducible or improperly-overlapping loops fall
//! back to the phase-1 flat lowering ([`super::lower_function`]), so output is
//! always correct.

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
/// goto-based lowering for functions with irreducible control flow.
pub fn structure_function(ctx: &Context, function_id: FunctionId) -> Program {
    let function = qcode::value::Function::from_id(ctx, function_id);
    let Some(root) = function.root() else {
        return Program::default();
    };
    let root_id = root.id;

    let nodes = reachable(ctx, root_id);
    let node_set: HashSet<BlockId> = nodes.iter().copied().collect();
    let doms = compute_dominators(ctx, root_id);

    // Natural-loop identification. Irreducible or improperly-overlapping loops
    // are out of scope: fall back to the always-correct flat lowering.
    let Some(loops) = find_loops(ctx, &nodes, &node_set, &doms) else {
        return lower_function(ctx, function_id);
    };

    let exit_set = exit_blocks(ctx, &nodes, &node_set);
    let pdom = compute_postdominators(ctx, &nodes, &node_set, &exit_set);

    let labels = assign_labels(ctx, &nodes);
    let mut structurer = Structurer {
        ctx,
        node_set,
        pdom,
        loops,
        visited: HashSet::new(),
        frames: Vec::new(),
    };
    let (stmts, _) = structurer.region(root_id, None);
    Program { stmts, labels }
}

/// A natural loop: the blocks it comprises and the single block control leaves
/// to on exit (its `break` target), if it has one.
struct LoopInfo {
    body: HashSet<BlockId>,
    exit: Option<BlockId>,
}

/// A loop currently being structured, pushed while its body is walked so that
/// an edge back to the header becomes `continue` and an edge to the exit `break`.
#[derive(Clone, Copy)]
struct LoopFrame {
    header: BlockId,
    exit: Option<BlockId>,
}

struct Structurer<'ctx, 'a> {
    ctx: &'a Context<'ctx>,
    node_set: HashSet<BlockId>,
    /// Post-dominator sets: `pdom[n]` is every block that post-dominates `n`
    /// (including `n` itself).
    pdom: HashMap<BlockId, HashSet<BlockId>>,
    /// Natural loops keyed by header block.
    loops: HashMap<BlockId, LoopInfo>,
    /// Blocks already emitted, to guarantee termination and avoid duplication.
    visited: HashSet<BlockId>,
    /// The stack of enclosing loops (innermost last).
    frames: Vec<LoopFrame>,
}

impl Structurer<'_, '_> {
    /// Structures the region entered at `from`, stopping before `stop` (the
    /// enclosing merge point). Returns the statement list plus whether control
    /// falls through to `stop` (vs. terminating via break/continue/return/goto),
    /// so callers can decide whether to emit the continuation.
    fn region(&mut self, from: BlockId, stop: Option<BlockId>) -> (Vec<Stmt>, bool) {
        let mut out = Vec::new();
        let mut cur = Some(from);
        let mut fell_through = false;

        while let Some(b) = cur {
            // A structured exit of the innermost loop (continue/break) or an edge
            // out of it takes precedence over every other transition.
            if let Some(term) = self.loop_boundary(b) {
                out.push(term);
                break;
            }
            // The enclosing acyclic merge: control falls through to it.
            if Some(b) == stop {
                fell_through = true;
                break;
            }
            // A target outside the analyzed subgraph or an already-emitted join:
            // reference it by goto.
            if !self.node_set.contains(&b) || self.visited.contains(&b) {
                out.push(Stmt::Goto(b));
                break;
            }
            // A loop header reached for the first time: structure the whole loop
            // here, then continue from its exit.
            if self.loops.contains_key(&b) && !self.is_active(b) {
                let (loop_stmt, exit) = self.structure_loop(b, stop);
                out.push(loop_stmt);
                cur = exit;
                continue;
            }
            self.visited.insert(b);

            // A join point (>1 predecessor) may be a goto target, so label it —
            // except an active loop header, whose only back edges are continues.
            if self.predecessor_count(b) > 1 && !self.is_active(b) {
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
                    let (then, then_ft) = self.region(true_target, merge);
                    let (els, els_ft) = self.region(false_target, merge);
                    out.push(Stmt::If {
                        cond: lower_expr(self.ctx, condition),
                        then,
                        els,
                    });
                    // Continue at the merge only if some arm reaches it.
                    cur = if then_ft || els_ft { merge } else { None };
                }
                BlockExit::Indirect { edges } | BlockExit::Unstructured { edges } => {
                    for (_, target) in edges {
                        out.push(Stmt::Goto(target));
                    }
                    cur = None;
                }
            }
        }
        (out, fell_through)
    }

    /// Structures the natural loop headed at `header` as a [`Stmt::Loop`],
    /// returning it and the block control leaves to on exit.
    fn structure_loop(&mut self, header: BlockId, stop: Option<BlockId>) -> (Stmt, Option<BlockId>) {
        let exit = self.loops[&header].exit;
        self.frames.push(LoopFrame { header, exit });
        let (body, _) = self.region(header, stop);
        self.frames.pop();
        (refine_loop(body), exit)
    }

    /// If `b` is a boundary of the innermost enclosing loop, the structured
    /// transfer to emit for it: `continue` back to the header, `break` to the
    /// exit, or a `goto` for any other edge leaving the loop body.
    fn loop_boundary(&self, b: BlockId) -> Option<Stmt> {
        let frame = self.frames.last()?;
        if b == frame.header && self.visited.contains(&b) {
            return Some(Stmt::Continue);
        }
        if !self.loops[&frame.header].body.contains(&b) {
            return Some(if frame.exit == Some(b) {
                Stmt::Break
            } else {
                Stmt::Goto(b)
            });
        }
        None
    }

    /// Whether `b` is a loop header currently being structured (on the frame
    /// stack), which must not be re-entered as a fresh loop.
    fn is_active(&self, b: BlockId) -> bool {
        self.frames.iter().any(|f| f.header == b)
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

/// Rewrites an endless loop body into its most specific loop form. For now this
/// is the identity (always [`Stmt::Loop`]); a later refinement recognises
/// `while`/`do-while` shapes.
fn refine_loop(body: Vec<Stmt>) -> Stmt {
    Stmt::Loop { body }
}

/// Identifies the natural loops of the subgraph, keyed by header block.
///
/// Returns `None` if the loops are not properly nested (they overlap without one
/// containing the other) — a signature of irreducible control flow this phase
/// does not structure.
fn find_loops(
    ctx: &Context,
    nodes: &[BlockId],
    node_set: &HashSet<BlockId>,
    doms: &DominatorTree<BlockId>,
) -> Option<HashMap<BlockId, LoopInfo>> {
    // Group back-edge sources (latches) by the header they jump to. A back edge
    // is `u -> v` where `v` dominates `u`.
    let mut latches: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for &u in nodes {
        for (_, v) in BasicBlock::from_id(ctx, u).successors() {
            if node_set.contains(&v) && doms.dominates(v, u) {
                latches.entry(v).or_default().push(u);
            }
        }
    }

    let mut loops: HashMap<BlockId, LoopInfo> = HashMap::new();
    for (&header, latches) in &latches {
        let body = natural_loop_body(ctx, header, latches, node_set);
        let exit = loop_exit(ctx, &body, node_set);
        loops.insert(header, LoopInfo { body, exit });
    }

    // Every pair of loops must be nested or disjoint; overlap without
    // containment is irreducible.
    let bodies: Vec<&HashSet<BlockId>> = loops.values().map(|l| &l.body).collect();
    for (i, a) in bodies.iter().enumerate() {
        for b in &bodies[i + 1..] {
            let overlaps = a.intersection(b).next().is_some();
            if overlaps && !a.is_subset(b) && !b.is_subset(a) {
                return None;
            }
        }
    }
    Some(loops)
}

/// The natural loop body of the back edges into `header`: `header` plus every
/// block that can reach a latch without passing through `header`.
fn natural_loop_body(
    ctx: &Context,
    header: BlockId,
    latches: &[BlockId],
    node_set: &HashSet<BlockId>,
) -> HashSet<BlockId> {
    let mut body = HashSet::new();
    body.insert(header);
    let mut stack = Vec::new();
    for &l in latches {
        if body.insert(l) {
            stack.push(l);
        }
    }
    while let Some(n) = stack.pop() {
        for (_, p) in BasicBlock::from_id(ctx, n).predecessors() {
            if node_set.contains(&p) && body.insert(p) {
                stack.push(p);
            }
        }
    }
    body
}

/// The block a loop most commonly transfers to on exit — the target of the most
/// edges leaving its body — or `None` for an endless loop with no exit edges.
fn loop_exit(
    ctx: &Context,
    body: &HashSet<BlockId>,
    node_set: &HashSet<BlockId>,
) -> Option<BlockId> {
    let mut counts: HashMap<BlockId, usize> = HashMap::new();
    for &a in body {
        for (_, s) in BasicBlock::from_id(ctx, a).successors() {
            if node_set.contains(&s) && !body.contains(&s) {
                *counts.entry(s).or_default() += 1;
            }
        }
    }
    // Most exit edges wins; ties break to the smallest id for determinism.
    counts
        .into_iter()
        .max_by_key(|&(b, c)| (c, std::cmp::Reverse(Into::<usize>::into(b))))
        .map(|(b, _)| b)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{ast::Stmt, emit_c, lower_function};
    use qcode_macro::qcode;

    /// Whether any loop node appears anywhere in the statement tree.
    fn has_loop(stmts: &[Stmt]) -> bool {
        stmts.iter().any(|s| match s {
            Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::Loop { .. } => true,
            Stmt::If { then, els, .. } => has_loop(then) || has_loop(els),
            _ => false,
        })
    }

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
    fn pretested_loop_structures_without_gotos() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i32 x;

            fn f:
            <entry>
                goto <head>;
            <head>
                %c = load(i8, &cond);
                if %c goto <body> else goto <exit_lbl>;
            <body>
                store(&x, i32 0x1);
                goto <head>;
            <exit_lbl>
                local i64 ptr;
                return [ptr];
            "
        );

        let program = structure_function(&ctx, f);
        // The loop (header test, body, back edge, exit) structures with no gotos:
        // the back edge is a `continue`, the exit a `break`.
        assert_eq!(
            program.goto_count(),
            0,
            "loop should structure without gotos:\n{}",
            emit_c(&ctx, &program)
        );
        assert!(
            has_loop(&program.stmts),
            "expected a loop node:\n{:#?}",
            program.stmts
        );
        // And it no longer degrades to the flat lowering.
        let flat = lower_function(&ctx, f);
        assert!(program.goto_count() < flat.goto_count());
    }

    #[test]
    fn posttested_single_block_loop_structures() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i32 x;

            fn f:
            <entry>
                goto <head>;
            <head>
                store(&x, i32 0x1);
                %c = load(i8, &cond);
                if %c goto <head> else goto <exit_lbl>;
            <exit_lbl>
                local i64 ptr;
                return [ptr];
            "
        );

        let program = structure_function(&ctx, f);
        assert_eq!(
            program.goto_count(),
            0,
            "self-looping block should structure without gotos:\n{}",
            emit_c(&ctx, &program)
        );
        assert!(
            has_loop(&program.stmts),
            "expected a loop node:\n{:#?}",
            program.stmts
        );
    }

    #[test]
    fn nested_loops_structure_without_gotos() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 outer_c;
            varnode i8 inner_c;
            varnode i32 x;

            fn f:
            <entry>
                goto <outer>;
            <outer>
                %oc = load(i8, &outer_c);
                if %oc goto <inner> else goto <exit_lbl>;
            <inner>
                %ic = load(i8, &inner_c);
                if %ic goto <inner_body> else goto <outer>;
            <inner_body>
                store(&x, i32 0x1);
                goto <inner>;
            <exit_lbl>
                local i64 ptr;
                return [ptr];
            "
        );

        let program = structure_function(&ctx, f);
        assert_eq!(
            program.goto_count(),
            0,
            "nested loops should structure without gotos:\n{}",
            emit_c(&ctx, &program)
        );
        assert!(
            has_loop(&program.stmts),
            "expected loop nodes:\n{:#?}",
            program.stmts
        );
    }
}
