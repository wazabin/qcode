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

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use jstd::graph::analysis::{DominatorTree, compute_dominators, compute_postdominators};
use qcode::{
    context::Context,
    value::{BasicBlock, BlockId, InstructionId, function::FunctionId},
};

use crate::pipeline::{DecompilePass, RegisteredPass, make_pass};

use super::{
    BlockExit,
    ast::{Program, Stmt, SwitchCase},
    block_exit,
    lower::{assign_labels, block_arg_moves, is_replaced_by_goto},
    lower_expr::lower_expr,
    lower_function,
};

/// The default decompilation pass pipeline, in run order: structure the CFG into
/// an AST, validate that structuring dropped no code (falling back to flat
/// lowering when it did), refine its loops, then recover switches. Validation
/// runs before the switch pass, which intentionally elides comparison setup that
/// the coverage check would otherwise read as dropped. SAILR deopt passes slot
/// in here as they land.
const DECOMPILE_PIPELINE: &[&str] = &[
    "structure",
    "validate_structuring",
    "refine_loops",
    "recover_switch",
    // Last, as SAILR prescribes for cross-jump reversion: it is the most
    // aggressive deopt, and running it earlier destroys the shapes the passes
    // above recognize — duplicating a shared default is exactly what stops
    // `recover_switch` seeing a comparison tree.
    "cross_jump_revert",
    "validate_labels",
];

/// The passes a speculative candidate must be put through before its goto count
/// means anything: whatever runs between structuring and the deopt.
const RESTRUCTURE_CHAIN: &[&str] = &["refine_loops", "recover_switch"];

/// Decompiles `function_id` to a high-level [`Program`] by running the
/// decompilation pass pipeline over it. Each pass reads the (immutable) qcode IR
/// and rewrites the shared AST.
///
/// Returns `Err` if a pass reports an error, so callers (notably the GUI, which
/// decompiles every function up front) can degrade a single failing function to
/// a placeholder instead of aborting the whole binary.
pub fn decompile_function(ctx: &Context, function_id: FunctionId) -> Result<Program, String> {
    let mut program = Program::default();
    for &name in DECOMPILE_PIPELINE {
        match make_pass(name) {
            Some(RegisteredPass::Decompile(pass)) => {
                pass.run(ctx, function_id, &mut program)
                    .map_err(|e| format!("decompile pass `{name}`: {e}"))?;
            }
            Some(_) => {
                return Err(format!(
                    "pipeline pass `{name}` is not a decompilation pass"
                ));
            }
            None => return Err(format!("unknown decompilation pass `{name}`")),
        }
    }
    Ok(program)
}

/// The region-structuring decompilation pass: builds the initial AST from qcode.
#[derive(Default)]
pub struct Structure;

impl DecompilePass for Structure {
    const NAME: &'static str = "structure";

    fn description(&self) -> &'static str {
        "region structuring: nested if / loops recovered from the CFG"
    }

    fn run(
        &self,
        ctx: &Context,
        fun_id: FunctionId,
        program: &mut Program,
    ) -> Result<bool, String> {
        *program = structure_regions(ctx, fun_id);
        Ok(true)
    }
}

crate::register_decompile_pass!(Structure);

/// Post-structuring validation: the structured AST must not silently drop code
/// or jump to a label it never prints — the two ways region structuring can
/// produce misleading pseudo-C (a dropped secondary loop exit, a jump-table arm
/// whose case bodies were never walked). When either invariant fails the
/// structured output is unsound, so this pass replaces it with the
/// always-correct flat lowering rather than emit code that lies.
#[derive(Default)]
pub struct ValidateStructuring;

impl DecompilePass for ValidateStructuring {
    const NAME: &'static str = "validate_structuring";

    fn description(&self) -> &'static str {
        "validate structured output; fall back to flat lowering if code was dropped"
    }

    fn run(
        &self,
        ctx: &Context,
        fun_id: FunctionId,
        program: &mut Program,
    ) -> Result<bool, String> {
        if validate_program(ctx, fun_id, program, true).is_err() {
            *program = lower_function(ctx, fun_id);
            return Ok(true);
        }
        Ok(false)
    }
}

crate::register_decompile_pass!(ValidateStructuring);

/// How many duplications to accept before stopping. Each one is re-structured
/// and measured, so the cap bounds work as well as output growth. Matches the
/// default angr's `CrossJumpReverter` uses.
const MAX_DUPLICATIONS: usize = 3;

/// SAILR cross-jump reversal: undo the compiler's code deduplication.
///
/// A shared tail — the `default:` arm of a switch reached from several paths, a
/// common epilogue — is emitted once by the compiler and jumped to from
/// everywhere else. Region structuring cannot place it: a block with N
/// predecessors has no single nesting position all N reach, so all but one path
/// gets a `goto`, often *into* a nested block. No amount of better region
/// selection fixes that, because the shape is an artifact of the optimization,
/// not of the source.
///
/// The reversal is duplication: give each path its own copy and the gotos
/// dissolve. Here that needs no CFG surgery — [`Structurer`] emits a goto
/// precisely when a block is already `visited`, so permitting a second emission
/// *is* the duplication (see [`Structurer::permit_dup`]).
///
/// Speculative, as in SAILR: each candidate is applied, the function
/// re-structured, and the result kept only if [`Program::goto_count`] strictly
/// drops. A duplication that does not pay for itself is reverted, so the pass
/// cannot make output worse by its own measure.
#[derive(Default)]
pub struct CrossJumpRevert;

impl DecompilePass for CrossJumpRevert {
    const NAME: &'static str = "cross_jump_revert";

    fn description(&self) -> &'static str {
        "SAILR deopt: duplicate shared tails to undo compiler code deduplication"
    }

    fn run(
        &self,
        ctx: &Context,
        fun_id: FunctionId,
        program: &mut Program,
    ) -> Result<bool, String> {
        let mut permit: HashSet<BlockId> = HashSet::default();
        let mut best_gotos = program.goto_count();
        let mut changed = false;

        for _ in 0..MAX_DUPLICATIONS {
            // Re-read the candidates each round: duplicating one tail can expose
            // or remove others.
            let mut targets = Vec::new();
            collect_goto_targets(&program.stmts, &mut targets);
            targets.retain(|b| !permit.contains(b));
            targets.dedup();
            if targets.is_empty() {
                break;
            }

            let Some((winner, candidate)) = targets
                .into_iter()
                .filter_map(|target| {
                    permit.insert(target);
                    let candidate = restructure_with(ctx, fun_id, &permit).ok();
                    permit.remove(&target);
                    let candidate = candidate?;
                    (candidate.goto_count() < best_gotos).then_some((target, candidate))
                })
                .min_by_key(|(_, candidate)| candidate.goto_count())
            else {
                // Nothing left that pays for itself.
                break;
            };

            permit.insert(winner);
            best_gotos = candidate.goto_count();
            *program = candidate;
            changed = true;
        }

        if changed {
            // Duplication removes the jumps that made a label necessary; left
            // behind it would be dead text, and the same label printed twice
            // reads to `validate_labels` as a dangling jump.
            let mut targets = Vec::new();
            collect_goto_targets(&program.stmts, &mut targets);
            let used: HashSet<BlockId> = targets.into_iter().collect();
            prune_unused_labels(&mut program.stmts, &used);
        }
        Ok(changed)
    }
}

crate::register_decompile_pass!(CrossJumpRevert);

/// Re-structure with `permit_dup`, then replay the passes that normally run
/// between structuring and this one.
///
/// Measuring a candidate on raw structuring alone would compare it against a
/// program those passes have already improved, and would miss that a duplication
/// can *destroy* what they recognize: splitting a shared default removes the very
/// shape `recover_switch` folds into a `match`, trading a goto for a whole
/// switch. Putting the candidate through the same chain makes the comparison
/// honest.
fn restructure_with(
    ctx: &Context,
    fun_id: FunctionId,
    permit_dup: &HashSet<BlockId>,
) -> Result<Program, String> {
    let mut program = structure_regions_with(ctx, fun_id, permit_dup);
    for &name in RESTRUCTURE_CHAIN {
        match make_pass(name) {
            Some(RegisteredPass::Decompile(pass)) => {
                pass.run(ctx, fun_id, &mut program)?;
            }
            _ => return Err(format!("`{name}` is not a decompilation pass")),
        }
    }
    Ok(program)
}

/// Final label-consistency validation, run after the switch pass. Switch recovery
/// absorbs a shared default block's label into the `match`; if some other part of
/// the function still jumps to that label, the jump would dangle. This re-checks
/// goto/label consistency (but not coverage — the switch pass legitimately elides
/// the comparison-tree setup that coverage counts) and falls back to the flat
/// lowering if a later pass orphaned a label.
#[derive(Default)]
pub struct ValidateLabels;

impl DecompilePass for ValidateLabels {
    const NAME: &'static str = "validate_labels";

    fn description(&self) -> &'static str {
        "validate goto/label consistency after switch recovery"
    }

    fn run(
        &self,
        ctx: &Context,
        fun_id: FunctionId,
        program: &mut Program,
    ) -> Result<bool, String> {
        if validate_program(ctx, fun_id, program, false).is_err() {
            *program = lower_function(ctx, fun_id);
            return Ok(true);
        }
        Ok(false)
    }
}

crate::register_decompile_pass!(ValidateLabels);

/// Checks the structural invariants the emitter relies on:
///
/// 1. **Coverage** (only when `coverage`) — every instruction of every reachable
///    block that is not replaced by a goto appears in the output. Checking whole
///    blocks instead was too coarse: a block whose ordinary body was emitted
///    counted as covered even when its *terminator* was dropped, which is exactly
///    how an unresolved indirect tail call used to vanish without a trace.
///    Skipped after switch recovery, which legitimately elides the comparison
///    setup that coverage would read as dropped.
/// 2. **Goto/label consistency** — every `goto` to an in-subgraph block targets
///    a label that is actually printed. A goto to a suppressed label is a jump
///    into the void.
///
/// (Gotos to blocks outside the analyzed subgraph — inter-function tails — carry
/// no label by design and are exempt.)
fn validate_program(
    ctx: &Context,
    function_id: FunctionId,
    program: &Program,
    coverage: bool,
) -> Result<(), String> {
    let function = qcode::value::FunctionRef::from_id(ctx, function_id);
    let Some(root) = function.root() else {
        return Ok(());
    };
    let node_set: HashSet<BlockId> = reachable(ctx, root.id).into_iter().collect();

    let mut labels = HashSet::default();
    let mut gotos = HashSet::default();
    let mut covered = HashSet::default();
    collect_validation(&program.stmts, &mut labels, &mut gotos, &mut covered);

    if coverage {
        for &b in &node_set {
            for insn in BasicBlock::from_id(ctx, b).instructions() {
                if !is_replaced_by_goto(&insn) && !covered.contains(&insn.id) {
                    return Err(format!(
                        "instruction {:?} of block {b:?} was dropped from structured output",
                        insn.id
                    ));
                }
            }
        }
    }
    for &g in &gotos {
        if node_set.contains(&g) && !labels.contains(&g) {
            return Err(format!(
                "goto targets in-subgraph block {g:?} with no printed label"
            ));
        }
    }
    Ok(())
}

/// Walks the statement tree, recording every emitted label, every goto target,
/// and every block whose body was emitted (via a [`Stmt::Raw`], or — once a
/// switch has folded its comparison tree — via the case's recorded provenance).
fn collect_validation(
    stmts: &[Stmt],
    labels: &mut HashSet<BlockId>,
    gotos: &mut HashSet<BlockId>,
    covered: &mut HashSet<InstructionId>,
) {
    let cover_insn = |id, covered: &mut HashSet<InstructionId>| {
        covered.insert(id);
    };
    for stmt in stmts {
        match stmt {
            Stmt::Label(b) => {
                labels.insert(*b);
            }
            Stmt::Goto(b) | Stmt::GotoIf { target: b, .. } => {
                gotos.insert(*b);
            }
            Stmt::Raw(id) => cover_insn(*id, covered),
            Stmt::If { then, els, .. } => {
                collect_validation(then, labels, gotos, covered);
                collect_validation(els, labels, gotos, covered);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::Loop { body } => {
                collect_validation(body, labels, gotos, covered);
            }
            Stmt::Switch { cases, default, .. } => {
                for case in cases {
                    // The equality tests a case folded away belong to the blocks
                    // they came from; count them as covered.
                    for &id in &case.insns {
                        cover_insn(id, covered);
                    }
                    collect_validation(&case.body, labels, gotos, covered);
                }
                collect_validation(default, labels, gotos, covered);
            }
            Stmt::Assign { .. }
            | Stmt::SaveTemp { .. }
            | Stmt::AssignTemp { .. }
            | Stmt::Break
            | Stmt::Continue => {}
        }
    }
}

/// Structures `function_id` into nested control flow (with loops as endless
/// [`Stmt::Loop`]s), falling back to flat goto-based lowering for functions with
/// irreducible control flow. The [`Structure`] pass's implementation.
fn structure_regions(ctx: &Context, function_id: FunctionId) -> Program {
    structure_regions_limited(ctx, function_id, MAX_REGION_DEPTH, &HashSet::default())
}

/// [`structure_regions`] with a set of *duplication roots*: blocks the structurer
/// may emit a second time rather than refer to by `goto`.
///
/// Each root is expanded to the subtree it dominates, because duplicating a
/// block means duplicating its whole region. Re-emitting only the header would
/// walk straight into its already-visited successors and emit a fresh goto for
/// each — trading one jump for several. Dominance is the right closure: a block
/// the root dominates is reachable only through it, so copying it alongside the
/// root cannot strand any other path.
fn structure_regions_with(
    ctx: &Context,
    function_id: FunctionId,
    permit_dup: &HashSet<BlockId>,
) -> Program {
    structure_regions_limited(ctx, function_id, MAX_REGION_DEPTH, permit_dup)
}

/// [`structure_regions`] with an explicit region-depth limit, so tests can drive
/// the deep-cascade fallback without materializing thousands of blocks.
fn structure_regions_limited(
    ctx: &Context,
    function_id: FunctionId,
    max_depth: usize,
    permit_dup: &HashSet<BlockId>,
) -> Program {
    let function = qcode::value::FunctionRef::from_id(ctx, function_id);
    let Some(root) = function.root() else {
        return Program {
            function: Some(function_id),
            ..Program::default()
        };
    };
    let root_id = root.id;

    // A tail-call edge is a real CFG edge, so whole-program reachability from the
    // root pulls in blocks owned by the callee. Restrict the region to blocks this
    // function owns: otherwise the structured path inlines the entire callee body
    // into the caller, and the flat fallback emits `goto bb_N;` into a block that
    // has no matching label. Foreign successors then fall outside `node_set` and
    // are correctly treated as region exits.
    let nodes: Vec<BlockId> = reachable(ctx, root_id)
        .into_iter()
        .filter(|&b| b.func == function_id)
        .collect();
    let node_set: HashSet<BlockId> = nodes.iter().copied().collect();
    let doms = compute_dominators(&function, root_id);

    // Natural-loop identification. Irreducible or improperly-overlapping loops
    // are out of scope: fall back to the always-correct flat lowering.
    let Some(loops) = find_loops(ctx, &nodes, &node_set, &doms) else {
        return lower_function(ctx, function_id);
    };

    let exit_set = exit_blocks(ctx, &nodes, &node_set);
    let pdom = compute_postdominators(&function, &nodes, &node_set, &exit_set);

    // Expand each duplication root to the blocks it dominates (see
    // `structure_regions_with`).
    let permit_dup: HashSet<BlockId> = if permit_dup.is_empty() {
        HashSet::default()
    } else {
        nodes
            .iter()
            .copied()
            .filter(|&b| permit_dup.iter().any(|&root| doms.dominates(root, b)))
            .collect()
    };

    let labels = assign_labels(ctx, &nodes);
    let mut structurer = Structurer {
        ctx,
        node_set,
        pdom,
        loops,
        visited: HashSet::default(),
        permit_dup,
        frames: Vec::new(),
        max_depth,
        overflowed: false,
    };
    let (mut stmts, _) = structurer.region(root_id, None, 0);

    // Emit continuation regions for virtualized edges. Structuring turns an edge
    // it cannot fit into a schema (most commonly a loop's secondary exit) into a
    // `goto`; a block reachable *only* through such an edge is never visited by
    // the main walk, so its body would be dropped and the goto would dangle —
    // demoting the whole function to flat lowering at validation. Instead, give
    // every such goto a home: append each unvisited in-subgraph goto target as a
    // labeled top-level region, repeating until no new targets appear (a
    // continuation can itself virtualize further edges).
    loop {
        if structurer.overflowed {
            break;
        }
        let mut targets = Vec::new();
        collect_goto_targets(&stmts, &mut targets);
        targets.retain(|b| structurer.node_set.contains(b) && !structurer.visited.contains(b));
        targets.dedup();
        if targets.is_empty() {
            break;
        }
        for b in targets {
            // An earlier continuation in this batch may already have emitted it.
            if structurer.visited.contains(&b) {
                continue;
            }
            // `region` labels join blocks itself; a single-predecessor target
            // (reachable only through the virtualized edge) needs the label here.
            if structurer.predecessor_count(b) <= 1 {
                stmts.push(Stmt::Label(b));
            }
            let (cont, _) = structurer.region(b, None, 0);
            stmts.extend(cont);
        }
    }

    // A pathologically deep region (a multi-thousand-arm comparison cascade)
    // would recurse deep enough to overflow the stack — which no `catch_unwind`
    // can recover. When the guard trips, discard the partial structuring and fall
    // back to the flat lowering, whose emission is iterative and always correct.
    if structurer.overflowed {
        return lower_function(ctx, function_id);
    }

    // The walk labels every join block (>1 predecessors) defensively, but a join
    // whose incoming paths all structured away is never jumped to — drop those
    // labels so structured output only shows labels a goto actually targets.
    let mut used = Vec::new();
    collect_goto_targets(&stmts, &mut used);
    let used: HashSet<BlockId> = used.into_iter().collect();
    prune_unused_labels(&mut stmts, &used);
    Program {
        function: Some(function_id),
        stmts,
        labels,
    }
}

/// The maximum region-recursion depth before structuring bails to flat lowering.
/// Real control flow nests only shallowly; a depth this large means a degenerate
/// (often obfuscated) cascade whose structured form would risk a stack overflow
/// in this pass or in downstream AST recursion (emit/refine/switch all descend
/// the tree structuring builds). The bound is kept well under a typical spawned
/// thread's stack so those descents stay safe too. Flat lowering renders such a
/// function correctly, if uglily.
const MAX_REGION_DEPTH: usize = 1000;

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
    /// Blocks that may be emitted more than once. A goto is the structurer's way
    /// of saying "this block is already printed elsewhere"; permitting a second
    /// emission is exactly SAILR's cross-jump reversal, expressed without
    /// touching the CFG. Populated by [`CrossJumpRevert`], which only keeps a
    /// permission that strictly reduces the goto count.
    permit_dup: HashSet<BlockId>,
    /// The stack of enclosing loops (innermost last).
    frames: Vec<LoopFrame>,
    /// The region-recursion depth beyond which structuring bails out.
    max_depth: usize,
    /// Set when region recursion reaches [`Structurer::max_depth`], signalling the
    /// caller to discard the partial structuring and fall back to flat lowering.
    overflowed: bool,
}

impl Structurer<'_, '_> {
    /// Structures the region entered at `from`, stopping before `stop` (the
    /// enclosing merge point). Returns the statement list plus whether control
    /// falls through to `stop` (vs. terminating via break/continue/return/goto),
    /// so callers can decide whether to emit the continuation.
    fn region(&mut self, from: BlockId, stop: Option<BlockId>, depth: usize) -> (Vec<Stmt>, bool) {
        let mut out = Vec::new();
        let mut cur = Some(from);
        let mut fell_through = false;

        // Deep enough to risk a stack overflow: stop recursing and signal the
        // caller to fall back to flat lowering.
        if depth >= self.max_depth {
            self.overflowed = true;
            return (out, false);
        }

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
            // A target outside the analyzed subgraph, or an already-emitted join
            // we have no permission to repeat: reference it by goto.
            if !self.node_set.contains(&b)
                || (self.visited.contains(&b) && !self.permit_dup.contains(&b))
            {
                out.push(Stmt::Goto(b));
                break;
            }
            // A loop header reached for the first time: structure the whole loop
            // here, then continue from its exit.
            if self.loops.contains_key(&b) && !self.is_active(b) {
                let (loop_stmt, exit) = self.structure_loop(b, stop, depth);
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
                BlockExit::Goto { target, .. } => {
                    // Phi copies for this edge run at the end of the source block,
                    // just before control transfers on to `target`.
                    out.extend(block_arg_moves(self.ctx, b, target));
                    cur = Some(target);
                }
                BlockExit::Branch {
                    condition,
                    true_target,
                    false_target,
                    ..
                } => {
                    let merge = self.immediate_postdom(b);
                    let (then, then_ft) = self.region(true_target, merge, depth + 1);
                    let (els, els_ft) = self.region(false_target, merge, depth + 1);
                    // Each edge's phi copies belong at the top of that arm, so
                    // they run only on the path that takes the edge.
                    out.push(Stmt::If {
                        cond: lower_expr(self.ctx, condition),
                        then: prepend(block_arg_moves(self.ctx, b, true_target), then),
                        els: prepend(block_arg_moves(self.ctx, b, false_target), els),
                    });
                    // Continue at the merge only if some arm reaches it.
                    cur = if then_ft || els_ft { merge } else { None };
                }
                // A resolved dispatch: structure each arm into its case, exactly
                // as the two sides of a conditional are structured, bounded by
                // the dispatch's own reconvergence point.
                //
                // Arms sharing a target need no special handling: `region`
                // degrades an already-emitted block to a `goto`, so the first arm
                // to reach a shared body inlines it and the rest jump to it.
                BlockExit::Switch {
                    scrutinee,
                    arms,
                    default,
                    ..
                } => {
                    let merge = self.immediate_postdom(b);
                    let mut any_fell_through = false;
                    let mut arm_region = |s: &mut Self, target: BlockId| {
                        let (body, fell) = s.region(target, merge, depth + 1);
                        any_fell_through |= fell;
                        // This edge's phi copies run only on the path that takes it.
                        prepend(block_arg_moves(s.ctx, b, target), body)
                    };

                    let mut cases = Vec::with_capacity(arms.len());
                    for (values, target) in arms {
                        let body = arm_region(self, target);
                        cases.push(SwitchCase {
                            values,
                            body,
                            insns: Vec::new(),
                        });
                    }
                    let default = match default {
                        Some(target) => arm_region(self, target),
                        None => Vec::new(),
                    };

                    out.push(Stmt::Switch {
                        scrutinee: lower_expr(self.ctx, scrutinee),
                        cases,
                        default,
                    });
                    // Continue past the dispatch only if some arm reaches the merge.
                    cur = if any_fell_through { merge } else { None };
                }
                BlockExit::Indirect { edges } | BlockExit::Unstructured { edges } => {
                    for (_, target) in edges {
                        out.extend(block_arg_moves(self.ctx, b, target));
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
    fn structure_loop(
        &mut self,
        header: BlockId,
        stop: Option<BlockId>,
        depth: usize,
    ) -> (Stmt, Option<BlockId>) {
        let exit = self.loops[&header].exit;
        self.frames.push(LoopFrame { header, exit });
        let (body, _) = self.region(header, stop, depth + 1);
        self.frames.pop();
        // Loops are emitted endless; the `refine_loops` pass rewrites the shape.
        (Stmt::Loop { body }, exit)
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
    ///
    /// The candidates are scanned in a fixed (block-id) order so the choice is
    /// deterministic: when a block has no exit its post-dominator set is the whole
    /// graph and several candidates tie, and an unordered scan would pick a
    /// different one — and structure the code differently — run to run.
    fn immediate_postdom(&self, b: BlockId) -> Option<BlockId> {
        let set = self.pdom.get(&b)?;
        let mut strict: Vec<BlockId> = set.iter().copied().filter(|&x| x != b).collect();
        strict.sort_unstable_by_key(|&x| usize::from(x.local));
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
    let mut latches: HashMap<BlockId, Vec<BlockId>> = HashMap::default();
    for &u in nodes {
        for (_, v) in BasicBlock::from_id(ctx, u).successors() {
            if node_set.contains(&v) && doms.dominates(v, u) {
                latches.entry(v).or_default().push(u);
            }
        }
    }

    let mut loops: HashMap<BlockId, LoopInfo> = HashMap::default();
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
    let mut body = HashSet::default();
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
    let mut counts: HashMap<BlockId, usize> = HashMap::default();
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
        .max_by_key(|&(b, c)| (c, std::cmp::Reverse(usize::from(b.local))))
        .map(|(b, _)| b)
}

/// Collects every goto target in the statement tree, in emission order. Used to
/// find the virtualized-edge targets that still need a continuation region.
fn collect_goto_targets(stmts: &[Stmt], out: &mut Vec<BlockId>) {
    for stmt in stmts {
        match stmt {
            Stmt::Goto(b) | Stmt::GotoIf { target: b, .. } => out.push(*b),
            Stmt::If { then, els, .. } => {
                collect_goto_targets(then, out);
                collect_goto_targets(els, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::Loop { body } => {
                collect_goto_targets(body, out);
            }
            Stmt::Switch { cases, default, .. } => {
                for case in cases {
                    collect_goto_targets(&case.body, out);
                }
                collect_goto_targets(default, out);
            }
            _ => {}
        }
    }
}

/// Removes every [`Stmt::Label`] whose block is not in `used` (the set of goto
/// targets), recursing into structured bodies.
fn prune_unused_labels(stmts: &mut Vec<Stmt>, used: &HashSet<BlockId>) {
    stmts.retain(|s| !matches!(s, Stmt::Label(b) if !used.contains(b)));
    for stmt in stmts {
        match stmt {
            Stmt::If { then, els, .. } => {
                prune_unused_labels(then, used);
                prune_unused_labels(els, used);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::Loop { body } => {
                prune_unused_labels(body, used);
            }
            Stmt::Switch { cases, default, .. } => {
                for case in cases {
                    prune_unused_labels(&mut case.body, used);
                }
                prune_unused_labels(default, used);
            }
            _ => {}
        }
    }
}

/// Returns `head` extended by `tail` — used to place an edge's phi copies before
/// the body of the arm that takes it.
fn prepend(mut head: Vec<Stmt>, tail: Vec<Stmt>) -> Vec<Stmt> {
    head.extend(tail);
    head
}

/// The blocks reachable from `root`, in DFS order.
fn reachable(ctx: &Context, root: BlockId) -> Vec<BlockId> {
    let function = qcode::value::FunctionRef::from_id(ctx, root.func);
    jstd::graph::analysis::reachable_from_root(&function, root)
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
    use wazabin_qcode_macro::qcode;

    /// Whether any loop node appears anywhere in the statement tree.
    fn has_loop(stmts: &[Stmt]) -> bool {
        stmts.iter().any(|s| match s {
            Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::Loop { .. } => true,
            Stmt::If { then, els, .. } => has_loop(then) || has_loop(els),
            _ => false,
        })
    }

    /// An unresolved indirect branch closing a block is a real transfer of
    /// control with no goto to stand for it, so it must reach the output — and
    /// the coverage validator must be able to see it go missing. Whole-block
    /// coverage could not: the block's ordinary body was emitted, so it counted
    /// as covered while its terminator was silently dropped.
    #[test]
    fn indirect_branch_survives_structuring() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i64 dst;

            fn f:
            <entry>
                %c = load(cond:1, &cond);
                if %c goto <tail> else goto <done>;
            <tail>
                %p = load(dst:8, &dst);
                goto [i64 %p];
            <done>
                return at i64 0x0;
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        let c = emit_c(&ctx, &program, None);
        assert!(
            c.contains("goto *dst"),
            "the indirect transfer must survive structuring:\n{c}"
        );

        // And the validator must count it: dropping it is an error, not a pass.
        let mut stripped = program.clone();
        strip_indirect(&mut stripped.stmts, &ctx);
        assert!(
            validate_program(&ctx, f, &stripped, true).is_err(),
            "a dropped indirect branch must fail coverage validation"
        );
    }

    /// Removes every `branchind` statement from a statement tree, to check that
    /// validation notices.
    fn strip_indirect(stmts: &mut Vec<Stmt>, ctx: &Context) {
        stmts.retain(|s| match s {
            Stmt::Raw(id) => !matches!(
                qcode::value::Instruction::from_id(ctx, *id).mnemonic(),
                qcode::value::insn::Mnemonic::BranchInd(_)
            ),
            _ => true,
        });
        for stmt in stmts.iter_mut() {
            match stmt {
                Stmt::If { then, els, .. } => {
                    strip_indirect(then, ctx);
                    strip_indirect(els, ctx);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::Loop { body } => {
                    strip_indirect(body, ctx)
                }
                Stmt::Switch { cases, default, .. } => {
                    for case in cases.iter_mut() {
                        strip_indirect(&mut case.body, ctx);
                    }
                    strip_indirect(default, ctx);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn decompile_pipeline_passes_resolve_as_decompile_scope() {
        // The driver's pipeline names must all resolve through the pass manager
        // as decompilation-scoped passes.
        for &name in DECOMPILE_PIPELINE {
            assert!(
                matches!(make_pass(name), Some(RegisteredPass::Decompile(_))),
                "`{name}` should resolve as a decompilation pass"
            );
        }
    }

    /// Each arm of a resolved dispatch is structured into its case, the same way
    /// the two sides of a conditional are — not left as a jump to a block printed
    /// after the switch. An arm sharing a target with an earlier one degrades to a
    /// `goto`, since the body has already been emitted.
    /// SAILR cross-jump reversal: a tail shared by two paths is emitted once by
    /// the compiler and jumped to from the other, which region structuring cannot
    /// place. Duplicating it removes the goto.
    #[test]
    fn shared_tail_is_duplicated_to_remove_a_goto() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i32 out;

            fn f:
            <entry>
                %c = load(cond:1, &cond);
                if %c goto <mid_lbl> else goto <tail_lbl>;
            <mid_lbl>
                store(out:4, &out <- i32 0x1);
                goto <tail_lbl>;
            <tail_lbl>
                store(out:4, &out <- i32 0x9);
                return at 0x0;
            "
        );

        // Structuring alone reaches `tail_lbl` from two places and must jump.
        let plain = structure_regions(&ctx, f);

        let mut deopted = plain.clone();
        let changed = CrossJumpRevert
            .run(&ctx, f, &mut deopted)
            .expect("deopt runs");

        if plain.goto_count() > 0 {
            assert!(changed, "a shared tail should have been duplicated");
            assert!(
                deopted.goto_count() < plain.goto_count(),
                "the deopt must strictly reduce gotos: {} -> {}",
                plain.goto_count(),
                deopted.goto_count()
            );
            // The tail's body is now printed on both paths.
            let c = emit_c(&ctx, &deopted, None);
            assert!(
                c.matches("out = 0x9").count() >= 2,
                "the shared tail should appear on each path:\n{c}"
            );
        }
    }

    /// The deopt is speculative: a candidate that does not strictly reduce the
    /// goto count is reverted, so the pass can never make its own metric worse.
    #[test]
    fn deopt_never_increases_the_goto_count() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i32 out;

            fn f:
            <entry>
                %c = load(cond:1, &cond);
                if %c goto <a_lbl> else goto <b_lbl>;
            <a_lbl>
                store(out:4, &out <- i32 0x1);
                return at 0x0;
            <b_lbl>
                store(out:4, &out <- i32 0x2);
                return at 0x0;
            "
        );

        let plain = structure_regions(&ctx, f);
        let mut deopted = plain.clone();
        CrossJumpRevert
            .run(&ctx, f, &mut deopted)
            .expect("deopt runs");
        assert!(
            deopted.goto_count() <= plain.goto_count(),
            "deopt must not increase gotos: {} -> {}",
            plain.goto_count(),
            deopted.goto_count()
        );
    }

    #[test]
    fn switch_arms_are_structured_into_their_cases() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 idx;
            varnode i64 out;

            fn f:
            <entry>
                %i = load(idx:8, &idx);
                switch %i { 0x0 => <a_lbl>, 0x1 => <b_lbl>, 0x4 => <a_lbl>, 0x2 => <c_lbl> };
            <a_lbl>
                store(out:8, &out <- i64 0xaa);
                return at 0x0;
            <b_lbl>
                store(out:8, &out <- i64 0xbb);
                return at 0x0;
            <c_lbl>
                store(out:8, &out <- i64 0xcc);
                return at 0x0;
            "
        );

        let program = structure_regions(&ctx, f);
        let c = emit_c(&ctx, &program, None);

        // Each arm's body sits inside its case: the store appears between the
        // arm's `=>` and its closing brace, not after the switch under a label.
        for (arm, value) in [("0x0 | 0x4", "0xaa"), ("0x1", "0xbb"), ("0x2", "0xcc")] {
            let body = c
                .split_once(&format!("{arm} =>"))
                .unwrap_or_else(|| panic!("no `{arm}` arm:\n{c}"))
                .1;
            let end = body.find('}').expect("arm closes");
            assert!(
                body[..end].contains(value),
                "arm `{arm}` should hold its body {value}, not a jump:\n{c}"
            );
        }

        // 0x0 and 0x4 reach one block with the same (empty) arguments, so they
        // are one arm with two labels rather than a duplicated body.
        assert_eq!(
            c.matches("0xaa").count(),
            1,
            "the shared target's body should be emitted once:\n{c}"
        );
        // Nothing is left over after the dispatch: every body went into an arm.
        assert!(
            !c.contains("goto"),
            "no arm should degrade to a jump here:\n{c}"
        );
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
                %c = load(cond:1, &cond);
                if %c goto <then_lbl> else goto <else_lbl>;
            <then_lbl>
                store(x:4, &x <- i32 0x1);
                goto <merge>;
            <else_lbl>
                store(x:4, &x <- i32 0x2);
                goto <merge>;
            <merge>
                return at i64 0;
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        // The branch is fully structured: no gotos remain (the merge is inlined).
        assert_eq!(
            program.goto_count(),
            0,
            "if/else should erase gotos:\n{}",
            emit_c(&ctx, &program, None)
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
    fn fully_structured_output_has_no_stray_labels() {
        // The merge block of an if/else is a join (>1 predecessors), so the walk
        // labels it defensively — but once both arms structure, nothing jumps to
        // it and the label must be pruned from the output.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i32 x;

            fn f:
            <entry>
                %c = load(cond:1, &cond);
                if %c goto <then_lbl> else goto <else_lbl>;
            <then_lbl>
                store(x:4, &x <- i32 0x1);
                goto <merge>;
            <else_lbl>
                store(x:4, &x <- i32 0x2);
                goto <merge>;
            <merge>
                return at i64 0;
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        let c = emit_c(&ctx, &program, None);
        assert!(
            !c.contains("merge:"),
            "unreferenced join label should be pruned:\n{c}"
        );
        assert!(
            !program.stmts.iter().any(|s| matches!(s, Stmt::Label(_))),
            "goto-free output should carry no labels:\n{c}"
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
                %c = load(cond:1, &cond);
                if %c goto <then_lbl> else goto <merge>;
            <then_lbl>
                store(x:4, &x <- i32 0x1);
                goto <merge>;
            <merge>
                return at i64 0;
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        assert_eq!(
            program.goto_count(),
            0,
            "if-then should erase gotos:\n{}",
            emit_c(&ctx, &program, None)
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
                %c = load(cond:1, &cond);
                if %c goto <then_lbl> else goto <else_lbl>;
            <then_lbl>
                store(x:4, &x <- i32 0x1);
                goto <merge>;
            <else_lbl>
                store(x:4, &x <- i32 0x2);
                goto <merge>;
            <merge>
                return at i64 0;
            "
        );

        let flat = lower_function(&ctx, f).goto_count();
        let structured = decompile_function(&ctx, f).unwrap().goto_count();
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
                %c = load(cond:1, &cond);
                if %c goto <body> else goto <exit_lbl>;
            <body>
                store(x:4, &x <- i32 0x1);
                goto <head>;
            <exit_lbl>
                return at i64 0;
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        // The loop (header test, body, back edge, exit) structures with no gotos:
        // the back edge is a `continue`, the exit a `break`.
        assert_eq!(
            program.goto_count(),
            0,
            "loop should structure without gotos:\n{}",
            emit_c(&ctx, &program, None)
        );
        // Header-tested: refines to a pre-tested `while (cond)`.
        let c = emit_c(&ctx, &program, None);
        assert!(
            program
                .stmts
                .iter()
                .any(|s| matches!(s, Stmt::While { .. })),
            "expected a while loop, got:\n{c}"
        );
        assert!(
            c.contains("while (cond) {"),
            "expected `while (cond)`:\n{c}"
        );
        assert!(!c.contains("while (true)"), "should not be endless:\n{c}");
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
                store(x:4, &x <- i32 0x1);
                %c = load(cond:1, &cond);
                if %c goto <head> else goto <exit_lbl>;
            <exit_lbl>
                return at i64 0;
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        assert_eq!(
            program.goto_count(),
            0,
            "self-looping block should structure without gotos:\n{}",
            emit_c(&ctx, &program, None)
        );
        // Latch-tested: refines to a post-tested `do { … } while (cond)`.
        let c = emit_c(&ctx, &program, None);
        assert!(
            program
                .stmts
                .iter()
                .any(|s| matches!(s, Stmt::DoWhile { .. })),
            "expected a do-while loop, got:\n{c}"
        );
        assert!(c.contains("do {"), "expected a do-block:\n{c}");
        assert!(
            c.contains("} while (cond)"),
            "expected trailing while test:\n{c}"
        );
    }

    #[test]
    fn exitless_function_structures_deterministically() {
        // An endless loop with no exit gives every block the whole graph as its
        // post-dominator set (pdom = ⊤). The immediate-post-dominator choice must
        // be canonical there, not hash-iteration-order dependent. Smoke-test that
        // it structures at all (the ⊤ path) and yields the same output twice.
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
                %c = load(cond:1, &cond);
                if %c goto <a> else goto <b>;
            <a>
                store(x:4, &x <- i32 0x1);
                goto <head>;
            <b>
                store(x:4, &x <- i32 0x2);
                goto <head>;
            "
        );

        let first = emit_c(&ctx, &decompile_function(&ctx, f).unwrap(), None);
        let second = emit_c(&ctx, &decompile_function(&ctx, f).unwrap(), None);
        assert_eq!(first, second, "structuring should be deterministic");
        assert!(
            first.contains("0x1") && first.contains("0x2"),
            "both arms should be present:\n{first}"
        );
    }

    #[test]
    fn block_argument_join_emits_phi_copies() {
        // `x = cond ? 1 : 2` after mem2reg: the merge block takes a parameter `v`
        // that each arm passes a different value to. The per-edge argument moves
        // must be emitted, or the merge's `x = v` reads a variable never assigned.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i32 x;

            fn f:
            <entry>
                %c = load(cond:1, &cond);
                if %c goto <then_lbl> else goto <else_lbl>;
            <then_lbl>
                goto <merge @v=i32 0x1>;
            <else_lbl>
                goto <merge @v=i32 0x2>;
            <merge @v:i32>
                store(x:4, &x <- @v);
                return at i64 0;
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        let c = emit_c(&ctx, &program, None);
        // Both edges assign the join parameter before it is used.
        assert!(c.contains("v = 0x1"), "then edge's phi copy missing:\n{c}");
        assert!(c.contains("v = 0x2"), "else edge's phi copy missing:\n{c}");
        // And the merge consumes that same parameter.
        assert!(
            c.contains("x = v"),
            "merge should read the join parameter:\n{c}"
        );
    }

    #[test]
    fn loop_carried_parameter_updates_on_the_back_edge() {
        // A counting loop: the header parameter `i` is seeded to 0 on entry and
        // updated to `i + 1` on the back edge. The entry copy and the back-edge
        // copy must both be emitted, or the loop counter is never initialized or
        // never advances.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;

            fn f:
            <entry>
                goto <head @i=i32 0x0>;
            <head @i:i32>
                %c = i32 @i s< i32 0xa;
                if %c goto <body> else goto <exit_lbl>;
            <body>
                %ni = i32 @i + i32 0x1;
                goto <head @i=%ni>;
            <exit_lbl>
                store(x:4, &x <- @i);
                return at i64 0;
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        let c = emit_c(&ctx, &program, None);
        assert!(c.contains("i = 0x0"), "counter should be initialized:\n{c}");
        assert!(
            c.contains("i = i + 0x1"),
            "counter should advance on the back edge:\n{c}"
        );
    }

    #[test]
    fn loop_with_two_exits_drops_no_code() {
        // A loop whose body leaves through two different blocks: the header's
        // false edge to `exit1`, and the body's error edge to `cleanup`. Only one
        // can be the structured `break` target; a naive walk turns the other into
        // a goto it never follows, dropping that block's body. Validation must
        // catch the drop and fall back to flat lowering, which keeps every block.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i8 err;
            varnode i32 x;

            fn f:
            <entry>
                goto <head>;
            <head>
                %c = load(cond:1, &cond);
                if %c goto <body> else goto <exit1>;
            <body>
                %e = load(err:1, &err);
                if %e goto <cleanup> else goto <head>;
            <cleanup>
                store(x:4, &x <- i32 0xdead);
                return at i64 0;
            <exit1>
                store(x:4, &x <- i32 0xbeef);
                return at i64 0;
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        let c = emit_c(&ctx, &program, None);
        // Neither exit's body may be lost, however the loop is (or isn't) structured.
        assert!(
            c.contains("0xdead"),
            "the cleanup exit body was dropped:\n{c}"
        );
        assert!(c.contains("0xbeef"), "the main exit body was dropped:\n{c}");
    }

    #[test]
    fn two_exit_loop_structures_with_a_continuation_region() {
        // Same shape as `loop_with_two_exits_drops_no_code`: the secondary exit
        // edge is virtualized to a goto. With continuation regions the goto's
        // target is emitted as a labeled region, so the function keeps its
        // structured loop instead of demoting wholesale to flat lowering.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i8 err;
            varnode i32 x;

            fn f:
            <entry>
                goto <head>;
            <head>
                %c = load(cond:1, &cond);
                if %c goto <body> else goto <exit1>;
            <body>
                %e = load(err:1, &err);
                if %e goto <cleanup> else goto <head>;
            <cleanup>
                store(x:4, &x <- i32 0xdead);
                return at i64 0;
            <exit1>
                store(x:4, &x <- i32 0xbeef);
                return at i64 0;
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        let c = emit_c(&ctx, &program, None);
        assert!(
            has_loop(&program.stmts),
            "the loop should stay structured, not fall back to flat:\n{c}"
        );
        // The virtualized exit is one goto; everything else is structured.
        assert_eq!(program.goto_count(), 1, "expected exactly one goto:\n{c}");
        // Both exit bodies survive, and the goto's target label is printed.
        assert!(c.contains("0xdead"), "cleanup body dropped:\n{c}");
        assert!(c.contains("0xbeef"), "main exit body dropped:\n{c}");
        // Which exit gets virtualized is a tie-break; whichever it is, its goto
        // must have a matching printed label.
        assert!(
            (c.contains("goto cleanup;") && c.contains("cleanup:"))
                || (c.contains("goto exit1;") && c.contains("exit1:")),
            "virtualized edge should have a labeled continuation:\n{c}"
        );
    }

    #[test]
    fn deep_cascade_falls_back_to_flat_lowering() {
        // A comparison cascade recurses one region level per arm. Past the depth
        // guard, structuring must abandon its partial result and fall back to the
        // flat lowering (iterative, always correct) rather than risk a stack
        // overflow. Driven with a tiny limit so the test needs only a few blocks.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 sel;
            varnode i32 out;

            fn f:
            <entry>
                %x = load(sel:4, &sel);
                %c1 = i32 %x == i32 0x1;
                if %c1 goto <case1> else goto <t2>;
            <case1>
                store(out:4, &out <- i32 0x10);
                goto <done>;
            <t2>
                %c2 = i32 %x == i32 0x2;
                if %c2 goto <case2> else goto <t3>;
            <case2>
                store(out:4, &out <- i32 0x20);
                goto <done>;
            <t3>
                %c3 = i32 %x == i32 0x3;
                if %c3 goto <case3> else goto <done>;
            <case3>
                store(out:4, &out <- i32 0x30);
                goto <done>;
            <done>
                return at i64 0;
            "
        );

        // A generous limit structures the cascade; a tiny one trips the guard and
        // yields exactly the flat lowering.
        let deep = structure_regions_limited(&ctx, f, 2, &HashSet::default());
        let flat = lower_function(&ctx, f);
        assert_eq!(
            emit_c(&ctx, &deep, None),
            emit_c(&ctx, &flat, None),
            "an over-deep cascade should fall back to flat lowering"
        );
        let shallow = structure_regions_limited(&ctx, f, MAX_REGION_DEPTH, &HashSet::default());
        assert!(
            shallow.goto_count() < flat.goto_count(),
            "within the limit it should still structure"
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
                %oc = load(outer_c:1, &outer_c);
                if %oc goto <inner> else goto <exit_lbl>;
            <inner>
                %ic = load(inner_c:1, &inner_c);
                if %ic goto <inner_body> else goto <outer>;
            <inner_body>
                store(x:4, &x <- i32 0x1);
                goto <inner>;
            <exit_lbl>
                return at i64 0;
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        assert_eq!(
            program.goto_count(),
            0,
            "nested loops should structure without gotos:\n{}",
            emit_c(&ctx, &program, None)
        );
        assert!(
            has_loop(&program.stmts),
            "expected loop nodes:\n{:#?}",
            program.stmts
        );
    }
}
