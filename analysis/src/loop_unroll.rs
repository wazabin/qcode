//! Restricted loop recognizer for the first loop-unrolling milestone.
//!
//! This pass only annotates loops. It deliberately accepts a narrow canonical
//! induction-variable shape so later unrolling work has a reliable starting point:
//!
//! ```text
//! preheader -> header(@i = C0)
//! header: if @i < C1 goto body else goto exit
//! body: ...
//! latch: %next = @i + C2; goto header(@i = %next)
//! ```

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::VecDeque;

use jstd::graph::analysis::{DominatorTree, compute_dominators, compute_postdominators};
use qcode::value::{
    BlockParamId, FunctionId, QCodeView, ValueId,
    block::BlockId,
    insn::{Binary, Binop, Branch, CBranch, InstructionId, IntBinop, Mnemonic},
    util::base_ref::BaseRef,
};

use crate::{ContextView, FunctionBody, FunctionPass, Outcome};

const COMMENT_PREFIX: &str = "loop_unroll:";
const MAX_UNROLL_ITERATIONS: u64 = 10;

#[derive(Default)]
pub struct RecognizeSimpleLoops;

#[derive(Default)]
pub struct UnrollSimpleLoops;

#[derive(Debug, Clone, PartialEq, Eq)]
struct SimpleLoop {
    header: BlockId,
    induction: BlockParamId,
    body: BlockId,
    latch: BlockId,
    initial: u64,
    bound: u64,
    step: u64,
    iterations: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BackEdge {
    latch: BlockId,
    header: BlockId,
}

struct LoopAnalysis {
    dominators: DominatorTree<BlockId>,
    postdominators: HashMap<BlockId, HashSet<BlockId>>,
    backedges: Vec<BackEdge>,
}

impl FunctionPass for RecognizeSimpleLoops {
    const NAME: &'static str = "recognize_simple_loops";

    fn description(&self) -> &'static str {
        "Recognize simple constant-bound induction-variable loops"
    }

    fn run<'str>(
        &self,
        f: &mut FunctionBody<'str>,
        m: ContextView<'_, 'str>,
        _next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String> {
        let fid = f.id();
        Ok(Outcome::changed(recognize_simple_loops_host(f, m, fid))
            .preserving_global::<crate::CallGraphAnalysis>()
            .preserving_local::<crate::AliasAnalysis>())
    }
}

crate::register_function_pass!(RecognizeSimpleLoops);

impl FunctionPass for UnrollSimpleLoops {
    const NAME: &'static str = "unroll_simple_loops";

    fn description(&self) -> &'static str {
        "Unroll simple constant-bound induction-variable loops with fewer than ten iterations"
    }

    fn run<'str>(
        &self,
        f: &mut FunctionBody<'str>,
        m: ContextView<'_, 'str>,
        _next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String> {
        let fid = f.id();
        Ok(Outcome::changed(unroll_simple_loops_host(f, m, fid)))
    }
}

crate::register_function_pass!(UnrollSimpleLoops);

/// Concrete `FunctionBody`/`ContextView` version of [`recognize_simple_loops`]
/// (stage 5b).
pub fn recognize_simple_loops_host<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    fun_id: FunctionId,
) -> bool {
    let Some(analysis) = LoopAnalysis::compute(cx.body_view(body), fun_id) else {
        return false;
    };
    let block_ids = analysis.block_ids();
    let recognized = analysis
        .backedges
        .iter()
        .filter_map(|&edge| recognize_simple_loop(cx.body_view(body), &analysis, edge))
        .fold(
            HashMap::<BlockId, Vec<SimpleLoop>>::default(),
            |mut acc, lp| {
                acc.entry(lp.header()).or_default().push(lp);
                acc
            },
        );

    let updates = block_ids
        .iter()
        .map(|&block| {
            let loop_comment = recognized.get(&block).and_then(|loops| {
                (loops.len() == 1).then(|| format_loop_comment(cx.body_view(body), &loops[0]))
            });
            let current = cx
                .body_view(body)
                .block_ref(block)
                .comment()
                .map(str::to_owned);
            (
                block,
                merge_loop_comment(current.as_deref(), loop_comment.as_deref()),
            )
        })
        .collect::<Vec<_>>();

    let mut changed = false;
    for (block, comment) in updates {
        let current = cx
            .body_view(body)
            .block_ref(block)
            .comment()
            .map(str::to_owned);
        if current != comment {
            // TODO(5b-ii): `BaseRef::set_comment` is not mirrored on
            // `FunctionBody`; go through a temporary host.
            let mut host = cx.host(body);
            BaseRef::new(host.reborrow(), block).set_comment(comment);
            changed = true;
        }
    }
    changed
}

/// Concrete `FunctionBody`/`ContextView` version of [`unroll_simple_loops`]
/// (stage 5b).
pub fn unroll_simple_loops_host<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    fun_id: FunctionId,
) -> bool {
    let Some(analysis) = LoopAnalysis::compute(cx.body_view(body), fun_id) else {
        return false;
    };

    let candidates = analysis
        .backedges
        .iter()
        .filter_map(|&edge| {
            let lp = recognize_simple_loop(cx.body_view(body), &analysis, edge)?;
            (lp.iterations < MAX_UNROLL_ITERATIONS).then_some((edge, lp))
        })
        .collect::<Vec<_>>();

    let mut changed = false;
    for (edge, lp) in candidates {
        let Some(plan) = UnrollPlan::build(cx.body_view(body), &analysis, edge, lp) else {
            continue;
        };
        if apply_unroll_plan(body, cx, plan) {
            changed = true;
        }
    }

    changed
}

struct UnrollPlan {
    lp: SimpleLoop,
    preheader: BlockId,
    exit: BlockId,
    preheader_args: Vec<ValueId>,
    exit_args: Vec<ValueId>,
    path: Vec<BlockId>,
    loop_nodes: HashSet<BlockId>,
}

impl UnrollPlan {
    fn build<'ctx, 'str: 'ctx>(
        host: impl QCodeView<'ctx, 'str>,
        analysis: &LoopAnalysis,
        edge: BackEdge,
        lp: SimpleLoop,
    ) -> Option<Self> {
        let loop_nodes = natural_loop(host, edge);
        let preheader = loop_preheader(host, lp.header, &loop_nodes)?;
        let cbranch = header_cbranch(host, lp.header)?;
        let (exit, exit_args) = header_exit(lp.header.func, &loop_nodes, &cbranch)?;
        let preheader_args = branch_args_to(host, preheader, lp.header)?;
        let path = linear_loop_path(host, lp.body, lp.latch, &loop_nodes)?;

        if !path
            .iter()
            .all(|block| host.block_ref(*block).params().next().is_none())
        {
            return None;
        }

        if analysis
            .backedges
            .iter()
            .filter(|candidate| candidate.header == lp.header)
            .count()
            != 1
        {
            return None;
        }

        Some(Self {
            lp,
            preheader,
            exit,
            preheader_args,
            exit_args,
            path,
            loop_nodes,
        })
    }
}

fn apply_unroll_plan<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    plan: UnrollPlan,
) -> bool {
    let mut carried = plan.preheader_args.clone();
    let mut first_new_block = None;
    let mut previous_new_block = None;
    let mut created_blocks = Vec::new();

    for iteration in 0..plan.lp.iterations {
        let mut value_map = header_value_map(cx.body_view(body), plan.lp.header, &carried);

        for (path_index, &old_block) in plan.path.iter().enumerate() {
            let new_block = body.make_block();
            {
                // TODO(5b-ii): `BaseRef::rename_local` is not mirrored on
                // `FunctionBody`; go through a temporary host.
                let mut host = cx.host(body);
                let _ = BaseRef::new(host.reborrow(), new_block).rename_local(
                    format!(
                        "unroll_{:x}_{iteration}_{path_index}",
                        usize::from(old_block.local)
                    )
                    .into(),
                );
            }
            created_blocks.push(new_block);
            first_new_block.get_or_insert(new_block);

            if let Some(previous) = previous_new_block {
                replace_terminator_with_branch(body, cx, previous, new_block, Vec::new());
            }

            let old_insns = cx
                .body_view(body)
                .block_ref(old_block)
                .instruction_ids()
                .to_vec();
            let Some((&terminator, body_insns)) = old_insns.split_last() else {
                return false;
            };

            for old_insn in body_insns.iter().copied() {
                let (type_id, mnemonic) = {
                    let old_ref = cx.body_view(body).insn_ref(old_insn);
                    (
                        old_ref.type_id(),
                        remap_mnemonic(old_ref.mnemonic(), &value_map),
                    )
                };
                let new_insn = body.push_mnemonic_with_type(mnemonic, type_id);
                let insert_at = cx
                    .body_view(body)
                    .block_ref(new_block)
                    .instruction_ids()
                    .len();
                {
                    // TODO(5b-ii): `BaseRef::insert_insn_at_index` is not mirrored
                    // on `FunctionBody`; go through a temporary host.
                    let mut host = cx.host(body);
                    BaseRef::new(host.reborrow(), new_block)
                        .insert_insn_at_index(insert_at, new_insn);
                }
                value_map.insert(
                    ValueId::Instruction(old_insn),
                    ValueId::Instruction(new_insn),
                );
            }

            let is_latch = path_index + 1 == plan.path.len();
            if is_latch {
                let Some(next_carried) = remapped_branch_args_to(
                    cx.body_view(body),
                    terminator,
                    plan.lp.header,
                    &value_map,
                ) else {
                    return false;
                };
                carried = next_carried;
                previous_new_block = Some(new_block);
            } else {
                previous_new_block = Some(new_block);
            }
        }
    }

    let final_target = first_new_block.unwrap_or(plan.exit);
    let preheader_args = if first_new_block.is_some() {
        Vec::new()
    } else {
        remap_values(
            &plan.exit_args,
            &header_value_map(cx.body_view(body), plan.lp.header, &plan.preheader_args),
        )
    };
    replace_terminator_with_branch(body, cx, plan.preheader, final_target, preheader_args);

    if let Some(last_new_block) = previous_new_block {
        let exit_args = remap_values(
            &plan.exit_args,
            &header_value_map(cx.body_view(body), plan.lp.header, &carried),
        );
        replace_terminator_with_branch(body, cx, last_new_block, plan.exit, exit_args);
    }

    // A header param may be read *directly* outside the loop: the header dominates
    // the exit, so a live-out can use the param without an exit-block param (e.g. a
    // returned register write-set referencing the loop counter — `fn_449740`'s
    // `pack(ECX=@counter)`). Deleting the header would dangle such uses, so rewrite
    // every header param to its final loop-carried value. `carried` holds, in param
    // order, each param's value at loop exit (the last latch's args; the initial
    // values when the loop ran zero times). Uses inside the about-to-be-deleted loop
    // blocks are rewritten too, harmlessly.
    let header_params: Vec<BlockParamId> = cx
        .body_view(body)
        .block_ref(plan.lp.header)
        .params()
        .map(|param| param.id)
        .collect();
    for (&param, &final_value) in header_params.iter().zip(carried.iter()) {
        body.replace_all_uses_with(ValueId::BlockParam(param), final_value);
    }

    for block in plan.loop_nodes {
        body.delete_block(block);
    }

    !created_blocks.is_empty() || plan.lp.iterations == 0
}

fn loop_preheader<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    header: BlockId,
    loop_nodes: &HashSet<BlockId>,
) -> Option<BlockId> {
    let header_ref = host.block_ref(header);
    let mut preheaders = header_ref
        .predecessors()
        .filter_map(|(_, pred)| (!loop_nodes.contains(&pred)).then_some(pred));
    let preheader = preheaders.next()?;
    preheaders.next().is_none().then_some(preheader)
}

fn branch_args_to<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    block: BlockId,
    target: BlockId,
) -> Option<Vec<ValueId>> {
    let term_id = *host.block_ref(block).instruction_ids().last()?;
    let Mnemonic::Branch(Branch {
        target: branch_target,
        args,
    }) = host.insn_ref(term_id).mnemonic()
    else {
        return None;
    };
    (BlockId::new(block.func, *branch_target) == target)
        .then(|| args.iter().map(|a| a.qualify(block.func)).collect())
}

fn header_exit(
    func: qcode::value::FunctionId,
    loop_nodes: &HashSet<BlockId>,
    cbranch: &CBranch,
) -> Option<(BlockId, Vec<ValueId>)> {
    // The header CBranch's targets are body-local indices in `func`'s arena.
    let sb = BlockId::new(func, cbranch.success_block);
    let fb = BlockId::new(func, cbranch.failure_block);
    let success_is_body = loop_nodes.contains(&sb);
    let failure_is_body = loop_nodes.contains(&fb);
    match (success_is_body, failure_is_body) {
        (true, false) => Some((
            fb,
            cbranch
                .failure_args
                .iter()
                .map(|a| a.qualify(func))
                .collect(),
        )),
        (false, true) => Some((
            sb,
            cbranch
                .success_args
                .iter()
                .map(|a| a.qualify(func))
                .collect(),
        )),
        _ => None,
    }
}

fn linear_loop_path<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    body: BlockId,
    latch: BlockId,
    loop_nodes: &HashSet<BlockId>,
) -> Option<Vec<BlockId>> {
    let mut path = Vec::new();
    let mut seen = HashSet::default();
    let mut current = body;

    loop {
        if !loop_nodes.contains(&current) || !seen.insert(current) {
            return None;
        }
        path.push(current);
        if current == latch {
            return Some(path);
        }

        let current_ref = host.block_ref(current);
        let mut successors = current_ref
            .successors()
            .filter_map(|(_, succ)| loop_nodes.contains(&succ).then_some(succ));
        let next = successors.next()?;
        if successors.next().is_some() {
            return None;
        }
        current = next;
    }
}

fn header_value_map<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    header: BlockId,
    values: &[ValueId],
) -> HashMap<ValueId, ValueId> {
    host.block_ref(header)
        .params()
        .map(|param| ValueId::BlockParam(param.id))
        .zip(values.iter().copied())
        .collect()
}

fn remapped_branch_args_to<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    terminator: InstructionId,
    target: BlockId,
    value_map: &HashMap<ValueId, ValueId>,
) -> Option<Vec<ValueId>> {
    let Mnemonic::Branch(Branch {
        target: branch_target,
        args,
    }) = host.insn_ref(terminator).mnemonic()
    else {
        return None;
    };
    (BlockId::new(terminator.func, *branch_target) == target).then(|| {
        let args: Vec<ValueId> = args.iter().map(|a| a.qualify(terminator.func)).collect();
        remap_values(&args, value_map)
    })
}

fn remap_values(values: &[ValueId], value_map: &HashMap<ValueId, ValueId>) -> Vec<ValueId> {
    values
        .iter()
        .map(|value| remap_value(*value, value_map))
        .collect()
}

fn remap_value(value: ValueId, value_map: &HashMap<ValueId, ValueId>) -> ValueId {
    value_map.get(&value).copied().unwrap_or(value)
}

fn remap_mnemonic(mnemonic: &Mnemonic, value_map: &HashMap<ValueId, ValueId>) -> Mnemonic {
    let mut remapped = mnemonic.clone();
    for (&old, &new) in value_map {
        // The map relates values within one function (the unrolled loop's own),
        // so both sides strip to the same body's local domain.
        remapped.replace_value(old.strip_func(), new.strip_func());
    }
    remapped
}

pub(crate) fn replace_terminator_with_branch<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    block: BlockId,
    target: BlockId,
    args: Vec<ValueId>,
) {
    let mut old_successors = cx
        .body_view(body)
        .block_ref(block)
        .successors()
        .map(|(edge, _)| edge)
        .collect::<Vec<_>>();
    old_successors.sort_unstable();
    old_successors.dedup();
    for edge in old_successors {
        body.remove_cfg_edge(edge);
    }

    // Reuse the existing terminator only if the block actually ends in one. The
    // freshly-created unrolled blocks hold only copied *body* instructions (no
    // terminator yet); their last instruction is a real value (e.g. the induction
    // increment), which must not be clobbered into the branch — doing so destroys
    // that value and, when it is the exit argument, yields a branch that passes
    // itself. In that case append the branch instead.
    let term_id = cx
        .body_view(body)
        .block_ref(block)
        .instruction_ids()
        .last()
        .copied()
        .filter(|&id| cx.body_view(body).insn_ref(id).mnemonic().is_terminator());
    let local_target = target.localize(block.func);
    let args: Vec<_> = args
        .into_iter()
        .map(|arg| arg.localize(block.func))
        .collect();
    if let Some(term_id) = term_id {
        body.replace_instruction_mnemonic(
            term_id,
            Mnemonic::Branch(Branch {
                target: local_target,
                args,
            }),
        );
    } else {
        let branch = body.push_mnemonic(
            cx.shr(),
            Mnemonic::Branch(Branch {
                target: local_target,
                args,
            }),
            0,
        );
        let end = cx.body_view(body).block_ref(block).instruction_ids().len();
        {
            // TODO(5b-ii): `BaseRef::insert_insn_at_index` is not mirrored on
            // `FunctionBody`; go through a temporary host.
            let mut host = cx.host(body);
            BaseRef::new(host.reborrow(), block).insert_insn_at_index(end, branch);
        }
    }
    body.add_cfg_edge(block, target);
}

impl LoopAnalysis {
    fn compute<'ctx, 'str: 'ctx>(
        host: impl QCodeView<'ctx, 'str>,
        fun_id: FunctionId,
    ) -> Option<Self> {
        let function = host.function_ref(fun_id);
        let root = function.root()?.id;
        let block_ids = function.iter().map(|block| block.id).collect::<Vec<_>>();
        if block_ids.is_empty() {
            return None;
        }

        let node_set = block_ids.iter().copied().collect::<HashSet<_>>();
        let exit_set = block_ids
            .iter()
            .copied()
            .filter(|&block| host.block_ref(block).successors().next().is_none())
            .collect::<HashSet<_>>();
        if exit_set.is_empty() {
            return None;
        }

        let cfg = host.function_ref(fun_id);
        let dominators = compute_dominators(&cfg, root);
        let postdominators = compute_postdominators(&cfg, &block_ids, &node_set, &exit_set);
        let mut backedges = Vec::new();
        for &latch in &block_ids {
            let successors = host
                .block_ref(latch)
                .successors()
                .map(|(_, header)| header)
                .collect::<Vec<_>>();
            for header in successors {
                if dominators.dominates(header, latch) {
                    backedges.push(BackEdge { latch, header });
                }
            }
        }

        Some(Self {
            dominators,
            postdominators,
            backedges,
        })
    }

    fn block_ids(&self) -> Vec<BlockId> {
        let mut blocks = self.postdominators.keys().copied().collect::<Vec<_>>();
        blocks.sort();
        blocks
    }
}

impl SimpleLoop {
    fn header(&self) -> BlockId {
        self.header
    }
}

fn recognize_simple_loop<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    analysis: &LoopAnalysis,
    edge: BackEdge,
) -> Option<SimpleLoop> {
    let header = edge.header;
    let cbranch = header_cbranch(host, header)?;
    let (induction, bound, signed) = condition_bound(host, cbranch.condition.qualify(header.func))?;
    let header_params = host
        .block_ref(header)
        .params()
        .map(|param| param.id)
        .collect::<Vec<_>>();
    let param_index = header_params.iter().position(|&param| param == induction)?;

    let loop_nodes = natural_loop(host, edge);
    let body = first_body_block(host, analysis, &loop_nodes, edge, &cbranch)?;
    let initial = loop_initial_value(host, header, &loop_nodes, param_index)?;
    let step = latch_step(host, edge.latch, header, param_index, induction)?;
    if step == 0 {
        return None;
    }

    let iterations = if signed {
        // Trip count for a signed comparison must be computed with signed
        // arithmetic: an initial value like `-10` is stored as a large u64,
        // and naive unsigned `initial >= bound` would wrongly yield 0.
        let initial = initial as i64;
        let bound = bound as i64;
        if initial >= bound {
            0
        } else {
            // `bound > initial` and `step > 0`, so the gap is positive:
            // ceil(gap / step) without the unstable signed `div_ceil`.
            let step = step as i64;
            ((bound - initial + step - 1) / step) as u64
        }
    } else if initial >= bound {
        0
    } else {
        (bound - initial).div_ceil(step)
    };

    Some(SimpleLoop {
        header,
        induction,
        body,
        latch: edge.latch,
        initial,
        bound,
        step,
        iterations,
    })
}

fn natural_loop<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    edge: BackEdge,
) -> HashSet<BlockId> {
    let mut nodes = HashSet::from_iter([edge.header, edge.latch]);
    let mut worklist = VecDeque::from([edge.latch]);

    while let Some(block) = worklist.pop_front() {
        for (_, pred) in host.block_ref(block).predecessors() {
            if nodes.insert(pred) && pred != edge.header {
                worklist.push_back(pred);
            }
        }
    }

    nodes
}

fn header_cbranch<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    header: BlockId,
) -> Option<CBranch> {
    let block = host.block_ref(header);
    let term_id = *block.instruction_ids().last()?;
    match host.insn_ref(term_id).mnemonic() {
        Mnemonic::CBranch(cbranch) => Some(cbranch.clone()),
        _ => None,
    }
}

fn condition_bound<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    condition: ValueId,
) -> Option<(BlockParamId, u64, bool)> {
    let ValueId::Instruction(condition_id) = condition else {
        return None;
    };
    let Mnemonic::Binop(Binary {
        op: op @ Binop::Int(IntBinop::Less | IntBinop::SLess),
        lhs,
        rhs,
    }) = host.insn_ref(condition_id).mnemonic()
    else {
        return None;
    };

    let signed = matches!(op, Binop::Int(IntBinop::SLess));
    let induction = lhs.qualify(condition_id.func).as_block_param()?;
    let bound = numeric_const(host, rhs.qualify(condition_id.func))?;
    Some((induction, bound, signed))
}

fn first_body_block<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    analysis: &LoopAnalysis,
    loop_nodes: &HashSet<BlockId>,
    edge: BackEdge,
    cbranch: &CBranch,
) -> Option<BlockId> {
    if cbranch.success_block == cbranch.failure_block {
        return None;
    }

    // The header CBranch's targets are body-local indices in the header's arena.
    let sb = BlockId::new(edge.header.func, cbranch.success_block);
    let fb = BlockId::new(edge.header.func, cbranch.failure_block);
    let success_is_body = loop_nodes.contains(&sb);
    let failure_is_body = loop_nodes.contains(&fb);

    // The trip-count formula in `condition_bound` treats the header comparison
    // (`i < bound`) as the loop-*continue* condition, i.e. it assumes the body
    // executes precisely while the condition is true. That only holds when the
    // condition-true edge enters the body. If the true edge exits the loop
    // instead (body on the failure branch), the polarity is inverted and the
    // computed iteration count would be wrong, so refuse to recognize the loop.
    let body = match (success_is_body, failure_is_body) {
        (true, false) => sb,
        _ => return None,
    };

    if !analysis.dominators.dominates(edge.header, body) {
        return None;
    }
    if !analysis
        .postdominators
        .get(&body)
        .is_some_and(|postdoms| postdoms.contains(&edge.header))
    {
        return None;
    }

    if !can_reach(host, body, edge.latch, loop_nodes) {
        return None;
    }

    Some(body)
}

fn loop_initial_value<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    header: BlockId,
    loop_nodes: &HashSet<BlockId>,
    param_index: usize,
) -> Option<u64> {
    let header_ref = host.block_ref(header);
    let mut preheaders = header_ref
        .predecessors()
        .filter_map(|(_, pred)| (!loop_nodes.contains(&pred)).then_some(pred));
    let preheader = preheaders.next()?;
    if preheaders.next().is_some() {
        return None;
    }

    let term_id = *host.block_ref(preheader).instruction_ids().last()?;
    let Mnemonic::Branch(Branch { target, args }) = host.insn_ref(term_id).mnemonic() else {
        return None;
    };
    if BlockId::new(preheader.func, *target) != header {
        return None;
    }
    numeric_const(host, args.get(param_index)?.qualify(preheader.func))
}

fn can_reach<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    from: BlockId,
    to: BlockId,
    allowed: &HashSet<BlockId>,
) -> bool {
    let mut seen = HashSet::default();
    let mut worklist = VecDeque::from([from]);

    while let Some(block) = worklist.pop_front() {
        if block == to {
            return true;
        }
        if !seen.insert(block) {
            continue;
        }
        for (_, succ) in host.block_ref(block).successors() {
            if allowed.contains(&succ) {
                worklist.push_back(succ);
            }
        }
    }

    false
}

fn latch_step<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    body: BlockId,
    header: BlockId,
    param_index: usize,
    induction: BlockParamId,
) -> Option<u64> {
    let body_ref = host.block_ref(body);
    if body_ref.successors().count() != 1 {
        return None;
    }

    let term_id = *body_ref.instruction_ids().last()?;
    let Mnemonic::Branch(Branch { target, args }) = host.insn_ref(term_id).mnemonic() else {
        return None;
    };
    if BlockId::new(body.func, *target) != header {
        return None;
    }
    induction_increment(host, args.get(param_index)?.qualify(body.func), induction)
}

fn induction_increment<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    value: ValueId,
    induction: BlockParamId,
) -> Option<u64> {
    let ValueId::Instruction(id) = value else {
        return None;
    };
    let Mnemonic::Binop(Binary {
        op: Binop::Int(IntBinop::Add),
        lhs,
        rhs,
    }) = host.insn_ref(id).mnemonic()
    else {
        return None;
    };

    let induction_value = ValueId::BlockParam(induction);
    let (lhs, rhs) = (lhs.qualify(id.func), rhs.qualify(id.func));
    if lhs == induction_value {
        numeric_const(host, rhs)
    } else if rhs == induction_value {
        numeric_const(host, lhs)
    } else {
        None
    }
}

fn numeric_const<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    value: ValueId,
) -> Option<u64> {
    let id = value.as_literal()?;
    let literal = &host.shared().values.literals[id];
    if literal.symbolic.is_some() {
        return None;
    }
    let size = host.shared().types.size_of(literal.type_id);
    Some(if size >= 8 {
        literal.value
    } else {
        literal.value & ((1u64 << (size * 8)) - 1)
    })
}

fn format_loop_comment<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    lp: &SimpleLoop,
) -> String {
    let induction = host.param_ref(lp.induction);
    let body = host
        .block_ref(lp.body)
        .name()
        .map(str::to_owned)
        .unwrap_or_else(|| format!("bb_{:x}", usize::from(lp.body.local)));
    let latch = host
        .block_ref(lp.latch)
        .name()
        .map(str::to_owned)
        .unwrap_or_else(|| format!("bb_{:x}", usize::from(lp.latch.local)));
    format!(
        "{COMMENT_PREFIX} simple induction {} = {:#x}; {} < {:#x}; {} += {:#x}; iterations={}; body=<{}>; latch=<{}>",
        induction, lp.initial, induction, lp.bound, induction, lp.step, lp.iterations, body, latch
    )
}

fn merge_loop_comment(existing: Option<&str>, loop_comment: Option<&str>) -> Option<String> {
    let mut lines = existing
        .into_iter()
        .flat_map(str::lines)
        .filter(|line| !line.trim_start().starts_with(COMMENT_PREFIX))
        .map(str::to_owned)
        .collect::<Vec<_>>();

    if let Some(loop_comment) = loop_comment {
        lines.push(loop_comment.to_owned());
    }

    (!lines.is_empty()).then(|| lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use qcode::{
        context::Context,
        value::{BasicBlock, FunctionBody},
    };
    use qcode_macro::qcode;

    use super::*;
    use crate::test_util::run_function_pass;

    #[test]
    fn annotates_simple_constant_bound_induction_loop() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <header @i=0x0 @sum=0x0>;
            <header @i:i64 @sum:i64>
                %cond = @i < 0x3;
                if %cond goto <body> else goto <exit>;
            <body>
                %sum_next = @sum + @i;
                %i_next = @i + 0x1;
                goto <header @i=%i_next @sum=%sum_next>;
            <exit>
                return at 0x0;
            "
        );

        assert!(run_function_pass::<RecognizeSimpleLoops>(&mut ctx, test).unwrap());

        let header = BasicBlock::from_id(&ctx, header);
        let comment = header.comment().expect("header should be annotated");
        assert!(comment.contains("loop_unroll: simple induction @i = 0x0"));
        assert!(comment.contains("@i < 0x3"));
        assert!(comment.contains("@i += 0x1"));
        assert!(comment.contains("iterations=3"));
    }

    #[test]
    fn signed_loop_with_negative_initial_computes_trip_count() {
        // Regression: `for (int i = -10; i < 0; i++)` lowers to a signed
        // comparison (`s<`) with initial = -10 stored as 0xffff…fff6. Computing
        // the trip count with unsigned arithmetic wrongly yields 0 iterations,
        // deleting the loop body. The correct count is 10.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <header @i=0xfffffffffffffff6 @sum=0x0>;
            <header @i:i64 @sum:i64>
                %cond = @i s< 0x0;
                if %cond goto <body> else goto <exit>;
            <body>
                %sum_next = @sum + @i;
                %i_next = @i + 0x1;
                goto <header @i=%i_next @sum=%sum_next>;
            <exit>
                return at 0x0;
            "
        );

        assert!(run_function_pass::<RecognizeSimpleLoops>(&mut ctx, test).unwrap());

        let header = BasicBlock::from_id(&ctx, header);
        let comment = header.comment().expect("header should be annotated");
        assert!(comment.contains("iterations=10"), "comment: {comment}");
    }

    #[test]
    fn annotates_loop_with_multiple_body_blocks() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <header @i=0x0 @sum=0x0>;
            <header @i:i64 @sum:i64>
                %cond = @i < 0x4;
                if %cond goto <first> else goto <exit>;
            <first>
                %partial = @sum + @i;
                goto <second>;
            <second>
                %sum_next = %partial + 0x2;
                goto <latch>;
            <latch>
                %i_next = @i + 0x1;
                goto <header @i=%i_next @sum=%sum_next>;
            <exit>
                return at 0x0;
            "
        );

        assert!(run_function_pass::<RecognizeSimpleLoops>(&mut ctx, test).unwrap());

        let header = BasicBlock::from_id(&ctx, header);
        let comment = header.comment().expect("header should be annotated");
        assert!(comment.contains("body=<first>"));
        assert!(comment.contains("latch=<latch>"));
        assert!(comment.contains("iterations=4"));
    }

    #[test]
    fn rejects_body_branch_that_could_be_break_or_continue() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <header @i=0x0>;
            <header @i:i64>
                %cond = @i < 0x3;
                if %cond goto <body> else goto <exit>;
            <body>
                %i_next = @i + 0x1;
                %keep_going = @i != 0x1;
                if %keep_going goto <header @i=%i_next> else goto <exit>;
            <exit>
                return at 0x0;
            "
        );

        assert!(!run_function_pass::<RecognizeSimpleLoops>(&mut ctx, test).unwrap());
        assert!(BasicBlock::from_id(&ctx, header).comment().is_none());
    }

    #[test]
    fn rejects_inverted_condition_polarity() {
        // The header compares `@i < 0x3` but the condition-*true* edge exits the
        // loop while the false edge enters the body. The trip-count formula
        // assumes "continue while i < bound", so accepting this loop would emit
        // the wrong number of unrolled copies. The recognizer must reject it.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <header @i=0x0 @sum=0x0>;
            <header @i:i64 @sum:i64>
                %cond = @i < 0x3;
                if %cond goto <exit> else goto <body>;
            <body>
                %sum_next = @sum + @i;
                %i_next = @i + 0x1;
                goto <header @i=%i_next @sum=%sum_next>;
            <exit>
                return at 0x0;
            "
        );

        assert!(!run_function_pass::<RecognizeSimpleLoops>(&mut ctx, test).unwrap());
        assert!(BasicBlock::from_id(&ctx, header).comment().is_none());
    }

    #[test]
    fn rejects_non_additive_latch_update() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <header @i=0x0>;
            <header @i:i64>
                %cond = @i < 0x3;
                if %cond goto <body> else goto <exit>;
            <body>
                %i_next = @i * 0x2;
                goto <header @i=%i_next>;
            <exit>
                return at 0x0;
            "
        );

        assert!(!run_function_pass::<RecognizeSimpleLoops>(&mut ctx, test).unwrap());
        assert!(BasicBlock::from_id(&ctx, header).comment().is_none());
    }

    #[test]
    fn unrolls_simple_loop_with_known_count_under_limit() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <header @i=0x0 @sum=0x0>;
            <header @i:i64 @sum:i64>
                %cond = @i < 0x3;
                if %cond goto <body> else goto <exit>;
            <body>
                %sum_next = @sum + @i;
                %i_next = @i + 0x1;
                goto <header @i=%i_next @sum=%sum_next>;
            <exit @result:i64>
                return at @result;
            "
        );

        assert!(run_function_pass::<UnrollSimpleLoops>(&mut ctx, test).unwrap());

        let blocks = FunctionBody::from_id(&ctx, test)
            .iter()
            .map(|block| block.id)
            .collect::<Vec<_>>();
        assert!(!blocks.contains(&header));
        assert!(!blocks.contains(&body));
        assert_eq!(blocks.len(), 5);
        assert_eq!(BasicBlock::from_id(&ctx, entry).successors().count(), 1);
    }

    #[test]
    fn unrolls_loop_body_with_memory_and_cast_instructions() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <header @i=0x0 @ptr=0x1000>;
            <header @i:i64 @ptr:i64>
                %cond = @i < 0x3;
                if %cond goto <body> else goto <exit>;
            <body>
                %addr = @ptr + @i;
                %byte = load(ram:1, %addr);
                %wide = zext(i64, %byte);
                %mixed = %wide ^ @i;
                %byte_next = %mixed + 0x1;
                store(ram:8, %addr <- %byte_next);
                %i_next = @i + 0x1;
                goto <header @i=%i_next @ptr=@ptr>;
            <exit>
                return at 0x0;
            "
        );

        assert!(run_function_pass::<UnrollSimpleLoops>(&mut ctx, test).unwrap());

        let blocks = FunctionBody::from_id(&ctx, test)
            .iter()
            .map(|block| block.id)
            .collect::<Vec<_>>();
        assert!(!blocks.contains(&header));
        assert!(!blocks.contains(&body));

        let stores = FunctionBody::from_id(&ctx, test)
            .iter()
            .flat_map(|block| block.instruction_ids().to_vec())
            .filter(|&insn| {
                matches!(
                    qcode::value::Instruction::from_id(&ctx, insn).mnemonic(),
                    Mnemonic::Store(_)
                )
            })
            .count();
        assert_eq!(stores, 3);
    }

    /// Reproduction: a loop-carried induction value passed to an *exit param* and
    /// used *after* the loop (the shape of a returned register write-set referencing
    /// the loop counter). After unrolling deletes the loop, that use must remain
    /// defined — not dangle on the removed header param.
    #[test]
    fn live_out_induction_value_stays_defined_after_unroll() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <header @i=0x0>;
            <header @i:i64>
                %cond = @i < 0x3;
                if %cond goto <body> else goto <exit @fin=@i>;
            <body>
                %i_next = @i + 0x1;
                goto <header @i=%i_next>;
            <exit @fin:i64>
                %p = (@fin);
                return at @fin;
            "
        );

        assert!(run_function_pass::<UnrollSimpleLoops>(&mut ctx, test).unwrap());

        // The exit block (which holds the live-out use) survives and keeps a param,
        // and every predecessor edge into it carries an argument for that param —
        // i.e. the live-out value is not dangling.
        let exit_params = BasicBlock::from_id(&ctx, exit).params().count();
        assert_eq!(exit_params, 1, "exit keeps its live-out param");
        let preds: Vec<_> = BasicBlock::from_id(&ctx, exit)
            .predecessors()
            .map(|(_, p)| p)
            .collect();
        assert!(!preds.is_empty(), "exit must still be reachable");
        for pred in preds {
            let args = branch_args_to(qcode::value::ModuleView::new(&ctx), pred, exit);
            assert!(
                args.is_some_and(|a| a.len() == exit_params),
                "pred {pred:?} must pass an arg for the live-out exit param"
            );
        }

        // The following `simplify_cfg` (which runs right after unrolling in the
        // pipeline) merges the now-single-pred exit into its predecessor: the
        // live-out use must be rewritten to the incoming value, never left dangling
        // on the removed exit param.
        let _ = run_function_pass::<crate::cfg::SimplifyCfg>(&mut ctx, test);
        let defined: rustc_hash::FxHashSet<ValueId> = FunctionBody::from_id(&ctx, test)
            .iter()
            .flat_map(|b| {
                b.params()
                    .map(|p| ValueId::BlockParam(p.id))
                    .chain(b.instruction_ids().iter().map(|&i| ValueId::Instruction(i)))
                    .collect::<Vec<_>>()
            })
            .collect();
        for insn in FunctionBody::from_id(&ctx, test)
            .iter()
            .flat_map(|b| b.instruction_ids().to_vec())
        {
            for operand in qcode::value::Instruction::from_id(&ctx, insn).operands() {
                if matches!(operand, ValueId::Instruction(_) | ValueId::BlockParam(_)) {
                    assert!(
                        defined.contains(&operand),
                        "operand {operand:?} of {insn:?} is undefined after simplify_cfg (dangling)"
                    );
                }
            }
        }
    }

    /// Asserts no instruction in `fid` references a value (instruction or block
    /// param) that is not defined anywhere in the function — i.e. no dangling use.
    fn assert_no_dangling(ctx: &Context, fid: FunctionId) {
        let defined: rustc_hash::FxHashSet<ValueId> = FunctionBody::from_id(ctx, fid)
            .iter()
            .flat_map(|b| {
                b.params()
                    .map(|p| ValueId::BlockParam(p.id))
                    .chain(b.instruction_ids().iter().map(|&i| ValueId::Instruction(i)))
                    .collect::<Vec<_>>()
            })
            .collect();
        for insn in FunctionBody::from_id(ctx, fid)
            .iter()
            .flat_map(|b| b.instruction_ids().to_vec())
        {
            for operand in qcode::value::Instruction::from_id(ctx, insn).operands() {
                if matches!(operand, ValueId::Instruction(_) | ValueId::BlockParam(_)) {
                    assert!(
                        defined.contains(&operand),
                        "operand {operand:?} of {insn:?} is undefined (dangling)"
                    );
                }
            }
        }
    }

    /// Reproduction: the induction variable used **directly** in the exit block
    /// (no exit param). This is valid SSA — the header dominates the exit — and is
    /// the shape of `fn_449740`'s `pack(ECX=@counter)` register write-set. Unrolling
    /// deletes the header, so it must replace the induction param's live-out uses
    /// with its final value, or they dangle.
    #[test]
    fn direct_live_out_use_of_induction_var_is_rewritten() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <header @i=0x0>;
            <header @i:i64>
                %cond = @i < 0x3;
                if %cond goto <body> else goto <exit>;
            <body>
                %i_next = @i + 0x1;
                goto <header @i=%i_next>;
            <exit>
                %p = (@i);
                return at @i;
            "
        );

        assert!(run_function_pass::<UnrollSimpleLoops>(&mut ctx, test).unwrap());
        assert_no_dangling(&ctx, test);
        let _ = run_function_pass::<crate::cfg::SimplifyCfg>(&mut ctx, test);
        assert_no_dangling(&ctx, test);
    }

    #[test]
    fn does_not_unroll_loop_at_iteration_limit() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <header @i=0x0>;
            <header @i:i64>
                %cond = @i < 0xa;
                if %cond goto <body> else goto <exit>;
            <body>
                %i_next = @i + 0x1;
                goto <header @i=%i_next>;
            <exit>
                return at 0x0;
            "
        );

        assert!(!run_function_pass::<UnrollSimpleLoops>(&mut ctx, test).unwrap());
        let blocks = FunctionBody::from_id(&ctx, test)
            .iter()
            .map(|block| block.id)
            .collect::<Vec<_>>();
        assert!(blocks.contains(&header));
        assert!(blocks.contains(&body));
    }
}
