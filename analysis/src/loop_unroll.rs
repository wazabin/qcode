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

use std::collections::{HashMap, HashSet, VecDeque};

use jstd::graph::analysis::{DominatorTree, compute_dominators, compute_postdominators};
use qcode::{
    context::Context,
    value::{
        BasicBlock, BlockParam, BlockParamId, Function, FunctionId, ValueId,
        block::BlockId,
        insn::{Binary, Binop, Branch, CBranch, IntBinop, Mnemonic},
    },
};

use crate::{FunctionPass, PipelineEnv};

const COMMENT_PREFIX: &str = "loop_unroll:";

#[derive(Default)]
pub struct RecognizeSimpleLoops;

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

    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        Ok(recognize_simple_loops(ctx, fun_id))
    }
}

crate::register_function_pass!(RecognizeSimpleLoops);

pub fn recognize_simple_loops(ctx: &mut Context, fun_id: FunctionId) -> bool {
    let Some(analysis) = LoopAnalysis::compute(ctx, fun_id) else {
        return false;
    };
    let block_ids = analysis.block_ids();
    let recognized = analysis
        .backedges
        .iter()
        .filter_map(|&edge| recognize_simple_loop(ctx, &analysis, edge))
        .fold(HashMap::<BlockId, Vec<SimpleLoop>>::new(), |mut acc, lp| {
            acc.entry(lp.header()).or_default().push(lp);
            acc
        });

    let updates = block_ids
        .iter()
        .map(|&block| {
            let loop_comment = recognized
                .get(&block)
                .and_then(|loops| (loops.len() == 1).then(|| format_loop_comment(ctx, &loops[0])));
            let current = BasicBlock::from_id(ctx, block).comment().map(str::to_owned);
            (
                block,
                merge_loop_comment(current.as_deref(), loop_comment.as_deref()),
            )
        })
        .collect::<Vec<_>>();

    let mut changed = false;
    for (block, comment) in updates {
        let current = BasicBlock::from_id(ctx, block).comment().map(str::to_owned);
        if current != comment {
            BasicBlock::from_id_mut(ctx, block).set_comment(comment);
            changed = true;
        }
    }
    changed
}

impl LoopAnalysis {
    fn compute(ctx: &Context, fun_id: FunctionId) -> Option<Self> {
        let function = Function::from_id(ctx, fun_id);
        let root = function.root()?.id;
        let block_ids = function.iter().map(|block| block.id).collect::<Vec<_>>();
        if block_ids.is_empty() {
            return None;
        }

        let node_set = block_ids.iter().copied().collect::<HashSet<_>>();
        let exit_set = block_ids
            .iter()
            .copied()
            .filter(|&block| {
                BasicBlock::from_id(ctx, block)
                    .successors()
                    .next()
                    .is_none()
            })
            .collect::<HashSet<_>>();
        if exit_set.is_empty() {
            return None;
        }

        let dominators = compute_dominators(ctx, root);
        let postdominators = compute_postdominators(ctx, &block_ids, &node_set, &exit_set);
        let mut backedges = Vec::new();
        for &latch in &block_ids {
            let successors = BasicBlock::from_id(ctx, latch)
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
        blocks.sort_by_key(|id| usize::from(*id));
        blocks
    }
}

impl SimpleLoop {
    fn header(&self) -> BlockId {
        self.header
    }
}

fn recognize_simple_loop(
    ctx: &Context,
    analysis: &LoopAnalysis,
    edge: BackEdge,
) -> Option<SimpleLoop> {
    let header = edge.header;
    let cbranch = header_cbranch(ctx, header)?;
    let (induction, bound) = condition_bound(ctx, cbranch.condition)?;
    let header_params = BasicBlock::from_id(ctx, header)
        .params()
        .map(|param| param.id)
        .collect::<Vec<_>>();
    let param_index = header_params.iter().position(|&param| param == induction)?;

    let loop_nodes = natural_loop(ctx, edge);
    let body = first_body_block(ctx, analysis, &loop_nodes, edge, &cbranch)?;
    let initial = loop_initial_value(ctx, header, &loop_nodes, param_index)?;
    let step = latch_step(ctx, edge.latch, header, param_index, induction)?;
    if step == 0 {
        return None;
    }

    let iterations = if initial >= bound {
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

fn natural_loop(ctx: &Context, edge: BackEdge) -> HashSet<BlockId> {
    let mut nodes = HashSet::from([edge.header, edge.latch]);
    let mut worklist = VecDeque::from([edge.latch]);

    while let Some(block) = worklist.pop_front() {
        for (_, pred) in BasicBlock::from_id(ctx, block).predecessors() {
            if nodes.insert(pred) && pred != edge.header {
                worklist.push_back(pred);
            }
        }
    }

    nodes
}

fn header_cbranch(ctx: &Context, header: BlockId) -> Option<CBranch> {
    let block = BasicBlock::from_id(ctx, header);
    let term_id = *block.instruction_ids().last()?;
    match qcode::value::Instruction::from_id(ctx, term_id).mnemonic() {
        Mnemonic::CBranch(cbranch) => Some(cbranch.clone()),
        _ => None,
    }
}

fn condition_bound(ctx: &Context, condition: ValueId) -> Option<(BlockParamId, u64)> {
    let ValueId::Instruction(condition_id) = condition else {
        return None;
    };
    let condition = qcode::value::Instruction::from_id(ctx, condition_id);
    let Mnemonic::Binop(Binary {
        op: Binop::Int(IntBinop::Less | IntBinop::SLess),
        lhs,
        rhs,
    }) = condition.mnemonic()
    else {
        return None;
    };

    let induction = lhs.as_block_param()?;
    let bound = numeric_const(ctx, *rhs)?;
    Some((induction, bound))
}

fn first_body_block(
    ctx: &Context,
    analysis: &LoopAnalysis,
    loop_nodes: &HashSet<BlockId>,
    edge: BackEdge,
    cbranch: &CBranch,
) -> Option<BlockId> {
    if cbranch.success_block == cbranch.failure_block {
        return None;
    }

    let success_is_body = loop_nodes.contains(&cbranch.success_block);
    let failure_is_body = loop_nodes.contains(&cbranch.failure_block);

    let body = match (success_is_body, failure_is_body) {
        (true, false) => cbranch.success_block,
        (false, true) => cbranch.failure_block,
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

    if !can_reach(ctx, body, edge.latch, loop_nodes) {
        return None;
    }

    Some(body)
}

fn loop_initial_value(
    ctx: &Context,
    header: BlockId,
    loop_nodes: &HashSet<BlockId>,
    param_index: usize,
) -> Option<u64> {
    let header_ref = BasicBlock::from_id(ctx, header);
    let mut preheaders = header_ref
        .predecessors()
        .filter_map(|(_, pred)| (!loop_nodes.contains(&pred)).then_some(pred));
    let preheader = preheaders.next()?;
    if preheaders.next().is_some() {
        return None;
    }

    let term_id = *BasicBlock::from_id(ctx, preheader)
        .instruction_ids()
        .last()?;
    let term = qcode::value::Instruction::from_id(ctx, term_id);
    let Mnemonic::Branch(Branch { target, args }) = term.mnemonic() else {
        return None;
    };
    if *target != header {
        return None;
    }
    numeric_const(ctx, *args.get(param_index)?)
}

fn can_reach(ctx: &Context, from: BlockId, to: BlockId, allowed: &HashSet<BlockId>) -> bool {
    let mut seen = HashSet::new();
    let mut worklist = VecDeque::from([from]);

    while let Some(block) = worklist.pop_front() {
        if block == to {
            return true;
        }
        if !seen.insert(block) {
            continue;
        }
        for (_, succ) in BasicBlock::from_id(ctx, block).successors() {
            if allowed.contains(&succ) {
                worklist.push_back(succ);
            }
        }
    }

    false
}

fn latch_step(
    ctx: &Context,
    body: BlockId,
    header: BlockId,
    param_index: usize,
    induction: BlockParamId,
) -> Option<u64> {
    let body_ref = BasicBlock::from_id(ctx, body);
    if body_ref.successors().count() != 1 {
        return None;
    }

    let term_id = *body_ref.instruction_ids().last()?;
    let term = qcode::value::Instruction::from_id(ctx, term_id);
    let Mnemonic::Branch(Branch { target, args }) = term.mnemonic() else {
        return None;
    };
    if *target != header {
        return None;
    }
    induction_increment(ctx, *args.get(param_index)?, induction)
}

fn induction_increment(ctx: &Context, value: ValueId, induction: BlockParamId) -> Option<u64> {
    let ValueId::Instruction(id) = value else {
        return None;
    };
    let insn = qcode::value::Instruction::from_id(ctx, id);
    let Mnemonic::Binop(Binary {
        op: Binop::Int(IntBinop::Add),
        lhs,
        rhs,
    }) = insn.mnemonic()
    else {
        return None;
    };

    let induction_value = ValueId::BlockParam(induction);
    if *lhs == induction_value {
        numeric_const(ctx, *rhs)
    } else if *rhs == induction_value {
        numeric_const(ctx, *lhs)
    } else {
        None
    }
}

fn numeric_const(ctx: &Context, value: ValueId) -> Option<u64> {
    let id = value.as_literal()?;
    let literal = &ctx.values.literals[id];
    if literal.symbolic.is_some() {
        return None;
    }
    let size = ctx.types.size_of(literal.type_id);
    Some(if size >= 8 {
        literal.value
    } else {
        literal.value & ((1u64 << (size * 8)) - 1)
    })
}

fn format_loop_comment(ctx: &Context, lp: &SimpleLoop) -> String {
    let induction = BlockParam::from_id(ctx, lp.induction);
    let body = BasicBlock::from_id(ctx, lp.body)
        .name()
        .map(str::to_owned)
        .unwrap_or_else(|| format!("bb_{:x}", usize::from(lp.body)));
    let latch = BasicBlock::from_id(ctx, lp.latch)
        .name()
        .map(str::to_owned)
        .unwrap_or_else(|| format!("bb_{:x}", usize::from(lp.latch)));
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
    use qcode::{context::Context, value::BasicBlock};
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
                return [0x0];
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
                return [0x0];
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
                return [0x0];
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
                return [0x0];
            "
        );

        assert!(!run_function_pass::<RecognizeSimpleLoops>(&mut ctx, test).unwrap());
        assert!(BasicBlock::from_id(&ctx, header).comment().is_none());
    }
}
