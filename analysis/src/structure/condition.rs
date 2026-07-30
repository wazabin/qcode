//! Recovery of a block's control-flow exit and its per-edge branch predicates.

use qcode::value::{BlockId, BlockRef, ValueId, block::EdgeId, insn::Mnemonic};

/// An atomic branch predicate: a boolean-valued IR operand together with the
/// polarity in which it must hold for a particular edge to be taken.
///
/// This is the leaf atom out of which later phases build reaching-condition
/// formulas. `value` is the condition operand of a [`Mnemonic::CBranch`]; the
/// IR treats it as "taken when non-zero".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeCondition {
    /// The boolean-valued IR operand of the conditional branch.
    pub value: ValueId,
    /// When `false`, the edge is taken iff `value` is true (the `cbranch`
    /// success side). When `true`, it is taken iff `value` is false (the
    /// fall-through / failure side).
    pub negated: bool,
}

impl EdgeCondition {
    /// The predicate "`value` is true".
    pub fn when_true(value: ValueId) -> Self {
        Self {
            value,
            negated: false,
        }
    }

    /// The predicate "`value` is false".
    pub fn when_false(value: ValueId) -> Self {
        Self {
            value,
            negated: true,
        }
    }

    /// The same predicate with inverted polarity.
    #[must_use]
    pub fn negated(self) -> Self {
        Self {
            value: self.value,
            negated: !self.negated,
        }
    }
}

/// The classified control-flow exit of a basic block: its successor edges tagged
/// with the structural role each plays, recovered from the terminator.
///
/// Recover one with [`block_exit`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockExit {
    /// The block leaves the function (`return`); it has no CFG successors to
    /// structure.
    Return,
    /// A single successor, taken unconditionally. Covers unconditional branches
    /// as well as calls, whose fall-through continuation is a lone successor.
    Goto { edge: EdgeId, target: BlockId },
    /// A two-way conditional branch. `true_*` is taken when `condition` is true,
    /// `false_*` when it is false.
    Branch {
        /// The branch predicate as an IR value.
        condition: ValueId,
        true_edge: EdgeId,
        true_target: BlockId,
        false_edge: EdgeId,
        false_target: BlockId,
    },
    /// A resolved multi-way dispatch. Unlike [`BlockExit::Indirect`], each
    /// successor is selected by known scrutinee values, so the backend can emit
    /// a `switch` instead of one goto per successor.
    Switch {
        /// The value dispatched on.
        scrutinee: ValueId,
        /// `(case values, target)` grouped by target in first-seen order.
        /// Several values may select one target — a table whose slots repeat is
        /// a shared `case 1: case 6:` arm.
        arms: Vec<(Vec<u64>, BlockId)>,
        /// Where an unlisted value goes, when the dispatch names a block for it.
        default: Option<BlockId>,
        /// Every outgoing edge. Kept alongside `arms` because a dispatch may
        /// reach one target through several parallel edges, one per case.
        edges: Vec<(EdgeId, BlockId)>,
    },
    /// An indirect / computed branch (resolved jump table, indirect tail call).
    /// Its successors carry no recoverable per-edge predicate at this phase.
    Indirect { edges: Vec<(EdgeId, BlockId)> },
    /// No terminator, or one whose successors could not be reconciled with the
    /// CFG edges. Successors are exposed without polarity so callers can still
    /// fall back to gotos.
    Unstructured { edges: Vec<(EdgeId, BlockId)> },
}

impl BlockExit {
    /// The predicate under which `edge` is taken from this block, if this exit
    /// carries per-edge polarity. `None` for unconditional, indirect, or
    /// unstructured exits, and for edges not belonging to this exit.
    pub fn condition_for(&self, edge: EdgeId) -> Option<EdgeCondition> {
        match *self {
            BlockExit::Branch {
                condition,
                true_edge,
                false_edge,
                ..
            } => {
                if edge == true_edge {
                    Some(EdgeCondition::when_true(condition))
                } else if edge == false_edge {
                    Some(EdgeCondition::when_false(condition))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// All outgoing edges of this exit, regardless of classification.
    pub fn edges(&self) -> Vec<EdgeId> {
        match self {
            BlockExit::Return => Vec::new(),
            BlockExit::Goto { edge, .. } => vec![*edge],
            BlockExit::Branch {
                true_edge,
                false_edge,
                ..
            } => vec![*true_edge, *false_edge],
            BlockExit::Switch { edges, .. }
            | BlockExit::Indirect { edges }
            | BlockExit::Unstructured { edges } => edges.iter().map(|&(e, _)| e).collect(),
        }
    }
}

/// Finds the outgoing edge of `block` whose target is `target`.
///
/// Returns `None` if there is no such edge, or if `target` is reached by more
/// than one edge (parallel edges are ambiguous for polarity assignment).
fn unique_edge_to(block: &BlockRef<'_, '_>, target: BlockId) -> Option<EdgeId> {
    let mut found = None;
    for (edge, succ) in block.successors() {
        if succ == target {
            if found.is_some() {
                return None; // parallel edges: ambiguous
            }
            found = Some(edge);
        }
    }
    found
}

/// Recovers the classified control-flow exit of `block` from its terminator.
///
/// This is the phase-0 primitive: it maps the abstract `from -> to` CFG edges to
/// the roles the terminator assigns them, exposing which edge is the taken side
/// of a conditional branch and the predicate that governs it.
pub fn block_exit(block: BlockRef<'_, '_>) -> BlockExit {
    let Some(term) = block.iter().last().filter(|insn| insn.is_terminator()) else {
        return unstructured(&block);
    };

    match term.mnemonic() {
        Mnemonic::Return(_) => BlockExit::Return,

        Mnemonic::Branch(b) => {
            let target = BlockId::new(block.id.func, b.target);
            match unique_edge_to(&block, target) {
                Some(edge) => BlockExit::Goto { edge, target },
                None => unstructured(&block),
            }
        }

        Mnemonic::CBranch(c) => {
            // The invariant is that the two sides are distinct blocks; if they
            // collapse to one, the condition is irrelevant and it is really an
            // unconditional edge.
            if c.success_block == c.failure_block {
                let target = BlockId::new(block.id.func, c.success_block);
                return match unique_edge_to(&block, target) {
                    Some(edge) => BlockExit::Goto { edge, target },
                    None => unstructured(&block),
                };
            }
            match (
                unique_edge_to(&block, BlockId::new(block.id.func, c.success_block)),
                unique_edge_to(&block, BlockId::new(block.id.func, c.failure_block)),
            ) {
                (Some(true_edge), Some(false_edge)) => BlockExit::Branch {
                    condition: c.condition.qualify(block.id.func),
                    true_edge,
                    true_target: BlockId::new(block.id.func, c.success_block),
                    false_edge,
                    false_target: BlockId::new(block.id.func, c.failure_block),
                },
                _ => unstructured(&block),
            }
        }

        // A resolved dispatch: the terminator records which value selects which
        // successor, so group the cases by target and hand the backend a real
        // multi-way exit. Repeated targets become one arm with several labels.
        Mnemonic::Switch(switch) => {
            let func = block.id.func;
            // Group by target *and* argument list: the arm body is emitted once,
            // so two cases reaching one block with different block arguments
            // cannot share an arm. (A jump table passes none, so in practice
            // every repeat merges.)
            let mut arms: Vec<(Vec<u64>, BlockId)> = Vec::new();
            let mut arm_args: Vec<&[qcode::value::LocalValueId]> = Vec::new();
            for case in &switch.cases {
                let target = BlockId::new(func, case.target);
                let existing = arms
                    .iter()
                    .position(|(_, t)| *t == target)
                    .filter(|&i| arm_args[i] == case.args.as_slice());
                match existing {
                    Some(i) => arms[i].0.push(case.value),
                    None => {
                        arms.push((vec![case.value], target));
                        arm_args.push(&case.args);
                    }
                }
            }
            BlockExit::Switch {
                scrutinee: switch.scrutinee.qualify(func),
                arms,
                default: switch.default.map(|d| BlockId::new(func, d)),
                edges: successors(&block),
            }
        }

        // Indirect branches have their real successors materialized as edges
        // (e.g. resolved jump tables) but no per-edge predicate.
        Mnemonic::BranchInd(_) => BlockExit::Indirect {
            edges: successors(&block),
        },

        // A call terminates the block but flows on to a single continuation.
        Mnemonic::Call(_) | Mnemonic::CallInd(_) => match successors(&block).as_slice() {
            [] => BlockExit::Return,
            [(edge, target)] => BlockExit::Goto {
                edge: *edge,
                target: *target,
            },
            _ => unstructured(&block),
        },

        // Not a control-flow terminator we model; expose edges opaquely.
        _ => unstructured(&block),
    }
}

fn successors(block: &BlockRef<'_, '_>) -> Vec<(EdgeId, BlockId)> {
    block.successors().collect()
}

fn unstructured(block: &BlockRef<'_, '_>) -> BlockExit {
    BlockExit::Unstructured {
        edges: successors(block),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{context::Context, value::BasicBlock};
    use qcode_macro::qcode;

    #[test]
    fn return_block_has_no_successors_to_structure() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                return at i64 0;
            "
        );
        let exit = block_exit(BasicBlock::from_id(&ctx, block));
        assert_eq!(exit, BlockExit::Return);
        assert!(exit.edges().is_empty());
    }

    #[test]
    fn unconditional_branch_is_goto() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                goto <done>;
            <done>
                goto <0x1001>;
            "
        );
        let exit = block_exit(BasicBlock::from_id(&ctx, block));
        match exit {
            BlockExit::Goto { target, edge } => {
                assert_eq!(target, done);
                assert_eq!(exit_edges(&ctx, block), vec![edge]);
            }
            other => panic!("expected goto, got {other:?}"),
        }
    }

    #[test]
    fn conditional_branch_recovers_polarity() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;

            <block>
                %c = load(cond:1, &cond);
                if %c goto <then_lbl> else goto <else_lbl>;
            <then_lbl>
                goto <0x1001>;
            <else_lbl>
                goto <0x1002>;
            "
        );
        let exit = block_exit(BasicBlock::from_id(&ctx, block));
        let BlockExit::Branch {
            condition,
            true_edge,
            true_target,
            false_edge,
            false_target,
        } = exit
        else {
            panic!("expected branch, got {exit:?}");
        };

        // The `if %c goto <then>` side is the true side.
        assert_eq!(true_target, then_lbl);
        assert_eq!(false_target, else_lbl);
        assert_ne!(true_edge, false_edge);

        // The predicate polarity matches the edge role.
        assert_eq!(
            exit.condition_for(true_edge),
            Some(EdgeCondition::when_true(condition))
        );
        assert_eq!(
            exit.condition_for(false_edge),
            Some(EdgeCondition::when_false(condition))
        );
    }

    fn exit_edges(ctx: &Context<'_>, block: qcode::value::BlockId) -> Vec<EdgeId> {
        block_exit(BasicBlock::from_id(ctx, block)).edges()
    }
}
