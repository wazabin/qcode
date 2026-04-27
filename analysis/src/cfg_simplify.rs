use qcode::{
    context::Context,
    value::{BasicBlock, FunctionId, insn::Mnemonic},
};

/// Merges basic blocks in `function_id` wherever the conditions allow:
/// if block A has exactly one successor B, B has exactly one predecessor A,
/// and A ends with an unconditional `Branch { target: B }`, then A and B are
/// merged into A (the branch is removed and B's instructions are appended).
///
/// The pass repeats until no further merges are possible.
pub fn simplify_cfg(ctx: &mut Context, function_id: FunctionId) {
    loop {
        let blocks = ctx.values.functions[function_id].blocks.clone();
        let mut merged = false;

        'outer: for a_id in blocks {
            // Collect at most 2 successors to check the "exactly one" condition.
            // Collecting eagerly releases the immutable borrow before any mutation.
            let a_succs: Vec<_> = BasicBlock::from_id(&*ctx, a_id)
                .successors()
                .take(2)
                .collect();

            let Some(&(edge_ab, b_id)) = a_succs.first() else {
                continue;
            };

            if a_succs.len() != 1 {
                continue; // A has more than one successor
            }

            if b_id == a_id {
                continue; // self-loop
            }

            if BasicBlock::from_id(&*ctx, b_id).predecessors().count() != 1 {
                continue;
            }

            // A's terminal must be an unconditional Branch to B.
            let a_terminal = ctx.values.basic_blocks[a_id].instructions.last().copied();
            let is_branch_to_b = a_terminal
                .map(|id| {
                    matches!(
                        ctx.values.instructions[id].mnemonic(),
                        Mnemonic::Branch(b) if b.target == b_id
                    )
                })
                .unwrap_or(false);

            if !is_branch_to_b {
                continue;
            }

            BasicBlock::from_id_mut(ctx, a_id).absorb_block(b_id, edge_ab, function_id);
            merged = true;
            break 'outer;
        }

        if !merged {
            break;
        }
    }
}

#[cfg(test)]
mod tests {

    use qcode::{
        context::Context,
        value::{
            BasicBlock, Function, InstructionRef,
            insn::{Branch, Mnemonic},
        },
    };

    use super::simplify_cfg;

    fn make_ctx() -> Context<'static> {
        Context::new()
    }

    /// Push an unconditional Branch instruction into block `from` targeting `to`,
    /// and add a CFG edge from `from` to `to`.
    fn add_branch(ctx: &mut Context, from: qcode::value::BlockId, to: qcode::value::BlockId) {
        let insn = InstructionRef::from_mnemonic(
            ctx,
            Mnemonic::Branch(Branch {
                target: to,
                args: vec![],
            }),
            0,
        )
        .id;
        ctx.values.basic_blocks[from].instructions.push(insn);
        ctx.add_cfg_edge(from, to);
    }

    #[test]
    fn merges_two_block_chain() {
        let mut ctx = make_ctx();

        let func_id = Function::make(&mut ctx, "f".into()).unwrap().id;

        let a = BasicBlock::make(&mut ctx).in_function(func_id).id;
        let b = BasicBlock::make(&mut ctx).in_function(func_id).id;
        add_branch(&mut ctx, a, b);

        simplify_cfg(&mut ctx, func_id);

        let blocks: Vec<_> = Function::from_id(&ctx, func_id)
            .blocks()
            .map(|b| b.id)
            .collect();
        assert_eq!(blocks.len(), 1, "two-block chain should merge into one");
        assert_eq!(blocks[0], a);
    }

    #[test]
    fn merges_three_block_chain() {
        let mut ctx = make_ctx();

        let func_id = Function::make(&mut ctx, "f".into()).unwrap().id;

        let a = BasicBlock::make(&mut ctx).in_function(func_id).id;
        let b = BasicBlock::make(&mut ctx).in_function(func_id).id;
        let c = BasicBlock::make(&mut ctx).in_function(func_id).id;

        add_branch(&mut ctx, a, b);
        add_branch(&mut ctx, b, c);

        simplify_cfg(&mut ctx, func_id);

        let blocks: Vec<_> = Function::from_id(&ctx, func_id)
            .blocks()
            .map(|b| b.id)
            .collect();
        assert_eq!(blocks.len(), 1, "three-block chain should collapse to one");
    }

    #[test]
    fn no_merge_when_a_has_two_successors() {
        // A→B and A→C (diamond entry): A has two successors, no merge possible.
        let mut ctx = make_ctx();
        let func_id = Function::make(&mut ctx, "f".into()).unwrap().id;
        let a = BasicBlock::make(&mut ctx).in_function(func_id).id;
        let b = BasicBlock::make(&mut ctx).in_function(func_id).id;
        let c = BasicBlock::make(&mut ctx).in_function(func_id).id;
        // Add a CBranch-like situation: two outgoing edges from A.
        ctx.add_cfg_edge(a, b);
        ctx.add_cfg_edge(a, c);

        simplify_cfg(&mut ctx, func_id);

        let blocks: Vec<_> = Function::from_id(&ctx, func_id)
            .blocks()
            .map(|b| b.id)
            .collect();
        assert_eq!(blocks.len(), 3, "diamond entry should not be merged");
    }

    #[test]
    fn no_merge_when_b_has_two_predecessors() {
        // A→B and D→B: B has two predecessors, no merge.
        let mut ctx = make_ctx();
        let func_id = Function::make(&mut ctx, "f".into()).unwrap().id;
        let a = BasicBlock::make(&mut ctx).in_function(func_id).id;
        let b = BasicBlock::make(&mut ctx).in_function(func_id).id;
        let d = BasicBlock::make(&mut ctx).in_function(func_id).id;
        add_branch(&mut ctx, a, b);
        ctx.add_cfg_edge(d, b);

        simplify_cfg(&mut ctx, func_id);

        let blocks: Vec<_> = Function::from_id(&ctx, func_id)
            .blocks()
            .map(|b| b.id)
            .collect();
        assert_eq!(blocks.len(), 3, "B has two predecessors, should not merge");
    }

    #[test]
    fn parent_cleared_on_merged_block() {
        let mut ctx = make_ctx();
        let func_id = Function::make(&mut ctx, "f".into()).unwrap().id;
        let a = BasicBlock::make(&mut ctx).in_function(func_id).id;
        let b = BasicBlock::make(&mut ctx).in_function(func_id).id;

        add_branch(&mut ctx, a, b);

        {
            let b = BasicBlock::from_id(&ctx, b);
            assert!(b.parent().is_some(), "b should have parent before merge");
        }

        simplify_cfg(&mut ctx, func_id);

        {
            let a = BasicBlock::from_id(&ctx, a);
            let b = BasicBlock::from_id(&ctx, b);
            assert!(
                b.parent().is_none(),
                "b's parent should be cleared after merge"
            );
            assert!(a.parent().is_some(), "a should still have a parent");
        }
    }
}
