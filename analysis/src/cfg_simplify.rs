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

            // Merging rewrites B's params to the branch's args, so the branch must
            // supply one arg per param. A mismatch means a malformed edge — e.g. a
            // CRT stub's tail `jmp` into another routine that mem2reg gave a param,
            // lifted as an intra-function `goto` carrying no args. Leave such edges
            // unmerged rather than absorbing an unsatisfiable param.
            let b_params = ctx.values.basic_blocks[b_id].params.len();
            let branch_args = a_terminal
                .and_then(|id| match ctx.values.instructions[id].mnemonic() {
                    Mnemonic::Branch(b) => Some(b.args.len()),
                    _ => None,
                })
                .unwrap_or(0);
            if b_params != branch_args {
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
            BasicBlock, Function, ValueId,
            insn::{Binary, Mnemonic},
        },
    };
    use qcode_macro::qcode;

    use super::simplify_cfg;

    fn make_ctx() -> Context<'static> {
        Context::new()
    }

    #[test]
    fn merges_two_block_chain() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <b>;
            <b>
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        let blocks: Vec<_> = Function::from_id(&ctx, f).blocks().map(|b| b.id).collect();
        assert_eq!(blocks.len(), 1, "two-block chain should merge into one");
        assert_eq!(blocks[0], a);
    }

    #[test]
    fn merges_three_block_chain() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <b>;
            <b>
                goto <c>;
            <c>
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        let blocks: Vec<_> = Function::from_id(&ctx, f).blocks().map(|b| b.id).collect();
        assert_eq!(blocks.len(), 1, "three-block chain should collapse to one");
    }

    #[test]
    fn no_merge_when_a_has_two_successors() {
        // A->B and A->C (diamond entry): A has two successors, no merge possible.
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a @cond:i8>
                if @cond goto <b> else goto <c>;
            <b>
                goto <0x1001>;
            <c>
                goto <0x1002>;
            "
        );

        simplify_cfg(&mut ctx, f);

        let blocks: Vec<_> = Function::from_id(&ctx, f).blocks().map(|b| b.id).collect();
        assert_eq!(blocks.len(), 3, "diamond entry should not be merged");
    }

    #[test]
    fn no_merge_when_b_has_two_predecessors() {
        // A->B and D->B: B has two predecessors, no merge.
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <b>;
            <d>
                goto <b>;
            <b>
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        let blocks: Vec<_> = Function::from_id(&ctx, f).blocks().map(|b| b.id).collect();
        assert_eq!(blocks.len(), 3, "B has two predecessors, should not merge");
    }

    #[test]
    fn parent_cleared_on_merged_block() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <b>;
            <b>
                goto <0x1001>;
            "
        );

        {
            let b = BasicBlock::from_id(&ctx, b);
            assert!(b.parent().is_some(), "b should have parent before merge");
        }

        simplify_cfg(&mut ctx, f);

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

    #[test]
    fn merged_block_arguments_are_rewritten_to_branch_args() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a @input:i64>
                goto <b @x=@input>;

            <b @x:i64>
                %sum = @x + 1;
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        let blocks: Vec<_> = Function::from_id(&ctx, f).blocks().map(|b| b.id).collect();
        assert_eq!(blocks, [a], "branch-with-args chain should merge");
        assert!(BasicBlock::from_id(&ctx, b).parent().is_none());

        let Mnemonic::Binop(Binary { lhs, .. }) = ctx.values.instructions[sum].mnemonic() else {
            panic!("expected merged sum to be a binop");
        };
        assert_eq!(*lhs, ValueId::BlockParam(input));
    }

    #[test]
    fn merged_instructions_have_correct_parent() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                %x = i64 1 + i64 1;
                goto <b>;
            <b>
                %y = i64 2 + i64 2;
                goto <0x1001>;
            "
        );

        let y_insn = y;
        simplify_cfg(&mut ctx, f);

        let insn = ctx.get_insn(y_insn);
        let parent_id = insn.parent().map(|b| b.id());
        assert_eq!(
            parent_id,
            Some(ValueId::BasicBlock(a)),
            "instruction from b should be reparented to a after merge"
        );
    }

    /// Regression: `absorb_block` must *drain* the absorbed block's instruction
    /// list, not merely copy it. Leaving the ids in both blocks puts every merged
    /// instruction in two blocks at once; a later `remove_instruction` (which
    /// unlinks via the instruction's `parent`) then clears one copy while the
    /// other lingers, corrupting block membership and spinning `remove_dead_insns`
    /// forever.
    #[test]
    fn absorbed_block_instruction_list_is_drained() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                %x = i64 1 + i64 1;
                goto <b>;
            <b>
                %y = i64 2 + i64 2;
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        assert!(
            ctx.values.basic_blocks[b].instructions.is_empty(),
            "absorbed block `b` must not retain its instructions after merge"
        );
    }

    /// Regression (invariant): after CFG simplification no instruction id may
    /// appear in more than one block's instruction list. This is the property the
    /// `absorb_block` drain fix restores; its violation was the root cause of an
    /// infinite loop in `remove_dead_insns`.
    #[test]
    fn no_instruction_belongs_to_two_blocks_after_simplify() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                %x = i64 1 + i64 1;
                goto <b>;
            <b>
                %y = i64 2 + i64 2;
                goto <c>;
            <c>
                %z = i64 3 + i64 3;
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        let mut seen = std::collections::HashSet::new();
        for block in ctx.values.basic_blocks.iter() {
            for &insn in &block.instructions {
                assert!(
                    seen.insert(insn),
                    "instruction {insn:?} appears in more than one block after simplify_cfg"
                );
            }
        }
    }

    /// Regression: `remove_dead_insns` must terminate (and actually remove the
    /// dead instruction) after a merge. With the duplicate-membership bug a dead
    /// instruction left in a stale block list was re-discovered every round, so
    /// this loop never finished.
    #[test]
    fn remove_dead_insns_terminates_after_merge() {
        use crate::remove_dead_insns;
        use qcode::value::Function;

        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <b>;
            <b>
                %y = i64 2 + i64 2;
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        let blocks: Vec<_> = Function::from_id(&ctx, f)
            .iter()
            .map(|blk| blk.id)
            .collect();
        for bid in blocks {
            remove_dead_insns(&mut ctx, bid);
        }

        assert!(
            Function::from_id(&ctx, f)
                .iter()
                .all(|blk| !blk.instruction_ids().contains(&y)),
            "dead merged instruction must be removed, not looped on"
        );
    }
}
