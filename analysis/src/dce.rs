use std::collections::HashSet;

use qcode::{
    context::Context,
    value::{BasicBlock, BlockId, InstructionId, insn::Mnemonic},
};

fn has_side_effects(mnemonic: &Mnemonic) -> bool {
    matches!(mnemonic, Mnemonic::Store(_) | Mnemonic::PCodeOp(_)) || mnemonic.is_terminator()
}

/// Returns instructions in `block_id` that are pure and have no users.
pub fn dead_insns(ctx: &Context, block_id: BlockId) -> HashSet<InstructionId> {
    let mut dead = HashSet::new();
    let insn_ids: Vec<InstructionId> = BasicBlock::from_id(ctx, block_id)
        .instruction_ids()
        .to_vec();
    for id in insn_ids {
        let insn = ctx.get_insn(id);
        let mnemonic = insn.mnemonic();
        let side_effects = has_side_effects(mnemonic);

        if !side_effects && ctx.users(id).is_empty() {
            dead.insert(id);
        }
    }
    dead
}

/// Removes dead pure instructions from `block_id` iteratively until fixed point,
/// updating the users reverse map after each round.
pub fn remove_dead_insns(ctx: &mut Context, block_id: BlockId) {
    loop {
        let dead = dead_insns(ctx, block_id);
        if dead.is_empty() {
            return;
        }

        BasicBlock::from_id_mut(ctx, block_id).retain_insns(|id| !dead.contains(id));

        ctx.values.remove_instructions(&dead);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use qcode::{
        builder::Builder,
        context::Context,
        space::SpaceId,
        testing::TestContext,
        value::{BlockId, Value, ValueId, insn::PCodeOpId},
    };

    fn reg_space(ctx: &Context) -> SpaceId {
        ctx.try_get_space("register").unwrap()
    }

    fn build_block(f: impl FnOnce(&mut Builder<'static, '_>)) -> (Context<'static>, BlockId) {
        let mut ctx = TestContext::new().ctx;
        let block_id = ctx.get_or_make_block(0x1000);
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        f(&mut builder);
        unsafe { builder.dont_finalize() };
        drop(builder);
        (ctx, block_id)
    }

    #[test]
    fn test_pure_binop_no_users_removed() {
        let (mut ctx, block_id) = build_block(|b| {
            let a = b.context_mut().get_const(1u64, 8).id();
            let c = b.context_mut().get_const(2u64, 8).id();
            b.push_add(a, c);
        });

        remove_dead_insns(&mut ctx, block_id);
        assert!(
            BasicBlock::from_id(&ctx, block_id)
                .instruction_ids()
                .is_empty(),
            "dead binop should be removed"
        );
    }

    #[test]
    fn test_pure_binop_with_users_kept() {
        let (ctx, block_id) = build_block(|b| {
            let a = b.context_mut().get_const(1u64, 8).id();
            let c = b.context_mut().get_const(2u64, 8).id();
            let sum = b.push_add(a, c).id();
            let d = b.context_mut().get_const(3u64, 8).id();
            b.push_add(sum, d);
        });

        // The top-level add (no users) is dead, but the first add feeds it —
        // after top-level is removed, first add has no users and is then also removed.
        // This test checks the chain case; use dead_insns (single round) for the kept case.
        let dead = dead_insns(&ctx, block_id);
        let insn_ids: Vec<_> = BasicBlock::from_id(&ctx, block_id)
            .instruction_ids()
            .to_vec();
        // Only the outermost add (no users) is dead in the first round.
        assert_eq!(
            dead.len(),
            1,
            "only the unused result should be dead initially"
        );
        assert!(
            !dead.contains(&insn_ids[0]),
            "first add (whose result is used) must not be dead"
        );
        assert!(
            dead.contains(&insn_ids[1]),
            "second add (no users) should be dead"
        );
    }

    #[test]
    fn test_chain_both_removed() {
        // c = a + b, d = c * 2, neither used → both removed after iterating.
        let (mut ctx, block_id) = build_block(|b| {
            let a = b.context_mut().get_const(1u64, 8).id();
            let bv = b.context_mut().get_const(2u64, 8).id();
            let c = b.push_add(a, bv).id();
            let two = b.context_mut().get_const(2u64, 8).id();
            b.push_mul(c, two);
        });

        remove_dead_insns(&mut ctx, block_id);
        assert!(
            BasicBlock::from_id(&ctx, block_id)
                .instruction_ids()
                .is_empty(),
            "entire dead chain should be removed"
        );
    }

    #[test]
    fn test_store_kept() {
        let (mut ctx, block_id) = build_block(|b| {
            let rax_vn = b.context().get_named("r0").unwrap().as_varnode().unwrap();
            let rax_ptr = ValueId::Varnode(rax_vn);
            let space = reg_space(b.context());
            let v = b.context_mut().get_const(42u64, 8).id();
            b.push_store(v, rax_ptr, space);
        });

        let block = BasicBlock::from_id_mut(&mut ctx, block_id);
        let block_len = block.instruction_ids().len();

        remove_dead_insns(&mut ctx, block_id);

        let block = BasicBlock::from_id(&ctx, block_id);

        assert_eq!(
            block.instruction_ids().len(),
            block_len,
            "store must not be removed, block:\n{block}",
        );
    }

    #[test]
    fn test_pcode_op_kept() {
        let dead = {
            let (ctx, block_id) = build_block(|b| {
                let op_id: PCodeOpId = b.context_mut().pcode_ops.push("syscall");
                b.push_pcode_op(op_id, vec![], None);
            });
            dead_insns(&ctx, block_id)
        };
        assert!(dead.is_empty(), "PCodeOp must not be marked dead");
    }

    #[test]
    fn test_terminator_kept() {
        let dead = {
            let (ctx, block_id) = build_block(|b| {
                let zero = b.context_mut().get_const(0u64, 8).id();
                b.push_return(zero);
            });
            dead_insns(&ctx, block_id)
        };
        assert!(dead.is_empty(), "terminator must not be marked dead");
    }
}
