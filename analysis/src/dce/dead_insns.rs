use rustc_hash::FxHashSet as HashSet;

use qcode::{
    context::Context,
    value::{
        BasicBlock, BlockId, Function, FunctionId, InstructionId,
        insn::{Call, Mnemonic},
    },
};

/// Returns instructions in `block_id` that are pure and have no users.
pub fn dead_insns(ctx: &Context, block_id: BlockId) -> HashSet<InstructionId> {
    let mut dead = HashSet::default();
    let insn_ids: Vec<InstructionId> = BasicBlock::from_id(ctx, block_id)
        .instruction_ids()
        .to_vec();
    for id in insn_ids {
        let insn = ctx.get_insn(id);
        let mnemonic = insn.mnemonic();
        let side_effects = mnemonic.has_side_effects();

        if !side_effects && ctx.users(id).is_empty() {
            dead.insert(id);
        }
    }
    dead
}

/// Removes dead pure instructions from `block_id` iteratively until fixed point,
/// updating the users reverse map after each round.
pub fn remove_dead_insns(ctx: &mut Context, block_id: BlockId) -> bool {
    let mut changed = false;
    loop {
        let dead = dead_insns(ctx, block_id);
        if dead.is_empty() {
            break;
        }

        changed = true;
        for id in &dead {
            ctx.remove_instruction(*id);
        }
    }

    let params_changed = remove_unused_no_pred_block_params(ctx, block_id);
    changed || params_changed
}

/// Removes block params that have no users when the block has no incoming
/// control-flow edges. This covers function-entry params introduced for
/// load-before-store registers that later become dead, without touching join
/// blocks whose predecessor terminators carry positional arguments.
pub fn remove_unused_no_pred_block_params(ctx: &mut Context, block_id: BlockId) -> bool {
    if BasicBlock::from_id(ctx, block_id)
        .predecessors()
        .next()
        .is_some()
    {
        return false;
    }

    // A `pure_reg` function's entry params are its canonical interface, aligned
    // index-for-index with `input_regs` and every caller's `Call.args`. Removing
    // one is an interface change, so route it through `remove_entry_param`, which
    // drops the param, the matching `input_regs` entry, and the matching argument
    // at every direct caller in lockstep — keeping the alignment the emulator's
    // positional arg-binding relies on. (Conventional functions keep the local
    // removal below; their call args are re-derived by `bind_call_args`.)
    if let Some((fid, is_entry, pure_reg)) =
        BasicBlock::from_id(ctx, block_id).function().map(|f| {
            (
                f.id,
                f.root().map(|b| b.id) == Some(block_id),
                f.is_pure_reg(),
            )
        })
        && is_entry
        && pure_reg
    {
        let params = ctx.values.basic_blocks[block_id].params.clone();
        let dead: Vec<usize> = params
            .iter()
            .enumerate()
            .filter(|(_, p)| ctx.users(**p).is_empty() && !ctx.values.block_params[**p].protected)
            .map(|(i, _)| i)
            .collect();
        // Remove high index first so the lower indices stay valid.
        for &index in dead.iter().rev() {
            remove_entry_param(ctx, fid, index);
        }
        return !dead.is_empty();
    }

    let params = ctx.values.basic_blocks[block_id].params.clone();
    let mut kept = Vec::with_capacity(params.len());
    let mut changed = false;
    for param in params {
        if ctx.users(param).is_empty() && !ctx.values.block_params[param].protected {
            ctx.values.block_params[param].parent = None;
            changed = true;
        } else {
            ctx.values.block_params[param].index = kept.len();
            kept.push(param);
        }
    }

    if changed {
        ctx.values.basic_blocks[block_id].params = kept;
    }
    changed
}

/// Remove the entry param at position `index` from `fid` and keep its interface
/// aligned: drop the root block param, the `input_regs[index]` entry, and the
/// `Call.args[index]` argument at every direct caller, all in lockstep.
///
/// This is the single ABI-consistent entry-param removal both this module's
/// dead-param sweep and `dead_signature` route through, so a `pure_reg`
/// function's `param[i] ↔ input_regs[i] ↔ arg[i]` alignment holds by
/// construction after any removal. The caller must ensure the param has no
/// remaining users.
pub fn remove_entry_param(ctx: &mut Context, fid: FunctionId, index: usize) {
    let Some(root) = Function::from_id(ctx, fid).root().map(|b| b.id) else {
        return;
    };

    // Drop the root param at `index`, reindexing the survivors.
    let mut params = ctx.values.basic_blocks[root].params.clone();
    if index >= params.len() {
        return;
    }
    let removed = params.remove(index);
    ctx.values.block_params[removed].parent = None;
    for (i, &p) in params.iter().enumerate() {
        ctx.values.block_params[p].index = i;
    }
    ctx.values.basic_blocks[root].params = params;

    // Drop the matching input-register entry.
    if let Some(inputs) = Function::from_id(ctx, fid).input_regs()
        && index < inputs.len()
    {
        let mut inputs = inputs.to_vec();
        inputs.remove(index);
        Function::from_id_mut(ctx, fid).set_input_regs(inputs);
    }

    // Drop the matching positional argument at every direct caller.
    let call_sites: Vec<InstructionId> = ctx
        .instructions()
        .filter_map(|insn| match insn.mnemonic() {
            Mnemonic::Call(c) if c.target == fid => Some(insn.id),
            _ => None,
        })
        .collect();
    for call_id in call_sites {
        let Mnemonic::Call(call) = ctx.get_insn(call_id).mnemonic().clone() else {
            continue;
        };
        if index >= call.args.len() {
            continue;
        }
        let mut args = call.args;
        args.remove(index);
        ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: call.target,
                args,
                clobbers: call.clobbers,
            }),
        );
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
        value::{BlockId, ValueId, insn::PCodeOpId},
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
                let op_id: PCodeOpId = b.context_mut().pcode_ops.push(Box::from("syscall"));
                b.push_pcode_op(op_id, vec![], None, 0);
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

    #[test]
    fn unused_entry_block_param_removed_after_dead_user_is_removed() {
        let (mut ctx, block_id) = build_block(|b| {
            let param = b.push_param(4).id();
            b.push_zext(param, 8);
        });

        assert_eq!(BasicBlock::from_id(&ctx, block_id).num_params(), 1);
        remove_dead_insns(&mut ctx, block_id);
        assert_eq!(
            BasicBlock::from_id(&ctx, block_id).num_params(),
            0,
            "unused entry block param should be pruned"
        );
    }
}

// ----- pass ------------------------------------------------------------------

use qcode::value::FunctionRef;

use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct Dce;

impl FunctionPass for Dce {
    const NAME: &'static str = "dce";
    fn description(&self) -> &'static str {
        "Remove unused pure instructions"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        let block_ids: Vec<_> = FunctionRef::from_id(ctx, fun_id)
            .blocks()
            .map(|b| b.id)
            .collect();
        let mut changed = false;
        for block_id in block_ids {
            changed |= remove_dead_insns(ctx, block_id);
        }
        Ok(changed)
    }
}

crate::register_function_pass!(Dce);
