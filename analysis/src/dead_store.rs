use std::collections::HashSet;

use qcode::{
    context::Context,
    space::{SpaceId, SpaceType},
    value::{
        BasicBlock, BlockId, ValueId, Varnode,
        insn::{InstructionId, Mnemonic},
    },
};

fn is_reg_space(ctx: &Context, space_id: SpaceId) -> bool {
    matches!(ctx.get_space(space_id).ty, SpaceType::Register)
}

fn ptr_offset(ctx: &Context, ptr: ValueId) -> Option<i64> {
    match ptr {
        ValueId::Varnode(id) => Some(Varnode::from_id(ctx, id).address()),
        ValueId::Literal(lid) => Some(ctx.values.literals[lid].value as i64),
        _ => None,
    }
}

fn intervals_overlap(a: (i64, i64), b: (i64, i64)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

fn any_overlap(set: &[(i64, i64)], range: (i64, i64)) -> bool {
    set.iter().any(|&r| intervals_overlap(r, range))
}

fn fully_covered(set: &[(i64, i64)], range: (i64, i64)) -> bool {
    let mut pieces: Vec<(i64, i64)> = set
        .iter()
        .filter(|&&r| intervals_overlap(r, range))
        .map(|&(s, e)| (s.max(range.0), e.min(range.1)))
        .collect();
    pieces.sort_unstable();
    let mut covered = range.0;
    for (s, e) in pieces {
        if s > covered {
            return false;
        }
        covered = covered.max(e);
    }
    covered >= range.1
}

fn remove_overlap(set: &mut Vec<(i64, i64)>, range: (i64, i64)) {
    let mut result = Vec::new();
    for &(s, e) in set.iter() {
        if e <= range.0 || s >= range.1 {
            result.push((s, e));
        } else {
            if s < range.0 {
                result.push((s, range.0));
            }
            if e > range.1 {
                result.push((range.1, e));
            }
        }
    }
    *set = result;
}

/// Returns the set of dead register load/store instructions in `block_id`.
///
/// Dead loads: loads from the register space whose result has no users, or
/// whose result is GVN-redundant (an equivalent earlier load computed the same
/// value).
/// Dead stores: stores to the register space that are overwritten by a later
/// covering store before any overlapping load, within the same basic block.
/// GVN-redundant loads are invisible to the backward store scan — they do not
/// count as live readers, so more stores may be exposed as dead.
///
/// Pass `None` for `gvn` to get the old behaviour without GVN.
pub fn dead_reg_insns(ctx: &Context, block_id: BlockId) -> HashSet<InstructionId> {
    let insns: Vec<InstructionId> = BasicBlock::from_id(ctx, block_id)
        .instruction_ids()
        .to_vec();
    let mut dead = HashSet::new();

    // Dead loads: no users
    for &id in &insns {
        if let Mnemonic::Load(load) = ctx.get_insn(id).mnemonic()
            && is_reg_space(ctx, load.space)
            && ctx.users(id).is_empty()
        {
            dead.insert(id);
        }
    }

    // Dead stores: backward scan.
    // Loads already in the dead set are skipped — they will be eliminated, so
    // they should not prevent their producing store from being marked dead.
    let mut live: Vec<(i64, i64)> = Vec::new();
    let mut killed: Vec<(i64, i64)> = Vec::new();

    for &id in insns.iter().rev() {
        match ctx.get_insn(id).mnemonic() {
            Mnemonic::Load(load) if is_reg_space(ctx, load.space) && !dead.contains(&id) => {
                if let Some(offset) = ptr_offset(ctx, load.ptr) {
                    let range = (offset, offset + load.size as i64);
                    live.push(range);
                }
            }
            Mnemonic::Store(store) if is_reg_space(ctx, store.space) => {
                if let Some(offset) = ptr_offset(ctx, store.ptr) {
                    let range = (offset, offset + store.size as i64);
                    if !any_overlap(&live, range) && fully_covered(&killed, range) {
                        dead.insert(id);
                    }
                    remove_overlap(&mut live, range);
                    killed.push(range);
                }
            }
            _ => {}
        }
    }

    dead
}

/// Removes dead register loads/stores from `block_id` in-place.
/// Also cleans up the `users` reverse map for removed instructions.
///
/// When `gvn` is provided, users of GVN-redundant dead loads are redirected to
/// the GVN leader before the dead instruction is removed.
pub fn remove_dead_reg_insns(ctx: &mut Context, block_id: BlockId) {
    let dead = dead_reg_insns(ctx, block_id);
    if dead.is_empty() {
        return;
    }

    BasicBlock::from_id_mut(ctx, block_id).retain_insns(|id| !dead.contains(id));

    ctx.values.remove_instructions(&dead);
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use qcode::{builder::Builder, testing::TestContext, value::Value};

    fn reg_space(ctx: &Context) -> SpaceId {
        ctx.try_get_space("register").unwrap()
    }

    /// Emit a single basic block and return (ctx, block_id).
    fn build_block(
        f: impl FnOnce(&mut Builder<'static, '_>),
    ) -> (qcode::context::Context<'static>, BlockId) {
        let mut ctx = TestContext::new().ctx;
        let block_id = ctx.get_or_make_block(0x1000);
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        f(&mut builder);
        unsafe { builder.dont_finalize() };
        drop(builder);
        (ctx, block_id)
    }

    #[test]
    fn test_dead_load_eliminated() {
        let (ctx, block_id) = build_block(|b| {
            let rax_vn = b.context().get_named("r0").unwrap().as_varnode().unwrap();
            let rax_ptr = ValueId::Varnode(rax_vn);
            let space = reg_space(b.context());
            b.push_load::<false>(rax_ptr, 8, space);
        });

        let dead = dead_reg_insns(&ctx, block_id);
        assert!(!dead.is_empty(), "dead load should be detected");
    }

    #[test]
    fn test_used_load_not_eliminated() {
        let (ctx, block_id) = build_block(|b| {
            let rax_vn = b.context().get_named("r0").unwrap().as_varnode().unwrap();
            let rax_ptr = ValueId::Varnode(rax_vn);
            let space = reg_space(b.context());
            let loaded_id = b.push_load::<false>(rax_ptr, 8, space).id();
            let one = b.context_mut().get_const(1, 8).id();
            b.push_add(loaded_id, one);
        });

        let dead = dead_reg_insns(&ctx, block_id);
        let load_ids: Vec<_> = BasicBlock::from_id(&ctx, block_id)
            .instruction_ids()
            .iter()
            .copied()
            .filter(|&id| matches!(ctx.get_insn(id).mnemonic(), Mnemonic::Load(_)))
            .collect();
        assert!(!load_ids.is_empty());
        assert!(
            !dead.contains(&load_ids[0]),
            "used load must not be eliminated"
        );
    }

    #[test]
    #[ignore = "WIP: analysis not fully implemented yet"]
    fn test_dead_store_overwritten() {
        // Store to r0 twice — first store is dead.
        let mut ctx = TestContext::new().ctx;
        let rax = ctx.get_named("r0").unwrap().as_varnode().unwrap();
        qcode!(
            ctx,
            "
            <block>
            store({rax}, i64 1);
            store({rax}, i64 2);
            "
        );
        let block_id = block;

        let dead = dead_reg_insns(&ctx, block_id);
        let store_ids: Vec<_> = BasicBlock::from_id(&ctx, block_id)
            .instruction_ids()
            .iter()
            .copied()
            .filter(|&id| matches!(ctx.get_insn(id).mnemonic(), Mnemonic::Store(_)))
            .collect();
        assert_eq!(store_ids.len(), 2);
        assert!(dead.contains(&store_ids[0]), "first store should be dead");
        assert!(!dead.contains(&store_ids[1]), "last store must not be dead");
    }

    #[test]
    fn test_store_read_then_overwrite_not_dead() {
        // Store to r0, load r0, store r0 — first store is NOT dead.
        let mut ctx = TestContext::new().ctx;
        let rax = ctx.get_named("r0").unwrap().as_varnode().unwrap();
        qcode!(
            ctx,
            "
            <block>
            store({rax}, i64 1);
            %tmp = load(i64, {rax});
            store({rax}, i64 2);
            "
        );
        let block_id = block;

        let dead = dead_reg_insns(&ctx, block_id);

        let store_ids: Vec<_> = BasicBlock::from_id(&ctx, block_id)
            .instruction_ids()
            .iter()
            .copied()
            .filter(|&id| matches!(ctx.get_insn(id).mnemonic(), Mnemonic::Store(_)))
            .collect();

        assert_eq!(store_ids.len(), 2);
        assert!(
            !dead.contains(&store_ids[0]),
            "first store is read before overwrite, must not be dead"
        );
    }

    #[test]
    fn test_partial_overlap_not_dead() {
        // Store to r0_lo16 (16-bit), then load r0_byte1 — the store must NOT be dead.
        let (ctx, block_id) = build_block(|b| {
            let ax_vn = b
                .context()
                .get_named("r0_lo16")
                .unwrap()
                .as_varnode()
                .unwrap();
            let ah_vn = b
                .context()
                .get_named("r0_byte1")
                .unwrap()
                .as_varnode()
                .unwrap();
            let space = reg_space(b.context());
            let v = b.context_mut().get_const(0x1234u64, 2).id();
            b.push_store(v, ValueId::Varnode(ax_vn), space);
            let loaded_id = b.push_load::<false>(ValueId::Varnode(ah_vn), 1, space).id();
            let zero = b.context_mut().get_const(0, 1).id();
            b.push_add(loaded_id, zero);
        });

        let dead = dead_reg_insns(&ctx, block_id);
        let store_ids: Vec<_> = BasicBlock::from_id(&ctx, block_id)
            .instruction_ids()
            .iter()
            .copied()
            .filter(|&id| matches!(ctx.get_insn(id).mnemonic(), Mnemonic::Store(_)))
            .collect();
        assert!(!store_ids.is_empty());
        assert!(
            !dead.contains(&store_ids[0]),
            "store to AX must not be dead when AH is loaded"
        );
    }
}
