use std::collections::HashSet;

use crate::AliasResult;
use qcode::{
    context::Context,
    space::{Space, SpaceId, SpaceType},
    value::{
        BasicBlock, BlockId, Function, FunctionId, ValueId, Varnode,
        insn::{InstructionId, Mnemonic},
    },
};

fn is_reg_space(ctx: &Context, space_id: SpaceId) -> bool {
    matches!(Space::from_id(ctx, space_id).ty, SpaceType::Register)
}

fn is_temp_space(ctx: &Context, space_id: SpaceId) -> bool {
    !is_reg_space(ctx, space_id) && space_id != ctx.default_space
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

/// Returns the set of dead load/store instructions in `block_id`.
///
/// Dead loads: loads whose result has no users (any address space).
///
/// Dead stores: stores overwritten by a later covering store before any
/// intervening load that may read from the same location.
///
/// When `aliases` is provided, the backward scan uses `may_alias` to detect
/// live readers and `must_alias` to confirm a later store kills an earlier one.
/// Without `aliases`, the scan falls back to interval arithmetic restricted to
/// the register address space.
pub fn dead_load_insns(
    ctx: &Context,
    block_id: BlockId,
    aliases: Option<&AliasResult>,
) -> HashSet<InstructionId> {
    let insns: Vec<InstructionId> = BasicBlock::from_id(ctx, block_id)
        .instruction_ids()
        .to_vec();
    let mut dead = HashSet::new();

    // Dead loads: no users (any address space).
    for &id in &insns {
        if let Mnemonic::Load(_) = ctx.get_insn(id).mnemonic()
            && ctx.users(id).is_empty()
        {
            dead.insert(id);
        }
    }

    if let Some(aliases) = aliases {
        // Alias-aware backward scan.
        // live: (ptr, access_size, access_space) of loads not yet satisfied.
        // The access_space is the space the load reads from, NOT the pointer's
        // derived space. A register store can only be observed by a register
        // load, so we scope the may_alias check to matching spaces.
        let mut live: Vec<(ValueId, usize, SpaceId)> = Vec::new();
        // killed: (ptr, access_size) of stores seen later in program order.
        let mut killed: Vec<(ValueId, usize)> = Vec::new();

        for &id in insns.iter().rev() {
            match ctx.get_insn(id).mnemonic() {
                Mnemonic::Load(load) if !dead.contains(&id) => {
                    live.push((load.ptr, load.size, load.space));
                }
                Mnemonic::Store(store) => {
                    let ptr = store.ptr;
                    let size = store.size;
                    let no_live_reader = !live
                        .iter()
                        .any(|(lp, _, ls)| *ls == store.space && aliases.may_alias(ctx, ptr, *lp));
                    // A later store kills this one when its interval fully covers ours.
                    let is_killed = killed.iter().any(|(kp, _)| aliases.covers(ptr, *kp));
                    if no_live_reader && is_killed {
                        dead.insert(id);
                    } else {
                        // Remove loads whose interval is fully covered by this store
                        // and whose access space matches.
                        live.retain(|(lp, _, ls)| *ls != store.space || !aliases.covers(*lp, ptr));
                        killed.push((ptr, size));
                    }
                }
                _ => {}
            }
        }
    } else {
        // Interval-based backward scan (register space only).
        // Dead loads already in the set are skipped so they don't prevent
        // their producing store from being marked dead.
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
    }

    dead
}

/// Removes dead loads/stores from `block_id` in-place.
pub fn remove_dead_load_insns_block(
    ctx: &mut Context,
    block_id: BlockId,
    aliases: Option<&AliasResult>,
) {
    let dead = dead_load_insns(ctx, block_id, aliases);
    if dead.is_empty() {
        return;
    }
    for id in &dead {
        ctx.remove_instruction(*id);
    }
}

/// Returns stores to temp/mysave spaces (not Register, not default RAM) from
/// which no load ever reads within `function_id`. These stores are dead because
/// the temp spaces are function-scoped and not observable outside.
fn unread_temp_space_stores(ctx: &Context, function_id: FunctionId) -> HashSet<InstructionId> {
    let fun = Function::from_id(ctx, function_id);
    let mut loaded_spaces: HashSet<SpaceId> = HashSet::new();
    let mut candidate_stores: Vec<(InstructionId, SpaceId)> = Vec::new();

    for block in &fun {
        for &insn_id in block.instruction_ids() {
            match ctx.get_insn(insn_id).mnemonic() {
                Mnemonic::Load(load) if is_temp_space(ctx, load.space) => {
                    loaded_spaces.insert(load.space);
                }
                Mnemonic::Store(store) if is_temp_space(ctx, store.space) => {
                    candidate_stores.push((insn_id, store.space));
                }
                _ => {}
            }
        }
    }

    candidate_stores
        .into_iter()
        .filter(|(_, space_id)| !loaded_spaces.contains(space_id))
        .map(|(id, _)| id)
        .collect()
}

/// Removes dead loads/stores from `function_id` in-place.
pub fn remove_dead_load_insns(
    ctx: &mut Context,
    function_id: FunctionId,
    aliases: Option<&AliasResult>,
) {
    let fun = Function::from_id(ctx, function_id);

    let mut dead = HashSet::new();
    dead.extend(unread_temp_space_stores(ctx, function_id));
    for block in &fun {
        dead.extend(dead_load_insns(ctx, block.id, aliases));
    }

    for id in &dead {
        ctx.remove_instruction(*id);
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use qcode::{builder::Builder, context::Context, testing::TestContext, value::Value};

    fn reg_space(ctx: &Context) -> SpaceId {
        ctx.try_get_space("register").unwrap()
    }

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

        let dead = dead_load_insns(&ctx, block_id, None);
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

        let dead = dead_load_insns(&ctx, block_id, None);
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
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;
            <block>
                store(&A, i64 1);
                store(&A, i64 2);
            "
        );
        let block_id = block;

        let dead = dead_load_insns(&ctx, block_id, None);
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
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;
            <block>
                store(&A, i64 1);
                %tmp = load(i64, &A);
                store(&A, i64 2);
            "
        );
        let block_id = block;

        let dead = dead_load_insns(&ctx, block_id, None);

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

        let dead = dead_load_insns(&ctx, block_id, None);
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

    #[test]
    fn test_dead_store_with_must_alias() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            <block>
                store(&A, i64 1);
                store(&A, i64 2);
            "
        );
        let block_id = block;

        let aliases = crate::AliasResult::simple(&ctx);
        let dead = dead_load_insns(&ctx, block_id, Some(&aliases));
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
    fn test_store_with_live_intervening_load_not_dead_with_aliases() {
        // %tmp is used by the second store, so the load is live and the first
        // store to A must not be eliminated.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            <block>
                store(&A, i64 1);
                %tmp = load(i64, &A);
                store(&B, %tmp);
                store(&A, i64 2);
            "
        );
        let block_id = block;

        let aliases = crate::AliasResult::simple(&ctx);
        let dead = dead_load_insns(&ctx, block_id, Some(&aliases));
        let store_a_ids: Vec<_> = BasicBlock::from_id(&ctx, block_id)
            .instruction_ids()
            .iter()
            .copied()
            .filter(|&id| {
                if let Mnemonic::Store(s) = ctx.get_insn(id).mnemonic() {
                    s.ptr == ValueId::Varnode(ctx.get_named("A").unwrap().as_varnode().unwrap())
                } else {
                    false
                }
            })
            .collect();
        assert_eq!(store_a_ids.len(), 2);
        assert!(
            !dead.contains(&store_a_ids[0]),
            "first store to A is read by a live load, must not be dead"
        );
    }

    // Regression: store to narrow sub-register (r0_lo32 / EAX-like) immediately
    // followed by a store to the wide register (r0 / RAX-like) that fully covers it.
    // The narrow store has no live readers and its interval is covered by the wide
    // store, so it must be detected as dead.
    #[test]
    fn test_narrow_store_killed_by_wide_register_store() {
        let (ctx, block_id) = build_block(|b| {
            let r0_lo32 = b
                .context()
                .get_named("r0_lo32")
                .unwrap()
                .as_varnode()
                .unwrap();
            let r0 = b.context().get_named("r0").unwrap().as_varnode().unwrap();
            let space = reg_space(b.context());

            let narrow_val = b.context_mut().get_const(1u64, 4).id();
            b.push_store(narrow_val, ValueId::Varnode(r0_lo32), space);

            let wide_val = b.context_mut().get_const(2u64, 8).id();
            b.push_store(wide_val, ValueId::Varnode(r0), space);
        });

        let aliases = crate::AliasResult::simple(&ctx);
        let dead = dead_load_insns(&ctx, block_id, Some(&aliases));

        let store_ids: Vec<_> = BasicBlock::from_id(&ctx, block_id)
            .instruction_ids()
            .iter()
            .copied()
            .filter(|&id| matches!(ctx.get_insn(id).mnemonic(), Mnemonic::Store(_)))
            .collect();

        assert_eq!(store_ids.len(), 2);
        assert!(
            dead.contains(&store_ids[0]),
            "store to r0_lo32 must be dead: r0 fully covers it with no intervening load"
        );
        assert!(
            !dead.contains(&store_ids[1]),
            "store to r0 must not be dead"
        );
    }
}
