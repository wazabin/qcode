use rustc_hash::FxHashSet as HashSet;

use crate::AliasResult;
use jstd::graph::analysis::compute_postdominators;
use qcode::{
    context::Context,
    space::{Space, SpaceId, SpaceType},
    value::{
        BasicBlock, BlockId, Function, FunctionId, ValueId, Varnode,
        insn::{InstructionId, Mnemonic},
    },
};

/// A memory location that may still be read before being overwritten.
///
/// `space` is the space the access reads from (not the pointer's derived
/// space): a register store can only be observed by a register load, so
/// `may_alias` checks are scoped to matching spaces.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct LiveLoc {
    pub ptr: ValueId,
    pub size: usize,
    pub space: SpaceId,
}

/// A byte interval `[start, end)` overwritten by a covering store before any
/// read.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct KilledInterval {
    pub space: SpaceId,
    pub start: u64,
    pub end: u64,
}

impl KilledInterval {
    fn from_alias(iv: (SpaceId, u64, u64)) -> Self {
        let (space, start, end) = iv;
        Self { space, start, end }
    }
}

/// Locations that may still be read.
pub(crate) type LiveSet = Vec<LiveLoc>;
/// Byte intervals overwritten before any read.
pub(crate) type KilledSet = Vec<KilledInterval>;

pub(crate) fn is_reg_space(ctx: &Context, space_id: SpaceId) -> bool {
    matches!(Space::from_id(ctx, space_id).ty, SpaceType::Register)
}

pub(crate) fn is_temp_space(ctx: &Context, space_id: SpaceId) -> bool {
    !is_reg_space(ctx, space_id) && space_id != ctx.default_space
}

/// True for spaces whose stores are eligible for cross-block dead-store
/// elimination: register space and function-scoped temp spaces. The default
/// (RAM/global) space is excluded because such stores may be observed outside
/// the function.
pub(crate) fn is_tracked_space(ctx: &Context, space_id: SpaceId) -> bool {
    is_reg_space(ctx, space_id) || is_temp_space(ctx, space_id)
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
/// `dead_regs` is a list of register varnodes (as `ValueId::Varnode`) whose
/// live-out value is never observable — any store to them with no subsequent
/// read in the block is unconditionally dead, even without a covering later store.
pub fn dead_load_insns(
    ctx: &Context,
    block_id: BlockId,
    aliases: Option<&AliasResult>,
    dead_regs: &[ValueId],
) -> HashSet<InstructionId> {
    let insns: Vec<InstructionId> = BasicBlock::from_id(ctx, block_id)
        .instruction_ids()
        .to_vec();
    let mut dead = block_dead_loads(ctx, block_id);

    if let Some(aliases) = aliases {
        // Alias-aware backward scan, seeded empty (single-block).
        scan_block_aliased(ctx, block_id, aliases, dead_regs, &mut dead, &[], &[]);
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
                        let is_dead_reg = dead_regs.contains(&store.ptr);
                        if !any_overlap(&live, range)
                            && (fully_covered(&killed, range) || is_dead_reg)
                        {
                            dead.insert(id);
                        } else {
                            remove_overlap(&mut live, range);
                            killed.push(range);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    dead
}

/// Loads in `block_id` whose result has no users (dead in any address space).
fn block_dead_loads(ctx: &Context, block_id: BlockId) -> HashSet<InstructionId> {
    let mut dead = HashSet::default();
    for &id in BasicBlock::from_id(ctx, block_id).instruction_ids() {
        if let Mnemonic::Load(_) = ctx.get_insn(id).mnemonic()
            && ctx.users(id).is_empty()
        {
            dead.insert(id);
        }
    }
    dead
}

/// True when `iv` is fully covered by the union of same-space `killed`
/// intervals.
fn fully_covered_iv(killed: &[KilledInterval], iv: KilledInterval) -> bool {
    let pieces: Vec<(i64, i64)> = killed
        .iter()
        .filter(|k| k.space == iv.space)
        .map(|k| (k.start as i64, k.end as i64))
        .collect();
    fully_covered(&pieces, (iv.start as i64, iv.end as i64))
}

/// Remove the byte range `iv` from the same-space `killed` intervals: a read of
/// `iv` means the location is no longer "overwritten before read" for earlier
/// instructions.
fn punch_killed(killed: &mut Vec<KilledInterval>, iv: KilledInterval) {
    let mut same: Vec<(i64, i64)> = killed
        .iter()
        .filter(|k| k.space == iv.space)
        .map(|k| (k.start as i64, k.end as i64))
        .collect();
    remove_overlap(&mut same, (iv.start as i64, iv.end as i64));
    killed.retain(|k| k.space != iv.space);
    killed.extend(same.into_iter().map(|(s, e)| KilledInterval {
        space: iv.space,
        start: s as u64,
        end: e as u64,
    }));
}

/// Shared alias-aware backward scan of a single block.
///
/// `dead` is extended with the dead stores found. Returns the block's
/// upward-exposed live set (`live_in`) and the locations guaranteed overwritten
/// before any read from block entry (`killed_in`), which the cross-block
/// dataflow in [`crate::mem::mem_liveness`] propagates to predecessors.
///
/// `live` tracks [`LiveLoc`]s of loads not yet satisfied; `killed` tracks
/// [`KilledInterval`]s of locations overwritten by a covering store before any
/// read.
fn scan_block_aliased(
    ctx: &Context,
    block_id: BlockId,
    aliases: &AliasResult,
    dead_regs: &[ValueId],
    dead: &mut HashSet<InstructionId>,
    live_seed: &[LiveLoc],
    killed_seed: &[KilledInterval],
) -> (LiveSet, KilledSet) {
    let insns: Vec<InstructionId> = BasicBlock::from_id(ctx, block_id)
        .instruction_ids()
        .to_vec();
    let mut live = live_seed.to_vec();
    let mut killed = killed_seed.to_vec();

    // In a `pure_reg` function the whole architectural register file is
    // functionalized into the returned write-set tuple — callers replay every
    // output register from the tuple, never from the register file — so no
    // register is live at the function's exit. A register store with no
    // in-function reader is therefore dead, exactly like an explicit `dead_reg`.
    // Like `dead_reg`s (and unlike `is_killed`) this does not require a covering
    // store, so it also sees through the call barrier below. Sound *only* for
    // `pure_reg`: otherwise registers are live-out per the calling convention.
    let regs_dead_at_exit = BasicBlock::from_id(ctx, block_id)
        .function()
        .is_some_and(|f| f.is_pure_reg());

    for &id in insns.iter().rev() {
        match ctx.get_insn(id).mnemonic() {
            Mnemonic::Load(load) if !dead.contains(&id) => {
                match aliases.interval(load.ptr) {
                    Some(iv) => punch_killed(&mut killed, KilledInterval::from_alias(iv)),
                    // Unknown read location: conservatively drop same-space kills.
                    None => killed.retain(|k| k.space != load.space),
                }
                live.push(LiveLoc {
                    ptr: load.ptr,
                    size: load.size,
                    space: load.space,
                });
            }
            Mnemonic::Store(store) => {
                let ptr = store.ptr;
                let ptr_iv = aliases.interval(ptr).map(KilledInterval::from_alias);
                let no_live_reader = !live
                    .iter()
                    .any(|l| l.space == store.space && aliases.may_alias(ctx, ptr, l.ptr));
                let is_killed = ptr_iv.is_some_and(|iv| fully_covered_iv(&killed, iv));
                let is_dead_reg = dead_regs.contains(&ptr)
                    || (regs_dead_at_exit && is_reg_space(ctx, store.space));
                if no_live_reader && (is_killed || is_dead_reg) {
                    dead.insert(id);
                } else {
                    // This store overwrites its location: it satisfies covered
                    // live loads and becomes a kill for earlier instructions.
                    live.retain(|l| l.space != store.space || !aliases.covers(l.ptr, ptr));
                    if let Some(iv) = ptr_iv {
                        killed.push(iv);
                    }
                }
            }
            // A call may read any register before its continuation overwrites it,
            // so a register store preceding the call cannot be proven dead by a
            // later (post-call) covering store. Drop register-space kills at the
            // call boundary; explicit `dead_reg` removals still apply (they do not
            // depend on `killed`). This keeps the caller's pre-call stack-pointer
            // decrement (see call_summary::decrement_stack_pointer) alive so the
            // callee's entry stack pointer is seeded correctly.
            Mnemonic::Call(_) | Mnemonic::CallInd(_) => {
                killed.retain(|k| !is_reg_space(ctx, k.space));
            }
            _ => {}
        }
    }
    (live, killed)
}

/// Backward transfer function for one block, used by the cross-block memory
/// liveness fixpoint. Given the live-out / killed-out seeds (from successors),
/// returns this block's `(live_in, killed_in)`.
pub(crate) fn block_transfer(
    ctx: &Context,
    block_id: BlockId,
    aliases: &AliasResult,
    dead_regs: &[ValueId],
    live_seed: &[LiveLoc],
    killed_seed: &[KilledInterval],
) -> (LiveSet, KilledSet) {
    let mut dead = block_dead_loads(ctx, block_id);
    scan_block_aliased(
        ctx,
        block_id,
        aliases,
        dead_regs,
        &mut dead,
        live_seed,
        killed_seed,
    )
}

/// Like [`dead_load_insns`] but seeds the backward scan with the block's
/// live-out / killed-out sets so stores dead across basic-block boundaries are
/// detected.
pub(crate) fn dead_load_insns_seeded(
    ctx: &Context,
    block_id: BlockId,
    aliases: &AliasResult,
    dead_regs: &[ValueId],
    live_seed: &[LiveLoc],
    killed_seed: &[KilledInterval],
) -> HashSet<InstructionId> {
    let mut dead = block_dead_loads(ctx, block_id);
    scan_block_aliased(
        ctx,
        block_id,
        aliases,
        dead_regs,
        &mut dead,
        live_seed,
        killed_seed,
    );
    dead
}

/// Removes dead loads/stores from `block_id` in-place.
pub fn remove_dead_load_insns_block(
    ctx: &mut Context,
    block_id: BlockId,
    aliases: Option<&AliasResult>,
    dead_regs: &[ValueId],
) {
    let dead = dead_load_insns(ctx, block_id, aliases, dead_regs);
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
///
/// When a store's pointer is a known literal, the check is address-precise:
/// the store is dead only if no load in the function reads from an overlapping
/// byte range in the same space. When the pointer is not a literal, the check
/// falls back to space-level coarseness (dead only if the space has no loads).
/// (insn_id, space_id, Some(start, end) if the store address is a literal).
type CandidateStore = (InstructionId, SpaceId, Option<(i64, i64)>);

fn unread_temp_space_stores(ctx: &Context, function_id: FunctionId) -> HashSet<InstructionId> {
    let fun = Function::from_id(ctx, function_id);
    let mut loaded_spaces: HashSet<SpaceId> = HashSet::default();
    // (space_id, byte_start, byte_end) for loads with known literal addresses
    let mut loaded_intervals: Vec<(SpaceId, i64, i64)> = Vec::new();
    // (insn_id, space_id, Some(start, end) if address is a literal)
    let mut candidate_stores: Vec<CandidateStore> = Vec::new();

    for block in &fun {
        for &insn_id in block.instruction_ids() {
            match ctx.get_insn(insn_id).mnemonic() {
                Mnemonic::Load(load) if is_temp_space(ctx, load.space) => {
                    loaded_spaces.insert(load.space);
                    if let ValueId::Literal(lid) = load.ptr {
                        let addr = ctx.values.literals[lid].value as i64;
                        loaded_intervals.push((load.space, addr, addr + load.size as i64));
                    }
                }
                Mnemonic::Store(store) if is_temp_space(ctx, store.space) => {
                    let interval = if let ValueId::Literal(lid) = store.ptr {
                        let addr = ctx.values.literals[lid].value as i64;
                        Some((addr, addr + store.size as i64))
                    } else {
                        None
                    };
                    candidate_stores.push((insn_id, store.space, interval));
                }
                _ => {}
            }
        }
    }

    candidate_stores
        .into_iter()
        .filter(|(_, space_id, interval)| match interval {
            Some((start, end)) => !loaded_intervals.iter().any(|(ls, ls_start, ls_end)| {
                ls == space_id && *ls_start < *end && *start < *ls_end
            }),
            None => !loaded_spaces.contains(space_id),
        })
        .map(|(id, _, _)| id)
        .collect()
}

#[derive(Clone, Copy)]
struct RegisterStore {
    id: InstructionId,
    block: BlockId,
    ptr: ValueId,
    space: SpaceId,
}

/// Finds register stores that are overwritten by a covering store in a
/// postdominating block, with no reads of that register anywhere in the
/// function. This complements the killed-set dataflow for loop shapes where an
/// exit-block overwrite postdominates the entry store but is not propagated as a
/// must-kill through the loop backedge.
fn postdominated_dead_register_stores(
    ctx: &Context,
    function_id: FunctionId,
    aliases: &AliasResult,
) -> HashSet<InstructionId> {
    let function = Function::from_id(ctx, function_id);
    let blocks: Vec<BlockId> = function.iter().map(|block| block.id).collect();
    if blocks.is_empty() {
        return HashSet::default();
    }

    // Calls can observe argument registers implicitly or through bound call args;
    // keep this cleanup for straight-line/local register traffic.
    if function
        .iter()
        .flat_map(|block| block.iter())
        .any(|insn| matches!(insn.mnemonic(), Mnemonic::Call(_) | Mnemonic::CallInd(_)))
    {
        return HashSet::default();
    }

    let node_set: HashSet<BlockId> = blocks.iter().copied().collect();
    let exit_set: HashSet<BlockId> = blocks
        .iter()
        .copied()
        .filter(|&block| {
            BasicBlock::from_id(ctx, block)
                .successors()
                .next()
                .is_none()
        })
        .collect();
    if exit_set.is_empty() {
        return HashSet::default();
    }
    let pdom = compute_postdominators(ctx, &blocks, &node_set, &exit_set);

    let mut stores = Vec::new();
    let mut loads = Vec::new();
    for block in &blocks {
        for &id in BasicBlock::from_id(ctx, *block).instruction_ids() {
            match ctx.get_insn(id).mnemonic() {
                Mnemonic::Store(store) if is_reg_space(ctx, store.space) => {
                    stores.push(RegisterStore {
                        id,
                        block: *block,
                        ptr: store.ptr,
                        space: store.space,
                    });
                }
                Mnemonic::Load(load) if is_reg_space(ctx, load.space) => {
                    loads.push((load.ptr, load.space));
                }
                _ => {}
            }
        }
    }

    stores
        .iter()
        .filter(|candidate| {
            !loads.iter().any(|&(ptr, space)| {
                space == candidate.space && aliases.may_alias(ctx, candidate.ptr, ptr)
            })
        })
        .filter(|candidate| {
            stores.iter().any(|killer| {
                killer.id != candidate.id
                    && killer.block != candidate.block
                    && killer.space == candidate.space
                    && pdom
                        .get(&candidate.block)
                        .is_some_and(|set| set.contains(&killer.block))
                    && aliases.covers(candidate.ptr, killer.ptr)
            })
        })
        .map(|store| store.id)
        .collect()
}

/// Removes dead loads/stores from `function_id` in-place.
///
/// When `aliases` is provided, a flow-sensitive memory-liveness dataflow
/// ([`crate::mem::mem_liveness`]) is run over the CFG so that register/temp-space
/// stores that are dead *across* basic-block boundaries — overwritten before
/// being read on every path, or written to a `dead_reg` and never read — are
/// removed. Without `aliases` each block is treated independently.
pub fn remove_dead_load_insns(
    ctx: &mut Context,
    function_id: FunctionId,
    aliases: Option<&AliasResult>,
    dead_regs: &[ValueId],
) -> bool {
    let block_ids: Vec<BlockId> = Function::from_id(ctx, function_id)
        .iter()
        .map(|block| block.id)
        .collect();

    let mut dead = HashSet::default();
    dead.extend(unread_temp_space_stores(ctx, function_id));

    match aliases {
        Some(aliases) => {
            dead.extend(postdominated_dead_register_stores(
                ctx,
                function_id,
                aliases,
            ));
            let liveness =
                crate::mem::compute_memory_liveness(ctx, function_id, aliases, dead_regs);
            for &block_id in &block_ids {
                dead.extend(dead_load_insns_seeded(
                    ctx,
                    block_id,
                    aliases,
                    dead_regs,
                    liveness.live_out(block_id),
                    liveness.killed_out(block_id),
                ));
            }
        }
        None => {
            for &block_id in &block_ids {
                dead.extend(dead_load_insns(ctx, block_id, None, dead_regs));
            }
        }
    }

    let changed = !dead.is_empty();
    for id in &dead {
        ctx.remove_instruction(*id);
    }
    changed
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

        let dead = dead_load_insns(&ctx, block_id, None, &[]);
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

        let dead = dead_load_insns(&ctx, block_id, None, &[]);
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

        let dead = dead_load_insns(&ctx, block_id, None, &[]);
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

        let dead = dead_load_insns(&ctx, block_id, None, &[]);

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

        let dead = dead_load_insns(&ctx, block_id, None, &[]);
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
        let dead = dead_load_insns(&ctx, block_id, Some(&aliases), &[]);
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
        let dead = dead_load_insns(&ctx, block_id, Some(&aliases), &[]);
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
        let dead = dead_load_insns(&ctx, block_id, Some(&aliases), &[]);

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

    #[test]
    fn postdominating_exit_store_kills_loop_entry_register_store() {
        let mut tc = TestContext::new();
        let r0_lo32 = tc.r0_lo32;
        let r1 = tc.r1;
        qcode!(
            tc.ctx,
            "
            fn test:
                <entry>
                    store({r0_lo32}, i32 1);
                    goto <loop_head>;

                <loop_head>
                    if i8 1 goto <loop_body> else goto <exit>;

                <loop_body>
                    store({r1}, i64 2);
                    goto <loop_head>;

                <exit>
                    store({r0_lo32}, i32 3);
                    return [i64 0];
            "
        );

        let aliases = crate::AliasResult::simple(&tc.ctx);
        let entry_store = *BasicBlock::from_id(&tc.ctx, entry)
            .instruction_ids()
            .iter()
            .find(|&&id| matches!(tc.ctx.get_insn(id).mnemonic(), Mnemonic::Store(_)))
            .expect("entry store exists");
        let exit_store = *BasicBlock::from_id(&tc.ctx, exit)
            .instruction_ids()
            .iter()
            .find(|&&id| matches!(tc.ctx.get_insn(id).mnemonic(), Mnemonic::Store(_)))
            .expect("exit store exists");

        remove_dead_load_insns(&mut tc.ctx, test, Some(&aliases), &[]);

        assert!(
            !BasicBlock::from_id(&tc.ctx, entry)
                .instruction_ids()
                .contains(&entry_store),
            "entry r0_lo32 store should be removed because exit store postdominates it"
        );
        assert!(
            BasicBlock::from_id(&tc.ctx, exit)
                .instruction_ids()
                .contains(&exit_store),
            "exit r0_lo32 store is the return-visible value and must remain"
        );
    }
}

// ----- pass ------------------------------------------------------------------

use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct DeadLoad;

impl FunctionPass for DeadLoad {
    const NAME: &'static str = "dead_load";
    fn description(&self) -> &'static str {
        "Remove dead memory loads"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        // Per-function pass: scope the alias oracle to this function so the
        // stage is O(program) total, not O(functions × program).
        let aliases = AliasResult::simple_for_function(ctx, fun_id);
        Ok(remove_dead_load_insns(ctx, fun_id, Some(&aliases), &[]))
    }
}

crate::register_function_pass!(DeadLoad);

/// Lives here (not in the orphaned `dead_store.rs`) because it shares
/// [`remove_dead_load_insns`] with [`DeadLoad`]; the only difference is that it
/// also treats the architecture's flag registers as dead.
#[derive(Default)]
pub struct DeadStore;

impl FunctionPass for DeadStore {
    const NAME: &'static str = "dead_store";
    fn description(&self) -> &'static str {
        "Remove dead register loads and overwritten flag stores"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        env: &PipelineEnv,
    ) -> Result<bool, String> {
        // Per-function pass: scope the alias oracle to this function so the
        // stage is O(program) total, not O(functions × program).
        let aliases = AliasResult::simple_for_function(ctx, fun_id);
        Ok(remove_dead_load_insns(
            ctx,
            fun_id,
            Some(&aliases),
            &env.cfg.dead_flag_regs,
        ))
    }
}

crate::register_function_pass!(DeadStore);
