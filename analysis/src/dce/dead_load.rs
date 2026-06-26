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

/// The byte intervals a resolved external `target` clobbers (its recorded
/// caller-saved register set), as kills for the backward scan. Empty for a
/// callee with no recorded clobber set.
fn call_clobber_intervals(ctx: &Context, target: FunctionId) -> Vec<KilledInterval> {
    let Some(clobbered) = Function::from_id(ctx, target).clobbered_regs() else {
        return Vec::new();
    };
    clobbered
        .iter()
        .map(|&vn| {
            let v = Varnode::from_id(ctx, vn);
            let start = v.address() as u64;
            KilledInterval {
                space: v.space().id,
                start,
                end: start + v.size() as u64,
            }
        })
        .collect()
}

/// Whether the kill interval `iv` fully covers the live register load `l` — so
/// the clobbering write satisfies it (the load reads the call's output). Only a
/// register-varnode load is matched; any other live location is left in place
/// (conservative — the store before it stays live).
fn killed_covers_loc(ctx: &Context, iv: KilledInterval, l: &LiveLoc) -> bool {
    let ValueId::Varnode(vn) = l.ptr else {
        return false;
    };
    let v = Varnode::from_id(ctx, vn);
    if v.space().id != iv.space {
        return false;
    }
    let start = v.address() as u64;
    let end = start + v.size() as u64;
    iv.start <= start && end <= iv.end
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

/// Decompose `ptr` into `(base, byte_offset)` by peeling literal `+`/`-` terms,
/// so `base + 8` and `base` (decomposing to the same `base`) can be compared
/// by their static offset. A pointer with no recognizable arithmetic is its own
/// base at offset 0. Mirrors `gvn::affine::base_offset` but is self-contained
/// (no `Numbering`) and used only for disjointness, where over-conservatism is
/// always safe.
fn base_plus_offset(ctx: &Context, ptr: ValueId) -> (ValueId, i64) {
    use qcode::value::insn::{Binop, IntBinop};
    let mut cur = ptr;
    let mut acc = 0i64;
    // Bound the walk so a malformed cyclic graph can't loop forever.
    for _ in 0..64 {
        let ValueId::Instruction(id) = cur else { break };
        // `gep(base, off)` ≡ `base + off` (a constant byte offset) — peel it like
        // a literal `add` so a field deref `gep(p.field)` compares offset-precisely
        // against the equivalent `p + off` arithmetic (e.g. an argpromote seed
        // store written to `p + off`).
        if let Mnemonic::Gep(g) = ctx.get_insn(id).mnemonic() {
            acc += g.offset as i64;
            cur = g.base;
            continue;
        }
        let Mnemonic::Binop(b) = ctx.get_insn(id).mnemonic() else {
            break;
        };
        let (op, lhs, rhs) = (b.op, b.lhs, b.rhs);
        let lit = |v: ValueId| match v {
            ValueId::Literal(lid) => Some(ctx.values.literals[lid].value as i64),
            _ => None,
        };
        match op {
            Binop::Int(IntBinop::Add) => {
                if let Some(c) = lit(rhs) {
                    acc += c;
                    cur = lhs;
                } else if let Some(c) = lit(lhs) {
                    acc += c;
                    cur = rhs;
                } else {
                    break;
                }
            }
            Binop::Int(IntBinop::Sub) => {
                if let Some(c) = lit(rhs) {
                    acc -= c;
                    cur = lhs;
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
    (cur, acc)
}

/// The base a pointer is anchored to, for offset-precise disjointness.
/// `Absolute` is a literal (global) address — all literals share one base, so
/// they compare purely by offset. `Sym` is an opaque SSA base value; two `Sym`s
/// compare only when they are the *same* value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AddrBase {
    Absolute,
    Sym(ValueId),
}

/// Decompose `ptr` into `(base, byte_offset)`, folding a literal base into an
/// `Absolute` anchor so `0x1000` and `p + 4` are each described precisely.
fn addr_key(ctx: &Context, ptr: ValueId) -> (AddrBase, i64) {
    if let ValueId::Literal(lid) = ptr {
        return (AddrBase::Absolute, ctx.values.literals[lid].value as i64);
    }
    let (base, off) = base_plus_offset(ctx, ptr);
    match base {
        ValueId::Literal(lid) => (
            AddrBase::Absolute,
            ctx.values.literals[lid].value as i64 + off,
        ),
        other => (AddrBase::Sym(other), off),
    }
}

/// True when `store` (a store of `store_size` bytes at `store_ptr`) provably does
/// not overlap any `load` at `load_ptr`/`load_size`: they share a comparable base
/// (both absolute, or the same symbolic base) and their byte ranges are disjoint.
/// Incomparable bases conservatively count as a possible overlap (not disjoint).
fn disjoint_access(
    ctx: &Context,
    store_ptr: ValueId,
    store_size: usize,
    load_ptr: ValueId,
    load_size: usize,
) -> bool {
    let (sb, so) = addr_key(ctx, store_ptr);
    let (lb, lo) = addr_key(ctx, load_ptr);
    sb == lb && !intervals_overlap((so, so + store_size as i64), (lo, lo + load_size as i64))
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

/// A byte interval overwritten before any read, keyed by an [`AddrBase`] so
/// instruction-computed pointers (`p`, `p + 4`, …) — which have no entry in the
/// alias interval map (`value_to_interval` only holds `Literal`/`Varnode` keys)
/// — can still be proven covered by a later store to the same base.
///
/// Purely block-local: created and consumed within a single backward scan, never
/// seeded or propagated across blocks. That keeps it sound without any
/// escape/locality proof — a kill only exists because a *later store in this same
/// block* must-covers the victim, so the victim is overwritten before any exit.
#[derive(Clone, Copy)]
struct RelKill {
    space: SpaceId,
    base: AddrBase,
    start: i64,
    end: i64,
}

/// True when the kills sharing `(space, base)` fully cover `range`.
fn rel_fully_covered(kills: &[RelKill], space: SpaceId, base: AddrBase, range: (i64, i64)) -> bool {
    let set: Vec<(i64, i64)> = kills
        .iter()
        .filter(|k| k.space == space && k.base == base)
        .map(|k| (k.start, k.end))
        .collect();
    fully_covered(&set, range)
}

/// Drop kills a load at `(space, base, range)` may read. Same-base kills have the
/// read range punched out (possibly splitting an interval); a kill on a
/// different — hence incomparable — base in the same space cannot be proven
/// disjoint from the read and is dropped (mirrors the absolute path dropping
/// same-space kills on an unknown read location).
fn rel_punch_on_load(kills: &mut Vec<RelKill>, space: SpaceId, base: AddrBase, range: (i64, i64)) {
    let mut next = Vec::with_capacity(kills.len());
    for k in kills.drain(..) {
        if k.space != space {
            next.push(k);
        } else if k.base != base {
            // Incomparable / possibly-aliasing read: cannot keep the kill.
        } else if k.end <= range.0 || k.start >= range.1 {
            next.push(k);
        } else {
            if k.start < range.0 {
                next.push(RelKill { end: range.0, ..k });
            }
            if k.end > range.1 {
                next.push(RelKill {
                    start: range.1,
                    ..k
                });
            }
        }
    }
    *kills = next;
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
    // Block-local overwrite tracking for instruction-computed pointers (which
    // have no alias interval). Never seeded/propagated — see [`RelKill`].
    let mut rel_killed: Vec<RelKill> = Vec::new();

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
                // Relative path: a read of `base + off` clears any same-base
                // overwrite it overlaps, and any incomparable same-space kill.
                let (lb, lo) = addr_key(ctx, load.ptr);
                rel_punch_on_load(&mut rel_killed, load.space, lb, (lo, lo + load.size as i64));
                live.push(LiveLoc {
                    ptr: load.ptr,
                    size: load.size,
                    space: load.space,
                });
            }
            Mnemonic::Store(store) => {
                let ptr = store.ptr;
                let ptr_iv = aliases.interval(ptr).map(KilledInterval::from_alias);
                // A live load blocks the store only if it may-alias *and* is not
                // provably offset-disjoint: `may_alias` is class-based (true for
                // any same-base access), so refine it with the offset-precise
                // `disjoint_access` so a disjoint read of `base + k` does not pin
                // an unrelated `base + j` store.
                let no_live_reader = !live.iter().any(|l| {
                    l.space == store.space
                        && aliases.may_alias(ctx, ptr, l.ptr)
                        && !disjoint_access(ctx, ptr, store.size, l.ptr, l.size)
                });
                // Relative coverage for instruction-computed pointers with no
                // alias interval: a later store to the same base must-covers this
                // one. Sound for any space (RAM included) — overwrite before exit.
                let (sb, so) = addr_key(ctx, ptr);
                let rel_range = (so, so + store.size as i64);
                let is_killed = ptr_iv.is_some_and(|iv| fully_covered_iv(&killed, iv))
                    || rel_fully_covered(&rel_killed, store.space, sb, rel_range);
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
                    rel_killed.push(RelKill {
                        space: store.space,
                        base: sb,
                        start: rel_range.0,
                        end: rel_range.1,
                    });
                }
            }
            // An `externally_resolved` call reads no registers and writes exactly
            // its caller-saved clobber set (a C prototype under a known calling
            // convention). So its clobbered registers are *killed* here — a
            // preceding store to one of them, with no read in between, is dead —
            // and the call satisfies any post-call read of them (that read is the
            // call's own output, not the caller's pre-call value). Registers it
            // does not clobber (callee-saved) it neither reads nor writes, so their
            // existing kills pass through untouched.
            Mnemonic::Call(call)
                if Function::from_id(ctx, call.target).is_externally_resolved() =>
            {
                for iv in call_clobber_intervals(ctx, call.target) {
                    live.retain(|l| l.space != iv.space || !killed_covers_loc(ctx, iv, l));
                    killed.push(iv);
                }
                // A callee may read memory through pointer arguments, so no
                // relative (RAM/temp) overwrite survives across the call.
                rel_killed.clear();
            }
            // Any other call may read any register before its continuation
            // overwrites it, so a register store preceding the call cannot be proven
            // dead by a later (post-call) covering store. Drop register-space kills
            // at the call boundary; explicit `dead_reg` removals still apply (they
            // do not depend on `killed`). This keeps the caller's pre-call
            // stack-pointer decrement (see call_summary::decrement_stack_pointer)
            // alive so the callee's entry stack pointer is seeded correctly.
            Mnemonic::Call(_) | Mnemonic::CallInd(_) => {
                killed.retain(|k| !is_reg_space(ctx, k.space));
                // The callee may read any memory it can reach (pointer args,
                // globals), so drop every relative overwrite at the barrier.
                rel_killed.clear();
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

/// A temp-space load: the space it reads, its pointer, and byte size.
type TempLoad = (SpaceId, ValueId, usize);
/// A candidate temp-space store: its instruction, space, pointer, and byte size.
type TempStore = (InstructionId, SpaceId, ValueId, usize);

/// Returns stores to temp/mysave spaces (not Register, not default RAM) from
/// which no load ever reads within `function_id`. These stores are dead because
/// the temp spaces are function-scoped and not observable outside.
///
/// The check is offset-precise: a store is dead only if every load in the same
/// space is provably disjoint from it ([`disjoint_access`]). This covers both
/// literal (global) addresses and symbolic `base + const` addresses sharing a
/// common base — the latter is what argpromote's shadow accesses look like, so a
/// store to one field of a base is removable even when another, disjoint field of
/// the same base is still loaded. A load on an *incomparable* base conservatively
/// keeps the store.
fn unread_temp_space_stores(ctx: &Context, function_id: FunctionId) -> HashSet<InstructionId> {
    let fun = Function::from_id(ctx, function_id);
    let mut loads: Vec<TempLoad> = Vec::new();
    let mut candidate_stores: Vec<TempStore> = Vec::new();

    for block in &fun {
        for &insn_id in block.instruction_ids() {
            match ctx.get_insn(insn_id).mnemonic() {
                Mnemonic::Load(load) if is_temp_space(ctx, load.space) => {
                    loads.push((load.space, load.ptr, load.size));
                }
                Mnemonic::Store(store) if is_temp_space(ctx, store.space) => {
                    candidate_stores.push((insn_id, store.space, store.ptr, store.size));
                }
                _ => {}
            }
        }
    }

    candidate_stores
        .into_iter()
        .filter(|&(_, space, sptr, ssize)| {
            !loads.iter().any(|&(lspace, lptr, lsize)| {
                lspace == space && !disjoint_access(ctx, sptr, ssize, lptr, lsize)
            })
        })
        .map(|(id, _, _, _)| id)
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

    /// Build a one-block function with a temp ("shadow") space modeling what
    /// argpromote leaves behind: `base` is an opaque pointer value (a register
    /// load), `store(base + store_off, _)` writes the shadow, and
    /// `v = load(base + load_off)` reads it back into a register (keeping the
    /// load live). Returns `(ctx, function, shadow_store_id)`.
    fn argpromote_shadow_fn(
        store_off: i64,
        load_off: i64,
    ) -> (Context<'static>, FunctionId, InstructionId) {
        use qcode::builder::Builder;
        let mut tc = TestContext::new();
        let shadow = tc.ctx.make_temp_space();
        let regsp = tc.reg_space;
        let r1 = tc.r1;

        let block_id = BasicBlock::make(&mut tc.ctx).id;
        let fid = {
            let mut f = Function::make(&mut tc.ctx, "f".into()).unwrap();
            f.add_block(block_id);
            f.id
        };

        let offset_ptr = |b: &mut Builder<'static, '_>, base: ValueId, off: i64| {
            if off == 0 {
                base
            } else {
                let c = b.context_mut().get_const(off as u64, 8).id();
                ValueId::Instruction(b.push_add(base, c).id)
            }
        };
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, block_id));
            let base = b.push_load::<false>(ValueId::Varnode(r1), 8, regsp).id();
            let store_ptr = offset_ptr(&mut b, base, store_off);
            let five = b.context_mut().get_const(5, 4).id();
            b.push_store(five, store_ptr, shadow);
            let load_ptr = offset_ptr(&mut b, base, load_off);
            let v = b.push_load::<false>(load_ptr, 4, shadow).id();
            b.push_store(v, ValueId::Varnode(r1), regsp);
            unsafe { b.dont_finalize() };
        }
        let store_id = BasicBlock::from_id(&tc.ctx, block_id)
            .instruction_ids()
            .iter()
            .copied()
            .find(|&id| {
                matches!(tc.ctx.get_insn(id).mnemonic(),
                    Mnemonic::Store(s) if is_temp_space(&tc.ctx, s.space))
            })
            .unwrap();
        (tc.ctx, fid, store_id)
    }

    /// Regression for argpromote leftover shadow stores: a store into a
    /// function-scoped temp space is dead at the function's exit, so a store to
    /// one offset of a base is removable even when a *disjoint* offset of the
    /// same base is still loaded. Before the offset-precise check this was kept,
    /// because the coarse "space has a load" fallback could not tell the two
    /// `base + const` addresses apart.
    #[test]
    fn temp_store_dead_when_only_disjoint_offset_is_read() {
        // store base+0 (4 bytes), load base+8 (4 bytes): disjoint => dead.
        let (mut ctx, fid, store_id) = argpromote_shadow_fn(0, 8);
        assert!(
            unread_temp_space_stores(&ctx, fid).contains(&store_id),
            "temp-space store to base+0 is dead when only base+8 is read"
        );
        let aliases = AliasResult::simple(&ctx);
        assert!(
            remove_dead_load_insns(&mut ctx, fid, Some(&aliases), &[]),
            "the dead shadow store should be removed end-to-end"
        );
        assert!(
            ctx.get_insn(store_id).parent().is_none(),
            "the dead shadow store is gone after DSE"
        );
    }

    /// Guard: the same store must be kept when the surviving load *does* overlap
    /// it (same base, overlapping byte range), so the offset-precise rule never
    /// drops a store whose value is still observed.
    #[test]
    fn temp_store_kept_when_overlapping_offset_is_read() {
        // store base+0 (4 bytes), load base+2 (4 bytes): ranges overlap => live.
        let (ctx, fid, store_id) = argpromote_shadow_fn(0, 2);
        assert!(
            !unread_temp_space_stores(&ctx, fid).contains(&store_id),
            "temp-space store must be kept when an overlapping offset is read"
        );
    }

    /// Build a one-block function modeling the array-build idiom: a base pointer
    /// (`@ESP - k` in real IR, an opaque load here), several constant 4-byte RAM
    /// stores at `base + 0/4/8`, then a wide store covering them. The covering
    /// store must-covers each narrow one (same symbolic base, contained offsets),
    /// so all three are dead — even though they are RAM-space stores through
    /// instruction-computed pointers with no alias interval. `with_read` inserts a
    /// load of `base + 4` between the narrow stores and the covering store.
    fn array_build_block(with_read: bool) -> (Context<'static>, BlockId) {
        build_block(|b| {
            let r1 = b.context().get_named("r1").unwrap().as_varnode().unwrap();
            let regsp = reg_space(b.context());
            let ram = b.context().default_space;
            let base = b.push_load::<false>(ValueId::Varnode(r1), 8, regsp).id();

            let at = |b: &mut Builder<'static, '_>, off: u64| {
                if off == 0 {
                    base
                } else {
                    let c = b.context_mut().get_const(off, 8).id();
                    ValueId::Instruction(b.push_add(base, c).id)
                }
            };

            for off in [0u64, 4, 8] {
                let v = b.context_mut().get_const(0x1111_1111 + off, 4).id();
                let p = at(b, off);
                b.push_store(v, p, ram);
            }

            if with_read {
                // A read of base+4 before the overwrite keeps that store live.
                let p = at(b, 4);
                let v = b.push_load::<false>(p, 4, ram).id();
                b.push_store(v, ValueId::Varnode(r1), regsp);
            }

            // Covering 12-byte store at base+0.
            let blob = b.context_mut().get_bytes(vec![0u8; 12]).id();
            b.push_store(blob, base, ram);
        })
    }

    /// REPRO of the reported sample: RAM stores through `base - k` pointers, an
    /// intervening non-memory instruction (`%m`, modeling the un-folded map),
    /// then a covering store to `base - 4`. The early `base-4` store and the
    /// duplicate `base-12` / `base-8` stores should all be dead.
    #[test]
    fn repro_sample_dead_stores() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 BASE;
            <block>
                %base = load(i32, &BASE);
                %p4 = %base - i32 4;
                %p8 = %base - i32 8;
                %pc = %base - i32 12;
                store(%p4, i32 0x1f1e1d2c);
                store(%p8, %p4);
                store(%pc, i32 0x44c420);
                store(%pc, i32 0x44c420);
                store(%p8, %p4);
                %m = %base + i32 0;
                store(%p4, %m);
            "
        );
        let block_id = block;
        let aliases = AliasResult::simple(&ctx);
        let dead = dead_load_insns(&ctx, block_id, Some(&aliases), &[]);
        let stores = ram_stores(&ctx, block_id);
        let which: Vec<bool> = stores.iter().map(|s| dead.contains(s)).collect();
        // Each earlier writer is overwritten before any read (the intervening
        // `%m` is not a memory op), so stores 0/1/2 are dead; the last writer to
        // each slot (3/4/5) survives. None of this depends on folding the map.
        assert_eq!(which, vec![true, true, true, false, false, false]);
    }

    /// The `<$>` macro surface lowers to a real `Map` instruction whose body is
    /// the named function — end-to-end check of the proc-macro support.
    #[test]
    fn qcode_map_lowers_to_map_insn() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 SRC;
            fn inc:
                <inc_entry @e:i8>
                    %r = @e + i8 1;
                    return [%r];
            fn host:
                <host_entry>
                    %s = load(i32, &SRC);
                    %m = inc <$> %s;
                    return [%m];
            "
        );
        let root = Function::from_id(&ctx, host).root().unwrap().id;
        let has_map = BasicBlock::from_id(&ctx, root)
            .iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Map(m) if m.body == inc));
        assert!(has_map, "`inc <$> %s` should lower to a Map with body=inc");
    }

    fn ram_stores(ctx: &Context, block_id: BlockId) -> Vec<InstructionId> {
        BasicBlock::from_id(ctx, block_id)
            .instruction_ids()
            .iter()
            .copied()
            .filter(|&id| matches!(ctx.get_insn(id).mnemonic(), Mnemonic::Store(_)))
            .collect()
    }

    /// RAM stores through `base + off` pointers (no alias interval) are killed by
    /// a later wide store to the same base — the array-build dead-store case.
    #[test]
    fn ram_stores_covered_by_wide_store_are_dead() {
        let (ctx, block_id) = array_build_block(false);
        let aliases = AliasResult::simple(&ctx);
        let dead = dead_load_insns(&ctx, block_id, Some(&aliases), &[]);

        let stores = ram_stores(&ctx, block_id);
        assert_eq!(stores.len(), 4, "three narrow stores + one covering store");
        for (i, &s) in stores[..3].iter().enumerate() {
            assert!(dead.contains(&s), "narrow store {i} is covered → dead");
        }
        assert!(
            !dead.contains(&stores[3]),
            "the covering store is live (it is the surviving value)"
        );
    }

    /// Offset-precise liveness: a read of `base + 4` before the covering store
    /// keeps *only* the `base + 4` store live; the disjoint `base + 0` / `base + 8`
    /// stores are still dead. Without `disjoint_access` the class-based
    /// `may_alias` would conservatively pin all three.
    #[test]
    fn ram_store_kept_only_for_read_offset() {
        let (ctx, block_id) = array_build_block(true);
        let aliases = AliasResult::simple(&ctx);
        let dead = dead_load_insns(&ctx, block_id, Some(&aliases), &[]);

        // Stores in program order: base+0, base+4, base+8, [read's reg store], cover.
        let stores = ram_stores(&ctx, block_id);
        assert!(
            !dead.contains(&stores[1]),
            "store to base+4 is read before the overwrite → kept"
        );
        assert!(
            dead.contains(&stores[0]),
            "base+0 is disjoint from the read → still dead"
        );
        assert!(
            dead.contains(&stores[2]),
            "base+8 is disjoint from the read → still dead"
        );
    }

    /// Build a one-block caller that stores `0x1` into register `r`, then calls
    /// `callee`. Returns `(ctx, block, store_id)`.
    fn store_then_call_fn(
        configure_callee: impl FnOnce(&mut Context, FunctionId),
    ) -> (Context<'static>, BlockId, InstructionId) {
        let mut tc = TestContext::new();
        let (r, reg) = (tc.r0, tc.reg_space);

        let callee = Function::make_external(&mut tc.ctx, 0x9000, Some("ext".into())).id;
        configure_callee(&mut tc.ctx, callee);

        let caller = Function::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        Function::from_id_mut(&mut tc.ctx, caller)
            .set_root(entry)
            .unwrap();
        let store_id;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            let v = b.context_mut().get_const(0x1u64, 8).id();
            store_id = b.push_store(v, ValueId::Varnode(r), reg).id;
            b.push_call(callee);
            unsafe { b.dont_finalize() };
        }
        (tc.ctx, entry, store_id)
    }

    /// A store to a register an `externally_resolved` callee clobbers, never read
    /// before the call, is dead: the call overwrites it and reads no registers.
    #[test]
    fn store_before_resolved_call_to_clobbered_reg_is_dead() {
        let (ctx, block, store_id) = store_then_call_fn(|ctx, callee| {
            let r0 = ctx.get_named("r0").unwrap().as_varnode().unwrap();
            Function::from_id_mut(ctx, callee).set_clobbered_regs(vec![r0]);
            Function::from_id_mut(ctx, callee).set_externally_resolved(true);
        });
        let aliases = AliasResult::simple(&ctx);
        let dead = dead_load_insns(&ctx, block, Some(&aliases), &[]);
        assert!(
            dead.contains(&store_id),
            "store to a clobbered register before a resolved call is dead"
        );
    }

    /// The same store is *kept* when the callee is not `externally_resolved`: the
    /// call may read the register (e.g. an argument), so the store is live.
    #[test]
    fn store_before_unresolved_call_is_kept() {
        let (ctx, block, store_id) = store_then_call_fn(|ctx, callee| {
            let r0 = ctx.get_named("r0").unwrap().as_varnode().unwrap();
            // Clobbers r0 but is *not* marked resolved.
            Function::from_id_mut(ctx, callee).set_clobbered_regs(vec![r0]);
        });
        let aliases = AliasResult::simple(&ctx);
        let dead = dead_load_insns(&ctx, block, Some(&aliases), &[]);
        assert!(
            !dead.contains(&store_id),
            "store before an unresolved call must be kept (the callee may read it)"
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
