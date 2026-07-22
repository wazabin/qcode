use qcode::value::QCodeMut;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{AliasResult, ContextView, FunctionBody, Outcome};
use jstd::graph::analysis::{compute_dominators, compute_postdominators};
use qcode::{
    context::Context,
    space::{LocalMemorySpaceId, Space, SpaceType},
    value::{
        BlockId, FunctionId, ModuleView, QCodeView, ValueId, Varnode,
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
    pub space: LocalMemorySpaceId,
}

/// A byte interval `[start, end)` overwritten by a covering store before any
/// read.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct KilledInterval {
    pub space: LocalMemorySpaceId,
    pub start: u64,
    pub end: u64,
}

impl KilledInterval {
    fn from_alias(iv: (LocalMemorySpaceId, u64, u64)) -> Self {
        let (space, start, end) = iv;
        Self { space, start, end }
    }
}

/// Locations that may still be read.
pub(crate) type LiveSet = Vec<LiveLoc>;
/// Byte intervals overwritten before any read.
pub(crate) type KilledSet = Vec<KilledInterval>;

pub(crate) fn is_reg_space<'a, 'str: 'a>(
    src: impl qcode::value::util::base_ref::AsShared<'a, 'str>,
    space_id: impl Into<LocalMemorySpaceId>,
) -> bool {
    space_id
        .into()
        .shared()
        .is_some_and(|space_id| matches!(Space::from_id(src, space_id).ty, SpaceType::Register))
}

/// The byte intervals a resolved external `target` clobbers (the outputs of its
/// materialized interface map: return register(s) ∪ ABI caller-saved set), as
/// kills for the backward scan. Empty for a callee with no materialized map.
fn call_clobber_intervals<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    target: FunctionId,
) -> Vec<KilledInterval> {
    let Some(clobbered) = host
        .interface(target)
        .effects
        .materialized()
        .map(|map| map.outputs.as_slice())
    else {
        return Vec::new();
    };
    clobbered
        .iter()
        .map(|&vn| {
            let v = Varnode::from_id(host.shared(), vn);
            let start = v.address() as u64;
            KilledInterval {
                space: v.space().id.into(),
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
fn killed_covers_loc<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    iv: KilledInterval,
    l: &LiveLoc,
) -> bool {
    let ValueId::Varnode(vn) = l.ptr else {
        return false;
    };
    let v = Varnode::from_id(host.shared(), vn);
    if v.space().id != iv.space {
        return false;
    }
    let start = v.address() as u64;
    let end = start + v.size() as u64;
    iv.start <= start && end <= iv.end
}

pub(crate) fn is_temp_space<'a, 'str: 'a>(
    src: impl qcode::value::util::base_ref::AsShared<'a, 'str>,
    space_id: impl Into<LocalMemorySpaceId>,
) -> bool {
    let shared = src.as_shared();
    match space_id.into() {
        LocalMemorySpaceId::Shared(space_id) => {
            !is_reg_space(shared, space_id) && space_id != shared.default_space
        }
        LocalMemorySpaceId::Temp(_) => true,
    }
}

/// True for spaces whose stores are eligible for cross-block dead-store
/// elimination: register space and function-scoped temp spaces. The default
/// (RAM/global) space is excluded because such stores may be observed outside
/// the function.
pub(crate) fn is_tracked_space<'a, 'str: 'a>(
    src: impl qcode::value::util::base_ref::AsShared<'a, 'str>,
    space_id: impl Into<LocalMemorySpaceId>,
) -> bool {
    let shared = src.as_shared();
    let space_id = space_id.into();
    is_reg_space(shared, space_id) || is_temp_space(shared, space_id)
}

/// Decompose `ptr` into `(base, byte_offset)` by peeling literal `+`/`-` terms,
/// so `base + 8` and `base` (decomposing to the same `base`) can be compared
/// by their static offset. A pointer with no recognizable arithmetic is its own
/// base at offset 0. Mirrors `gvn::affine::base_offset` but is self-contained
/// (no `Numbering`) and used only for disjointness, where over-conservatism is
/// always safe.
fn base_plus_offset<'a, 'str: 'a>(host: impl QCodeView<'a, 'str>, ptr: ValueId) -> (ValueId, i64) {
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
        if let Mnemonic::Gep(g) = host.insn_ref(id).mnemonic() {
            acc += g.offset as i64;
            cur = g.base.qualify(id.func);
            continue;
        }
        let Mnemonic::Binop(b) = host.insn_ref(id).mnemonic() else {
            break;
        };
        let (op, lhs, rhs) = (b.op, b.lhs.qualify(id.func), b.rhs.qualify(id.func));
        let lit = |v: ValueId| match v {
            ValueId::Literal(lid) => Some(host.shared().values.literals[lid].value as i64),
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
fn addr_key<'a, 'str: 'a>(host: impl QCodeView<'a, 'str>, ptr: ValueId) -> (AddrBase, i64) {
    if let ValueId::Literal(lid) = ptr {
        return (
            AddrBase::Absolute,
            host.shared().values.literals[lid].value as i64,
        );
    }
    let (base, off) = base_plus_offset(host, ptr);
    match base {
        ValueId::Literal(lid) => (
            AddrBase::Absolute,
            host.shared().values.literals[lid].value as i64 + off,
        ),
        other => (AddrBase::Sym(other), off),
    }
}

/// True when `store` (a store of `store_size` bytes at `store_ptr`) provably does
/// not overlap any `load` at `load_ptr`/`load_size`: they share a comparable base
/// (both absolute, or the same symbolic base) and their byte ranges are disjoint.
/// Incomparable bases conservatively count as a possible overlap (not disjoint).
fn disjoint_access<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    store_ptr: ValueId,
    store_size: usize,
    load_ptr: ValueId,
    load_size: usize,
) -> bool {
    let (sb, so) = addr_key(host, store_ptr);
    let (lb, lo) = addr_key(host, load_ptr);
    sb == lb && !intervals_overlap((so, so + store_size as i64), (lo, lo + load_size as i64))
}

fn ptr_offset<'a, 'str: 'a>(host: impl QCodeView<'a, 'str>, ptr: ValueId) -> Option<i64> {
    match ptr {
        ValueId::Varnode(id) => Some(Varnode::from_id(host.shared(), id).address()),
        ValueId::Literal(lid) => Some(host.shared().values.literals[lid].value as i64),
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
    space: LocalMemorySpaceId,
    base: AddrBase,
    start: i64,
    end: i64,
}

/// True when the kills sharing `(space, base)` fully cover `range`.
fn rel_fully_covered(
    kills: &[RelKill],
    space: LocalMemorySpaceId,
    base: AddrBase,
    range: (i64, i64),
) -> bool {
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
fn rel_punch_on_load(
    kills: &mut Vec<RelKill>,
    space: LocalMemorySpaceId,
    base: AddrBase,
    range: (i64, i64),
) {
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
pub fn dead_load_insns<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    block_id: BlockId,
    aliases: Option<&AliasResult>,
    dead_regs: &[ValueId],
) -> HashSet<InstructionId> {
    let insns: Vec<InstructionId> = host.block_ref(block_id).instruction_ids().to_vec();
    let mut dead = block_dead_loads(host, block_id);

    if let Some(aliases) = aliases {
        // Alias-aware backward scan, seeded empty (single-block).
        scan_block_aliased(host, block_id, aliases, dead_regs, &mut dead, &[], &[]);
    } else {
        // Interval-based backward scan (register space only).
        // Dead loads already in the set are skipped so they don't prevent
        // their producing store from being marked dead.
        let mut live: Vec<(i64, i64)> = Vec::new();
        let mut killed: Vec<(i64, i64)> = Vec::new();

        for &id in insns.iter().rev() {
            match host.insn_ref(id).mnemonic() {
                Mnemonic::Load(load)
                    if is_reg_space(host.shared(), load.space) && !dead.contains(&id) =>
                {
                    if let Some(offset) = ptr_offset(host, load.ptr.qualify(id.func)) {
                        let range = (offset, offset + load.size as i64);
                        live.push(range);
                    }
                }
                Mnemonic::Store(store) if is_reg_space(host.shared(), store.space) => {
                    if let Some(offset) = ptr_offset(host, store.ptr.qualify(id.func)) {
                        let range = (offset, offset + store.size as i64);
                        let is_dead_reg = dead_regs.contains(&store.ptr.qualify(id.func));
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
fn block_dead_loads<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    block_id: BlockId,
) -> HashSet<InstructionId> {
    let mut dead = HashSet::default();
    let insns = host.block_ref(block_id).instruction_ids().to_vec();
    for id in insns {
        if let Mnemonic::Load(_) = host.insn_ref(id).mnemonic()
            && host
                .function(id.func)
                .users_of(ValueId::Instruction(id))
                .is_empty()
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
fn scan_block_aliased<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    block_id: BlockId,
    aliases: &AliasResult,
    dead_regs: &[ValueId],
    dead: &mut HashSet<InstructionId>,
    live_seed: &[LiveLoc],
    killed_seed: &[KilledInterval],
) -> (LiveSet, KilledSet) {
    let insns: Vec<InstructionId> = host.block_ref(block_id).instruction_ids().to_vec();
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
    let regs_dead_at_exit = host
        .block_ref(block_id)
        .function()
        .is_some_and(|f| f.is_reg_materialized());

    for &id in insns.iter().rev() {
        match host.insn_ref(id).mnemonic() {
            Mnemonic::Load(load) if !dead.contains(&id) => {
                let load_ptr = load.ptr.qualify(id.func);
                match aliases.interval(load_ptr) {
                    Some(iv) => punch_killed(&mut killed, KilledInterval::from_alias(iv)),
                    // Unknown read location: conservatively drop same-space kills.
                    None => killed.retain(|k| k.space != load.space),
                }
                // Relative path: a read of `base + off` clears any same-base
                // overwrite it overlaps, and any incomparable same-space kill.
                let (lb, lo) = addr_key(host, load_ptr);
                rel_punch_on_load(&mut rel_killed, load.space, lb, (lo, lo + load.size as i64));
                live.push(LiveLoc {
                    ptr: load_ptr,
                    size: load.size,
                    space: load.space,
                });
            }
            Mnemonic::Store(store) => {
                let ptr = store.ptr.qualify(id.func);
                let ptr_iv = aliases.interval(ptr).map(KilledInterval::from_alias);
                // A live load blocks the store only if it may-alias *and* is not
                // provably offset-disjoint: `may_alias` is class-based (true for
                // any same-base access), so refine it with the offset-precise
                // `disjoint_access` so a disjoint read of `base + k` does not pin
                // an unrelated `base + j` store.
                let no_live_reader = !live.iter().any(|l| {
                    l.space == store.space
                        && aliases.may_alias(host, ptr, l.ptr)
                        && !disjoint_access(host, ptr, store.size, l.ptr, l.size)
                });
                // Relative coverage for instruction-computed pointers with no
                // alias interval: a later store to the same base must-covers this
                // one. Sound for any space (RAM included) — overwrite before exit.
                let (sb, so) = addr_key(host, ptr);
                let rel_range = (so, so + store.size as i64);
                let is_killed = ptr_iv.is_some_and(|iv| fully_covered_iv(&killed, iv))
                    || rel_fully_covered(&rel_killed, store.space, sb, rel_range);
                let is_dead_reg = dead_regs.contains(&ptr)
                    || (regs_dead_at_exit && is_reg_space(host.shared(), store.space));
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
            // A call to a *materialized external* (a C prototype under a known
            // calling convention) reads no registers implicitly — its argument
            // reads are explicit `load`s at the (regpure-rewritten) site — and
            // writes exactly its interface map's outputs (return ∪ caller-saved
            // clobbers). So those registers are *killed* here — a preceding
            // store to one of them, with no read in between, is dead — and the
            // call satisfies any post-call read of them (that read is the call's
            // own output, not the caller's pre-call value). Registers it does
            // not clobber (callee-saved) it neither reads nor writes, so their
            // existing kills pass through untouched.
            // Guarded on the regpure tag: only a rewritten site has explicit
            // argument loads — an Opaque site to the same callee still reads its
            // argument registers implicitly from the register file, and on
            // x86-64 those are caller-saved (∈ outputs), so killing them would
            // delete the pre-call argument-setup stores.
            Mnemonic::Call(call)
                if call.tag.is_regpure()
                    && call.target.real().is_some_and(|target| {
                        let interface = host.interface(target);
                        interface.is_external && interface.effects.materialized().is_some()
                    }) =>
            {
                let target = call.target.real().unwrap();
                for iv in call_clobber_intervals(host, target) {
                    live.retain(|l| l.space != iv.space || !killed_covers_loc(host, iv, l));
                    killed.push(iv);
                }
                // A callee may read memory through pointer arguments, so no
                // relative (RAM/temp) overwrite survives across the call.
                rel_killed.clear();
                // Axiom sweep (argpromote shadow materialization): a private
                // body-local (shadow/temp) space is no longer unreachable by a
                // callee — a rebased shadow pointer handed to a prototyped
                // external lets the callee read that space. So a pre-call store
                // to a private space can no longer be proven dead by a covering
                // store after the call; drop private-space kills at the barrier.
                killed.retain(|k| k.space.shared().is_some());
            }
            // Any other call may read any register before its continuation
            // overwrites it, so a register store preceding the call cannot be proven
            // dead by a later (post-call) covering store. Drop register-space kills
            // at the call boundary; explicit `dead_reg` removals still apply (they
            // do not depend on `killed`). This keeps the caller's pre-call
            // stack-pointer decrement (see call_summary::decrement_stack_pointer)
            // alive so the callee's entry stack pointer is seeded correctly.
            Mnemonic::Call(_) | Mnemonic::CallInd(_) => {
                // Drop register-space kills, and (axiom sweep) private
                // body-local space kills too: a rebased shadow pointer passed to
                // a callee lets it read the caller's shadow/temp space, so a
                // pre-call private-space store is no longer dead across the call.
                // Shared non-register (RAM/global) kills pass through as before.
                killed.retain(|k| {
                    k.space.shared().is_some() && !is_reg_space(host.shared(), k.space)
                });
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
pub(crate) fn block_transfer<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    block_id: BlockId,
    aliases: &AliasResult,
    dead_regs: &[ValueId],
    live_seed: &[LiveLoc],
    killed_seed: &[KilledInterval],
) -> (LiveSet, KilledSet) {
    let mut dead = block_dead_loads(host, block_id);
    scan_block_aliased(
        host,
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
pub(crate) fn dead_load_insns_seeded<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    block_id: BlockId,
    aliases: &AliasResult,
    dead_regs: &[ValueId],
    live_seed: &[LiveLoc],
    killed_seed: &[KilledInterval],
) -> HashSet<InstructionId> {
    let mut dead = block_dead_loads(host, block_id);
    scan_block_aliased(
        host,
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
    let mut dead: Vec<_> = dead_load_insns(ModuleView::new(&*ctx), block_id, aliases, dead_regs)
        .into_iter()
        .collect();
    if dead.is_empty() {
        return;
    }
    dead.sort_unstable();
    for id in dead {
        ctx.remove_instruction(id);
    }
}

/// A temp-space load: the space it reads, its pointer, and byte size.
type TempLoad = (LocalMemorySpaceId, ValueId, usize);
/// A candidate temp-space store: its instruction, space, pointer, and byte size.
type TempStore = (InstructionId, LocalMemorySpaceId, ValueId, usize);

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
fn unread_temp_space_stores<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    function_id: FunctionId,
) -> HashSet<InstructionId> {
    let fun = host.function_ref(function_id);
    let mut loads: Vec<TempLoad> = Vec::new();
    let mut candidate_stores: Vec<TempStore> = Vec::new();

    for block in &fun {
        for insn_id in block.instruction_ids() {
            match host.insn_ref(insn_id).mnemonic() {
                Mnemonic::Load(load) if is_temp_space(host.shared(), load.space) => {
                    loads.push((load.space, load.ptr.qualify(insn_id.func), load.size));
                }
                Mnemonic::Store(store) if is_temp_space(host.shared(), store.space) => {
                    candidate_stores.push((
                        insn_id,
                        store.space,
                        store.ptr.qualify(insn_id.func),
                        store.size,
                    ));
                }
                _ => {}
            }
        }
    }

    candidate_stores
        .into_iter()
        .filter(|&(_, space, sptr, ssize)| {
            !loads.iter().any(|&(lspace, lptr, lsize)| {
                lspace == space && !disjoint_access(host, sptr, ssize, lptr, lsize)
            })
        })
        .map(|(id, _, _, _)| id)
        .collect()
}

/// Stores to **own-frame stack locals** whose slot is never read in the function.
///
/// An own-frame local (`@SP - k`, below the entry stack pointer) is private to
/// this activation: by frame freshness no incoming pointer names it, and the
/// frame is deallocated at return, so the caller can never observe it. A store to
/// such a slot is therefore dead when no load in the function may read it — no
/// covering later store required (like the `dead_reg`/temp-space rules, and unlike
/// the killed-set dataflow).
///
/// Conservatively disabled when the function makes any call (or indirect branch):
/// a callee could read the frame through a pointer it was handed, which is not
/// modeled as an explicit load here (mirrors `postdominated_dead_register_stores`).
/// Escape of a slot *address* into memory needs no special case: recovering the
/// slot requires a load through the reloaded (opaque) pointer, which `may_alias`
/// conservatively treats as reading every slot, so such a store is kept. Run to a
/// fixpoint, a chain of slots that only hold each other's addresses collapses.
fn unread_frame_local_stores<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    function_id: FunctionId,
    aliases: &AliasResult,
) -> HashSet<InstructionId> {
    let fun = host.function_ref(function_id);
    if fun.iter().flat_map(|block| block.iter()).any(|insn| {
        matches!(
            insn.mnemonic(),
            Mnemonic::Call(_) | Mnemonic::CallInd(_) | Mnemonic::BranchInd(_)
        )
    }) {
        return HashSet::default();
    }

    let mut loads: Vec<(ValueId, usize)> = Vec::new();
    let mut stores: Vec<(InstructionId, ValueId, usize)> = Vec::new();
    for block in &fun {
        for id in block.instruction_ids() {
            match host.insn_ref(id).mnemonic() {
                Mnemonic::Load(load) => loads.push((load.ptr.qualify(id.func), load.size)),
                Mnemonic::Store(store)
                    if aliases.is_own_frame_local(store.ptr.qualify(id.func)) =>
                {
                    stores.push((id, store.ptr.qualify(id.func), store.size))
                }
                _ => {}
            }
        }
    }

    stores
        .into_iter()
        .filter(|&(_, sptr, ssize)| {
            !loads.iter().any(|&(lptr, lsize)| {
                aliases.may_alias(host, sptr, lptr)
                    && !disjoint_access(host, sptr, ssize, lptr, lsize)
            })
        })
        .map(|(id, _, _)| id)
        .collect()
}

#[derive(Clone, Copy)]
struct RegisterStore {
    id: InstructionId,
    block: BlockId,
    ptr: ValueId,
    space: LocalMemorySpaceId,
}

/// Finds register stores that are overwritten by a covering store in a
/// postdominating block, with no reads of that register anywhere in the
/// function. This complements the killed-set dataflow for loop shapes where an
/// exit-block overwrite postdominates the entry store but is not propagated as a
/// must-kill through the loop backedge.
fn postdominated_dead_register_stores<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    function_id: FunctionId,
    aliases: &AliasResult,
) -> HashSet<InstructionId> {
    let function = host.function_ref(function_id);
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
        .filter(|&block| host.block_ref(block).successors().next().is_none())
        .collect();
    if exit_set.is_empty() {
        return HashSet::default();
    }
    let pdom = compute_postdominators(&function, &blocks, &node_set, &exit_set);

    let mut stores = Vec::new();
    let mut loads = Vec::new();
    for block in &blocks {
        let insns = host.block_ref(*block).instruction_ids().to_vec();
        for id in insns {
            match host.insn_ref(id).mnemonic() {
                Mnemonic::Store(store) if is_reg_space(host.shared(), store.space) => {
                    stores.push(RegisterStore {
                        id,
                        block: *block,
                        ptr: store.ptr.qualify(id.func),
                        space: store.space,
                    });
                }
                Mnemonic::Load(load) if is_reg_space(host.shared(), load.space) => {
                    loads.push((load.ptr.qualify(id.func), load.space));
                }
                _ => {}
            }
        }
    }

    stores
        .iter()
        .filter(|candidate| {
            !loads.iter().any(|&(ptr, space)| {
                space == candidate.space && aliases.may_alias(host, candidate.ptr, ptr)
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

/// True when the store `killer_ptr`/`killer_size` fully covers `cand_ptr`/`cand_size`:
/// both decompose to the same [`AddrBase`] and the killer's byte range encloses the
/// candidate's. Offset-precise (unlike the class-based [`AliasResult::covers`]), so a
/// whole-region write-back at `base` covers an earlier same-base `base + k` field
/// store even when neither has an alias interval.
fn ram_covers<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    cand_ptr: ValueId,
    cand_size: usize,
    killer_ptr: ValueId,
    killer_size: usize,
) -> bool {
    let (cb, co) = addr_key(host, cand_ptr);
    let (kb, ko) = addr_key(host, killer_ptr);
    cb == kb && ko <= co && co + cand_size as i64 <= ko + killer_size as i64
}

/// A RAM store/load with the block and in-block position needed to reason about
/// program order.
#[derive(Clone, Copy)]
struct RamAccess {
    id: InstructionId,
    block: BlockId,
    /// Index within its block's instruction list.
    pos: usize,
    ptr: ValueId,
    size: usize,
}

/// Blocks reachable from `start` by following ≥1 CFG edge (so `start` itself is
/// included only when it lies on a cycle).
fn forward_reachable<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    start: BlockId,
) -> HashSet<BlockId> {
    let mut seen: HashSet<BlockId> = HashSet::default();
    let mut stack: Vec<BlockId> = host.block_ref(start).successors().map(|(_, s)| s).collect();
    while let Some(b) = stack.pop() {
        if seen.insert(b) {
            stack.extend(host.block_ref(b).successors().map(|(_, s)| s));
        }
    }
    seen
}

/// RAM analog of [`postdominated_dead_register_stores`]. A store to the default
/// (RAM/global) space is dead when a covering store in a **postdominating** block
/// overwrites its region before every function exit and no surviving load can read
/// the store's value. Postdominators (not the killed-set dataflow) provide the
/// coverage, so the loop shape array_promote leaves behind — a pre-loop init/seed
/// store overwritten by the exit-block write-back across the loop backedge — is
/// recognized (the must-dataflow cannot cross the backedge).
///
/// A load `L` blocks candidate `S` (covered by killer `C`) only when it can observe
/// `S`'s value: it may-aliases the region, `S` can reach `L`, and `C` does **not**
/// dominate `L`. This excludes a load ordered before `S` (reads an earlier value) and
/// a load after the cover (reads `C`'s value) — e.g. array_promote's exit write-back
/// read — so only a genuine intervening read keeps the store.
///
/// The dominance exclusion is only valid for a killer that cannot **reach** the
/// candidate: with a `C →* S` path, `C` dominating `L` no longer implies `L` reads
/// `C`'s value (execution can run `C … S … L` with the next cover only after `L`),
/// so such killers are skipped.
///
/// Sound only for **call-free** functions (v1): a callee could read the region
/// through a pointer argument, which no explicit load models — array_promote's own
/// precondition. `return` needs no special case: an exit block has no postdominating
/// killer, so a store reaching it stays live.
fn postdominated_dead_ram_stores<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    function_id: FunctionId,
    aliases: &AliasResult,
) -> HashSet<InstructionId> {
    let function = host.function_ref(function_id);
    let blocks: Vec<BlockId> = function.iter().map(|block| block.id).collect();
    let Some(&entry) = blocks.first() else {
        return HashSet::default();
    };

    // A callee could observe the region through a pointer argument, and an indirect
    // branch makes the CFG (hence dominance) unreliable — gate the whole pass.
    if function.iter().flat_map(|block| block.iter()).any(|insn| {
        matches!(
            insn.mnemonic(),
            Mnemonic::Call(_) | Mnemonic::CallInd(_) | Mnemonic::BranchInd(_)
        )
    }) {
        return HashSet::default();
    }

    let ram = host.shared().default_space;
    let node_set: HashSet<BlockId> = blocks.iter().copied().collect();
    let exit_set: HashSet<BlockId> = blocks
        .iter()
        .copied()
        .filter(|&block| host.block_ref(block).successors().next().is_none())
        .collect();
    if exit_set.is_empty() {
        return HashSet::default();
    }
    let pdom = compute_postdominators(&function, &blocks, &node_set, &exit_set);
    let dom = compute_dominators(&function, entry);

    let mut stores: Vec<RamAccess> = Vec::new();
    let mut loads: Vec<RamAccess> = Vec::new();
    for &block in &blocks {
        let insns = host.block_ref(block).instruction_ids().to_vec();
        for (pos, id) in insns.into_iter().enumerate() {
            match host.insn_ref(id).mnemonic() {
                Mnemonic::Store(store) if store.space == ram => stores.push(RamAccess {
                    id,
                    block,
                    pos,
                    ptr: store.ptr.qualify(id.func),
                    size: store.size,
                }),
                Mnemonic::Load(load) if load.space == ram => loads.push(RamAccess {
                    id,
                    block,
                    pos,
                    ptr: load.ptr.qualify(id.func),
                    size: load.size,
                }),
                _ => {}
            }
        }
    }

    // `S` reaches `L`: another block reachable forward, or a same-block load that is
    // later in program order (or `S`'s block is itself on a cycle).
    let mut reach_cache: HashMap<BlockId, HashSet<BlockId>> = HashMap::default();
    let mut reaches = |s: &RamAccess, l: &RamAccess| -> bool {
        let set = reach_cache
            .entry(s.block)
            .or_insert_with(|| forward_reachable(host, s.block));
        set.contains(&l.block) || (s.block == l.block && l.pos > s.pos)
    };
    // `C` dominates `L`: strictly across blocks, or earlier in the same block.
    let c_dominates_l = |c: &RamAccess, l: &RamAccess| -> bool {
        if c.block == l.block {
            c.pos < l.pos
        } else {
            dom.dominates(c.block, l.block)
        }
    };

    let mut dead = HashSet::default();
    for cand in &stores {
        for killer in &stores {
            if killer.id == cand.id
                || killer.block == cand.block
                || !pdom
                    .get(&cand.block)
                    .is_some_and(|set| set.contains(&killer.block))
                || !ram_covers(host, cand.ptr, cand.size, killer.ptr, killer.size)
            {
                continue;
            }
            // The killer must not reach the candidate. Otherwise "C dominates L"
            // below stops implying "L reads C's value": with a C →* S path (cover
            // on a cycle back to the store, e.g. cover in a loop header, store in
            // the body), execution can run C … S … L with no cover in between —
            // C only re-executes *after* L observed S.
            if reaches(killer, cand) {
                continue;
            }
            let blocked = loads.iter().any(|l| {
                aliases.may_alias(host, cand.ptr, l.ptr)
                    && !disjoint_access(host, cand.ptr, cand.size, l.ptr, l.size)
                    && reaches(cand, l)
                    && !c_dominates_l(killer, l)
            });
            if !blocked {
                dead.insert(cand.id);
                break;
            }
        }
    }
    dead
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
    crate::with_body_mut(ctx, function_id, |body, cx| {
        remove_dead_load_insns_body(body, cx, function_id, aliases, dead_regs)
    })
}

/// Body-local core of [`remove_dead_load_insns`].
pub fn remove_dead_load_insns_body<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    function_id: FunctionId,
    aliases: Option<&AliasResult>,
    dead_regs: &[ValueId],
) -> bool {
    let block_ids: Vec<BlockId> = cx
        .body_view(body)
        .function_ref(function_id)
        .iter()
        .map(|block| block.id)
        .collect();

    let mut dead = HashSet::default();
    dead.extend(unread_temp_space_stores(cx.body_view(body), function_id));

    match aliases {
        Some(aliases) => {
            dead.extend(postdominated_dead_register_stores(
                cx.body_view(body),
                function_id,
                aliases,
            ));
            dead.extend(unread_frame_local_stores(
                cx.body_view(body),
                function_id,
                aliases,
            ));
            dead.extend(postdominated_dead_ram_stores(
                cx.body_view(body),
                function_id,
                aliases,
            ));
            let liveness = crate::mem::compute_memory_liveness(
                cx.body_view(body),
                function_id,
                aliases,
                dead_regs,
            );
            for &block_id in &block_ids {
                dead.extend(dead_load_insns_seeded(
                    cx.body_view(body),
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
                dead.extend(dead_load_insns(
                    cx.body_view(body),
                    block_id,
                    None,
                    dead_regs,
                ));
            }
        }
    }

    let changed = !dead.is_empty();
    let mut dead: Vec<_> = dead.into_iter().collect();
    dead.sort_unstable();
    for id in dead {
        body.remove_instruction(id);
    }
    changed
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use qcode::{
        builder::Builder,
        context::Context,
        testing::TestContext,
        value::{BasicBlock, FunctionBody, Value},
    };

    fn build_block(
        f: impl FnOnce(&mut Builder<'static, '_>),
    ) -> (qcode::context::Context<'static>, BlockId) {
        let mut ctx = TestContext::new().ctx;
        let block_id = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let mut builder = ctx.builder_at(0x1000);
        f(&mut builder);
        drop(builder);
        (ctx, block_id)
    }

    #[test]
    fn test_dead_load_eliminated() {
        let (ctx, block_id) = build_block(|b| {
            let rax_vn = b.shr().get_named("r0").unwrap().as_varnode().unwrap();
            let rax_ptr = ValueId::Varnode(rax_vn);
            let space = b.shr().named_spaces["register"];
            b.push_load::<false>(rax_ptr, 8, space);
        });

        let dead = dead_load_insns(ModuleView::new(&ctx), block_id, None, &[]);
        assert!(!dead.is_empty(), "dead load should be detected");
    }

    #[test]
    fn test_used_load_not_eliminated() {
        let (ctx, block_id) = build_block(|b| {
            let rax_vn = b.shr().get_named("r0").unwrap().as_varnode().unwrap();
            let rax_ptr = ValueId::Varnode(rax_vn);
            let space = b.shr().named_spaces["register"];
            let loaded_id = b.push_load::<false>(rax_ptr, 8, space).id();
            let one = b.shr().get_const(1, 8);
            b.push_add(loaded_id, one);
        });

        let dead = dead_load_insns(ModuleView::new(&ctx), block_id, None, &[]);
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
        let regsp = tc.reg_space;
        let r1 = tc.r1;

        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let shadow = LocalMemorySpaceId::Temp(
            tc.ctx.bodies[fid]
                .push_temp_space(qcode::value::TempSpace::new(Some("shadow"), 1, 8))
                .local,
        );
        let block_id = BasicBlock::make(&mut tc.ctx, fid).id;

        let offset_ptr = |b: &mut Builder<'static, '_>, base: ValueId, off: i64| {
            if off == 0 {
                base
            } else {
                let c = b.shr().get_const(off as u64, 8);
                ValueId::Instruction(b.push_add(base, c).id)
            }
        };
        {
            let mut b = tc.ctx.builder(block_id);
            let base = b.push_load::<false>(ValueId::Varnode(r1), 8, regsp).id();
            let store_ptr = offset_ptr(&mut b, base, store_off);
            let five = b.shr().get_const(5, 4);
            b.push_store(five, store_ptr, shadow);
            let load_ptr = offset_ptr(&mut b, base, load_off);
            let v = b.push_load::<false>(load_ptr, 4, shadow).id();
            b.push_store(v, ValueId::Varnode(r1), regsp);
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
            unread_temp_space_stores(ModuleView::new(&ctx), fid).contains(&store_id),
            "temp-space store to base+0 is dead when only base+8 is read"
        );
        let aliases = AliasResult::simple_for_function(&ctx, fid);
        assert!(
            remove_dead_load_insns(&mut ctx, fid, Some(&aliases), &[]),
            "the dead shadow store should be removed end-to-end"
        );
        assert!(
            !ctx.contains_instruction(store_id),
            "the dead shadow store payload is gone after DSE"
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
            !unread_temp_space_stores(ModuleView::new(&ctx), fid).contains(&store_id),
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
            let r1 = b.shr().get_named("r1").unwrap().as_varnode().unwrap();
            let regsp = b.shr().named_spaces["register"];
            let ram = b.shr().default_space;
            let base = b.push_load::<false>(ValueId::Varnode(r1), 8, regsp).id();

            let at = |b: &mut Builder<'static, '_>, off: u64| {
                if off == 0 {
                    base
                } else {
                    let c = b.shr().get_const(off, 8);
                    ValueId::Instruction(b.push_add(base, c).id)
                }
            };

            for off in [0u64, 4, 8] {
                let v = b.shr().get_const(0x1111_1111 + off, 4);
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
            let blob = b.shr().get_bytes(vec![0u8; 12]);
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
                %base = load(BASE:4, &BASE);
                %p4 = %base - i32 4;
                %p8 = %base - i32 8;
                %pc = %base - i32 12;
                store(ram:4, %p4 <- i32 0x1f1e1d2c);
                store(ram:4, %p8 <- %p4);
                store(ram:4, %pc <- i32 0x44c420);
                store(ram:4, %pc <- i32 0x44c420);
                store(ram:4, %p8 <- %p4);
                %m = %base + i32 0;
                store(ram:4, %p4 <- %m);
            "
        );
        let block_id = block;
        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        let dead = dead_load_insns(ModuleView::new(&ctx), block_id, Some(&aliases), &[]);
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
                    return at %r;
            fn host:
                <host_entry>
                    %s = load(SRC:4, &SRC);
                    %m = inc <$> %s;
                    return at %m;
            "
        );
        let root = FunctionBody::from_id(&ctx, host).root().unwrap().id;
        let has_map = BasicBlock::from_id(&ctx, root)
            .iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Map(m) if m.body.real() == Some(inc)));
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
        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        let dead = dead_load_insns(ModuleView::new(&ctx), block_id, Some(&aliases), &[]);

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
        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        let dead = dead_load_insns(ModuleView::new(&ctx), block_id, Some(&aliases), &[]);

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
    /// `callee` with the given site tag. Returns `(ctx, block, store_id)`.
    fn store_then_call_fn(
        tag: qcode::value::insn::CallTag,
        configure_callee: impl FnOnce(&mut Context, FunctionId),
    ) -> (Context<'static>, BlockId, InstructionId) {
        let mut tc = TestContext::new();
        let (r, reg) = (tc.r0, tc.reg_space);

        let callee = FunctionBody::make_external(&mut tc.ctx, 0x9000, Some("ext".into())).id;
        configure_callee(&mut tc.ctx, callee);

        let caller = FunctionBody::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let entry = { tc.ctx.get_or_make_block(0x1000, caller) };
        FunctionBody::from_id_mut(&mut tc.ctx, caller)
            .set_root(entry)
            .unwrap();
        let store_id;
        let call_id;
        {
            let mut b = tc.ctx.builder_at(0x1000);
            let v = b.shr().get_const(0x1u64, 8);
            store_id = b.push_store(v, ValueId::Varnode(r), reg).id;
            call_id = b.push_call(callee).id;
        }
        let Mnemonic::Call(mut call) = tc.ctx.get_insn(call_id).mnemonic().clone() else {
            unreachable!()
        };
        call.tag = tag;
        tc.ctx
            .replace_instruction_mnemonic(call_id, Mnemonic::Call(call));
        (tc.ctx, entry, store_id)
    }

    fn materialize_r0_out(ctx: &mut Context, callee: FunctionId) {
        let r0 = ctx.get_named("r0").unwrap().as_varnode().unwrap();
        FunctionBody::from_id_mut(ctx, callee).set_register_effects(
            qcode::value::RegisterChannelState::Materialized(qcode::value::RegisterInterfaceMap {
                inputs: vec![],
                outputs: vec![r0],
                returns: 0,
            }),
        );
    }

    /// A store to a register a materialized external callee clobbers, never read
    /// before the *regpure-rewritten* call, is dead: the call overwrites it and
    /// its argument reads are explicit SSA args, not register-file reads.
    #[test]
    fn store_before_resolved_regpure_call_to_clobbered_reg_is_dead() {
        let (ctx, block, store_id) =
            store_then_call_fn(qcode::value::insn::CallTag::RegPure, materialize_r0_out);
        let aliases = AliasResult::simple_for_function(
            &ctx,
            BasicBlock::from_id(&ctx, block).function().unwrap().id,
        );
        let dead = dead_load_insns(ModuleView::new(&ctx), block, Some(&aliases), &[]);
        assert!(
            dead.contains(&store_id),
            "store to a clobbered register before a regpure call is dead"
        );
    }

    /// Regression: the same store at a still-*Opaque* site must be KEPT. An
    /// Opaque call reads its argument registers implicitly from the register
    /// file, and on x86-64 those are caller-saved (∈ the output pack) — the
    /// materialized-external kill arm must not fire without the regpure tag.
    #[test]
    fn store_before_opaque_call_to_materialized_external_is_kept() {
        let (ctx, block, store_id) =
            store_then_call_fn(qcode::value::insn::CallTag::Opaque, materialize_r0_out);
        let aliases = AliasResult::simple_for_function(
            &ctx,
            BasicBlock::from_id(&ctx, block).function().unwrap().id,
        );
        let dead = dead_load_insns(ModuleView::new(&ctx), block, Some(&aliases), &[]);
        assert!(
            !dead.contains(&store_id),
            "store before an Opaque site must be kept (implicit argument reads)"
        );
    }

    /// The same store is *kept* when the callee's effects are unsolved: the call
    /// may read the register (e.g. an argument), so the store is live.
    #[test]
    fn store_before_unresolved_call_is_kept() {
        let (ctx, block, store_id) =
            store_then_call_fn(qcode::value::insn::CallTag::Opaque, |_ctx, _callee| {
                // Left with the default `Unsolved` effects (not materialized).
            });
        let aliases = AliasResult::simple_for_function(
            &ctx,
            BasicBlock::from_id(&ctx, block).function().unwrap().id,
        );
        let dead = dead_load_insns(ModuleView::new(&ctx), block, Some(&aliases), &[]);
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
                store(A:8, &A <- i64 1);
                store(A:8, &A <- i64 2);
            "
        );
        let block_id = block;

        let dead = dead_load_insns(ModuleView::new(&ctx), block_id, None, &[]);
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

    /// A redundant store to an `@SP`-rooted stack slot is dead even when a later
    /// load from a *global* address sits between exit and the store: under the
    /// stack-vs-global disjointness rule the global load cannot read the slot, so
    /// it no longer pins the overwritten store. Without frame freshness the global
    /// load conservatively may-alias the slot and keeps the store alive (the bug).
    #[test]
    fn redundant_stack_store_dead_despite_global_load() {
        use qcode::{
            testing::TestContext,
            value::{BasicBlock, FunctionBody},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = { tc.ctx.get_or_make_block(0x1000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx.block_param_mut(sp_pid).origin =
            Some(ValueId::Varnode(sp_reg).localize(sp_pid.func));
        let sp = ValueId::BlockParam(sp_pid);

        let ram = tc.ctx.shared.default_space;
        {
            let mut b = tc.ctx.builder(root);
            let c18 = b.shr().get_const(0x18, 8);
            let slot = b.push_sub(sp, c18).id(); // @SP - 0x18
            let val = b.shr().get_const(0x2f45c825, 4);
            b.push_store(val, slot, ram); // S1
            b.push_store(val, slot, ram); // S2 (identical → S1 dead)
            let glob = b.shr().get_const(0x454df8, 4);
            let g = b.push_load::<false>(glob, 4, ram).id(); // global load after the stores
            // Give the load a user so it is a live reader (not a pruned dead load).
            let zero = b.shr().get_const(0, 4);
            b.push_add(g, zero);
        }

        let store_ids: Vec<_> = BasicBlock::from_id(&tc.ctx, root)
            .instruction_ids()
            .iter()
            .copied()
            .filter(|&id| matches!(tc.ctx.get_insn(id).mnemonic(), Mnemonic::Store(_)))
            .collect();
        assert_eq!(store_ids.len(), 2);
        let (s1, s2) = (store_ids[0], store_ids[1]);

        // Without frame freshness the global load pins the redundant store.
        let plain = crate::AliasResult::simple_for_function(&tc.ctx, fid);
        let dead_plain = dead_load_insns(ModuleView::new(&tc.ctx), root, Some(&plain), &[]);
        assert!(
            !dead_plain.contains(&s1),
            "without stack/global disjointness the global load pins S1"
        );

        // With frame freshness, stack ⊥ global, so S1 is correctly dead.
        let r = crate::AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            qcode::value::ModuleView::new(&tc.ctx),
            fid,
            Some(sp_reg),
        );
        let dead = dead_load_insns(ModuleView::new(&tc.ctx), root, Some(&r), &[]);
        assert!(dead.contains(&s1), "the redundant first store is dead");
        assert!(!dead.contains(&s2), "the surviving last store is kept");
    }

    /// A store to an own-frame local (`@SP - k`) that no load ever reads is dead;
    /// a caller-frame slot (`@SP + k`) and a local whose slot *is* read are kept.
    #[test]
    fn unread_own_frame_local_store_is_dead() {
        use qcode::{
            testing::TestContext,
            value::{BasicBlock, FunctionBody},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = { tc.ctx.get_or_make_block(0x1000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx.block_param_mut(sp_pid).origin =
            Some(ValueId::Varnode(sp_reg).localize(sp_pid.func));
        let sp = ValueId::BlockParam(sp_pid);

        let ram = tc.ctx.shared.default_space;
        {
            let mut b = tc.ctx.builder(root);
            let c8 = b.shr().get_const(8, 8);
            let c10 = b.shr().get_const(0x10, 8);
            let v = b.shr().get_const(0x1234, 4);
            let local = b.push_sub(sp, c8).id(); // @SP - 8 (never read)
            b.push_store(v, local, ram); // dead
            let caller = b.push_add(sp, c8).id(); // @SP + 8 (caller frame)
            b.push_store(v, caller, ram); // kept (caller-visible)
            let read_local = b.push_sub(sp, c10).id(); // @SP - 0x10 (read below)
            b.push_store(v, read_local, ram); // kept (its slot is read)
            b.push_load::<false>(read_local, 4, ram);
        }

        let stores: Vec<_> = BasicBlock::from_id(&tc.ctx, root)
            .instruction_ids()
            .iter()
            .copied()
            .filter(|&id| matches!(tc.ctx.get_insn(id).mnemonic(), Mnemonic::Store(_)))
            .collect();
        let (s_local, s_caller, s_read) = (stores[0], stores[1], stores[2]);

        let r = crate::AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            qcode::value::ModuleView::new(&tc.ctx),
            fid,
            Some(sp_reg),
        );
        let dead = unread_frame_local_stores(ModuleView::new(&tc.ctx), fid, &r);

        assert!(
            dead.contains(&s_local),
            "unread own-frame local store is dead"
        );
        assert!(
            !dead.contains(&s_caller),
            "a caller-frame slot is observable, not dead"
        );
        assert!(
            !dead.contains(&s_read),
            "a local whose slot is loaded is not dead"
        );
    }

    #[test]
    fn test_store_read_then_overwrite_not_dead() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;
            <block>
                store(A:8, &A <- i64 1);
                %tmp = load(A:8, &A);
                store(A:8, &A <- i64 2);
            "
        );
        let block_id = block;

        let dead = dead_load_insns(ModuleView::new(&ctx), block_id, None, &[]);

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
            let ax_vn = b.shr().get_named("r0_lo16").unwrap().as_varnode().unwrap();
            let ah_vn = b.shr().get_named("r0_byte1").unwrap().as_varnode().unwrap();
            let space = b.shr().named_spaces["register"];
            let v = b.shr().get_const(0x1234u64, 2);
            b.push_store(v, ValueId::Varnode(ax_vn), space);
            let loaded_id = b.push_load::<false>(ValueId::Varnode(ah_vn), 1, space).id();
            let zero = b.shr().get_const(0, 1);
            b.push_add(loaded_id, zero);
        });

        let dead = dead_load_insns(ModuleView::new(&ctx), block_id, None, &[]);
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
                store(A:8, &A <- i64 1);
                store(A:8, &A <- i64 2);
            "
        );
        let block_id = block;

        let aliases = crate::AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        let dead = dead_load_insns(ModuleView::new(&ctx), block_id, Some(&aliases), &[]);
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
                store(A:8, &A <- i64 1);
                %tmp = load(A:8, &A);
                store(B:8, &B <- %tmp);
                store(A:8, &A <- i64 2);
            "
        );
        let block_id = block;

        let aliases = crate::AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        let dead = dead_load_insns(ModuleView::new(&ctx), block_id, Some(&aliases), &[]);
        let store_a_ids: Vec<_> = BasicBlock::from_id(&ctx, block_id)
            .instruction_ids()
            .iter()
            .copied()
            .filter(|&id| {
                if let Mnemonic::Store(s) = ctx.get_insn(id).mnemonic() {
                    s.ptr.qualify(id.func)
                        == ValueId::Varnode(ctx.get_named("A").unwrap().as_varnode().unwrap())
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
            let r0_lo32 = b.shr().get_named("r0_lo32").unwrap().as_varnode().unwrap();
            let r0 = b.shr().get_named("r0").unwrap().as_varnode().unwrap();
            let space = b.shr().named_spaces["register"];

            let narrow_val = b.shr().get_const(1u64, 4);
            b.push_store(narrow_val, ValueId::Varnode(r0_lo32), space);

            let wide_val = b.shr().get_const(2u64, 8);
            b.push_store(wide_val, ValueId::Varnode(r0), space);
        });

        let aliases = crate::AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        let dead = dead_load_insns(ModuleView::new(&ctx), block_id, Some(&aliases), &[]);

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
                    store(register:4, {r0_lo32} <- i32 1);
                    goto <loop_head>;

                <loop_head>
                    if i8 1 goto <loop_body> else goto <exit>;

                <loop_body>
                    store(register:8, {r1} <- i64 2);
                    goto <loop_head>;

                <exit>
                    store(register:4, {r0_lo32} <- i32 3);
                    return at i64 0;
            "
        );

        let aliases = crate::AliasResult::simple_for_function(&tc.ctx, test);
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

    // ----- RAM (default-space) dead-store elimination -----------------------

    /// Count `store`s to the default (RAM) space remaining in the whole function.
    fn ram_store_count(ctx: &Context, fid: FunctionId) -> usize {
        FunctionBody::from_id(ctx, fid)
            .iter()
            .flat_map(|b| b.iter().map(|i| i.id).collect::<Vec<_>>())
            .filter(|&id| {
                matches!(ctx.get_insn(id).mnemonic(), Mnemonic::Store(s) if s.space == ctx.shared.default_space)
            })
            .count()
    }

    /// The array_promote leftover shape: a pre-loop init store at `@base`, a loop that
    /// touches nothing in RAM, and an exit-block write-back covering `@base`. The exit
    /// store postdominates the entry store across the loop backedge, so the init store
    /// is dead — the killed-set dataflow can't cross the backedge, postdominance can.
    #[test]
    fn ram_covered_store_across_loop_is_dead() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn mix:
            <entry @base:i64 @src:i64>
                %v = load(ram:8, @src);
                store(ram:8, @base <- %v);
                goto <head @i=0>;
            <head @i:i64>
                %done = @i == 24;
                if %done goto <exit> else goto <body @j=@i>;
            <body @j:i64>
                %j1 = @j + 1;
                goto <head @i=%j1>;
            <exit>
                store(ram:8, @base <- i64 7);
                return at i64 0x0;
            "
        );
        let aliases = crate::AliasResult::simple_for_function(&ctx, mix);
        remove_dead_load_insns(&mut ctx, mix, Some(&aliases), &[]);
        // Only the exit write-back survives; the load from the *unrelated* `@src`
        // pointer, ordered before the store, must not keep it alive.
        assert_eq!(
            ram_store_count(&ctx, mix),
            1,
            "the covered pre-loop init store should be removed"
        );
    }

    /// A store with no covering later store reaches `return` and stays live.
    #[test]
    fn ram_store_reaching_return_lives() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn mix:
            <entry @base:i64>
                store(ram:8, @base <- i64 5);
                goto <exit>;
            <exit>
                return at i64 0x0;
            "
        );
        let aliases = crate::AliasResult::simple_for_function(&ctx, mix);
        remove_dead_load_insns(&mut ctx, mix, Some(&aliases), &[]);
        assert_eq!(
            ram_store_count(&ctx, mix),
            1,
            "uncovered store must survive"
        );
    }

    /// A read of the region between the store and its cover keeps the store live.
    #[test]
    fn ram_intervening_read_keeps_store() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn mix:
            <entry @base:i64 @sink:i64>
                store(ram:8, @base <- i64 5);
                goto <head @i=0>;
            <head @i:i64>
                %done = @i == 24;
                if %done goto <exit> else goto <body @j=@i>;
            <body @j:i64>
                %r = load(ram:8, @base);
                store(ram:8, @sink <- %r);
                %j1 = @j + 1;
                goto <head @i=%j1>;
            <exit>
                store(ram:8, @base <- i64 7);
                return at i64 0x0;
            "
        );
        let aliases = crate::AliasResult::simple_for_function(&ctx, mix);
        remove_dead_load_insns(&mut ctx, mix, Some(&aliases), &[]);
        // Both `@base` stores stay: the loop reads `@base` before the cover. (`@sink`
        // may alias `@base`, so its store is conservatively kept too.)
        assert!(
            ram_store_count(&ctx, mix) >= 2,
            "an intervening region read must keep the store"
        );
    }

    /// A load of the region *after* the covering store (reading the cover's value,
    /// not the candidate's) does not block removal.
    #[test]
    fn ram_read_after_cover_does_not_block() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn mix:
            <entry @base:i64 @sink:i64>
                store(ram:8, @base <- i64 5);
                goto <exit>;
            <exit>
                store(ram:8, @base <- i64 7);
                %r = load(ram:8, @base);
                store(ram:8, @sink <- %r);
                return at i64 0x0;
            "
        );
        let aliases = crate::AliasResult::simple_for_function(&ctx, mix);
        remove_dead_load_insns(&mut ctx, mix, Some(&aliases), &[]);
        // The entry store dies; the exit cover and the `@sink` store remain (2).
        assert_eq!(
            ram_store_count(&ctx, mix),
            2,
            "a post-cover read reads the cover's value and must not keep the store"
        );
    }

    /// A cover on a cycle back to the store must not qualify: with the cover in the
    /// loop header and the candidate + read in the body, execution runs
    /// `C … S … L` each iteration — `C` dominating `L` does not mean `L` reads `C`'s
    /// value, so deleting the body store would miscompile the read.
    #[test]
    fn ram_cover_reaching_back_to_store_does_not_kill() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn mix:
            <entry @base:i64>
                goto <head @i=0>;
            <head @i:i64>
                store(ram:8, @base <- i64 0);
                %done = @i == 24;
                if %done goto <exit> else goto <body @j=@i>;
            <body @j:i64>
                store(ram:1, @base <- i8 7);
                %v = load(ram:1, @base);
                %w = zext(i64, %v);
                %j1 = @j + %w;
                goto <head @i=%j1>;
            <exit>
                return at i64 0x0;
            "
        );
        let aliases = crate::AliasResult::simple_for_function(&ctx, mix);
        remove_dead_load_insns(&mut ctx, mix, Some(&aliases), &[]);
        assert_eq!(
            ram_store_count(&ctx, mix),
            2,
            "a cover that reaches back to the store must not kill it"
        );
    }

    /// End-to-end: array_promote leaves a pre-loop init store (read by its snapshot
    /// load), `gvn` forwards the snapshot away, and RAM-DSE then removes the now-dead
    /// init store — leaving only the exit write-back. Exercises the pipeline stage
    /// `["array_promote", …, "gvn", "dce", "dead_store"]`.
    #[test]
    fn array_promote_leftover_init_store_is_swept() {
        use crate::gvn::Gvn;
        use crate::mem::array_promote::ArrayPromote;
        use crate::test_util::run_function_pass;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn mix:
            <entry @src:i64 @base:i64>
                %init = load(ram:24, @src);
                store(ram:24, @base <- %init);
                goto <head @i=0>;
            <head @i:i64>
                %done = @i == 24;
                if %done goto <exit> else goto <body @j=@i>;
            <body @j:i64>
                %addr = @base + @j;
                %cur = load(ram:1, %addr);
                %jt = trunc(i8, @j);
                %x = %cur ^ %jt;
                store(ram:1, %addr <- %x);
                %j1 = @j + 1;
                goto <head @i=%j1>;
            <exit>
                return at i64 0x0;
            "
        );
        assert!(
            run_function_pass::<ArrayPromote>(&mut ctx, mix).unwrap(),
            "the enveloped byte fill should promote"
        );
        run_function_pass::<Gvn>(&mut ctx, mix).unwrap();
        let aliases = crate::AliasResult::simple_for_function(&ctx, mix);
        remove_dead_load_insns(&mut ctx, mix, Some(&aliases), &[]);
        // Only the exit write-back of the carried array survives; the init store and
        // the (forwarded) snapshot load of `@base` are gone.
        let ir = format!("{}", FunctionBody::from_id(&ctx, mix));
        assert_eq!(
            ram_store_count(&ctx, mix),
            1,
            "the leftover init store should be swept: {ir}"
        );
        assert!(
            !ir.contains("load(ram:24, i64 @base)"),
            "the snapshot load should be forwarded away: {ir}"
        );
    }

    /// The whole mechanism is gated on functions with no calls or indirect
    /// branches: control transfer to code out of view could read the region, so
    /// nothing is removed. `@base`'s cross-block cover would otherwise be dead.
    #[test]
    fn ram_dse_disabled_with_indirect_transfer() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn mix:
            <entry @base:i64 @p:i64>
                store(ram:8, @base <- i64 5);
                goto <mid>;
            <mid>
                store(ram:8, @base <- i64 7);
                goto [i64 @p];
            "
        );
        let aliases = crate::AliasResult::simple_for_function(&ctx, mix);
        remove_dead_load_insns(&mut ctx, mix, Some(&aliases), &[]);
        assert_eq!(
            ram_store_count(&ctx, mix),
            2,
            "an indirect transfer must disable cross-block RAM DSE"
        );
    }
}

// ----- pass ------------------------------------------------------------------

use crate::FunctionPass;

#[derive(Default)]
pub struct DeadLoad;

impl FunctionPass for DeadLoad {
    const NAME: &'static str = "dead_load";
    fn description(&self) -> &'static str {
        "Remove dead memory loads"
    }
    fn run<'str>(
        &self,
        f: &mut FunctionBody<'str>,
        m: ContextView<'_, 'str>,
        _next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String> {
        let fid = f.id();
        let aliases = frame_aware_aliases(m, m.body_view(f), fid);
        Ok(
            Outcome::changed(remove_dead_load_insns_body(f, m, fid, Some(&aliases), &[]))
                .preserving_global::<crate::CallGraphAnalysis>()
                .preserving_global::<crate::AddressAnalysis>(),
        )
    }
}

crate::register_function_pass!(DeadLoad);

/// Lives here (not in the orphaned `dead_store.rs`) because it shares
/// [`remove_dead_load_insns_body`] with [`DeadLoad`]; the only difference is that
/// it also treats the architecture's flag registers as dead.
#[derive(Default)]
pub struct DeadStore;

impl FunctionPass for DeadStore {
    const NAME: &'static str = "dead_store";
    fn description(&self) -> &'static str {
        "Remove dead register loads and overwritten flag stores"
    }
    fn run<'str>(
        &self,
        f: &mut FunctionBody<'str>,
        m: ContextView<'_, 'str>,
        _next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String> {
        let fid = f.id();
        let dead_regs = m.env().cfg.dead_flag_regs.clone();
        let aliases = frame_aware_aliases(m, m.body_view(f), fid);
        Ok(Outcome::changed(remove_dead_load_insns_body(
            f,
            m,
            fid,
            Some(&aliases),
            &dead_regs,
        ))
        .preserving_global::<crate::CallGraphAnalysis>()
        .preserving_global::<crate::AddressAnalysis>())
    }
}

crate::register_function_pass!(DeadStore);

/// Build a per-function alias oracle with frame-freshness populated (the same way
/// [`crate::gvn::Gvn`] does), so the dead-store/dead-load scans get the
/// stack-vs-global and own-frame disjointness rules. Falls back to an inert frame
/// when no stack-pointer register is registered. Reads the function through the
/// selected body view; the module-wide alias base and stack-pointer register
/// come from the [`ContextView`].
fn frame_aware_aliases<'a, 'str: 'a>(
    m: ContextView<'_, 'str>,
    host: impl QCodeView<'a, 'str>,
    fun_id: FunctionId,
) -> AliasResult {
    let shared = m.shr();
    let env = m.env();
    let sp_reg = shared.registers.get(&env.cfg.stack_pointer).copied();
    env.alias_base(shared)
        .for_function(host, fun_id)
        .with_frame_freshness(host, fun_id, sp_reg)
}
