//! Array promotion: "mem2reg for arrays".
//!
//! Recognizes an in-place, strided, real-RAM array-fill loop whose per-element
//! write depends on a *previously written* element re-read from memory (a
//! memory-carried recurrence), and rewrites the memory traffic into a value-level
//! carried array threaded through the loop with the `insert` / `at` intrinsics,
//! plus a single wide `store(ram, base <- arr)` at the loop exit.
//!
//! The motivating case is the honest MT19937 seeding loop (`mt19937_init.qcode`):
//!
//! ```text
//!   mt[0] = seed
//!   mt[i] = 1812433253 * (mt[i-1] ^ (mt[i-1] >> 30)) + i,   i = 1..624
//! ```
//!
//! Input shape (impure `fn`, base is an incoming pointer, result observed only
//! through `@base`):
//!
//! ```text
//!   <entry @seed @base>
//!       %e0 = trunc(i32, @seed);
//!       store(ram:4, @base <- %e0);              // seed lane 0
//!       goto <head @i=1 @buf=@base>;
//!   <head @i @buf>  %d = @i == N; if %d goto <exit> else goto <body @j=@i @b=@buf>;
//!   <body @j @b>
//!       %prev = load(ram:4, @b + (@j-1)*4);       // memory-carried read
//!       %next = f(%prev, @j);
//!       store(ram:4, @b + @j*4 <- %next);         // lane store
//!       goto <head @i=@j+1 @buf=@b>;
//!   <exit>  return …;
//! ```
//!
//! Output shape (the carried array `@arr:[i32;N]` threaded through the loop; the
//! `insert`-loop that [`loop_to_scan`](crate::calls) then folds into a `scanl`):
//!
//! ```text
//!   <entry @seed @base>
//!       %e0 = trunc(i32, @seed);
//!       %a0 = insert(zero:[i32;N], 0, %e0);
//!       goto <head @i=1 @buf=@base @arr=%a0>;
//!   <head @i @buf @arr>  … if … goto <exit @arr> else goto <body @j=@i @b=@buf @arr>;
//!   <body @j @b @arr>
//!       %prev = at(@arr, @j-1);
//!       %next = f(%prev, @j);
//!       %arr' = insert(@arr, @j, %next);
//!       goto <head @i=@j+1 @buf=@b @arr=%arr'>;
//!   <exit @arr>  store(ram, @base <- @arr); return …;
//! ```
//!
//! v1 soundness gate: the base pointer is a loop-invariant root param (proved via
//! pass-through phi union-find); the loop has the canonical preheader/header/body/
//! exit shape; the index steps by one and is bounded; the *only* memory accesses
//! reaching the region are the seed store, the one lane store, and the one lane
//! load; those lanes tile `[0, N)` exactly once; and the function contains no
//! calls (nothing else can observe or mutate the region mid-fill).

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    builder::Builder,
    context::Context,
    space::{Space, SpaceId, SpaceType},
    value::{
        BasicBlock, BlockId, Function, FunctionId, ValueId, ValueRef,
        insn::{Branch, CBranch, InstructionId, Mnemonic},
    },
};

use crate::gvn::affine::precompute_forms;
use crate::sequence::{affine_base_const, affine_strided_lane};
use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct ArrayPromote;

/// A recognized in-place array-fill loop.
///
/// Two loop shapes are recognized. In the *split* shape the loop body ends in a
/// `goto header` and a distinct header block holds the guard `cbranch`; in the
/// *rotated* (do-while) shape the body is its own header — it ends in the guard
/// `cbranch` with one edge back to itself. In the rotated shape `header == body`.
///
/// The carried per-element recurrence reaches the body one of two ways. Either
/// it is *memory-carried* — a strided lane load re-reads a previously written
/// element (`mt[j-1]`), the clean `mem2reg`-hasn't-forwarded-it shape — or it is
/// *register-carried*: a loop block param (`@EBX`) already holds the previous
/// element and the memory reload is gone (what the real pipeline produces once
/// `mem2reg`/argpromote have threaded the reload through a register). Either way
/// the carry becomes `at(arr, index-1)`; in the register-carried case the
/// accumulator param's uses are rewritten to that `at` and the now-dead param is
/// left for `dce`.
struct PromoteMatch {
    preheader: BlockId,
    header: BlockId,
    body: BlockId,
    exit: BlockId,
    /// `header == body` (single-block do-while loop).
    rotated: bool,
    /// Loop-invariant root pointer param the region is based at.
    base_root: ValueId,
    /// Region origin word offset from `base_root`: element 0 sits at
    /// `base_root + origin_word * elem_size`. All element positions below are
    /// relative to this origin (so element 0 is the seed regardless of a nonzero
    /// pointer bias in the lifted addresses).
    origin_word: i64,
    /// Body induction parameter (the lane index in the store/load addresses).
    index: ValueId,
    elem_size: usize,
    count: usize,
    /// Pre-loop element-0 seed store and the value it writes.
    seed_id: InstructionId,
    seed_val: ValueId,
    /// The RAM space the region lives in — all region accesses (seed, lane
    /// store/load, exit loads) share it, and the exit-store writes it back. Its
    /// word size is 1 (byte-addressed), so lane arithmetic in bytes is exact.
    region_space: SpaceId,
    /// The one strided lane store, writing element `index + store_delta`.
    lane_store_id: InstructionId,
    stored_val: ValueId,
    store_delta: i64,
    /// The carry value to rewrite into `at(arr, index + load_delta)`: either the
    /// memory-carried lane-load result or the register accumulator param.
    carry_replace: ValueId,
    /// `Some(load)` when the carry is a memory lane load (also removed).
    carry_remove: Option<InstructionId>,
    load_delta: i64,
    /// Additional const-offset region loads in the exit block, each rewritten to
    /// `at(arr_exit, element)`. Pairs of `(load insn, element index)`.
    extra_loads: Vec<(InstructionId, i64)>,
    /// Dead pre-loop stores to element 0 (overwritten by the seed store), removed
    /// with the rest of the promoted traffic.
    dead_stores: Vec<InstructionId>,
}

/// Minimal union-find over `ValueId` for pass-through phi resolution.
#[derive(Default)]
struct Uf {
    parent: HashMap<ValueId, ValueId>,
}

impl Uf {
    fn find(&mut self, v: ValueId) -> ValueId {
        let p = *self.parent.get(&v).unwrap_or(&v);
        if p == v {
            return v;
        }
        let r = self.find(p);
        self.parent.insert(v, r);
        r
    }

    fn union(&mut self, a: ValueId, b: ValueId) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent.insert(ra, rb);
        }
    }
}

/// Values feeding header-param index `k` from every predecessor edge.
fn header_incoming(ctx: &Context, header: BlockId, k: usize) -> Vec<ValueId> {
    let mut out = Vec::new();
    let preds: Vec<BlockId> = BasicBlock::from_id(ctx, header)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    for pred in preds {
        let Some(term) = BasicBlock::from_id(ctx, pred).iter().last() else {
            continue;
        };
        match term.mnemonic() {
            Mnemonic::Branch(b) => out.extend(b.args.get(k).copied()),
            Mnemonic::CBranch(cb) => {
                if cb.success_block == header {
                    out.extend(cb.success_args.get(k).copied());
                }
                if cb.failure_block == header {
                    out.extend(cb.failure_args.get(k).copied());
                }
            }
            _ => {}
        }
    }
    out
}

/// `c` if `v` is the integer literal `c`, else `None`.
fn literal(ctx: &Context, v: ValueId) -> Option<u64> {
    match ValueRef::new(v, ctx) {
        ValueRef::Literal(l) => Some(l.value()),
        _ => None,
    }
}

/// The trip bound `N` from a header guard `hi <cmp> N`, so the last body index is
/// `N-1`. `exit_on_true` says whether the guard's success edge leaves the loop.
/// Accepts only the two canonical polarities: `hi == N` exiting on true, or
/// `hi < N` (unsigned/signed) continuing on true.
fn guard_bound(ctx: &Context, cond: ValueId, hi: ValueId, exit_on_true: bool) -> Option<i64> {
    use qcode::value::insn::{Binary, Binop, IntBinop};
    let ValueId::Instruction(id) = cond else {
        return None;
    };
    let Mnemonic::Binop(Binary { lhs, rhs, op }) = ctx.get_insn(id).mnemonic() else {
        return None;
    };
    let konst = if *lhs == hi {
        *rhs
    } else if *rhs == hi {
        *lhs
    } else {
        return None;
    };
    let ok = match op {
        Binop::Int(IntBinop::Equal) => exit_on_true,
        Binop::Int(IntBinop::Less | IntBinop::SLess) => !exit_on_true && *lhs == hi,
        _ => false,
    };
    if !ok {
        return None;
    }
    Some(literal(ctx, konst)? as i64)
}

/// `true` if `v` is `idx + 1` (either operand order).
fn is_increment(ctx: &Context, v: ValueId, idx: ValueId) -> bool {
    use qcode::value::insn::{Binary, Binop, IntBinop};
    let ValueId::Instruction(id) = v else {
        return false;
    };
    let Mnemonic::Binop(Binary { lhs, rhs, op }) = ctx.get_insn(id).mnemonic() else {
        return false;
    };
    let lit = |x: ValueId| matches!(qcode::value::ValueRef::new(x, ctx), ValueRef::Literal(l) if l.value() == 1);
    matches!(op, Binop::Int(IntBinop::Add))
        && ((*lhs == idx && lit(*rhs)) || (*rhs == idx && lit(*lhs)))
}

/// The arguments `from` passes to `to` on their connecting edge (the first edge
/// to `to`, so it is unambiguous only when `from` has a single edge to `to` —
/// which is the case for every loop edge here).
fn edge_args(ctx: &Context, from: BlockId, to: BlockId) -> Vec<ValueId> {
    let Some(term) = BasicBlock::from_id(ctx, from).iter().last() else {
        return Vec::new();
    };
    match term.mnemonic() {
        Mnemonic::Branch(b) if b.target == to => b.args.clone(),
        Mnemonic::CBranch(cb) => {
            if cb.success_block == to {
                cb.success_args.clone()
            } else if cb.failure_block == to {
                cb.failure_args.clone()
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

/// Values feeding block-param `p` of `block` from every predecessor edge.
fn param_incomings(ctx: &Context, block: BlockId, p: ValueId) -> Vec<ValueId> {
    let Some(k) = BasicBlock::from_id(ctx, block)
        .params()
        .position(|q| q.id() == p)
    else {
        return Vec::new();
    };
    header_incoming(ctx, block, k)
}

/// Trip bound `N` for a rotated (do-while) loop whose guard compares the
/// *incremented* index `inc` (`= index + 1`) to a constant `N`, so the last body
/// index is `N-1`. Accepts only the canonical do-while polarities: `inc < N`
/// (unsigned/signed) continuing on true, or `inc == N` exiting on true. Keying on
/// `inc` (never the pre-increment `index`) keeps the meaning unambiguous: in a
/// do-while the body has already run at `index` before the guard, so a guard on
/// `index` would tile a different range.
fn rotated_bound(ctx: &Context, cond: ValueId, inc: ValueId, exit_on_true: bool) -> Option<i64> {
    use qcode::value::insn::{Binary, Binop, IntBinop};
    let ValueId::Instruction(id) = cond else {
        return None;
    };
    let Mnemonic::Binop(Binary { lhs, rhs, op }) = ctx.get_insn(id).mnemonic() else {
        return None;
    };
    let konst = if *lhs == inc {
        *rhs
    } else if *rhs == inc {
        *lhs
    } else {
        return None;
    };
    let ok = match op {
        Binop::Int(IntBinop::Equal) => exit_on_true,
        Binop::Int(IntBinop::Less | IntBinop::SLess) => !exit_on_true && *lhs == inc,
        _ => false,
    };
    ok.then(|| literal(ctx, konst).map(|k| k as i64)).flatten()
}

/// Union-find over all block params ↔ their incoming values, resolved into a pure
/// `value → root-pointer-param` map. A value maps to root `R` iff its pass-through
/// phi class contains exactly one root param `R` (ambiguous classes are omitted).
fn base_roots(ctx: &mut Context, fid: FunctionId) -> HashMap<ValueId, ValueId> {
    let mut uf = Uf::default();
    // Every value the union-find touches. A node that ends up as its tree's
    // representative is only ever stored as a parent *value*, never a key, so
    // `uf.parent.keys()` alone would miss it — track membership explicitly.
    let mut nodes: HashSet<ValueId> = HashSet::default();
    let blocks: Vec<BlockId> = Function::from_id(ctx, fid).iter().map(|b| b.id).collect();
    for &bid in &blocks {
        let params: Vec<ValueId> = BasicBlock::from_id(ctx, bid)
            .params()
            .map(|p| p.id())
            .collect();
        for (k, p) in params.into_iter().enumerate() {
            for v in header_incoming(ctx, bid, k) {
                uf.union(p, v);
                nodes.insert(p);
                nodes.insert(v);
            }
        }
    }
    let roots: Vec<ValueId> = Function::from_id(ctx, fid)
        .root()
        .map(|r| r.params().map(|p| p.id()).collect())
        .unwrap_or_default();
    for &r in &roots {
        nodes.insert(r);
    }
    // Representative → unique root (or None if two roots collide in one class).
    let mut rep_root: HashMap<ValueId, Option<ValueId>> = HashMap::default();
    for &r in &roots {
        let rep = uf.find(r);
        rep_root
            .entry(rep)
            .and_modify(|e| *e = None)
            .or_insert(Some(r));
    }
    // Resolve every value seen by the union-find to its class root.
    let mut val_root: HashMap<ValueId, ValueId> = HashMap::default();
    for k in nodes {
        let rep = uf.find(k);
        if let Some(Some(r)) = rep_root.get(&rep) {
            val_root.insert(k, *r);
        }
    }
    val_root
}

/// Recognize the in-place array-fill loop in `fid`.
fn try_match(ctx: &mut Context, fid: FunctionId) -> Option<PromoteMatch> {
    // Reject anything with a call: another routine could observe/mutate the region.
    for block in Function::from_id(ctx, fid).iter() {
        for insn in block.iter() {
            if matches!(
                insn.mnemonic(),
                Mnemonic::Call(_) | Mnemonic::CallInd(_) | Mnemonic::BranchInd(_)
            ) {
                return None;
            }
        }
    }

    // Collect every RAM access, recording the space each lives in. Accesses in
    // different spaces never alias, so the region's space (fixed below by the lane
    // store) partitions these: same-space accesses touch the region, others are
    // trivially disjoint.
    struct Acc {
        id: InstructionId,
        block: BlockId,
        ptr: ValueId,
        size: usize,
        space: SpaceId,
        stored: Option<ValueId>,
    }
    let is_ram = |ctx: &Context, sp: SpaceId| matches!(Space::from_id(ctx, sp).ty, SpaceType::Ram);
    let mut accesses: Vec<Acc> = Vec::new();
    for block in Function::from_id(ctx, fid).iter() {
        let bid = block.id;
        for insn in block.iter() {
            match insn.mnemonic() {
                Mnemonic::Load(l) if is_ram(ctx, l.space) => accesses.push(Acc {
                    id: insn.id,
                    block: bid,
                    ptr: l.ptr,
                    size: l.size,
                    space: l.space,
                    stored: None,
                }),
                Mnemonic::Store(s) if is_ram(ctx, s.space) => accesses.push(Acc {
                    id: insn.id,
                    block: bid,
                    ptr: s.ptr,
                    size: s.size,
                    space: s.space,
                    stored: Some(s.src),
                }),
                _ => {}
            }
        }
    }

    let numbering = precompute_forms(ctx, fid);
    let val_root = base_roots(ctx, fid);
    let root_of = |v: ValueId| -> Option<ValueId> { val_root.get(&v).copied() };

    // The single strided lane store establishes (base_root, elem_size, index) and
    // the region's space.
    struct Lane {
        store_id: InstructionId,
        base_root: ValueId,
        index: ValueId,
        stored_val: ValueId,
        c_lane: i64,
        body: BlockId,
        esz: usize,
        space: SpaceId,
    }
    let mut lane: Option<Lane> = None;
    for a in &accesses {
        let Some(src) = a.stored else { continue };
        let Some((base, idx, c)) = affine_strided_lane(&numbering, a.ptr, a.size) else {
            continue;
        };
        if !matches!(idx, ValueId::BlockParam(_)) {
            continue;
        }
        let Some(r) = root_of(base) else { continue };
        if lane.is_some() {
            return None; // more than one strided store — not the canonical shape
        }
        lane = Some(Lane {
            store_id: a.id,
            base_root: r,
            index: idx,
            stored_val: src,
            c_lane: c,
            body: a.block,
            esz: a.size,
            space: a.space,
        });
    }
    let Lane {
        store_id: lane_store_id,
        base_root,
        index,
        stored_val,
        c_lane,
        body,
        esz,
        space: region_space,
    } = lane?;
    if esz == 0 || c_lane % esz as i64 != 0 {
        return None;
    }
    // v1 works in bytes (offset % esz, count * esz, per-lane addresses). A
    // word-addressed region would miscount lanes, so require a byte-addressed space.
    if Space::from_id(ctx, region_space).word_size != 1 {
        return None;
    }
    // Only accesses in the region's space can touch it; others are a different
    // memory and never alias, even when the pointer arithmetic coincides.
    let in_region = |a: &Acc| a.space == region_space;
    let store_word = c_lane / esz as i64;
    let is_r = |v: ValueId| root_of(v) == Some(base_root);

    // Pre-loop const-offset region stores. Their word offset defines the region
    // origin — element 0 lives at the lowest such word, so a nonzero pointer bias
    // in the lifted addresses (e.g. `mt[0]` at `base + 4`) is normalized away by
    // measuring every other access relative to it. v1 supports only element-0
    // pre-loop stores; when several write element 0, the last in program order is
    // the seed and the earlier ones are dead (overwritten before any read).
    let mut seed_candidates: Vec<(usize, InstructionId, BlockId, ValueId, i64)> = Vec::new();
    for (pos, a) in accesses.iter().enumerate() {
        if a.id == lane_store_id || a.size != esz || !in_region(a) {
            continue;
        }
        let Some(src) = a.stored else { continue };
        let Some((_, c_seed)) = affine_base_const(&numbering, a.ptr, &is_r) else {
            continue;
        };
        if c_seed % esz as i64 != 0 {
            return None;
        }
        seed_candidates.push((pos, a.id, a.block, src, c_seed / esz as i64));
    }
    if seed_candidates.is_empty() {
        return None;
    }
    let origin_word = seed_candidates.iter().map(|c| c.4).min().unwrap();
    // v1: every pre-loop region store targets element 0, all in one block.
    let seed_block = seed_candidates[0].2;
    if seed_candidates
        .iter()
        .any(|c| c.4 != origin_word || c.2 != seed_block)
    {
        return None;
    }
    seed_candidates.sort_by_key(|c| c.0);
    let (_, seed_id, _, seed_val, _) = *seed_candidates.last().unwrap();
    let dead_stores: Vec<InstructionId> = seed_candidates[..seed_candidates.len() - 1]
        .iter()
        .map(|c| c.1)
        .collect();
    // Element positions relative to the origin.
    let store_delta = store_word - origin_word;

    // Optional memory-carried lane load: the strided read of a previous element.
    let mut load: Option<(InstructionId, i64)> = None;
    for a in &accesses {
        if a.id == lane_store_id || a.stored.is_some() || a.size != esz || !in_region(a) {
            continue;
        }
        let Some((base, idx, c)) = affine_strided_lane(&numbering, a.ptr, a.size) else {
            continue;
        };
        if idx != index || root_of(base) != Some(base_root) || c % esz as i64 != 0 {
            continue;
        }
        if load.is_some() {
            return None;
        }
        load = Some((a.id, c / esz as i64 - origin_word));
    }

    // --- Loop structure ---
    // `index` is the body's induction parameter.
    let ValueId::BlockParam(ipid) = index else {
        return None;
    };
    if ctx.values.block_params[ipid].parent != Some(body) {
        return None;
    }
    // The body's terminator tells the two shapes apart: a `goto header` is the
    // split shape (a distinct header holds the guard); a `cbranch` with an edge
    // back to the body itself is the rotated do-while shape (`header == body`).
    let body_term = BasicBlock::from_id(ctx, body).iter().last()?;
    let (rotated, header, exit, cond, exit_on_true) = match body_term.mnemonic() {
        Mnemonic::Branch(b) => {
            let header = b.target;
            let hterm = BasicBlock::from_id(ctx, header).iter().last()?;
            let Mnemonic::CBranch(cb) = hterm.mnemonic() else {
                return None;
            };
            let (sb, fb, cond) = (cb.success_block, cb.failure_block, cb.condition);
            let exit = if sb == body {
                fb
            } else if fb == body {
                sb
            } else {
                return None;
            };
            (false, header, exit, cond, exit == sb)
        }
        Mnemonic::CBranch(cb) => {
            let (sb, fb, cond) = (cb.success_block, cb.failure_block, cb.condition);
            let exit = if sb == body {
                fb
            } else if fb == body {
                sb
            } else {
                return None;
            };
            (true, body, exit, cond, exit == sb)
        }
        _ => return None,
    };
    if exit == body || exit == header {
        return None;
    }
    // header preds = {preheader, body(back-edge)}.
    let header_preds: Vec<BlockId> = BasicBlock::from_id(ctx, header)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    if !header_preds.contains(&body) {
        return None;
    }
    let mut ph_iter = header_preds.iter().copied().filter(|&p| p != body);
    let preheader = ph_iter.next()?;
    if ph_iter.next().is_some() {
        return None; // more than one entry into the loop
    }
    let exit_preds: Vec<BlockId> = BasicBlock::from_id(ctx, exit)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    if exit_preds != [header] {
        return None;
    }
    let ph_term = BasicBlock::from_id(ctx, preheader).iter().last()?;
    if !matches!(ph_term.mnemonic(), Mnemonic::Branch(b) if b.target == header) {
        return None;
    }
    // The pre-loop seed stores must sit in the preheader (they run once, before
    // the loop, dominating every lane).
    if seed_block != preheader {
        return None;
    }

    // Induction: the values feeding `index` (directly, in the rotated shape, or
    // via the header param it copies, in the split shape) start at a single
    // literal `s` and step by one.
    let feeds = if rotated {
        param_incomings(ctx, body, index)
    } else {
        let kj = BasicBlock::from_id(ctx, body)
            .params()
            .position(|p| p.id() == index)?;
        let hi = edge_args(ctx, header, body).get(kj).copied()?;
        let ValueId::BlockParam(hpid) = hi else {
            return None;
        };
        if ctx.values.block_params[hpid].parent != Some(header) {
            return None;
        }
        param_incomings(ctx, header, hi)
    };
    let inc_val = feeds
        .iter()
        .copied()
        .find(|&v| is_increment(ctx, v, index))?;
    let inits: HashSet<u64> = feeds
        .iter()
        .filter(|&&v| !is_increment(ctx, v, index))
        .filter_map(|&v| literal(ctx, v))
        .collect();
    if inits.len() != 1 {
        return None;
    }
    let s = *inits.iter().next().unwrap() as i64;

    // Trip bound `N`; the last body index is `N-1`.
    let n = if rotated {
        rotated_bound(ctx, cond, inc_val, exit_on_true)?
    } else {
        // Split shape: the guard compares the header param feeding `index`.
        let kj = BasicBlock::from_id(ctx, body)
            .params()
            .position(|p| p.id() == index)?;
        let hi = edge_args(ctx, header, body).get(kj).copied()?;
        guard_bound(ctx, cond, hi, exit_on_true)?
    };

    // Coverage: element 0 is the seed, lane stores tile `[1, count-1]`.
    let store_lo = s + store_delta;
    let store_hi = (n - 1) + store_delta;
    if store_lo != 1 || store_hi < store_lo {
        return None;
    }
    let count = (store_hi + 1) as usize;
    if count == 0 || count.saturating_mul(esz) > (1 << 20) {
        return None;
    }

    // Carry: either the memory lane load, or a register accumulator param whose
    // back-edge value is `stored_val` (`@acc' = %next`) and whose preheader init
    // is the seed. Both become `at(arr, index-1)`.
    let (carry_replace, carry_remove, load_delta) = match load {
        Some((load_id, ld)) => {
            // A valid memory carry reads a strictly-earlier, already-written lane.
            if ld >= store_delta || s + ld < 0 {
                return None;
            }
            (ValueId::Instruction(load_id), Some(load_id), ld)
        }
        None => {
            // Register-carried: v1 supports this only for the rotated shape, whose
            // single-block back-edge makes the accumulator unambiguous.
            if !rotated {
                return None;
            }
            let back = edge_args(ctx, body, header);
            let init = edge_args(ctx, preheader, header);
            let params: Vec<ValueId> = BasicBlock::from_id(ctx, body)
                .params()
                .map(|p| p.id())
                .collect();
            let mut acc = None;
            for (k, p) in params.iter().enumerate() {
                if *p == index {
                    continue;
                }
                if back.get(k) == Some(&stored_val) && init.get(k) == Some(&seed_val) {
                    if acc.is_some() {
                        return None;
                    }
                    acc = Some(*p);
                }
            }
            (acc?, None, store_delta - 1)
        }
    };

    // Additional const-offset region loads (e.g. the exit read of element 0 the
    // function returns). Only const loads in the exit block are supported — there
    // the whole array is available, so they forward to `at(arr_exit, element)`.
    let mut extra_loads: Vec<(InstructionId, i64)> = Vec::new();
    for a in &accesses {
        if a.stored.is_some() || a.id == lane_store_id || Some(a.id) == carry_remove || !in_region(a)
        {
            continue;
        }
        let Some((_, c)) = affine_base_const(&numbering, a.ptr, &is_r) else {
            continue;
        };
        if c % esz as i64 != 0 {
            return None;
        }
        let elem = c / esz as i64 - origin_word;
        if a.block != exit || elem < 0 || elem as usize >= count {
            return None;
        }
        extra_loads.push((a.id, elem));
    }

    // Every remaining RAM access reaching the region must be accounted for.
    let mut allowed: HashSet<InstructionId> = [seed_id, lane_store_id].into_iter().collect();
    if let Some(load_id) = carry_remove {
        allowed.insert(load_id);
    }
    allowed.extend(extra_loads.iter().map(|&(id, _)| id));
    allowed.extend(dead_stores.iter().copied());
    for a in &accesses {
        if allowed.contains(&a.id) {
            continue;
        }
        // An access in a different space is a different memory — trivially disjoint
        // from the region regardless of pointer arithmetic. Only same-space
        // accesses that alias the region need accounting.
        if !in_region(a) {
            continue;
        }
        let strided_r = affine_strided_lane(&numbering, a.ptr, a.size)
            .and_then(|(b, _, _)| root_of(b))
            == Some(base_root);
        let const_r = affine_base_const(&numbering, a.ptr, &is_r).is_some();
        if strided_r || const_r {
            return None; // an unmodelled access to the region
        }
    }

    Some(PromoteMatch {
        preheader,
        header,
        body,
        exit,
        rotated,
        base_root,
        origin_word,
        index,
        elem_size: esz,
        count,
        region_space,
        seed_id,
        seed_val,
        lane_store_id,
        stored_val,
        store_delta,
        carry_replace,
        carry_remove,
        load_delta,
        extra_loads,
        dead_stores,
    })
}

/// Append `arg` to the branch terminator of `from` on the edge to `to`.
fn append_edge_arg(ctx: &mut Context, from: BlockId, to: BlockId, arg: ValueId) {
    let Some(term) = BasicBlock::from_id(ctx, from).iter().last() else {
        return;
    };
    let term_id = term.id;
    let mut m = term.mnemonic().clone();
    match &mut m {
        Mnemonic::Branch(Branch { target, args }) if *target == to => args.push(arg),
        Mnemonic::CBranch(CBranch {
            success_block,
            success_args,
            failure_block,
            failure_args,
            ..
        }) => {
            if *success_block == to {
                success_args.push(arg);
            }
            if *failure_block == to {
                failure_args.push(arg);
            }
        }
        _ => return,
    }
    ctx.replace_instruction_mnemonic(term_id, m);
}

/// Build `index + delta` (as `index`, `index - 1`, or `index + c`) at `index`'s
/// own width, so the arithmetic wraps exactly as the lifted address did. The
/// `index - 1` form is emitted verbatim as a `sub` so `loop_to_scan`'s
/// `is_decrement` recognizes it.
fn index_plus(b: &mut Builder, index: ValueId, delta: i64) -> ValueId {
    if delta == 0 {
        return index;
    }
    let ty = b.context_mut().type_of(index);
    let width = b.context_mut().types.size_of(ty);
    if delta == -1 {
        let one = b.context_mut().get_const(1, width).id();
        return b.push_sub(index, one).id();
    }
    let mask = if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    };
    let c = b.context_mut().get_const(delta as u64 & mask, width).id();
    b.push_add(index, c).id()
}

fn apply(ctx: &mut Context, m: &PromoteMatch) -> bool {
    let esz = m.elem_size;
    let elem_ty = ctx.types.get_or_make_int(esz);
    let arr_ty = ctx.types.get_or_make_array(elem_ty, m.count);
    let arr_sz = m.count * esz;

    let insert_id =
        qcode::value::insn::IntrinsicId::from_name("insert").expect("insert registered");
    let at_id = qcode::value::insn::IntrinsicId::from_name("at").expect("at registered");

    // New carried-array params (pushed last on each block so appended edge args
    // line up positionally). In the rotated shape the header *is* the body, so
    // they share one param.
    let new_param = |ctx: &mut Context, bid: BlockId| {
        let pid = BasicBlock::from_id_mut(ctx, bid).push_param(arr_sz).id;
        ctx.values.block_params[pid].type_id = arr_ty;
        ValueId::BlockParam(pid)
    };
    let arr_h = new_param(ctx, m.header);
    let arr_b = if m.rotated {
        arr_h
    } else {
        new_param(ctx, m.body)
    };
    let arr_e = new_param(ctx, m.exit);

    // Zero-initialized array literal, retyped to [elem;count].
    let zero = {
        let id = ctx.get_bytes(vec![0u8; arr_sz]).id();
        if let ValueId::Bytes(bid) = id {
            ctx.values.bytes[bid].type_id = arr_ty;
        }
        id
    };

    // Preheader: %a0 = insert(zero, 0, seed_val), before the branch.
    let a0 = {
        let term_id = BasicBlock::from_id(ctx, m.preheader)
            .iter()
            .last()
            .unwrap()
            .id;
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.preheader));
        b.set_insert_point_before(term_id);
        let idx0 = b.context_mut().get_const(0, 8).id();
        b.push_intrinsic(insert_id, vec![zero, idx0, m.seed_val])
            .id()
    };

    // Body: at(arr_b, index + load_delta) becomes the carry. For a memory carry it
    // replaces the lane load (inserted just before it); for a register carry it
    // replaces the accumulator param's uses (inserted at the body top, before the
    // first instruction that reads the accumulator).
    let anchor = match m.carry_remove {
        Some(load_id) => Some(load_id),
        None => BasicBlock::from_id(ctx, m.body).iter().next().map(|i| i.id),
    };
    let at_val = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.body));
        if let Some(anchor) = anchor {
            b.set_insert_point_before(anchor);
        }
        let idx = index_plus(&mut b, m.index, m.load_delta);
        b.push_intrinsic(at_id, vec![arr_b, idx]).id()
    };
    ctx.replace_all_uses_with(m.carry_replace, at_val);

    // Body: %arr' = insert(arr_b, index + store_delta, stored_val), before branch.
    let arr_next = {
        let term_id = BasicBlock::from_id(ctx, m.body).iter().last().unwrap().id;
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.body));
        b.set_insert_point_before(term_id);
        let idx = index_plus(&mut b, m.index, m.store_delta);
        b.push_intrinsic(insert_id, vec![arr_b, idx, m.stored_val])
            .id()
    };

    // Exit: rewrite the extra const-offset region loads to `at(arr_e, element)`.
    for &(load_id, elem) in &m.extra_loads {
        let at_val = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.exit));
            b.set_insert_point_before(load_id);
            let idx = b.context_mut().get_const(elem as u64, 8).id();
            b.push_intrinsic(at_id, vec![arr_e, idx]).id()
        };
        ctx.replace_all_uses_with(ValueId::Instruction(load_id), at_val);
        ctx.remove_instruction(load_id);
    }

    // Exit: store(ram, base(+origin) <- arr_e) before the exit terminator.
    {
        let term_id = BasicBlock::from_id(ctx, m.exit).iter().last().unwrap().id;
        let ram = m.region_space;
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.exit));
        b.set_insert_point_before(term_id);
        let dst = if m.origin_word == 0 {
            m.base_root
        } else {
            let ty = b.context_mut().type_of(m.base_root);
            let width = b.context_mut().types.size_of(ty);
            let off = b
                .context_mut()
                .get_const((m.origin_word * m.elem_size as i64) as u64, width)
                .id();
            b.push_add(m.base_root, off).id()
        };
        b.push_store(arr_e, dst, ram);
    }

    // Drop the now-dead memory traffic.
    if let Some(load_id) = m.carry_remove {
        ctx.remove_instruction(load_id);
    }
    for &dead in &m.dead_stores {
        ctx.remove_instruction(dead);
    }
    ctx.remove_instruction(m.lane_store_id);
    ctx.remove_instruction(m.seed_id);

    // Thread the array through every loop edge. The header always passes its own
    // array param (`arr_h`) to the exit and (in the split shape) to the body; in
    // the rotated shape header==body, so the back-edge is the body's self-edge and
    // there is no separate header→body edge.
    append_edge_arg(ctx, m.preheader, m.header, a0);
    append_edge_arg(ctx, m.body, m.header, arr_next);
    if !m.rotated {
        append_edge_arg(ctx, m.header, m.body, arr_h);
    }
    append_edge_arg(ctx, m.header, m.exit, arr_h);

    true
}

impl FunctionPass for ArrayPromote {
    const NAME: &'static str = "array_promote";

    fn description(&self) -> &'static str {
        "Promote in-place strided RAM array-fill loops to a value-carried array (insert/at)"
    }

    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        match try_match(ctx, fun_id) {
            Some(m) => Ok(apply(ctx, &m)),
            None => Ok(false),
        }
    }
}

crate::register_function_pass!(ArrayPromote);

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use crate::test_util::run_function_pass;

    // The honest MT19937 seeding loop: in-place strided RAM fill with a
    // memory-carried recurrence (reload `mt[j-1]`). `qcode!` requires a string
    // literal at expansion time, so the source is inlined in each test.

    #[test]
    fn mt_init_is_promoted() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn mt_init:
            <entry @seed:i64 @base:i64>
                %e0 = trunc(i32, @seed);
                store(ram:4, @base <- %e0);
                goto <head @i=1 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 624;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %jm1 = @j - 1;
                %roff = %jm1 * 4;
                %raddr = @b + %roff;
                %prev = load(ram:4, %raddr);
                %sh = %prev >> 30;
                %xo = %prev ^ %sh;
                %ml = %xo * 1812433253;
                %jt = trunc(i32, @j);
                %next = %ml + %jt;
                %woff = @j * 4;
                %waddr = @b + %woff;
                store(ram:4, %waddr <- %next);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        let changed = run_function_pass::<ArrayPromote>(&mut ctx, mt_init).unwrap();
        assert!(changed, "array_promote should recognize the mt_init loop");
        let ir = format!("{}", Function::from_id(&ctx, mt_init));
        // The memory-carried reload became `at`, the lane store became `insert`,
        // and the region is stored back once at the exit.
        assert!(ir.contains("$at("), "lane load should become at(): {ir}");
        assert!(
            ir.contains("$insert("),
            "lane store should become insert(): {ir}"
        );
        assert!(
            ir.contains("store(ram:2496"),
            "the whole [i32;624] region should be stored once at exit: {ir}"
        );
        // The per-lane RAM traffic is gone.
        assert!(
            !ir.contains("load(ram:4"),
            "lane load should be removed: {ir}"
        );
        assert!(
            !ir.contains("store(ram:4"),
            "seed + lane stores should be removed: {ir}"
        );
    }

    #[test]
    fn idempotent_second_run_is_noop() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn mt_init:
            <entry @seed:i64 @base:i64>
                %e0 = trunc(i32, @seed);
                store(ram:4, @base <- %e0);
                goto <head @i=1 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 624;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %jm1 = @j - 1;
                %roff = %jm1 * 4;
                %raddr = @b + %roff;
                %prev = load(ram:4, %raddr);
                %sh = %prev >> 30;
                %xo = %prev ^ %sh;
                %ml = %xo * 1812433253;
                %jt = trunc(i32, @j);
                %next = %ml + %jt;
                %woff = @j * 4;
                %waddr = @b + %woff;
                store(ram:4, %waddr <- %next);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        assert!(run_function_pass::<ArrayPromote>(&mut ctx, mt_init).unwrap());
        // After promotion there is no strided RAM lane store left to match.
        assert!(
            !run_function_pass::<ArrayPromote>(&mut ctx, mt_init).unwrap(),
            "second run should find nothing to promote"
        );
    }

    // Soundness guard for the iota-based scan: a lane may only be forwarded to the
    // carried array when it was *written earlier this trip*. If the loop reads a
    // lane's original (pre-existing) memory value, zero-seeding the promoted array
    // would change the result, so the pass must refuse — the scan cannot be
    // expressed over a bare `iota`, it would need the original array.

    #[test]
    fn reads_current_lane_original_value_is_rejected() {
        // Prefix sum in place: mt[i] = mt[i] + mt[i-1]. Lane `i` is loaded (its
        // original value) before it is stored, so the fill is not self-contained.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn reads_orig:
            <entry @seed:i64 @base:i64>
                %e0 = trunc(i32, @seed);
                store(ram:4, @base <- %e0);
                goto <head @i=1 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 624;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %jm1 = @j - 1;
                %roff = %jm1 * 4;
                %raddr = @b + %roff;
                %prev = load(ram:4, %raddr);
                %coff = @j * 4;
                %caddr = @b + %coff;
                %cur = load(ram:4, %caddr);
                %next = %cur + %prev;
                store(ram:4, %caddr <- %next);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        assert!(
            !run_function_pass::<ArrayPromote>(&mut ctx, reads_orig).unwrap(),
            "reading a lane's original value before writing it must block promotion"
        );
    }

    #[test]
    fn index_map_over_original_array_is_rejected() {
        // Indexed map over the original data: mt[i] = mt[i] * 3 + i. Reads the
        // original lane `i` and the index `i` — the `enumerate(l)` shape. There is
        // no memory-carried recurrence and no seed store; the functional form would
        // read the original array, so the zero-seeding pass must refuse.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn reads_enum:
            <entry @base:i64>
                goto <head @i=0 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 624;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %off = @j * 4;
                %addr = @b + %off;
                %cur = load(ram:4, %addr);
                %m = %cur * 3;
                %jt = trunc(i32, @j);
                %next = %m + %jt;
                store(ram:4, %addr <- %next);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        assert!(
            !run_function_pass::<ArrayPromote>(&mut ctx, reads_enum).unwrap(),
            "an indexed map reading the original array must block promotion"
        );
    }
}
