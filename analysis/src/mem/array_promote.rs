//! Array promotion: "mem2reg for arrays".
//!
//! Recognizes an in-place, strided array-fill loop — a loop that writes a memory
//! region through an affine, loop-invariant-based address `base + i*esz (+ c)` as
//! its induction variable `i` sweeps `[0, N)` — and rewrites the region's memory
//! traffic into **one** value-level carried array `arr:[elem;N]` threaded through
//! the loop with the `insert` / `at` intrinsics, plus a single wide
//! `store(base <- arr)` at the exit.
//!
//! The model is uniform: every region **load** at lane `index + od` becomes
//! `at(arr, index + od)` and the lane **store** becomes `insert(arr, index + od)`.
//! A read of an already-written earlier lane (`od < store_delta`) sees the fresh
//! carry; a read of the not-yet-written own lane (`od >= store_delta`) sees the
//! original value. The carried array is initialized to the region's original
//! contents (`arr0 = load(base, N*esz)`) when any such original read exists, else
//! to `splat(0, N)` — so pure generation introduces no new observable read.
//!
//! Loop structure and pass-through pointer identity come from [`crate::loop_info`].
//!
//! Soundness gate (aliasing): the base is a loop-invariant root pointer; the
//! function contains no calls; the written lanes tile the region `[0, N)` exactly;
//! and every region access is accounted for (seed, lane store, region loads, exit
//! reads) — any unmodelled region access declines the promotion.

use rustc_hash::FxHashSet as HashSet;

use qcode::{
    builder::Builder,
    space::{Space, SpaceId, SpaceType},
    types::TypeId,
    value::{
        BlockId, BlockRef, FunctionId, FunctionRef, ValueId,
        insn::{Branch, CBranch, InstructionId, IntrinsicApp, IntrinsicId, Load, Mnemonic},
        util::{
            base_ref::{BaseRef, HostRef},
            host_mut::HostMut,
        },
    },
};

use crate::gvn::affine::precompute_forms;
use crate::sequence::{affine_base_const, affine_strided_lane};

#[derive(Default)]
pub struct ArrayPromote;

/// A recognized in-place array-fill loop over a single carried array.
struct PromoteMatch {
    preheader: BlockId,
    header: BlockId,
    body: BlockId,
    exit: BlockId,
    /// `header == body` (single-block do-while loop).
    rotated: bool,
    /// Loop-invariant root pointer the region is based at.
    base_root: ValueId,
    /// Element 0 sits at `base_root + origin_word * elem_size`.
    origin_word: i64,
    /// Body induction parameter (the lane index in the store/load addresses).
    index: ValueId,
    elem_size: usize,
    count: usize,
    /// Pre-loop element-0 seed store; `None` for a seedless indexed fill.
    seed: Option<Seed>,
    /// The RAM/Temporary space the region lives in (word size 1).
    region_space: SpaceId,
    /// The one strided lane store, writing element `index + store_delta`.
    lane_store_id: InstructionId,
    stored_val: ValueId,
    store_delta: i64,
    /// Every strided region load at lane `index + od`, rewritten to `at(arr, index+od)`.
    region_reads: Vec<(InstructionId, i64)>,
    /// Set when any read touches a not-yet-written lane; then `arr` is initialized
    /// from the region's original contents rather than `splat(0)`.
    reads_original: bool,
    /// Const-offset region loads in the exit block, rewritten to `at(arr_exit, element)`.
    extra_loads: Vec<(InstructionId, i64)>,
}

/// A pre-loop element-0 seed `l[0] = val`, folded into the array as `insert(_, 0, val)`.
struct Seed {
    id: InstructionId,
    val: ValueId,
}

/// Recognize the in-place array-fill loop in `fid`.
fn try_match(host: HostRef, fid: FunctionId) -> Option<PromoteMatch> {
    // Reject anything with a call: another routine could observe/mutate the region.
    for block in FunctionRef::new(host, fid).iter() {
        for insn in block.iter() {
            if matches!(
                insn.mnemonic(),
                Mnemonic::Call(_) | Mnemonic::CallInd(_) | Mnemonic::BranchInd(_)
            ) {
                return None;
            }
        }
    }

    // A promotable region lives in real RAM or in argpromote's functionalized
    // shadow (an unnamed `Temporary` space); both are byte-addressed memory.
    struct Acc {
        id: InstructionId,
        block: BlockId,
        ptr: ValueId,
        size: usize,
        space: SpaceId,
        stored: Option<ValueId>,
    }
    let is_ram = |sp: SpaceId| {
        matches!(
            Space::from_id(host.shared(), sp).ty,
            SpaceType::Ram | SpaceType::Temporary
        )
    };
    let mut accesses: Vec<Acc> = Vec::new();
    for block in FunctionRef::new(host, fid).iter() {
        let bid = block.id;
        for insn in block.iter() {
            match insn.mnemonic() {
                Mnemonic::Load(l) if is_ram(l.space) => accesses.push(Acc {
                    id: insn.id,
                    block: bid,
                    ptr: l.ptr,
                    size: l.size,
                    space: l.space,
                    stored: None,
                }),
                Mnemonic::Store(s) if is_ram(s.space) => accesses.push(Acc {
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

    let loops = crate::loop_info::recognize_loops(host, fid);
    let numbering = precompute_forms(host, fid);
    let val_root = crate::loop_info::value_roots(host, fid);
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
        let Some((base, idx, c)) =
            affine_strided_lane(&numbering, a.ptr, a.size, |v| root_of(v).is_some())
        else {
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
    // v1 works in bytes, so require a byte-addressed region.
    if Space::from_id(host.shared(), region_space).word_size != 1 {
        return None;
    }
    let in_region = |a: &Acc| a.space == region_space;
    let store_word = c_lane / esz as i64;
    let is_r = |v: ValueId| root_of(v) == Some(base_root);

    // Pre-loop const-offset region store — the seed. At most one is allowed; a
    // duplicate should have been removed by dead-store/dce before this pass.
    let mut seed: Option<(InstructionId, BlockId, ValueId, i64)> = None; // (id, block, val, word)
    for a in &accesses {
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
        if seed.is_some() {
            return None; // two seed candidates — not the canonical shape
        }
        seed = Some((a.id, a.block, src, c_seed / esz as i64));
    }

    // --- Loop structure (from the shared recognizer) ---
    let lp = loops.iter().find(|l| l.body == body)?;
    let (preheader, header, exit, rotated) = (lp.preheader, lp.header, lp.exit, lp.rotated);
    let ind = lp.unit_induction(host, index)?;
    let s = ind.start;
    let n = ind.count;
    // The seed store runs once, before the loop.
    if let Some((_, seed_block, ..)) = &seed
        && *seed_block != preheader
    {
        return None;
    }
    let seeded = seed.is_some();

    // Fix the region origin (element 0's word offset) and hence `store_delta`.
    let (origin_word, store_delta) = match &seed {
        Some((_, _, _, seed_word)) => (*seed_word, store_word - *seed_word),
        None => (s + store_word, -s),
    };
    let store_lo = s + store_delta;
    let store_hi = (n - 1) + store_delta;
    let expected_lo = if seeded { 1 } else { 0 };
    if store_lo != expected_lo || store_hi < store_lo {
        return None;
    }
    let count = (store_hi + 1) as usize;
    if count == 0 || count.saturating_mul(esz) > (1 << 20) {
        return None;
    }

    // Body instruction order, for the program-order check below.
    let body_order: Vec<InstructionId> = BlockRef::new(host, body).iter().map(|i| i.id).collect();
    let store_pos = body_order.iter().position(|&id| id == lane_store_id)?;

    // Strided region loads at `index + od`: one uniform list, each rewritten to
    // `at(arr, index+od)`. `od < store_delta` reads an already-written earlier lane
    // (the carry); `od >= store_delta` reads a not-yet-written lane (the original).
    let mut region_reads: Vec<(InstructionId, i64)> = Vec::new();
    let mut reads_original = false;
    for a in &accesses {
        if a.id == lane_store_id || a.stored.is_some() || a.size != esz || !in_region(a) {
            continue;
        }
        let Some((base, idx, c)) =
            affine_strided_lane(&numbering, a.ptr, a.size, |v| root_of(v).is_some())
        else {
            continue;
        };
        if idx != index || root_of(base) != Some(base_root) || c % esz as i64 != 0 {
            continue;
        }
        let od = c / esz as i64 - origin_word;
        // The lane `index+od` must stay in `[0, count)` across all iterations.
        let lane_lo = s + od;
        let lane_hi = (n - 1) + od;
        if lane_lo < 0 || lane_hi >= count as i64 {
            return None;
        }
        if od >= store_delta {
            // An original read: correct only if the load precedes the lane store
            // this trip. (Given coverage, forward reads `od > store_delta` fail the
            // bounds check above, so in practice only `od == store_delta` reaches
            // here — but the uniform rule stays correct if coverage ever loosens.)
            reads_original = true;
            let pos = body_order.iter().position(|&id| id == a.id)?;
            if pos >= store_pos {
                return None;
            }
        }
        region_reads.push((a.id, od));
    }

    // Const-offset, element-width region loads in the exit block, forwarded to
    // `at(arr_exit, elem)`. Wider loads (e.g. a whole-region `load(base)`) are *not*
    // matched here — they must not be narrowed to a 1-byte `at`; they fall through to
    // the accounting loop, which leaves a whole-region exit load in place.
    let region_bytes = count * esz;
    let read_ids: HashSet<InstructionId> = region_reads.iter().map(|&(id, _)| id).collect();
    let mut extra_loads: Vec<(InstructionId, i64)> = Vec::new();
    for a in &accesses {
        if a.stored.is_some()
            || a.id == lane_store_id
            || a.size != esz
            || read_ids.contains(&a.id)
            || !in_region(a)
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

    // Every remaining region access must be accounted for.
    let mut allowed: HashSet<InstructionId> = [lane_store_id].into_iter().collect();
    if let Some((seed_id, ..)) = &seed {
        allowed.insert(*seed_id);
    }
    allowed.extend(read_ids.iter().copied());
    allowed.extend(extra_loads.iter().map(|&(id, _)| id));
    for a in &accesses {
        if allowed.contains(&a.id) || !in_region(a) {
            continue;
        }
        // Pre-loop region *stores* (any size/offset) are harmless: the snapshot
        // `arr0 = load(region)` is taken at the *end* of the preheader, so it sees
        // every preheader store. They stay in place — a RAM store through the base
        // pointer is observable and is never DSE'd (mem_liveness is register/temp
        // only); it is the snapshot *load* that store→load forwarding collapses.
        // Loads are NOT given this allowance: the seed store is deleted by the
        // rewrite, so a preheader load placed after it could change value.
        if a.stored.is_some() && a.block == preheader {
            continue;
        }
        // A whole-region load at the origin in the exit block is fed by the exit
        // write-back `store(region <- arr_e)` (emitted at the top of the exit
        // block), so it reads the promoted result. Left in place, to be forwarded.
        if a.stored.is_none()
            && a.block == exit
            && a.size == region_bytes
            && affine_base_const(&numbering, a.ptr, &is_r)
                .is_some_and(|(_, c)| c == origin_word * esz as i64)
        {
            continue;
        }
        let strided_r = affine_strided_lane(&numbering, a.ptr, a.size, |v| root_of(v).is_some())
            .and_then(|(b, _, _)| root_of(b))
            == Some(base_root);
        let const_r = affine_base_const(&numbering, a.ptr, &is_r).is_some();
        if strided_r || const_r {
            return None; // an unmodelled access to the region
        }
    }

    let seed = seed.map(|(id, _, val, _)| Seed { id, val });

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
        seed,
        lane_store_id,
        stored_val,
        store_delta,
        region_reads,
        reads_original,
        extra_loads,
    })
}

/// Append `arg` to the branch terminator of `from` on the edge to `to`.
fn append_edge_arg<'str, H: HostMut<'str>>(host: &mut H, from: BlockId, to: BlockId, arg: ValueId) {
    let Some(term) = BlockRef::new(host.read_host(), from).iter().last() else {
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
    host.replace_instruction_mnemonic(term_id, m);
}

/// Build `index + delta` (as `index`, `index - 1`, or `index + c`) at `index`'s
/// own width, so the arithmetic wraps exactly as the lifted address did. The
/// `index - 1` form is emitted verbatim as a `sub` so `loop_to_scan`'s
/// `is_decrement` recognizes it.
fn index_plus<'str, 'ctx, Ctx: HostMut<'str>>(
    b: &mut Builder<'str, 'ctx, Ctx>,
    index: ValueId,
    delta: i64,
    width: usize,
) -> ValueId {
    if delta == 0 {
        return index;
    }
    if delta == -1 {
        let one = b.context().get_const(1, width).id();
        return b.push_sub(index, one).id();
    }
    let mask = if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    };
    let c = b.context().get_const(delta as u64 & mask, width).id();
    b.push_add(index, c).id()
}

/// The region base pointer `base_root (+ origin_word*esz)`, pushing the offset add
/// through `b` when the origin is nonzero.
fn region_base<'str, 'ctx, Ctx: HostMut<'str>>(
    b: &mut Builder<'str, 'ctx, Ctx>,
    base_root: ValueId,
    origin_word: i64,
    esz: usize,
    width: usize,
) -> ValueId {
    if origin_word == 0 {
        return base_root;
    }
    let off = b
        .context()
        .get_const((origin_word * esz as i64) as u64, width)
        .id();
    b.push_add(base_root, off).id()
}

/// The byte width of a value's type, routed through the host so a checked-out
/// function's own instruction/param results are read from its owned arena rather
/// than the (sentinel) shared registry slot.
fn width_of<'str, H: HostMut<'str>>(host: &H, v: ValueId) -> usize {
    let ty = match v {
        ValueId::Instruction(iid) => {
            qcode::value::InstructionRef::new(host.read_host(), iid).type_id()
        }
        ValueId::BlockParam(pid) => host.read_host().block_param(pid).type_id,
        other => host
            .shared()
            .stored_type_of(other)
            .expect("value has a stored type"),
    };
    host.shared().types.size_of(ty)
}

/// Push an `index_plus(index, delta)` value into `block` before `before`, through
/// a checkout-safe builder (const/add/sub only). Returns the index value.
fn make_index<'str, H: HostMut<'str>>(
    host: &mut H,
    block: BlockId,
    before: InstructionId,
    index: ValueId,
    delta: i64,
) -> ValueId {
    let width = width_of(host, index);
    let mut b = Builder::from_block(BaseRef::new(host.reborrow_host(), block));
    b.set_insert_point_before(before);
    let v = index_plus(&mut b, index, delta, width);
    unsafe { b.dont_finalize() };
    v
}

/// Create a typed instruction with `mnemonic` and splice it before `before` in
/// `block` (avoids the Builder's `context_mut` type-mint path).
fn insert_before<'str, H: HostMut<'str>>(
    host: &mut H,
    block: BlockId,
    before: InstructionId,
    mnemonic: Mnemonic,
    ty: TypeId,
) -> ValueId {
    let id = host.push_mnemonic_with_type(block.func, mnemonic, ty);
    host.insert_insn_before(block, before, id);
    ValueId::Instruction(id)
}

/// Create a typed instruction and insert it before `block`'s first instruction.
/// The preheader always ends in a `goto header`, so it is never empty here.
fn insert_at_top<'str, H: HostMut<'str>>(
    host: &mut H,
    block: BlockId,
    mnemonic: Mnemonic,
    ty: TypeId,
) -> ValueId {
    let first = BlockRef::new(host.read_host(), block)
        .iter()
        .next()
        .expect("preheader has a terminator")
        .id;
    let id = host.push_mnemonic_with_type(block.func, mnemonic, ty);
    host.insert_insn_before(block, first, id);
    ValueId::Instruction(id)
}

/// The last instruction id of `block`.
fn last_insn<'str, H: HostMut<'str>>(host: &H, block: BlockId) -> InstructionId {
    BlockRef::new(host.read_host(), block)
        .iter()
        .last()
        .unwrap()
        .id
}

fn apply<'str, H: HostMut<'str>>(host: &mut H, m: &PromoteMatch) -> bool {
    let esz = m.elem_size;
    let elem_ty = host.shared().types.get_or_make_int(esz);
    let arr_ty = host.shared().types.get_or_make_array(elem_ty, m.count);
    let arr_sz = m.count * esz;

    let insert_id = IntrinsicId::from_name("insert").expect("insert registered");
    let at_id = IntrinsicId::from_name("at").expect("at registered");

    // New carried-array params (pushed last so appended edge args line up). In the
    // rotated shape the header *is* the body, so they share one param.
    // Push a fresh array-typed param onto `bid` (host-routed mirror of
    // `BasicBlock::push_param` followed by the original's `type_id = arr_ty`).
    let new_param = |host: &mut H, bid: BlockId| {
        let index = host.read_host().block(bid).params.len();
        let pid = host.push_block_param(
            bid.func,
            qcode::value::block_param::BlockParam {
                index,
                type_id: arr_ty,
                parent: Some(bid),
                name: None,
                origin: None,
                protected: false,
            },
        );
        host.block_mut(bid).params.push(pid);
        ValueId::BlockParam(pid)
    };
    let arr_h = new_param(host, m.header);
    let arr_b = if m.rotated {
        arr_h
    } else {
        new_param(host, m.body)
    };
    let arr_e = new_param(host, m.exit);

    // Snapshot at the *end* of the preheader — after every preheader store, so the
    // carried array starts from the region's fully-initialized contents — when any
    // original lane is read; else a symbolic zero splat (no new observable read).
    let arr0 = if m.reads_original {
        let term_id = last_insn(host, m.preheader);
        let base_width = width_of(host, m.base_root);
        let dst = {
            let mut b = Builder::from_block(BaseRef::new(host.reborrow_host(), m.preheader));
            b.set_insert_point_before(term_id);
            let dst = region_base(&mut b, m.base_root, m.origin_word, esz, base_width);
            unsafe { b.dont_finalize() };
            dst
        };
        insert_before(
            host,
            m.preheader,
            term_id,
            Mnemonic::Load(Load {
                space: m.region_space,
                ptr: dst,
                size: arr_sz,
            }),
            arr_ty,
        )
    } else {
        let splat_id = IntrinsicId::from_name("splat").expect("splat registered");
        let zero_elem = host.shared().get_const(0, esz).id();
        let count_const = host.shared().get_const(m.count as u64, 8).id();
        insert_at_top(
            host,
            m.preheader,
            Mnemonic::Intrinsic(IntrinsicApp {
                id: splat_id,
                args: vec![zero_elem, count_const],
            }),
            arr_ty,
        )
    };

    // Seed: arr1 = insert(arr0, 0, seed_val) before the preheader terminator, else arr0.
    let arr1 = match &m.seed {
        Some(seed) => {
            let term_id = last_insn(host, m.preheader);
            let idx0 = host.shared().get_const(0, 8).id();
            insert_before(
                host,
                m.preheader,
                term_id,
                Mnemonic::Intrinsic(IntrinsicApp {
                    id: insert_id,
                    args: vec![arr0, idx0, seed.val],
                }),
                arr_ty,
            )
        }
        None => arr0,
    };

    // Body reads: each region load at `index+od` becomes `at(arr_b, index+od)`.
    for &(load_id, od) in &m.region_reads {
        let idx = make_index(host, m.body, load_id, m.index, od);
        let at_val = insert_before(
            host,
            m.body,
            load_id,
            Mnemonic::Intrinsic(IntrinsicApp {
                id: at_id,
                args: vec![arr_b, idx],
            }),
            elem_ty,
        );
        host.replace_all_uses_with(ValueId::Instruction(load_id), at_val);
        host.remove_instruction(load_id);
    }

    // Body write: arr_next = insert(arr_b, index+store_delta, stored_val), before the branch.
    let arr_next = {
        let term_id = last_insn(host, m.body);
        let idx = make_index(host, m.body, term_id, m.index, m.store_delta);
        insert_before(
            host,
            m.body,
            term_id,
            Mnemonic::Intrinsic(IntrinsicApp {
                id: insert_id,
                args: vec![arr_b, idx, m.stored_val],
            }),
            arr_ty,
        )
    };

    // Exit const reads: rewrite to `at(arr_e, element)`.
    for &(load_id, elem) in &m.extra_loads {
        let idx = host.shared().get_const(elem as u64, 8).id();
        let at_val = insert_before(
            host,
            m.exit,
            load_id,
            Mnemonic::Intrinsic(IntrinsicApp {
                id: at_id,
                args: vec![arr_e, idx],
            }),
            elem_ty,
        );
        host.replace_all_uses_with(ValueId::Instruction(load_id), at_val);
        host.remove_instruction(load_id);
    }

    // Exit write-back: store(region, base(+origin) <- arr_e) at the *top* of the exit
    // block, so any whole-region exit load left in place reads the promoted result.
    {
        let first_id = BlockRef::new(host.read_host(), m.exit)
            .iter()
            .next()
            .unwrap()
            .id;
        let base_width = width_of(host, m.base_root);
        let mut b = Builder::from_block(BaseRef::new(host.reborrow_host(), m.exit));
        b.set_insert_point_before(first_id);
        let dst = region_base(&mut b, m.base_root, m.origin_word, esz, base_width);
        b.push_store(arr_e, dst, m.region_space);
        unsafe { b.dont_finalize() };
    }

    // Drop the now-dead memory traffic. (Region loads were removed inline above.)
    host.remove_instruction(m.lane_store_id);
    if let Some(seed) = &m.seed {
        host.remove_instruction(seed.id);
    }

    // Thread the carried array through every loop edge.
    append_edge_arg(host, m.preheader, m.header, arr1);
    append_edge_arg(host, m.body, m.header, arr_next);
    if !m.rotated {
        append_edge_arg(host, m.header, m.body, arr_h);
    }
    append_edge_arg(host, m.header, m.exit, arr_h);

    true
}

use crate::{FunctionBody, FunctionPassV2, ModuleView};

impl FunctionPassV2 for ArrayPromote {
    const NAME: &'static str = "array_promote";

    fn description(&self) -> &'static str {
        "Promote in-place strided RAM array-fill loops to a value-carried array (insert/at)"
    }

    fn run<'str>(
        &self,
        m: &ModuleView<'_, 'str>,
        f: &mut FunctionBody<'str>,
    ) -> Result<bool, String> {
        let fid = f.id();
        let mut host = f.host(m);
        match try_match(host.read_host(), fid) {
            Some(matched) => Ok(apply(&mut host, &matched)),
            None => Ok(false),
        }
    }
}

crate::register_function_pass_v2!(ArrayPromote);

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use crate::test_util::run_function_pass_v2;
    use qcode::context::Context;
    use qcode::value::{BasicBlock, Function};

    // A seeded, memory-carried strided fill: `out[0] = seed`, `out[i] =
    // out[i-1] + i` reloading the previous lane. The reload is a *carry* read
    // (`od = -1 < store_delta = 0`), so `reads_original == false` → splat init.
    #[test]
    fn memory_carried_fill_promoted() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn fill:
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
                %jt = trunc(i32, @j);
                %next = %prev + %jt;
                %woff = @j * 4;
                %waddr = @b + %woff;
                store(ram:4, %waddr <- %next);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        let changed = run_function_pass_v2::<ArrayPromote>(&mut ctx, fill).unwrap();
        assert!(changed, "the memory-carried fill should be recognized");
        let ir = format!("{}", Function::from_id(&ctx, fill));
        assert!(ir.contains("$at("), "lane load should become at(): {ir}");
        assert!(
            ir.contains("$insert("),
            "lane store should become insert(): {ir}"
        );
        assert!(
            ir.contains("store(ram:2496"),
            "the whole [i32;624] region should be stored once at exit: {ir}"
        );
        assert!(
            !ir.contains("load(ram:4"),
            "lane load should be removed: {ir}"
        );
        assert!(
            !ir.contains("store(ram:4"),
            "lane stores should be removed: {ir}"
        );
    }

    // Register-carried, write-only shape: no in-loop lane load, only indexed lane
    // stores. Still an affine strided fill, so it promotes. Rotated (do-while).
    #[test]
    fn register_carried_rotated_is_promoted() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn reg_rot:
            <entry @seed:i64 @base:i64>
                %e0 = trunc(i32, @seed);
                store(ram:4, @base <- %e0);
                goto <body @j=1 @b=@base @a=%e0>;
            <body @j:i64 @b:i64 @a:i32>
                %m = @a * 3;
                %jt = trunc(i32, @j);
                %next = %m + %jt;
                %woff = @j * 4;
                %waddr = @b + %woff;
                store(ram:4, %waddr <- %next);
                %j1 = @j + 1;
                %done = %j1 < 624;
                if %done goto <body @j=%j1 @b=@b @a=%next> else goto <exit>;
            <exit>
                return at i64 0x0;
            "
        );
        let changed = run_function_pass_v2::<ArrayPromote>(&mut ctx, reg_rot).unwrap();
        let ir = format!("{}", Function::from_id(&ctx, reg_rot));
        assert!(
            changed,
            "rotated register-carried fill should promote: {ir}"
        );
        assert!(
            ir.contains("$insert("),
            "lane store should become insert(): {ir}"
        );
        assert!(
            !ir.contains("store(ram:4"),
            "lane stores should be removed: {ir}"
        );
    }

    // The split shape of the same register-carried fill promotes identically.
    #[test]
    fn register_carried_split_is_promoted() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn reg_split:
            <entry @seed:i64 @base:i64>
                %e0 = trunc(i32, @seed);
                store(ram:4, @base <- %e0);
                goto <head @i=1 @buf=@base @acc=%e0>;
            <head @i:i64 @buf:i64 @acc:i32>
                %done = @i == 624;
                if %done goto <exit> else goto <body @j=@i @b=@buf @a=@acc>;
            <body @j:i64 @b:i64 @a:i32>
                %m = @a * 3;
                %jt = trunc(i32, @j);
                %next = %m + %jt;
                %woff = @j * 4;
                %waddr = @b + %woff;
                store(ram:4, %waddr <- %next);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b @acc=%next>;
            <exit>
                return at i64 0x0;
            "
        );
        let changed = run_function_pass_v2::<ArrayPromote>(&mut ctx, reg_split).unwrap();
        let ir = format!("{}", Function::from_id(&ctx, reg_split));
        assert!(
            changed,
            "split-shape register-carried fill should promote: {ir}"
        );
        assert!(
            ir.contains("$insert("),
            "lane store should become insert(): {ir}"
        );
        assert!(
            !ir.contains("store(ram:4"),
            "lane stores should be removed: {ir}"
        );
    }

    // A seedless pure generation `out[i] = i * 3`: write-only, splat init.
    #[test]
    fn write_only_generated_fill_promoted() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn generate:
            <entry @base:i64>
                goto <head @i=0 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 624;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %jt = trunc(i32, @j);
                %next = %jt * 3;
                %woff = @j * 4;
                %waddr = @b + %woff;
                store(ram:4, %waddr <- %next);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        let changed = run_function_pass_v2::<ArrayPromote>(&mut ctx, generate).unwrap();
        let ir = format!("{}", Function::from_id(&ctx, generate));
        assert!(changed, "a write-only generated fill should promote: {ir}");
        assert!(
            ir.contains("$insert("),
            "lane store should become insert(): {ir}"
        );
        assert!(
            !ir.contains("store(ram:4"),
            "lane stores should be removed: {ir}"
        );
    }

    // A byte buffer wrapped in a whole-region envelope: a pre-loop wide store of
    // all `count*esz` bytes at the base (the region initializer) and an exit wide
    // load. The initializer store must NOT trip the accounting veto — it is the
    // counterpart of the exit envelope read, and the promoted `arr0 = load(region)`
    // reads back exactly what it wrote. Mirrors the lifted `buf[i] = f(buf[i], i)`
    // over an argpromote-functionalized stack buffer.
    #[test]
    fn whole_region_initializer_store_does_not_veto() {
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
                %out = load(ram:24, @base);
                return at i64 0x0;
            "
        );
        let changed = run_function_pass_v2::<ArrayPromote>(&mut ctx, mix).unwrap();
        let ir = format!("{}", Function::from_id(&ctx, mix));
        assert!(changed, "the enveloped byte fill should promote: {ir}");
        assert!(ir.contains("$at("), "lane load becomes at(): {ir}");
        assert!(ir.contains("$insert("), "lane store becomes insert(): {ir}");
        assert!(
            !ir.contains("load(ram:1") && !ir.contains("store(ram:1"),
            "byte lane traffic should be gone: {ir}"
        );
        // Ordering soundness: the snapshot `load(ram:24, @base)` must be taken
        // *after* the init store, so the carried array sees the initialized bytes.
        let init_store = ir
            .find("store(ram:24, i64 @base <- i192")
            .expect("init store kept");
        let snapshot = ir
            .find("load(ram:24, i64 @base)")
            .expect("snapshot load present");
        assert!(
            init_store < snapshot,
            "snapshot must follow the init store: {ir}"
        );
    }

    // The exit wide load keeps its width: it is fed by the write-back store, not
    // narrowed to a 1-byte `at`. The write-back must precede it so it reads the
    // promoted result.
    #[test]
    fn wide_exit_load_keeps_width_and_reads_result() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn mix:
            <entry @base:i64>
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
                %out = load(ram:24, @base);
                %r = trunc(i64, %out);
                return at i64 %r;
            "
        );
        let changed = run_function_pass_v2::<ArrayPromote>(&mut ctx, mix).unwrap();
        let ir = format!("{}", Function::from_id(&ctx, mix));
        assert!(changed, "should promote: {ir}");
        assert!(
            ir.contains("load(ram:24, i64 @base)"),
            "wide exit load keeps its width (not narrowed to a 1-byte at): {ir}"
        );
        let store = ir
            .find("store(ram:24, i64 @base <-")
            .expect("write-back present");
        // The exit wide load is the last such load (the first is the preheader snapshot).
        let load = ir
            .rfind("load(ram:24, i64 @base)")
            .expect("wide load present");
        assert!(
            store < load,
            "write-back must precede the exit wide load so it reads the result: {ir}"
        );
    }

    // A split loop whose param-less body reads the *header* induction param
    // directly (`@i`) instead of copying it into a body param. This is the shape
    // an SSA-minimal byte-fill loop takes; `unit_induction` must recognize the
    // header-carried induction, and the fill must promote just like the
    // body-copied form.
    #[test]
    fn header_carried_byte_fill_promoted() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <entry @base:i64>
                goto <head @i=0>;
            <head @i:i64>
                %done = @i < 24;
                if %done goto <body> else goto <exit>;
            <body>
                %addr = @base + @i;
                %cur = load(ram:1, %addr);
                %jt = trunc(i8, @i);
                %x = %cur ^ %jt;
                store(ram:1, %addr <- %x);
                %i1 = @i + 1;
                goto <head @i=%i1>;
            <exit>
                return at i64 0x0;
            "
        );
        let changed = run_function_pass_v2::<ArrayPromote>(&mut ctx, f).unwrap();
        let ir = format!("{}", Function::from_id(&ctx, f));
        assert!(changed, "header-carried byte fill should promote: {ir}");
        assert!(ir.contains("$at("), "lane load becomes at(): {ir}");
        assert!(ir.contains("$insert("), "lane store becomes insert(): {ir}");
        assert!(
            !ir.contains("load(ram:1"),
            "lane load should be removed: {ir}"
        );
        assert!(
            !ir.contains("store(ram:1"),
            "lane store should be removed: {ir}"
        );
    }

    // Base/index classification for a unit element stride must not depend on
    // `ValueId` order. Here the induction body param `@j` is created *before* the
    // base body param `@b`, so it sorts ahead of the base in the affine form; the
    // lane address `@j + @b` still has to bind `@b` as base and `@j` as index.
    #[test]
    fn byte_stride_base_index_order_independent() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <entry @base:i64>
                goto <head @i=0 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i < 24;
                if %done goto <body @j=@i @b=@buf> else goto <exit>;
            <body @j:i64 @b:i64>
                %addr = @j + @b;
                %cur = load(ram:1, %addr);
                %x = %cur - 0x28;
                store(ram:1, %addr <- %x);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        let changed = run_function_pass_v2::<ArrayPromote>(&mut ctx, f).unwrap();
        let ir = format!("{}", Function::from_id(&ctx, f));
        assert!(
            changed,
            "induction sorting ahead of the base must still promote: {ir}"
        );
        assert!(ir.contains("$insert("), "lane store becomes insert(): {ir}");
        assert!(
            !ir.contains("store(ram:1"),
            "lane store should be removed: {ir}"
        );
    }

    // A *partial* pre-loop store (4 of 24 bytes, off the origin) in the preheader
    // must not veto: pre-loop region traffic is absorbed by the end-of-preheader
    // snapshot regardless of shape. Guards against the deleted shape-match's
    // over-specificity.
    #[test]
    fn partial_preloop_store_does_not_veto() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn mix:
            <entry @base:i64 @v:i64>
                %v32 = trunc(i32, @v);
                %off = @base + 8;
                store(ram:4, %off <- %v32);
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
        let changed = run_function_pass_v2::<ArrayPromote>(&mut ctx, mix).unwrap();
        let ir = format!("{}", Function::from_id(&ctx, mix));
        assert!(changed, "a partial pre-loop store should not block: {ir}");
        assert!(ir.contains("$insert("), "lane store becomes insert(): {ir}");
    }

    // A whole-region store *inside the loop body* is an unmodelled write the carried
    // array never sees — it must veto (positional rule: only preheader stores are free).
    #[test]
    fn whole_region_store_in_body_vetoes() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn mix:
            <entry @base:i64 @src:i64>
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
                %blob = load(ram:24, @src);
                store(ram:24, @base <- %blob);
                %j1 = @j + 1;
                goto <head @i=%j1>;
            <exit>
                return at i64 0x0;
            "
        );
        let changed = run_function_pass_v2::<ArrayPromote>(&mut ctx, mix).unwrap();
        assert!(
            !changed,
            "a whole-region store in the body must block promotion"
        );
    }

    #[test]
    fn idempotent_second_run_is_noop() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn fill:
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
                %jt = trunc(i32, @j);
                %next = %prev + %jt;
                %woff = @j * 4;
                %waddr = @b + %woff;
                store(ram:4, %waddr <- %next);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        assert!(run_function_pass_v2::<ArrayPromote>(&mut ctx, fill).unwrap());
        assert!(
            !run_function_pass_v2::<ArrayPromote>(&mut ctx, fill).unwrap(),
            "second run should find nothing to promote"
        );
    }

    // Seeded prefix sum `out[0]=seed; out[i]=out[i-1]+l[i]`: reads the carry
    // `l[i-1]` (od=-1) AND the original own lane `l[i]` (od=0). `reads_original`
    // → the array is initialized from the region's original contents, threaded as
    // the single carried array (no separate snapshot param).
    #[test]
    fn reads_original_lane_uses_original_init() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn reads_orig:
            <entry @seed:i64 @base:i64>
                %e0 = @seed[0:4];
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
            run_function_pass_v2::<ArrayPromote>(&mut ctx, reads_orig).unwrap(),
            "an original-lane read should promote via an original init"
        );
        let ir = format!("{}", Function::from_id(&ctx, reads_orig));
        // Exactly one wide original load, in the preheader, threaded as the carried
        // array — no separate snapshot, no second wide load.
        assert_eq!(
            ir.matches("load(ram:2496").count(),
            1,
            "exactly one whole-region original init load: {ir}"
        );
        assert!(ir.contains("$at("), "reads should become at(): {ir}");
        assert!(
            !ir.contains("load(ram:4"),
            "per-lane loads should be removed: {ir}"
        );
    }

    // Seedless indexed map `l[i] = l[i]*3 + i`: reads its own original lane, no
    // carry. `reads_original` → one original init load; own-lane read → at().
    #[test]
    fn index_map_over_original_array_promoted() {
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
                %jt = @j[0:4];
                %next = %m + %jt;
                store(ram:4, %addr <- %next);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        assert!(
            run_function_pass_v2::<ArrayPromote>(&mut ctx, reads_enum).unwrap(),
            "a seedless indexed map over the original array should promote"
        );
        let ir = format!("{}", Function::from_id(&ctx, reads_enum));
        assert!(
            ir.contains("load(ram:2496"),
            "a whole-region original init load should be inserted: {ir}"
        );
        assert!(
            ir.contains("$at("),
            "the original read should become at(): {ir}"
        );
        assert!(
            !ir.contains("load(ram:4"),
            "the per-lane load should be removed: {ir}"
        );
    }

    // Both a carry read `l[i-1]` and an original read `l[i]` are served by ONE
    // carried array — the promoted body has exactly one array-typed param reaching
    // it (no separate `%l0` snapshot).
    #[test]
    fn original_and_carry_share_one_array() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn shared:
            <entry @seed:i64 @base:i64>
                %e0 = @seed[0:4];
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
        assert!(run_function_pass_v2::<ArrayPromote>(&mut ctx, shared).unwrap());
        // Count array-typed params reaching the body: exactly one carried array.
        let body = Function::from_id(&ctx, shared)
            .iter()
            .find(|b| {
                b.params()
                    .any(|p| p.name() == Some("j") || p.name().is_none())
                    && b.iter()
                        .any(|i| matches!(i.mnemonic(), Mnemonic::Intrinsic(_)))
            })
            .map(|b| b.id)
            .expect("body block");
        let arr_params = BasicBlock::from_id(&ctx, body)
            .params()
            .filter(|p| ctx.types.array_of(p.type_id()).is_some())
            .count();
        assert_eq!(
            arr_params, 1,
            "exactly one carried array param reaches the body"
        );
    }

    // An own-lane load placed *after* the lane store is not a read of the original
    // value, so the program-order check must decline the promotion.
    #[test]
    fn own_lane_read_after_store_not_promoted() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn after:
            <entry @base:i64>
                goto <head @i=0 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 624;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %off = @j * 4;
                %addr = @b + %off;
                %jt = @j[0:4];
                store(ram:4, %addr <- %jt);
                %cur = load(ram:4, %addr);
                %sink = %cur + %jt;
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        // A single lane store, then an own-lane load *after* it: the load no longer
        // observes the original value, so the program-order check declines.
        assert!(
            !run_function_pass_v2::<ArrayPromote>(&mut ctx, after).unwrap(),
            "an own-lane read after the lane store must not promote"
        );
    }
}
