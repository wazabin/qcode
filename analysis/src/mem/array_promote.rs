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
struct PromoteMatch {
    preheader: BlockId,
    header: BlockId,
    body: BlockId,
    exit: BlockId,
    /// Loop-invariant root pointer param the region is based at.
    base_root: ValueId,
    /// Body induction parameter (the lane index in the store/load addresses).
    index: ValueId,
    elem_size: usize,
    count: usize,
    /// Pre-loop lane-0 seed store and the value it writes.
    seed_id: InstructionId,
    seed_val: ValueId,
    /// The one strided lane store: `arr[index + store_delta] = stored_val`.
    lane_store_id: InstructionId,
    stored_val: ValueId,
    store_delta: i64,
    /// The one strided lane load: `at(arr, index + load_delta)`.
    lane_load_id: InstructionId,
    load_delta: i64,
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
    let ram = ctx.default_space;

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

    // Collect every RAM access.
    struct Acc {
        id: InstructionId,
        block: BlockId,
        ptr: ValueId,
        size: usize,
        stored: Option<ValueId>,
    }
    let mut accesses: Vec<Acc> = Vec::new();
    for block in Function::from_id(ctx, fid).iter() {
        let bid = block.id;
        for insn in block.iter() {
            match insn.mnemonic() {
                Mnemonic::Load(l) if l.space == ram => accesses.push(Acc {
                    id: insn.id,
                    block: bid,
                    ptr: l.ptr,
                    size: l.size,
                    stored: None,
                }),
                Mnemonic::Store(s) if s.space == ram => accesses.push(Acc {
                    id: insn.id,
                    block: bid,
                    ptr: s.ptr,
                    size: s.size,
                    stored: Some(s.src),
                }),
                _ => {}
            }
        }
    }

    let numbering = precompute_forms(ctx, fid);
    let val_root = base_roots(ctx, fid);
    let root_of = |v: ValueId| -> Option<ValueId> { val_root.get(&v).copied() };

    // The single strided lane store establishes (base_root, elem_size, index).
    let mut lane: Option<(
        InstructionId,
        ValueId,
        ValueId,
        ValueId,
        i64,
        BlockId,
        usize,
    )> = None;
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
        lane = Some((a.id, r, idx, src, c, a.block, a.size));
    }
    let (lane_store_id, base_root, index, stored_val, c_lane, body, esz) = lane?;
    if esz == 0 || c_lane % esz as i64 != 0 {
        return None;
    }
    let store_delta = c_lane / esz as i64;
    let is_r = |v: ValueId| root_of(v) == Some(base_root);

    // Seed store: a const-offset store of one element at the region base.
    let mut seed: Option<(InstructionId, ValueId, i64)> = None;
    for a in &accesses {
        if a.id == lane_store_id || a.size != esz {
            continue;
        }
        let Some(src) = a.stored else { continue };
        let Some((_, c_seed)) = affine_base_const(&numbering, a.ptr, &is_r) else {
            continue;
        };
        if c_seed % esz as i64 != 0 {
            continue;
        }
        if seed.is_some() {
            return None;
        }
        seed = Some((a.id, src, c_seed / esz as i64));
    }
    let (seed_id, seed_val, seed_word) = seed?;

    // Lane load: the strided read (the memory-carried previous element).
    let mut load: Option<(InstructionId, i64)> = None;
    for a in &accesses {
        if a.id == lane_store_id || a.stored.is_some() || a.size != esz {
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
        load = Some((a.id, c / esz as i64));
    }
    let (lane_load_id, load_delta) = load?;

    // Every RAM access reaching the region must be one of {seed, lane store, load}.
    let allowed: HashSet<InstructionId> =
        [seed_id, lane_store_id, lane_load_id].into_iter().collect();
    for a in &accesses {
        if allowed.contains(&a.id) {
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

    // --- Loop structure ---
    // `index` is the body's induction parameter; the body's single predecessor is
    // the header, whose param feeds `index` on the header→body edge.
    let ValueId::BlockParam(ipid) = index else {
        return None;
    };
    if ctx.values.block_params[ipid].parent != Some(body) {
        return None;
    }
    let body_preds: Vec<BlockId> = BasicBlock::from_id(ctx, body)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    if body_preds.len() != 1 {
        return None;
    }
    let header = body_preds[0];
    let kj = BasicBlock::from_id(ctx, body)
        .params()
        .position(|p| p.id() == index)?;
    let hi = match header_incoming(ctx, body, kj)[..] {
        [v] => v,
        _ => return None,
    };
    let ValueId::BlockParam(hpid) = hi else {
        return None;
    };
    if ctx.values.block_params[hpid].parent != Some(header) {
        return None;
    }

    // Header cbranch → {body, exit}; header preds = {preheader, body}.
    let term = BasicBlock::from_id(ctx, header).iter().last()?;
    let Mnemonic::CBranch(cb) = term.mnemonic() else {
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
    if exit == body {
        return None;
    }
    let header_preds: Vec<BlockId> = BasicBlock::from_id(ctx, header)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    if header_preds.len() != 2 || !header_preds.contains(&body) {
        return None;
    }
    let preheader = *header_preds.iter().find(|&&p| p != body)?;
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

    // Induction: `hi` starts at a single literal `s` and steps by one.
    let khi = BasicBlock::from_id(ctx, header)
        .params()
        .position(|p| p.id() == hi)?;
    // The back-edge increments the induction variable by one. The IR carries the
    // *body* copy (`index`), so the increment reads `index + 1`, not `hi + 1`.
    let inc = header_incoming(ctx, header, khi);
    if !inc.iter().any(|&v| is_increment(ctx, v, index)) {
        return None;
    }
    let inits: HashSet<u64> = inc
        .iter()
        .filter(|&&v| !is_increment(ctx, v, index))
        .filter_map(|&v| literal(ctx, v))
        .collect();
    if inits.len() != 1 {
        return None;
    }
    let s = *inits.iter().next().unwrap() as i64;

    // Trip bound `N` from the guard `hi <cmp> N`; the last body index is `N-1`.
    let n = guard_bound(ctx, cond, hi, exit == sb)?;

    // Coverage: seed word 0, lane stores tile [1, count-1] contiguously.
    if seed_word != 0 {
        return None;
    }
    let store_lo = s + store_delta;
    let store_hi = (n - 1) + store_delta;
    if store_lo != 1 || store_hi < store_lo {
        return None;
    }
    let count = (store_hi + 1) as usize;
    if count == 0 || count.saturating_mul(esz) > (1 << 20) {
        return None;
    }

    Some(PromoteMatch {
        preheader,
        header,
        body,
        exit,
        base_root,
        index,
        elem_size: esz,
        count,
        seed_id,
        seed_val,
        lane_store_id,
        stored_val,
        store_delta,
        lane_load_id,
        load_delta,
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

fn apply(ctx: &mut Context, m: &PromoteMatch) -> bool {
    let esz = m.elem_size;
    let elem_ty = ctx.types.get_or_make_int(esz);
    let arr_ty = ctx.types.get_or_make_array(elem_ty, m.count);
    let arr_sz = m.count * esz;

    let insert_id =
        qcode::value::insn::IntrinsicId::from_name("insert").expect("insert registered");
    let at_id = qcode::value::insn::IntrinsicId::from_name("at").expect("at registered");

    // New carried-array params (pushed last on each block so appended edge args
    // line up positionally).
    let arr_h = {
        let pid = BasicBlock::from_id_mut(ctx, m.header).push_param(arr_sz).id;
        ctx.values.block_params[pid].type_id = arr_ty;
        ValueId::BlockParam(pid)
    };
    let arr_b = {
        let pid = BasicBlock::from_id_mut(ctx, m.body).push_param(arr_sz).id;
        ctx.values.block_params[pid].type_id = arr_ty;
        ValueId::BlockParam(pid)
    };
    let arr_e = {
        let pid = BasicBlock::from_id_mut(ctx, m.exit).push_param(arr_sz).id;
        ctx.values.block_params[pid].type_id = arr_ty;
        ValueId::BlockParam(pid)
    };

    // Zero-initialized array literal, retyped to [elem;count].
    let zero = {
        let id = ctx.get_bytes(vec![0u8; arr_sz]).id();
        if let ValueId::Bytes(bid) = id {
            ctx.values.bytes[bid].type_id = arr_ty;
        }
        id
    };

    // Preheader: %a0 = insert(zero, seed_word=0, seed_val), before the branch.
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

    // Body: at(arr_b, index + load_delta) replacing the lane load.
    let at_val = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.body));
        b.set_insert_point_before(m.lane_load_id);
        let idx = if m.load_delta == 0 {
            m.index
        } else {
            let c = b.context_mut().get_const(m.load_delta as u64, 8).id();
            b.push_add(m.index, c).id()
        };
        b.push_intrinsic(at_id, vec![arr_b, idx]).id()
    };
    ctx.replace_all_uses_with(ValueId::Instruction(m.lane_load_id), at_val);

    // Body: %arr' = insert(arr_b, index + store_delta, stored_val), before branch.
    let arr_next = {
        let term_id = BasicBlock::from_id(ctx, m.body).iter().last().unwrap().id;
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.body));
        b.set_insert_point_before(term_id);
        let idx = if m.store_delta == 0 {
            m.index
        } else {
            let c = b.context_mut().get_const(m.store_delta as u64, 8).id();
            b.push_add(m.index, c).id()
        };
        b.push_intrinsic(insert_id, vec![arr_b, idx, m.stored_val])
            .id()
    };

    // Exit: store(ram, base <- arr_e) before the exit terminator.
    {
        let term_id = BasicBlock::from_id(ctx, m.exit).iter().last().unwrap().id;
        let ram = ctx.default_space;
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.exit));
        b.set_insert_point_before(term_id);
        b.push_store(arr_e, m.base_root, ram);
    }

    // Drop the now-dead memory traffic.
    ctx.remove_instruction(m.lane_load_id);
    ctx.remove_instruction(m.lane_store_id);
    ctx.remove_instruction(m.seed_id);

    // Thread the array through every loop edge.
    append_edge_arg(ctx, m.preheader, m.header, a0);
    append_edge_arg(ctx, m.body, m.header, arr_next);
    append_edge_arg(ctx, m.header, m.body, arr_h);
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
}
