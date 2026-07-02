//! Loop-to-scan: fold the value-carried `insert`/`at` fill loop that
//! [`array_promote`](crate::mem::array_promote) produces into a single `scanl`.
//!
//! `array_promote` rewrites a memory-carried array-fill loop into a carried
//! array threaded through the loop with `insert`/`at`, seeded at lane 0 and
//! stored wide at the exit:
//!
//! ```text
//!   <entry @seed @base>
//!       %e0 = trunc(i32, @seed);
//!       %a0 = insert(zero:[i32;N], 0, %e0);
//!       goto <head @i=1 @buf=@base @arr=%a0>;
//!   <head @i @buf @arr>  if @i == N goto <exit @arr> else goto <body @j @b @arr>;
//!   <body @j @b @arr>
//!       %prev = at(@arr, @j-1);           // previous element = previous accumulator
//!       %next = f(%prev, @j);             // pure body of (acc, index)
//!       %arr' = insert(@arr, @j, %next);
//!       goto <head @i=@j+1 @buf=@b @arr=%arr'>;
//!   <exit @arr>  store(ram, @base <- @arr); return …;
//! ```
//!
//! The loop computes `arr[0] = seed`, `arr[j] = f(arr[j-1], j)` for `j ∈ [1, N)`.
//! That is exactly a left-scan: with `acc_0 = seed` and `acc_{k+1} = f(acc_k, k+1)`,
//! the `N-1` produced elements are `arr[1..N]`, and `arr[0]` is the seed itself.
//! So the whole fill folds to
//!
//! ```text
//!   %scan = scanl @f seed iota(N-1);              // arr[1..N]
//!   %full = concat(singleton(seed), %scan);       // arr[0..N]
//!   store(ram, @base <- %full);
//! ```
//!
//! where `@f` is the outlined pure body of `(accumulator, index)`. The residual
//! `insert`/`at` loop is left in place (its array is now unused) for `dce` to
//! remove. The recognizer trusts `array_promote`'s coverage proof — the array's
//! full length `N` comes straight from the exit store's element type.

use rustc_hash::FxHashSet as HashSet;

use qcode::{
    builder::Builder,
    context::Context,
    value::{
        BasicBlock, BlockId, Function, FunctionId, InstructionRef, ValueId, ValueRef,
        insn::{Binary, Binop, InstructionId, IntBinop, IntrinsicApp, IntrinsicId, Mnemonic},
    },
};

use super::loop_to_map::{ScanElem, outline_scan_body};
use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct LoopToScan;

/// A recognized `insert`/`at` fill loop (see the module docs).
struct ScanMatch {
    /// The loop-exit block holding the wide store.
    exit: BlockId,
    /// The `store(ram, base <- arr)` to rewrite.
    store_id: InstructionId,
    /// The lane-0 seed value (`acc_0`, also the first output element).
    seed_val: ValueId,
    /// The body's per-lane result (`%next`), the scan body's return value.
    stored_val: ValueId,
    /// The `at(arr, index-1)` result — the previous element, i.e. the accumulator.
    prev_val: ValueId,
    /// The body induction parameter used in the body expression.
    index: ValueId,
    /// The loop index at the first body iteration (`iota` element 0 maps here).
    index_start: i64,
    /// The full array length `N`.
    count: usize,
    /// The array element type.
    elem_ty: qcode::types::TypeId,
    /// Set when the body reads the region's *original* element at its own lane
    /// (`at(%l0, index)`) — the loop is a scan over the original array rather than
    /// a pure generation over `iota`. Holds `(element-read value, exit snapshot
    /// param to slice)`; the scan then ranges over `l0[1..]`.
    elem: Option<(ValueId, ValueId)>,
}

/// `c` if `v` is the integer literal `c`, else `None`.
fn literal(ctx: &Context, v: ValueId) -> Option<u64> {
    match ValueRef::new(v, ctx) {
        ValueRef::Literal(l) => Some(l.value()),
        _ => None,
    }
}

/// `true` if `v` is `idx + 1` (either operand order) — a unit step of `idx`.
fn is_increment(ctx: &Context, v: ValueId, idx: ValueId) -> bool {
    let ValueId::Instruction(id) = v else {
        return false;
    };
    let Mnemonic::Binop(Binary { lhs, rhs, op }) = ctx.get_insn(id).mnemonic() else {
        return false;
    };
    let one = |x: ValueId| literal(ctx, x) == Some(1);
    matches!(op, Binop::Int(IntBinop::Add))
        && ((*lhs == idx && one(*rhs)) || (*rhs == idx && one(*lhs)))
}

/// `true` if `v` is `idx - 1`, expressed either as `idx - 1` or as `idx + (-1)`
/// (the wrapping representation `array_promote` emits for the `at` back-index).
fn is_decrement(ctx: &mut Context, v: ValueId, idx: ValueId) -> bool {
    let ValueId::Instruction(id) = v else {
        return false;
    };
    let Mnemonic::Binop(Binary { lhs, rhs, op }) = ctx.get_insn(id).mnemonic() else {
        return false;
    };
    let (lhs, rhs, op) = (*lhs, *rhs, *op);
    let idx_ty = ctx.type_of(idx);
    let width = ctx.types.size_of(idx_ty);
    let neg_one = if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    };
    match op {
        Binop::Int(IntBinop::Sub) => lhs == idx && literal(ctx, rhs) == Some(1),
        Binop::Int(IntBinop::Add) => {
            (lhs == idx && literal(ctx, rhs) == Some(neg_one))
                || (rhs == idx && literal(ctx, lhs) == Some(neg_one))
        }
        _ => false,
    }
}

/// Values feeding block-param index `k` of `block` from every predecessor edge.
fn incoming(ctx: &Context, block: BlockId, k: usize) -> Vec<ValueId> {
    let mut out = Vec::new();
    let preds: Vec<BlockId> = BasicBlock::from_id(ctx, block)
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
                if cb.success_block == block {
                    out.extend(cb.success_args.get(k).copied());
                }
                if cb.failure_block == block {
                    out.extend(cb.failure_args.get(k).copied());
                }
            }
            _ => {}
        }
    }
    out
}

/// Index of block-param `p` within `block`'s parameter list.
fn param_pos(ctx: &Context, block: BlockId, p: ValueId) -> Option<usize> {
    BasicBlock::from_id(ctx, block)
        .params()
        .position(|q| q.id() == p)
}

/// Parent block of a block-param value.
fn param_parent(ctx: &Context, v: ValueId) -> Option<BlockId> {
    let ValueId::BlockParam(pid) = v else {
        return None;
    };
    ctx.values.block_params[pid].parent
}

/// Recognize the `insert`/`at` fill loop in `fid`.
fn try_match(ctx: &mut Context, fid: FunctionId) -> Option<ScanMatch> {
    let insert_id = IntrinsicId::from_name("insert")?;
    let at_id = IntrinsicId::from_name("at")?;

    // Anchor: a single `store(ram, base <- arr)` whose source is an array-typed
    // block param of the storing block (the loop's exit-carried array). Collect
    // candidate RAM stores first (the block walk borrows `ctx`), then resolve the
    // array type.
    let mut candidates: Vec<(InstructionId, BlockId, ValueId, ValueId)> = Vec::new();
    for block in Function::from_id(ctx, fid).iter() {
        let bid = block.id;
        for insn in block.iter() {
            let Mnemonic::Store(s) = insn.mnemonic() else {
                continue;
            };
            // Any RAM space — array_promote emits the wide store into whatever RAM
            // space the region lived in, not necessarily the default space.
            if matches!(
                qcode::space::Space::from_id(ctx, s.space).ty,
                qcode::space::SpaceType::Ram
            ) {
                candidates.push((insn.id, bid, s.ptr, s.src));
            }
        }
    }
    let mut anchor = None;
    for (id, bid, ptr, src) in candidates {
        // The store source must be an array-typed block param — the loop-carried
        // array reaching the exit, either the exit's own pass-through param
        // (split shape) or the loop header's param directly (rotated shape, where
        // gvn has already coalesced the trivial exit pass-through away).
        if !matches!(src, ValueId::BlockParam(_)) {
            continue;
        }
        let src_ty = ctx.type_of(src);
        let Some((elem_ty, count)) = ctx.types.array_of(src_ty) else {
            continue;
        };
        if count == 0 {
            continue;
        }
        if anchor.is_some() {
            return None; // more than one array store — not the canonical shape
        }
        anchor = Some((id, bid, ptr, src, elem_ty, count));
    }
    let (store_id, exit, _base, arr_src, elem_ty, count) = anchor?;

    // Exit's single predecessor is the loop header.
    let exit_preds: Vec<BlockId> = BasicBlock::from_id(ctx, exit)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    let [header] = exit_preds[..] else {
        return None;
    };
    // The header's own carried array param: the store source directly if it lives
    // on the header (rotated/coalesced), otherwise the value the exit's
    // pass-through param copies from the header on the exit edge (split).
    let arr_h = if param_parent(ctx, arr_src) == Some(header) {
        arr_src
    } else {
        let k_arre = param_pos(ctx, exit, arr_src)?;
        match incoming(ctx, exit, k_arre)[..] {
            [v] => v,
            _ => return None,
        }
    };
    if param_parent(ctx, arr_h) != Some(header) {
        return None;
    }

    // The header array param's two incomings: the preheader seed insert and the
    // body carry insert.
    let k_arrh = param_pos(ctx, header, arr_h)?;
    let mut seed_val = None;
    let mut carry = None; // (arr_b, stored_val, insert_index)
    for v in incoming(ctx, header, k_arrh) {
        let ValueId::Instruction(iid) = v else {
            return None;
        };
        let Mnemonic::Intrinsic(IntrinsicApp { id, args }) = ctx.get_insn(iid).mnemonic() else {
            return None;
        };
        if *id != insert_id {
            return None;
        }
        let [arr0, idx, val] = args[..] else {
            return None;
        };
        if let ValueId::BlockParam(_) = arr0 {
            // Carry: insert into a loop-carried array param.
            if carry.is_some() {
                return None;
            }
            carry = Some((arr0, val, idx));
        } else if literal(ctx, idx) == Some(0) {
            // Seed: insert the lane-0 value into a fresh (non-param) array.
            if seed_val.is_some() {
                return None;
            }
            seed_val = Some(val);
        } else {
            return None;
        }
    }
    let seed_val = seed_val?;
    let (arr_b, stored_val, ins_idx) = carry?;
    let body = param_parent(ctx, arr_b)?;
    if !matches!(ins_idx, ValueId::BlockParam(_)) || param_parent(ctx, ins_idx) != Some(body) {
        return None;
    }
    let index = ins_idx;

    // The body must be a pred of the header (the loop back-edge).
    let header_preds: HashSet<BlockId> = BasicBlock::from_id(ctx, header)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    if !header_preds.contains(&body) {
        return None;
    }

    // The single `at(arr_b, index-1)` reading the previous element (the carry).
    let mut prev = None;
    for insn in BasicBlock::from_id(ctx, body).iter() {
        let Mnemonic::Intrinsic(IntrinsicApp { id, args }) = insn.mnemonic() else {
            continue;
        };
        if *id != at_id || args.first() != Some(&arr_b) {
            continue;
        }
        if prev.is_some() {
            return None;
        }
        prev = Some((insn.id, args[1]));
    }
    let (prev_id, at_idx) = prev?;
    if !is_decrement(ctx, at_idx, index) {
        return None;
    }
    let prev_val = ValueId::Instruction(prev_id);

    // Original-element read: a second `at(l0_b, index)` on a *different*
    // loop-invariant array param `l0_b` (the whole-region snapshot `array_promote`
    // threads when the loop reads its own lane). Its presence turns the fold into a
    // scan over the original array `l0[1..]` instead of over `iota`.
    let mut elem = None;
    for insn in BasicBlock::from_id(ctx, body).iter() {
        let Mnemonic::Intrinsic(IntrinsicApp { id, args }) = insn.mnemonic() else {
            continue;
        };
        if *id != at_id || args.len() != 2 || args[0] == arr_b {
            continue;
        }
        let (l0_b, e_idx) = (args[0], args[1]);
        if !matches!(l0_b, ValueId::BlockParam(_)) || e_idx != index {
            continue;
        }
        if elem.is_some() {
            return None;
        }
        elem = Some((ValueId::Instruction(insn.id), l0_b));
    }
    // Resolve the snapshot's exit view (to slice `l0[1..]` at the store site). The
    // snapshot is loop-invariant, so it reaches the exit as an exit param whose
    // header-edge value is the header param feeding `l0_b`. v1 array-input support
    // is limited to the split shape (distinct body/exit blocks).
    let elem = match elem {
        Some((e_read, l0_b)) => {
            if body == header {
                return None; // rotated array-input not supported in v1
            }
            let [l0_h] = incoming(ctx, body, param_pos(ctx, body, l0_b)?)[..] else {
                return None;
            };
            if param_parent(ctx, l0_h) != Some(header) {
                return None;
            }
            let mut l0_exit = None;
            for p in BasicBlock::from_id(ctx, exit).params().map(|p| p.id()) {
                if incoming(ctx, exit, param_pos(ctx, exit, p)?)[..] == [l0_h] {
                    if l0_exit.is_some() {
                        return None;
                    }
                    l0_exit = Some(p);
                }
            }
            Some((e_read, l0_exit?))
        }
        None => None,
    };

    // Induction: the values feeding `index` start at a single literal `s` and step
    // by one on the back-edge. In the rotated shape (`body == header`) the index
    // param's own two incomings are the preheader init and the back-edge
    // increment; in the split shape `index` copies a header param, whose incomings
    // carry the init and increment.
    let k_index = param_pos(ctx, body, index)?;
    let feeds = if body == header {
        incoming(ctx, body, k_index)
    } else {
        let [hp] = incoming(ctx, body, k_index)[..] else {
            return None;
        };
        if param_parent(ctx, hp) != Some(header) {
            return None;
        }
        let k_hp = param_pos(ctx, header, hp)?;
        incoming(ctx, header, k_hp)
    };
    if !feeds.iter().any(|&v| is_increment(ctx, v, index)) {
        return None;
    }
    let inits: Vec<i64> = feeds
        .iter()
        .filter(|&&v| !is_increment(ctx, v, index))
        .filter_map(|&v| literal(ctx, v).map(|x| x as i64))
        .collect();
    let [index_start] = inits[..] else {
        return None;
    };

    Some(ScanMatch {
        exit,
        store_id,
        seed_val,
        stored_val,
        prev_val,
        index,
        index_start,
        count,
        elem_ty,
        elem,
    })
}

/// Rewrite a matched fill loop: outline the `(acc, index)` body and replace the
/// wide exit store with `store(ram, base <- concat(singleton(seed), scanl @body
/// seed iota(N-1)))`. The residual loop is left for `dce`.
fn apply(ctx: &mut Context, fid: FunctionId, m: &ScanMatch) -> bool {
    let index_ty = ctx.type_of(m.index);
    let i64_ty = ctx.types.get_or_make_int(8);
    let n1 = m.count - 1;

    let iota_id = IntrinsicId::from_name("iota").expect("iota registered");
    let singleton_id = IntrinsicId::from_name("singleton").expect("singleton registered");
    let concat_id = IntrinsicId::from_name("concat").expect("concat registered");

    let name = format!("{}_scan_body", Function::from_id(ctx, fid).name());

    // Two source shapes. When the body reads the region's original element
    // (`m.elem`), the scan ranges over the *original array* `l0[1..]` and the body
    // is `f(acc, x)` over that data element. Otherwise it is a pure generation and
    // the scan ranges over `iota(N-1)`, whose element is the loop index (no
    // original memory read — the property is then syntactic).
    let (body_fn, src) = match m.elem {
        Some((elem_read, l0_exit)) => {
            let esz = ctx.types.size_of(m.elem_ty);
            let src_arr_ty = ctx.types.get_or_make_array(m.elem_ty, n1);
            let Some(body_fn) = outline_scan_body(
                ctx,
                &name,
                m.stored_val,
                m.prev_val,
                m.index,
                Some(elem_read),
                m.index_start,
                m.elem_ty,
                index_ty,
                ScanElem::Data(m.elem_ty),
            ) else {
                return false;
            };
            // `l0[1..]` — the original elements at lanes `1..N`, a byte-slice of the
            // snapshot array from element 1 (length `N-1`).
            let slice = InstructionRef::from_mnemonic_with_type(
                ctx,
                Mnemonic::Range(qcode::value::insn::Range {
                    src: l0_exit,
                    start: esz,
                    size: n1 * esz,
                }),
                src_arr_ty,
            )
            .id;
            BasicBlock::from_id_mut(ctx, m.exit).insert_insn_before(m.store_id, slice);
            (body_fn, ValueId::Instruction(slice))
        }
        None => {
            let src_arr_ty = ctx.types.get_or_make_array(i64_ty, n1);
            let Some(body_fn) = outline_scan_body(
                ctx,
                &name,
                m.stored_val,
                m.prev_val,
                m.index,
                None,
                m.index_start,
                m.elem_ty,
                index_ty,
                ScanElem::Scalar(i64_ty),
            ) else {
                return false;
            };
            // `iota(N-1)` typed to the concrete `[i64; N-1]` (construction does not fold).
            let n1_const = ctx.get_const(n1 as u64, 8).id();
            let iota_id_insn = InstructionRef::from_mnemonic_with_type(
                ctx,
                Mnemonic::Intrinsic(IntrinsicApp {
                    id: iota_id,
                    args: vec![n1_const],
                }),
                src_arr_ty,
            )
            .id;
            BasicBlock::from_id_mut(ctx, m.exit).insert_insn_before(m.store_id, iota_id_insn);
            (body_fn, ValueId::Instruction(iota_id_insn))
        }
    };

    // scan(iota) → singleton(seed) → concat, all before the wide store.
    let full = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.exit));
        b.set_insert_point_before(m.store_id);
        let scan = b.push_scan(body_fn, m.seed_val, src, Vec::new()).id();
        let sing = b.push_intrinsic(singleton_id, vec![m.seed_val]).id();
        b.push_intrinsic(concat_id, vec![sing, scan]).id()
    };

    // Point the wide store at the folded array (leaving the loop's own array,
    // now dead, for `dce`).
    let mut sm = ctx.get_insn(m.store_id).mnemonic().clone();
    if let Mnemonic::Store(s) = &mut sm {
        s.src = full;
    }
    ctx.replace_instruction_mnemonic(m.store_id, sm);
    true
}

/// A recognized carry-free indexed-map fill loop (`l[j] = f(l[j], j)`), the
/// seedless dual of [`ScanMatch`]: no accumulator, every lane read from the
/// original snapshot and written once. Folds to `map @f (enumerate l0)`.
struct MapMatch {
    exit: BlockId,
    store_id: InstructionId,
    /// The body's per-lane result (`%next`), the map body's return value.
    stored_val: ValueId,
    /// The body induction parameter (`j`), read as the map index.
    index: ValueId,
    /// The `at(l0, j)` element-read value bound to the body's element input.
    elem_read: ValueId,
    /// The exit view of the whole-region snapshot to enumerate/map over.
    l0_exit: ValueId,
    /// The array element type and full length.
    elem_ty: qcode::types::TypeId,
    count: usize,
}

/// Recognize the carry-free `insert`/`at` map loop `array_promote` emits for an
/// indexed map over the original array.
fn try_match_map(ctx: &mut Context, fid: FunctionId) -> Option<MapMatch> {
    let insert_id = IntrinsicId::from_name("insert")?;
    let at_id = IntrinsicId::from_name("at")?;

    // Anchor: a single array-typed RAM store whose source is a block param.
    let mut candidates: Vec<(InstructionId, BlockId, ValueId)> = Vec::new();
    for block in Function::from_id(ctx, fid).iter() {
        let bid = block.id;
        for insn in block.iter() {
            let Mnemonic::Store(s) = insn.mnemonic() else {
                continue;
            };
            if matches!(
                qcode::space::Space::from_id(ctx, s.space).ty,
                qcode::space::SpaceType::Ram
            ) {
                candidates.push((insn.id, bid, s.src));
            }
        }
    }
    let mut anchor = None;
    for (id, bid, src) in candidates {
        if !matches!(src, ValueId::BlockParam(_)) {
            continue;
        }
        let src_ty = ctx.type_of(src);
        let Some((elem_ty, count)) = ctx.types.array_of(src_ty) else {
            continue;
        };
        if count == 0 {
            continue;
        }
        if anchor.is_some() {
            return None;
        }
        anchor = Some((id, bid, src, elem_ty, count));
    }
    let (store_id, exit, arr_src, elem_ty, count) = anchor?;

    let exit_preds: Vec<BlockId> = BasicBlock::from_id(ctx, exit)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    let [header] = exit_preds[..] else {
        return None;
    };
    let arr_h = if param_parent(ctx, arr_src) == Some(header) {
        arr_src
    } else {
        let k = param_pos(ctx, exit, arr_src)?;
        match incoming(ctx, exit, k)[..] {
            [v] => v,
            _ => return None,
        }
    };
    if param_parent(ctx, arr_h) != Some(header) {
        return None;
    }

    // The header array param's incomings: the body carry `insert(arr_b, j, next)`
    // and a non-insert initial array (the zero seed). A `map` has no accumulator,
    // so there is exactly one insert and no lane-0 seed insert.
    let k_arrh = param_pos(ctx, header, arr_h)?;
    let mut carry = None;
    let mut saw_init = false;
    for v in incoming(ctx, header, k_arrh) {
        match v {
            ValueId::Instruction(iid) => {
                let Mnemonic::Intrinsic(IntrinsicApp { id, args }) = ctx.get_insn(iid).mnemonic()
                else {
                    return None;
                };
                if *id != insert_id {
                    return None;
                }
                let [arr0, idx, val] = args[..] else {
                    return None;
                };
                if !matches!(arr0, ValueId::BlockParam(_)) {
                    return None;
                }
                if carry.is_some() {
                    return None;
                }
                carry = Some((arr0, val, idx));
            }
            // The zero-initialized array literal.
            ValueId::Bytes(_) => saw_init = true,
            _ => return None,
        }
    }
    if !saw_init {
        return None;
    }
    let (arr_b, stored_val, ins_idx) = carry?;
    let body = param_parent(ctx, arr_b)?;
    if body == header {
        return None; // rotated map deferred to v1's split-shape support
    }
    if !matches!(ins_idx, ValueId::BlockParam(_)) || param_parent(ctx, ins_idx) != Some(body) {
        return None;
    }
    let index = ins_idx;

    let header_preds: HashSet<BlockId> = BasicBlock::from_id(ctx, header)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    if !header_preds.contains(&body) {
        return None;
    }

    // The body reads the original element `at(l0_b, j)` on a loop-invariant param
    // `l0_b != arr_b`, and reads no accumulator (`at(arr_b, …)` would be a scan).
    let mut elem = None;
    for insn in BasicBlock::from_id(ctx, body).iter() {
        let Mnemonic::Intrinsic(IntrinsicApp { id, args }) = insn.mnemonic() else {
            continue;
        };
        if *id != at_id || args.len() != 2 {
            continue;
        }
        if args[0] == arr_b {
            return None; // an accumulator read — this is a scan, not a map
        }
        if !matches!(args[0], ValueId::BlockParam(_)) || args[1] != index {
            continue;
        }
        if elem.is_some() {
            return None;
        }
        elem = Some((ValueId::Instruction(insn.id), args[0]));
    }
    let (elem_read, l0_b) = elem?;

    // Induction: `j` starts at 0 and steps by 1.
    let k_index = param_pos(ctx, body, index)?;
    let [hp] = incoming(ctx, body, k_index)[..] else {
        return None;
    };
    if param_parent(ctx, hp) != Some(header) {
        return None;
    }
    let k_hp = param_pos(ctx, header, hp)?;
    let feeds = incoming(ctx, header, k_hp);
    if !feeds.iter().any(|&v| is_increment(ctx, v, index)) {
        return None;
    }
    let inits: Vec<i64> = feeds
        .iter()
        .filter(|&&v| !is_increment(ctx, v, index))
        .filter_map(|&v| literal(ctx, v).map(|x| x as i64))
        .collect();
    if inits != [0] {
        return None; // v1 maps tile [0, N) from index 0
    }

    // The exit view of the snapshot to enumerate over.
    let [l0_h] = incoming(ctx, body, param_pos(ctx, body, l0_b)?)[..] else {
        return None;
    };
    if param_parent(ctx, l0_h) != Some(header) {
        return None;
    }
    let mut l0_exit = None;
    for p in BasicBlock::from_id(ctx, exit).params().map(|p| p.id()) {
        if incoming(ctx, exit, param_pos(ctx, exit, p)?)[..] == [l0_h] {
            if l0_exit.is_some() {
                return None;
            }
            l0_exit = Some(p);
        }
    }
    let l0_exit = l0_exit?;

    Some(MapMatch {
        exit,
        store_id,
        stored_val,
        index,
        elem_read,
        l0_exit,
        elem_ty,
        count,
    })
}

/// Rewrite a matched map loop to `store(ram, base <- map @f (enumerate l0))`.
fn apply_map(ctx: &mut Context, fid: FunctionId, m: &MapMatch) -> bool {
    use super::loop_to_map::outline_tupled;

    let enum_id = IntrinsicId::from_name("enumerate").expect("enumerate registered");
    let l0_ty = ctx.types.get_or_make_array(m.elem_ty, m.count);
    let enum_ty = enum_id.desc().result_type(&mut ctx.types, &[l0_ty]);
    let Some((tuple_ty, _)) = ctx.types.array_of(enum_ty) else {
        return false;
    };

    let name = format!("{}_map_body", Function::from_id(ctx, fid).name());
    let Some(body_fn) = outline_tupled(
        ctx,
        &name,
        m.stored_val,
        m.index,
        m.elem_read,
        tuple_ty,
    ) else {
        return false;
    };

    let out = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.exit));
        b.set_insert_point_before(m.store_id);
        let en = b.push_intrinsic(enum_id, vec![m.l0_exit]).id();
        b.push_map(body_fn, en, Vec::new()).id()
    };
    let mut sm = ctx.get_insn(m.store_id).mnemonic().clone();
    if let Mnemonic::Store(s) = &mut sm {
        s.src = out;
    }
    ctx.replace_instruction_mnemonic(m.store_id, sm);
    true
}

impl FunctionPass for LoopToScan {
    const NAME: &'static str = "loop_to_scan";

    fn description(&self) -> &'static str {
        "Fold the value-carried insert/at fill loop from array_promote into a scanl"
    }

    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        if let Some(m) = try_match(ctx, fun_id) {
            return Ok(apply(ctx, fun_id, &m));
        }
        // The carry-free indexed map (`l[j] = f(l[j], j)`) → `map @f (enumerate l0)`.
        if let Some(m) = try_match_map(ctx, fun_id) {
            return Ok(apply_map(ctx, fun_id, &m));
        }
        Ok(false)
    }
}

crate::register_function_pass!(LoopToScan);
