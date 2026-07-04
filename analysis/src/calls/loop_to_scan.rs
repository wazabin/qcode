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
        BasicBlock, BlockId, Function, FunctionId, InstructionRef, ValueId,
        insn::{Branch, InstructionId, IntrinsicApp, IntrinsicId, Mnemonic},
    },
};

use super::carried_array::{
    classify_body_reads, exit_view, find_carried_array, incoming, is_increment, literal,
    param_parent, param_pos,
};
use super::loop_to_map::{ScanElem, outline_scan_body};
use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct LoopToScan;

/// A recognized `insert`/`at` fill loop (see the module docs).
struct ScanMatch {
    /// The loop-exit block holding the wide store.
    exit: BlockId,
    /// The loop header (guard block) and body; `header == body` in the rotated
    /// (do-while) shape. Used to delete the residual loop once it is private.
    header: BlockId,
    body: BlockId,
    /// The loop-carried array as seen by the exit block (the store's source).
    /// Every other exit use of it (e.g. `at(arr, k)` from `array_promote`'s
    /// exit-load rewrite) is redirected to the folded array so the residual
    /// loop actually dies.
    arr_exit: ValueId,
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
    /// (`at(arr, index)` on the single carried array) — the loop is a scan over the
    /// original array rather than a pure generation over `iota`. Holds
    /// `(element-read value, original-array value to slice)`, where the array is the
    /// seed insert's base operand (the preheader wide `Load`); the scan ranges over
    /// `l0[1..]`.
    elem: Option<(ValueId, ValueId)>,
}

/// Recognize the `insert`/`at` fill loop in `fid`.
fn try_match(ctx: &mut Context, fid: FunctionId) -> Option<ScanMatch> {
    // Anchor on the shared carried-array matcher, then narrow to the scan shape:
    // a lane-0 accumulator seed insert must exist (`ca.seed`).
    let ca = find_carried_array(ctx, fid)?;
    let (seed_val, seed_arr0) = ca.seed?;
    let header = ca.header;
    let body = ca.body;
    let index = ca.index;
    let elem_ty = ca.elem_ty;
    let count = ca.count;
    let stored_val = ca.stored_val;

    // The loop body carries the array back to the header; the guard block's other
    // successor is the loop exit. `header`'s terminator is the `CBranch` guard
    // (rotated: it lives on the body/header itself; split: on the distinct guard).
    let hterm = BasicBlock::from_id(ctx, header).iter().last()?;
    let Mnemonic::CBranch(cb) = hterm.mnemonic() else {
        return None;
    };
    let (sb, fb) = (cb.success_block, cb.failure_block);
    // The back-edge block (loop body) is a predecessor of the header; the exit is
    // the guard's other successor.
    let header_preds_set: HashSet<BlockId> = BasicBlock::from_id(ctx, header)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    let exit = if header_preds_set.contains(&sb) && !header_preds_set.contains(&fb) {
        fb
    } else if header_preds_set.contains(&fb) && !header_preds_set.contains(&sb) {
        sb
    } else {
        return None;
    };

    // The array value as the exit block sees it: an exit pass-through param that
    // copies `arr_h` on the header→exit edge, or `arr_h` itself when gvn coalesced
    // that trivial pass-through away (then the exit reads the header param directly).
    let arr_src = exit_view(ctx, &ca, exit);

    // The body's `at(arr_b, ·)` reads on the single carried array: the carry
    // `at(arr_b, index-1)` (always present) and, for an original-array scan, the
    // own-lane original read `at(arr_b, index)`.
    let reads = classify_body_reads(ctx, &ca)?;
    let prev_id = reads.prev?;
    let prev_val = ValueId::Instruction(prev_id);
    let elem_read = reads.own;

    // An own-lane original read turns the fold into a scan over the original array
    // `l0[1..]`. The original array is the seed insert's base operand `seed_arr0`,
    // defined in the preheader (so it dominates the exit rewrite site). It must be a
    // wide `Load` of the region — a `splat` base would mean literal zero, not
    // `l0[i]`. v1 array-input support is split-shape only.
    let elem = match elem_read {
        Some(e_read) => {
            if body == header {
                return None; // rotated array-input not supported in v1
            }
            let ValueId::Instruction(a0) = seed_arr0 else {
                return None;
            };
            if !matches!(ctx.get_insn(a0).mnemonic(), Mnemonic::Load(_)) {
                return None;
            }
            Some((e_read, seed_arr0))
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
        header,
        body,
        arr_exit: arr_src,
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

    // Anchor every new instruction before the exit block's *first* instruction,
    // not just before the wide store: `array_promote` may have rewritten other
    // exit loads to `at(arr_exit, k)` earlier in the block, and those get
    // redirected to the folded array below — which must therefore dominate them.
    let Some(anchor) = BasicBlock::from_id(ctx, m.exit).iter().next().map(|i| i.id) else {
        return false;
    };
    // Pre-existing exit instructions whose `arr_exit` uses are redirected.
    let preexisting: Vec<InstructionId> = BasicBlock::from_id(ctx, m.exit)
        .iter()
        .map(|i| i.id)
        .collect();

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
            BasicBlock::from_id_mut(ctx, m.exit).insert_insn_before(anchor, slice);
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
            BasicBlock::from_id_mut(ctx, m.exit).insert_insn_before(anchor, iota_id_insn);
            (body_fn, ValueId::Instruction(iota_id_insn))
        }
    };

    // scan(iota) → singleton(seed) → concat, ahead of every exit use.
    let full = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.exit));
        b.set_insert_point_before(anchor);
        let scan = b.push_scan(body_fn, m.seed_val, src, Vec::new()).id();
        let sing = b.push_intrinsic(singleton_id, vec![m.seed_val]).id();
        b.push_intrinsic(concat_id, vec![sing, scan]).id()
    };

    // Redirect every exit use of the loop-carried array — the wide store and any
    // `at(arr, k)` reads `array_promote` left for exit loads — to the folded
    // array, leaving the loop's own array dead for `dce`.
    for id in preexisting {
        let mut mn = ctx.get_insn(id).mnemonic().clone();
        mn.replace_value(m.arr_exit, full);
        ctx.replace_instruction_mnemonic(id, mn);
    }

    // Delete the residual loop when it is now wholly private (mirrors
    // `loop_to_map`'s deletable path): the exit carries no params and every
    // value the loop defines is used only inside it. Then the loop computes
    // nothing observable — reroute the single preheader straight to the exit
    // and delete the loop blocks. Otherwise (a loop value threaded past the
    // exit, e.g. a clobbered register in the return envelope) leave it for
    // later dead-arg/dce rounds.
    let loop_blocks: Vec<BlockId> = if m.header == m.body {
        vec![m.body]
    } else {
        vec![m.header, m.body]
    };
    let in_loop = |ctx: &Context, v: ValueId| {
        ctx.users(v).iter().all(|&u| {
            ctx.get_insn(u)
                .parent()
                .is_some_and(|b| loop_blocks.contains(&b.id))
        })
    };
    let private = loop_blocks.iter().all(|&blk| {
        let b = BasicBlock::from_id(ctx, blk);
        b.params().all(|p| in_loop(ctx, p.id()))
            && b.iter().all(|i| in_loop(ctx, ValueId::Instruction(i.id)))
    });
    let defined_in_loop = |ctx: &Context, v: ValueId| match v {
        ValueId::BlockParam(_) => param_parent(ctx, v).is_some_and(|b| loop_blocks.contains(&b)),
        ValueId::Instruction(id) => ctx
            .get_insn(id)
            .parent()
            .is_some_and(|b| loop_blocks.contains(&b.id)),
        _ => false,
    };
    // Each exit param (a now-dead array pass-through the later gvn would have
    // coalesced) must be re-fed from a preheader-available value: its
    // header-edge incoming directly if loop-invariant, or — when it copies a
    // loop param — that param's own loop-invariant (preheader) incoming.
    let exit_args: Option<Vec<ValueId>> = BasicBlock::from_id(ctx, m.exit)
        .params()
        .map(|p| p.id())
        .collect::<Vec<_>>()
        .into_iter()
        .map(|p| {
            let k = param_pos(ctx, m.exit, p)?;
            let [v] = incoming(ctx, m.exit, k)[..] else {
                return None;
            };
            if !defined_in_loop(ctx, v) {
                return Some(v);
            }
            if !matches!(v, ValueId::BlockParam(_)) {
                return None;
            }
            let kv = param_pos(ctx, m.header, v)?;
            let init: Vec<ValueId> = incoming(ctx, m.header, kv)
                .into_iter()
                .filter(|&w| !defined_in_loop(ctx, w))
                .collect();
            match init[..] {
                [w] => Some(w),
                _ => None,
            }
        })
        .collect();
    if private && let Some(exit_args) = exit_args {
        let preheaders: Vec<BlockId> = BasicBlock::from_id(ctx, m.header)
            .predecessors()
            .map(|(_, p)| p)
            .filter(|p| !loop_blocks.contains(p))
            .collect();
        if let [preheader] = preheaders[..] {
            if let Some(term_id) = BasicBlock::from_id(ctx, preheader)
                .iter()
                .last()
                .map(|t| t.id)
            {
                ctx.replace_instruction_mnemonic(
                    term_id,
                    Mnemonic::Branch(Branch {
                        target: m.exit,
                        args: exit_args,
                    }),
                );
                ctx.add_cfg_edge(preheader, m.exit);
            }
            for &blk in &loop_blocks {
                BasicBlock::from_id_mut(ctx, blk).delete(fid);
            }
        }
    }
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
    let Some(body_fn) = outline_tupled(ctx, &name, m.stored_val, m.index, m.elem_read, tuple_ty)
    else {
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

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use crate::mem::array_promote::ArrayPromote;
    use crate::test_util::run_function_pass;

    // The exact seam this rewrite touches: `array_promote` promotes the seeded
    // prefix sum `out[0]=seed; out[i]=out[i-1]+l[i]` into a single carried array
    // (original init + carry read `at(arr,i-1)` + own-lane read `at(arr,i)`), and
    // `loop_to_scan` must then fold it to a `scanl` over the original array. A
    // recognizer mismatch across the two passes is invisible to their per-pass
    // tests, so exercise the chain directly.
    #[test]
    fn array_promote_then_scan_folds_prefix_sum() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn prefix:
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
            run_function_pass::<ArrayPromote>(&mut ctx, prefix).unwrap(),
            "array_promote should promote the prefix sum"
        );
        assert!(
            run_function_pass::<LoopToScan>(&mut ctx, prefix).unwrap(),
            "loop_to_scan should fold the promoted single-array loop"
        );
        let ir = format!("{}", Function::from_id(&ctx, prefix));
        assert!(
            ir.contains("scanl"),
            "the promoted loop should fold to a scanl over the original array: {ir}"
        );
    }
}
