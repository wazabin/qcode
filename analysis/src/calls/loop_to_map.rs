//! Loop-to-map: recognize a total element-wise array loop on the shared
//! carried-array form (see [`super::carried_array`], produced by
//! [`array_promote`](crate::mem::array_promote)) and rewrite it as a single
//! [`Map`](qcode::value::insn::Map) over the array value, outlining the
//! per-element body into a fresh pure function.
//!
//! This file owns two pattern families over that form: the total-map recognizer
//! ([`try_match`]/[`apply`]) and the bounded NUL-scan "strlen" recognizers
//! ([`try_match_strlen`] on the `at`-form snapshot, and the raw-`char*` Layer 2).
//! Accumulator scans live in the sibling [`super::loop_to_scan`].
//!
//! The per-element bodies are outlined into fresh pure functions by the shared
//! [`super::outline`] machinery.

use rustc_hash::FxHashSet as HashSet;

use qcode::{
    builder::Builder,
    context::Context,
    space::{Space, SpaceId, SpaceType},
    value::{
        BasicBlock, BlockId, Function, FunctionId, ValueId,
        insn::{
            Binop, Branch, CBranch, InstructionId, IntBinop, IntrinsicApp, IntrinsicId, Mnemonic,
            Unary, Unop,
        },
    },
};

use super::carried_array::{CarriedArray, classify_body_reads, exit_view, find_carried_array};
use super::outline::{outline_expression, outline_tupled, pure_slice};
use crate::loop_info::{
    cbranch_exit, delete_private_loop, incoming, is_increment, is_loop_private, literal,
    param_parent, param_pos,
};
use crate::{Pass, PipelineEnv};

// ===========================================================================
// Total-map recognizer
// ===========================================================================
//
// After the single-carried-array `array_promote`, an element-wise fill loop over
// a bounded region threads one array-typed header param through the loop with
// `insert`/`at`, initialized from a preheader wide `Load` (in-place map over the
// original data) or a `splat(0, N)` (pure generation), and stored wide at the
// exit:
//
//   <entry @base>
//       %arr0 = load(ram:N*esz, @base)   // reads-original init  — OR —
//       %arr0 = splat(0, N)              // pure-generation init
//       goto <head @i=0 @arr=%arr0>;
//   <head @i @arr:[e;N]>
//       if @i == N goto <exit @arr> else goto <body @j=@i @a=@arr>;
//   <body @j @a>
//       %x   = at(@a, @j)                // optional own-lane original read
//       %v   = body(@j, %x)              // pure
//       %ins = insert(@a, @j, %v)        // carry
//       goto <head @i=@j+1 @arr=%ins>;
//   <exit @arr>  store(ram:N*esz, @base <- @arr); ...
//
// A map is exactly a carried array *without an accumulator*: the body writes lane
// `@j` but never reads lane `@j-1` (that carry read is the scan shape,
// [`loop_to_scan`](super::loop_to_scan)'s pattern). This recognizer shares the
// carried-array matcher ([`super::carried_array`]) with `loop_to_scan` and is
// mutually exclusive with it by the single test `ca.seed.is_none()`.
//
// It rewrites the exit view of the carried array to a `map`: when the body reads
// only the element it is `map(body, arr0)` with a unary `body(elem)`; when it
// also reads the index it is `map(body, enumerate(arr0))` with `body(tuple)`
// unpacking the `(index, elem)` pair. The dead loop is deleted when wholly
// private, else left for later cleanup; the function stays pure and correct.
//
// v1 promotes a *single* region, so the old two-buffer `dst[i] = body(src[i])`
// copy no longer surfaces as a clean map (the second-region read makes the body
// impure for outlining and the match bails) — an accepted generality regression.

/// A recognized total-map loop built on the shared carried-array matcher.
struct MapMatch {
    /// The carried array and its loop structure (header/body/arr/index/…).
    ca: CarriedArray,
    /// The loop exit block (holds the wide store / return-envelope pack).
    exit: BlockId,
    /// The carried array as the exit block sees it (the rewrite target).
    arr_exit: ValueId,
    /// The `at(arr_b, index)` own-lane original read, if the body reads its lane.
    elem_read: Option<ValueId>,
    /// The initial array the map ranges over: the preheader wide `Load` (reads
    /// original) or the `splat(0, N)` (pure generation).
    init_arr: ValueId,
    /// The RAM store consuming `arr_exit`, when the region is real memory (else
    /// the exit uses are a private-shadow return envelope and `arr_exit`'s uses
    /// are simply replaced).
    store_id: Option<InstructionId>,
    /// Whether the loop is wholly private and can be deleted after the rewrite
    /// (exit carries only the array pass-through, every loop value used only in
    /// the loop). Otherwise the residual loop is left for later dce.
    deletable: bool,
}

fn is_temp(ctx: &Context, s: SpaceId) -> bool {
    matches!(Space::from_id(ctx, s).ty, SpaceType::Temporary)
}

/// Match the canonical total-map loop in `fid` on the shared carried-array form,
/// or `None` for any other shape (the function is then left untouched).
fn try_match(ctx: &mut Context, fid: FunctionId) -> Option<MapMatch> {
    // A map is a carried array with no lane-0 accumulator seed (a seed is the scan
    // shape, `loop_to_scan`'s pattern — the two are mutually exclusive here) and a
    // real initial array to range over.
    let ca = find_carried_array(ctx, fid)?;
    if ca.seed.is_some() {
        return None;
    }
    let init_arr = ca.init?;

    // No accumulator carry read (`at(arr_b, index-1)`); the own-lane original read
    // (`at(arr_b, index)`) is optional.
    let reads = classify_body_reads(ctx, &ca)?;
    if reads.prev.is_some() {
        return None;
    }
    let elem_read = reads.own;

    // Split shape only: the rewrite and deletion address header and body
    // separately. A rotated (`header == body`) map is deferred (mirrors
    // `loop_to_scan`'s v1 rejection).
    if ca.header == ca.body {
        return None;
    }

    // Totality is `array_promote`'s coverage proof: the carried array has exactly
    // `count` lanes and the promotion writes every one. We only confirm the index
    // the body carries inits at 0 and steps by +1 on the back-edge (the promoted
    // guard is an equality `@i == N`, which `value_range` cannot narrow, so we do
    // not lean on it here — mirroring `loop_to_scan`).
    let k_index = param_pos(ctx, ca.body, ca.index)?;
    let feeds = {
        let [hp] = incoming(ctx, ca.body, k_index)[..] else {
            return None;
        };
        if param_parent(ctx, hp) != Some(ca.header) {
            return None;
        }
        let k_hp = param_pos(ctx, ca.header, hp)?;
        incoming(ctx, ca.header, k_hp)
    };
    if !feeds.iter().any(|&v| is_increment(ctx, v, ca.index)) {
        return None;
    }
    let inits: Vec<i64> = feeds
        .iter()
        .filter(|&&v| !is_increment(ctx, v, ca.index))
        .filter_map(|&v| literal(ctx, v).map(|x| x as i64))
        .collect();
    if inits != [0] {
        return None; // v1 maps tile [0, N) from index 0
    }

    // Exit discovery: the header terminator is the loop guard; the exit is the
    // successor that is not itself a header predecessor.
    let (exit, _stay) = cbranch_exit(ctx, ca.header)?;
    let arr_exit = exit_view(ctx, &ca, exit);

    // When the body reads its own original lane, the map ranges over the *original*
    // array, so the init must be the preheader wide `Load` — reading lane `j` of a
    // `splat`-init region is legal but yields literal zero, not `l0[j]`, so matching
    // it as an element read would be wrong (mirrors `loop_to_scan`'s scan check).
    if elem_read.is_some() {
        let ValueId::Instruction(iid) = init_arr else {
            return None;
        };
        if !matches!(ctx.get_insn(iid).mnemonic(), Mnemonic::Load(_)) {
            return None;
        }
    }

    // Consumer: a RAM store of the carried array's exit view (real memory). When
    // absent (private argpromote shadow, wide temp store already dce'd) the exit
    // uses of `arr_exit` are the return envelope — the rewrite just replaces them.
    let store_id = BasicBlock::from_id(ctx, exit)
        .iter()
        .find_map(|i| match i.mnemonic() {
            Mnemonic::Store(s)
                if matches!(Space::from_id(ctx, s.space).ty, SpaceType::Ram)
                    && s.src == arr_exit =>
            {
                Some(i.id)
            }
            _ => None,
        });

    // Deletable iff wholly private: every value the loop defines is used only inside
    // it (the exit's array pass-through is re-fed from the preheader init in `apply`).
    let loop_blocks = [ca.header, ca.body];
    let deletable = is_loop_private(ctx, &loop_blocks);

    Some(MapMatch {
        ca,
        exit,
        arr_exit,
        elem_read,
        init_arr,
        store_id,
        deletable,
    })
}

/// Does the per-element body actually read the loop index? `enumerate` is only
/// worth inserting when it does; a value-only body maps directly over the array.
fn body_uses_index(
    ctx: &Context,
    stored_val: ValueId,
    index: ValueId,
    elem_read: Option<ValueId>,
) -> bool {
    if stored_val == index {
        return true;
    }
    let mut inputs = vec![index];
    inputs.extend(elem_read);
    match pure_slice(ctx, stored_val, &inputs) {
        Some(slice) => slice
            .iter()
            .any(|&iid| ctx.get_insn(iid).mnemonic().args().contains(&index)),
        None => false,
    }
}

/// Rewrite a matched loop: outline the per-element body, replace the exit view of
/// the carried array with a `map`, then (when wholly private) reroute the
/// preheader past the loop and delete the dead loop blocks.
///
/// The map source depends on whether the body reads the index. A value-only body
/// maps over the array directly — `map(body, arr0)`, `body(elem)`. An index-aware
/// body maps over `enumerate(arr0)`, whose element is the `(index, elem)` tuple —
/// `map(body, enumerate(arr0))`, `body(tuple)` unpacking it. Returns `false` if
/// the body is not a closed pure expression of `(index, element?)` (then nothing
/// is changed — outlining is all-or-nothing and runs before any rewrite).
fn apply(ctx: &mut Context, fid: FunctionId, m: &MapMatch) -> bool {
    let name = format!("{}_map_body", Function::from_id(ctx, fid).name());
    let enum_id = IntrinsicId::from_name("enumerate").expect("enumerate registered");
    let uses_index = body_uses_index(ctx, m.ca.stored_val, m.ca.index, m.elem_read);

    // Outline the body before any rewrite, so a non-closed body leaves the loop
    // untouched.
    let body_fn = if uses_index {
        let arr_ty = ctx.type_of(m.init_arr);
        let enum_ty = enum_id.desc().result_type(&mut ctx.types, &[arr_ty]);
        let Some((tuple_ty, _)) = ctx.types.array_of(enum_ty) else {
            return false;
        };
        match outline_tupled(
            ctx,
            &name,
            m.ca.stored_val,
            m.ca.index,
            m.elem_read,
            tuple_ty,
        ) {
            Some(f) => f,
            None => return false,
        }
    } else {
        // A value-only body needs an element to map over; without an own-lane read
        // there is nothing to bind (a pure constant fill is out of v1 scope).
        let Some(elem) = m.elem_read else {
            return false;
        };
        match outline_expression(ctx, &name, m.ca.stored_val, &[elem]) {
            Some(f) => f,
            None => return false,
        }
    };

    // Build `map(body, enumerate(arr0))` (index-aware) or `map(body, arr0)`
    // (value-only) ahead of the consumer, then forward the exit view to it.
    let anchor = m
        .store_id
        .or_else(|| BasicBlock::from_id(ctx, m.exit).iter().next().map(|i| i.id));
    let map_val = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.exit));
        if let Some(at) = anchor {
            b.set_insert_point_before(at);
        }
        let src = if uses_index {
            b.push_intrinsic(enum_id, vec![m.init_arr]).id()
        } else {
            m.init_arr
        };
        b.push_map(body_fn, src, Vec::new()).id()
    };
    ctx.replace_all_uses_with(m.arr_exit, map_val);

    // Delete the residual loop when wholly private (mirrors `loop_to_scan::apply`):
    // reroute the single preheader straight to the exit, re-feeding each exit param
    // from a preheader-available value, then delete the loop blocks.
    let loop_blocks = [m.ca.header, m.ca.body];
    let defined_in_loop = |ctx: &Context, v: ValueId| match v {
        ValueId::BlockParam(_) => param_parent(ctx, v).is_some_and(|b| loop_blocks.contains(&b)),
        ValueId::Instruction(id) => ctx
            .get_insn(id)
            .parent()
            .is_some_and(|b| loop_blocks.contains(&b.id)),
        _ => false,
    };
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
            let kv = param_pos(ctx, m.ca.header, v)?;
            let init: Vec<ValueId> = incoming(ctx, m.ca.header, kv)
                .into_iter()
                .filter(|&w| !defined_in_loop(ctx, w))
                .collect();
            match init[..] {
                [w] => Some(w),
                _ => None,
            }
        })
        .collect();
    if m.deletable
        && let Some(exit_args) = exit_args
    {
        let preheaders: Vec<BlockId> = BasicBlock::from_id(ctx, m.ca.header)
            .predecessors()
            .map(|(_, p)| p)
            .filter(|p| !loop_blocks.contains(p))
            .collect();
        if let [preheader] = preheaders[..] {
            delete_private_loop(ctx, fid, preheader, &loop_blocks, m.exit, exit_args);
        }
    }
    true
}

/// Recognize total-map loops across all pure functions, rewriting each to a
/// `map`. Returns `true` if anything changed.
pub(crate) fn recognize_total_maps(ctx: &mut Context) -> bool {
    let fids: Vec<FunctionId> = ctx.function_ids();
    let mut changed = false;
    for fid in fids {
        if !Function::from_id(ctx, fid).is_pure() {
            continue;
        }
        if let Some(m) = try_match(ctx, fid) {
            changed |= apply(ctx, fid, &m);
        }
    }
    changed
}

// ===========================================================================
// strlen recognizer — len(take_while(arr))
// ===========================================================================
//
// A bounded NUL-scan over a snapshot array — `while (arr[i]) i++;` capped by the
// snapshot bound `N` — counts the leading nonzero bytes. That count is exactly
// `len(take_while(arr))`: `take_while` truncates the array at its first zero, and
// `len` of the resulting list is its data-dependent length. The recognizer proves
// the loop is such a scan and rewrites the escaping count to `len(take_while(@arr))`
// — the projectable form — leaving the now-dead scan for DCE.
//
// Shape (carried-array form, produced by `array_reads` from an argpromote
// read-only snapshot): a read-only scan whose lane read is `at(@arr, i)` and whose
// exit is governed by the loaded byte rather than a static index bound:
//
//   <entry @arr:[i8;N] ...>  goto <head @i=0>;
//   <head @i>   %b = at(@arr, @i);                     // lane read
//               if %b != 0 goto <body> else goto <exit @i>;   // NUL test
//   <body>      goto <head @i=@i+1>;
//   <exit @count>  ... uses of @count = strlen ...
//
// The lane byte governs the loop exit (the take_while predicate), and the index
// carried out on the exit edge is the scan length. v1: byte lanes, snapshot-backed
// source (the array value `@arr`); the unbounded raw-`char*` case is Layer 2.

/// A recognized bounded NUL-scan whose escaping count is `len(take_while(arr))`.
struct StrlenMatch {
    /// The `Array` snapshot scanned (a root `[i8;N]` param).
    arr: ValueId,
    /// The exit-block parameter holding the scan length (its uses become `len`).
    count_param: ValueId,
    /// The loop header (carries the induction param, holds the `at` and NUL test).
    header_block: BlockId,
    /// The loop body (the per-iteration increment).
    body_block: BlockId,
    /// The exit block (where `len`/`take_while` are inserted; holds `count_param`).
    exit_block: BlockId,
    /// The header's sole out-of-loop predecessor (rerouted to the exit on delete).
    preheader: BlockId,
    /// Whether the loop is wholly private and can be deleted (see [`MapMatch`]).
    deletable: bool,
}

/// Whether `cond` is true exactly when `elem` is **nonzero** (`Some(true)`, a
/// "continue while nonzero" guard), true exactly when `elem` is **zero**
/// (`Some(false)`), or not a zero-test of `elem` at all (`None`). Handles the bare
/// byte used as a predicate, `elem != 0` / `elem == 0` (either operand order), and
/// a `BoolNot` wrapper.
fn nonzero_polarity(ctx: &Context, cond: ValueId, elem: ValueId) -> Option<bool> {
    if cond == elem {
        return Some(true); // the raw byte as a bool: true ⟺ nonzero
    }
    let ValueId::Instruction(id) = cond else {
        return None;
    };
    match ctx.get_insn(id).mnemonic() {
        Mnemonic::Binop(b) => {
            let zero_test = (b.lhs == elem && literal(ctx, b.rhs) == Some(0))
                || (b.rhs == elem && literal(ctx, b.lhs) == Some(0));
            if !zero_test {
                return None;
            }
            match b.op {
                Binop::Int(IntBinop::NotEqual) => Some(true),
                Binop::Int(IntBinop::Equal) => Some(false),
                _ => None,
            }
        }
        Mnemonic::Unop(Unary {
            op: Unop::BoolNot,
            src,
        }) => nonzero_polarity(ctx, *src, elem).map(|p| !p),
        _ => None,
    }
}

/// Match a bounded NUL-scan in `fid`, or `None` for any other shape.
fn try_match_strlen(ctx: &Context, fid: FunctionId) -> Option<StrlenMatch> {
    let at_id = IntrinsicId::from_name("at")?;
    let insert_id = IntrinsicId::from_name("insert")?;
    let root_params: Vec<ValueId> = Function::from_id(ctx, fid)
        .root()?
        .params()
        .map(|p| p.id())
        .collect();
    let is_root = |v: ValueId| root_params.contains(&v);

    // The lane read is the sole `at(@arr, idx)` over a root `[i8;N]` param. More than
    // one `at` on that array (or any `insert` into it) means the loop is not a plain
    // read-only NUL scan, so bail.
    let mut lane: Option<(InstructionId, ValueId, ValueId)> = None; // (at id, arr, idx)
    for block in Function::from_id(ctx, fid).iter() {
        for insn in block.iter() {
            let Mnemonic::Intrinsic(IntrinsicApp { id, args }) = insn.mnemonic() else {
                continue;
            };
            if *id == insert_id && args.first().is_some_and(|&a| is_root(a)) {
                return None; // a write into a snapshot — not read-only
            }
            if *id != at_id || args.len() != 2 || !is_root(args[0]) {
                continue;
            }
            let byte_array = ctx
                .stored_type_of(args[0])
                .and_then(|t| ctx.types.array_of(t))
                .is_some_and(|(elem, _)| ctx.types.size_of(elem) == 1);
            if !byte_array {
                continue;
            }
            if lane.is_some() {
                return None; // more than one lane read — not the canonical scan
            }
            lane = Some((insn.id, args[0], args[1]));
        }
    }
    let (at_insn, arr, index) = lane?;
    let elem_val = ValueId::Instruction(at_insn);

    // The index is a header param initialised to 0 and stepped by +1 — so it counts
    // iterations from 0. (Unlike the map recognizer there is no static upper bound:
    // termination is the NUL test below, and the snapshot bound `N` caps it.)
    let ValueId::BlockParam(pid) = index else {
        return None;
    };
    let header = ctx.values.block_params[pid].parent?;
    // The lane read must live in the header: the NUL test that governs the loop
    // reads it there, and the count is the index at that test.
    if ctx.get_insn(at_insn).parent().map(|b| b.id) != Some(header) {
        return None;
    }
    let k = BasicBlock::from_id(ctx, header)
        .params()
        .position(|p| p.id() == index)?;
    // The index must start at 0 on *every* entry edge and step by +1 on the
    // back-edge: each non-increment (preheader) incoming has to be the literal 0,
    // or some entry could start the count off-zero and the rewrite would be wrong.
    let incoming = incoming(ctx, header, k);
    let inits: Vec<ValueId> = incoming
        .iter()
        .copied()
        .filter(|&v| !is_increment(ctx, v, index))
        .collect();
    if inits.is_empty()
        || !inits.iter().all(|&v| literal(ctx, v) == Some(0))
        || !incoming.iter().any(|&v| is_increment(ctx, v, index))
    {
        return None;
    }

    // Termination: the header ends in a CBranch governed by the loaded byte's
    // zero-test. Its continue edge re-enters the loop (the body); its other edge
    // leaves to the exit, carrying the index — that carried value is the scan
    // length. Keying on the *header*'s terminator (not any matching CBranch in the
    // function) ensures the NUL test is the loop's governing exit.
    {
        let Some(term) = BasicBlock::from_id(ctx, header).iter().last() else {
            return None;
        };
        let Mnemonic::CBranch(CBranch {
            condition,
            success_block,
            success_args,
            failure_block,
            failure_args,
        }) = term.mnemonic()
        else {
            return None;
        };
        let nonzero_continues = nonzero_polarity(ctx, *condition, elem_val)?;
        // The exit edge is the one taken when the byte is zero.
        let (exit_block, exit_args) = if nonzero_continues {
            (*failure_block, failure_args)
        } else {
            (*success_block, success_args)
        };
        // The exit edge must carry the index (the count) to an exit-block param.
        let kx = exit_args.iter().position(|&v| v == index)?;
        let count_param = BasicBlock::from_id(ctx, exit_block)
            .params()
            .nth(kx)
            .map(|p| p.id())?;

        // The body is the continue target; header/body/exit must be distinct so the
        // rewrite (and any deletion) can address them separately.
        let body_block = if nonzero_continues {
            *success_block
        } else {
            *failure_block
        };
        if header == body_block || header == exit_block || body_block == exit_block {
            return None;
        }

        // The body's terminator must be an unconditional branch back to the header:
        // the back-edge is the only way out of the body, so the header's NUL test is
        // the loop's *sole* data-dependent exit. Without this, a second `break`
        // (e.g. on another byte value) would make the count not the first-zero index.
        let back_ok = BasicBlock::from_id(ctx, body_block).iter().last().is_some_and(|t| {
            matches!(t.mnemonic(), Mnemonic::Branch(Branch { target, .. }) if *target == header)
        });
        if !back_ok {
            return None;
        }

        // The preheader is the header's sole out-of-loop predecessor: the reroute
        // source when the dead loop is deleted (there is no seed store to key on).
        let out_of_loop: Vec<BlockId> = BasicBlock::from_id(ctx, header)
            .predecessors()
            .map(|(_, p)| p)
            .filter(|&p| p != body_block)
            .collect();
        let [preheader] = out_of_loop[..] else {
            return None;
        };

        // Deletable iff wholly private (mirrors the map recognizer): the exit carries
        // only the count, and every value the loop defines is used only inside it.
        let loop_blocks = [header, body_block];
        let in_loop = |v: ValueId| {
            ctx.users(v).iter().all(|&u| {
                ctx.get_insn(u)
                    .parent()
                    .is_some_and(|b| loop_blocks.contains(&b.id))
            })
        };
        let exit_only_count = BasicBlock::from_id(ctx, exit_block)
            .params()
            .all(|p| p.id() == count_param);
        let deletable = exit_only_count
            && loop_blocks.iter().all(|&blk| {
                let b = BasicBlock::from_id(ctx, blk);
                b.params().all(|p| p.id() == index || in_loop(p.id()))
                    && b.iter().all(|i| in_loop(ValueId::Instruction(i.id)))
            });

        Some(StrlenMatch {
            arr,
            count_param,
            header_block: header,
            body_block,
            exit_block,
            preheader,
            deletable,
        })
    }
}

/// Rewrite a matched NUL-scan: replace the escaping count with
/// `len(take_while(@arr))`, then (when the scan is wholly private) strip the seed
/// and delete the dead loop.
fn apply_strlen(ctx: &mut Context, fid: FunctionId, m: &StrlenMatch) -> bool {
    // take_while(@arr) then len(...) of it, inserted at the top of the exit block.
    let tw_id = IntrinsicId::from_name("take_while").expect("take_while registered");
    let len_id = IntrinsicId::from_name("len").expect("len registered");
    let first = BasicBlock::from_id(ctx, m.exit_block)
        .iter()
        .next()
        .map(|i| i.id);
    let len_val = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.exit_block));
        if let Some(at) = first {
            b.set_insert_point_before(at);
        }
        let tw = b.push_intrinsic(tw_id, vec![m.arr]).id();
        b.push_intrinsic(len_id, vec![tw]).id()
    };
    ctx.replace_all_uses_with(m.count_param, len_val);

    if m.deletable {
        // The count was the loop's only escape and is now forwarded to `len`, so the
        // whole scan is dead. Order matters:
        //   1. drop the exit's count param (rewrites the header cbranch's exit-edge
        //      args while the header still exists),
        //   2. reroute the preheader straight to the (now param-less) exit,
        //   3. delete the dead loop blocks. (There is no seed store to strip — the
        //      `at`-form scan reads the root array param directly.)
        let kx = BasicBlock::from_id(ctx, m.exit_block)
            .params()
            .position(|p| p.id() == m.count_param);
        if let Some(kx) = kx {
            crate::dce::remove_params_from_block(ctx, m.exit_block, &HashSet::from_iter([kx]));
        }
        // Reroute the preheader straight to the (now param-less) exit and delete the
        // dead loop blocks (body before header, as they were emitted).
        delete_private_loop(
            ctx,
            fid,
            m.preheader,
            &[m.body_block, m.header_block],
            m.exit_block,
            Vec::new(),
        );
    }
    // Otherwise the loop also feeds outside consumers, so it keeps running; only the
    // count was forwarded to `len` above.
    true
}

/// Recognize bounded NUL-scan loops across all pure functions, rewriting each
/// escaping count to `len(take_while(arr))`. Returns `true` if anything changed.
pub(crate) fn recognize_strlens(ctx: &mut Context) -> bool {
    let fids: Vec<FunctionId> = ctx.function_ids();
    let mut changed = false;
    for fid in fids {
        if !Function::from_id(ctx, fid).is_pure() {
            continue;
        }
        if let Some(m) = try_match_strlen(ctx, fid) {
            changed |= apply_strlen(ctx, fid, &m);
        }
    }
    changed
}

// ===========================================================================
// strlen recognizer (Layer 2) — raw pointer scan, no snapshot
// ===========================================================================
//
// The general `strlen` is a raw `char*` scan with no bounded snapshot (argpromote
// only snapshots *bounded* regions, and a NUL scan's index is unbounded). After
// `argpromote_registers` the pointer is a param; the loop steps it and stops at the
// first zero byte, and the length is the pointer *difference* `end - base`:
//
//   <entry @s0>   goto <head @s=@s0>;
//   <head @s>     %b = load(ram:1, @s);
//                 if %b != 0 goto <body> else goto <exit @s>;     // NUL test
//   <body>        goto <head @s=@s+1>;
//   <exit @end>   %len = @end - @s0;     ... uses of %len = strlen ...
//
// The source for `take_while` is the *pointer* `@s0` (an unbounded string view, see
// [`TypeManager::get_or_make_unbounded_list`]), and `end - base` is exactly
// `len(take_while(@s0))`. Unlike Layer 1 this needs no purity gate (the raw loads
// make the function impure) and no snapshot; the soundness is purely structural —
// a +1 pointer induction whose sole exit is the NUL test, with the length read off
// as the end-minus-base difference. The dead scan is left for later DCE.
//
// [`TypeManager::get_or_make_unbounded_list`]: qcode::types::TypeManager::get_or_make_unbounded_list

/// A recognized raw-pointer NUL-scan whose `end - base` difference is `strlen`.
struct StrlenPtrMatch {
    /// The string base pointer `@s0` (the `take_while` source).
    base: ValueId,
    /// The `end - base` pointer-difference instruction (becomes `len(take_while)`).
    diff_id: InstructionId,
    /// The block holding that difference (where `len`/`take_while` are inserted).
    diff_block: BlockId,
}

/// Match a raw-pointer NUL-scan in `fid`, or `None` for any other shape.
fn try_match_strlen_ptr(ctx: &Context, fid: FunctionId) -> Option<StrlenPtrMatch> {
    for block in Function::from_id(ctx, fid).iter() {
        let header = block.id;
        let params: Vec<ValueId> = BasicBlock::from_id(ctx, header)
            .params()
            .map(|p| p.id())
            .collect();
        for (k, &s) in params.iter().enumerate() {
            // Induction pointer: stepped by +1 on the back-edge, initialised to a
            // single base pointer `@s0`. Requiring *exactly one* non-increment
            // incoming pins the base unambiguously — with two entry pointers, the
            // `end - base` length would only be `strlen` on the matching entry.
            let incoming = incoming(ctx, header, k);
            if !incoming.iter().any(|&v| is_increment(ctx, v, s)) {
                continue;
            }
            let bases: Vec<ValueId> = incoming
                .iter()
                .copied()
                .filter(|&v| !is_increment(ctx, v, s))
                .collect();
            let [base] = bases[..] else {
                continue;
            };

            // A byte load at the pointer, in real memory (not a shadow snapshot —
            // that is Layer 1's `is_temp` region).
            let load = BasicBlock::from_id(ctx, header)
                .iter()
                .find_map(|i| match i.mnemonic() {
                    Mnemonic::Load(l) if l.ptr == s && l.size == 1 && !is_temp(ctx, l.space) => {
                        Some(i.id)
                    }
                    _ => None,
                });
            let Some(load_id) = load else {
                continue;
            };
            let elem_val = ValueId::Instruction(load_id);

            // The header's terminator is the NUL test; its continue edge re-enters
            // the loop and its other edge leaves, carrying the end pointer.
            let Some(term) = BasicBlock::from_id(ctx, header).iter().last() else {
                continue;
            };
            let Mnemonic::CBranch(CBranch {
                condition,
                success_block,
                success_args,
                failure_block,
                failure_args,
            }) = term.mnemonic()
            else {
                continue;
            };
            let Some(nonzero_continues) = nonzero_polarity(ctx, *condition, elem_val) else {
                continue;
            };
            let (exit_block, exit_args, body_block) = if nonzero_continues {
                (*failure_block, failure_args, *success_block)
            } else {
                (*success_block, success_args, *failure_block)
            };
            // The exit edge must carry the induction pointer (the end pointer).
            if !exit_args.iter().any(|&v| v == s) {
                continue;
            }
            if header == body_block || header == exit_block {
                continue;
            }
            // The body is an unconditional back-edge: the NUL test is the loop's
            // sole exit (else `end - base` is not the first-zero offset).
            let back_ok = BasicBlock::from_id(ctx, body_block).iter().last().is_some_and(|t| {
                matches!(t.mnemonic(), Mnemonic::Branch(Branch { target, .. }) if *target == header)
            });
            if !back_ok {
                continue;
            }

            // The escaping length is the pointer difference `end - base`, where the
            // end pointer is the exit-block param fed the induction pointer. Matched
            // by `lhs - base` with `lhs` an exit param carrying `s`.
            let kx = exit_args.iter().position(|&v| v == s)?;
            let end_param = BasicBlock::from_id(ctx, exit_block)
                .params()
                .nth(kx)
                .map(|p| p.id())?;
            for b2 in Function::from_id(ctx, fid).iter() {
                let bid = b2.id;
                for i in b2.iter() {
                    if let Mnemonic::Binop(bin) = i.mnemonic()
                        && matches!(bin.op, Binop::Int(IntBinop::Sub))
                        && bin.lhs == end_param
                        && bin.rhs == base
                    {
                        return Some(StrlenPtrMatch {
                            base,
                            diff_id: i.id,
                            diff_block: bid,
                        });
                    }
                }
            }
        }
    }
    None
}

/// Rewrite a matched raw-pointer scan: replace its `end - base` difference with
/// `len(take_while(@base))` over the unbounded string at `@base`.
fn apply_strlen_ptr(ctx: &mut Context, m: &StrlenPtrMatch) -> bool {
    let tw_id = IntrinsicId::from_name("take_while").expect("take_while registered");
    let len_id = IntrinsicId::from_name("len").expect("len registered");
    let len_val = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.diff_block));
        b.set_insert_point_before(m.diff_id);
        let tw = b.push_intrinsic(tw_id, vec![m.base]).id();
        b.push_intrinsic(len_id, vec![tw]).id()
    };
    ctx.replace_all_uses_with(ValueId::Instruction(m.diff_id), len_val);
    ctx.remove_instruction(m.diff_id);
    // The scan now produces nothing used outside it; later DCE removes the dead loop.
    true
}

/// Recognize raw-pointer NUL-scan loops across all functions, rewriting each
/// `end - base` length to `len(take_while(base))`. Returns `true` if anything changed.
pub(crate) fn recognize_strlens_ptr(ctx: &mut Context) -> bool {
    let fids: Vec<FunctionId> = ctx.function_ids();
    let mut changed = false;
    for fid in fids {
        if let Some(m) = try_match_strlen_ptr(ctx, fid) {
            changed |= apply_strlen_ptr(ctx, &m);
        }
    }
    changed
}

#[derive(Default)]
pub struct LoopToMap;

impl Pass for LoopToMap {
    const NAME: &'static str = "loop_to_map";
    fn description(&self) -> &'static str {
        "Rewrite a total element-wise array loop as a single map, and a NUL-scan as len(take_while)"
    }
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        let mut changed = recognize_total_maps(ctx);
        changed |= recognize_strlens(ctx);
        changed |= recognize_strlens_ptr(ctx);
        Ok(changed)
    }
}

crate::register_module_pass!(LoopToMap);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{Value, insn::Mnemonic},
    };
    use qcode_macro::qcode;

    use crate::mem::array_promote::ArrayPromote;
    use crate::test_util::run_function_pass;

    /// Promote `fid`, mark it pure (the real pipeline functionalizes it before
    /// `loop_to_map`, which gates on purity), then fold its total map. Returns
    /// `(promoted, folded)`.
    fn promote_then_map(ctx: &mut Context, fid: FunctionId) -> (bool, bool) {
        let promoted = run_function_pass::<ArrayPromote>(ctx, fid).unwrap();
        Function::from_id_mut(ctx, fid).set_is_pure(true);
        let folded = recognize_total_maps(ctx);
        (promoted, folded)
    }

    // ===== recognizer (two-buffer) =========================================

    /// Build a counted, byte-lane total-map loop `dst[i] = src[i]` over `N`
    /// elements in the canonical post-argpromote shape:
    ///   entry: seed `*[shadow]:N base_src = arr`; goto head(i=0)
    ///   head:  if i < N goto body else goto exit
    ///   body:  e = *[shadow]:1 (base_src+i); *[shadow]:1 (base_dst+i) = e; i+1
    ///   exit:  wv = *[shadow]:N base_dst; store(ram, &out = wv); return
    /// When `distinct_dst` is false, `base_dst == base_src` (the in-place case).
    /// When `stray` is true, an extra shadow load on the destination base is
    /// injected so the region bookkeeping should reject the loop.
    ///
    /// Returns `(fid, exit_block, arr_value)`.
    fn build_copy_loop(
        tc: &mut TestContext,
        distinct_dst: bool,
        stray: bool,
    ) -> (FunctionId, BlockId, ValueId) {
        const N: usize = 4;
        let i8 = tc.ctx.types.get_or_make_int(1);
        let arr_ty = tc.ctx.types.get_or_make_array(i8, N);
        let shadow = tc.ctx.make_temp_space();
        let ram = tc.ctx.default_space;

        let fid = Function::make(&mut tc.ctx, "copy".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let header = tc.ctx.get_or_make_block(0x1010);
        let body = tc.ctx.get_or_make_block(0x1020);
        let exit = tc.ctx.get_or_make_block(0x1030);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(header);
            f.add_block(body);
            f.add_block(exit);
        }

        // Root params: arr:[i8;N], base_src, base_dst (the call interface).
        let arr_pid = BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(N).id;
        tc.ctx.values.block_params[arr_pid].type_id = arr_ty;
        let arr = ValueId::BlockParam(arr_pid);
        let base_src =
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id);
        let base_dst = if distinct_dst {
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id)
        } else {
            base_src
        };

        // Header induction param.
        let i = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(8)
                .id,
        );

        // entry: seed store + preheader branch.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let zero = b.context_mut().get_const(0, 8).id();
            b.push_store(arr, base_src, shadow); // *[shadow]:N base_src = arr
            b.push_branch_with_args(header, vec![zero]);
        }
        // header: counted guard.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, header));
            let n = b.context_mut().get_const(N as u64, 8).id();
            let cond = b.push_lt(i, n).id();
            b.push_cbranch_with_args(cond, body, vec![], exit, vec![]);
        }
        // body: read src lane, write dst lane, increment.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, body));
            let one = b.context_mut().get_const(1, 8).id();
            let addr_src = b.push_add(base_src, i).id();
            let elem = b.push_load::<false>(addr_src, 1, shadow).id();
            let addr_dst = b.push_add(base_dst, i).id();
            b.push_store(elem, addr_dst, shadow); // *[shadow]:1 (base_dst+i) = elem
            let inc = b.push_add(i, one).id();
            b.push_branch_with_args(header, vec![inc]);
        }
        // exit: wide reload (write-set) + an external use + return.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, exit));
            if stray {
                // An extra shadow access on the destination region: should defeat
                // the exactness check.
                b.push_load::<false>(base_dst, 1, shadow);
            }
            let wv = b.push_load::<false>(base_dst, N, shadow).id();
            let out = b.context_mut().get_const(0x9000, 8).id();
            b.push_store(wv, out, ram); // external consumer of the write-set
            let dummy = b.context_mut().get_const(0, 8).id();
            b.push_return(dummy);
        }

        Function::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);
        (fid, exit, arr)
    }

    /// An in-place indexed map whose element body is *index-free*
    /// (`out[i] = out[i] ^ 0x5a`): `array_promote` threads one carried array with
    /// an own-lane `at` read, and `loop_to_map` folds it to `map(f, arr0)` — a
    /// value-only body, so **no** `enumerate` is inserted.
    #[test]
    fn array_promote_then_map_folds_xor_fill() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn xorbuf:
            <entry @base:i64>
                goto <head @i=0 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 16;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %off = @j * 4;
                %addr = @b + %off;
                %x = load(ram:4, %addr);
                %v = %x ^ 0x5a;
                store(ram:4, %addr <- %v);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        let (promoted, folded) = promote_then_map(&mut ctx, xorbuf);
        assert!(promoted, "the indexed fill should promote");
        assert!(folded, "the promoted loop should fold to a map");
        let ir = format!("{}", Function::from_id(&ctx, xorbuf));
        assert!(ir.contains("<$>"), "folds to a map: {ir}");
        assert!(
            !ir.contains("enumerate"),
            "an index-free body needs no enumerate: {ir}"
        );
        // Exactly one outlined pure body, and it recomputes the xor.
        let bodies: Vec<FunctionId> = ctx
            .function_ids()
            .into_iter()
            .filter(|&f| f != xorbuf && Function::from_id(&ctx, f).is_pure())
            .collect();
        assert_eq!(bodies.len(), 1, "one map body outlined");
        let broot = Function::from_id(&ctx, bodies[0]).root().unwrap().id;
        assert!(
            BasicBlock::from_id(&ctx, broot).iter().any(|i| matches!(
                i.mnemonic(),
                Mnemonic::Binop(b) if matches!(b.op, Binop::Int(IntBinop::Xor))
            )),
            "the body recomputes the xor"
        );
    }

    /// The same loop with an *index-aware* body (`out[i] = out[i] + i`) folds to
    /// `map(f, enumerate(arr0))` — the index forces an `enumerate`.
    #[test]
    fn index_aware_body_maps_enumerate() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn addidx:
            <entry @base:i64>
                goto <head @i=0 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 16;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %off = @j * 4;
                %addr = @b + %off;
                %x = load(ram:4, %addr);
                %jt = @j[0:4];
                %v = %x + %jt;
                store(ram:4, %addr <- %v);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        let (promoted, folded) = promote_then_map(&mut ctx, addidx);
        assert!(promoted && folded);
        let ir = format!("{}", Function::from_id(&ctx, addidx));
        assert!(
            ir.contains("<$>") && ir.contains("enumerate"),
            "an index-aware body maps over enumerate: {ir}"
        );
    }

    /// A write-only generation `out[i] = i * 3` promotes to a `splat(0, N)` init
    /// (no region load); folding maps over `enumerate(splat)` — the map source
    /// chains back to the splat, and no new region `Load` is introduced.
    #[test]
    fn generation_loop_maps_over_splat() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn generate:
            <entry @base:i64>
                goto <head @i=0 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 16;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %jt = trunc(i32, @j);
                %next = %jt * 3;
                %off = @j * 4;
                %addr = @b + %off;
                store(ram:4, %addr <- %next);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        let (promoted, folded) = promote_then_map(&mut ctx, generate);
        assert!(promoted && folded);
        let ir = format!("{}", Function::from_id(&ctx, generate));
        assert!(
            ir.contains("<$>") && ir.contains("$splat("),
            "maps over splat: {ir}"
        );
        assert!(
            !ir.contains("load(ram"),
            "generation introduces no region load: {ir}"
        );
    }

    /// The seeded prefix sum (a scan: lane-0 seed insert + `at(arr, i-1)` carry
    /// read) is `loop_to_scan`'s shape, not a map — `recognize_total_maps` must
    /// decline it (mutual exclusion via `ca.seed`).
    #[test]
    fn scan_shape_is_not_matched_by_map() {
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
        assert!(run_function_pass::<ArrayPromote>(&mut ctx, prefix).unwrap());
        Function::from_id_mut(&mut ctx, prefix).set_is_pure(true);
        assert!(
            !recognize_total_maps(&mut ctx),
            "a seeded scan is not a map"
        );
    }

    /// All-or-nothing outlining: a body reaching an unrelated RAM load is not a
    /// closed pure expression, so the map is not applied and nothing changes.
    #[test]
    fn impure_body_left_alone() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn impure:
            <entry @base:i64 @other:i64>
                goto <head @i=0 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 16;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %off = @j * 4;
                %addr = @b + %off;
                %x = load(ram:4, %addr);
                %y = load(ram:4, @other);
                %v = %x + %y;
                store(ram:4, %addr <- %v);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        run_function_pass::<ArrayPromote>(&mut ctx, impure).unwrap();
        Function::from_id_mut(&mut ctx, impure).set_is_pure(true);
        assert!(
            !recognize_total_maps(&mut ctx),
            "an impure body must not fold to a map"
        );
    }

    // ===== recognizer (strlen) =============================================

    /// Build a bounded NUL-scan `while (arr[i]) i++;` over an `N`-byte snapshot in
    /// the canonical post-argpromote shape, with the scan length escaping via the
    /// exit block param (consumed by an external store):
    ///   entry: seed `*[shadow]:N base = arr`; goto head(i=0)
    ///   head:  %b = *[shadow]:1 (base+i); if %b != 0 goto body else goto exit(i)
    ///   body:  goto head(i+1)
    ///   exit:  store(ram, &out = count); return
    /// When `extra_break` is true the body ends in a *second* data-dependent exit
    /// (a `break` on another condition) instead of a clean back-edge, so the NUL
    /// test is no longer the loop's sole exit and recognition must decline.
    ///
    /// Returns `(fid, exit_block, arr_value)`.
    fn build_strlen_loop(
        tc: &mut TestContext,
        extra_break: bool,
    ) -> (FunctionId, BlockId, ValueId) {
        const N: usize = 8;
        let i8 = tc.ctx.types.get_or_make_int(1);
        let arr_ty = tc.ctx.types.get_or_make_array(i8, N);
        let shadow = tc.ctx.make_temp_space();
        let ram = tc.ctx.default_space;

        let fid = Function::make(&mut tc.ctx, "slen".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let header = tc.ctx.get_or_make_block(0x1010);
        let body = tc.ctx.get_or_make_block(0x1020);
        let exit = tc.ctx.get_or_make_block(0x1030);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(header);
            f.add_block(body);
            f.add_block(exit);
        }

        let arr_pid = BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(N).id;
        tc.ctx.values.block_params[arr_pid].type_id = arr_ty;
        let arr = ValueId::BlockParam(arr_pid);
        let base =
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id);

        let i = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(8)
                .id,
        );
        let count =
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, exit).push_param(8).id);

        // entry: seed store + preheader branch (i = 0).
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let zero = b.context_mut().get_const(0, 8).id();
            b.push_store(arr, base, shadow);
            b.push_branch_with_args(header, vec![zero]);
        }
        // head: load lane, NUL test; continue while nonzero, else exit carrying i.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, header));
            let addr = b.push_add(base, i).id();
            let byte = b.push_load::<false>(addr, 1, shadow).id();
            let zero1 = b.context_mut().get_const(0, 1).id();
            let nz = b.push_ne(byte, zero1).id();
            b.push_cbranch_with_args(nz, body, vec![], exit, vec![i]);
        }
        // body: increment, then either a clean back-edge or (extra_break) a second
        // exit on `i < 100` — a break that defeats the sole-exit requirement.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, body));
            let one = b.context_mut().get_const(1, 8).id();
            let inc = b.push_add(i, one).id();
            if extra_break {
                let hundred = b.context_mut().get_const(100, 8).id();
                let lt = b.push_lt(inc, hundred).id();
                b.push_cbranch_with_args(lt, header, vec![inc], exit, vec![inc]);
            } else {
                b.push_branch_with_args(header, vec![inc]);
            }
        }
        // exit: external consumer of the length + return.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, exit));
            let out = b.context_mut().get_const(0x9000, 8).id();
            b.push_store(count, out, ram);
            let dummy = b.context_mut().get_const(0, 8).id();
            b.push_return(dummy);
        }

        Function::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);
        (fid, exit, arr)
    }

    /// `len(take_while(@arr))` over `@arr`, if such a chain appears in `block`.
    fn len_take_while_src(ctx: &Context, block: BlockId) -> Option<ValueId> {
        BasicBlock::from_id(ctx, block).iter().find_map(|i| {
            let Mnemonic::Intrinsic(len) = i.mnemonic() else {
                return None;
            };
            if len.id.name() != "len" {
                return None;
            }
            let ValueId::Instruction(tw_id) = *len.args.first()? else {
                return None;
            };
            match ctx.get_insn(tw_id).mnemonic() {
                Mnemonic::Intrinsic(tw) if tw.id.name() == "take_while" => tw.args.first().copied(),
                _ => None,
            }
        })
    }

    /// A bounded NUL-scan over an argpromote snapshot: `array_reads` first rewrites
    /// the shadow lane loads to `at(@arr, i)`, then the strlen recognizer folds the
    /// escaping length to `len(take_while(@arr))` over the source array.
    #[test]
    fn array_reads_then_strlen_folds_nul_scan() {
        let mut tc = TestContext::new();
        let (fid, exit, arr) = build_strlen_loop(&mut tc, /*extra_break*/ false);

        assert!(
            run_function_pass::<crate::mem::array_reads::ArrayReads>(&mut tc.ctx, fid).unwrap(),
            "array_reads promotes the read-only shadow snapshot to at(@arr, i)"
        );
        assert!(
            recognize_strlens(&mut tc.ctx),
            "bounded NUL-scan must recognize on the at-form"
        );
        assert_eq!(
            len_take_while_src(&tc.ctx, exit),
            Some(arr),
            "the escaping count becomes len(take_while(@arr))"
        );
        assert!(Function::from_id(&tc.ctx, fid).is_pure());
        // The private scan is deleted: only entry + exit remain.
        assert_eq!(
            Function::from_id(&tc.ctx, fid).iter().count(),
            2,
            "the dead scan loop is removed"
        );
    }

    /// A loop with a second data-dependent exit (a `break` besides the NUL test) is
    /// not a clean `take_while`: its count is the first of *either* terminator, not
    /// the first zero, so recognition must decline even after `array_reads`.
    #[test]
    fn second_break_is_not_a_strlen() {
        let mut tc = TestContext::new();
        let (fid, exit, _arr) = build_strlen_loop(&mut tc, /*extra_break*/ true);

        run_function_pass::<crate::mem::array_reads::ArrayReads>(&mut tc.ctx, fid).unwrap();
        assert!(
            !recognize_strlens(&mut tc.ctx),
            "a loop with a second exit is not a NUL-scan strlen"
        );
        assert_eq!(
            len_take_while_src(&tc.ctx, exit),
            None,
            "no len/take_while inserted"
        );
    }

    /// A copy loop (which *stores*) is not a NUL-scan strlen: `array_reads` refuses
    /// the region (a non-seed store), and the strlen recognizer finds no `at` scan.
    #[test]
    fn copy_loop_is_not_a_strlen() {
        let mut tc = TestContext::new();
        let (fid, exit, _arr) =
            build_copy_loop(&mut tc, /*distinct_dst*/ true, /*stray*/ false);

        assert!(
            !run_function_pass::<crate::mem::array_reads::ArrayReads>(&mut tc.ctx, fid).unwrap(),
            "array_reads declines a region with a store"
        );
        assert!(
            !recognize_strlens(&mut tc.ctx),
            "a loop that stores is not a read-only NUL-scan"
        );
        assert_eq!(
            len_take_while_src(&tc.ctx, exit),
            None,
            "no len/take_while inserted"
        );
    }

    // ===== recognizer (strlen, raw pointer / Layer 2) ======================

    /// Build a raw `char*` NUL-scan with the length as a pointer difference, the
    /// general post-`argpromote_registers` shape (no snapshot, raw RAM loads):
    ///   entry: goto head(s = s0)
    ///   head:  %b = load(ram:1, s); if %b != 0 goto body else goto exit(s)
    ///   body:  goto head(s + 1)
    ///   exit:  %len = end - s0; store(ram, &out = %len); return
    /// When `with_diff` is false the exit consumes the end pointer directly instead
    /// of `end - s0`, so there is no length difference to rewrite and recognition
    /// must decline.
    ///
    /// Returns `(fid, exit_block, s0_base)`.
    fn build_strlen_ptr_loop(
        tc: &mut TestContext,
        with_diff: bool,
    ) -> (FunctionId, BlockId, ValueId) {
        let ram = tc.ctx.default_space;
        let fid = Function::make(&mut tc.ctx, "strlen".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x2000);
        let header = tc.ctx.get_or_make_block(0x2010);
        let body = tc.ctx.get_or_make_block(0x2020);
        let exit = tc.ctx.get_or_make_block(0x2030);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(header);
            f.add_block(body);
            f.add_block(exit);
        }

        // The base string pointer `@s0` (a root param).
        let s0 = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id);
        // Header induction pointer and exit end-pointer params.
        let s = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(8)
                .id,
        );
        let end = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, exit).push_param(8).id);

        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_branch_with_args(header, vec![s0]);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, header));
            let byte = b.push_load::<false>(s, 1, ram).id();
            let zero = b.context_mut().get_const(0, 1).id();
            let nz = b.push_ne(byte, zero).id();
            b.push_cbranch_with_args(nz, body, vec![], exit, vec![s]);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, body));
            let one = b.context_mut().get_const(1, 8).id();
            let s1 = b.push_add(s, one).id();
            b.push_branch_with_args(header, vec![s1]);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, exit));
            // strlen = end - base; or (no diff) the end pointer is consumed directly.
            let escaping = if with_diff {
                b.push_sub(end, s0).id()
            } else {
                end
            };
            let out = b.context_mut().get_const(0x9000, 8).id();
            b.push_store(escaping, out, ram);
            let dummy = b.context_mut().get_const(0, 8).id();
            b.push_return(dummy);
        }
        (fid, exit, s0)
    }

    /// A raw `char*` NUL-scan has its `end - base` length rewritten to
    /// `len(take_while(@base))` over the unbounded string at the base pointer.
    #[test]
    fn raw_pointer_nul_scan_recognized_as_len_take_while() {
        let mut tc = TestContext::new();
        let (_fid, exit, s0) = build_strlen_ptr_loop(&mut tc, /*with_diff*/ true);

        assert!(
            recognize_strlens_ptr(&mut tc.ctx),
            "raw-pointer NUL-scan must recognize"
        );
        assert_eq!(
            len_take_while_src(&tc.ctx, exit),
            Some(s0),
            "the pointer-difference length becomes len(take_while(@base))"
        );
        // The take_while source is the bare pointer, so its result is an *unbounded*
        // list (no static footprint).
        let tw_ty = BasicBlock::from_id(&tc.ctx, exit)
            .iter()
            .find_map(|i| match i.mnemonic() {
                Mnemonic::Intrinsic(intr) if intr.id.name() == "take_while" => Some(i.type_id()),
                _ => None,
            });
        let tw_ty = tw_ty.expect("a take_while was inserted");
        assert_eq!(
            tc.ctx.types.list_of(tw_ty).map(|(_, b)| b),
            Some(None),
            "unbounded list"
        );
    }

    /// Without the `end - base` difference there is no length expression to rewrite,
    /// so the scan is left untouched (no spurious `take_while`/`len` is invented).
    #[test]
    fn raw_pointer_scan_without_difference_declined() {
        let mut tc = TestContext::new();
        let (_fid, exit, _s0) = build_strlen_ptr_loop(&mut tc, /*with_diff*/ false);

        assert!(
            !recognize_strlens_ptr(&mut tc.ctx),
            "no end-base difference means nothing to rewrite"
        );
        assert_eq!(
            len_take_while_src(&tc.ctx, exit),
            None,
            "no len/take_while inserted"
        );
    }
}
