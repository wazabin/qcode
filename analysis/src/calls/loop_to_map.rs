//! Loop-to-map: recognize a total element-wise array loop and rewrite it as a
//! single [`Map`](qcode::value::insn::Map) over the array value, outlining the
//! per-element body into a fresh pure function. See `ARGPROMOTE_ARRAY_MAP.md`.
//!
//! This module currently provides [`outline_expression`], the mechanical core:
//! copying a pure expression DAG into a standalone function. The recognizer that
//! drives it (finding the loop, proving element-locality, calling
//! [`push_map`](qcode::builder::Builder::push_map)) builds on top.
//!
//! `outline_expression` is the recognizer's mechanical core.

use std::borrow::Cow;

use rustc_hash::FxHashMap as HashMap;

use qcode::{
    builder::Builder,
    context::Context,
    space::{Space, SpaceId, SpaceType},
    types::TypeId,
    value::{
        BasicBlock, BlockId, Function, FunctionId, Instruction, InstructionRef, Renameable,
        ValueId,
        insn::{Binop, Branch, Extract, InstructionId, IntBinop, IntrinsicId, Mnemonic, Return},
    },
};

use crate::value_range;
use crate::{Pass, PipelineEnv};

/// Whether `m` is a pure value-computing op that may appear inside an outlined
/// per-element body: arithmetic, casts, bit ops, aggregate projection. Anything
/// with a side effect or an untracked source (load/store/call/indirect/pcode/
/// nested map) or a terminator is rejected — such an expression is not a closed
/// pure function of the designated inputs.
fn is_pure_expr_op(m: &Mnemonic) -> bool {
    !matches!(
        m,
        Mnemonic::Load(_)
            | Mnemonic::Store(_)
            | Mnemonic::Call(_)
            | Mnemonic::CallInd(_)
            | Mnemonic::BranchInd(_)
            | Mnemonic::PCodeOp(_)
            | Mnemonic::Map(_)
    ) && !m.is_terminator()
}

/// The backward slice of pure instructions computing `result`, in
/// operands-before-result (post-order) order, or `None` if the expression is not
/// **closed** over `inputs` + literals: it reaches a free block-param, a raw
/// varnode, a function ref, or an impure op. A value in `inputs` is a leaf (it
/// becomes a parameter); a literal is a leaf (referenced directly).
fn pure_slice(ctx: &Context, result: ValueId, inputs: &[ValueId]) -> Option<Vec<InstructionId>> {
    let is_input = |v: ValueId| inputs.contains(&v);
    let mut order: Vec<InstructionId> = Vec::new();
    let mut seen: HashMap<ValueId, ()> = HashMap::default();
    // (value, expanded). The expanded marker emits in post order.
    let mut stack = vec![(result, false)];
    while let Some((v, expanded)) = stack.pop() {
        if is_input(v) || matches!(v, ValueId::Literal(_)) {
            continue; // leaf
        }
        if expanded {
            if let ValueId::Instruction(id) = v {
                order.push(id);
            }
            continue;
        }
        if seen.insert(v, ()).is_some() {
            continue;
        }
        let ValueId::Instruction(id) = v else {
            // A free block-param, varnode, or function ref: not closed.
            return None;
        };
        let m = ctx.get_insn(id).mnemonic().clone();
        if !is_pure_expr_op(&m) {
            return None;
        }
        stack.push((v, true));
        for a in m.args() {
            stack.push((a, false));
        }
    }
    Some(order)
}

/// Outline the pure expression that computes `result` into a fresh standalone
/// function `body(inputs…) -> result`, marked [`is_pure`](Function::is_pure).
/// Each value in `inputs` becomes a parameter (in order, with the input's
/// width/type); every instruction on the backward slice is cloned with operands
/// remapped (inputs→params, clones→clones, literals passed through); the function
/// returns the cloned `result`.
///
/// Returns `None` if the expression is not closed over `inputs` + literals (see
/// [`pure_slice`]) — the caller then leaves the loop unrecognized.
pub(crate) fn outline_expression(
    ctx: &mut Context,
    name: &str,
    result: ValueId,
    inputs: &[ValueId],
) -> Option<FunctionId> {
    let slice = pure_slice(ctx, result, inputs)?;
    let inputs = inputs.to_vec();
    outline_core(ctx, name, result, &slice, move |ctx, root| {
        // Parameters, in input order, typed as the host inputs.
        let mut value_map: HashMap<ValueId, ValueId> = HashMap::default();
        for inp in inputs {
            let ty = ctx.type_of(inp);
            let size = ctx.types.size_of(ty);
            let pid = BasicBlock::from_id_mut(ctx, root).push_param(size).id;
            ctx.values.block_params[pid].type_id = ty;
            value_map.insert(inp, ValueId::BlockParam(pid));
        }
        value_map
    })
}

/// Outline a body that takes a single `enumerate` tuple param `(index, elem)` and
/// unpacks it: the host `index_input` / `elem_input` are bound to `Extract(t, 0)`
/// / `Extract(t, 1)` of the tuple param `t` before the expression computing
/// `result` is cloned. This is the body shape for a `map` over `enumerate(arr)`
/// (rather than over `arr`) — a unary body whose element is the index/value pair.
///
/// Returns `None` if the expression is not closed over `(index, elem)` + literals
/// (see [`pure_slice`]).
fn outline_tupled(
    ctx: &mut Context,
    name: &str,
    result: ValueId,
    index_input: ValueId,
    elem_input: ValueId,
    tuple_ty: TypeId,
) -> Option<FunctionId> {
    let slice = pure_slice(ctx, result, &[index_input, elem_input])?;
    outline_core(ctx, name, result, &slice, move |ctx, root| {
        let mut value_map: HashMap<ValueId, ValueId> = HashMap::default();
        let tsz = ctx.types.size_of(tuple_ty);
        let pid = BasicBlock::from_id_mut(ctx, root).push_param(tsz).id;
        ctx.values.block_params[pid].type_id = tuple_ty;
        let tuple = ValueId::BlockParam(pid);
        // index = t.0, elem = t.1 — the two extracts the body unpacks.
        for (field, input) in [(0usize, index_input), (1usize, elem_input)] {
            let fty = ctx
                .types
                .field_type(tuple_ty, field)
                .expect("enumerate tuple field");
            let ex = InstructionRef::from_mnemonic_with_type(
                ctx,
                Mnemonic::Extract(Extract {
                    agg: tuple,
                    index: field,
                }),
                fty,
            )
            .id;
            BasicBlock::from_id_mut(ctx, root).push_insn(ex);
            value_map.insert(input, ValueId::Instruction(ex));
        }
        value_map
    })
}

/// Build a fresh single-block pure function `name` returning `result`, cloning
/// the pure `slice` (in definition order) with operands remapped through the
/// value map that `seed` installs. `seed` creates the root block's parameters
/// (and any unpacking instructions) and returns the initial input→value map.
fn outline_core(
    ctx: &mut Context,
    name: &str,
    result: ValueId,
    slice: &[InstructionId],
    seed: impl FnOnce(&mut Context, BlockId) -> HashMap<ValueId, ValueId>,
) -> Option<FunctionId> {
    let fid = Function::make(ctx, Cow::Owned(name.to_owned())).ok()?.id;
    let root = Function::from_id_mut(ctx, fid).make_root().id;

    // Give the root block a name so it renders with a real label in the GUI
    // (otherwise it falls back to an opaque `<bb_N>`). The name must be unique
    // across the context name map.
    let block_name = ctx.get_unique_name(Cow::Owned(format!("{name}_entry")));
    BasicBlock::from_id_mut(ctx, root)
        .rename(block_name)
        .expect("name was deduplicated");

    // Params / unpacking, installed by the caller; seeds the input→value map.
    let mut value_map = seed(ctx, root);

    // Clone the slice in definition order, remapping operands through the map.
    for &iid in slice {
        let (mut m, ty) = {
            let insn = Instruction::from_id(ctx, iid);
            (insn.mnemonic().clone(), insn.type_id())
        };
        for a in m.args() {
            if let Some(&n) = value_map.get(&a) {
                m.replace_value(a, n);
            }
        }
        let new_id = InstructionRef::from_mnemonic_with_type(ctx, m, ty).id;
        BasicBlock::from_id_mut(ctx, root).push_insn(new_id);
        value_map.insert(ValueId::Instruction(iid), ValueId::Instruction(new_id));
    }

    // Return the element value. `ptr` is the (irrelevant) return-address slot.
    let ret_val = value_map.get(&result).copied().unwrap_or(result);
    let dummy_ptr = ctx.get_const(0, 8).id();
    let ret_ty = ctx.types.get_or_make_int(1);
    let ret = InstructionRef::from_mnemonic_with_type(
        ctx,
        Mnemonic::Return(Return {
            ptr: dummy_ptr,
            value: Some(ret_val),
        }),
        ret_ty,
    )
    .id;
    BasicBlock::from_id_mut(ctx, root).push_insn(ret);

    // Fully pure: a deterministic function of its params. `is_pure` is the strong
    // flag; also set `pure_reg` (which it implies) so the GUI — whose purity badge
    // keys off `is_pure_reg` — marks the outlined body as functionalized.
    let mut body = Function::from_id_mut(ctx, fid);
    body.set_is_pure(true);
    body.set_pure_reg(true);
    Some(fid)
}

/// Inline the pure straight-line body of `body_fn` (a single-block function
/// returning a value, as produced by [`outline_expression`]) with its parameters
/// bound to `args`, splicing the cloned computation into `block` immediately
/// before `at`. Returns the value the body returns, remapped onto the inserted
/// clones (or directly an arg/literal when the body just returns a param or
/// constant). `None` on an arity mismatch, a missing body, or no return value.
///
/// This is the dual of [`outline_expression`] and the engine of element
/// projection: `Range` of a `Map` inlines `body_fn(src[k])` here (the body is
/// unary in the element).
pub(crate) fn inline_pure_body(
    ctx: &mut Context,
    body_fn: FunctionId,
    args: &[ValueId],
    block: BlockId,
    at: InstructionId,
) -> Option<ValueId> {
    let root = Function::from_id(ctx, body_fn).root().map(|b| b.id)?;
    let params: Vec<ValueId> = BasicBlock::from_id(ctx, root)
        .params()
        .map(|p| p.id())
        .collect();
    if params.len() != args.len() {
        return None;
    }
    let mut value_map: HashMap<ValueId, ValueId> = HashMap::default();
    for (&p, &a) in params.iter().zip(args) {
        value_map.insert(p, a);
    }
    let insns: Vec<InstructionId> = BasicBlock::from_id(ctx, root)
        .iter()
        .map(|i| i.id)
        .collect();
    for iid in insns {
        let m = ctx.get_insn(iid).mnemonic().clone();
        if let Mnemonic::Return(r) = &m {
            let v = r.value?;
            return Some(value_map.get(&v).copied().unwrap_or(v));
        }
        let ty = ctx.get_insn(iid).type_id();
        let mut nm = m;
        for a in nm.args() {
            if let Some(&n) = value_map.get(&a) {
                nm.replace_value(a, n);
            }
        }
        let new_id = InstructionRef::from_mnemonic_with_type(ctx, nm, ty).id;
        BasicBlock::from_id_mut(ctx, block).insert_insn_before(at, new_id);
        value_map.insert(ValueId::Instruction(iid), ValueId::Instruction(new_id));
    }
    None
}

// ===========================================================================
// Total-map recognizer
// ===========================================================================
//
// After step-1 argpromote, a dynamic-index buffer loop has this canonical shape
// (see `ARGPROMOTE_ARRAY_MAP.md`): an `Array` snapshot param seeded into a shadow
// space, a counted loop that read-modify-writes one shadow lane per iteration,
// and a wide shadow reload that becomes the returned write-set value:
//
//   <entry @base @arr:[i8;N]>
//       *[shadow]:N @base = @arr;                 // seed
//       goto <head @i=0>;
//   <head @i>   if @i < N goto <body> else goto <exit>;
//   <body>      %a = @base + @i;
//               %b = *[shadow]:1 %a;              // load lane
//               %v = body(@i, %b);                // pure
//               *[shadow]:1 %a = %v;              // store lane
//               goto <head @i=@i+1>;
//   <exit>      %wv = *[shadow]:N @base;          // write-set value
//               ... pack(write_value=%wv) ...
//
// The recognizer proves this is a *total* element-wise map (every lane written
// once by a pure body) and rewrites the write-set value to a map — the
// projectable form. When the body reads only the element it is `map(body, @arr)`
// with a unary `body(elem)`; when it also reads the index it is `map(body,
// enumerate(@arr))` with `body(tuple)` unpacking the `(index, elem)` pair. The
// dead shadow loop is left for later cleanup; the function stays pure and
// correct. v1 handles byte lanes (`elem == 1`) with a body closed over
// `(index, element)` only.

/// A recognized total-map loop (all fields are values/ids stable across the
/// rewrite, which only *adds* a function and instructions).
struct MapMatch {
    /// The `Array` snapshot param mapped over.
    arr: ValueId,
    /// The header induction parameter (`@i`).
    index: ValueId,
    /// The per-lane loaded element (`%b`).
    elem_val: ValueId,
    /// The per-lane stored value (`%v`), the body's result.
    stored_val: ValueId,
    /// The wide shadow reload whose uses become the map result (`%wv`).
    wv_id: InstructionId,
    /// The seed store `*[shadow]:N base = arr` (dead after the rewrite).
    seed_id: InstructionId,
    /// The root/entry block holding the seed store and the loop preheader branch.
    entry_block: BlockId,
    /// The loop header block (carries the induction param).
    header_block: BlockId,
    /// The loop body block (the per-lane read-modify-write).
    body_block: BlockId,
    /// The block holding `%wv` (where the map is inserted).
    exit_block: BlockId,
    /// Whether the loop is *private* — every value it defines is used only inside
    /// it and the exit carries no loop-carried params. When `true` the whole loop
    /// is dead after the rewrite and is deleted; when `false` the loop also
    /// computes values consumed outside it (e.g. a clobbered register threaded
    /// into the return envelope), so only the array's shadow accesses are stripped
    /// and the residual loop is left running. See [`apply`].
    deletable: bool,
}

fn is_temp(ctx: &Context, s: SpaceId) -> bool {
    matches!(Space::from_id(ctx, s).ty, SpaceType::Temporary)
}

/// `c` if `v` is the integer literal `c`, else `None`.
fn literal(ctx: &Context, v: ValueId) -> Option<u64> {
    match qcode::value::ValueRef::new(v, ctx) {
        qcode::value::ValueRef::Literal(l) => Some(l.value()),
        _ => None,
    }
}

/// `idx` if `addr` is `base + idx` (either operand order) with `idx` a block
/// param, else `None`.
fn base_plus_param(ctx: &Context, addr: ValueId, base: ValueId) -> Option<ValueId> {
    let ValueId::Instruction(id) = addr else {
        return None;
    };
    let Mnemonic::Binop(b) = ctx.get_insn(id).mnemonic() else {
        return None;
    };
    if !matches!(b.op, Binop::Int(IntBinop::Add)) {
        return None;
    }
    let other = if b.lhs == base {
        b.rhs
    } else if b.rhs == base {
        b.lhs
    } else {
        return None;
    };
    matches!(other, ValueId::BlockParam(_)).then_some(other)
}

/// `true` if `v` is `idx + 1` (either operand order).
fn is_increment(ctx: &Context, v: ValueId, idx: ValueId) -> bool {
    let ValueId::Instruction(id) = v else {
        return false;
    };
    let Mnemonic::Binop(b) = ctx.get_insn(id).mnemonic() else {
        return false;
    };
    matches!(b.op, Binop::Int(IntBinop::Add))
        && ((b.lhs == idx && literal(ctx, b.rhs) == Some(1))
            || (b.rhs == idx && literal(ctx, b.lhs) == Some(1)))
}

/// The values feeding header block-param index `k` from every predecessor edge.
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

/// One shadow-space memory access: `(insn, block, ptr, size, stored)`. `stored`
/// is `Some` for a store.
struct Access {
    id: InstructionId,
    block: BlockId,
    ptr: ValueId,
    size: usize,
    stored: Option<ValueId>,
}

/// Match the canonical total-map loop in `fid`, or `None` if it is any other
/// shape (the function is then left untouched).
fn try_match(ctx: &Context, fid: FunctionId) -> Option<MapMatch> {
    // Collect every temporary-space (shadow) access, with its block.
    let mut accesses: Vec<Access> = Vec::new();
    for block in Function::from_id(ctx, fid).iter() {
        let bid = block.id;
        for insn in block.iter() {
            let acc = match insn.mnemonic() {
                Mnemonic::Load(l) if is_temp(ctx, l.space) => Access {
                    id: insn.id,
                    block: bid,
                    ptr: l.ptr,
                    size: l.size,
                    stored: None,
                },
                Mnemonic::Store(s) if is_temp(ctx, s.space) => Access {
                    id: insn.id,
                    block: bid,
                    ptr: s.ptr,
                    size: s.size,
                    stored: Some(s.src),
                },
                _ => continue,
            };
            accesses.push(acc);
        }
    }

    // Seed store: `*[shadow]:N base = arr`, `arr` a root Array param, `base` a
    // root param. Identifies (arr, base, N).
    let root_params: Vec<ValueId> = Function::from_id(ctx, fid)
        .root()?
        .params()
        .map(|p| p.id())
        .collect();
    let is_root = |v: ValueId| root_params.contains(&v);
    let seed = accesses.iter().find(|a| {
        a.stored.is_some_and(|src| {
            is_root(src)
                && ctx
                    .stored_type_of(src)
                    .and_then(|t| ctx.types.array_of(t))
                    .is_some()
        }) && is_root(a.ptr)
    })?;
    let arr = seed.stored.unwrap();
    let base = seed.ptr;
    let count = seed.size;
    // v1: byte lanes only.
    let (elem_ty, arr_count) = ctx.types.array_of(ctx.stored_type_of(arr)?)?;
    if ctx.types.size_of(elem_ty) != 1 || arr_count != count {
        return None;
    }

    // Exactly four shadow accesses touch the *region base* — seed store and wide
    // reload at `base`, plus the body load+store at `base + idx`. Any other access
    // to the region means it is not a clean total map. Shadow accesses to *other*
    // bases (e.g. argpromote's coexisting frame seed stores) are unrelated and
    // ignored: the region is disjoint from them (see `regions_disjoint`).
    let region_accesses = accesses
        .iter()
        .filter(|a| a.ptr == base || base_plus_param(ctx, a.ptr, base).is_some())
        .count();
    if region_accesses != 4 {
        return None;
    }

    // Body store: `*[shadow]:1 (base + idx) = v`.
    let body_store = accesses.iter().find(|a| {
        a.stored.is_some() && a.size == 1 && base_plus_param(ctx, a.ptr, base).is_some()
    })?;
    let index = base_plus_param(ctx, body_store.ptr, base)?;
    let stored_val = body_store.stored.unwrap();
    let body_block = body_store.block;

    // Body load: `*[shadow]:1 (base + idx)` at the same address value.
    let body_load = accesses
        .iter()
        .find(|a| a.stored.is_none() && a.size == 1 && a.ptr == body_store.ptr)?;
    let elem_val = ValueId::Instruction(body_load.id);

    // Wide reload: `*[shadow]:N base` — the write-set value.
    let wv = accesses
        .iter()
        .find(|a| a.stored.is_none() && a.size == count && a.ptr == base)?;

    // Totality: the index covers exactly `[0, count)` with init 0 and step +1, so
    // every lane is written once.
    let range = value_range(ctx, index, body_block);
    if range.min != 0 || range.max as usize != count - 1 {
        return None;
    }
    let ValueId::BlockParam(pid) = index else {
        return None;
    };
    let header = ctx.values.block_params[pid].parent?;
    let k = BasicBlock::from_id(ctx, header)
        .params()
        .position(|p| p.id() == index)?;
    let incoming = header_incoming(ctx, header, k);
    let inits_at_zero = incoming.iter().any(|&v| literal(ctx, v) == Some(0));
    let steps_by_one = incoming.iter().any(|&v| is_increment(ctx, v, index));
    if !inits_at_zero || !steps_by_one {
        return None;
    }

    // The header and body must be distinct blocks: the rewrite (and, in the
    // deletable case, the block deletion) addresses them separately.
    if header == body_block {
        return None;
    }

    // The map replacement only needs `region_accesses == 4` (nothing else touches
    // the region) plus totality: the wide reload then equals `body(arr[i])` per
    // lane regardless of what *else* the loaded element feeds. So the loaded
    // element is allowed to leak into other results (e.g. an escaping `@EDX`) — we
    // simply keep the loop's shadow channel running in that case (see `deletable`
    // below and the stripping in [`apply`]). No element-locality gate is required.

    // The loop is *deletable* iff it is wholly private: the exit carries no
    // loop-carried params and every value the loop defines is used only inside it.
    // Then the residual loop is dead after the rewrite and is removed entirely.
    // Otherwise the loop also feeds outside consumers (a clobbered register
    // threaded into the return envelope, say), so it must keep running and only
    // the array's shadow accesses are stripped.
    let loop_blocks = [header, body_block];
    let in_loop = |v: ValueId| {
        ctx.users(v).iter().all(|&u| {
            ctx.get_insn(u)
                .parent()
                .is_some_and(|b| loop_blocks.contains(&b.id))
        })
    };
    let exit_has_params = BasicBlock::from_id(ctx, wv.block).params().next().is_some();
    let deletable = !exit_has_params
        && loop_blocks.iter().all(|&blk| {
            let b = BasicBlock::from_id(ctx, blk);
            b.params().all(|p| in_loop(p.id()))
                && b.iter().all(|i| in_loop(ValueId::Instruction(i.id)))
        });

    Some(MapMatch {
        arr,
        index,
        elem_val,
        stored_val,
        wv_id: wv.id,
        seed_id: seed.id,
        entry_block: seed.block,
        header_block: header,
        body_block,
        exit_block: wv.block,
        deletable,
    })
}

/// Does the per-element body actually read the loop index? `enumerate` is only
/// worth inserting when it does; a value-only body maps directly over `arr`.
fn body_uses_index(ctx: &Context, m: &MapMatch) -> bool {
    if m.stored_val == m.index {
        return true;
    }
    match pure_slice(ctx, m.stored_val, &[m.index, m.elem_val]) {
        Some(slice) => slice
            .iter()
            .any(|&iid| ctx.get_insn(iid).mnemonic().args().contains(&m.index)),
        None => false,
    }
}

/// Rewrite a matched loop: outline the per-element body, replace the write-set
/// value with a `map`, then delete the now-dead loop (reroute the preheader
/// straight to the exit and remove the loop blocks, the seed store, and the wide
/// reload).
///
/// The map source depends on whether the body reads the index. A value-only body
/// maps over `arr` directly — `map(body, arr)`, `body(elem)`. An index-aware body
/// maps over `enumerate(arr)`, whose element is the `(index, elem)` tuple —
/// `map(body, enumerate(arr))`, `body(tuple)` unpacking it (the `map` is unary in
/// the element). Returns `false` if the body is not a closed pure expression of
/// `(index, element)` (then nothing is changed — outlining is all-or-nothing and
/// runs before any rewrite).
fn apply(ctx: &mut Context, fid: FunctionId, m: &MapMatch) -> bool {
    let name = format!("{}_map_body", Function::from_id(ctx, fid).name());

    // Outline the body and pick the map source, both before any rewrite so a
    // non-closed body leaves the loop untouched.
    let (body_fn, src_val) = if body_uses_index(ctx, m) {
        let arr_ty = ctx.type_of(m.arr);
        let enum_id = IntrinsicId::from_name("enumerate").expect("enumerate registered");
        let enum_result_ty = enum_id.desc().result_type(&mut ctx.types, &[arr_ty]);
        let Some((tuple_ty, _)) = ctx.types.array_of(enum_result_ty) else {
            return false;
        };
        let Some(body_fn) = outline_tupled(ctx, &name, m.stored_val, m.index, m.elem_val, tuple_ty)
        else {
            return false;
        };
        // enumerate(arr), inserted just before the wide reload it feeds.
        let enum_val = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.exit_block));
            b.set_insert_point_before(m.wv_id);
            b.push_intrinsic(enum_id, vec![m.arr]).id()
        };
        (body_fn, enum_val)
    } else {
        let Some(body_fn) = outline_expression(ctx, &name, m.stored_val, &[m.elem_val]) else {
            return false;
        };
        (body_fn, m.arr)
    };

    // Forward the returned write-set value to the map result. `push_map` types the
    // result from the body's return (the element output type), so it is correct
    // even when the source element is an `enumerate` tuple.
    let wv_val = ValueId::Instruction(m.wv_id);
    let map_val = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.exit_block));
        b.set_insert_point_before(m.wv_id);
        b.push_map(body_fn, src_val, Vec::new()).id()
    };
    ctx.replace_all_uses_with(wv_val, map_val);

    // The wide reload is now forwarded to the map and has no remaining readers
    // (`region_accesses == 4` guarantees nothing else reads the region), so drop
    // it unconditionally.
    ctx.remove_instruction(m.wv_id);

    if m.deletable {
        // The loop is wholly private, so nothing outside it reads what it computes.
        // Strip the seed store (the per-lane store/load vanish with `body_block`
        // below), then reroute the preheader past the loop, straight to the exit
        // (`replace_…` does not touch CFG edges, so add the new edge explicitly;
        // deleting the header/body blocks unlinks the stale `entry → header` edge),
        // then delete the loop blocks.
        ctx.remove_instruction(m.seed_id);
        if let Some(term) = BasicBlock::from_id(ctx, m.entry_block).iter().last() {
            let term_id = term.id;
            ctx.replace_instruction_mnemonic(
                term_id,
                Mnemonic::Branch(Branch {
                    target: m.exit_block,
                    args: Vec::new(),
                }),
            );
            ctx.add_cfg_edge(m.entry_block, m.exit_block);
        }
        BasicBlock::from_id_mut(ctx, m.body_block).delete(fid);
        BasicBlock::from_id_mut(ctx, m.header_block).delete(fid);
    }
    // Otherwise the loop also computes values consumed outside it, so it keeps
    // running with its shadow channel intact: the seed store, per-lane store, and
    // per-lane load are *left in place* — removing them would leave the surviving
    // lane load reading an unseeded region. Only the wide reload was extracted into
    // the map above.
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

#[derive(Default)]
pub struct LoopToMap;

impl Pass for LoopToMap {
    const NAME: &'static str = "loop_to_map";
    fn description(&self) -> &'static str {
        "Rewrite a total element-wise array loop as a single map"
    }
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        Ok(recognize_total_maps(ctx))
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

    /// Build a host function `idx + zext(elem)` and outline it into a body of
    /// `(i64 idx, i8 elem)`. The clone must have two params, recompute the
    /// expression, return it, and be pure.
    #[test]
    fn outlines_closed_pure_expression() {
        let mut tc = TestContext::new();
        let host = Function::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let (idx, elem, result) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let idx = b.push_param(8).id();
            let elem = b.push_param(1).id();
            let widened = b.push_zext(elem, 8).id();
            let result = b.push_add(idx, widened).id();
            (idx, elem, result)
        };

        let body = outline_expression(&mut tc.ctx, "body", result, &[idx, elem])
            .expect("expression is closed over (idx, elem)");

        // Two params, in input order, with the input widths.
        let params: Vec<usize> = Function::from_id(&tc.ctx, body)
            .root()
            .unwrap()
            .params()
            .map(|p| p.size())
            .collect();
        assert_eq!(params, vec![8, 1]);

        // The body recomputes the expression (a zext and an add) and returns it.
        let root = Function::from_id(&tc.ctx, body).root().unwrap().id;
        let has_zext = BasicBlock::from_id(&tc.ctx, root)
            .iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Zext(_)));
        let has_add = BasicBlock::from_id(&tc.ctx, root)
            .iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)));
        assert!(has_zext && has_add, "body recomputes zext + add");
        let returns_value = BasicBlock::from_id(&tc.ctx, root)
            .iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Return(r) if r.value.is_some()));
        assert!(returns_value, "the outlined body returns the element");
        assert!(Function::from_id(&tc.ctx, body).is_pure());
        // Full purity implies register purity — the GUI badge keys off the latter.
        assert!(Function::from_id(&tc.ctx, body).is_pure_reg());
        // The root block carries a real label (not an opaque `<bb_N>` fallback).
        assert!(
            BasicBlock::from_id(&tc.ctx, root).name().is_some(),
            "the outlined root block is named for display"
        );
    }

    /// An expression that reaches a memory load is not a closed pure function of
    /// its inputs — outlining must refuse it.
    #[test]
    fn refuses_open_expression_with_load() {
        let mut tc = TestContext::new();
        let host = Function::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let ram = tc.ctx.default_space;
        let (idx, result) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let idx = b.push_param(8).id();
            // A load is an untracked source: the expression is not closed.
            let loaded = b.push_load::<false>(idx, 8, ram).id();
            let result = b.push_add(idx, loaded).id();
            (idx, result)
        };

        assert!(
            outline_expression(&mut tc.ctx, "body", result, &[idx]).is_none(),
            "an expression reaching a load must not outline"
        );
    }

    /// `outline_tupled` produces a unary body taking the `(index, elem)` tuple and
    /// unpacking it with two extracts before recomputing `index + zext(elem)`.
    #[test]
    fn outlines_tupled_index_aware_body() {
        let mut tc = TestContext::new();
        let host = Function::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let (idx, elem, result) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let idx = b.push_param(8).id();
            let elem = b.push_param(1).id();
            let widened = b.push_zext(elem, 8).id();
            let result = b.push_add(idx, widened).id();
            (idx, elem, result)
        };

        // The enumerate tuple `(index: i64, elem: i8)`, via enumerate's own rule.
        let i8 = tc.ctx.types.get_or_make_int(1);
        let arr_ty = tc.ctx.types.get_or_make_array(i8, 1);
        let enum_id = IntrinsicId::from_name("enumerate").unwrap();
        let enum_ty = enum_id.desc().result_type(&mut tc.ctx.types, &[arr_ty]);
        let (tuple_ty, _) = tc.ctx.types.array_of(enum_ty).unwrap();

        let body = outline_tupled(&mut tc.ctx, "body", result, idx, elem, tuple_ty)
            .expect("expression is closed over (idx, elem)");

        // A single param — the tuple — sized to the `(i64, i8)` aggregate.
        let params: Vec<usize> = Function::from_id(&tc.ctx, body)
            .root()
            .unwrap()
            .params()
            .map(|p| p.size())
            .collect();
        assert_eq!(params, vec![tc.ctx.types.size_of(tuple_ty)]);

        // The body unpacks the tuple (two extracts) and recomputes zext + add.
        let root = Function::from_id(&tc.ctx, body).root().unwrap().id;
        let extracts = BasicBlock::from_id(&tc.ctx, root)
            .iter()
            .filter(|i| matches!(i.mnemonic(), Mnemonic::Extract(_)))
            .count();
        assert_eq!(
            extracts, 2,
            "the body unpacks index and elem from the tuple"
        );
        let has_zext = BasicBlock::from_id(&tc.ctx, root)
            .iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Zext(_)));
        let has_add = BasicBlock::from_id(&tc.ctx, root)
            .iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)));
        assert!(has_zext && has_add, "body recomputes zext + add");
        assert!(Function::from_id(&tc.ctx, body).is_pure());
    }
}
