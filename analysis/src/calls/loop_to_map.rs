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

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    builder::Builder,
    context::Context,
    space::{Space, SpaceId, SpaceType},
    types::TypeId,
    value::{
        BasicBlock, BlockId, Function, FunctionId, Instruction, InstructionRef, Renameable,
        ValueId,
        insn::{
            Binary, Binop, Branch, CBranch, Extract, InstructionId, IntBinop, IntrinsicId,
            Mnemonic, Range, Return, Unary, Unop,
        },
    },
};

use crate::gvn::affine::precompute_forms;
use crate::sequence::{affine_base_const, affine_strided_lane};
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

/// Outline a [`Scan`](qcode::value::insn::Scan) body: a **binary** function
/// `body(acc, tuple)` where `acc` is the carried accumulator (param 0, typed as
/// `acc_ty`) and `tuple` is the `enumerate` element `(index, elem)` (param 1).
/// The host `acc_input` is bound to param 0 and `index_input` to `Extract(t, 0)`.
/// When the original loop also reads the current source lane, `elem_input` is
/// bound to `Extract(t, 1)`, so the body can depend on `(acc, index, elem)`.
///
/// Returns `None` if the expression is not closed over those inputs + literals.
pub(crate) fn outline_scan_body(
    ctx: &mut Context,
    name: &str,
    result: ValueId,
    acc_input: ValueId,
    index_input: ValueId,
    elem_input: Option<ValueId>,
    index_start: i64,
    acc_ty: TypeId,
    index_ty: TypeId,
    tuple_ty: TypeId,
) -> Option<FunctionId> {
    let mut inputs = vec![acc_input, index_input];
    if let Some(elem) = elem_input {
        inputs.push(elem);
    }
    let slice = pure_slice(ctx, result, &inputs)?;
    outline_core(ctx, name, result, &slice, move |ctx, root| {
        let mut value_map: HashMap<ValueId, ValueId> = HashMap::default();
        // Param 0: the accumulator.
        let acc_sz = ctx.types.size_of(acc_ty);
        let apid = BasicBlock::from_id_mut(ctx, root).push_param(acc_sz).id;
        ctx.values.block_params[apid].type_id = acc_ty;
        value_map.insert(acc_input, ValueId::BlockParam(apid));
        // Param 1: the enumerate tuple. The body's loop index is
        // `(index_ty)(t.0) + index_start` — the `i64` enumerate index narrowed to
        // the loop index's width, shifted so `enumerate index 0` maps to the loop's
        // first index (it may count from 1 while the array is 0-based).
        let tsz = ctx.types.size_of(tuple_ty);
        let pid = BasicBlock::from_id_mut(ctx, root).push_param(tsz).id;
        ctx.values.block_params[pid].type_id = tuple_ty;
        let tuple = ValueId::BlockParam(pid);
        let fty = ctx
            .types
            .field_type(tuple_ty, 0)
            .expect("enumerate tuple index field");
        let mut idx = InstructionRef::from_mnemonic_with_type(
            ctx,
            Mnemonic::Extract(Extract {
                agg: tuple,
                index: 0,
            }),
            fty,
        )
        .id;
        BasicBlock::from_id_mut(ctx, root).push_insn(idx);
        // Narrow `i64` index → loop index width.
        let isz = ctx.types.size_of(index_ty);
        if isz < ctx.types.size_of(fty) {
            let r = InstructionRef::from_mnemonic_with_type(
                ctx,
                Mnemonic::Range(Range {
                    src: ValueId::Instruction(idx),
                    start: 0,
                    size: isz,
                }),
                index_ty,
            )
            .id;
            BasicBlock::from_id_mut(ctx, root).push_insn(r);
            idx = r;
        }
        // Shift by the start index.
        if index_start != 0 {
            let mask = if isz >= 8 {
                u64::MAX
            } else {
                (1u64 << (isz * 8)) - 1
            };
            let c = ctx.get_const((index_start as u64) & mask, isz).id();
            let add = InstructionRef::from_mnemonic_with_type(
                ctx,
                Mnemonic::Binop(Binary {
                    lhs: ValueId::Instruction(idx),
                    rhs: c,
                    op: Binop::Int(IntBinop::Add),
                }),
                index_ty,
            )
            .id;
            BasicBlock::from_id_mut(ctx, root).push_insn(add);
            idx = add;
        }
        value_map.insert(index_input, ValueId::Instruction(idx));
        if let Some(elem_input) = elem_input {
            let elem_ty = ctx
                .types
                .field_type(tuple_ty, 1)
                .expect("enumerate tuple elem field");
            let elem = InstructionRef::from_mnemonic_with_type(
                ctx,
                Mnemonic::Extract(Extract {
                    agg: tuple,
                    index: 1,
                }),
                elem_ty,
            )
            .id;
            BasicBlock::from_id_mut(ctx, root).push_insn(elem);
            value_map.insert(elem_input, ValueId::Instruction(elem));
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
// The load-base and store-base need not coincide. In the shape above they are
// the same region (an in-place read-modify-write). A two-buffer copy/transform
// `dst[i] = body(src[i])` instead reads `*[shadow]:1 (base_src+i)` and writes
// `*[shadow]:1 (base_dst+i)` with a distinct destination base, whose wide reload
// is the write-set; the source is still seeded from `@arr`. Both bases are root
// params, so distinct ones name provably disjoint shadow regions. The in-place
// case is one region of four accesses; the two-buffer case is two disjoint
// regions of two (seed+load on the source, store+reload on the destination).
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
    let base_src = seed.ptr;
    let count = seed.size;
    // v1: byte lanes only.
    let (elem_ty, arr_count) = ctx.types.array_of(ctx.stored_type_of(arr)?)?;
    if ctx.types.size_of(elem_ty) != 1 || arr_count != count {
        return None;
    }

    // Wide reload: `*[shadow]:N base_dst` — the write-set value. Its base may be
    // the seeded source base (an in-place read-modify-write) *or* a distinct
    // destination base (a two-buffer `dst[i] = body(src[i])` copy/transform). It
    // must be a root param so its shadow region is provably disjoint from the
    // source's (distinct root params seed disjoint regions). Identified as the
    // unique wide (`size == count`) load; the seed is a *store*, so it never
    // matches here.
    let wv = accesses
        .iter()
        .find(|a| a.stored.is_none() && a.size == count)?;
    let base_dst = wv.ptr;
    if !is_root(base_dst) {
        return None;
    }

    // Body store: `*[shadow]:1 (base_dst + idx) = v` — writes the destination lane.
    let body_store = accesses.iter().find(|a| {
        a.stored.is_some() && a.size == 1 && base_plus_param(ctx, a.ptr, base_dst).is_some()
    })?;
    let index = base_plus_param(ctx, body_store.ptr, base_dst)?;
    let stored_val = body_store.stored.unwrap();
    let body_block = body_store.block;

    // Body load: `*[shadow]:1 (base_src + idx)` — reads the source at the *same*
    // lane index. Matched by the index param (not the address value): in the
    // two-buffer case the source and destination addresses differ even though
    // they share the induction variable.
    let body_load = accesses.iter().find(|a| {
        a.stored.is_none() && a.size == 1 && base_plus_param(ctx, a.ptr, base_src) == Some(index)
    })?;
    let elem_val = ValueId::Instruction(body_load.id);

    // Region exactness: nothing beyond these four accesses may touch either
    // region — otherwise it is not a clean total map. In-place (base_src ==
    // base_dst) is a single region of 4 accesses; two-buffer is two disjoint
    // regions of 2 each (seed + load on the source, store + wide reload on the
    // destination). Shadow accesses to *unrelated* bases (e.g. argpromote's frame
    // seed stores) are disjoint from both and ignored.
    let touches =
        |base: ValueId, a: &Access| a.ptr == base || base_plus_param(ctx, a.ptr, base).is_some();
    let src_region = accesses.iter().filter(|a| touches(base_src, a)).count();
    if base_src == base_dst {
        if src_region != 4 {
            return None;
        }
    } else {
        let dst_region = accesses.iter().filter(|a| touches(base_dst, a)).count();
        if src_region != 2 || dst_region != 2 {
            return None;
        }
    }

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

// ===========================================================================
// Scan recognizer — a map with a loop-carried accumulator
// ===========================================================================
//
// The total-map shape above is element-local: `out[i] = body(arr[i], i)`. A loop
// whose per-element write depends on the *previous* iteration's result —
// `out[i] = body(out[i-1], i)` — cannot be a map (its body reaches a free
// loop-carried param), but it is a left-`scan`. The MT19937 seeding loop
// `mt[i] = 1812433253 * (mt[i-1] ^ (mt[i-1] >> 30)) + i` is the motivating case.
//
// Post-argpromote shadow shape (in-place; the previous value is threaded through
// a register/param, not re-read from memory). Some recurrences also read the
// current source lane before overwriting it:
//
//   <entry @base @arr:[i8;N] @init>
//       *[shadow]:N @base = @arr;                  // seed
//       goto <head @i=0 @acc=@init>;
//   <head @i @acc>  if @i < N goto <body> else goto <exit …>;
//   <body>      %b = *[shadow]:1 (@base+@i);        // optional source lane
//               %v = body(@acc, @i[, %b]);          // pure, depends on @acc
//               *[shadow]:1 (@base+@i) = %v;        // store lane
//               goto <head @i=@i+1 @acc=%v>;        // carry acc' = v
//   <exit>      %wv = *[shadow]:N @base;            // write-set value
//
// The recognizer proves: a counted index `[0,N)`, a second header param `@acc`
// whose back-edge value is exactly the stored value and whose preheader value is
// the initial accumulator, a body that is a closed pure expression of `(acc,
// index[, elem])` actually using `acc`, and an in-place region. It rewrites the
// write-set reload to `scan(init, enumerate(@arr), body)` and leaves the residual
// loop running (v1 does not delete the scan loop).

/// A recognized scan loop. Like [`MapMatch`] but with a carried accumulator.
struct ScanMatch {
    /// The `Array` snapshot seeded into the shadow region (the scan source).
    arr: ValueId,
    /// The header induction parameter (`@i`).
    index: ValueId,
    /// The index value at the array's first slot — the `idx` for which the lane
    /// address hits `arr[0]`. The scan body's `index` is therefore `enumerate
    /// index + index_start` (the loop may count from 1 while the array is 0-based).
    index_start: i64,
    /// The header accumulator parameter (`@acc`), threaded each iteration.
    acc: ValueId,
    /// The initial accumulator (`acc_0`), fed to `@acc` from the preheader.
    init: ValueId,
    /// The per-lane stored value (`%v`) — the body's result *and* the carry.
    stored_val: ValueId,
    /// Optional per-lane loaded source element. Present for scan loops whose body
    /// depends on both the previous accumulator and the current array lane.
    elem_val: Option<ValueId>,
    /// The wide shadow reload whose uses become the scan result (`%wv`).
    wv_id: InstructionId,
    /// The block holding `%wv` (where the scan is inserted).
    exit_block: BlockId,
}

/// Match a scan loop in `fid`, or `None` for any other shape.
fn try_match_scan(ctx: &Context, fid: FunctionId) -> Option<ScanMatch> {
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

    let numbering = precompute_forms(ctx, fid);
    let root_params: Vec<ValueId> = Function::from_id(ctx, fid)
        .root()?
        .params()
        .map(|p| p.id())
        .collect();
    let is_root = |v: ValueId| root_params.contains(&v);

    // Seed store: `*[shadow]:bytes (root + c_arr) = arr`, `arr` a root Array param.
    // The array body may sit at a non-zero offset of an incoming pointer (a struct
    // field) — MT19937's state is at `+8`, after the `index`/`seed` words — so the
    // seed address is `root + c_arr`, not a bare root param.
    let seed = accesses.iter().find_map(|a| {
        let src = a.stored?;
        if !is_root(src) {
            return None;
        }
        let (elem_ty, count) = ctx.types.array_of(ctx.stored_type_of(src)?)?;
        let (root, c_arr) = affine_base_const(&numbering, a.ptr, &is_root)?;
        let esz = ctx.types.size_of(elem_ty);
        (esz != 0 && count != 0 && a.size == count * esz).then_some((src, root, c_arr, esz, count))
    })?;
    let (arr, root, c_arr, esz, count) = seed;
    let bytes = count * esz;

    // Wide reload: `*[shadow]:bytes (root + c_arr)` — the write-set value.
    let wv = accesses.iter().find(|a| {
        a.stored.is_none()
            && a.size == bytes
            && affine_base_const(&numbering, a.ptr, &is_root) == Some((root, c_arr))
    })?;

    // Lane store: `*[shadow]:esz (root + idx*esz + c_lane) = v`, `idx` a block
    // param. No matching lane *load* exists (the previous value is carried, not
    // re-read) — that absence is the scan signature, enforced by the coverage
    // check below (only the seed/lane/reload touch the array region).
    let lane = accesses.iter().find_map(|a| {
        let v = a.stored?;
        if a.size != esz {
            return None;
        }
        let (base, idx, c_lane) = affine_strided_lane(&numbering, a.ptr, esz)?;
        (base == root).then_some((a.id, v, idx, c_lane, a.block))
    })?;
    let (lane_store_id, stored_val, index, c_lane, body_block) = lane;

    // Optional source lane load at the same indexed address. Older scan support
    // required this to be absent; MT-style recurrences often need both the
    // previous accumulator and the current original lane.
    let elem_val = accesses.iter().find_map(|a| {
        if a.id == wv.id || a.id == lane_store_id || a.stored.is_some() || a.size != esz {
            return None;
        }
        let (base, idx, c) = affine_strided_lane(&numbering, a.ptr, esz)?;
        (base == root && idx == index && c == c_lane).then_some(ValueId::Instruction(a.id))
    });

    // Coverage: the lane offsets `{idx*esz + c_lane : idx ∈ [min, max]}` must tile
    // the array region `[c_arr, c_arr + count*esz)` exactly. This subsumes the
    // start index (the loop may count from 1), the constant offset, and totality.
    let range = value_range(ctx, index, body_block);
    let esz_i = esz as i64;
    let lo_off = esz_i * range.min as i64 + c_lane;
    let hi_off = esz_i * range.max as i64 + c_lane;
    if lo_off != c_arr || hi_off != c_arr + (count as i64 - 1) * esz_i {
        return None;
    }
    // The index value at the first array slot (`arr[0]`): `idx = (c_arr - c_lane)/esz`.
    let index_start = range.min as i64;

    let ValueId::BlockParam(pid) = index else {
        return None;
    };
    let header = ctx.values.block_params[pid].parent?;
    let ki = BasicBlock::from_id(ctx, header)
        .params()
        .position(|p| p.id() == index)?;
    let steps_by_one = header_incoming(ctx, header, ki)
        .iter()
        .any(|&v| is_increment(ctx, v, index));
    if !steps_by_one {
        return None;
    }

    // Accumulator: a *distinct* header param whose back-edge value is exactly the
    // stored value (the carry `acc' = v`) and whose other incoming is the initial
    // accumulator. `header_incoming` yields one value per predecessor edge — the
    // preheader (init) and the body back-edge (stored_val).
    let header_params: Vec<ValueId> = BasicBlock::from_id(ctx, header)
        .params()
        .map(|p| p.id())
        .collect();
    let (acc, init) = header_params.iter().enumerate().find_map(|(ka, &p)| {
        if p == index {
            return None;
        }
        let inc = header_incoming(ctx, header, ka);
        let carries = inc.iter().filter(|&&v| v == stored_val).count();
        if carries != 1 {
            return None;
        }
        // The unique non-carry incoming is the initial accumulator.
        let inits: Vec<ValueId> = inc.into_iter().filter(|&v| v != stored_val).collect();
        let [init] = inits[..] else { return None };
        Some((p, init))
    })?;

    // The body must be a closed pure expression of `(acc, index[, elem])` that
    // actually uses the accumulator — otherwise it is index-local (a map/generate),
    // not a scan.
    let mut inputs = vec![acc, index];
    if let Some(elem) = elem_val {
        inputs.push(elem);
    }
    let slice = pure_slice(ctx, stored_val, &inputs)?;
    let uses_acc = stored_val == acc
        || slice
            .iter()
            .any(|&iid| ctx.get_insn(iid).mnemonic().args().contains(&acc));
    if !uses_acc {
        return None;
    }

    Some(ScanMatch {
        arr,
        index,
        index_start,
        acc,
        init,
        stored_val,
        elem_val,
        wv_id: wv.id,
        exit_block: wv.block,
    })
}

/// Rewrite a matched scan loop: outline the `(acc, index)` body and replace the
/// write-set reload with `scan(init, enumerate(@arr), body)`. The residual loop
/// is left running (v1 does not delete it). Returns `false` if the body is not a
/// closed pure expression (then nothing is changed).
fn apply_scan(ctx: &mut Context, fid: FunctionId, m: &ScanMatch) -> bool {
    let name = format!("{}_scan_body", Function::from_id(ctx, fid).name());
    let arr_ty = ctx.type_of(m.arr);
    let acc_ty = ctx.type_of(m.acc);
    let index_ty = ctx.type_of(m.index);
    let enum_id = IntrinsicId::from_name("enumerate").expect("enumerate registered");
    let enum_result_ty = enum_id.desc().result_type(&mut ctx.types, &[arr_ty]);
    let Some((tuple_ty, _)) = ctx.types.array_of(enum_result_ty) else {
        return false;
    };
    let Some(body_fn) = outline_scan_body(
        ctx,
        &name,
        m.stored_val,
        m.acc,
        m.index,
        m.elem_val,
        m.index_start,
        acc_ty,
        index_ty,
        tuple_ty,
    ) else {
        return false;
    };

    // enumerate(@arr) then scan(init, that, body), inserted before the wide reload.
    let wv_val = ValueId::Instruction(m.wv_id);
    let scan_val = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.exit_block));
        b.set_insert_point_before(m.wv_id);
        let enum_val = b.push_intrinsic(enum_id, vec![m.arr]).id();
        b.push_scan(body_fn, m.init, enum_val, Vec::new()).id()
    };
    ctx.replace_all_uses_with(wv_val, scan_val);
    ctx.remove_instruction(m.wv_id);
    let _ = fid;
    true
}

/// Recognize scan loops across all pure functions, rewriting each to a `scan`.
/// Returns `true` if anything changed.
pub(crate) fn recognize_scans(ctx: &mut Context) -> bool {
    let fids: Vec<FunctionId> = ctx.function_ids();
    let mut changed = false;
    for fid in fids {
        if !Function::from_id(ctx, fid).is_pure() {
            continue;
        }
        if let Some(m) = try_match_scan(ctx, fid) {
            changed |= apply_scan(ctx, fid, &m);
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
// Shape (post-argpromote snapshot): like the total-map shape but *read-only* and
// terminated by the loaded byte rather than a static index bound:
//
//   <entry @base @arr:[i8;N]>
//       *[shadow]:N @base = @arr;                 // seed
//       goto <head @i=0>;
//   <head @i>   %b = *[shadow]:1 (@base+@i);      // load lane
//               if %b != 0 goto <body> else goto <exit @i>;   // NUL test
//   <body>      goto <head @i=@i+1>;
//   <exit @count>  ... uses of @count = strlen ...
//
// The loaded byte governs the loop exit (the take_while predicate), and the index
// carried out on the exit edge is the scan length. v1: byte lanes, snapshot-backed
// source (the array value `@arr`); the unbounded raw-`char*` case is a follow-up.

/// A recognized bounded NUL-scan whose escaping count is `len(take_while(arr))`.
struct StrlenMatch {
    /// The `Array` snapshot scanned.
    arr: ValueId,
    /// The exit-block parameter holding the scan length (its uses become `len`).
    count_param: ValueId,
    /// The seed store `*[shadow]:N base = arr` (dead after the rewrite).
    seed_id: InstructionId,
    /// The block holding the seed store and the loop preheader branch.
    entry_block: BlockId,
    /// The loop header (carries the induction param, holds the NUL test).
    header_block: BlockId,
    /// The loop body (the per-iteration increment).
    body_block: BlockId,
    /// The exit block (where `len`/`take_while` are inserted; holds `count_param`).
    exit_block: BlockId,
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
    // Collect every shadow (temporary-space) access.
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

    // Seed store: `*[shadow]:N base = arr`, `arr` a root Array param, `base` root.
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
    // v1: byte lanes.
    let (elem_ty, arr_count) = ctx.types.array_of(ctx.stored_type_of(arr)?)?;
    if ctx.types.size_of(elem_ty) != 1 || arr_count != count {
        return None;
    }

    // The scan is read-only: a single per-lane load `*[shadow]:1 (base + idx)`,
    // and no store other than the seed. Exactly two accesses touch the region
    // (seed + lane load); any store-back or extra access means it is not a pure
    // NUL-scan (a copy/transform is the map recognizer's job).
    if accesses
        .iter()
        .any(|a| a.stored.is_some() && a.id != seed.id)
    {
        return None;
    }
    let lane = accesses.iter().find(|a| {
        a.stored.is_none() && a.size == 1 && base_plus_param(ctx, a.ptr, base).is_some()
    })?;
    let index = base_plus_param(ctx, lane.ptr, base)?;
    let elem_val = ValueId::Instruction(lane.id);
    let touches = |a: &Access| a.ptr == base || base_plus_param(ctx, a.ptr, base).is_some();
    if accesses.iter().filter(|a| touches(a)).count() != 2 {
        return None;
    }

    // The index is a header param initialised to 0 and stepped by +1 — so it counts
    // iterations from 0. (Unlike the map recognizer there is no static upper bound:
    // termination is the NUL test below, and the snapshot bound `N` caps it.)
    let ValueId::BlockParam(pid) = index else {
        return None;
    };
    let header = ctx.values.block_params[pid].parent?;
    // The lane load must live in the header: the NUL test that governs the loop
    // reads it there, and the count is the index at that test.
    if lane.block != header {
        return None;
    }
    let k = BasicBlock::from_id(ctx, header)
        .params()
        .position(|p| p.id() == index)?;
    // The index must start at 0 on *every* entry edge and step by +1 on the
    // back-edge: each non-increment (preheader) incoming has to be the literal 0,
    // or some entry could start the count off-zero and the rewrite would be wrong.
    let incoming = header_incoming(ctx, header, k);
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
            seed_id: seed.id,
            entry_block: seed.block,
            header_block: header,
            body_block,
            exit_block,
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
        //   3. strip the seed store and delete the dead loop blocks.
        let kx = BasicBlock::from_id(ctx, m.exit_block)
            .params()
            .position(|p| p.id() == m.count_param);
        if let Some(kx) = kx {
            crate::dce::remove_params_from_block(ctx, m.exit_block, &HashSet::from_iter([kx]));
        }
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
        ctx.remove_instruction(m.seed_id);
        BasicBlock::from_id_mut(ctx, m.body_block).delete(fid);
        BasicBlock::from_id_mut(ctx, m.header_block).delete(fid);
    }
    // Otherwise the loop also feeds outside consumers, so it keeps running with its
    // shadow channel intact; only the count was forwarded to `len` above.
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
            let incoming = header_incoming(ctx, header, k);
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
        changed |= recognize_scans(ctx);
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

    /// The map inserted into `block`, if any, with its source value.
    fn map_src_in(ctx: &Context, block: BlockId) -> Option<ValueId> {
        BasicBlock::from_id(ctx, block)
            .iter()
            .find_map(|i| match i.mnemonic() {
                Mnemonic::Map(m) => Some(m.src),
                _ => None,
            })
    }

    /// A two-buffer counted copy `dst[i] = src[i]` is recognized: the write-set
    /// reload over `base_dst` becomes `map(identity, arr)` over the *source*
    /// array, even though load-base and store-base differ.
    #[test]
    fn two_buffer_copy_recognized() {
        let mut tc = TestContext::new();
        let (fid, exit, arr) =
            build_copy_loop(&mut tc, /*distinct_dst*/ true, /*stray*/ false);

        assert!(
            recognize_total_maps(&mut tc.ctx),
            "two-buffer copy must recognize"
        );
        assert_eq!(
            map_src_in(&tc.ctx, exit),
            Some(arr),
            "the map source is the source array @arr"
        );
        // The host stays pure and the body was outlined as a pure function.
        assert!(Function::from_id(&tc.ctx, fid).is_pure());
    }

    /// Regression: the in-place single-buffer case (`base_dst == base_src`) still
    /// recognizes after the two-buffer generalization.
    #[test]
    fn in_place_single_buffer_still_recognized() {
        let mut tc = TestContext::new();
        let (_fid, exit, arr) =
            build_copy_loop(&mut tc, /*distinct_dst*/ false, /*stray*/ false);

        assert!(
            recognize_total_maps(&mut tc.ctx),
            "in-place map must still recognize"
        );
        assert_eq!(map_src_in(&tc.ctx, exit), Some(arr));
    }

    /// A stray extra access to the destination region breaks region exactness
    /// (`dst_region != 2`), so the loop is left unrecognized — no map inserted.
    #[test]
    fn stray_destination_access_rejects() {
        let mut tc = TestContext::new();
        let (_fid, exit, _arr) =
            build_copy_loop(&mut tc, /*distinct_dst*/ true, /*stray*/ true);

        assert!(
            !recognize_total_maps(&mut tc.ctx),
            "a stray dst access must not recognize"
        );
        assert_eq!(map_src_in(&tc.ctx, exit), None, "no map is inserted");
    }

    // ===== recognizer (scan) ===============================================

    /// Build a scan loop `out[i] = acc; acc = acc + 1` over an `N`-lane snapshot
    /// of `esz`-byte elements, in the canonical post-argpromote shape: a carried
    /// accumulator threaded through a header param, an in-place lane store with
    /// **no** matching lane load, and a wide write-set reload. The lane address is
    /// `base + i` for `esz == 1` and `base + i*esz` for a wider lane (the MT19937
    /// shape).
    ///   entry: seed `*[shadow]:N·esz base = arr`; goto head(i=0, acc=init)
    ///   head:  if i < N goto body else goto exit
    ///   body:  v = acc + 1; *[shadow]:esz (base + i*esz) = v; goto head(i+1, acc=v)
    ///   exit:  wv = *[shadow]:N·esz base; store(ram, &out = wv); return
    fn build_scan_loop(
        tc: &mut TestContext,
        esz: usize,
        read_lane: bool,
    ) -> (FunctionId, BlockId, ValueId, ValueId) {
        const N: usize = 4;
        let elem = tc.ctx.types.get_or_make_int(esz);
        let arr_ty = tc.ctx.types.get_or_make_array(elem, N);
        let shadow = tc.ctx.make_temp_space();
        let ram = tc.ctx.default_space;

        let fid = Function::make(&mut tc.ctx, "scanfn".into()).unwrap().id;
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

        // Root params: arr:[elem;N], base, init:elem (the initial accumulator).
        let arr_pid = BasicBlock::from_id_mut(&mut tc.ctx, entry)
            .push_param(N * esz)
            .id;
        tc.ctx.values.block_params[arr_pid].type_id = arr_ty;
        let arr = ValueId::BlockParam(arr_pid);
        let base =
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id);
        let init = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, entry)
                .push_param(esz)
                .id,
        );

        // Header params: induction `i` (width 8) then accumulator `acc` (elem).
        let i = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(8)
                .id,
        );
        let acc = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(esz)
                .id,
        );

        // entry: seed store + preheader branch carrying (i=0, acc=init).
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let zero = b.context_mut().get_const(0, 8).id();
            b.push_store(arr, base, shadow);
            b.push_branch_with_args(header, vec![zero, init]);
        }
        // header: counted guard.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, header));
            let n = b.context_mut().get_const(N as u64, 8).id();
            let cond = b.push_lt(i, n).id();
            b.push_cbranch_with_args(cond, body, vec![], exit, vec![]);
        }
        // body: v = acc + 1, or `acc + arr[i] + 1` when `read_lane`; store lane
        // at `base + i*esz`; carry (i+1, acc=v).
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, body));
            let addr = if esz == 1 {
                b.push_add(base, i).id()
            } else {
                let e = b.context_mut().get_const(esz as u64, 8).id();
                let scaled = b.push_mul(i, e).id();
                b.push_add(base, scaled).id()
            };
            let one_e = b.context_mut().get_const(1, esz).id();
            let v = if read_lane {
                let elem = b.push_load::<false>(addr, esz, shadow).id();
                let sum = b.push_add(acc, elem).id();
                b.push_add(sum, one_e).id()
            } else {
                b.push_add(acc, one_e).id()
            };
            b.push_store(v, addr, shadow);
            let one8 = b.context_mut().get_const(1, 8).id();
            let inc = b.push_add(i, one8).id();
            b.push_branch_with_args(header, vec![inc, v]);
        }
        // exit: wide reload (write-set) + external consumer + return.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, exit));
            let wv = b.push_load::<false>(base, N * esz, shadow).id();
            let out = b.context_mut().get_const(0x9000, 8).id();
            b.push_store(wv, out, ram);
            let dummy = b.context_mut().get_const(0, 8).id();
            b.push_return(dummy);
        }

        Function::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);
        (fid, exit, arr, init)
    }

    /// The scan inserted into `block`, if any, with its `(init, src)`.
    fn scan_in(ctx: &Context, block: BlockId) -> Option<(ValueId, ValueId)> {
        BasicBlock::from_id(ctx, block)
            .iter()
            .find_map(|i| match i.mnemonic() {
                Mnemonic::Scan(s) => Some((s.init, s.src)),
                _ => None,
            })
    }

    /// A loop whose per-lane store depends on a carried accumulator is recognized
    /// as a `scan(init, enumerate(@arr), body)`, with a freshly outlined pure body.
    /// Assert that `fid`'s scan loop is recognized and rewritten to
    /// `scan(@init, enumerate(@arr), body)` with a fresh pure body.
    fn assert_scan_recognized(
        tc: &mut TestContext,
        fid: FunctionId,
        exit: BlockId,
        arr: ValueId,
        init: ValueId,
    ) {
        // A map recognizer must NOT claim it (the carried body is not a map).
        assert!(
            !recognize_total_maps(&mut tc.ctx),
            "a carried-accumulator loop is not a total map"
        );
        assert!(recognize_scans(&mut tc.ctx), "the scan must be recognized");

        let (scan_init, scan_src) =
            scan_in(&tc.ctx, exit).expect("a scan is inserted at the write-set reload");
        assert_eq!(scan_init, init, "the scan is seeded with @init");
        let ValueId::Instruction(src_id) = scan_src else {
            panic!("scan src should be the enumerate result");
        };
        match tc.ctx.get_insn(src_id).mnemonic() {
            Mnemonic::Intrinsic(app) => {
                assert_eq!(
                    app.id.desc().name(),
                    "enumerate",
                    "scan src is enumerate(...)"
                );
                assert_eq!(app.args, vec![arr], "enumerate is over @arr");
            }
            other => panic!("expected enumerate intrinsic, got {other:?}"),
        }
        let has_body = tc.ctx.function_ids().iter().any(|&f| {
            Function::from_id(&tc.ctx, f).name().ends_with("_scan_body")
                && Function::from_id(&tc.ctx, f).is_pure()
        });
        assert!(has_body, "a pure scan body function was outlined");
        assert!(
            Function::from_id(&tc.ctx, fid).is_pure(),
            "the host stays pure"
        );
    }

    /// A byte-lane loop whose per-lane store depends on a carried accumulator is
    /// recognized as a `scan`.
    #[test]
    fn carried_accumulator_loop_recognized_as_scan() {
        let mut tc = TestContext::new();
        let (fid, exit, arr, init) =
            build_scan_loop(&mut tc, /*esz*/ 1, /*read_lane*/ false);
        assert_scan_recognized(&mut tc, fid, exit, arr, init);
    }

    /// A scan body may also read the current source lane: `v = acc + arr[i] + 1`.
    /// This is the recurrence shape needed before MT-style twist loops can be
    /// lifted: the lane update is not element-local because it uses the previous
    /// accumulator, but it is also not seed-only because it reads the current lane.
    #[test]
    fn carried_accumulator_with_lane_load_recognized_as_scan() {
        let mut tc = TestContext::new();
        let (fid, exit, arr, init) =
            build_scan_loop(&mut tc, /*esz*/ 4, /*read_lane*/ true);
        assert_scan_recognized(&mut tc, fid, exit, arr, init);
    }

    /// The same scan with **i32 lanes** and strided `base + i*4` addressing (the
    /// MT19937 shape) is recognized — the lane-width + scaled-address generalization.
    #[test]
    fn wide_lane_scan_recognized() {
        let mut tc = TestContext::new();
        let (fid, exit, arr, init) =
            build_scan_loop(&mut tc, /*esz*/ 4, /*read_lane*/ false);
        assert_scan_recognized(&mut tc, fid, exit, arr, init);
    }

    /// The real MT19937 `init_genrand` shadow shape: the array snapshot sits at a
    /// **byte offset** of an incoming pointer (`base + 8`, a struct field), the lane
    /// address carries a **constant** (`base + i*4 + 4`), the index counts **from 1**,
    /// and the loop is a **single self-looping block**. Exercises the affine
    /// decomposition + index-offset path.
    #[test]
    fn struct_offset_scan_recognized() {
        const N: usize = 4;
        let mut tc = TestContext::new();
        let i32t = tc.ctx.types.get_or_make_int(4);
        let arr_ty = tc.ctx.types.get_or_make_array(i32t, N);
        let shadow = tc.ctx.make_temp_space();
        let ram = tc.ctx.default_space;

        let fid = Function::make(&mut tc.ctx, "mt".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let head = tc.ctx.get_or_make_block(0x1010);
        let exit = tc.ctx.get_or_make_block(0x1020);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(head);
            f.add_block(exit);
        }

        // Root params: base (incoming pointer), arr:[i32;N] snapshot, init.
        let base =
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(4).id);
        let arr_pid = BasicBlock::from_id_mut(&mut tc.ctx, entry)
            .push_param(N * 4)
            .id;
        tc.ctx.values.block_params[arr_pid].type_id = arr_ty;
        let arr = ValueId::BlockParam(arr_pid);
        let init =
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(4).id);

        // Self-loop header params: accumulator `ebx`, induction `i`.
        let ebx = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, head).push_param(4).id);
        let i = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, head).push_param(4).id);

        // entry: seed the array at `base + 8`; enter the loop with (ebx=init, i=1).
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let eight = b.context_mut().get_const(8, 4).id();
            let seedaddr = b.push_add(base, eight).id();
            b.push_store(arr, seedaddr, shadow);
            let one = b.context_mut().get_const(1, 4).id();
            b.push_branch_with_args(head, vec![init, one]);
        }
        // head (single self-loop block): v = ebx + i; store at base + i*4 + 4;
        // carry (ebx=v, i=i+1) while i+1 < N+1.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, head));
            let v = b.push_add(ebx, i).id();
            let four = b.context_mut().get_const(4, 4).id();
            let off = b.push_mul(i, four).id();
            let a = b.push_add(off, base).id();
            let addr = b.push_add(a, four).id();
            b.push_store(v, addr, shadow);
            let one = b.context_mut().get_const(1, 4).id();
            let ni = b.push_add(i, one).id();
            let bound = b.context_mut().get_const(N as u64 + 1, 4).id();
            let c = b.push_lt(ni, bound).id();
            b.push_cbranch_with_args(c, head, vec![v, ni], exit, vec![]);
        }
        // exit: wide reload at `base + 8` (write-set) + external consumer.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, exit));
            let eight = b.context_mut().get_const(8, 4).id();
            let seedaddr = b.push_add(base, eight).id();
            let wv = b.push_load::<false>(seedaddr, N * 4, shadow).id();
            let out = b.context_mut().get_const(0x9000, 8).id();
            b.push_store(wv, out, ram);
            let dummy = b.context_mut().get_const(0, 8).id();
            b.push_return(dummy);
        }
        Function::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);

        assert!(
            recognize_scans(&mut tc.ctx),
            "the MT struct-offset shadow shape must be recognized as a scan"
        );
        let (scan_init, _) = scan_in(&tc.ctx, exit).expect("a scan is inserted");
        assert_eq!(scan_init, init, "the scan is seeded with @init");
        let has_body = tc.ctx.function_ids().iter().any(|&f| {
            Function::from_id(&tc.ctx, f).name().ends_with("_scan_body")
                && Function::from_id(&tc.ctx, f).is_pure()
        });
        assert!(has_body, "a pure scan body was outlined");
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

    /// A bounded NUL-scan over a snapshot is rewritten so its escaping length is
    /// `len(take_while(@arr))` over the source array.
    #[test]
    fn bounded_nul_scan_recognized_as_len_take_while() {
        let mut tc = TestContext::new();
        let (fid, exit, arr) = build_strlen_loop(&mut tc, /*extra_break*/ false);

        assert!(
            recognize_strlens(&mut tc.ctx),
            "bounded NUL-scan must recognize"
        );
        assert_eq!(
            len_take_while_src(&tc.ctx, exit),
            Some(arr),
            "the escaping count becomes len(take_while(@arr))"
        );
        assert!(Function::from_id(&tc.ctx, fid).is_pure());
    }

    /// A loop with a second data-dependent exit (a `break` besides the NUL test) is
    /// not a clean `take_while`: its count is the first of *either* terminator, not
    /// the first zero, so recognition must decline.
    #[test]
    fn second_break_is_not_a_strlen() {
        let mut tc = TestContext::new();
        let (_fid, exit, _arr) = build_strlen_loop(&mut tc, /*extra_break*/ true);

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

    /// A copy loop (which *stores*) is not a NUL-scan strlen — the read-only gate
    /// rejects it, leaving it for the map recognizer.
    #[test]
    fn copy_loop_is_not_a_strlen() {
        let mut tc = TestContext::new();
        let (_fid, exit, _arr) =
            build_copy_loop(&mut tc, /*distinct_dst*/ true, /*stray*/ false);

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
