//! Form-agnostic outlining machinery shared by the loop recognizers.
//!
//! [`outline_expression`] / [`outline_tupled`] / [`outline_scan_body`] copy a pure
//! expression DAG out of a host loop body into a fresh standalone pure function
//! (the mechanical core is [`outline_core`]); [`inline_pure_body`] is the dual,
//! splicing such a body back in with its parameters bound to concrete arguments
//! (used by [`gvn/array_project`](crate::gvn) to project a `map` element). These
//! are used by [`loop_to_map`](super::loop_to_map),
//! [`loop_to_scan`](super::loop_to_scan), and the element-projection rewrite; they
//! know nothing about any particular recognizer's shape.

use rustc_hash::FxHashMap as HashMap;

use qcode::{
    context::Context,
    types::TypeId,
    value::{
        BasicBlock, BlockId, BodyView, FunctionBody, FunctionId, FunctionKind, InstructionRef,
        LocalValueId, QCodeView, ValueId, VarnodeId,
        block_param::BlockParam,
        insn::{Binary, Binop, Callee, Extract, InstructionId, IntBinop, Mnemonic, Range, Return},
        util::{base_ref::BaseRef, pass_backing::PassBacking},
    },
};

use crate::pipeline::{ContextView, Minted};

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
pub(crate) fn pure_slice<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    result: ValueId,
    inputs: &[ValueId],
) -> Option<Vec<InstructionId>> {
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
        let m = host.instruction(id).mnemonic().clone();
        if !is_pure_expr_op(&m) {
            return None;
        }
        stack.push((v, true));
        for a in m.args() {
            stack.push((a.qualify(id.func), false));
        }
    }
    Some(order)
}

/// Outline the pure expression that computes `result` into a fresh standalone
/// function `body(inputs…) -> result`, marked
/// [`is_pure`](qcode::value::FunctionRef::is_pure).
/// Each value in `inputs` becomes a parameter (in order, with the input's
/// width/type); every instruction on the backward slice is cloned with operands
/// remapped (inputs→params, clones→clones, literals passed through); the function
/// returns the cloned `result`.
///
/// Returns `None` if the expression is not closed over `inputs` + literals (see
/// [`pure_slice`]) — the caller then leaves the loop unrecognized.
pub(crate) fn outline_expression<'str>(
    m: ContextView<'_, 'str>,
    body: &mut FunctionBody<'str>,
    next_minted: &mut u32,
    minted_out: &mut Vec<Minted<'str>>,
    name: &str,
    result: ValueId,
    inputs: &[ValueId],
) -> Option<Callee> {
    let slice = pure_slice(m.body_view(body), result, inputs)?;
    let inputs = inputs.to_vec();
    outline_core(
        m,
        body,
        next_minted,
        minted_out,
        name,
        result,
        &slice,
        move |own, minted, root| {
            // Parameters, in input order, typed as the host inputs.
            let mut value_map: HashMap<ValueId, ValueId> = HashMap::default();
            for inp in inputs {
                let ty = own.type_of(inp);
                let pid = push_param_into(minted, root, ty);
                value_map.insert(inp, pid);
            }
            value_map
        },
    )
}

/// Outline a body that takes a single `enumerate` tuple param `(index, elem)` and
/// unpacks it: the host `index_input` / `elem_input` are bound to `Extract(t, 0)`
/// / `Extract(t, 1)` of the tuple param `t` before the expression computing
/// `result` is cloned. This is the body shape for a `map` over `enumerate(arr)`
/// (rather than over `arr`) — a unary body whose element is the index/value pair.
///
/// `elem_input` is `None` for a pure index-driven generation whose body reads the
/// index but not the lane element (the `t.1` extract is then omitted); the map
/// still ranges over `enumerate(arr)` so the body receives the index in `t.0`.
///
/// Returns `None` if the expression is not closed over `(index, elem?)` + literals
/// (see [`pure_slice`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn outline_tupled<'str>(
    m: ContextView<'_, 'str>,
    body: &mut FunctionBody<'str>,
    next_minted: &mut u32,
    minted_out: &mut Vec<Minted<'str>>,
    name: &str,
    result: ValueId,
    index_input: ValueId,
    elem_input: Option<ValueId>,
    tuple_ty: TypeId,
) -> Option<Callee> {
    let mut inputs = vec![index_input];
    inputs.extend(elem_input);
    let slice = pure_slice(m.body_view(body), result, &inputs)?;
    outline_core(
        m,
        body,
        next_minted,
        minted_out,
        name,
        result,
        &slice,
        move |own, minted, root| {
            let mut value_map: HashMap<ValueId, ValueId> = HashMap::default();
            let tuple = push_param_into(minted, root, tuple_ty);
            // index = t.0, elem = t.1 — the extracts the body unpacks. `elem` is only
            // extracted when the body actually consumes the lane element.
            let fields = [(0usize, Some(index_input)), (1usize, elem_input)];
            for (field, input) in fields {
                let Some(input) = input else { continue };
                let fty = own
                    .shared()
                    .types
                    .field_type(tuple_ty, field)
                    .expect("enumerate tuple field");
                let ex = push_insn_into(
                    minted,
                    root,
                    Mnemonic::Extract(Extract {
                        agg: tuple.localize(root.func),
                        index: field,
                    }),
                    fty,
                );
                value_map.insert(input, ex);
            }
            value_map
        },
    )
}

/// How a [`Scan`](qcode::value::insn::Scan) body receives its per-lane input
/// (param 1), which the body turns into the loop index.
///
/// - [`Scalar`](ScanElem::Scalar): the source is a bare index driver (an `iota`),
///   so param 1 *is* the index element directly. There is no data lane, so the
///   only array the scan touches is the freshly-built `iota` — no original memory
///   is read. Used by [`loop_to_scan`](crate::calls::loop_to_scan).
pub(crate) enum ScanElem {
    /// The scan source's scalar element type (the `iota` element, e.g. `i64`).
    Scalar(TypeId),
    /// The scan ranges over a real data array (`l0[1..]`); param 1 *is* the data
    /// element, bound to `elem_input`. The loop index is not exposed (the body of
    /// this shape depends only on the accumulator and the current element, e.g. a
    /// prefix sum `acc + l[i]`). Used by `loop_to_scan`'s array-input path.
    Data(TypeId),
}

/// Outline a [`Scan`](qcode::value::insn::Scan) body: a **binary** function
/// `body(acc, x)` where `acc` is the carried accumulator (param 0, typed as
/// `acc_ty`) and `x` is the per-lane input (param 1), whose shape is set by
/// `elem`. The host `acc_input` is bound to param 0 and `index_input` to the
/// derived loop index. [`ScanElem::Scalar`] mode has no data lane, so `elem_input`
/// must be `None`; [`ScanElem::Data`] mode binds the current element to
/// `elem_input` and does not expose the index.
///
/// Returns `None` if the expression is not closed over those inputs + literals,
/// or if `elem_input` is given in scalar mode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn outline_scan_body<'str>(
    m: ContextView<'_, 'str>,
    body: &mut FunctionBody<'str>,
    next_minted: &mut u32,
    minted_out: &mut Vec<Minted<'str>>,
    name: &str,
    result: ValueId,
    acc_input: ValueId,
    index_input: ValueId,
    elem_input: Option<ValueId>,
    index_start: i64,
    acc_ty: TypeId,
    index_ty: TypeId,
    elem: ScanElem,
) -> Option<Callee> {
    let inputs: Vec<ValueId> = match elem {
        // A scalar source has no separate data lane to bind.
        ScanElem::Scalar(_) => {
            if elem_input.is_some() {
                return None;
            }
            vec![acc_input, index_input]
        }
        // Data source: the body reads only the accumulator and the current element;
        // the index is not exposed.
        ScanElem::Data(_) => vec![acc_input, elem_input?],
    };
    let slice = pure_slice(m.body_view(body), result, &inputs)?;
    outline_core(
        m,
        body,
        next_minted,
        minted_out,
        name,
        result,
        &slice,
        move |own, minted, root| {
            let mut value_map: HashMap<ValueId, ValueId> = HashMap::default();
            // Param 0: the accumulator.
            let apid = push_param_into(minted, root, acc_ty);
            value_map.insert(acc_input, apid);
            // Param 1: the per-lane input. The body's loop index is
            // `(index_ty)(raw) + index_start` — the `i64` driver value narrowed to the
            // loop index's width, shifted so element 0 maps to the loop's first index
            // (it may count from 1 while the array is 0-based). In tuple mode `raw` is
            // `t.0` of the `enumerate` element; in scalar mode `raw` is the param
            // itself (the `iota` lane).
            let param_ty = match elem {
                ScanElem::Scalar(elem_ty) => elem_ty,
                ScanElem::Data(elem_ty) => elem_ty,
            };
            let param = push_param_into(minted, root, param_ty);
            // Data mode: the param *is* the element; bind it and skip index derivation.
            if let ScanElem::Data(_) = elem {
                value_map.insert(elem_input.expect("data mode requires elem_input"), param);
                return value_map;
            }
            // `idx` is the raw index driver value before narrow/shift: in scalar mode
            // the param itself (the `iota` lane), directly.
            let (mut idx, fty): (ValueId, TypeId) = match elem {
                // Scalar: the param is the index driver value directly.
                ScanElem::Scalar(elem_ty) => (param, elem_ty),
                // Data mode returned above.
                ScanElem::Data(_) => unreachable!("data mode handled above"),
            };
            let types = &own.shared().types;
            // Narrow `i64` index → loop index width.
            let isz = types.size_of(index_ty);
            if isz < types.size_of(fty) {
                let r = push_insn_into(
                    minted,
                    root,
                    Mnemonic::Range(Range {
                        src: idx.localize(root.func),
                        start: 0,
                        size: isz,
                    }),
                    index_ty,
                );
                idx = r;
            }
            // Shift by the start index.
            if index_start != 0 {
                let mask = if isz >= 8 {
                    u64::MAX
                } else {
                    (1u64 << (isz * 8)) - 1
                };
                let c = own.shared().get_const((index_start as u64) & mask, isz);
                let add = push_insn_into(
                    minted,
                    root,
                    Mnemonic::Binop(Binary {
                        lhs: idx.localize(root.func),
                        rhs: c.localize(root.func),
                        op: Binop::Int(IntBinop::Add),
                    }),
                    index_ty,
                );
                idx = add;
            }
            value_map.insert(index_input, idx);
            value_map
        },
    )
}

/// The result type of a `map`/`scan` over `src` whose body returns `body_ret`,
/// mirroring [`Builder::push_map`](crate::builder::Builder::push_map)'s type rule
/// (a sequence of `body_ret` with the source's length/kind, falling back to the
/// source type). The pass must compute this itself: the Builder derives it by
/// reading the body function's return type, but a *minted* body is not yet
/// installed in the registry, so it cannot be queried there and the node would
/// otherwise receive the wrong fallback type.
pub(crate) fn seq_result_type<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    src: ValueId,
    body_ret: TypeId,
) -> TypeId {
    let src_ty = host.type_of(src);
    match host.shared().types.seq_of(src_ty) {
        Some((_, len, is_list)) => host.shared().types.get_or_make_seq(body_ret, len, is_list),
        None => src_ty,
    }
}

/// Push a fresh param typed `ty` onto `block` in the minted host (host-routed
/// mirror of `BasicBlock::push_param` + the `type_id` write). Returns its value.
fn push_param_into<'str>(host: &mut PassBacking<'_, 'str>, block: BlockId, ty: TypeId) -> ValueId {
    let index = host.view().block(block).params.len();
    let pid = host.push_block_param(block.func, BlockParam::new(index, ty, block.local));
    host.block_mut(block).params.push(pid.localize(block.func));
    ValueId::BlockParam(pid)
}

/// Mint an instruction with an explicit result type into the minted host and
/// append it to `block` (host-routed mirror of `InstructionRef::from_mnemonic_with_type`
/// + `push_insn`). Returns its value.
fn push_insn_into<'str>(
    host: &mut PassBacking<'_, 'str>,
    block: BlockId,
    mnemonic: Mnemonic,
    ty: TypeId,
) -> ValueId {
    let id = host.push_mnemonic_with_type(block.func, mnemonic, ty);
    BaseRef::new(host.reborrow(), block).push_insn(id);
    ValueId::Instruction(id)
}

/// Simultaneously substitute `pairs` (`old → new`) in `mn`'s operands. Plain
/// sequential `replace_value` calls corrupt a **cross-arena** clone remap: bare
/// body-local operands carry no function half, so a freshly-written target-arena
/// local can numerically collide with a not-yet-processed source-arena key and
/// get double-replaced. Route through unique varnode sentinels instead — a
/// closed pure slice never carries a varnode operand ([`pure_slice`] rejects
/// them), so the sentinels are guaranteed fresh.
pub(crate) fn substitute_operands(mn: &mut Mnemonic, pairs: &[(LocalValueId, LocalValueId)]) {
    let sentinel = |k: usize| LocalValueId::Varnode(VarnodeId::from(0x7fff_0000 + k));
    for (k, &(old, _)) in pairs.iter().enumerate() {
        mn.replace_value(old, sentinel(k));
    }
    for (k, &(_, new)) in pairs.iter().enumerate() {
        mn.replace_value(sentinel(k), new);
    }
}

/// Build a fresh single-block pure function `name` returning `result` by
/// [minting](crate::pipeline::mint_function) it (kind `Machine`, `is_pure`),
/// cloning the pure `slice` (read from `body`'s own function, in definition
/// order) with operands remapped through the value map that `seed` installs.
/// `seed` receives a read view of the owning function and the minted mutation
/// host, creates the root block's parameters (and any unpacking instructions),
/// and returns the initial input→value map. The returned callee remains a
/// pass-local placeholder until the install barrier patches its owner sites.
#[allow(clippy::too_many_arguments)]
fn outline_core<'str>(
    m: ContextView<'_, 'str>,
    body: &mut FunctionBody<'str>,
    next_minted: &mut u32,
    minted_out: &mut Vec<Minted<'str>>,
    name: &str,
    result: ValueId,
    slice: &[InstructionId],
    seed: impl for<'a> FnOnce(
        BodyView<'a, 'str>,
        &mut qcode::value::util::pass_backing::PassBacking<'a, 'str>,
        BlockId,
    ) -> HashMap<ValueId, ValueId>,
) -> Option<Callee> {
    // Fully pure: a deterministic function of its params. `is_pure` (with the
    // implied `pure_reg`) is set by `mint_function`; the GUI purity badge keys
    // off `pure_reg`.
    let callee = crate::pipeline::mint_function(
        body,
        next_minted,
        minted_out,
        std::borrow::Cow::Owned(name.to_owned()),
        FunctionKind::Machine,
        /*pure*/ true,
    );
    let fid = body.id();

    // Read the slice mnemonics/types from the owning function up front, so the
    // minted-host borrow below does not overlap the owner read.
    let cloned: Vec<(InstructionId, Mnemonic, TypeId)> = {
        let own = m.body_view(body);
        slice
            .iter()
            .map(|&iid| {
                let r = InstructionRef::new(own, iid);
                (iid, r.mnemonic().clone(), r.type_id())
            })
            .collect()
    };
    let dummy_ptr = m.shr().get_const(0, 8);
    let ret_ty = m.shr().types.get_or_make_int(1);

    let (own, mut minted) = crate::pipeline::host_with_minted(body, minted_out, m, callee);
    // Root block, set as the minted function's entry, named for display (block
    // names are function-scoped, so uniqueness is within the new function).
    let root = minted.make_block(fid);
    minted.function_mut(fid).set_root_id(Some(root.local));
    let block_name = format!("{name}_entry");
    let _ = BaseRef::new(minted.reborrow(), root).rename_local(std::borrow::Cow::Owned(block_name));

    // Params / unpacking, installed by the caller; seeds the input→value map.
    let mut value_map = seed(own, &mut minted, root);

    // Clone the slice in definition order, remapping operands through the map.
    // The clone's operands are the *source* function's locals (qualify with
    // `iid.func` for the map lookup); the replacements live in the minted
    // function's arena (localize against `fid`). `pure_slice` guarantees every
    // non-literal operand is in the map, so no source-local id survives.
    for (iid, mut mn, ty) in cloned {
        let pairs: Vec<(LocalValueId, LocalValueId)> = mn
            .args()
            .iter()
            .filter_map(|&a| {
                value_map
                    .get(&a.qualify(iid.func))
                    .map(|&n| (a, n.localize(fid)))
            })
            .collect();
        substitute_operands(&mut mn, &pairs);
        let new = push_insn_into(&mut minted, root, mn, ty);
        value_map.insert(ValueId::Instruction(iid), new);
    }

    // Return the element value. `ptr` is the (irrelevant) return-address slot.
    let ret_val = value_map.get(&result).copied().unwrap_or(result);
    push_insn_into(
        &mut minted,
        root,
        Mnemonic::Return(Return {
            ptr: dummy_ptr.localize(fid),
            value: Some(ret_val.localize(fid)),
        }),
        ret_ty,
    );
    Some(callee)
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
    let root = FunctionBody::from_id(ctx, body_fn).root().map(|b| b.id)?;
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
            let v = r.value?.qualify(iid.func);
            return Some(value_map.get(&v).copied().unwrap_or(v));
        }
        let ty = ctx.get_insn(iid).type_id();
        let mut nm = m;
        // The clone's operands are `body_fn`'s locals (qualify with `iid.func`
        // for the lookup); the replacements live in the caller's arena.
        let pairs: Vec<(LocalValueId, LocalValueId)> = nm
            .args()
            .iter()
            .filter_map(|&a| {
                value_map
                    .get(&a.qualify(iid.func))
                    .map(|&n| (a, n.localize(block.func)))
            })
            .collect();
        substitute_operands(&mut nm, &pairs);
        let new_id = InstructionRef::from_mnemonic_with_type(ctx, block.func, nm, ty).id;
        BasicBlock::from_id_mut(ctx, block).insert_insn_before(at, new_id);
        value_map.insert(ValueId::Instruction(iid), ValueId::Instruction(new_id));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{Value, insn::Mnemonic},
    };

    use qcode::value::insn::IntrinsicId;

    /// Build a host function `idx + zext(elem)` and outline it into a body of
    /// `(i64 idx, i8 elem)`. The clone must have two params, recompute the
    /// expression, return it, and be pure.
    #[test]
    fn outlines_closed_pure_expression() {
        let mut tc = TestContext::new();
        let host = FunctionBody::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000, host);
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, host);
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

        let outlined =
            crate::test_util::with_minting(&mut tc.ctx, host, |m, body, next, minted| {
                outline_expression(m, body, next, minted, "body", result, &[idx, elem])
            });
        assert!(outlined.is_some(), "expression is closed over (idx, elem)");
        let body = FunctionBody::from_name(&tc.ctx, "body")
            .expect("the mint barrier installs the outlined body")
            .id;

        // Two params, in input order, with the input widths.
        let params: Vec<usize> = FunctionBody::from_id(&tc.ctx, body)
            .root()
            .unwrap()
            .params()
            .map(|p| p.size())
            .collect();
        assert_eq!(params, vec![8, 1]);

        // The body recomputes the expression (a zext and an add) and returns it.
        let root = FunctionBody::from_id(&tc.ctx, body).root().unwrap().id;
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
        assert!(FunctionBody::from_id(&tc.ctx, body).is_pure());
        // Full purity implies register purity — the GUI badge keys off the latter.
        assert!(FunctionBody::from_id(&tc.ctx, body).is_pure_reg());
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
        let host = FunctionBody::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000, host);
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let ram = tc.ctx.shared.default_space;
        let (idx, result) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let idx = b.push_param(8).id();
            // A load is an untracked source: the expression is not closed.
            let loaded = b.push_load::<false>(idx, 8, ram).id();
            let result = b.push_add(idx, loaded).id();
            (idx, result)
        };

        assert!(
            crate::test_util::with_minting(&mut tc.ctx, host, |m, body, next, minted| {
                outline_expression(m, body, next, minted, "body", result, &[idx])
            })
            .is_none(),
            "an expression reaching a load must not outline"
        );
    }

    /// `outline_tupled` produces a unary body taking the `(index, elem)` tuple and
    /// unpacking it with two extracts before recomputing `index + zext(elem)`.
    #[test]
    fn outlines_tupled_index_aware_body() {
        let mut tc = TestContext::new();
        let host = FunctionBody::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000, host);
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, host);
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
        let i8 = tc.ctx.shared.types.get_or_make_int(1);
        let arr_ty = tc.ctx.shared.types.get_or_make_array(i8, 1);
        let enum_id = IntrinsicId::from_name("enumerate").unwrap();
        let enum_ty = enum_id.desc().result_type(&tc.ctx.shared.types, &[arr_ty]);
        let (tuple_ty, _) = tc.ctx.shared.types.array_of(enum_ty).unwrap();

        let outlined =
            crate::test_util::with_minting(&mut tc.ctx, host, |m, body, next, minted| {
                outline_tupled(
                    m,
                    body,
                    next,
                    minted,
                    "body",
                    result,
                    idx,
                    Some(elem),
                    tuple_ty,
                )
            });
        assert!(outlined.is_some(), "expression is closed over (idx, elem)");
        let body = FunctionBody::from_name(&tc.ctx, "body")
            .expect("the mint barrier installs the outlined body")
            .id;

        // A single param — the tuple — sized to the `(i64, i8)` aggregate.
        let params: Vec<usize> = FunctionBody::from_id(&tc.ctx, body)
            .root()
            .unwrap()
            .params()
            .map(|p| p.size())
            .collect();
        assert_eq!(params, vec![tc.ctx.shared.types.size_of(tuple_ty)]);

        // The body unpacks the tuple (two extracts) and recomputes zext + add.
        let root = FunctionBody::from_id(&tc.ctx, body).root().unwrap().id;
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
        assert!(FunctionBody::from_id(&tc.ctx, body).is_pure());
    }
}
