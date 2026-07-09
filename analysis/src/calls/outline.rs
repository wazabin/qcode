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

use std::borrow::Cow;

use rustc_hash::FxHashMap as HashMap;

use qcode::{
    context::Context,
    types::TypeId,
    value::{
        BasicBlock, BlockId, Function, FunctionId, Instruction, InstructionRef, Renameable,
        ValueId,
        insn::{Binary, Binop, Extract, InstructionId, IntBinop, Mnemonic, Range, Return},
    },
};

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
pub(crate) fn pure_slice(
    ctx: &Context,
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
            ctx.values.block_param_mut(pid).type_id = ty;
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
/// `elem_input` is `None` for a pure index-driven generation whose body reads the
/// index but not the lane element (the `t.1` extract is then omitted); the map
/// still ranges over `enumerate(arr)` so the body receives the index in `t.0`.
///
/// Returns `None` if the expression is not closed over `(index, elem?)` + literals
/// (see [`pure_slice`]).
pub(crate) fn outline_tupled(
    ctx: &mut Context,
    name: &str,
    result: ValueId,
    index_input: ValueId,
    elem_input: Option<ValueId>,
    tuple_ty: TypeId,
) -> Option<FunctionId> {
    let mut inputs = vec![index_input];
    inputs.extend(elem_input);
    let slice = pure_slice(ctx, result, &inputs)?;
    outline_core(ctx, name, result, &slice, move |ctx, root| {
        let mut value_map: HashMap<ValueId, ValueId> = HashMap::default();
        let tsz = ctx.types.size_of(tuple_ty);
        let pid = BasicBlock::from_id_mut(ctx, root).push_param(tsz).id;
        ctx.values.block_param_mut(pid).type_id = tuple_ty;
        let tuple = ValueId::BlockParam(pid);
        // index = t.0, elem = t.1 — the extracts the body unpacks. `elem` is only
        // extracted when the body actually consumes the lane element.
        let fields = [(0usize, Some(index_input)), (1usize, elem_input)];
        for (field, input) in fields {
            let Some(input) = input else { continue };
            let fty = ctx
                .types
                .field_type(tuple_ty, field)
                .expect("enumerate tuple field");
            let ex = InstructionRef::from_mnemonic_with_type(
                ctx,
                root.func,
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
    elem: ScanElem,
) -> Option<FunctionId> {
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
    let slice = pure_slice(ctx, result, &inputs)?;
    outline_core(ctx, name, result, &slice, move |ctx, root| {
        let mut value_map: HashMap<ValueId, ValueId> = HashMap::default();
        // Param 0: the accumulator.
        let acc_sz = ctx.types.size_of(acc_ty);
        let apid = BasicBlock::from_id_mut(ctx, root).push_param(acc_sz).id;
        ctx.values.block_param_mut(apid).type_id = acc_ty;
        value_map.insert(acc_input, ValueId::BlockParam(apid));
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
        let psz = ctx.types.size_of(param_ty);
        let pid = BasicBlock::from_id_mut(ctx, root).push_param(psz).id;
        ctx.values.block_param_mut(pid).type_id = param_ty;
        let param = ValueId::BlockParam(pid);
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
        // Narrow `i64` index → loop index width.
        let isz = ctx.types.size_of(index_ty);
        if isz < ctx.types.size_of(fty) {
            let r = InstructionRef::from_mnemonic_with_type(
                ctx,
                root.func,
                Mnemonic::Range(Range {
                    src: idx,
                    start: 0,
                    size: isz,
                }),
                index_ty,
            )
            .id;
            BasicBlock::from_id_mut(ctx, root).push_insn(r);
            idx = ValueId::Instruction(r);
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
                root.func,
                Mnemonic::Binop(Binary {
                    lhs: idx,
                    rhs: c,
                    op: Binop::Int(IntBinop::Add),
                }),
                index_ty,
            )
            .id;
            BasicBlock::from_id_mut(ctx, root).push_insn(add);
            idx = ValueId::Instruction(add);
        }
        value_map.insert(index_input, idx);
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
    // (otherwise it falls back to an opaque `<bb_N>`). Block names are
    // function-scoped, so deduplicate within the new function's own table.
    let block_name = ctx.get_unique_name_in(fid, Cow::Owned(format!("{name}_entry")));
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
        let new_id = InstructionRef::from_mnemonic_with_type(ctx, root.func, m, ty).id;
        BasicBlock::from_id_mut(ctx, root).push_insn(new_id);
        value_map.insert(ValueId::Instruction(iid), ValueId::Instruction(new_id));
    }

    // Return the element value. `ptr` is the (irrelevant) return-address slot.
    let ret_val = value_map.get(&result).copied().unwrap_or(result);
    let dummy_ptr = ctx.get_const(0, 8).id();
    let ret_ty = ctx.types.get_or_make_int(1);
    let ret = InstructionRef::from_mnemonic_with_type(
        ctx,
        root.func,
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
        let host = Function::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x1000, __f)
        };
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
        let entry = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x1000, __f)
        };
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
        let entry = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x1000, __f)
        };
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
        let enum_ty = enum_id.desc().result_type(&tc.ctx.types, &[arr_ty]);
        let (tuple_ty, _) = tc.ctx.types.array_of(enum_ty).unwrap();

        let body = outline_tupled(&mut tc.ctx, "body", result, idx, Some(elem), tuple_ty)
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
