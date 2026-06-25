//! Loop-to-map: recognize a total element-wise array loop and rewrite it as a
//! single [`Map`](qcode::value::insn::Map) over the array value, outlining the
//! per-element body into a fresh pure function. See `ARGPROMOTE_ARRAY_MAP.md`.
//!
//! This module currently provides [`outline_expression`], the mechanical core:
//! copying a pure expression DAG into a standalone function. The recognizer that
//! drives it (finding the loop, proving element-locality, calling
//! [`push_map`](qcode::builder::Builder::push_map)) builds on top.
//!
//! `outline_expression` and its helpers are exercised by tests today; the
//! recognizer (next) is their first production caller.
#![allow(dead_code)]

use std::borrow::Cow;

use rustc_hash::FxHashMap as HashMap;

use qcode::{
    context::Context,
    value::{
        BasicBlock, Function, FunctionId, Instruction, InstructionRef, ValueId,
        insn::{InstructionId, Mnemonic, Return},
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
fn pure_slice(
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

    let fid = Function::make(ctx, Cow::Owned(name.to_owned())).ok()?.id;
    let root = Function::from_id_mut(ctx, fid).make_root().id;

    // Parameters, in input order, typed as the host inputs.
    let mut value_map: HashMap<ValueId, ValueId> = HashMap::default();
    for &inp in inputs {
        let ty = ctx.type_of(inp);
        let size = ctx.types.size_of(ty);
        let pid = BasicBlock::from_id_mut(ctx, root).push_param(size).id;
        ctx.values.block_params[pid].type_id = ty;
        value_map.insert(inp, ValueId::BlockParam(pid));
    }

    // Clone the slice in definition order, remapping operands through the map.
    for &iid in &slice {
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

    Function::from_id_mut(ctx, fid).set_is_pure(true);
    Some(fid)
}

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
}
