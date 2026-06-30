//! Pure-function emulation sub-pass: harvest constant return values from calls
//! to pure functions.
//!
//! When an `extract(call_result, i)` projects field `i` of a [`Call`] to a
//! **pure** function (one argpromote has fully functionalized — see
//! [`Function::is_pure`]), and that field's value depends only on call arguments
//! that are constant literals, the field is computed by emulating the callee and
//! the `extract` is replaced with the resulting literal. See
//! `PURE_EMULATION_DESIGN.md`.
//!
//! Partial constness is supported: only the field's backward
//! [`projection`](crate::calls::project_return) must be constant, so a call with
//! a mix of literal and symbolic arguments can still have some fields harvested.
//! Symbolic arguments are bound to a poison value (`0`) for emulation; the
//! projection guarantees the harvested field is independent of that choice.

use rustc_hash::FxHashSet as HashSet;

use qcode::{
    context::Context,
    value::{
        Function, ValueId,
        insn::{Call, Extract, Mnemonic, Tuple},
    },
};

use qcode_emulator::{SizedValue, StandaloneEmulator};

use crate::calls::{project_return, return_field};

use super::fold::const_value;
use std::any::Any;

use super::walk::{Claim, Editor, InsnCtx, SubPass};

/// Upper bound on emulated instructions per harvested field. Pure functions are
/// loop-free (an argpromote invariant), so this only guards against a function
/// slipping that invariant past the verifier.
const STEP_BUDGET: usize = 100_000;

/// Replace `extract` of a constant pure-call field with the emulated literal.
pub(super) struct PureCall;

impl SubPass for PureCall {
    fn init_state(&self) -> Box<dyn Any> {
        Box::new(())
    }

    fn clone_state(&self, _state: &dyn Any) -> Box<dyn Any> {
        Box::new(())
    }

    fn on_insn(&self, ctx: &mut Context, _state: &mut dyn Any, ic: &InsnCtx, ed: &mut Editor) -> Claim {
        let Mnemonic::Extract(Extract { agg, index }) = *ic.mnemonic else {
            return Claim::Pass;
        };
        let ValueId::Instruction(call_id) = agg else {
            return Claim::Pass;
        };
        // The aggregate must be a direct call to a fully pure function.
        let Mnemonic::Call(Call { target, args, .. }) = ctx.get_insn(call_id).mnemonic().clone()
        else {
            return Claim::Pass;
        };
        if !Function::from_id(ctx, target).is_pure() {
            return Claim::Pass;
        }

        // Pre-filter: at least one literal argument (otherwise nothing folds and
        // the projection work is wasted).
        let literal_indices: HashSet<usize> = args
            .iter()
            .enumerate()
            .filter(|&(_, &a)| const_value(ctx, a).is_some())
            .map(|(i, _)| i)
            .collect();
        if literal_indices.is_empty() {
            return Claim::Pass;
        }

        // The field must depend only on literal arguments.
        let Some(proj) = project_return(ctx, target, index) else {
            return Claim::Pass;
        };
        if !proj.is_constant_over(&literal_indices) {
            return Claim::Pass;
        }

        // Build the positional argument vector: literal value, or poison (0) for a
        // symbolic argument the projection has proven irrelevant to this field.
        let Some(root) = ctx.values.functions[target].root else {
            return Claim::Pass;
        };
        let param_sizes: Vec<usize> = qcode::value::BasicBlock::from_id(ctx, root)
            .params()
            .map(|p| p.size())
            .collect();
        if param_sizes.len() != args.len() {
            return Claim::Pass;
        }
        let arg_values: Vec<SizedValue> = args
            .iter()
            .zip(&param_sizes)
            .map(|(&a, &size)| SizedValue::new(const_value(ctx, a).unwrap_or(0), size))
            .collect();

        // Emulate the callee on the concrete arguments and read the field back.
        let Some(value) = emulate_field(ctx, target, root, &arg_values, index) else {
            return Claim::Pass;
        };

        let lit = ctx.get_const(value, ic.size).id();
        ed.replace(ctx, ic.insn_id, lit);
        Claim::Done
    }
}

/// Run pure function `target` on `arg_values` and read field `index` of the
/// returned aggregate at the reached return. `None` on any emulation fault, step
/// budget exhaustion, or a non-scalar (aggregate) field.
fn emulate_field(
    ctx: &Context,
    target: qcode::value::function::FunctionId,
    root: qcode::value::block::BlockId,
    arg_values: &[SizedValue],
    index: usize,
) -> Option<u64> {
    let mut emu = StandaloneEmulator::new(root);
    emu.run_pure(ctx, target, arg_values, STEP_BUDGET).ok()?;

    let field = return_field(ctx, emu.current_block(), index)?;
    // Scalar-only (v1): a field that is itself an aggregate is not harvested.
    if let ValueId::Instruction(id) = field
        && matches!(ctx.get_insn(id).mnemonic(), Mnemonic::Tuple(Tuple { .. }))
    {
        return None;
    }
    emu.get_value(ctx, field)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{
            BasicBlock, Function,
            function::{FunctionId, FunctionSignature},
            insn::{Return, Store},
        },
    };

    /// Build pure `foo(a, b) = (a, b*69 + 42)` and mark it `is_pure`.
    fn build_pure_foo(tc: &mut TestContext) -> FunctionId {
        let fid = Function::make(&mut tc.ctx, "foo".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let (ret, ptr, tuple);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let a = b.push_param(8).id();
            let bp = b.push_param(8).id();
            let c69 = b.context_mut().get_const(69, 8).id();
            let c42 = b.context_mut().get_const(42, 8).id();
            let b69 = b.push_mul(bp, c69).id();
            let body = b.push_add(b69, c42).id();
            tuple = b.push_tuple(vec![a, body]).id();
            ptr = b.context_mut().get_const(0x2000, 8).id();
            ret = b.push_return(ptr).id();
            unsafe { b.dont_finalize() };
        }
        let ValueId::Instruction(iid) = ret else {
            unreachable!()
        };
        tc.ctx.replace_instruction_mnemonic(
            iid,
            Mnemonic::Return(Return {
                ptr,
                value: Some(tuple),
            }),
        );
        Function::from_id_mut(&mut tc.ctx, fid).set_signature(FunctionSignature {
            pure_reg: true,
            is_pure: true,
            ..Default::default()
        });
        fid
    }

    /// Build caller `g` that calls `foo(a_in, b_const)` and stores both extracted
    /// fields. Returns `(g, g_cont)`. `b_const` is `None` to pass a symbolic value
    /// for the second argument too.
    fn build_caller(
        tc: &mut TestContext,
        foo: FunctionId,
        b_const: Option<u64>,
    ) -> (FunctionId, qcode::value::block::BlockId) {
        let agg_ty = {
            let i64_ty = tc.ctx.types.get_or_make_int(8);
            tc.ctx.types.get_or_make_aggregate(vec![i64_ty, i64_ty])
        };
        let gid = Function::make(&mut tc.ctx, "g".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x4000);
        let cont = tc.ctx.get_or_make_block(0x4100);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, gid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(cont);
        }
        let (a_in, b_arg, call_id);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            a_in = b.push_param(8).id();
            b_arg = match b_const {
                Some(v) => b.context_mut().get_const(v, 8).id(),
                None => b.push_param(8).id(),
            };
            let ValueId::Instruction(id) = b.push_call(foo).id() else {
                unreachable!()
            };
            call_id = id;
            unsafe { b.dont_finalize() };
        }
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: foo,
                args: vec![a_in, b_arg],
                clobbers: vec![],
            }),
        );
        qcode::value::Instruction::from_id_mut(&mut tc.ctx, call_id).set_type(agg_ty);
        tc.ctx.add_cfg_edge(entry, cont);
        let (r0, r1, reg_space) = (tc.r0, tc.r1, tc.reg_space);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, cont));
            let cr = ValueId::Instruction(call_id);
            let e0 = b.push_extract(cr, 0).id();
            let e1 = b.push_extract(cr, 1).id();
            b.push_store(e0, ValueId::Varnode(r0), reg_space);
            b.push_store(e1, ValueId::Varnode(r1), reg_space);
            let ptr = b.context_mut().get_const(0, 8).id();
            b.push_return(ptr);
        }
        (gid, cont)
    }

    fn extract_count(tc: &TestContext, block: qcode::value::block::BlockId) -> usize {
        BasicBlock::from_id(&tc.ctx, block)
            .iter()
            .filter(|i| matches!(i.mnemonic(), Mnemonic::Extract(_)))
            .count()
    }

    /// The literal stored into varnode `reg` in `block`, if its source is a const.
    fn stored_const(
        tc: &TestContext,
        block: qcode::value::block::BlockId,
        reg: ValueId,
    ) -> Option<u64> {
        BasicBlock::from_id(&tc.ctx, block).iter().find_map(|i| {
            let Mnemonic::Store(Store { ptr, src, .. }) = i.mnemonic() else {
                return None;
            };
            if *ptr != reg {
                return None;
            }
            match src {
                ValueId::Literal(lid) => Some(tc.ctx.values.literals[*lid].value),
                _ => None,
            }
        })
    }

    /// `foo(a_in, 7)`: field 1 (`b*69+42 = 525`) is harvested into a literal; field
    /// 0 (the symbolic `a_in`) is left as a live extract.
    #[test]
    fn harvests_constant_field_from_partial_const_call() {
        let mut tc = TestContext::new();
        let foo = build_pure_foo(&mut tc);
        let (g, cont) = build_caller(&mut tc, foo, Some(7));
        let r1 = ValueId::Varnode(tc.r1);

        let aliases = crate::AliasResult::simple(&tc.ctx);
        super::super::gvn_function(&mut tc.ctx, g, Some(&aliases));

        assert_eq!(
            stored_const(&tc, cont, r1),
            Some(7 * 69 + 42),
            "field 1 must be emulated to 525 and stored as a literal"
        );
        assert_eq!(
            extract_count(&tc, cont),
            1,
            "only the symbolic field-0 extract should remain"
        );
    }

    /// With no literal argument the pre-filter rejects the call; nothing is
    /// harvested even though field 1 is otherwise pure.
    #[test]
    fn no_harvest_without_literal_arg() {
        let mut tc = TestContext::new();
        let foo = build_pure_foo(&mut tc);
        let (g, cont) = build_caller(&mut tc, foo, None);

        let aliases = crate::AliasResult::simple(&tc.ctx);
        super::super::gvn_function(&mut tc.ctx, g, Some(&aliases));

        assert_eq!(
            extract_count(&tc, cont),
            2,
            "both extracts must survive when no argument is constant"
        );
    }
}
