//! Pure-function emulation sub-pass: harvest constant return values from calls
//! to pure functions.
//!
//! When an `extract(call_result, i)` projects field `i` of a [`Call`] to a
//! **pure** function (one argpromote has fully functionalized — see
//! [`FunctionRef::is_pure`](qcode::value::FunctionRef::is_pure)), and that field's
//! value depends only on call arguments
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
        FunctionBody, ValueId,
        insn::{Call, Extract, Mnemonic, Tuple},
    },
};

use qcode_emulator::{SizedValue, StandaloneEmulator};

use crate::calls::{project_return, return_field};

use super::fold::const_value;
use std::any::Any;

use super::walk::{Claim, Editor, InsnCtx, ModuleSubPass};

/// Upper bound on emulated instructions per harvested field. Pure functions are
/// loop-free (an argpromote invariant), so this only guards against a function
/// slipping that invariant past the verifier.
const STEP_BUDGET: usize = 100_000;

/// Replace `extract` of a constant pure-call field with the emulated literal.
/// Reads the pure *callee*'s body directly, so it runs only on the module host
/// (dispatched by the [`concretize`](super::concretize) module pass).
pub(super) struct PureCall;

impl<'str> ModuleSubPass<'str> for PureCall {
    fn init_state(&self) -> Box<dyn Any> {
        Box::new(())
    }

    fn clone_state(&self, _state: &dyn Any) -> Box<dyn Any> {
        Box::new(())
    }

    fn on_insn(
        &self,
        host: &mut Context<'str>,
        _state: &mut dyn Any,
        ic: &InsnCtx,
        ed: &mut Editor,
    ) -> Claim {
        let ctx: &mut Context = host;
        let Mnemonic::Extract(Extract { agg, index }) = *ic.mnemonic else {
            return Claim::Pass;
        };
        let ValueId::Instruction(call_id) = agg.qualify(ic.insn_id.func) else {
            return Claim::Pass;
        };
        // The aggregate must be a direct call to a fully pure function.
        let Mnemonic::Call(Call { target, args, .. }) = ctx.get_insn(call_id).mnemonic().clone()
        else {
            return Claim::Pass;
        };
        let Some(target) = target.real() else {
            return Claim::Pass;
        };
        if !FunctionBody::from_id(ctx, target).is_pure() {
            return Claim::Pass;
        }

        // Pre-filter: at least one literal argument (otherwise nothing folds and
        // the projection work is wasted).
        let literal_indices: HashSet<usize> = args
            .iter()
            .enumerate()
            .filter(|&(_, &a)| const_value(&*ctx, a.qualify(ic.insn_id.func)).is_some())
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
        let Some(root) = FunctionBody::from_id(&*ctx, target)
            .root()
            .map(|block| block.id)
        else {
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
            .map(|(&a, &size)| {
                SizedValue::new(
                    const_value(&*ctx, a.qualify(ic.insn_id.func)).unwrap_or(0),
                    size,
                )
            })
            .collect();

        // Emulate the callee on the concrete arguments and read the field back.
        let Some(value) = emulate_field(ctx, target, root, &arg_values, index) else {
            // All static gates passed but emulation still yielded nothing — this
            // fall-through was silent before, which is exactly why the
            // block-param `resolve_array` gap stayed invisible.
            qcode::pass_log!(
                debug,
                "pure-call emulation failed after static gates: target {target:?}, field {index}"
            );
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
            BasicBlock, FunctionBody, LocalValueId,
            function::{FunctionId, FunctionSignature},
            insn::{Return, Store},
        },
    };

    /// Build pure `foo(a, b) = (a, b*69 + 42)` and mark it `is_pure`.
    fn build_pure_foo(tc: &mut TestContext) -> FunctionId {
        let fid = FunctionBody::make(&mut tc.ctx, "foo".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000, fid);
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
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
                ptr: ptr.localize(fid),
                value: Some(tuple.localize(fid)),
            }),
        );
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_signature(FunctionSignature {
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
            let i64_ty = tc.ctx.shared.types.get_or_make_int(8);
            tc.ctx
                .shared
                .types
                .get_or_make_aggregate(vec![i64_ty, i64_ty])
        };
        let gid = FunctionBody::make(&mut tc.ctx, "g".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x4000, gid);
        let cont = tc.ctx.get_or_make_block(0x4100, gid);
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, gid);
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
                target: qcode::value::insn::Callee::Real(foo),
                args: vec![a_in.localize(gid), b_arg.localize(gid)],
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
            if ptr.qualify(block.func) != reg {
                return None;
            }
            match src {
                LocalValueId::Literal(lid) => Some(tc.ctx.shared.values.literals[*lid].value),
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

        super::super::concretize::concretize_function(&mut tc.ctx, g);

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

    /// Build pure `dec(sp: i32, arr: [i8;4])` returning `(sp, zext(arr[0:1]))`
    /// and mark it `is_pure`. This mirrors the `fn_410770` byte-wise decoder
    /// shape: the array-typed root param is read via a `Range` slice, exercising
    /// the `BlockParam` arm of `resolve_array`.
    fn build_pure_decoder(tc: &mut TestContext) -> FunctionId {
        let arr_ty = {
            let i8_ty = tc.ctx.shared.types.get_or_make_int(1);
            tc.ctx.shared.types.get_or_make_array(i8_ty, 4)
        };
        let fid = FunctionBody::make(&mut tc.ctx, "dec".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000, fid);
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        // Params in signature order (sp, arr) so the caller's positional args
        // align. Array-typed param can't be minted via `push_param` (scalar
        // only), so set its stored type directly, matching how argpromote does.
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(4).id;
        let arr_pid = BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(4).id;
        tc.ctx.block_param_mut(arr_pid).type_id = arr_ty;

        let at_id = qcode::value::insn::IntrinsicId::from_name("at").expect("at registered");
        let (ret, ptr, tuple);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let sp = ValueId::BlockParam(sp_pid);
            let arr = ValueId::BlockParam(arr_pid);
            // `at(arr, 0)` bridges the array param to a scalar lane (byte 0),
            // routing the read through `resolve_array`'s new `BlockParam` arm.
            let zero = b.context_mut().get_const(0, 8).id();
            let b0 = b.push_intrinsic(at_id, vec![arr, zero]).id();
            let z = b.push_zext(b0, 4).id();
            tuple = b.push_tuple(vec![sp, z]).id();
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
                ptr: ptr.localize(fid),
                value: Some(tuple.localize(fid)),
            }),
        );
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_signature(FunctionSignature {
            pure_reg: true,
            is_pure: true,
            ..Default::default()
        });
        fid
    }

    /// Caller `g` = `dec(0x40ea20, 0x2f76bfc2)`, extracting field 1 into `r1`.
    /// Both args are literals (`sp` scalar, `arr` a 4-byte array-shaped literal).
    fn build_decoder_caller(
        tc: &mut TestContext,
        dec: FunctionId,
    ) -> (FunctionId, qcode::value::block::BlockId) {
        let agg_ty = {
            let i32_ty = tc.ctx.shared.types.get_or_make_int(4);
            tc.ctx
                .shared
                .types
                .get_or_make_aggregate(vec![i32_ty, i32_ty])
        };
        let gid = FunctionBody::make(&mut tc.ctx, "g".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x4000, gid);
        let cont = tc.ctx.get_or_make_block(0x4100, gid);
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, gid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(cont);
        }
        let (sp_arg, arr_arg, call_id);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            sp_arg = b.context_mut().get_const(0x40ea20, 4).id();
            arr_arg = b.context_mut().get_const(0x2f76bfc2, 4).id();
            let ValueId::Instruction(id) = b.push_call(dec).id() else {
                unreachable!()
            };
            call_id = id;
            unsafe { b.dont_finalize() };
        }
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: qcode::value::insn::Callee::Real(dec),
                args: vec![sp_arg.localize(gid), arr_arg.localize(gid)],
                clobbers: vec![],
            }),
        );
        qcode::value::Instruction::from_id_mut(&mut tc.ctx, call_id).set_type(agg_ty);
        tc.ctx.add_cfg_edge(entry, cont);
        let (r1, reg_space) = (tc.r1, tc.reg_space);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, cont));
            let cr = ValueId::Instruction(call_id);
            let e1 = b.push_extract(cr, 1).id();
            b.push_store(e1, ValueId::Varnode(r1), reg_space);
            let ptr = b.context_mut().get_const(0, 8).id();
            b.push_return(ptr);
        }
        (gid, cont)
    }

    /// The `fn_410770` regression: an all-literal call to a pure decoder whose
    /// `arr` param is array-typed folds field 1 = `zext(arr[0])` to the low byte
    /// of the little-endian arg (`0x2f76bfc2 & 0xff = 0xc2`). This is exactly the
    /// path that stayed silent while `resolve_array` lacked a `BlockParam` arm.
    #[test]
    fn folds_array_param_field_from_all_literal_call() {
        let mut tc = TestContext::new();
        let dec = build_pure_decoder(&mut tc);
        let (g, cont) = build_decoder_caller(&mut tc, dec);
        let r1 = ValueId::Varnode(tc.r1);

        super::super::concretize::concretize_function(&mut tc.ctx, g);

        assert_eq!(
            stored_const(&tc, cont, r1),
            Some(0xc2),
            "field 1 = zext(arr[0]) must fold to the LE low byte of 0x2f76bfc2"
        );
        assert_eq!(
            extract_count(&tc, cont),
            0,
            "the sole extract must fold away"
        );
    }

    /// With no literal argument the pre-filter rejects the call; nothing is
    /// harvested even though field 1 is otherwise pure.
    #[test]
    fn no_harvest_without_literal_arg() {
        let mut tc = TestContext::new();
        let foo = build_pure_foo(&mut tc);
        let (g, cont) = build_caller(&mut tc, foo, None);

        super::super::concretize::concretize_function(&mut tc.ctx, g);

        assert_eq!(
            extract_count(&tc, cont),
            2,
            "both extracts must survive when no argument is constant"
        );
    }
}
