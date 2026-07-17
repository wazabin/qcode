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
//! Partial constness is gated by the field's backward
//! [`projection`](crate::calls::project_return): only fields the projection
//! proves constant over the literal arguments are considered.
//!
//! Symbolic arguments are bound to **poison** for emulation, not `0` (argpromote
//! v2, P.2). Emulating the callee body demands the concrete value of every
//! operand it touches, so a call carrying any symbolic argument that the body
//! actually reads — including into the return tuple — bails on the poison-read
//! trap instead of silently computing a wrong result on a fabricated `0`. This
//! is the intended correctness shift: a fold now happens only when the emulation
//! never reads an unknown value.

use qcode::value::QCodeMut;
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

use super::{ModuleInsn, fold::const_value, module_insn, module_instruction_snapshot};

/// Upper bound on emulated instructions per harvested field. Pure functions are
/// loop-free (an argpromote invariant), so this only guards against a function
/// slipping that invariant past the verifier.
const STEP_BUDGET: usize = 100_000;

/// Replace `extract` of a constant pure-call field with the emulated literal.
/// One invocation performs one frozen module sweep; pipeline configuration owns
/// the whole-program fixpoint.
#[derive(Default)]
pub(super) struct PureCall;

impl PureCall {
    fn rewrite(&self, ctx: &mut Context, ic: &ModuleInsn) -> bool {
        let Mnemonic::Extract(Extract { agg, index }) = ic.mnemonic.clone() else {
            return false;
        };
        let ValueId::Instruction(call_id) = agg.qualify(ic.insn_id.func) else {
            return false;
        };
        // The aggregate must be a direct call to a fully pure function.
        let Mnemonic::Call(Call { target, args, .. }) = ctx.get_insn(call_id).mnemonic().clone()
        else {
            return false;
        };
        let Some(target) = target.real() else {
            return false;
        };
        if !FunctionBody::from_id(ctx, target).is_pure() {
            return false;
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
            return false;
        }

        // The field must depend only on literal arguments.
        let Some(proj) = project_return(ctx, target, index) else {
            return false;
        };
        if !proj.is_constant_over(&literal_indices) {
            return false;
        }

        let Some(root) = FunctionBody::from_id(&*ctx, target)
            .root()
            .map(|block| block.id)
        else {
            return false;
        };
        let param_sizes: Vec<usize> = qcode::value::BasicBlock::from_id(ctx, root)
            .params()
            .map(|p| p.size())
            .collect();
        if param_sizes.len() != args.len() {
            return false;
        }
        // Build the positional argument vector: a literal binds its concrete
        // value, a symbolic argument binds **poison** (`None`). The projection has
        // proven this field independent of the symbolic arguments, so a correct
        // fold never reads the poison; if it does, the emulator's poison-read trap
        // makes the fold bail rather than compute on a bogus value.
        let arg_values: Vec<Option<SizedValue>> = args
            .iter()
            .zip(&param_sizes)
            .map(|(&a, &size)| {
                const_value(&*ctx, a.qualify(ic.insn_id.func)).map(|v| SizedValue::new(v, size))
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
            return false;
        };

        let lit = ctx.get_const(value, ic.size).id();
        ctx.replace_instruction(ic.insn_id, lit);
        true
    }
}

impl crate::Pass for PureCall {
    const NAME: &'static str = "pure_call";

    fn description(&self) -> &'static str {
        "Emulate constant fields returned by pure calls"
    }

    fn run(
        &self,
        ctx: &mut Context,
        _env: &crate::PipelineEnv,
        targets: &[qcode::value::FunctionId],
    ) -> Result<crate::ModulePassOutcome, String> {
        let targets: rustc_hash::FxHashSet<_> = targets.iter().copied().collect();
        let snapshot = module_instruction_snapshot(ctx)
            .into_iter()
            .filter(|id| targets.contains(&id.func))
            .collect::<Vec<_>>();
        let mut changed = rustc_hash::FxHashSet::default();
        for insn_id in snapshot {
            let ic = module_insn(ctx, insn_id);
            if self.rewrite(ctx, &ic) {
                changed.insert(insn_id.func);
            }
        }
        Ok(crate::ModulePassOutcome::functions(changed)
            .preserving_global::<crate::AddressAnalysis>())
    }
}

crate::register_module_pass!(PureCall);

/// Run pure function `target` on `arg_values` and read field `index` of the
/// returned aggregate at the reached return. `None` on any emulation fault, step
/// budget exhaustion, or a non-scalar (aggregate) field.
fn emulate_field(
    ctx: &Context,
    target: qcode::value::function::FunctionId,
    root: qcode::value::block::BlockId,
    arg_values: &[Option<SizedValue>],
    index: usize,
) -> Option<u64> {
    let mut emu = StandaloneEmulator::new(root);
    emu.run_pure_partial(ctx, target, arg_values, STEP_BUDGET)
        .ok()?;

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
        testing::TestContext,
        value::{
            BasicBlock, FunctionBody, LocalValueId,
            function::{FunctionId, FunctionSignature},
            insn::{Return, Store},
        },
    };

    fn run_pure_call(tc: &mut TestContext) {
        let env = crate::PipelineEnv::headless(&tc.ctx);
        loop {
            let targets = tc.ctx.function_ids();
            if !crate::Pass::run(&PureCall, &mut tc.ctx, &env, &targets)
                .unwrap()
                .changed()
            {
                break;
            }
        }
    }

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
            let mut b = tc.ctx.builder(entry);
            let a = b.push_param(8).id();
            let bp = b.push_param(8).id();
            let c69 = b.shr().get_const(69, 8);
            let c42 = b.shr().get_const(42, 8);
            let b69 = b.push_mul(bp, c69).id();
            let body = b.push_add(b69, c42).id();
            tuple = b.push_tuple(vec![a, body]).id();
            ptr = b.shr().get_const(0x2000, 8);
            ret = b.push_return(ptr).id();
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
            is_pure: true,
            ..Default::default()
        });
        // Materialize the register channel (what `is_pure_reg` now reads).
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_pure_reg(true);
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
            let mut b = tc.ctx.builder(entry);
            a_in = b.push_param(8).id();
            b_arg = match b_const {
                Some(v) => b.shr().get_const(v, 8),
                None => b.push_param(8).id(),
            };
            let ValueId::Instruction(id) = b.push_call(foo).id() else {
                unreachable!()
            };
            call_id = id;
        }
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: qcode::value::insn::Callee::Real(foo),
                args: vec![a_in.localize(gid), b_arg.localize(gid)],
                clobbers: vec![],
                tag: Default::default(),
            }),
        );
        qcode::value::Instruction::from_id_mut(&mut tc.ctx, call_id).set_type(agg_ty);
        tc.ctx.add_cfg_edge(entry, cont);
        let (r0, r1, reg_space) = (tc.r0, tc.r1, tc.reg_space);
        {
            let mut b = tc.ctx.builder(cont);
            let cr = ValueId::Instruction(call_id);
            let e0 = b.push_extract(cr, 0).id();
            let e1 = b.push_extract(cr, 1).id();
            b.push_store(e0, ValueId::Varnode(r0), reg_space);
            b.push_store(e1, ValueId::Varnode(r1), reg_space);
            let ptr = b.shr().get_const(0, 8);
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

    /// `foo(a_in, 7)` with a symbolic `a_in`: emulating the body builds the
    /// return tuple `(a_in, b*69+42)`, whose first field reads the **poison**
    /// bound for `a_in` — a hard error (argpromote v2, P.2). The pure-call fold
    /// therefore bails: neither field is harvested and no `0`-derived (or here,
    /// `525`) constant is stored. This is the intended correctness shift — a call
    /// with an unknown argument no longer emulates on a fabricated concrete value.
    #[test]
    fn symbolic_arg_bails_instead_of_computing_on_poison() {
        let mut tc = TestContext::new();
        let foo = build_pure_foo(&mut tc);
        let (_g, cont) = build_caller(&mut tc, foo, Some(7));
        let r1 = ValueId::Varnode(tc.r1);

        run_pure_call(&mut tc);

        assert_eq!(
            stored_const(&tc, cont, r1),
            None,
            "the fold must bail (poison read) rather than store a constant"
        );
        assert_eq!(
            extract_count(&tc, cont),
            2,
            "both extracts survive — nothing is harvested from a poison-arg call"
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
            let mut b = tc.ctx.builder(entry);
            let sp = ValueId::BlockParam(sp_pid);
            let arr = ValueId::BlockParam(arr_pid);
            // `at(arr, 0)` bridges the array param to a scalar lane (byte 0),
            // routing the read through `resolve_array`'s new `BlockParam` arm.
            let zero = b.shr().get_const(0, 8);
            let b0 = b.push_intrinsic(at_id, vec![arr, zero]).id();
            let z = b.push_zext(b0, 4).id();
            tuple = b.push_tuple(vec![sp, z]).id();
            ptr = b.shr().get_const(0x2000, 8);
            ret = b.push_return(ptr).id();
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
            is_pure: true,
            ..Default::default()
        });
        // Materialize the register channel (what `is_pure_reg` now reads).
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_pure_reg(true);
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
            let mut b = tc.ctx.builder(entry);
            sp_arg = b.shr().get_const(0x40ea20, 4);
            arr_arg = b.shr().get_const(0x2f76bfc2, 4);
            let ValueId::Instruction(id) = b.push_call(dec).id() else {
                unreachable!()
            };
            call_id = id;
        }
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: qcode::value::insn::Callee::Real(dec),
                args: vec![sp_arg.localize(gid), arr_arg.localize(gid)],
                clobbers: vec![],
                tag: Default::default(),
            }),
        );
        qcode::value::Instruction::from_id_mut(&mut tc.ctx, call_id).set_type(agg_ty);
        tc.ctx.add_cfg_edge(entry, cont);
        let (r1, reg_space) = (tc.r1, tc.reg_space);
        {
            let mut b = tc.ctx.builder(cont);
            let cr = ValueId::Instruction(call_id);
            let e1 = b.push_extract(cr, 1).id();
            b.push_store(e1, ValueId::Varnode(r1), reg_space);
            let ptr = b.shr().get_const(0, 8);
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
        let (_g, cont) = build_decoder_caller(&mut tc, dec);
        let r1 = ValueId::Varnode(tc.r1);

        run_pure_call(&mut tc);

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
        let (_g, cont) = build_caller(&mut tc, foo, None);

        run_pure_call(&mut tc);

        assert_eq!(
            extract_count(&tc, cont),
            2,
            "both extracts must survive when no argument is constant"
        );
    }
}
