//! Verify every function-private-space call argument targets a prototyped
//! external's pointer parameter.
//!
//! `argpromote`'s externals-into-shadow materialization rebases an admitted
//! prototyped-external's own-frame pointer argument into the promoting function's
//! private (shadow/temp) space (see `ram::apply`, rule 4). That is the *only*
//! legitimate way a private-space pointer becomes a call argument: the callee must
//! be a bodyless external whose `ExternArgmem` marks the corresponding parameter a
//! pointer kind (`OutPtr`/`MutPtr`/`ConstPtr`), so the RAM effect channel bounds
//! its footprint to the shadow object. A private-space pointer flowing into any
//! other call — an internal callee, an un-prototyped external, or a non-pointer
//! parameter slot — is a shadow leak: the callee would dereference a body-local
//! space it can neither name nor bound. Cheap and dirty-scoped: a private-space
//! provenance is function-local, so the per-function scan cannot miss one.

use qcode::{
    context::Context,
    value::{FunctionBody, ValueRef, insn::Mnemonic},
};

pub fn verify_private_space_call_args(ctx: &Context) -> Vec<String> {
    verify_private_space_call_args_scoped(ctx, super::Scope::All)
}

pub(crate) fn verify_private_space_call_args_scoped(
    ctx: &Context,
    scope: super::Scope<'_>,
) -> Vec<String> {
    let mut out = Vec::new();
    for insn in scope.instructions(ctx) {
        let Mnemonic::Call(c) = insn.mnemonic() else {
            continue;
        };
        let func = insn.id.func;
        for (i, arg) in c.args.iter().enumerate() {
            let arg = arg.qualify(func);
            // Only function-private (temp/shadow) provenance is constrained; a
            // shared space or an untyped value is unremarkable.
            let is_private = ValueRef::new(arg, ctx)
                .memory_space()
                .is_some_and(|ms| ms.shared().is_none());
            if !is_private {
                continue;
            }
            let caller_name = FunctionBody::from_id(ctx, func).name().to_string();
            let Some(callee) = c.target.real() else {
                out.push(format!(
                    "{caller_name}: passes a function-private-space pointer to an unresolved \
                     call target (arg {i}) — only a prototyped external's pointer parameter \
                     may receive a shadow pointer"
                ));
                continue;
            };
            let cf = FunctionBody::from_id(ctx, callee);
            let ok = cf.is_external()
                && cf.argmem().is_some_and(|am| {
                    matches!(
                        am.params.get(i),
                        Some(
                            qcode::value::ArgMemKind::OutPtr
                                | qcode::value::ArgMemKind::MutPtr
                                | qcode::value::ArgMemKind::ConstPtr
                        )
                    )
                });
            if !ok {
                out.push(format!(
                    "{caller_name}: passes a function-private-space pointer (arg {i}) to {} — \
                     only a prototyped external's pointer parameter may receive a shadow pointer",
                    cf.name(),
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::space::{LocalMemorySpaceId, MemorySpaceId};
    use qcode::value::insn::{Call, CallTag, Callee};
    use qcode::value::{ArgMemKind, BasicBlock, ExternArgmem, FunctionBody, QCodeMut, TempSpace};
    use wazabin_qcode_macro::qcode;

    /// Build a caller with a private (temp) shadow-typed pointer value, and give
    /// its single call `args = [shadow_ptr]` targeting `target`.
    fn setup(
        tc: &mut qcode::testing::TestContext,
        caller: qcode::value::FunctionId,
        entry: qcode::value::BlockId,
        call: qcode::value::InstructionId,
        target: qcode::value::FunctionId,
    ) {
        let shadow = tc.ctx.bodies[caller]
            .push_temp_space(TempSpace::new(Some("shadow"), 1, 8))
            .local;
        let shadow_mem = MemorySpaceId::Temp(qcode::value::TempSpaceId::new(caller, shadow));
        // A shadow-typed address built from the entry's SP-ish base.
        let ptr = {
            let base = BasicBlock::from_id(&tc.ctx, entry)
                .params()
                .next()
                .unwrap()
                .id();
            let mut b = tc.ctx.builder(entry);
            b.set_insert_point_to_start();
            let z = b.shr().get_const(0, 8);
            b.push_load::<false>(base, 8, LocalMemorySpaceId::Temp(shadow));
            b.push_add(base, z).id()
        };
        let sty = tc.ctx.shared.types.get_or_make_space_address(8, shadow_mem);
        if let ValueId::Instruction(iid) = ptr {
            qcode::value::Instruction::from_id_mut(&mut tc.ctx, iid).set_type(sty);
        }
        tc.ctx.replace_instruction_mnemonic(
            call,
            Mnemonic::Call(Call {
                target: Callee::Real(target),
                args: vec![ptr.localize(call.func)],
                clobbers: vec![],
                tag: CallTag::RegPure,
            }),
        );
    }

    use qcode::value::ValueId;

    fn caller_call(
        tc: &qcode::testing::TestContext,
        entry: qcode::value::BlockId,
    ) -> qcode::value::InstructionId {
        BasicBlock::from_id(&tc.ctx, entry)
            .iter()
            .find(|i| matches!(i.mnemonic(), Mnemonic::Call(_)))
            .unwrap()
            .id
    }

    /// A private-space pointer passed to a non-argmem (internal) callee is flagged.
    #[test]
    fn private_ptr_to_internal_callee_is_flagged() {
        let mut tc = qcode::testing::TestContext::new();
        let internal = FunctionBody::make_at_addr(&mut tc.ctx, 0x100, None).id;
        qcode!(
            tc.ctx,
            "
            fn caller:
                <k_entry @p:i64>
                    call fn caller();
                <k_cont>
                    return at i64 0;
            "
        );
        let _ = (k_entry, k_cont);
        let call = caller_call(&tc, k_entry);
        setup(&mut tc, caller, k_entry, call, internal);
        let diags = verify_private_space_call_args(&tc.ctx);
        assert!(
            !diags.is_empty(),
            "a shadow pointer into an internal callee must be flagged: {diags:?}"
        );
    }

    /// A private-space pointer passed to a prototyped external's pointer parameter
    /// is accepted.
    #[test]
    fn private_ptr_to_argmem_external_pointer_param_is_ok() {
        let mut tc = qcode::testing::TestContext::new();
        let ext = FunctionBody::make_external(&mut tc.ctx, 0x9000, Some("memset".into())).id;
        FunctionBody::from_id_mut(&mut tc.ctx, ext).set_argmem(ExternArgmem {
            params: vec![ArgMemKind::OutPtr],
            variadic: false,
        });
        qcode!(
            tc.ctx,
            "
            fn caller:
                <k_entry @p:i64>
                    call fn caller();
                <k_cont>
                    return at i64 0;
            "
        );
        let _ = (k_entry, k_cont);
        let call = caller_call(&tc, k_entry);
        setup(&mut tc, caller, k_entry, call, ext);
        assert!(
            verify_private_space_call_args(&tc.ctx).is_empty(),
            "a shadow pointer into an argmem external's pointer param is well-formed"
        );
    }

    /// A private-space pointer landing on a NON-pointer external parameter is
    /// flagged (the argmem slot must be a pointer kind).
    #[test]
    fn private_ptr_to_nonptr_external_param_is_flagged() {
        let mut tc = qcode::testing::TestContext::new();
        let ext = FunctionBody::make_external(&mut tc.ctx, 0x9000, Some("ext".into())).id;
        FunctionBody::from_id_mut(&mut tc.ctx, ext).set_argmem(ExternArgmem {
            params: vec![ArgMemKind::NonPtr],
            variadic: false,
        });
        qcode!(
            tc.ctx,
            "
            fn caller:
                <k_entry @p:i64>
                    call fn caller();
                <k_cont>
                    return at i64 0;
            "
        );
        let _ = (k_entry, k_cont);
        let call = caller_call(&tc, k_entry);
        setup(&mut tc, caller, k_entry, call, ext);
        assert!(
            !verify_private_space_call_args(&tc.ctx).is_empty(),
            "a shadow pointer on a non-pointer external param slot must be flagged"
        );
    }
}
