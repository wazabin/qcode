use std::collections::HashSet;

use qcode::{
    context::Context,
    space::SpaceType,
    value::{Function, FunctionId, ValueId, Varnode, VarnodeId, insn::Mnemonic},
};

/// Returns the set of register-space varnodes written by `function_id`.
///
/// Only direct stores to named register varnodes are counted; stores through
/// computed pointers or into other address spaces are ignored.
pub fn compute_clobbered_regs(ctx: &Context, function_id: FunctionId) -> Vec<VarnodeId> {
    let mut seen: HashSet<VarnodeId> = HashSet::new();
    let mut result: Vec<VarnodeId> = Vec::new();

    for block in Function::from_id(ctx, function_id).iter() {
        for insn in block.iter() {
            if let Mnemonic::Store(store) = insn.mnemonic() {
                if let ValueId::Varnode(vn_id) = store.ptr {
                    if matches!(Varnode::from_id(ctx, vn_id).space().ty, SpaceType::Register)
                        && seen.insert(vn_id)
                    {
                        result.push(vn_id);
                    }
                }
            }
        }
    }

    result
}

/// Computes and stores the clobbered-register set on `function_id`.
pub fn set_clobbered_regs(ctx: &mut Context, function_id: FunctionId) {
    let regs = compute_clobbered_regs(ctx, function_id);
    Function::from_id_mut(ctx, function_id).set_clobbered_regs(regs);
}

#[cfg(test)]
mod tests {
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{
            BasicBlock, Function, FunctionId, ValueId, VarnodeId, function::FunctionSignature,
        },
    };

    use super::*;

    /// Build a test function in a TestContext, invoke `f` to populate its entry block,
    /// and return the context and function ID.
    fn build_fn(f: impl FnOnce(&mut Builder<'static, '_>)) -> (TestContext, FunctionId) {
        let mut tc = TestContext::new();
        let fun_id = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
        let block_id = tc.ctx.get_or_make_block(0x1000);
        Function::from_id_mut(&mut tc.ctx, fun_id)
            .set_root(block_id)
            .unwrap();
        let mut builder = Builder::from_context(&mut tc.ctx, 0x1000);
        f(&mut builder);
        unsafe { builder.dont_finalize() };
        drop(builder);
        (tc, fun_id)
    }

    #[test]
    fn single_store_is_clobbered() {
        let (tc, fun_id) = build_fn(|b| {
            let r0 = b.context().get_named("r0").unwrap().as_varnode().unwrap();
            let reg_space = b.context().try_get_space("register").unwrap();
            let val = b.context_mut().get_const(1u64, 8).id();
            b.push_store(val, ValueId::Varnode(r0), reg_space);
        });
        let regs = compute_clobbered_regs(&tc.ctx, fun_id);
        assert!(regs.contains(&tc.r0));
    }

    #[test]
    fn load_only_not_clobbered() {
        let (tc, fun_id) = build_fn(|b| {
            let r0 = b.context().get_named("r0").unwrap().as_varnode().unwrap();
            let reg_space = b.context().try_get_space("register").unwrap();
            b.push_load::<false>(ValueId::Varnode(r0), 8, reg_space);
        });
        let regs = compute_clobbered_regs(&tc.ctx, fun_id);
        assert!(regs.is_empty(), "load-only should have no clobbered regs");
    }

    #[test]
    fn each_varnode_appears_once() {
        let (tc, fun_id) = build_fn(|b| {
            let r0 = b.context().get_named("r0").unwrap().as_varnode().unwrap();
            let reg_space = b.context().try_get_space("register").unwrap();
            let v1 = b.context_mut().get_const(1u64, 8).id();
            b.push_store(v1, ValueId::Varnode(r0), reg_space);
            let v2 = b.context_mut().get_const(2u64, 8).id();
            b.push_store(v2, ValueId::Varnode(r0), reg_space);
        });
        let regs = compute_clobbered_regs(&tc.ctx, fun_id);
        assert_eq!(
            regs.iter().filter(|&&v| v == tc.r0).count(),
            1,
            "duplicate stores should be deduplicated"
        );
    }

    #[test]
    fn multiple_distinct_regs_all_collected() {
        let (tc, fun_id) = build_fn(|b| {
            let r0 = b.context().get_named("r0").unwrap().as_varnode().unwrap();
            let r1 = b.context().get_named("r1").unwrap().as_varnode().unwrap();
            let reg_space = b.context().try_get_space("register").unwrap();
            let v = b.context_mut().get_const(0u64, 8).id();
            b.push_store(v, ValueId::Varnode(r0), reg_space);
            b.push_store(v, ValueId::Varnode(r1), reg_space);
        });
        let regs = compute_clobbered_regs(&tc.ctx, fun_id);
        assert!(regs.contains(&tc.r0));
        assert!(regs.contains(&tc.r1));
        assert_eq!(regs.len(), 2);
    }

    #[test]
    fn set_clobbered_regs_populates_signature() {
        let (mut tc, fun_id) = build_fn(|b| {
            let r0 = b.context().get_named("r0").unwrap().as_varnode().unwrap();
            let reg_space = b.context().try_get_space("register").unwrap();
            let val = b.context_mut().get_const(42u64, 8).id();
            b.push_store(val, ValueId::Varnode(r0), reg_space);
        });
        set_clobbered_regs(&mut tc.ctx, fun_id);
        let clobbered = Function::from_id(&tc.ctx, fun_id)
            .clobbered_regs()
            .expect("clobbered_regs should be set after set_clobbered_regs");
        assert!(clobbered.contains(&tc.r0));
    }

    #[test]
    fn preserves_existing_signature_fields() {
        let r1_id: VarnodeId;
        let (mut tc, fun_id) = build_fn(|b| {
            let r0 = b.context().get_named("r0").unwrap().as_varnode().unwrap();
            let reg_space = b.context().try_get_space("register").unwrap();
            let val = b.context_mut().get_const(1u64, 8).id();
            b.push_store(val, ValueId::Varnode(r0), reg_space);
        });
        r1_id = tc.r1;
        Function::from_id_mut(&mut tc.ctx, fun_id).set_signature(FunctionSignature {
            outputs: Some(vec![r1_id]),
            ..Default::default()
        });
        set_clobbered_regs(&mut tc.ctx, fun_id);
        let sig = Function::from_id(&tc.ctx, fun_id).signature().unwrap();
        assert!(sig.outputs.is_some(), "outputs should be preserved");
        assert!(sig.clobbered.is_some(), "clobbered should be set");
    }
}
