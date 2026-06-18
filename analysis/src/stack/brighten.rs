use qcode::{
    builder::Builder,
    context::Context,
    error::{Error, ErrorTy, Result},
    space::{Space, SpaceId, SpaceType},
    types::stack_base,
    value::{
        BasicBlock, BlockId, Function, FunctionId, RegisterId, ValueId,
        insn::Mnemonic,
        literal::Literal,
    },
};

const STACK_NAME: &str = "stack";

pub fn get_or_make_stack_space(ctx: &mut Context, addr_size: usize) -> SpaceId {
    ctx.try_get_space(STACK_NAME).unwrap_or_else(|| {
        ctx.add_space(Space {
            name: Some(Box::from(STACK_NAME)),
            word_size: 1,
            addr_size,
            ty: SpaceType::Ram,
        })
    })
}

fn make_stack_base(ctx: &mut Context, ptr_width: usize) -> ValueId {
    let stack_space_id = get_or_make_stack_space(ctx, ptr_width);
    let type_id = ctx
        .types
        .get_or_make_stack_address(ptr_width, Some(stack_space_id));

    ctx.values
        .push_literal(Literal {
            value: stack_base(ptr_width),
            type_id,
            symbolic: None,
        })
        .into()
}

/// Inject a synthetic `*[register]:8 <stack_ptr> = @stack_base` store at the
/// entry of `function_id`, giving the stack pointer a recognizable symbolic
/// origin for downstream mem2reg promotion of stack-relative addresses.
///
/// The literal `STACK_BASE` is tagged with a `SymbolicRef::Space`
/// pointing to a lazily-created "stack" address space.  Running this pass
/// before `mem2reg` lets the promoter treat stack slots as promotable memory.
pub fn brighten_stack(
    ctx: &mut Context,
    function_id: FunctionId,
    stack_ptr: RegisterId,
) -> Result<bool> {
    // The pointer width is the stack-pointer register's own width (8 for RSP, 4
    // for ESP); it sizes the synthetic stack base so the injected store matches
    // the register and the base survives truncation to that register.
    let ptr_width = ctx.get_register(stack_ptr).size();

    let root_id = Function::from_id(ctx, function_id)
        .root()
        .ok_or_else(|| Error::spanless(ErrorTy::NoRootBlock))?
        .id;

    // Idempotent: the pass may be re-run across fixpoint rounds, so bail out if the
    // synthetic stack-base store is already present at the root. Doing this before
    // creating the literal/space avoids leaking a duplicate literal.
    if root_has_stack_base_store(ctx, root_id, ptr_width) {
        return Ok(false);
    }

    let stack_ptr = ctx.get_register(stack_ptr);
    let reg_space_id = stack_ptr.space().id;
    let stack_ptr = stack_ptr.id();

    // For a `pure_reg` function, `argpromote_registers` has already seeded the
    // stack pointer from a by-value input param: `store(rsp_param, RSP)` is the
    // first thing in the entry. Injecting a base store *ahead* of it (as below)
    // would be immediately clobbered by that seed, so the symbolic origin would
    // never reach mem2reg. Instead, replace every use of that param with
    // `@stack_base` — including the seed (which becomes the base store) and any
    // further use the param flowed to (e.g. a loop-header block argument). This
    // makes the entry's stack-pointer origin identical to a non-functionalized
    // function; the now-unused param is reclaimed by DCE / `dead_signature`.
    if Function::from_id(ctx, function_id).is_pure_reg()
        && let Some(param) = entry_stack_ptr_seed_param(ctx, root_id, stack_ptr, reg_space_id)
    {
        let literal_id = make_stack_base(ctx, ptr_width);
        ctx.replace_all_uses_with(param, literal_id);
        return Ok(true);
    }

    let literal_id: ValueId = make_stack_base(ctx, ptr_width);

    let mut builder = Builder::from_block(BasicBlock::from_id_mut(ctx, root_id));
    builder.set_insert_point_to_start();
    unsafe { builder.dont_finalize() };
    builder.push_store(literal_id, stack_ptr, reg_space_id);

    Ok(true)
}

/// The by-value param the argpromote register seed writes into the stack-pointer
/// register (`store(rsp_param, RSP)` in register space) at entry — i.e. the
/// `src` of that seed store, if present. Replacing every use of this param with
/// `@stack_base` gives the stack pointer its symbolic origin in a functionalized
/// function.
fn entry_stack_ptr_seed_param(
    ctx: &Context,
    root_id: BlockId,
    stack_ptr: ValueId,
    reg_space: SpaceId,
) -> Option<ValueId> {
    BasicBlock::from_id(ctx, root_id).iter().find_map(|insn| {
        match insn.mnemonic() {
            Mnemonic::Store(store) if store.ptr == stack_ptr && store.space == reg_space => {
                Some(store.src)
            }
            _ => None,
        }
    })
}

/// True if `root_id` already begins with the synthetic stack-base store this pass
/// injects — i.e. a store whose source is the `@stack_base` literal of the given
/// pointer width. Used to keep [`brighten_stack`] idempotent when re-run.
fn root_has_stack_base_store(
    ctx: &Context,
    root_id: qcode::value::BlockId,
    ptr_width: usize,
) -> bool {
    use qcode::value::insn::Mnemonic;
    let base = stack_base(ptr_width);
    BasicBlock::from_id(ctx, root_id).iter().any(|insn| {
        if let Mnemonic::Store(store) = insn.mnemonic() {
            store
                .src
                .as_literal()
                .is_some_and(|lit| ctx.values.literals[lit].value == base)
        } else {
            false
        }
    })
}

#[cfg(test)]
mod tests {
    use qcode::{
        context::Context,
        error::ErrorTy,
        space::{Space, SpaceType},
        value::{BasicBlock, Function, RegisterId, Varnode, insn::Mnemonic},
    };
    use qcode_macro::qcode;

    use super::*;

    /// Creates a register space and a single 8-byte stack-pointer varnode,
    /// registering it under `RegisterId(0)`.
    fn setup_sp(ctx: &mut Context) -> RegisterId {
        let reg_space = ctx.add_space(Space {
            name: Some(Box::from("register")),
            word_size: 1,
            addr_size: 8,
            ty: SpaceType::Register,
        });
        let sp_id = Varnode::make(ctx, 0, 8, reg_space).id;
        let reg_id = RegisterId::from(0usize);
        ctx.registers.insert(reg_id, sp_id);
        reg_id
    }

    #[test]
    fn store_is_first_instruction_in_root_block() {
        let mut ctx = Context::new();
        let reg_id = setup_sp(&mut ctx);

        qcode!(ctx, "fn test: <entry> goto <0x1001>;");

        brighten_stack(&mut ctx, test, reg_id).unwrap();

        let root = Function::from_id(&ctx, test).root().unwrap();
        let first = root.iter().next().expect("root block has instructions");
        assert!(
            matches!(first.mnemonic(), Mnemonic::Store(_)),
            "first instruction should be a store, got: {}",
            first
        );
    }

    #[test]
    fn brighten_is_idempotent_when_rerun() {
        let mut ctx = Context::new();
        let reg_id = setup_sp(&mut ctx);

        qcode!(ctx, "fn test: <entry> goto <0x1001>;");

        brighten_stack(&mut ctx, test, reg_id).unwrap();
        brighten_stack(&mut ctx, test, reg_id).unwrap();

        let root = Function::from_id(&ctx, test).root().unwrap();
        let stores = root
            .iter()
            .filter(|insn| matches!(insn.mnemonic(), Mnemonic::Store(_)))
            .count();
        assert_eq!(
            stores, 1,
            "re-running brighten must not duplicate the store"
        );
    }

    #[test]
    fn store_precedes_existing_instructions() {
        let mut ctx = Context::new();
        let reg_id = setup_sp(&mut ctx);

        qcode!(
            ctx,
            "
            varnode i64 A;
            fn test:
                <entry>
                    %x = load(i64, &A);
                    goto <0x1001>;
        "
        );

        let root_id = Function::from_id(&ctx, test).root().unwrap().id;
        let count_before = BasicBlock::from_id(&ctx, root_id).instruction_ids().len();

        brighten_stack(&mut ctx, test, reg_id).unwrap();

        let root = BasicBlock::from_id(&ctx, root_id);
        assert_eq!(root.instruction_ids().len(), count_before + 1);

        let first = root.iter().next().unwrap();
        assert!(
            matches!(first.mnemonic(), Mnemonic::Store(_)),
            "injected store should be first"
        );
    }

    #[test]
    fn store_src_is_stack_address_typed() {
        let mut ctx = Context::new();
        let reg_id = setup_sp(&mut ctx);

        qcode!(ctx, "fn test: <entry> goto <0x1001>;");

        brighten_stack(&mut ctx, test, reg_id).unwrap();

        let root = Function::from_id(&ctx, test).root().unwrap();
        let first = root.iter().next().unwrap();
        let Mnemonic::Store(store) = first.mnemonic() else {
            panic!("expected store");
        };

        let ValueId::Literal(lit_id) = store.src else {
            panic!("store src should be a literal, got: {:?}", store.src);
        };

        let lit = &ctx.values.literals[lit_id];
        assert_eq!(lit.value, stack_base(8));
        assert_eq!(ctx.types.size_of(lit.type_id), 8);
        assert!(
            ctx.types.is_stack_address(lit.type_id),
            "literal should be StackAddress type"
        );
    }

    #[test]
    fn stack_space_reused_across_functions() {
        let mut ctx = Context::new();
        let reg_id = setup_sp(&mut ctx);

        qcode!(ctx, "fn f1: <e1> goto <0x1001>;");
        qcode!(ctx, "fn f2: <e2> goto <0x2001>;");

        brighten_stack(&mut ctx, f1, reg_id).unwrap();
        let space_after_first = ctx.try_get_space("stack").expect("stack space created");

        brighten_stack(&mut ctx, f2, reg_id).unwrap();
        let space_after_second = ctx
            .try_get_space("stack")
            .expect("stack space still present");

        assert_eq!(
            space_after_first, space_after_second,
            "stack space should be reused, not duplicated"
        );
    }

    /// For a `pure_reg` function the stack pointer is already seeded from an
    /// input param at entry; brighten must redirect that seed to `@stack_base`
    /// (not inject a second, clobbered store), leaving the param dead.
    #[test]
    fn redirects_rsp_seed_for_pure_reg_function() {
        let mut ctx = Context::new();
        let reg_id = setup_sp(&mut ctx);

        qcode!(ctx, "fn test: <entry> goto <0x1001>;");

        // Emulate argpromote_registers: a by-value param seeded into RSP at entry.
        let sp_vn = ctx.get_register(reg_id).id();
        let reg_space = ctx.get_register(reg_id).space().id;
        let root_id = Function::from_id(&ctx, test).root().unwrap().id;
        let param = {
            let p = BasicBlock::from_id_mut(&mut ctx, root_id).push_param(8).id;
            ValueId::BlockParam(p)
        };
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, root_id));
            b.set_insert_point_to_start();
            unsafe { b.dont_finalize() };
            b.push_store(param, sp_vn, reg_space);
        }
        Function::from_id_mut(&mut ctx, test).set_pure_reg(true);

        assert!(brighten_stack(&mut ctx, test, reg_id).unwrap());

        let root = Function::from_id(&ctx, test).root().unwrap();
        let stores: Vec<_> = root
            .iter()
            .filter(|i| matches!(i.mnemonic(), Mnemonic::Store(_)))
            .collect();
        assert_eq!(stores.len(), 1, "the seed is redirected, not duplicated");
        let Mnemonic::Store(store) = stores[0].mnemonic() else {
            unreachable!()
        };
        assert!(
            matches!(store.src, ValueId::Literal(_)),
            "RSP is now seeded from the @stack_base literal"
        );
        assert!(
            ctx.users(param).is_empty(),
            "the input param that seeded RSP is now dead"
        );

        // Idempotent: re-running finds the base store and makes no change.
        assert!(!brighten_stack(&mut ctx, test, reg_id).unwrap());
    }

    #[test]
    fn err_on_function_without_root() {
        let mut ctx = Context::new();
        let reg_id = setup_sp(&mut ctx);

        let fn_id = Function::make(&mut ctx, "rootless".into()).unwrap().id;

        let result = brighten_stack(&mut ctx, fn_id, reg_id);
        assert!(
            matches!(result, Err(ref e) if matches!(e.ty, ErrorTy::NoRootBlock)),
            "expected NoRootBlock error, got: {result:?}"
        );
    }
}

// ----- pass ------------------------------------------------------------------

use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct Brighten;

impl FunctionPass for Brighten {
    const NAME: &'static str = "brighten";
    fn description(&self) -> &'static str {
        "Inject symbolic stack base store at function entry"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        env: &PipelineEnv,
    ) -> std::result::Result<bool, String> {
        brighten_stack(ctx, fun_id, env.cfg.stack_pointer).map_err(|e| e.to_string())
    }
}

crate::register_function_pass!(Brighten);
