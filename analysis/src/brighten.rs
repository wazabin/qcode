use qcode::{
    builder::Builder,
    context::Context,
    error::{Error, ErrorTy, Result},
    space::{Space, SpaceId, SpaceType},
    types::stack_base,
    value::{BasicBlock, Function, FunctionId, RegisterId, ValueId, literal::Literal},
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

    let literal_id: ValueId = make_stack_base(ctx, ptr_width);

    let stack_ptr = ctx.get_register(stack_ptr);
    let reg_space_id = stack_ptr.space().id;
    let stack_ptr = stack_ptr.id();

    let mut builder = Builder::from_block(BasicBlock::from_id_mut(ctx, root_id));
    builder.set_insert_point_to_start();
    unsafe { builder.dont_finalize() };
    builder.push_store(literal_id, stack_ptr, reg_space_id);

    Ok(true)
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
