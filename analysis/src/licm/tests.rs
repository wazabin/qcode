//! Unit tests for loop-invariant code motion.

use qcode::{
    context::Context,
    value::{
        BasicBlock,
        block::BlockId,
        insn::{Binop, IntBinop, Mnemonic},
    },
};
use qcode_macro::qcode;

use super::Licm;
use crate::test_util::run_function_pass;

/// Number of integer `add` instructions in `block`.
fn count_adds(ctx: &Context, block: BlockId) -> usize {
    BasicBlock::from_id(ctx, block)
        .instruction_ids()
        .iter()
        .filter(|&&id| {
            matches!(
                ctx.get_insn(id).mnemonic(),
                Mnemonic::Binop(b) if b.op == Binop::Int(IntBinop::Add)
            )
        })
        .count()
}

/// Number of `load` instructions in `block`.
fn count_loads(ctx: &Context, block: BlockId) -> usize {
    BasicBlock::from_id(ctx, block)
        .instruction_ids()
        .iter()
        .filter(|&&id| matches!(ctx.get_insn(id).mnemonic(), Mnemonic::Load(_)))
        .count()
}

/// An invariant pure add in the loop body is moved into the preheader; the
/// loop-carried induction increment stays put.
#[test]
fn hoists_invariant_arithmetic() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
            varnode i64 A;
            varnode i64 B;

            fn f:
                <entry>
                    %x = load(i64, &A);
                    %y = load(i64, &B);
                    goto <header @i=0x0>;
                <header @i:i64>
                    %cond = @i < 0x3;
                    if %cond goto <body> else goto <exit>;
                <body>
                    %z = %x + %y;
                    %i_next = @i + 0x1;
                    goto <header @i=%i_next>;
                <exit>
                    return at 0x0;
        "
    );

    assert_eq!(count_adds(&ctx, entry), 0, "entry starts with no adds");
    assert_eq!(count_adds(&ctx, body), 2, "body starts with z and i_next");

    assert!(run_function_pass::<Licm>(&mut ctx, f).unwrap());

    assert_eq!(
        count_adds(&ctx, entry),
        1,
        "the invariant add %z is hoisted into the preheader"
    );
    assert_eq!(
        count_adds(&ctx, body),
        1,
        "only the loop-carried induction increment remains in the body"
    );
    assert!(
        !BasicBlock::from_id(&ctx, body)
            .instruction_ids()
            .contains(&z),
        "the original %z is gone from the body"
    );
}

/// A transitive chain of invariant ops all hoist, even when an inner op depends
/// on a previously-hoisted one.
#[test]
fn hoists_transitive_chain() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
            varnode i64 A;
            varnode i64 B;

            fn f:
                <entry>
                    %x = load(i64, &A);
                    %y = load(i64, &B);
                    goto <header @i=0x0>;
                <header @i:i64>
                    %cond = @i < 0x3;
                    if %cond goto <body> else goto <exit>;
                <body>
                    %s = %x + %y;
                    %t = %s * %x;
                    %u = %t + %y;
                    %i_next = @i + 0x1;
                    goto <header @i=%i_next>;
                <exit>
                    return at 0x0;
        "
    );

    assert!(run_function_pass::<Licm>(&mut ctx, f).unwrap());

    for v in [s, t, u] {
        assert!(
            !BasicBlock::from_id(&ctx, body)
                .instruction_ids()
                .contains(&v),
            "every invariant op leaves the body"
        );
    }
    assert_eq!(
        count_adds(&ctx, body),
        1,
        "only the induction increment remains"
    );
}

/// A use of a loop-carried block parameter is variant and must not move.
#[test]
fn keeps_loop_variant_in_body() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
            varnode i64 A;

            fn f:
                <entry>
                    %x = load(i64, &A);
                    goto <header @i=0x0>;
                <header @i:i64>
                    %cond = @i < 0x3;
                    if %cond goto <body> else goto <exit>;
                <body>
                    %v = %x + @i;
                    %i_next = @i + 0x1;
                    goto <header @i=%i_next>;
                <exit>
                    return at 0x0;
        "
    );

    assert!(
        !run_function_pass::<Licm>(&mut ctx, f).unwrap(),
        "nothing is invariant: %v depends on the loop counter @i"
    );
    assert!(
        BasicBlock::from_id(&ctx, body)
            .instruction_ids()
            .contains(&v),
        "%v stays in the body"
    );
}

/// An invariant load whose location no loop store touches is hoisted.
#[test]
fn hoists_invariant_load() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
            varnode i32 G;

            fn f:
                <entry>
                    goto <header @i=0x0>;
                <header @i:i64>
                    %cond = @i < 0x3;
                    if %cond goto <body> else goto <exit>;
                <body>
                    %v = load(i32, &G);
                    %i_next = @i + 0x1;
                    goto <header @i=%i_next>;
                <exit>
                    return at 0x0;
        "
    );

    assert_eq!(count_loads(&ctx, entry), 0);
    assert_eq!(count_loads(&ctx, body), 1);

    assert!(run_function_pass::<Licm>(&mut ctx, f).unwrap());

    assert_eq!(
        count_loads(&ctx, entry),
        1,
        "the invariant load is hoisted into the preheader"
    );
    assert_eq!(count_loads(&ctx, body), 0, "no load remains in the body");
}

/// A load is NOT hoisted when a store in the loop may write the same location.
/// (With no alias oracle in the test env, any loop store is conservatively
/// assumed to alias.)
#[test]
fn keeps_load_with_aliasing_store() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
            varnode i32 G;

            fn f:
                <entry>
                    goto <header @i=0x0>;
                <header @i:i64>
                    %cond = @i < 0x3;
                    if %cond goto <body> else goto <exit>;
                <body>
                    %v = load(i32, &G);
                    store(&G, i32 0x5);
                    %i_next = @i + 0x1;
                    goto <header @i=%i_next>;
                <exit>
                    return at 0x0;
        "
    );

    assert!(
        !run_function_pass::<Licm>(&mut ctx, f).unwrap(),
        "the loop stores into the loaded location, so the load is not invariant"
    );
    assert_eq!(count_loads(&ctx, body), 1, "the load stays in the body");
}

/// With a real alias oracle, a load IS hoisted across a store the oracle proves
/// disjoint (a store to a *different* global). This is the case the conservative
/// no-oracle path cannot handle and the motivating example needs.
#[test]
fn hoists_load_over_disjoint_store_with_oracle() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
            varnode i32 G;
            varnode i32 H;

            fn f:
                <entry>
                    goto <header @i=0x0>;
                <header @i:i64>
                    %cond = @i < 0x3;
                    if %cond goto <body> else goto <exit>;
                <body>
                    %v = load(i32, &G);
                    store(&H, i32 0x5);
                    %i_next = @i + 0x1;
                    goto <header @i=%i_next>;
                <exit>
                    return at 0x0;
        "
    );

    let aliases = crate::RegisterBase::build(&ctx).for_function(&ctx, f);
    assert!(
        super::hoist_loop_invariants_with_aliases(&mut ctx, f, Some(&aliases)),
        "load of G should hoist: the only loop store targets the disjoint global H"
    );
    assert_eq!(
        count_loads(&ctx, entry),
        1,
        "the load is hoisted into the preheader"
    );
    assert_eq!(count_loads(&ctx, body), 0, "no load remains in the body");
}

/// The same oracle keeps a load whose location a loop store may write.
#[test]
fn keeps_load_over_aliasing_store_with_oracle() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
            varnode i32 G;

            fn f:
                <entry>
                    goto <header @i=0x0>;
                <header @i:i64>
                    %cond = @i < 0x3;
                    if %cond goto <body> else goto <exit>;
                <body>
                    %v = load(i32, &G);
                    store(&G, i32 0x5);
                    %i_next = @i + 0x1;
                    goto <header @i=%i_next>;
                <exit>
                    return at 0x0;
        "
    );

    let aliases = crate::RegisterBase::build(&ctx).for_function(&ctx, f);
    assert!(
        !super::hoist_loop_invariants_with_aliases(&mut ctx, f, Some(&aliases)),
        "load of G must stay: the loop stores into G"
    );
    assert_eq!(count_loads(&ctx, body), 1, "the load stays in the body");
}

/// With no unique preheader (two non-loop predecessors enter the header),
/// nothing is hoisted.
#[test]
fn skips_loop_without_preheader() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
            fn f:
                <entry>
                    if i8 0x1 goto <p1> else goto <p2>;
                <p1>
                    goto <header @i=0x0>;
                <p2>
                    goto <header @i=0x0>;
                <header @i:i64>
                    %x = i64 0x5 + i64 0x7;
                    %cond = @i < 0x3;
                    if %cond goto <body> else goto <exit>;
                <body>
                    %i_next = @i + 0x1;
                    goto <header @i=%i_next>;
                <exit>
                    return at 0x0;
        "
    );

    assert!(
        !run_function_pass::<Licm>(&mut ctx, f).unwrap(),
        "the header has two preheaders, so no hoist target exists"
    );
    assert!(
        BasicBlock::from_id(&ctx, header)
            .instruction_ids()
            .contains(&x),
        "%x remains in the header"
    );
}

/// A second run reaches a fixpoint: once invariants are in the preheader there
/// is nothing left in the loop to move.
#[test]
fn is_idempotent() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
            varnode i64 A;
            varnode i64 B;

            fn f:
                <entry>
                    %x = load(i64, &A);
                    %y = load(i64, &B);
                    goto <header @i=0x0>;
                <header @i:i64>
                    %cond = @i < 0x3;
                    if %cond goto <body> else goto <exit>;
                <body>
                    %z = %x + %y;
                    %i_next = @i + 0x1;
                    goto <header @i=%i_next>;
                <exit>
                    return at 0x0;
        "
    );

    assert!(
        run_function_pass::<Licm>(&mut ctx, f).unwrap(),
        "first run hoists"
    );
    assert!(
        !run_function_pass::<Licm>(&mut ctx, f).unwrap(),
        "second run finds nothing to hoist"
    );
}

/// A hoisted value still in use inside the loop is correctly rewired: the
/// consumer now references the copy that lives in the preheader.
#[test]
fn rewires_uses_to_hoisted_copy() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
            varnode i64 A;
            varnode i64 B;
            varnode i64 OUT;

            fn f:
                <entry>
                    %x = load(i64, &A);
                    %y = load(i64, &B);
                    goto <header @i=0x0>;
                <header @i:i64>
                    %cond = @i < 0x3;
                    if %cond goto <body> else goto <exit>;
                <body>
                    %z = %x + %y;
                    store(&OUT, %z);
                    %i_next = @i + 0x1;
                    goto <header @i=%i_next>;
                <exit>
                    return at 0x0;
        "
    );

    assert!(run_function_pass::<Licm>(&mut ctx, f).unwrap());

    // The store in the body must now read an add that lives in the preheader.
    let store_id = *BasicBlock::from_id(&ctx, body)
        .instruction_ids()
        .iter()
        .find(|&&id| matches!(ctx.get_insn(id).mnemonic(), Mnemonic::Store(_)))
        .expect("body still has the store");
    let Mnemonic::Store(store) = ctx.get_insn(store_id).mnemonic() else {
        unreachable!()
    };
    let src = store.src;
    let qcode::value::ValueId::Instruction(src_id) = src else {
        panic!("store source should be the hoisted add")
    };
    let parent = ctx.get_insn(src_id).parent().map(|b| b.id);
    assert_eq!(
        parent,
        Some(entry),
        "the hoisted add lives in the preheader (entry), and the body store reads it"
    );
}
