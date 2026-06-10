use qcode_macro::qcode;

use super::*;
use qcode::{
    context::Context,
    value::{
        BasicBlock, Value,
        insn::{Binary, Binop, IntBinop},
    },
};

// 1. Same binop -> second is redundant, leader is first
#[test]
fn test_same_binop_redundant() {
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
            varnode i64 A;
            varnode i64 B;

            <block>
                %a = load(i64, &A);
                %b = load(i64, &B);

                %v1 = %a + %b;
                %v2 = %a + %b;

                goto <0x1001>;
        "
    );

    let mut block = BasicBlock::from_id_mut(&mut ctx, block);

    assert!(block.instruction_ids().contains(&v1));
    assert!(block.instruction_ids().contains(&v2));

    gvn(&mut block, None);

    assert!(block.instruction_ids().contains(&v1));
    assert!(!block.instruction_ids().contains(&v2));
}

// 2. Commutative normalization: a + b and b + a -> same value number
#[test]
fn test_commutative_normalization() {
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
            varnode i64 A;
            varnode i64 B;

            <block>
                %a = load(i64, &A);
                %b = load(i64, &B);
                %v1 = %a + %b;
                %v2 = %b + %a;
                goto <0x1001>;
        "
    );

    let mut block = BasicBlock::from_id_mut(&mut ctx, block);

    assert!(block.instruction_ids().contains(&v1));
    assert!(block.instruction_ids().contains(&v2));

    gvn(&mut block, None);

    assert!(block.instruction_ids().contains(&v1));
    assert!(!block.instruction_ids().contains(&v2));
}

// 3. Non-commutative not swapped: a - b != b - a
#[test]
fn test_non_commutative_not_swapped() {
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
            varnode i64 A;
            varnode i64 B;

            <block>
                %a = load(i64, &A);
                %b = load(i64, &B);
                %v1 = %a - %b;
                %v2 = %b - %a;
                goto <0x1001>;
        "
    );

    let mut block = BasicBlock::from_id_mut(&mut ctx, block);

    assert!(block.instruction_ids().contains(&v1));
    assert!(block.instruction_ids().contains(&v2));

    gvn(&mut block, None);

    assert!(block.instruction_ids().contains(&v1));
    assert!(block.instruction_ids().contains(&v2));
}

// 4. Different ops -> distinct
#[test]
fn test_different_ops_distinct() {
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
            varnode i64 A;
            varnode i64 B;

            <block>
                %a = load(i64, &A);
                %b = load(i64, &B);
                %v1 = %a + %b;
                %v2 = %a * %b;
                goto <0x1001>;
        "
    );

    let mut block = BasicBlock::from_id_mut(&mut ctx, block);

    assert!(block.instruction_ids().contains(&v1));
    assert!(block.instruction_ids().contains(&v2));

    gvn(&mut block, None);

    assert!(block.instruction_ids().contains(&v1));
    assert!(block.instruction_ids().contains(&v2));
}

// 5. Cross-block redundancy: a+b in entry propagates to dominated successor
#[test]
fn test_gvn_function_cross_block_redundancy() {
    // Layout: entry → succ (linear chain, succ dominated by entry)
    // entry: v1 = a + b
    // succ:  v2 = a + b  <- redundant, dominated by entry
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
            varnode i64 A;
            varnode i64 B;

            fn f:
                <entry>
                    %a = load(i64, &A);
                    %b = load(i64, &B);

                    %v1 = %a + %b;
                    goto <succ>;

                <succ>
                    %v2 = %a + %b;
                    return [0x1000];
            "
    );

    gvn_function(&mut ctx, f, None);

    assert!(
        BasicBlock::from_id(&ctx, entry)
            .instruction_ids()
            .contains(&v1)
    );
    assert!(
        !BasicBlock::from_id(&ctx, succ)
            .instruction_ids()
            .contains(&v2),
        "a+b in dominated successor should be eliminated"
    );
}

// 6. Diamond CFG: a+b computed in one branch is NOT available at merge
#[test]
fn test_gvn_function_no_propagation_across_merge() {
    // Layout: entry → left, entry → right, left → merge, right → merge
    // left:  v1 = a + b
    // right: (no a+b)
    // merge: v2 = a + b  <- NOT redundant; merge is dominated only by entry
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
            varnode i64 A;
            varnode i64 B;

            fn g:
                <entry>
                    %a = load(i64, &A);
                    %b = load(i64, &B);
                    if i8 1 goto <left> else goto <right>;

                <left>
                    %v1 = %a + %b;
                    goto <merge>;

                <right>
                    goto <merge>;

                <merge>
                    %v2 = %a + %b;"
    );

    gvn_function(&mut ctx, g, None);

    assert!(
        BasicBlock::from_id(&ctx, left)
            .instruction_ids()
            .contains(&v1)
    );
    assert!(
        BasicBlock::from_id(&ctx, merge)
            .instruction_ids()
            .contains(&v2),
        "a+b at merge must NOT be eliminated: merge is not dominated by left"
    );
}

#[test]
fn test_gvn_function_does_not_forward_loads_across_loop_header() {
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
            varnode i32 A;

            fn loop_load:
                <entry>
                    store(&A, i32 0);
                    goto <header>;

                <header>
                    %v = load(i32, &A);
                    if i8 1 goto <body> else goto <exit>;

                <body>
                    store(&A, i32 1);
                    goto <header>;

                <exit>
                    return [0x1000];
            "
    );

    let aliases = AliasResult::simple(&ctx);
    gvn_function(&mut ctx, loop_load, Some(&aliases));

    assert!(
        BasicBlock::from_id(&ctx, header)
            .instruction_ids()
            .contains(&v),
        "header load must not be replaced by the entry store; the backedge may overwrite it"
    );
}

#[test]
fn test_gvn_function_forwards_loads_across_loop_header_when_body_stores_do_not_alias() {
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
            varnode i32 A;
            varnode i32 B;

            fn loop_load:
                <entry>
                    store(&A, i32 7);
                    goto <header>;

                <header>
                    %v = load(i32, &A);
                    if i8 1 goto <body> else goto <exit>;

                <body>
                    store(&B, i32 1);
                    goto <header>;

                <exit>
                    return [0x1000];
            "
    );

    let aliases = AliasResult::simple(&ctx);
    gvn_function(&mut ctx, loop_load, Some(&aliases));

    assert!(
        !BasicBlock::from_id(&ctx, header)
            .instruction_ids()
            .contains(&v),
        "header load should be replaced by the dominating store when loop stores do not alias"
    );
}

// 7. Constant propagation
#[test]
fn test_constant_propagation() {
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
            varnode i64 A;
            varnode i64 B;
            <block>
                store(&A, i64 5);
                %a = load(i64, &A);
                %v1 = %a + 2;
                %v2 = %v1 + 3;
                store(&B, %v2);
                goto <0x1001>;"
    );

    let aliases = AliasResult::simple(&ctx);

    let mut block = BasicBlock::from_id_mut(&mut ctx, block);

    assert!(block.instruction_ids().contains(&v1));
    assert!(block.instruction_ids().contains(&v2));

    println!("{}", block);
    gvn(&mut block, Some(&aliases));
    println!("{}", block);

    assert!(!block.instruction_ids().contains(&v1));
    assert!(!block.instruction_ids().contains(&v2));
    assert!(block.to_string().contains("B = 0xa"));
}

#[test]
#[should_panic(expected = "type error in binop constant folding")]
fn constant_folding_reports_mixed_size_literals() {
    let mut ctx = Context::new();
    let lhs = ctx.get_const(0xf0, 4).id();
    let rhs = ctx.get_const(0xff, 1).id();

    let _ = constant_folding(
        &mut ctx,
        &Mnemonic::Binop(Binary {
            op: Binop::Int(IntBinop::And),
            lhs,
            rhs,
        }),
        4,
    );
}

#[test]
fn constant_folding_preserves_stack_address_type_through_folding() {
    let mut ctx = Context::new();
    let stack = ctx.add_space(qcode::space::Space {
        name: Some(Box::from("stack")),
        word_size: 1,
        addr_size: 8,
        ty: qcode::space::SpaceType::Ram,
    });
    // StackAddress-typed literals carry provenance in their TypeId, not in a
    // symbolic annotation. Constant folding must propagate that type so that
    // alias analysis can still distinguish SA results from plain integers.
    let sa_type = ctx.types.get_or_make_stack_address(8, Some(stack));
    let base_lid = ctx
        .values
        .get_or_make_typed_literal(0x1000_0000_0000_0000, sa_type, 8);
    let base = qcode::value::ValueId::Literal(base_lid);
    let offset = ctx.get_const(8, 8).id();

    let folded = constant_folding(
        &mut ctx,
        &Mnemonic::Binop(Binary {
            op: Binop::Int(IntBinop::Sub),
            lhs: base,
            rhs: offset,
        }),
        8,
    );

    let folded_id = folded.expect("SA - Int should constant-fold to a SA-typed literal");
    let qcode::value::ValueId::Literal(lid) = folded_id else {
        panic!("folded result must be a literal");
    };
    assert_eq!(
        ctx.values.literals[lid].type_id, sa_type,
        "folded SA - Int must preserve the StackAddress TypeId for alias analysis"
    );
}

// -----------------------------------------------------------------------
// Algebraic identities
// -----------------------------------------------------------------------

/// `x & x` is idempotent: the AND is replaced by `x` itself.
#[test]
fn test_algebraic_and_self_is_idempotent() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
            varnode i64 A;
            varnode i64 B;
            <block>
                %a = load(i64, &A);
                %v = %a & %a;
                store(&B, %v);
                goto <0x1001>;
        "
    );

    let aliases = AliasResult::simple(&ctx);
    let mut block = BasicBlock::from_id_mut(&mut ctx, block);
    assert!(block.instruction_ids().contains(&v));

    gvn(&mut block, Some(&aliases));

    assert!(
        !block.instruction_ids().contains(&v),
        "x & x should be eliminated"
    );
    assert!(
        block.instruction_ids().contains(&a),
        "the AND should be replaced by x itself, which stays live"
    );
}

/// `x + 0` collapses to `x`.
#[test]
fn test_algebraic_add_zero_identity() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
            varnode i64 A;
            varnode i64 B;
            <block>
                %a = load(i64, &A);
                %v = %a + 0x0;
                store(&B, %v);
                goto <0x1001>;
        "
    );

    let aliases = AliasResult::simple(&ctx);
    let mut block = BasicBlock::from_id_mut(&mut ctx, block);

    gvn(&mut block, Some(&aliases));

    assert!(
        !block.instruction_ids().contains(&v),
        "x + 0 should be eliminated"
    );
    assert!(block.instruction_ids().contains(&a));
}

/// `x ^ x` folds to the zero constant.
#[test]
fn test_algebraic_xor_self_is_zero() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
            varnode i64 A;
            varnode i64 B;
            <block>
                %a = load(i64, &A);
                %v = %a ^ %a;
                store(&B, %v);
                goto <0x1001>;
        "
    );

    let aliases = AliasResult::simple(&ctx);
    let mut block = BasicBlock::from_id_mut(&mut ctx, block);

    gvn(&mut block, Some(&aliases));

    assert!(
        !block.instruction_ids().contains(&v),
        "x ^ x should be eliminated"
    );
    assert!(
        block.to_string().contains("B = 0x0"),
        "x ^ x should fold to the zero constant, got:\n{block}"
    );
}

// -----------------------------------------------------------------------
// Signed-compare flag idiom
// -----------------------------------------------------------------------

/// `sborrow(a, b) != ((a - b) s< 0)` is the x86 signed-less-than idiom and
/// must collapse to a single `a s< b`, leaving the flag math dead.
#[test]
fn test_flag_idiom_collapses_to_signed_less_than() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
            varnode i32 A;
            varnode i32 B;

            fn cmp:
                <entry>
                    %a = load(i32, &A);
                    %b = load(i32, &B);
                    %of = sborrow(%a, %b);
                    %sub = %a - %b;
                    %sf = %sub s< 0x0;
                    %lt = %of != %sf;
                    if %lt goto <t> else goto <e>;
                <t>
                    return [0x1000];
                <e>
                    return [0x2000];
            "
    );

    let aliases = AliasResult::simple(&ctx);
    gvn_function(&mut ctx, cmp, Some(&aliases));

    assert!(
        !BasicBlock::from_id(&ctx, entry)
            .instruction_ids()
            .contains(&lt),
        "the `!=` flag combination should be rewritten away"
    );

    // After DCE the dead sborrow/sub/slt chain disappears, leaving the
    // single signed comparison the idiom was recognized as.
    crate::remove_dead_insns(&mut ctx, entry);
    let text = BasicBlock::from_id(&ctx, entry).to_string();
    assert!(
        text.contains("s<"),
        "expected a single signed-less-than, got:\n{text}"
    );
    assert!(
        !text.contains("sborrow") && !text.contains("!="),
        "the flag math should be dead after the rewrite, got:\n{text}"
    );
}

// -----------------------------------------------------------------------
// Register store→load forwarding
// -----------------------------------------------------------------------

/// Storing to a register and reading it straight back must forward the
/// stored value, even when overlapping sub-registers (r0/r0_lo32/...) put
/// the location in a multi-member alias class. Mirrors the post-call
/// `*[register]:4 EAX = v; %r = *[register]:4 EAX` reload chains the lifter
/// emits across every fixture.
#[test]
fn test_register_store_load_forwarding() {
    use qcode::{builder::Builder, testing::TestContext, value::Function};

    let mut tc = TestContext::new();
    let fun_id = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
    let block_id = tc.ctx.get_or_make_block(0x1000);
    Function::from_id_mut(&mut tc.ctx, fun_id)
        .set_root(block_id)
        .unwrap();

    let reg_space = tc.reg_space;
    let eax = ValueId::Varnode(tc.r0_lo32);
    let other = ValueId::Varnode(tc.r1);

    let load_id;
    {
        let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
        let c = b.context_mut().get_const(0x12345678, 4).id();
        b.push_store(c, eax, reg_space); // EAX = c
        let loaded = b.push_load::<false>(eax, 4, reg_space).id(); // %r = EAX
        b.push_store(loaded, other, reg_space); // use %r (keeps it live)
        load_id = loaded;
        unsafe { b.dont_finalize() };
    }

    let aliases = AliasResult::simple(&tc.ctx);
    gvn_function(&mut tc.ctx, fun_id, Some(&aliases));

    let ValueId::Instruction(load_insn) = load_id else {
        panic!("push_load should produce an instruction value");
    };
    assert!(
        !BasicBlock::from_id(&tc.ctx, block_id)
            .instruction_ids()
            .contains(&load_insn),
        "register reload should be forwarded to the stored value, got:\n{}",
        BasicBlock::from_id(&tc.ctx, block_id)
    );
}

/// A `call` is a block terminator with no CFG edge to its fall-through, so
/// the post-call block is unreachable from the function root. `gvn_function`
/// must still optimize it — otherwise the register reload chains the lifter
/// emits after every call survive untouched.
#[test]
fn test_register_forwarding_in_orphaned_post_call_block() {
    use qcode::{builder::Builder, testing::TestContext, value::Function};

    let mut tc = TestContext::new();
    let fun_id = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
    let entry = tc.ctx.get_or_make_block(0x1000);
    let post_call = tc.ctx.get_or_make_block(0x2000);
    {
        let mut f = Function::from_id_mut(&mut tc.ctx, fun_id);
        f.set_root(entry).unwrap();
        f.add_block(entry);
        f.add_block(post_call);
    }

    let reg_space = tc.reg_space;
    let eax = ValueId::Varnode(tc.r0_lo32);
    let other = ValueId::Varnode(tc.r1);

    // Entry ends in a `call`; no edge links it to `post_call`.
    {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
        b.push_call(fun_id);
        unsafe { b.dont_finalize() };
    }

    // Orphaned fall-through: store a register and read it straight back.
    let load_id;
    {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, post_call));
        let c = b.context_mut().get_const(0x42, 4).id();
        b.push_store(c, eax, reg_space);
        let loaded = b.push_load::<false>(eax, 4, reg_space).id();
        b.push_store(loaded, other, reg_space);
        load_id = loaded;
        unsafe { b.dont_finalize() };
    }

    let aliases = AliasResult::simple(&tc.ctx);
    gvn_function(&mut tc.ctx, fun_id, Some(&aliases));

    let ValueId::Instruction(load_insn) = load_id else {
        panic!("push_load should produce an instruction value");
    };
    assert!(
        !BasicBlock::from_id(&tc.ctx, post_call)
            .instruction_ids()
            .contains(&load_insn),
        "forwarding must reach the orphaned post-call block, got:\n{}",
        BasicBlock::from_id(&tc.ctx, post_call)
    );
}
