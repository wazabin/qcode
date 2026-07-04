use qcode::{
    builder::Builder,
    context::Context,
    space::{Space, SpaceId, SpaceType},
    testing::TestContext,
    value::{BasicBlock, BlockId, Varnode},
};
use qcode_macro::qcode;

use crate::gvn::gvn;

use super::{AliasResult, NodeId};

fn make_space(ctx: &mut Context<'static>, name: &'static str) -> SpaceId {
    let mut space = Space::new(Some(name), 1, 8);
    space.ty = SpaceType::Register;
    ctx.add_space(space)
}

fn build_in_custom_space(
    f: impl FnOnce(&mut Builder<'static, '_>, SpaceId),
) -> (Context<'static>, BlockId, SpaceId) {
    let mut ctx = Context::new();
    let space = make_space(&mut ctx, "register");
    let block_id = ctx.get_or_make_block(0x1000);
    let mut builder = Builder::from_context(&mut ctx, 0x1000);
    f(&mut builder, space);
    unsafe { builder.dont_finalize() };
    drop(builder);
    (ctx, block_id, space)
}

#[test]
fn separate_varnodes_do_not_alias() {
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
        varnode i32 A;
        varnode i32 B;

        <block>
            %a = load(i32, &A);
            %b = load(i32, &B);
            return [0];
    "
    );

    let result = AliasResult::simple(&ctx);
    assert!(
        !result.may_alias(&ctx, A.into(), B.into()),
        "distinct non-overlapping varnodes in the same space must not alias"
    );
}

#[test]
fn complex_operations_in_same_space_become_may_alias() {
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
        varnode i32 A;
        varnode i32 B;

        <block>
            %ptr = &A + i32 4;
            %a = load(i32, %ptr);
            %b = load(i32, &A);
            return [0];
    "
    );

    let result = AliasResult::simple(&ctx);
    assert!(
        result.may_alias(&ctx, ptr.into(), A.into()),
        "IR-derived pointer expressions should conservatively become may-alias"
    );
}

#[test]
fn complex_operations_in_other_space_do_not_alias() {
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
        varnode i32 A;
        varnode i32 B;

        <block>
            %ptr = &A + i32 4;
            %a = load(i32, %ptr);
            %b = load(i32, &A);
            return [0];
    "
    );

    let result = AliasResult::simple(&ctx);
    assert!(
        !result.may_alias(&ctx, ptr.into(), B.into()),
        "IR-derived pointer expressions in other spaces should not become may-alias"
    );
}

#[test]
fn overlapping_registers_alias() {
    let test_ctx = TestContext::new();

    let r0 = test_ctx.r0;
    let r0_lo32 = test_ctx.r0_lo32;

    let ctx = test_ctx.ctx;

    let result = AliasResult::simple(&ctx);
    assert!(
        result.may_alias(&ctx, r0.into(), r0_lo32.into()),
        "overlapping registers in the same space must alias"
    );
}

#[test]
fn irrelevant_instructions_and_constants_do_not_appear() {
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
        <block>
            %sum = i32 1 + i32 2;
            return [0];
    "
    );

    let one = ctx.get_const(1, 4).id();

    let result = AliasResult::simple(&ctx);

    assert_eq!(
        result.alias_class(one),
        None,
        "plain arithmetic constants should not appear in alias analysis"
    );
    assert_eq!(
        result.alias_class(sum.into()),
        None,
        "non-pointer arithmetic instructions should not appear in alias analysis"
    );
    assert_eq!(
        result.alias_class(block.into()),
        None,
        "basic block should not appear in alias analysis"
    );
}

#[test]
fn pointer_literals_are_tracked() {
    let test_ctx = TestContext::new();
    let reg_space = test_ctx.reg_space;
    let r0 = test_ctx.r0;
    let mut ctx = test_ctx.ctx;

    let _block_id = ctx.get_or_make_block(0x1000);
    let mut builder = Builder::from_context(&mut ctx, 0x1000);
    let literal_ptr = builder.context_mut().get_const(0, 8).id();
    builder.push_load::<false>(literal_ptr, 8, reg_space);
    unsafe { builder.dont_finalize() };
    drop(builder);

    let result = AliasResult::simple(&ctx);
    assert!(
        result.alias_class(literal_ptr).is_some(),
        "pointer literals used for memory accesses should appear in alias analysis"
    );
    assert!(
        result.may_alias(&ctx, literal_ptr, r0.into()),
        "literal register address should alias the overlapping register varnode"
    );
}

#[test]
fn literal_straddles_two_disjoint_varnode_classes_joins_them() {
    let mut a = None;
    let mut b = None;
    let mut literal_ptr = None;

    let (ctx, _, _) = build_in_custom_space(|builder, space| {
        let a_id = Varnode::make(builder.context_mut(), 0, 4, space).id;
        let b_id = Varnode::make(builder.context_mut(), 8, 4, space).id;
        let ptr = builder.context_mut().get_const(2, 8).id();

        builder.push_load::<false>(ptr, 8, space);

        a = Some(a_id);
        b = Some(b_id);
        literal_ptr = Some(ptr);
    });

    let a = a.unwrap();
    let b = b.unwrap();
    let literal_ptr = literal_ptr.unwrap();
    let result = AliasResult::simple(&ctx);

    assert!(result.may_alias(&ctx, literal_ptr, a.into()));
    assert!(result.may_alias(&ctx, literal_ptr, b.into()));
    assert!(result.may_alias(&ctx, a.into(), b.into()));
}

#[test]
fn two_literal_pointers_same_addr_alias_without_varnode() {
    let mut lit1 = None;
    let mut lit2 = None;

    let (ctx, _, _) = build_in_custom_space(|builder, space| {
        let first = builder.context_mut().get_const(0x10, 8).id();
        let second = builder
            .context_mut()
            .get_const(0xdead_beef_0000_0010, 4)
            .id();

        builder.push_load::<false>(first, 4, space);
        builder.push_load::<false>(second, 4, space);

        lit1 = Some(first);
        lit2 = Some(second);
    });

    let result = AliasResult::simple(&ctx);
    assert!(result.may_alias(&ctx, lit1.unwrap(), lit2.unwrap()));
}

#[test]
fn two_literal_pointers_overlapping_ranges_alias() {
    let mut lit1 = None;
    let mut lit2 = None;

    let (ctx, _, _) = build_in_custom_space(|builder, space| {
        let first = builder.context_mut().get_const(0x100, 8).id();
        let second = builder.context_mut().get_const(0x104, 8).id();

        builder.push_load::<false>(first, 8, space);
        builder.push_load::<false>(second, 4, space);

        lit1 = Some(first);
        lit2 = Some(second);
    });

    let result = AliasResult::simple(&ctx);
    assert!(result.may_alias(&ctx, lit1.unwrap(), lit2.unwrap()));
}

#[test]
fn two_literal_pointers_different_spaces_do_not_alias() {
    let mut ctx = Context::new();
    let reg_space = make_space(&mut ctx, "register");
    let alt_space = make_space(&mut ctx, "other");
    let _block_id = ctx.get_or_make_block(0x1000);
    let mut builder = Builder::from_context(&mut ctx, 0x1000);

    let lit1 = builder.context_mut().get_const(0x20, 8).id();
    let lit2 = builder
        .context_mut()
        .get_const(0xfeed_face_0000_0020, 4)
        .id();
    builder.push_load::<false>(lit1, 4, reg_space);
    builder.push_load::<false>(lit2, 4, alt_space);

    unsafe { builder.dont_finalize() };
    drop(builder);

    let result = AliasResult::simple(&ctx);
    assert!(!result.may_alias(&ctx, lit1, lit2));
}

#[test]
fn same_pointer_used_in_multiple_spaces_degrades_to_unknown() {
    let mut ctx = Context::new();
    let reg_space = make_space(&mut ctx, "register");
    let alt_space = make_space(&mut ctx, "other");
    let _block_id = ctx.get_or_make_block(0x1000);
    let mut builder = Builder::from_context(&mut ctx, 0x1000);

    let ptr = builder.context_mut().get_const(0x20, 8).id();
    builder.push_load::<false>(ptr, 4, reg_space);
    builder.push_load::<false>(ptr, 4, alt_space);

    unsafe { builder.dont_finalize() };
    drop(builder);

    // A literal interned across two spaces must not panic; the pointer degrades
    // to Unknown (may-alias everything) rather than killing the process.
    let result = AliasResult::simple(&ctx);
    assert_eq!(result.alias_class(ptr), Some(NodeId::Unknown));
}

#[test]
fn odd_pointer_arithmetic_degrades_to_unknown() {
    // r = sub(a, b) with both operands in the access space, then load(r) in that
    // same space: hits the "odd pointer arithmetic" branch, which must degrade to
    // Unknown rather than panic.
    let mut vn_a = None;
    let mut vn_b = None;
    let mut r = None;

    let (ctx, _, _) = build_in_custom_space(|builder, space| {
        let a_id = Varnode::make(builder.context_mut(), 0, 8, space).id;
        let b_id = Varnode::make(builder.context_mut(), 8, 8, space).id;
        let sub = builder
            .push_sub(a_id.into(), b_id.into())
            .id()
            .as_instruction()
            .expect("push_sub yields an instruction");
        builder.push_load::<false>(sub.into(), 8, space);

        vn_a = Some(a_id);
        vn_b = Some(b_id);
        r = Some(sub);
    });

    let vn_a = vn_a.unwrap();
    let r = r.unwrap();
    let _ = vn_b;
    let result = AliasResult::simple(&ctx);

    assert_eq!(result.alias_class(r.into()), Some(NodeId::Unknown));
    assert!(
        result.may_alias(&ctx, r.into(), vn_a.into()),
        "an Unknown pointer conservatively may-aliases the operand varnode"
    );
}

#[test]
fn unresolvable_load_ptr_becomes_unknown_and_aliases_everything() {
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
        <block>
            %ptr1 = i64 0x10 + i64 0x2;
            %ptr2 = i64 0x10 + i64 0x1;
            %v1 = load(i64, %ptr1);
            %v2 = load(i64, %ptr2);
            return [0];
    "
    );

    let result = AliasResult::simple(&ctx);
    assert_eq!(result.alias_class(ptr1.into()), Some(NodeId::Unknown));
    assert!(result.may_alias(&ctx, ptr1.into(), ptr2.into()));
}

#[test]
fn unresolvable_load_ptr_in_register_space_does_not_alias_registers() {
    let mut test_ctx = TestContext::new();
    let r0 = test_ctx.r0;

    qcode!(
        test_ctx.ctx,
        "
        <block>
            %lhs = load(i64, {r0});
            %rhs = load(i64, {r0});
            %ptr = %lhs + %rhs;
            %value = load(i64, %ptr);
            return [%value];
    "
    );

    let result = AliasResult::simple(&test_ctx.ctx);
    let alias_class = result.alias_class(ptr.into());
    assert!(matches!(alias_class, Some(NodeId::Unknown)));
    assert!(
        !result.may_alias(&test_ctx.ctx, ptr.into(), r0.into()),
        "register-space built pointers must not alias register varnodes"
    );
}

#[test]
fn store_then_load_invalidation_is_conservative_for_unknown_ptr() {
    let mut ctx = Context::new();

    qcode!(
        ctx,
        "
        varnode i64 A;
        varnode i64 B;

        <block>
            %before = load(i64, &A);
            %base = load(i64, &B);
            %ptr = &A + %base;
            store(%ptr, i64 0x7);
            %after = load(i64, &A);
            return [%after];
    "
    );

    let aliases = AliasResult::simple(&ctx);
    assert!(aliases.may_alias(&ctx, A.into(), ptr.into()));

    let mut block = BasicBlock::from_id_mut(&mut ctx, block);
    assert!(block.instruction_ids().contains(&after));

    gvn(&mut block, Some(&aliases));

    assert!(
        block.instruction_ids().contains(&after),
        "unknown store pointers must conservatively invalidate cached loads"
    );
}

#[test]
fn literal_with_high_bit_set_aliases_overlapping_literals() {
    let mut lit1 = None;
    let mut lit2 = None;

    let (ctx, _, _) = build_in_custom_space(|builder, space| {
        let first = builder
            .context_mut()
            .get_const(0x8000_0000_0000_0000, 8)
            .id();
        let second = builder
            .context_mut()
            .get_const(0x8000_0000_0000_0004, 8)
            .id();

        builder.push_load::<false>(first, 8, space);
        builder.push_load::<false>(second, 4, space);

        lit1 = Some(first);
        lit2 = Some(second);
    });

    let result = AliasResult::simple(&ctx);
    assert!(result.may_alias(&ctx, lit1.unwrap(), lit2.unwrap()));
}

#[test]
fn literal_with_upper_junk_bits_is_masked_to_size() {
    let mut a = None;
    let mut literal_ptr = None;

    let (ctx, _, _) = build_in_custom_space(|builder, space| {
        let a_id = Varnode::make(builder.context_mut(), 0x10, 4, space).id;
        let ptr = builder
            .context_mut()
            .get_const(0xdead_beef_0000_0010, 4)
            .id();

        builder.push_load::<false>(ptr, 4, space);

        a = Some(a_id);
        literal_ptr = Some(ptr);
    });

    let result = AliasResult::simple(&ctx);
    assert!(result.may_alias(&ctx, literal_ptr.unwrap(), a.unwrap().into()));
}

#[test]
fn untracked_value_may_alias_conservatively() {
    let mut ctx = Context::new();
    let space = make_space(&mut ctx, "register");
    let _block_id = ctx.get_or_make_block(0x1000);
    let mut builder = Builder::from_context(&mut ctx, 0x1000);

    // Two tracked, non-overlapping literal pointers (positive control).
    let p = builder.context_mut().get_const(0x1000, 8).id();
    let p2 = builder.context_mut().get_const(0x2000, 8).id();
    builder.push_load::<false>(p, 4, space);
    builder.push_load::<false>(p2, 4, space);
    unsafe { builder.dont_finalize() };
    drop(builder);

    let result = AliasResult::simple(&ctx);

    // Append an instruction the analysis never saw.
    let mut builder = Builder::from_context(&mut ctx, 0x1000);
    let eight = builder.context_mut().get_const(8, 8).id();
    let q = builder.push_add(p, eight).id();
    unsafe { builder.dont_finalize() };
    drop(builder);

    assert!(
        result.alias_class(q).is_none(),
        "precondition: q is genuinely untracked"
    );
    assert!(
        result.may_alias(&ctx, q, p),
        "an untracked value must answer may-alias, not no-alias"
    );
    assert!(
        !result.may_alias(&ctx, p, p2),
        "tracked non-overlapping literals still answer no-alias"
    );
}

#[test]
fn many_overlapping_subregisters_still_join_in_one_class() {
    let mut varnodes = Vec::new();

    let (ctx, _, _) = build_in_custom_space(|builder, space| {
        for size in (1..=32).rev() {
            let id = Varnode::make(builder.context_mut(), 0, size, space).id;
            varnodes.push(id);
        }
    });

    let result = AliasResult::simple(&ctx);
    let first = varnodes[0];

    for &varnode in &varnodes[1..] {
        assert!(result.may_alias(&ctx, first.into(), varnode.into()));
    }
}
