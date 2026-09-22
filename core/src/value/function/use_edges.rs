//! The use edges of a body: one per operand occurrence, threaded into a
//! list per used value. These exercise the verbs that keep the edges in
//! step with the operands — creation, deletion (one at a time and in bulk),
//! operand and mnemonic rewrites — and check the edges against the
//! operands with the arena integrity pass after each.

use super::*;
use crate::arena_integrity::verify_body_arena_integrity;
use crate::value::QCodeMut;
use wazabin_qcode_macro::qcode;

fn assert_clean(ctx: &Context<'_>) {
    assert_eq!(verify_body_arena_integrity(ctx), Vec::<String>::new());
}

/// The users of `value` in `f`, sorted, so tests can compare sets without
/// depending on the (unspecified) list order.
fn users(ctx: &Context<'_>, f: FunctionId, value: impl Into<ValueId>) -> Vec<InstructionId> {
    let mut users = ctx.bodies[f].users_of(value.into());
    users.sort();
    users
}

#[test]
fn creating_and_removing_instructions_maintains_the_edges() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
        fn f:
            <entry>
                %a = i64 1 + i64 2;
                %b = %a + i64 3;
                %c = %a + %b;
                return at %c;
        "
    );
    assert_clean(&ctx);
    assert_eq!(users(&ctx, f, a), vec![b, c]);
    assert_eq!(users(&ctx, f, b), vec![c]);
    assert!(ctx.bodies[f].has_users(ValueId::Instruction(c)));

    // Literals are shared values: this body tracks its own uses of them.
    let three = ctx.get_const(3, 8).id();
    assert_eq!(users(&ctx, f, three), vec![b]);

    ctx.remove_instruction(b);
    assert_eq!(users(&ctx, f, a), vec![c], "b's use of a came down");
    assert!(
        !ctx.bodies[f].has_users(three),
        "b's use of the literal came down"
    );
    // `c` still names `b`: a dangling operand, allowed while a
    // transformation is in progress and reported until it is resolved.
    assert!(!ctx.contains_instruction(b));
    assert_eq!(
        verify_body_arena_integrity(&ctx),
        vec![format!(
            "instruction {c:?} references removed local value {:?}",
            ValueId::Instruction(b)
        )]
    );
    let ret = BasicBlock::from_id(&ctx, entry).last_instruction().unwrap();
    ctx.remove_instruction(ret);
    ctx.remove_instruction(c);
    assert_clean(&ctx);
    assert!(!ctx.bodies[f].has_users(ValueId::Instruction(a)));
}

#[test]
fn bulk_removal_walks_each_shared_operand_once() {
    let mut ctx = Context::new();
    let f = ctx.anon_function();
    let block = ctx.get_or_make_block(0x1000, f);
    let shared = ctx.get_const(7, 8).id();
    let ids: Vec<InstructionId> = {
        let mut b = ctx.builder(block);
        (0..8).map(|_| b.push_bit_negate(shared).id).collect()
    };
    let keep = ids[3];
    let dead: FxHashSet<LocalInsnId> = ids
        .iter()
        .filter(|&&id| id != keep)
        .map(|id| id.local)
        .collect();

    ctx.bodies[f].remove_block_instructions(block, &dead);
    assert_clean(&ctx);
    assert_eq!(users(&ctx, f, shared), vec![keep]);
    assert_eq!(ctx.bodies[f].uses.len(), 1);

    ctx.bodies[f].clear_block_instructions(block);
    assert_clean(&ctx);
    assert!(!ctx.bodies[f].has_users(shared));
    assert_eq!(ctx.bodies[f].uses.len(), 0);
    assert!(
        ctx.bodies[f].shared_first_use.is_empty(),
        "a shared value with no use has no head entry"
    );
}

#[test]
fn replacing_all_uses_moves_every_edge() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
        fn f:
            <entry>
                %a = i64 1 + i64 2;
                %b = %a + i64 3;
                %c = %a + %a;
                return at %c;
        "
    );
    let ten = ctx.get_const(10, 8).id();
    ctx.replace_all_uses_with(ValueId::Instruction(a), ten);
    assert_clean(&ctx);
    assert!(!ctx.bodies[f].has_users(ValueId::Instruction(a)));
    assert_eq!(users(&ctx, f, ten), vec![b, c, c], "one per occurrence");
    assert_eq!(
        ctx.instruction(c).mnemonic().args().to_vec(),
        vec![ten.strip_func(), ten.strip_func()]
    );

    // Back onto a local value — a shared value's uses are this body's to
    // replace — then the instruction itself goes.
    ctx.bodies[f].replace_all_uses_with(ten, ValueId::Instruction(a));
    assert_clean(&ctx);
    assert_eq!(users(&ctx, f, a), vec![b, c, c]);
    assert!(!ctx.bodies[f].has_users(ten));
    ctx.replace_instruction(a, ten);
    assert_clean(&ctx);
    assert!(!ctx.contains_instruction(a));
    assert_eq!(users(&ctx, f, ten), vec![b, c, c]);
}

#[test]
fn selective_replacement_picks_edges_by_user_and_operand() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
        fn f:
            <entry>
                %a = i64 1 + i64 2;
                %here = %a + i64 3;
                goto <other>;
            <other>
                %there = %a + %a;
                return at %there;
        "
    );
    let ten = ctx.get_const(10, 8).id();

    // Only the uses in `<other>`.
    ctx.bodies[f].replace_uses_where(ValueId::Instruction(a), ten, |body, user, _| {
        body.insn(user).parent.get() == Some(other.local)
    });
    assert_clean(&ctx);
    assert_eq!(users(&ctx, f, a), vec![here]);
    assert_eq!(users(&ctx, f, ten), vec![there, there]);

    // Only the second of two occurrences: `%there = 10 + 10` → `%there = 10 + %a`.
    ctx.bodies[f].replace_uses_where(ten, ValueId::Instruction(a), |_, user, index| {
        user == there && index == 1
    });
    assert_clean(&ctx);
    assert_eq!(
        ctx.instruction(there).mnemonic().args().to_vec(),
        vec![ten.strip_func(), ValueId::Instruction(a).strip_func()]
    );
    assert_eq!(users(&ctx, f, a), vec![here, there]);
    assert_eq!(users(&ctx, f, ten), vec![there]);

    // One operand, by position.
    ctx.bodies[f].replace_operand(there, 0, ValueId::Instruction(a));
    assert_clean(&ctx);
    assert_eq!(users(&ctx, f, a), vec![here, there, there]);
    assert!(!ctx.bodies[f].has_users(ten));
}

#[test]
fn swapping_a_mnemonic_swaps_its_edges() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
        fn f:
            <entry>
                %a = i64 1 + i64 2;
                %b = i64 3 + i64 4;
                %c = %a + i64 5;
                return at %c;
        "
    );
    let five = ctx.get_const(5, 8).id();
    let replacement = Mnemonic::Binop(crate::value::insn::Binary {
        op: crate::value::insn::Binop::Int(crate::value::insn::IntBinop::Add),
        lhs: ValueId::Instruction(b).strip_func(),
        rhs: ValueId::Instruction(b).strip_func(),
    });
    ctx.replace_instruction_mnemonic(c, replacement);
    assert_clean(&ctx);
    assert!(!ctx.bodies[f].has_users(ValueId::Instruction(a)));
    assert!(!ctx.bodies[f].has_users(five));
    assert_eq!(users(&ctx, f, b), vec![c, c]);
}

#[test]
fn block_params_and_temps_carry_their_own_heads() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
        fn f:
            <entry @x:i64>
                %y = i64 @x + i64 1;
                goto <exit @p=%y>;
            <exit @p:i64>
                return at i64 @p;
        "
    );
    assert_clean(&ctx);
    assert_eq!(users(&ctx, f, ValueId::BlockParam(x)), vec![y]);
    let branch = BasicBlock::from_id(&ctx, entry).last_instruction().unwrap();
    let ret = BasicBlock::from_id(&ctx, exit).last_instruction().unwrap();
    assert_eq!(users(&ctx, f, ValueId::Instruction(y)), vec![branch]);
    assert_eq!(users(&ctx, f, ValueId::BlockParam(p)), vec![ret]);

    let space = ctx.bodies[f].push_temp_space(TempSpace::new(None, 1, 8));
    let temp = ctx.bodies[f].push_temp(Temp::new(0, 8, space.local));
    let load = {
        let mut b = ctx.builder(entry);
        b.set_insert_point_before(y);
        b.push_load::<false>(
            ValueId::Temp(temp),
            8,
            crate::space::LocalMemorySpaceId::Temp(space.local),
        )
        .id()
    };
    let ValueId::Instruction(load) = load else {
        panic!("a load is an instruction");
    };
    assert_clean(&ctx);
    assert_eq!(users(&ctx, f, ValueId::Temp(temp)), vec![load]);

    ctx.remove_instruction(ret);
    assert_clean(&ctx);
    assert!(!ctx.bodies[f].has_users(ValueId::BlockParam(p)));
}

#[test]
fn shared_values_are_tracked_per_body() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
        fn first:
            <first_entry>
                %a = i64 7 + i64 1;
                return at %a;

        fn second:
            <entry>
                %b = i64 7 + i64 2;
                return at %b;
        "
    );
    let seven = ctx.get_const(7, 8).id();
    assert_eq!(users(&ctx, first, seven), vec![a]);
    assert_eq!(users(&ctx, second, seven), vec![b]);
    let mut all = ctx.users_across_functions(seven);
    all.sort();
    assert_eq!(all, vec![a, b]);

    let ret = BasicBlock::from_id(&ctx, first_entry)
        .last_instruction()
        .unwrap();
    ctx.remove_instruction(ret);
    ctx.remove_instruction(a);
    assert_clean(&ctx);
    assert!(!ctx.bodies[first].has_users(seven));
    assert_eq!(
        users(&ctx, second, seven),
        vec![b],
        "the other body is untouched"
    );
}

#[test]
fn cloning_keeps_the_edges_and_serializing_rebuilds_them() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
        fn f:
            <entry @x:i64>
                %a = i64 @x + i64 2;
                %b = %a + %a;
                %c = %b + i64 3;
                return at %c;
        "
    );
    let ret = BasicBlock::from_id(&ctx, entry).last_instruction().unwrap();
    ctx.remove_instruction(ret);
    ctx.remove_instruction(c);
    ctx.replace_all_uses_with(ValueId::Instruction(a), ValueId::BlockParam(x));
    assert_clean(&ctx);

    let cloned = ctx.clone();
    assert_clean(&cloned);
    assert_eq!(
        users(&cloned, f, ValueId::BlockParam(x)),
        users(&ctx, f, ValueId::BlockParam(x))
    );

    let config = bincode::config::standard();
    let bytes = bincode::serde::encode_to_vec(&ctx, config).expect("encode");
    let (restored, _): (Context<'static>, usize) =
        bincode::serde::decode_from_slice(&bytes, config).expect("decode");
    assert_clean(&restored);
    assert_eq!(
        users(&restored, f, ValueId::BlockParam(x)),
        vec![a, b, b],
        "rebuilt from the operands"
    );
    assert!(!restored.bodies[f].has_users(ValueId::Instruction(a)));
    assert_eq!(restored.bodies[f].uses.len(), ctx.bodies[f].uses.len());
}

#[test]
fn a_removed_use_slot_is_reused() {
    let mut ctx = Context::new();
    let f = ctx.anon_function();
    let block = ctx.get_or_make_block(0x1000, f);
    let c = ctx.get_const(1, 8).id();
    let ids: Vec<InstructionId> = {
        let mut b = ctx.builder(block);
        (0..4).map(|_| b.push_bit_negate(c).id).collect()
    };
    let before = ctx.bodies[f].uses.len();
    ctx.remove_instruction(ids[1]);
    ctx.remove_instruction(ids[2]);
    let fresh = ctx.builder(block).push_bit_negate(c).id;
    let again = ctx.builder(block).push_bit_negate(c).id;
    assert_clean(&ctx);
    assert_eq!(ctx.bodies[f].uses.len(), before);
    assert_eq!(users(&ctx, f, c), vec![ids[0], ids[3], fresh, again]);
}

/// The highest use slot in use, plus one: how far the slab has grown.
fn issued(ctx: &Context<'_>, f: FunctionId) -> usize {
    ctx.bodies[f]
        .use_edges()
        .map(|edge| usize::from(edge.id) + 1)
        .max()
        .unwrap_or(0)
}

#[test]
fn churning_a_block_does_not_grow_the_slab_past_its_peak() {
    let mut ctx = Context::new();
    let f = ctx.anon_function();
    let block = ctx.get_or_make_block(0x1000, f);
    let c = ctx.get_const(1, 8).id();
    let mut live: Vec<InstructionId> = {
        let mut b = ctx.builder(block);
        (0..16).map(|_| b.push_bit_negate(c).id).collect()
    };
    let peak = issued(&ctx, f);
    assert_eq!(peak, 16);

    for round in 0..50 {
        // Delete a mix of old and new instructions, then mint as many again.
        for _ in 0..8 {
            let victim = live.remove((round * 3) % live.len());
            ctx.remove_instruction(victim);
        }
        for _ in 0..8 {
            live.push(ctx.builder(block).push_bit_negate(c).id);
        }
        assert_clean(&ctx);
        assert_eq!(ctx.bodies[f].uses.len(), 16);
        assert_eq!(issued(&ctx, f), peak, "round {round} grew the slab");
    }
    let mut expected = live.clone();
    expected.sort();
    assert_eq!(users(&ctx, f, c), expected);
}

#[test]
fn use_edges_iterate_over_the_holes_left_by_deletion() {
    let mut ctx = Context::new();
    let f = ctx.anon_function();
    let block = ctx.get_or_make_block(0x1000, f);
    let c = ctx.get_const(1, 8).id();
    let ids: Vec<InstructionId> = {
        let mut b = ctx.builder(block);
        (0..5).map(|_| b.push_bit_negate(c).id).collect()
    };
    ctx.remove_instruction(ids[1]);
    ctx.remove_instruction(ids[3]);
    assert_clean(&ctx);

    let edges: Vec<(usize, InstructionId)> = ctx.bodies[f]
        .use_edges()
        .map(|edge| (usize::from(edge.id), InstructionId::new(f, edge.user)))
        .collect();
    assert_eq!(edges, vec![(0, ids[0]), (2, ids[2]), (4, ids[4])]);
    assert_eq!(ctx.bodies[f].uses.len(), 3);
    assert!(!ctx.bodies[f].uses.contains(UseId::from(1)));
    assert!(!ctx.bodies[f].uses.contains(UseId::from(3)));
}

#[test]
fn rebuilding_forgets_the_holes_and_restarts_the_slab() {
    let mut ctx = Context::new();
    let f = ctx.anon_function();
    let block = ctx.get_or_make_block(0x1000, f);
    let c = ctx.get_const(1, 8).id();
    let ids: Vec<InstructionId> = {
        let mut b = ctx.builder(block);
        (0..6).map(|_| b.push_bit_negate(c).id).collect()
    };
    for &id in &ids[..4] {
        ctx.remove_instruction(id);
    }
    assert_eq!(issued(&ctx, f), 6, "the freed slots are still issued");
    let before = users(&ctx, f, c);

    ctx.bodies[f].rebuild_uses();
    assert_clean(&ctx);
    assert_eq!(users(&ctx, f, c), before);
    assert_eq!(issued(&ctx, f), 2, "the rebuilt slab is dense");
    let slots: Vec<usize> = ctx.bodies[f]
        .use_edges()
        .map(|edge| usize::from(edge.id))
        .collect();
    assert_eq!(slots, vec![0, 1]);

    // The next edge takes the slot after the rebuilt ones, not one of the
    // ids that were free before the rebuild.
    let fresh = ctx.builder(block).push_bit_negate(c).id;
    assert_clean(&ctx);
    assert_eq!(issued(&ctx, f), 3);
    assert_eq!(users(&ctx, f, c), vec![ids[4], ids[5], fresh]);
}

#[test]
fn a_freed_slot_is_never_reachable_from_a_use_list() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
        fn f:
            <entry @x:i64>
                %a = i64 @x + i64 2;
                %b = %a + @x;
                %c = %b + @x;
                return at %c;
        "
    );
    assert_eq!(users(&ctx, f, ValueId::BlockParam(x)), vec![a, b, c]);
    let ret = BasicBlock::from_id(&ctx, entry).last_instruction().unwrap();
    ctx.remove_instruction(ret);
    ctx.remove_instruction(c);
    // `c`'s two edges are free; the lists of `x` and `b` must have been
    // unlinked from them, and the next edges take those slots over.
    assert_eq!(users(&ctx, f, ValueId::BlockParam(x)), vec![a, b]);
    assert!(!ctx.bodies[f].has_users(ValueId::Instruction(b)));
    assert_clean(&ctx);

    let d = ctx
        .builder(entry)
        .push_binop(
            crate::value::insn::Binop::Int(crate::value::insn::IntBinop::Add),
            ValueId::Instruction(b),
            ValueId::BlockParam(x),
        )
        .id;
    assert_clean(&ctx);
    assert_eq!(users(&ctx, f, ValueId::BlockParam(x)), vec![a, b, d]);
    assert_eq!(users(&ctx, f, ValueId::Instruction(b)), vec![d]);
    assert_eq!(
        ctx.bodies[f].use_edges().count(),
        ctx.bodies[f].uses.len(),
        "every slot is either live or on the free list"
    );
}
