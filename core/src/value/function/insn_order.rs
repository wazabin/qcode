//! The instruction order of a block: a doubly linked list threaded through
//! the instructions, the block holding its ends and length. These exercise
//! the four link verbs and every editing verb built on them, and check the
//! links with the arena integrity pass after each.

use super::*;
use crate::arena_integrity::verify_body_arena_integrity;
use wazabin_qcode_macro::qcode;

/// A function with one block, and the block.
fn fixture() -> (Context<'static>, FunctionId, BlockId) {
    let mut ctx = Context::new();
    let f = ctx.anon_function();
    let block = ctx.get_or_make_block(0x1000, f);
    (ctx, f, block)
}

/// A fresh instruction of `f`, in no block: `~<n>` for a constant `n` of
/// its own, so its operand's user list holds it alone.
fn mint(ctx: &mut Context<'static>, f: FunctionId, block: BlockId, n: u64) -> InstructionId {
    let c = ctx.get_const(n, 8).id();
    let mut b = ctx.builder(block);
    let id = b.push_bit_negate(c).id;
    ctx.bodies[f].unlink(id.local);
    id
}

fn order(ctx: &Context<'_>, block: BlockId) -> Vec<InstructionId> {
    ctx.bodies[block.func].block_insn_ids(block)
}

fn reverse_order(ctx: &Context<'_>, block: BlockId) -> Vec<InstructionId> {
    ctx.bodies[block.func]
        .insn_ids(block.local)
        .rev()
        .map(|local| InstructionId::new(block.func, local))
        .collect()
}

/// The block's list agrees with itself from both ends, with its count, and
/// with the integrity pass.
fn assert_consistent(ctx: &Context<'_>, block: BlockId, expected: &[InstructionId]) {
    assert_eq!(order(ctx, block), expected, "forward order");
    let mut reversed = expected.to_vec();
    reversed.reverse();
    assert_eq!(reverse_order(ctx, block), reversed, "reverse order");
    let b = ctx.block(block);
    assert_eq!(b.insn_count(), expected.len());
    assert_eq!(b.has_insns(), !expected.is_empty());
    assert_eq!(
        b.first_insn().map(|l| InstructionId::new(block.func, l)),
        expected.first().copied()
    );
    assert_eq!(
        b.last_insn().map(|l| InstructionId::new(block.func, l)),
        expected.last().copied()
    );
    let ids = ctx.bodies[block.func].insn_ids(block.local);
    assert_eq!(ids.len(), expected.len(), "exact size");
    for id in expected {
        assert_eq!(ctx.bodies[block.func].insn(*id).parent, Some(block.local));
    }
    assert_eq!(verify_body_arena_integrity(ctx), Vec::<String>::new());
}

#[test]
fn linking_at_the_ends_and_around_an_anchor() {
    let (mut ctx, f, block) = fixture();
    assert_consistent(&ctx, block, &[]);

    let a = mint(&mut ctx, f, block, 1);
    ctx.bodies[f].link_last(block.local, a.local);
    assert_consistent(&ctx, block, &[a]);

    // Before the only instruction, then after it: the ends move.
    let b = mint(&mut ctx, f, block, 2);
    ctx.bodies[f].link_before(block.local, a.local, b.local);
    assert_consistent(&ctx, block, &[b, a]);
    let c = mint(&mut ctx, f, block, 3);
    ctx.bodies[f].link_after(block.local, a.local, c.local);
    assert_consistent(&ctx, block, &[b, a, c]);

    // In the middle, both ways.
    let d = mint(&mut ctx, f, block, 4);
    ctx.bodies[f].link_before(block.local, a.local, d.local);
    let e = mint(&mut ctx, f, block, 5);
    ctx.bodies[f].link_after(block.local, a.local, e.local);
    assert_consistent(&ctx, block, &[b, d, a, e, c]);

    // By index, walking: at the start, in the middle, and at the end.
    let g = mint(&mut ctx, f, block, 6);
    ctx.bodies[f].insert_insn_at(block.local, 0, g.local);
    let h = mint(&mut ctx, f, block, 7);
    ctx.bodies[f].insert_insn_at(block.local, 3, h.local);
    let i = mint(&mut ctx, f, block, 8);
    ctx.bodies[f].insert_insn_at(block.local, 7, i.local);
    assert_consistent(&ctx, block, &[g, b, d, h, a, e, c, i]);
}

#[test]
#[should_panic(expected = "index past the end")]
fn linking_past_the_end_is_refused() {
    let (mut ctx, f, block) = fixture();
    let a = mint(&mut ctx, f, block, 1);
    ctx.bodies[f].insert_insn_at(block.local, 1, a.local);
}

#[test]
fn unlinking_the_first_middle_and_last() {
    let (mut ctx, f, block) = fixture();
    let ids: Vec<InstructionId> = (0..5).map(|n| mint(&mut ctx, f, block, n)).collect();
    for id in &ids {
        ctx.bodies[f].link_last(block.local, id.local);
    }
    assert_consistent(&ctx, block, &ids);

    ctx.bodies[f].unlink(ids[0].local);
    assert_consistent(&ctx, block, &ids[1..]);
    ctx.bodies[f].unlink(ids[4].local);
    assert_consistent(&ctx, block, &ids[1..4]);
    ctx.bodies[f].unlink(ids[2].local);
    assert_consistent(&ctx, block, &[ids[1], ids[3]]);

    // An unlinked instruction is in no block and carries no links, and
    // unlinking it again is a no-op.
    let gone = &ctx.bodies[f].insns[ids[2].local];
    assert_eq!(gone.parent, None);
    assert_eq!((gone.prev_in_block(), gone.next_in_block()), (None, None));
    ctx.bodies[f].unlink(ids[2].local);
    assert_consistent(&ctx, block, &[ids[1], ids[3]]);

    // Down to one, then none.
    ctx.bodies[f].unlink(ids[1].local);
    assert_consistent(&ctx, block, &[ids[3]]);
    ctx.bodies[f].unlink(ids[3].local);
    assert_consistent(&ctx, block, &[]);

    // Relinking after emptying starts a fresh list.
    ctx.bodies[f].link_last(block.local, ids[2].local);
    ctx.bodies[f].link_before(block.local, ids[2].local, ids[0].local);
    assert_consistent(&ctx, block, &[ids[0], ids[2]]);
}

#[test]
fn iteration_walks_both_ends_to_the_middle() {
    let (mut ctx, f, block) = fixture();
    let ids: Vec<InstructionId> = (0..4).map(|n| mint(&mut ctx, f, block, n)).collect();
    for id in &ids {
        ctx.bodies[f].link_last(block.local, id.local);
    }
    let mut it = ctx.bodies[f].insn_ids(block.local);
    assert_eq!(it.size_hint(), (4, Some(4)));
    assert_eq!(it.next(), Some(ids[0].local));
    assert_eq!(it.next_back(), Some(ids[3].local));
    assert_eq!(it.len(), 2);
    assert_eq!(it.next_back(), Some(ids[2].local));
    assert_eq!(it.next(), Some(ids[1].local));
    assert_eq!(it.next(), None);
    assert_eq!(it.next_back(), None);

    // The same through a block view.
    let view = BasicBlock::from_id(&ctx, block);
    let mut it = view.instructions();
    assert_eq!(it.len(), 4);
    assert_eq!(it.next_back().map(|i| i.id), Some(ids[3]));
    assert_eq!(it.next().map(|i| i.id), Some(ids[0]));
    assert_eq!(it.map(|i| i.id).collect::<Vec<_>>(), &ids[1..3]);
    assert_eq!(view.instruction_ids(), ids);
}

#[test]
fn removing_instructions_unlinks_them() {
    let (mut ctx, f, block) = fixture();
    let ids: Vec<InstructionId> = (0..6).map(|n| mint(&mut ctx, f, block, n)).collect();
    for id in &ids {
        ctx.bodies[f].link_last(block.local, id.local);
    }

    // One at a time: first, last, middle.
    ctx.bodies[f].remove_instruction(ids[0]);
    ctx.bodies[f].remove_instruction(ids[5]);
    ctx.bodies[f].remove_instruction(ids[3]);
    assert_consistent(&ctx, block, &[ids[1], ids[2], ids[4]]);
    assert!(!ctx.contains_instruction(ids[3]));

    // In bulk.
    let dead: FxHashSet<LocalInsnId> = [ids[1].local, ids[4].local].into_iter().collect();
    ctx.bodies[f].remove_block_instructions(block, &dead);
    assert_consistent(&ctx, block, &[ids[2]]);

    // Everything, keeping the block.
    ctx.bodies[f].clear_block_instructions(block);
    assert_consistent(&ctx, block, &[]);
    assert!(ctx.contains_block(block));
}

#[test]
fn block_views_edit_the_list() {
    let (mut ctx, f, block) = fixture();
    let ids: Vec<InstructionId> = (0..5).map(|n| mint(&mut ctx, f, block, n)).collect();

    BasicBlock::from_id_mut(&mut ctx, block).push_insn(ids[0]);
    BasicBlock::from_id_mut(&mut ctx, block).extend_insns(&ids[1..3]);
    assert_consistent(&ctx, block, &ids[..3]);

    BasicBlock::from_id_mut(&mut ctx, block).insert_insn_before(ids[0], ids[3]);
    BasicBlock::from_id_mut(&mut ctx, block).insert_insn_after(ids[2], ids[4]);
    assert_consistent(&ctx, block, &[ids[3], ids[0], ids[1], ids[2], ids[4]]);

    BasicBlock::from_id_mut(&mut ctx, block).pop_insn();
    assert_consistent(&ctx, block, &[ids[3], ids[0], ids[1], ids[2]]);

    // Retaining deletes the rest.
    BasicBlock::from_id_mut(&mut ctx, block).retain_insns(|id| *id == ids[0] || *id == ids[2]);
    assert_consistent(&ctx, block, &[ids[0], ids[2]]);
    assert!(!ctx.contains_instruction(ids[1]));
    assert!(!ctx.contains_instruction(ids[3]));

    let fresh = mint(&mut ctx, f, block, 5);
    BasicBlock::from_id_mut(&mut ctx, block).insert_insn_at_index(1, fresh);
    assert_consistent(&ctx, block, &[ids[0], fresh, ids[2]]);
}

#[test]
fn the_builder_inserts_before_its_point() {
    let (mut ctx, f, block) = fixture();
    let c = ctx.get_const(0, 8).id();
    let (a, b) = {
        let mut bld = ctx.builder(block);
        (bld.push_bit_negate(c).id, bld.push_bit_negate(c).id)
    };
    // Pushes before an anchor land in push order, before it.
    let (x, y) = {
        let mut bld = ctx.builder(block);
        bld.set_insert_point_before(b);
        (bld.push_bit_negate(c).id, bld.push_bit_negate(c).id)
    };
    assert_consistent(&ctx, block, &[a, x, y, b]);
    // At the start of a block, then back to the end.
    let (s, e) = {
        let mut bld = ctx.builder(block);
        bld.set_insert_point_to_start();
        let s = bld.push_bit_negate(c).id;
        bld.set_insert_point_to_end();
        (s, bld.push_bit_negate(c).id)
    };
    assert_consistent(&ctx, block, &[s, a, x, y, b, e]);
    let _ = f;
}

#[test]
fn moving_an_instruction_within_and_across_blocks() {
    let (mut ctx, f, block) = fixture();
    let other = ctx.get_or_make_block(0x2000, f);
    let ids: Vec<InstructionId> = (0..4).map(|n| mint(&mut ctx, f, block, n)).collect();
    for id in &ids {
        ctx.bodies[f].link_last(block.local, id.local);
    }
    let o = mint(&mut ctx, f, other, 9);
    ctx.bodies[f].link_last(other.local, o.local);

    // Backwards and forwards within one block.
    ctx.bodies[f].move_insn_before(ids[3], ids[1]);
    assert_consistent(&ctx, block, &[ids[0], ids[3], ids[1], ids[2]]);
    ctx.bodies[f].move_insn_before(ids[0], ids[2]);
    assert_consistent(&ctx, block, &[ids[3], ids[1], ids[0], ids[2]]);
    // Before itself: nothing happens.
    ctx.bodies[f].move_insn_before(ids[1], ids[1]);
    assert_consistent(&ctx, block, &[ids[3], ids[1], ids[0], ids[2]]);

    // Across blocks, from the ends and the middle.
    ctx.bodies[f].move_insn_before(ids[3], o);
    ctx.bodies[f].move_insn_before(ids[2], o);
    ctx.bodies[f].move_insn_before(ids[1], o);
    assert_consistent(&ctx, block, &[ids[0]]);
    assert_consistent(&ctx, other, &[ids[3], ids[2], ids[1], o]);
}

#[test]
fn splitting_moves_the_tail_and_absorbing_brings_it_back() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
        fn split_me:
            <entry>
                %a = i64 1 + i64 2;
                %b = %a + i64 3;
                %c = %b + i64 4;
                %d = %c + i64 5;
                return at %d;
        "
    );
    let f = split_me;
    let ids = order(&ctx, entry);
    assert_eq!(ids.len(), 5);

    // Split before %c: the head keeps [%a, %b], the tail gets the rest.
    let tail = ctx.split_block_before(entry, ids[2]);
    assert_consistent(&ctx, entry, &ids[..2]);
    assert_consistent(&ctx, tail, &ids[2..]);

    // Terminate the head with a branch to the tail, then absorb it back.
    let branch = ctx.builder(entry).push_branch(tail).id;
    assert_consistent(&ctx, entry, &[ids[0], ids[1], branch]);
    let edge = ctx.bodies[f]
        .block(entry)
        .edges
        .iter()
        .copied()
        .find(|&e| {
            let e = ctx.bodies[f].edge(e);
            e.from == entry.local && e.to == tail.local
        })
        .expect("the branch's edge");
    ctx.bodies[f].absorb_block(entry, tail, edge);
    assert!(!ctx.contains_block(tail));
    assert_consistent(&ctx, entry, &ids);

    // Splitting at the first instruction empties the block; at the last
    // moves only the terminator.
    let tail = ctx.split_block_before(entry, ids[0]);
    assert_consistent(&ctx, entry, &[]);
    assert_consistent(&ctx, tail, &ids);
    let end = ctx.split_block_before(tail, ids[4]);
    assert_consistent(&ctx, tail, &ids[..4]);
    assert_consistent(&ctx, end, &ids[4..]);
}

#[test]
fn a_cloned_block_keeps_the_order() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
        fn clone_me:
            <entry>
                %a = i64 1 + i64 2;
                %b = %a + i64 3;
                %c = %b + i64 4;
                return at %c;
        "
    );
    let target = ctx.anon_function();
    let mut value_map = FxHashMap::default();
    let cloned = BasicBlock::clone_block_into(&mut ctx, entry, target, &mut value_map);
    let opcodes = |ctx: &Context<'_>, block: BlockId| -> Vec<String> {
        BasicBlock::from_id(ctx, block)
            .instructions()
            .map(|i| i.opcode().to_string())
            .collect()
    };
    assert_eq!(opcodes(&ctx, cloned), opcodes(&ctx, entry));
    let expected = order(&ctx, cloned);
    assert_eq!(expected.len(), 4);
    assert_consistent(&ctx, cloned, &expected);
    let _ = clone_me;
}

#[test]
fn links_survive_a_bincode_round_trip() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
        fn round_trip:
            <entry>
                %a = i64 1 + i64 2;
                %b = %a + i64 3;
                %c = %b + i64 4;
                return at %c;
        "
    );
    // Edit the list first, so the links are not the order the instructions
    // were minted in.
    let ids = order(&ctx, entry);
    ctx.bodies[round_trip].move_insn_before(ids[1], ids[0]);
    let before = order(&ctx, entry);
    assert_eq!(before, [ids[1], ids[0], ids[2], ids[3]]);

    let config = bincode::config::standard();
    let bytes = bincode::serde::encode_to_vec(&ctx, config).expect("encode");
    let (restored, _): (Context<'static>, usize) =
        bincode::serde::decode_from_slice(&bytes, config).expect("decode");
    assert_consistent(&restored, entry, &before);
}

#[test]
fn integrity_reports_broken_links() {
    let (mut ctx, f, block) = fixture();
    let ids: Vec<InstructionId> = (0..3).map(|n| mint(&mut ctx, f, block, n)).collect();
    for id in &ids {
        ctx.bodies[f].link_last(block.local, id.local);
    }
    assert_eq!(verify_body_arena_integrity(&ctx), Vec::<String>::new());

    let has = |ctx: &Context<'_>, needle: &str| {
        let report = verify_body_arena_integrity(ctx);
        assert!(
            report.iter().any(|line| line.contains(needle)),
            "expected {needle:?} in {report:#?}"
        );
    };

    // A back link that skips an instruction.
    ctx.bodies[f].insns[ids[2].local].prev = Some(ids[0].local);
    has(&ctx, "links back to");
    ctx.bodies[f].insns[ids[2].local].prev = Some(ids[1].local);

    // A count that disagrees with the walk.
    ctx.bodies[f].blocks[block.local].instructions.len = 2;
    has(&ctx, "links more instructions than its count");
    ctx.bodies[f].blocks[block.local].instructions.len = 4;
    has(&ctx, "links 3 instructions but counts 4");
    ctx.bodies[f].blocks[block.local].instructions.len = 3;

    // An end that is not where the walk ends.
    ctx.bodies[f].blocks[block.local].instructions.last = Some(ids[1].local);
    has(&ctx, "ends at");
    ctx.bodies[f].blocks[block.local].instructions.last = Some(ids[2].local);

    // A forward link into another block's instruction.
    let other = ctx.get_or_make_block(0x2000, f);
    let o = mint(&mut ctx, f, other, 9);
    ctx.bodies[f].link_last(other.local, o.local);
    ctx.bodies[f].insns[ids[2].local].next = Some(o.local);
    ctx.bodies[f].blocks[block.local].instructions.len = 4;
    has(&ctx, "block memberships");
}
