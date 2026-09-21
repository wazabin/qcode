//! A block the VM empties and lifts again holds as many instructions as
//! before, under new ids. Code compiled from the old ids reads and writes the
//! interpreter's value table by ids that no longer exist in the block — a
//! continuation entered after a hook's interrupt would import a stale value
//! (Embench's edn pushed a return address through a stale stack pointer and
//! returned into the wrong function). The cache has to go by the block's
//! revision, not its length.

use qcode::{
    address_index::AddressIndex,
    context::Context,
    value::{BlockId, QCodeMut},
};
use qcode_jit::Jit;
use wazabin_qcode_sleigh::SleighLifter;

/// `add eax, ebx`.
const ADD: &[u8] = &[0x01, 0xd8];

fn lifter() -> &'static SleighLifter<'static> {
    static LIFTER: std::sync::OnceLock<SleighLifter<'static>> = std::sync::OnceLock::new();
    LIFTER.get_or_init(|| SleighLifter::new(sleigh_precompile::x64::spec()))
}

/// Lifts `ADD` at 0x1000 into `ctx`, returning its block: the block already
/// at that address when there is one.
fn lift_at_0x1000(ctx: &mut Context<'static>) -> BlockId {
    let mut index = AddressIndex::analyze(ctx);
    lifter()
        .decode_and_lift_indexed(ctx, &mut index, 0x1000, ADD, None)
        .expect("the instruction decodes and lifts")
        .entry()
}

/// Lifts `ADD` at 0x1000 and returns its block.
fn lift() -> (Context<'static>, BlockId) {
    let mut ctx = lifter().new_context();
    let block = lift_at_0x1000(&mut ctx);
    (ctx, block)
}

#[test]
fn a_block_rebuilt_to_the_same_length_is_compiled_again() {
    let (mut ctx, block) = lift();
    let mut jit = Jit::new();
    jit.try_compile(&ctx, block)
        .expect("plain arithmetic compiles");
    jit.try_compile(&ctx, block).expect("still compiles");
    assert_eq!(
        jit.stats.compiled, 1,
        "an unchanged block is served from the cache"
    );

    // The same instruction lifted again into the emptied block, as the VM
    // does after the guest writes over it: the block is as long as it was,
    // and is not the block the code was compiled from.
    let before = ctx.block(block).insn_count();
    ctx.clear_block_instructions(block);
    assert_eq!(
        lift_at_0x1000(&mut ctx),
        block,
        "an empty block at an address is refilled"
    );
    assert_eq!(ctx.block(block).insn_count(), before);

    jit.try_compile(&ctx, block).expect("compiles again");
    assert_eq!(jit.stats.compiled, 2, "a rebuilt block is compiled afresh");
}
