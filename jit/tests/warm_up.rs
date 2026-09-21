//! A block is compiled on the entry the warm-up names, not before, and a
//! rebuilt block earns its compilation again: the entries that matter are
//! entries to the block as it is now.

use qcode::{
    address_index::AddressIndex,
    context::Context,
    value::{BlockId, QCodeMut},
};
use qcode_emulator::{EmulatorMemory, StandaloneEmulator};
use qcode_jit::Jit;
use qcode_vm::VmMemory;
use wazabin_qcode_sleigh::SleighLifter;

/// `add eax, ebx`.
const ADD: &[u8] = &[0x01, 0xd8];

fn lifter() -> &'static SleighLifter<'static> {
    static LIFTER: std::sync::OnceLock<SleighLifter<'static>> = std::sync::OnceLock::new();
    LIFTER.get_or_init(|| SleighLifter::new(sleigh_precompile::x64::spec()))
}

fn lift_at_0x1000(ctx: &mut Context<'static>) -> BlockId {
    let mut index = AddressIndex::analyze(ctx);
    lifter()
        .decode_and_lift_indexed(ctx, &mut index, 0x1000, ADD, None)
        .expect("the instruction decodes and lifts")
        .entry()
}

fn machine(ctx: &Context<'static>, block: BlockId) -> StandaloneEmulator<VmMemory> {
    let mut emu = StandaloneEmulator::<VmMemory>::new_in(block);
    emu.memory.configure_spaces(ctx);
    emu
}

/// Enters `block` once through the executor, reporting whether it ran
/// natively.
fn enter(
    jit: &mut Jit,
    ctx: &Context<'static>,
    emu: &mut StandaloneEmulator<VmMemory>,
    block: BlockId,
) -> bool {
    emu.block = block;
    emu.idx = 0;
    jit.run_block(ctx, emu, block, 0, 0)
        .expect("the block runs")
        .is_some()
}

#[test]
fn a_block_is_compiled_on_the_entry_the_warm_up_names() {
    let mut ctx = lifter().new_context();
    let block = lift_at_0x1000(&mut ctx);
    let mut emu = machine(&ctx, block);
    let mut jit = Jit::new();
    jit.set_warm_up(3);
    assert!(
        !enter(&mut jit, &ctx, &mut emu, block),
        "first entry: interpreted"
    );
    assert!(
        !enter(&mut jit, &ctx, &mut emu, block),
        "second entry: interpreted"
    );
    assert_eq!(jit.stats.compiled, 0);
    assert!(
        enter(&mut jit, &ctx, &mut emu, block),
        "third entry: compiled"
    );
    assert!(
        enter(&mut jit, &ctx, &mut emu, block),
        "and served from the cache"
    );
    assert_eq!(jit.stats.compiled, 1);
    assert_eq!(jit.stats.declined, 0, "warming is not a decline");
}

#[test]
fn a_rebuilt_block_warms_up_again() {
    let mut ctx = lifter().new_context();
    let block = lift_at_0x1000(&mut ctx);
    let mut emu = machine(&ctx, block);
    let mut jit = Jit::new();
    assert!(!enter(&mut jit, &ctx, &mut emu, block));
    assert!(
        enter(&mut jit, &ctx, &mut emu, block),
        "the default compiles on the second entry"
    );
    assert_eq!(jit.stats.compiled, 1);

    ctx.clear_block_instructions(block);
    assert_eq!(lift_at_0x1000(&mut ctx), block);
    assert!(
        !enter(&mut jit, &ctx, &mut emu, block),
        "a new revision starts over"
    );
    assert!(enter(&mut jit, &ctx, &mut emu, block));
    assert_eq!(jit.stats.compiled, 2);
}

#[test]
fn try_compile_does_not_wait() {
    let mut ctx = lifter().new_context();
    let block = lift_at_0x1000(&mut ctx);
    let mut emu = machine(&ctx, block);
    let mut jit = Jit::new();
    jit.try_compile(&ctx, block).expect("compiles at once");
    assert_eq!(jit.stats.compiled, 1);
    assert!(
        enter(&mut jit, &ctx, &mut emu, block),
        "and the run path finds it"
    );
    assert_eq!(jit.stats.compiled, 1);
}
