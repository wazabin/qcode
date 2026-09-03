//! The JIT must be invisible: a block run as native code has to leave exactly
//! the state the interpreter would have left.
//!
//! This is the gate that makes a partial backend safe. Coverage can grow freely
//! as long as every block the compiler *accepts* agrees with the interpreter,
//! and every block it declines is run by the interpreter unchanged.

use qcode::{address_index::AddressIndex, context::Context, value::BlockId};
use qcode_emulator::{EmulatorMemory, StandaloneEmulator};
use qcode_jit::Jit;
use qcode_vm::VmMemory;
use wazabin_qcode_sleigh::SleighLifter;

/// Registers compared after each run: the architectural state an x86-64
/// integer instruction can touch.
const WATCHED: &[&str] = &[
    "RAX", "RBX", "RCX", "RDX", "EAX", "EBX", "ECX", "EDX", "CF", "ZF", "SF", "OF", "AF", "PF",
];

fn lifter() -> &'static SleighLifter<'static> {
    static LIFTER: std::sync::OnceLock<SleighLifter<'static>> = std::sync::OnceLock::new();
    LIFTER.get_or_init(|| SleighLifter::new(sleigh_precompile::x64::spec()))
}

/// Lifts one instruction at 0x1000 and returns its block.
fn lift(code: &[u8]) -> (Context<'static>, BlockId) {
    let lifter = lifter();
    let mut ctx = lifter.new_context();
    let mut index = AddressIndex::analyze(&ctx);
    let block = lifter
        .decode_and_lift_indexed(&mut ctx, &mut index, 0x1000, code, None)
        .expect("the instruction decodes and lifts");
    (ctx, block)
}

/// Seeds the registers both runs start from.
fn seed(emu: &mut StandaloneEmulator<VmMemory>, ctx: &Context<'_>) {
    for (name, value) in [
        ("RAX", 0x1234_5678_9abc_def0u64),
        ("RBX", 0x0fed_cba9_8765_4321),
    ] {
        emu.set_varnode_by_name(ctx, name, value)
            .expect("register exists");
    }
}

fn state(emu: &mut StandaloneEmulator<VmMemory>, ctx: &Context<'_>) -> Vec<(String, Option<u64>)> {
    WATCHED
        .iter()
        .map(|name| ((*name).to_owned(), emu.read_varnode_by_name(ctx, name)))
        .collect()
}

fn machine(block: BlockId, ctx: &Context<'_>) -> StandaloneEmulator<VmMemory> {
    let mut emu = StandaloneEmulator::<VmMemory>::new_in(block);
    emu.memory.configure_spaces(ctx);
    seed(&mut emu, ctx);
    emu
}

/// Runs `code` both ways and requires the resulting state to be identical.
/// Returns whether the JIT actually accepted the block.
fn agree(code: &[u8]) -> bool {
    let (ctx, block) = lift(code);

    let mut interpreted = machine(block, &ctx);
    interpreted
        .run_block(&ctx)
        .expect("the interpreter runs the block");
    let expected = state(&mut interpreted, &ctx);

    let mut jitted = machine(block, &ctx);
    let mut jit = Jit::new();
    let ran = jit
        .run_block(&ctx, &mut jitted, block)
        .expect("running compiled code does not fault")
        .is_some();
    if !ran {
        return false;
    }
    let actual = state(&mut jitted, &ctx);

    assert_eq!(
        expected, actual,
        "compiled code disagreed with the interpreter"
    );
    true
}

#[test]
fn compiled_blocks_match_the_interpreter() {
    let programs: [(&str, &[u8]); 7] = [
        ("add eax, ebx", &[0x01, 0xd8]),
        ("sub eax, ebx", &[0x29, 0xd8]),
        ("xor eax, ebx", &[0x31, 0xd8]),
        ("and eax, ebx", &[0x21, 0xd8]),
        ("or eax, ebx", &[0x09, 0xd8]),
        ("dec ecx", &[0xff, 0xc9]),
        ("mov eax, 1", &[0xb8, 0x01, 0x00, 0x00, 0x00]),
    ];

    let mut accepted = 0;
    for (name, code) in programs {
        if agree(code) {
            accepted += 1;
            eprintln!("compiled and matched: {name}");
        } else {
            eprintln!("declined (interpreter handles it): {name}");
        }
    }
    // A backend that compiles nothing would pass every equality check above
    // vacuously, so require that it actually took some real work.
    assert!(
        accepted > 0,
        "the compiler accepted no blocks at all; the agreement checks were vacuous"
    );
    eprintln!("accepted {accepted} of 7 blocks");
}

#[test]
fn a_declined_block_is_reported_rather_than_miscompiled() {
    // Integer division is deliberately declined: its trap behaviour is the
    // guest architecture's, not Cranelift's.
    let (ctx, block) = lift(&[0xf7, 0xf3]); // div ebx
    let mut emu = machine(block, &ctx);
    let mut jit = Jit::new();
    let ran = jit
        .run_block(&ctx, &mut emu, block)
        .expect("declining is not an error")
        .is_some();
    assert!(!ran, "division must be left to the interpreter");
    assert_eq!(jit.stats.declined, 1);
    assert_eq!(jit.stats.compiled, 0);
}
