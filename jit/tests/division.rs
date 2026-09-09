//! Integer division under the JIT, at the values that make it interesting.
//!
//! QCode leaves nothing about division undefined — a zero divisor yields zero,
//! and signed division of the most negative value by `-1` wraps — while the
//! machine instruction faults on both. So compiled code has to steer around
//! exactly the cases a random test never generates, which is why they are
//! enumerated here rather than sampled.
//!
//! x86-64 divides a 128-bit dividend (`RDX:RAX`) by a 64-bit divisor, so these
//! also cover the width the host has no instruction for and the runtime
//! performs.

use qcode::{address_index::AddressIndex, context::Context, value::BlockId};
use qcode_emulator::{EmulatorMemory, StandaloneEmulator};
use qcode_jit::Jit;
use qcode_vm::VmMemory;
use wazabin_qcode_sleigh::SleighLifter;

const WATCHED: &[&str] = &["RAX", "RDX", "RBX"];

fn lifter() -> &'static SleighLifter<'static> {
    static LIFTER: std::sync::OnceLock<SleighLifter<'static>> = std::sync::OnceLock::new();
    LIFTER.get_or_init(|| SleighLifter::new(sleigh_precompile::x64::spec()))
}

fn lift(code: &[u8]) -> (Context<'static>, BlockId) {
    let lifter = lifter();
    let mut ctx = lifter.new_context();
    let mut index = AddressIndex::analyze(&ctx);
    let block = lifter
        .decode_and_lift_indexed(&mut ctx, &mut index, 0x1000, code, None)
        .expect("the instruction decodes and lifts");
    (ctx, block)
}

fn seed(
    emu: &mut StandaloneEmulator<VmMemory>,
    ctx: &Context<'_>,
    block: BlockId,
    (rdx, rax, rbx): (u64, u64, u64),
) {
    for (name, value) in [("RDX", rdx), ("RAX", rax), ("RBX", rbx)] {
        emu.set_varnode_by_name(ctx, name, value)
            .expect("register exists");
    }
    emu.invalidate_block_cache();
    emu.block = block;
    emu.idx = 0;
}

/// A machine seeded with one set of operands.
///
/// One machine serves every seed of a check: a compiled block resolves its
/// flat-space slots against the machine it first ran on.
fn machine(
    block: BlockId,
    ctx: &Context<'_>,
    operands: (u64, u64, u64),
) -> StandaloneEmulator<VmMemory> {
    let mut emu = StandaloneEmulator::<VmMemory>::new_in(block);
    emu.memory.configure_spaces(ctx);
    seed(&mut emu, ctx, block, operands);
    emu
}

fn state(emu: &mut StandaloneEmulator<VmMemory>, ctx: &Context<'_>) -> Vec<Option<u64>> {
    WATCHED
        .iter()
        .map(|name| emu.read_varnode_by_name(ctx, name))
        .collect()
}

/// Runs one instruction both ways over `seeds` and requires them to agree.
///
/// The interpreter is the specification here: whatever it computes for a
/// dividend of zero over zero is what compiled code owes.
fn agree_on(code: &[u8], what: &str, seeds: &[(u64, u64, u64)]) {
    let (ctx, block) = lift(code);
    let mut jit = Jit::new();
    let mut compiled_any = false;
    let mut interpreted = machine(block, &ctx, seeds[0]);
    let mut jitted = machine(block, &ctx, seeds[0]);

    for &operands in seeds {
        seed(&mut interpreted, &ctx, block, operands);
        interpreted
            .run_block(&ctx)
            .expect("the interpreter runs the block");
        let expected = state(&mut interpreted, &ctx);

        seed(&mut jitted, &ctx, block, operands);
        let ran = jit
            .run_block(&ctx, &mut jitted, block, 0, false)
            .expect("compiled code does not fault")
            .is_some();
        if !ran {
            continue;
        }
        compiled_any = true;
        assert_eq!(
            expected,
            state(&mut jitted, &ctx),
            "{what} disagreed with the interpreter on RDX:RAX/RBX = {operands:x?}"
        );
    }
    assert!(
        compiled_any,
        "{what} was declined, so nothing above was checked"
    );
}

/// Dividends and divisors worth trying: zero, one, the most negative value,
/// all-ones, and something with no special structure at all.
fn seeds() -> Vec<(u64, u64, u64)> {
    const MOST_NEGATIVE: u64 = 1 << 63;
    let halves = [0, 1, MOST_NEGATIVE, u64::MAX, 0x0123_4567_89ab_cdef];
    let divisors = [0, 1, u64::MAX, 2, 7, MOST_NEGATIVE];
    let mut out = Vec::new();
    for &high in &halves {
        for &low in &halves {
            for &divisor in &divisors {
                out.push((high, low, divisor));
            }
        }
    }
    out
}

#[test]
fn unsigned_division_agrees_with_the_interpreter() {
    // div rbx: RDX:RAX / RBX, quotient to RAX and remainder to RDX.
    agree_on(&[0x48, 0xf7, 0xf3], "div rbx", &seeds());
}

#[test]
fn signed_division_agrees_with_the_interpreter() {
    // idiv rbx, which is where the most-negative-over-minus-one case lives.
    agree_on(&[0x48, 0xf7, 0xfb], "idiv rbx", &seeds());
}

#[test]
fn narrow_division_agrees_with_the_interpreter() {
    // 32-bit: EDX:EAX / EBX, which the machine *can* do — and traps on, for
    // the same two inputs.
    agree_on(&[0xf7, 0xf3], "div ebx", &seeds());
    agree_on(&[0xf7, 0xfb], "idiv ebx", &seeds());
}
