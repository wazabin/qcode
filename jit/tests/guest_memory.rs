//! Guest RAM under the JIT: the same state the interpreter would leave, and the
//! same faults where it would fault.
//!
//! Compiled code reaches RAM through an inlined translation and permission
//! check, falling back to the VM for anything it cannot settle. That is two
//! paths where the interpreter has one, and which of them runs depends on
//! *history* — a page is only cached once an access to it has already been
//! served the slow way. So every check here runs the block twice: the first
//! execution exercises the fallback, the second the inline path, and both have
//! to agree with the interpreter.

use qcode::{address_index::AddressIndex, context::Context, value::BlockId};
use qcode_emulator::{EmulatorMemory, StandaloneEmulator};
use qcode_jit::Jit;
use qcode_vm::{FaultKind, PAGE_SIZE, VmMemory, perm};
use wazabin_qcode_sleigh::SleighLifter;

/// Where the test programs live, and where their data does.
const CODE: u64 = 0x1000;
const DATA: u64 = 0x40000;

const WATCHED: &[&str] = &["RAX", "RBX", "RCX", "RDX"];

fn lifter() -> &'static SleighLifter<'static> {
    static LIFTER: std::sync::OnceLock<SleighLifter<'static>> = std::sync::OnceLock::new();
    LIFTER.get_or_init(|| SleighLifter::new(sleigh_precompile::x64::spec()))
}

fn lift(code: &[u8]) -> (Context<'static>, BlockId) {
    let lifter = lifter();
    let mut ctx = lifter.new_context();
    let mut index = AddressIndex::analyze(&ctx);
    let block = lifter
        .decode_and_lift_indexed(&mut ctx, &mut index, CODE, code, None)
        .expect("the instruction decodes and lifts");
    (ctx, block)
}

/// Puts the machine back to its starting state: a known data page, `RBX`
/// pointing into it, and a known value in `RAX` to store.
fn seed(emu: &mut StandaloneEmulator<VmMemory>, ctx: &Context<'_>, block: BlockId, pointer: u64) {
    emu.memory
        .mmu
        .write_unchecked(DATA, &[0x11; 64], perm::RW_INIT);
    for (name, value) in [
        ("RAX", 0x1234_5678_9abc_def0u64),
        ("RBX", pointer),
        ("RCX", 0),
        ("RDX", 0),
    ] {
        emu.set_varnode_by_name(ctx, name, value)
            .expect("register exists");
    }
    emu.invalidate_block_cache();
    emu.block = block;
    emu.idx = 0;
}

/// A machine with two mapped data pages, seeded.
///
/// One machine serves every pass of a check rather than one per pass: a
/// compiled block resolves its flat-space slots against the machine it first
/// ran on, and the translation cache a second pass needs to be warm lives on
/// this machine's memory too.
fn machine(block: BlockId, ctx: &Context<'_>, pointer: u64) -> StandaloneEmulator<VmMemory> {
    let mut emu = StandaloneEmulator::<VmMemory>::new_in(block);
    emu.memory.configure_spaces(ctx);
    emu.memory
        .mmu
        .map(DATA, 2 * PAGE_SIZE, perm::RW_INIT)
        .expect("the data pages map");
    seed(&mut emu, ctx, block, pointer);
    emu
}

/// The architectural state and the data page, which is where a store lands.
fn state(emu: &mut StandaloneEmulator<VmMemory>, ctx: &Context<'_>) -> (Vec<Option<u64>>, Vec<u8>) {
    let registers = WATCHED
        .iter()
        .map(|name| emu.read_varnode_by_name(ctx, name))
        .collect();
    let mut page = vec![0u8; 64];
    emu.memory
        .mmu
        .read(DATA, &mut page)
        .expect("the data page is readable");
    (registers, page)
}

/// Runs `code` interpreted once and jitted twice, requiring all three to leave
/// the same state. Panics unless the JIT actually took the block.
fn agree(code: &[u8], pointer: u64) {
    let (ctx, block) = lift(code);

    let mut interpreted = machine(block, &ctx, pointer);
    interpreted
        .run_block(&ctx)
        .expect("the interpreter runs the block");
    let expected = state(&mut interpreted, &ctx);

    let mut jit = Jit::new();
    let mut jitted = machine(block, &ctx, pointer);
    for pass in 0..2 {
        seed(&mut jitted, &ctx, block, pointer);
        let ran = jit
            .run_block(&ctx, &mut jitted, block, false)
            .expect("running compiled code does not fault")
            .is_some();
        assert!(
            ran,
            "the block was declined; this test has nothing to check"
        );
        assert_eq!(
            expected,
            state(&mut jitted, &ctx),
            "compiled code disagreed with the interpreter on pass {pass}"
        );
    }
    // The second pass had a warm translation, so it ran wholly inline. A run
    // that never got there would still pass the equality above, on the
    // fallback alone.
    assert!(
        jit.stats.native_runs >= 2,
        "both passes must have run as native code"
    );
}

#[test]
fn ram_accesses_agree_with_the_interpreter() {
    let programs: [(&str, &[u8]); 6] = [
        ("mov eax, [rbx]", &[0x8b, 0x03]),
        ("mov rax, [rbx]", &[0x48, 0x8b, 0x03]),
        ("mov [rbx], eax", &[0x89, 0x03]),
        ("mov [rbx], rax", &[0x48, 0x89, 0x03]),
        ("mov al, [rbx]", &[0x8a, 0x03]),
        ("mov [rbx], al", &[0x88, 0x03]),
    ];
    for (name, code) in programs {
        eprintln!("checking {name}");
        agree(code, DATA + 8);
    }
}

#[test]
fn an_unaligned_access_within_a_page_stays_inline() {
    // Nothing about a guest access promises alignment, and refusing one would
    // be a silent trip to the fallback on ordinary code.
    agree(&[0x48, 0x8b, 0x03], DATA + 3);
    agree(&[0x48, 0x89, 0x03], DATA + 5);
}

#[test]
fn an_access_straddling_two_pages_still_agrees() {
    // The inline path cannot serve this — one translation covers one page — so
    // it must decline to the fallback rather than read across the boundary.
    agree(&[0x48, 0x8b, 0x03], DATA + PAGE_SIZE - 4);
    agree(&[0x48, 0x89, 0x03], DATA + PAGE_SIZE - 4);
}

/// Runs `code` under the JIT twice and returns the fault it took, if any.
fn fault_from(code: &[u8], pointer: u64, prepare: impl Fn(&mut VmMemory)) -> Option<FaultKind> {
    let (ctx, block) = lift(code);
    let mut jit = Jit::new();
    let mut emu = machine(block, &ctx, pointer);
    prepare(&mut emu.memory);
    let mut faulted = None;
    for _ in 0..2 {
        seed(&mut emu, &ctx, block, pointer);
        prepare(&mut emu.memory);
        if jit.run_block(&ctx, &mut emu, block, false).is_err() {
            // The fault is left for the VM to turn into an exit, exactly as an
            // interpreted one is.
            faulted = emu.memory.take_fault().map(|fault| fault.kind);
        }
    }
    faulted
}

#[test]
fn an_unmapped_access_faults_where_the_interpreter_would() {
    assert_eq!(
        fault_from(&[0x48, 0x8b, 0x03], 0x9000_0000, |_| {}),
        Some(FaultKind::ReadUnmapped)
    );
    assert_eq!(
        fault_from(&[0x48, 0x89, 0x03], 0x9000_0000, |_| {}),
        Some(FaultKind::WriteUnmapped)
    );
}

#[test]
fn a_permission_refusal_survives_a_warm_translation() {
    // The inline permission check is the only thing standing between a
    // compiled store and a read-only byte, and it is easy to write a test that
    // never reaches it: every operation that changes permissions flushes the
    // translation cache, so a page protected just beforehand is cold and the
    // fallback answers. Permissions here are per *byte*, so the cache is
    // warmed on a writable byte of the very page whose other half refuses the
    // write.
    let (ctx, block) = lift(&[0x48, 0x89, 0x03]); // mov [rbx], rax
    let mut jit = Jit::new();
    let mut emu = machine(block, &ctx, DATA);
    emu.memory
        .mmu
        .protect(DATA + PAGE_SIZE / 2, PAGE_SIZE / 2, perm::READ)
        .expect("the second half is mapped");

    // Warm the cache on the writable half.
    seed(&mut emu, &ctx, block, DATA);
    jit.run_block(&ctx, &mut emu, block, false)
        .expect("the writable half accepts the store");
    assert!(
        emu.memory.mmu.cache_translation(DATA),
        "the page must be cacheable for this test to check anything"
    );

    // Same page, same cached translation, a byte that refuses the write.
    seed(&mut emu, &ctx, block, DATA + PAGE_SIZE / 2);
    let refused = jit.run_block(&ctx, &mut emu, block, false).is_err();
    assert!(refused, "a compiled store must not bypass permissions");
    let fault = emu.memory.take_fault().expect("the fault was recorded");
    assert_eq!(fault.kind, FaultKind::WritePerm);
    assert_eq!(fault.addr, DATA + PAGE_SIZE / 2);
    // Nothing landed: the refused half is still the zeroes `map` left there.
    let mut out = [0u8; 8];
    emu.memory.mmu.read(DATA + PAGE_SIZE / 2, &mut out).unwrap();
    assert_eq!(out, [0; 8]);
}

#[test]
fn a_store_through_compiled_code_marks_its_bytes_initialized() {
    // Nothing in a normal run reads the INIT bits, which is exactly why an
    // inline store forgetting to set them would go unnoticed until a
    // `check_uninit` run reported a byte the guest had plainly written.
    let (ctx, block) = lift(&[0x48, 0x89, 0x03]); // mov [rbx], rax
    let mut jit = Jit::new();
    let mut emu = machine(block, &ctx, DATA + 8);
    // Start from bytes that are mapped and writable but undefined.
    emu.memory
        .mmu
        .protect(DATA, PAGE_SIZE, perm::READ | perm::WRITE)
        .expect("the page is mapped");

    for _ in 0..2 {
        emu.invalidate_block_cache();
        emu.block = block;
        emu.idx = 0;
        jit.run_block(&ctx, &mut emu, block, false)
            .expect("the store succeeds");
    }

    emu.memory.mmu.set_check_uninit(true);
    let mut out = [0u8; 8];
    emu.memory
        .mmu
        .read(DATA + 8, &mut out)
        .expect("the bytes the guest just wrote are defined");
    assert_eq!(u64::from_le_bytes(out), 0x1234_5678_9abc_def0);
}
