//! Snapshot and restore: a machine put back must run exactly as it did.

use qcode_jit::Jit;
use qcode_vm::{PAGE_SIZE, Vm, VmExit, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

const CODE: u64 = 0x1000;
const DATA: u64 = 0x20000;

fn machine(code: &[u8], jit: bool) -> Vm<SleighCodeSource<'static>> {
    let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
    let ctx = source.new_context();
    let mut memory = VmMemory::new();
    memory
        .mmu
        .write_unchecked(CODE, code, perm::RW_INIT | perm::EXEC);
    memory.mmu.map(DATA, 4 * PAGE_SIZE, perm::RW_INIT).unwrap();
    let mut vm = Vm::at_address(ctx, CODE, source, memory).expect("the entry decodes");
    if jit {
        vm.set_block_executor(Box::new(Jit::new()));
    }
    vm
}

/// A linear congruential generator that scatters its state through a
/// buffer: registers and memory both change on every pass, and depend on
/// everything before.
///
/// ```text
/// 1000: mov ecx, 10000
/// 1005: mov eax, 1
/// 100a: mov ebx, 0x20000
/// loop:
/// 100f: imul eax, eax, 1103515245
/// 1015: add eax, 12345
/// 101a: mov edx, eax
/// 101c: and edx, 0xfff
/// 1022: mov [rbx+rdx*4], eax        ; 4 KiB window inside the 16 KiB buffer
/// 1025: add [rbx+rdx*2], eax
/// 1028: dec ecx
/// 102a: jnz loop
/// 102c: jmp $
/// ```
const LCG: &[u8] = &[
    0xb9, 0x10, 0x27, 0x00, 0x00, // mov ecx, 10000
    0xb8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
    0xbb, 0x00, 0x00, 0x02, 0x00, // mov ebx, 0x20000
    0x69, 0xc0, 0x6d, 0x4e, 0xc6, 0x41, // imul eax, eax, 1103515245
    0x05, 0x39, 0x30, 0x00, 0x00, // add eax, 12345
    0x89, 0xc2, // mov edx, eax
    0x81, 0xe2, 0xff, 0x0f, 0x00, 0x00, // and edx, 0xfff
    0x89, 0x04, 0x93, // mov [rbx+rdx*4], eax
    0x01, 0x04, 0x53, // add [rbx+rdx*2], eax
    0xff, 0xc9, // dec ecx
    0x75, 0xe3, // jnz loop
    0xeb, 0xfe, // jmp $
];
const END: u64 = 0x102c;

const WATCHED: &[&str] = &["RAX", "RBX", "RCX", "RDX", "CF", "ZF", "SF", "OF"];

/// Registers plus the whole data buffer.
fn digest(vm: &mut Vm<SleighCodeSource<'static>>) -> (Vec<Option<u64>>, Vec<u8>) {
    let ctx = vm.context().clone();
    let registers = WATCHED
        .iter()
        .map(|name| vm.emulator().read_varnode_by_name(&ctx, name))
        .collect();
    let mut buffer = vec![0u8; 4 * PAGE_SIZE as usize];
    vm.memory().mmu.read(DATA, &mut buffer).unwrap();
    (registers, buffer)
}

/// Snapshot at the start, run to the end, restore, run to the end again:
/// the same machine both times, and the restore itself undid everything.
#[test]
fn a_restored_machine_recomputes_the_same_answer() {
    for jit in [false, true] {
        let mut vm = machine(LCG, jit);
        vm.add_breakpoint(END);
        let start = digest(&mut vm);
        let snapshot = vm.snapshot().expect("nothing is pending at the entry");

        assert!(matches!(vm.run(100_000_000), VmExit::Breakpoint(END)));
        let first = digest(&mut vm);
        assert_ne!(first, start, "the program did nothing");

        vm.restore(&snapshot).expect("the entry is still lifted");
        assert_eq!(
            digest(&mut vm),
            start,
            "jit={jit}: restore left changes behind"
        );
        assert!(matches!(vm.run(100_000_000), VmExit::Breakpoint(END)));
        assert_eq!(digest(&mut vm), first, "jit={jit}: the second run differed");
    }
}

/// A snapshot taken wherever a step budget happened to stop, stepped on to
/// the next instruction boundary, resumes exactly — with the values the
/// rest of its block still needs.
#[test]
fn a_snapshot_mid_flight_resumes_exactly() {
    for jit in [false, true] {
        for budget in [1, 7, 100, 1234, 65_537] {
            let mut vm = machine(LCG, jit);
            vm.add_breakpoint(END);
            assert!(matches!(vm.run(budget), VmExit::InstructionLimit));
            while vm.instruction_boundary().is_none() {
                assert!(vm.step().is_none());
            }
            let snapshot = vm.snapshot().expect("no interrupt is pending");

            assert!(matches!(vm.run(100_000_000), VmExit::Breakpoint(END)));
            let once = digest(&mut vm);
            vm.restore(&snapshot)
                .unwrap_or_else(|e| panic!("jit={jit} budget={budget}: {e}"));
            assert!(matches!(vm.run(100_000_000), VmExit::Breakpoint(END)));
            assert_eq!(digest(&mut vm), once, "jit={jit} budget={budget}");
        }
    }
}

/// Between two ops of one instruction, a snapshot is exact for as long as
/// its block stands. Here nothing reshapes the block, so it does.
#[test]
fn a_snapshot_inside_an_instruction_resumes_while_its_block_stands() {
    for jit in [false, true] {
        let mut vm = machine(LCG, jit);
        vm.add_breakpoint(END);
        // Well into the loop: the run is one block by now, and stays one.
        assert!(matches!(vm.run(50_000), VmExit::InstructionLimit));
        let snapshot = vm.snapshot().unwrap();
        let inside = vm.instruction_boundary().is_none();

        assert!(matches!(vm.run(100_000_000), VmExit::Breakpoint(END)));
        let once = digest(&mut vm);
        vm.restore(&snapshot).unwrap();
        assert!(matches!(vm.run(100_000_000), VmExit::Breakpoint(END)));
        assert_eq!(digest(&mut vm), once, "jit={jit} inside={inside}");
    }
}

/// Restoring more than once, and out of order, from the same snapshot.
#[test]
fn a_snapshot_can_be_restored_repeatedly() {
    let mut vm = machine(LCG, true);
    vm.add_breakpoint(END);
    let snapshot = vm.snapshot().unwrap();
    let mut results = Vec::new();
    for budget in [100, 5000, 100_000_000, 5000] {
        vm.restore(&snapshot).unwrap();
        vm.run(budget);
        results.push(digest(&mut vm));
    }
    assert_eq!(results[1], results[3]);
    assert_ne!(results[0], results[1]);
    assert_ne!(results[1], results[2]);
}

/// The self-patching loop from `self_modifying.rs`: a snapshot taken before
/// the patch puts the original bytes back, and the code lifted from the
/// patched bytes must go with them.
#[test]
fn a_restore_over_patched_code_reverts_the_code() {
    const PATCHER: &[u8] = &[
        0xb9, 0x03, 0x00, 0x00, 0x00, // 1000: mov ecx, 3
        0xbb, 0x01, 0x00, 0x00, 0x00, // 1005: mov ebx, 1
        0x80, 0x05, 0xf5, 0xff, 0xff, 0xff, 0x01, // 100a: add byte [rip-0xb], 1
        0x01, 0xd8, // 1011: add eax, ebx
        0xff, 0xc9, // 1013: dec ecx
        0x75, 0xee, // 1015: jnz 1005
        0xeb, 0xfe, // 1017: jmp $
    ];
    for jit in [false, true] {
        let mut vm = machine(PATCHER, jit);
        vm.add_breakpoint(0x1017);
        let snapshot = vm.snapshot().unwrap();
        let ctx = vm.context().clone();
        for round in 0..3 {
            assert!(matches!(vm.run(10_000), VmExit::Breakpoint(0x1017)));
            let eax = vm.emulator().read_varnode_by_name(&ctx, "EAX");
            assert_eq!(eax, Some(6), "jit={jit} round={round}");
            vm.restore(&snapshot).unwrap();
            let mut imm = [0u8; 1];
            vm.memory().mmu.read(0x1006, &mut imm).unwrap();
            assert_eq!(imm, [1], "the patched byte was not put back");
        }
    }
}

/// A snapshot is a table of shared pages, and a restore touches only the
/// pages written since: the acceptance figure is a 64 MiB image with 100
/// dirty pages restored in under a millisecond.
#[test]
fn restoring_a_large_image_costs_only_its_dirty_pages() {
    let mut vm = machine(LCG, false);
    const IMAGE: u64 = 0x1000_0000;
    const SIZE: u64 = 64 << 20;
    vm.memory_mut().mmu.map(IMAGE, SIZE, perm::RW_INIT).unwrap();
    let snapshot = vm.snapshot().unwrap();
    for page in 0..100u64 {
        let addr = IMAGE + page * (SIZE / 100);
        vm.memory_mut().mmu.write(addr, &[page as u8; 8]).unwrap();
    }
    assert_eq!(vm.memory().mmu.dirty_pages(), 100);

    let started = std::time::Instant::now();
    vm.restore(&snapshot).unwrap();
    let took = started.elapsed();
    eprintln!("restore of 64 MiB with 100 dirty pages: {took:?}");

    let mut out = [0u8; 8];
    vm.memory()
        .mmu
        .read(IMAGE + 3 * (SIZE / 100), &mut out)
        .unwrap();
    assert_eq!(out, [0; 8]);
    // The bound is for an optimised build; a debug build is only checked
    // for being in the right order of magnitude.
    let bound = if cfg!(debug_assertions) { 20 } else { 1 };
    assert!(
        took < std::time::Duration::from_millis(bound),
        "restore took {took:?}, bound {bound} ms"
    );
}
