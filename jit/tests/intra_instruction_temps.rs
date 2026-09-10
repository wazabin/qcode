//! A SLEIGH unique that outlives its block: an instruction with a branch in
//! its semantics writes a temporary before the branch and reads it after,
//! in a block of its own. The VM's cleanup forwards temp stores to their
//! loads and removes what nothing reads; it must not remove the store the
//! next block of the same instruction reads.
//!
//! `lock cmpxchg [r9], r10d` is the case that surfaced this: the address
//! and the loaded value are parked in temporaries, the compare branches, and
//! the taken side stores through the parked address. With the store removed
//! the parked address reads as zero, and musl's first lock in busybox `sort`
//! wrote to address 0.

use qcode_jit::Jit;
use qcode_vm::{Vm, VmExit, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

const DATA: u64 = 0x40000;
const SENTINEL: u64 = 0xdead_0000;

/// `mov r9, DATA; mov eax, 0x11; mov r10d, 0x22; lock cmpxchg [r9], r10d;
/// jmp [rip]` with the sentinel as the jump's target.
fn program() -> Vec<u8> {
    let mut code = vec![
        0x49, 0xc7, 0xc1, 0x00, 0x00, 0x04, 0x00, // mov r9, 0x40000
        0xb8, 0x11, 0x00, 0x00, 0x00, // mov eax, 0x11
        0x41, 0xba, 0x22, 0x00, 0x00, 0x00, // mov r10d, 0x22
        0xf0, 0x45, 0x0f, 0xb1, 0x11, // lock cmpxchg [r9], r10d
        0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmp [rip + 0]
    ];
    code.extend_from_slice(&SENTINEL.to_le_bytes());
    code
}

fn run(jit: bool, in_memory: u32) -> (VmExit, u32, Option<u64>) {
    let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
    let ctx = source.new_context();
    let mut memory = VmMemory::new();
    memory
        .mmu
        .write_unchecked(0x1000, &program(), perm::RW_INIT | perm::EXEC);
    memory
        .mmu
        .write_unchecked(DATA, &in_memory.to_le_bytes(), perm::RW_INIT);
    let mut vm = Vm::at_address(ctx, 0x1000, source, memory).expect("the entry decodes");
    if jit {
        vm.set_block_executor(Box::new(Jit::new()));
    }
    let exit = vm.run(100_000);
    let mut after = [0u8; 4];
    vm.memory_mut().mmu.read(DATA, &mut after).unwrap();
    let ctx = vm.context().clone();
    let rax = vm.emulator().read_varnode_by_name(&ctx, "RAX");
    (exit, u32::from_le_bytes(after), rax)
}

fn check(jit: bool) {
    // Equal: the source is written, EAX keeps the old value.
    let (exit, memory, rax) = run(jit, 0x11);
    assert!(
        matches!(exit, VmExit::Unlifted { addr, .. } if addr == SENTINEL),
        "jit={jit}: the program did not run to its end: {exit:?}"
    );
    assert_eq!(memory, 0x22, "jit={jit}: the exchange did not happen");
    assert_eq!(
        rax,
        Some(0x11),
        "jit={jit}: EAX changed on a successful exchange"
    );

    // Different: memory is untouched and EAX takes what was there.
    let (exit, memory, rax) = run(jit, 0x33);
    assert!(
        matches!(exit, VmExit::Unlifted { addr, .. } if addr == SENTINEL),
        "jit={jit}: the program did not run to its end: {exit:?}"
    );
    assert_eq!(memory, 0x33, "jit={jit}: a failed exchange wrote memory");
    assert_eq!(
        rax,
        Some(0x33),
        "jit={jit}: EAX did not take the memory value"
    );
}

#[test]
fn a_temporary_read_after_the_branch_survives_cleanup_interpreted() {
    check(false);
}

#[test]
fn a_temporary_read_after_the_branch_survives_cleanup_jitted() {
    check(true);
}
