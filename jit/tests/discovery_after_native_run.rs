//! Two discoveries in one step.
//!
//! A direct branch into code that is not lifted yet lands in an empty
//! placeholder block, which the VM fills and then retries the step. With an
//! executor installed the retry runs the whole body of the new block
//! natively and leaves the interpreter its terminator, and when that is an
//! indirect branch to code that is not lifted either, the step needs a
//! second discovery. The interpreter alone never needs two: it runs one
//! operation per attempt, so the terminator comes in a later step.
//!
//! busybox `sh` crashed on this under the JIT at a `call *%rax` into a
//! function no earlier path had lifted.

use qcode_jit::Jit;
use qcode_vm::{Vm, VmExit, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

const SENTINEL: u64 = 0xdead_0000;

/// `jmp next; next: jmp [rip]` with the sentinel as the jump's target.
fn program() -> Vec<u8> {
    let mut code = vec![
        0xeb, 0x00, // jmp next
        0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // next: jmp [rip + 0]
    ];
    code.extend_from_slice(&SENTINEL.to_le_bytes());
    code
}

fn run(jit: bool) -> VmExit {
    let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
    let ctx = source.new_context();
    let mut memory = VmMemory::new();
    memory
        .mmu
        .write_unchecked(0x1000, &program(), perm::RW_INIT | perm::EXEC);
    let mut vm = Vm::at_address(ctx, 0x1000, source, memory).expect("the entry decodes");
    if jit {
        vm.set_block_executor(Box::new(Jit::new()));
    }
    vm.run(1_000)
}

#[test]
fn an_indirect_branch_right_after_a_discovery_is_discovered_too_interpreted() {
    let exit = run(false);
    assert!(
        matches!(exit, VmExit::Unlifted { addr, .. } if addr == SENTINEL),
        "{exit:?}"
    );
}

#[test]
fn an_indirect_branch_right_after_a_discovery_is_discovered_too_jitted() {
    let exit = run(true);
    assert!(
        matches!(exit, VmExit::Unlifted { addr, .. } if addr == SENTINEL),
        "{exit:?}"
    );
}
