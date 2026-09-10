//! An indirect branch to an address a block absorbed.
//!
//! Straight-line discovery folds a callee into the block that called it
//! directly, and the index then answers the callee's address with that
//! block. A later `call *%rax` to the same address must not enter that
//! block at its start and rerun the caller's instructions: it has to break
//! the block apart and enter at the callee.
//!
//! busybox `sort` crashed on this: `__stdio_write` was absorbed into the
//! block that called it from `__stdout_write`, and `fflush` calling it
//! through `f->write` landed on the register shuffle before the call with
//! its own registers, so `__stdio_write` got `f == 1`.

use qcode_jit::Jit;
use qcode_vm::{Vm, VmExit, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

const DATA: u64 = 0x40000;
const SENTINEL: u64 = 0xdead_0000;
const STACK: u64 = 0x7fff_0000;
const STACK_TOP: u64 = 0x7fff_8000;

/// ```text
/// 1000: mov r8, DATA
/// 1007: inc dword [r8 + 4]      ; runs once
/// 100b: lea rax, [f]
/// 1012: call f                  ; direct: f gets absorbed here
/// 1017: call rax                ; indirect, to the absorbed address
/// 1019: jmp [rip]               ; to the sentinel
/// 1030: f: mov dword [r8], 0x22
/// 1037:    inc dword [r8 + 8]   ; runs twice
/// 103b:    ret
/// ```
fn program() -> Vec<u8> {
    let mut code = vec![
        0x49, 0xc7, 0xc0, 0x00, 0x00, 0x04, 0x00, // mov r8, 0x40000
        0x41, 0xff, 0x40, 0x04, // inc dword [r8 + 4]
        0x48, 0x8d, 0x05, 0x1e, 0x00, 0x00, 0x00, // lea rax, [rip + 0x1e] = 0x1030
        0xe8, 0x19, 0x00, 0x00, 0x00, // call 0x1030
        0xff, 0xd0, // call rax
        0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmp [rip + 0]
    ];
    code.extend_from_slice(&SENTINEL.to_le_bytes());
    code.resize(0x30, 0x90);
    code.extend_from_slice(&[
        0x41, 0xc7, 0x00, 0x22, 0x00, 0x00, 0x00, // mov dword [r8], 0x22
        0x41, 0xff, 0x40, 0x08, // inc dword [r8 + 8]
        0xc3, // ret
    ]);
    code
}

fn run(jit: bool) -> (VmExit, [u32; 3]) {
    let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
    let ctx = source.new_context();
    let mut memory = VmMemory::new();
    memory
        .mmu
        .write_unchecked(0x1000, &program(), perm::RW_INIT | perm::EXEC);
    memory.mmu.write_unchecked(DATA, &[0u8; 12], perm::RW_INIT);
    memory.mmu.map(STACK, 0x40000, perm::RW_INIT).unwrap();
    let mut vm = Vm::at_address(ctx, 0x1000, source, memory).expect("the entry decodes");
    let ctx = vm.context().clone();
    vm.emulator()
        .set_varnode_by_name(&ctx, "RSP", STACK_TOP)
        .unwrap();
    if jit {
        vm.set_block_executor(Box::new(Jit::new()));
    }
    let exit = vm.run(10_000);
    let mut bytes = [0u8; 12];
    vm.memory_mut().mmu.read(DATA, &mut bytes).unwrap();
    let word = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    (exit, [word(0), word(1), word(2)])
}

fn check(jit: bool) {
    let (exit, [stored, prologue, callee]) = run(jit);
    assert!(
        matches!(exit, VmExit::Unlifted { addr, .. } if addr == SENTINEL),
        "jit={jit}: the program did not run to its end: {exit:?}"
    );
    assert_eq!(stored, 0x22, "jit={jit}");
    assert_eq!(callee, 2, "jit={jit}: the callee must run once per call");
    assert_eq!(
        prologue, 1,
        "jit={jit}: the indirect call reran the caller's instructions"
    );
}

#[test]
fn an_indirect_call_into_an_absorbed_block_enters_at_the_callee_interpreted() {
    check(false);
}

#[test]
fn an_indirect_call_into_an_absorbed_block_enters_at_the_callee_jitted() {
    check(true);
}
