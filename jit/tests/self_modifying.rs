//! Self-modifying code: a store into a page that holds lifted code must be
//! visible to the next execution of that code, interpreted or compiled.

use qcode_jit::Jit;
use qcode_vm::{Vm, VmExit, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

fn machine(code: &[u8]) -> Vm<SleighCodeSource<'static>> {
    let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
    let ctx = source.new_context();
    let mut memory = VmMemory::new();
    memory
        .mmu
        .write_unchecked(0x1000, code, perm::RW_INIT | perm::EXEC);
    Vm::at_address(ctx, 0x1000, source, memory).expect("the entry decodes")
}

/// A loop that bumps the immediate of its own `mov ebx, imm` each pass, so
/// `eax` sums 1 + 2 + 3 only if every pass runs the freshly written bytes.
///
/// The patch is a store from inside the loop body into the same page, and
/// on the third pass the body has been entered at its start twice already,
/// which is when a JIT compiles it — so the store that must be seen is one
/// compiled code performs.
const PATCHER: &[u8] = &[
    0xb9, 0x03, 0x00, 0x00, 0x00, // 1000: mov ecx, 3
    0xbb, 0x01, 0x00, 0x00, 0x00, // 1005: mov ebx, 1         (imm at 1006)
    0x80, 0x05, 0xf5, 0xff, 0xff, 0xff, 0x01, // 100a: add byte [rip-0xb], 1  (-> 1006)
    0x01, 0xd8, // 1011: add eax, ebx
    0xff, 0xc9, // 1013: dec ecx
    0x75, 0xee, // 1015: jnz 1005
    0xeb, 0xfe, // 1017: jmp $
];

/// Runs the patcher to its final `jmp $`, reporting `eax`, how many blocks
/// were thrown away for being written over, and how many ran natively.
fn run(jit: bool) -> (Option<u64>, u64, u64) {
    let mut vm = machine(PATCHER);
    if jit {
        vm.set_block_executor(Box::new(Jit::new()));
    }
    vm.add_breakpoint(0x1017);
    let exit = vm.run(2000);
    assert!(
        matches!(exit, VmExit::Breakpoint(0x1017)),
        "jit={jit}: {exit:?}"
    );
    let ctx = vm.context().clone();
    let eax = vm.emulator().read_varnode_by_name(&ctx, "EAX");
    (eax, vm.stats.evicted, vm.stats.native_bodies)
}

#[test]
fn a_store_into_lifted_code_is_seen_by_the_interpreter() {
    let (eax, evicted, _) = run(false);
    assert_eq!(eax, Some(6));
    assert!(
        evicted > 0,
        "nothing was evicted, so the test never patched lifted code"
    );
}

#[test]
fn a_store_into_lifted_code_is_seen_by_the_jit() {
    let (eax, evicted, native) = run(true);
    assert_eq!(eax, Some(6));
    assert!(
        evicted > 0,
        "nothing was evicted, so the test never patched lifted code"
    );
    assert!(
        native > 0,
        "no block ran natively; the comparison was vacuous"
    );
}
