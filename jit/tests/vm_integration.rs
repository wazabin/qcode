//! Running a whole program through the VM with the JIT installed must give the
//! same answer as running it on the interpreter alone.

use qcode_jit::Jit;
use qcode_vm::{Vm, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

fn machine(code: &[u8]) -> Vm<SleighCodeSource<'static>> {
    let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
    let ctx = source.new_context();
    let mut memory = VmMemory::new();
    memory
        .mmu
        .write_unchecked(0x1000, code, perm::READ | perm::EXEC);
    memory.mmu.map(0x20000, 0x2000, perm::RW_INIT).unwrap();
    Vm::at_address(ctx, 0x1000, source, memory).expect("the entry decodes")
}

/// Runs `code`, optionally with the JIT installed, and reports the watched
/// registers plus how much the JIT took on.
fn run(code: &[u8], jit: bool, budget: u64) -> (Vec<Option<u64>>, u64) {
    let mut vm = machine(code);
    if jit {
        vm.set_block_executor(Box::new(Jit::new()));
    }
    vm.run(budget);
    let ctx = vm.context().clone();
    let state = ["RAX", "RBX", "RCX", "EAX", "EBX", "ECX", "CF", "ZF", "SF", "OF", "PF"]
        .iter()
        .map(|name| vm.emulator().read_varnode_by_name(&ctx, name))
        .collect();
    (state, vm.stats.native_bodies)
}

#[test]
fn the_jit_does_not_change_what_a_program_computes() {
    let programs: [(&str, &[u8]); 3] = [
        (
            "arithmetic",
            &[
                0xb8, 0x39, 0x05, 0x00, 0x00, // mov eax, 1337
                0xbb, 0x07, 0x00, 0x00, 0x00, // mov ebx, 7
                0x01, 0xd8, // add eax, ebx
                0x29, 0xd8, // sub eax, ebx
                0x31, 0xd8, // xor eax, ebx
            ],
        ),
        (
            "countdown loop",
            &[
                0xb9, 0xd0, 0x07, 0x00, 0x00, // mov ecx, 2000
                0xff, 0xc9, // dec ecx
                0x75, 0xfc, // jnz -4
            ],
        ),
        (
            "logical ops writing undefined flags",
            &[
                0xb8, 0xff, 0x00, 0x00, 0x00, // mov eax, 255
                0x21, 0xd8, // and eax, ebx
                0x09, 0xd8, // or  eax, ebx
            ],
        ),
    ];

    for (name, code) in programs {
        let (interpreted, _) = run(code, false, 200_000);
        let (jitted, native) = run(code, true, 200_000);
        assert_eq!(
            interpreted, jitted,
            "program `{name}` computed a different result with the JIT installed"
        );
        assert!(
            native > 0,
            "program `{name}` ran no block natively; the comparison was vacuous"
        );
        eprintln!("{name}: identical, {native} block bodies ran natively");
    }
}

#[test]
fn an_uninstalled_jit_leaves_the_machine_on_the_interpreter() {
    let (_, native) = run(&[0xb8, 0x01, 0x00, 0x00, 0x00], false, 1000);
    assert_eq!(native, 0);
}
