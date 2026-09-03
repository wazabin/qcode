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

/// The registers every program in this file is checked on.
const WATCHED: [&str; 11] = [
    "RAX", "RBX", "RCX", "EAX", "EBX", "ECX", "CF", "ZF", "SF", "OF", "PF",
];

/// Runs `code`, optionally with the JIT installed, and reports the watched
/// registers plus how much the JIT took on.
fn run(code: &[u8], jit: bool, budget: u64) -> (Vec<Option<u64>>, u64) {
    let mut vm = machine(code);
    if jit {
        vm.set_block_executor(Box::new(Jit::new()));
    }
    vm.run(budget);
    let ctx = vm.context().clone();
    let state = WATCHED
        .iter()
        .map(|name| vm.emulator().read_varnode_by_name(&ctx, name))
        .collect();
    (state, vm.stats.native_bodies)
}

#[test]
fn the_jit_does_not_change_what_a_program_computes() {
    let programs: [(&str, &[u8], &[(&str, u64)]); 3] = [
        (
            "arithmetic",
            &[
                0xb8, 0x39, 0x05, 0x00, 0x00, // mov eax, 1337
                0xbb, 0x07, 0x00, 0x00, 0x00, // mov ebx, 7
                0x01, 0xd8, // add eax, ebx
                0x29, 0xd8, // sub eax, ebx
                0x31, 0xd8, // xor eax, ebx
            ],
            &[("EAX", 1337 ^ 7), ("EBX", 7)],
        ),
        (
            "countdown loop",
            &[
                0xb9, 0xd0, 0x07, 0x00, 0x00, // mov ecx, 2000
                0xff, 0xc9, // dec ecx
                0x75, 0xfc, // jnz -4
            ],
            &[("ECX", 0)],
        ),
        (
            "logical ops writing undefined flags",
            &[
                0xb8, 0xff, 0x00, 0x00, 0x00, // mov eax, 255
                0x21, 0xd8, // and eax, ebx
                0x09, 0xd8, // or  eax, ebx
            ],
            &[("EAX", 0)],
        ),
    ];

    for (name, code, expect) in programs {
        let (interpreted, _) = run(code, false, 200_000);
        let (jitted, native) = run(code, true, 200_000);
        // Checked against a stated answer, not only against each other: an
        // equivalence test alone passes happily when both strategies are broken
        // the same way, which is exactly what a shared bug in the machinery
        // underneath them looks like.
        for &(register, want) in expect {
            let index = WATCHED
                .iter()
                .position(|&name| name == register)
                .expect("the expectation names a watched register");
            assert_eq!(
                interpreted[index],
                Some(want),
                "program `{name}` interpreted {register} wrongly"
            );
        }
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

/// Throughput with and without the JIT, on the same loop the interpreter
/// benchmarks use. Ignored by default: it runs for seconds.
#[test]
#[ignore = "long-running throughput benchmark"]
fn jit_throughput() {
    // mov ecx, 1000000 ; loop: dec ecx ; jnz loop
    let code: &[u8] = &[
        0xb9, 0x40, 0x42, 0x0f, 0x00, // mov ecx, 1000000
        0xff, 0xc9, // dec ecx
        0x75, 0xfc, // jnz -4
    ];
    let instructions = 1_000_000u64 * 2 + 1;

    for jit in [false, true] {
        let mut vm = machine(code);
        if jit {
            vm.set_block_executor(Box::new(Jit::new()));
        }
        let start = std::time::Instant::now();
        vm.run(u64::MAX);
        let elapsed = start.elapsed();
        let ctx = vm.context().clone();
        assert_eq!(
            vm.emulator().read_varnode_by_name(&ctx, "ECX"),
            Some(0),
            "the loop must run to completion"
        );
        eprintln!(
            "jit={jit}: {instructions} insns in {elapsed:?} ({:.2}M guest-insn/s) \
             native_bodies={}",
            instructions as f64 / elapsed.as_secs_f64() / 1e6,
            vm.stats.native_bodies,
        );
    }
}
