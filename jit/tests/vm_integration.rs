//! Running a whole program through the VM with the JIT installed must give the
//! same answer as running it on the interpreter alone.

use qcode_jit::Jit;
use qcode_vm::{Vm, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

/// A JIT that compiles on first sight: the programs here run once, and the
/// point is that their blocks ran natively.
fn eager() -> Jit {
    let mut jit = Jit::new();
    jit.set_warm_up(1);
    jit
}

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

/// A program, its bytes, and what the watched registers must hold after it.
type Program = (&'static str, &'static [u8], &'static [(&'static str, u64)]);

/// The registers every program in this file is checked on.
const WATCHED: [&str; 11] = [
    "RAX", "RBX", "RCX", "EAX", "EBX", "ECX", "CF", "ZF", "SF", "OF", "PF",
];

/// Runs `code`, optionally with the JIT installed, and reports the watched
/// registers plus how much the JIT took on.
fn run(code: &[u8], jit: bool, budget: u64) -> (Vec<Option<u64>>, u64) {
    let mut vm = machine(code);
    if jit {
        vm.set_block_executor(Box::new(eager()));
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
    let programs: [Program; 3] = [
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
            vm.set_block_executor(Box::new(eager()));
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

/// A user op the interpreter cannot run — `rdtsc` — stops both strategies at
/// the same place, and the compiled prefix of its block is still native code.
#[test]
fn an_intrinsic_interrupt_stops_the_jit_at_the_same_place_as_the_interpreter() {
    use qcode_vm::{InterruptKind, VmExit};

    let code: &[u8] = &[
        0xb8, 0x05, 0x00, 0x00, 0x00, // mov eax, 5
        0x0f, 0x31, // rdtsc
        0x89, 0xc3, // mov ebx, eax
        0x89, 0xd1, // mov ecx, edx
    ];
    let tsc: u128 = 0x1122_3344_5566_7788;

    let mut stops = Vec::new();
    let mut results = Vec::new();
    for jit in [false, true] {
        let mut vm = machine(code);
        if jit {
            vm.set_block_executor(Box::new(eager()));
        }
        let exit = vm.run(10_000);
        let VmExit::Interrupt(interrupt) = exit else {
            panic!("jit={jit}: expected an interrupt, got {exit:?}");
        };
        assert!(
            matches!(&interrupt.kind, InterruptKind::Intrinsic { name, .. } if name.as_ref() == "rdtsc"),
            "jit={jit}: stopped at {:?}",
            interrupt.kind
        );
        assert_eq!(interrupt.pc, Some(0x1005), "jit={jit}");
        assert_eq!(interrupt.size, 8, "jit={jit}: rdtsc yields a 64-bit value");
        stops.push((interrupt.insn, vm.emulator().block, vm.emulator().idx));
        let native_before = vm.stats.native_bodies;
        let idx_at_stop = vm.emulator().idx;

        vm.resume(Some(tsc)).unwrap();
        vm.run(10_000);
        if jit {
            assert!(
                idx_at_stop > 0,
                "the interrupt should sit after a compiled prefix in its block"
            );
            assert!(
                vm.stats.native_bodies > native_before,
                "the rest of the block after rdtsc should have run natively"
            );
        }
        let ctx = vm.context().clone();
        let read = |vm: &mut Vm<_>, name: &str| vm.emulator().read_varnode_by_name(&ctx, name);
        results.push((
            read(&mut vm, "EAX"),
            read(&mut vm, "EBX"),
            read(&mut vm, "ECX"),
        ));
        if jit {
            assert!(
                vm.stats.native_bodies > 0,
                "the prefix before rdtsc should have run natively"
            );
        }
    }
    assert_eq!(
        stops[0], stops[1],
        "both strategies stop at the same instruction"
    );
    assert_eq!(
        results[0],
        (Some(0x5566_7788), Some(0x5566_7788), Some(0x1122_3344))
    );
    assert_eq!(results[0], results[1]);
}

/// A loop compiled whole never leaves native code on its own; the run's
/// budget has to be what brings it back.
#[test]
fn the_budget_stops_a_loop_that_stays_in_compiled_code() {
    // `l: dec ecx; jmp l`
    let mut vm = machine(&[0xff, 0xc9, 0xeb, 0xfc]);
    vm.set_block_executor(Box::new(eager()));
    let exit = vm.run(10_000);
    assert!(
        matches!(exit, qcode_vm::VmExit::InstructionLimit),
        "expected the budget to end the run, got {exit:?}"
    );
    // Overshoot is at most the block the budget ran out in.
    assert!(
        vm.stats.steps >= 10_000 && vm.stats.steps < 10_100,
        "{}",
        vm.stats.steps
    );
}

/// A return is an indirect branch on the popped address, and a chain follows
/// it into the caller's compiled code instead of handing back after every
/// call. The interpreter is never involved once everything is compiled, so
/// only the first entry into each block is a native entry the VM counts.
#[test]
fn a_chain_follows_returns_and_indirect_calls() {
    let code: &[u8] = &[
        0xb9, 0xe8, 0x03, 0x00, 0x00, // 1000: mov ecx, 1000
        0x48, 0x8d, 0x05, 0x0b, 0x00, 0x00, 0x00, // 1005: lea rax, [rip+0xb] ; f
        0xff, 0xd0, // 100c: call rax
        0xff, 0xc9, // 100e: dec ecx
        0x75, 0xfa, // 1010: jnz 100c
        0xe9, 0xe9, 0x0f, 0x00, 0x00, // 1012: jmp 2000 ; unmapped: ends the run
        0xff, 0xc3, // 1017: f: inc ebx
        0xc3, // 1019: ret
    ];
    let mut vm = machine(code);
    let ctx = vm.context().clone();
    vm.emulator()
        .set_varnode_by_name(&ctx, "RSP", 0x21000)
        .unwrap();
    vm.set_block_executor(Box::new(eager()));
    let exit = vm.run(u64::MAX);
    assert!(
        matches!(exit, qcode_vm::VmExit::Unlifted { addr: 0x2000, .. }),
        "{exit:?}"
    );
    assert_eq!(vm.emulator().read_varnode_by_name(&ctx, "EBX"), Some(1000));
    assert_eq!(vm.emulator().read_varnode_by_name(&ctx, "ECX"), Some(0));
    // A handful of blocks, each discovered once; the thousand calls and
    // returns in between never left native code.
    assert!(
        vm.stats.native_bodies < 20,
        "{} native entries",
        vm.stats.native_bodies
    );
}

/// `step` means one block at most, so stepping a loop that stays in compiled
/// code comes back after every pass round it.
#[test]
fn a_step_never_chains_past_the_block_it_starts() {
    // `l: dec ecx; jmp l`
    let mut vm = machine(&[0xff, 0xc9, 0xeb, 0xfc]);
    vm.set_block_executor(Box::new(eager()));
    for _ in 0..100 {
        assert!(vm.step().is_none());
    }
    let ctx = vm.context().clone();
    // One decrement per block, and at most one block per step; the first
    // steps went on discovery rather than running anything.
    let ecx = vm.emulator().read_varnode_by_name(&ctx, "ECX").unwrap();
    let passes = ecx.wrapping_neg() & 0xffff_ffff;
    assert!((1..=100).contains(&passes), "{passes} passes in 100 steps");
}

/// A block that is a terminator and nothing else retires no operation when
/// it runs, so any positive allowance would still let a chain through it
/// into the next block. A step must therefore hand the executor the
/// allowance that forbids chaining outright, which is `0`; `run` hands it
/// what is left of its budget.
#[test]
fn a_step_forbids_chaining_where_a_run_rations_it() {
    use qcode_vm::BlockExecutor;
    use std::{cell::RefCell, rc::Rc};

    /// Records the allowance of every call and runs nothing itself.
    struct Recorder(Rc<RefCell<Vec<u64>>>);
    impl BlockExecutor for Recorder {
        fn run_block(
            &mut self,
            _: &qcode::context::Context<'_>,
            _: &mut qcode_emulator::StandaloneEmulator<VmMemory>,
            _: qcode::value::BlockId,
            _: usize,
            chain: u64,
        ) -> Result<Option<qcode_vm::Executed>, qcode_emulator::EmulatorErrorKind> {
            self.0.borrow_mut().push(chain);
            Ok(None)
        }
    }

    // `l: dec ecx; jmp l`
    let mut vm = machine(&[0xff, 0xc9, 0xeb, 0xfc]);
    let allowances = Rc::new(RefCell::new(Vec::new()));
    vm.set_block_executor(Box::new(Recorder(allowances.clone())));

    for _ in 0..50 {
        assert!(vm.step().is_none());
    }
    let stepped = allowances.borrow_mut().drain(..).collect::<Vec<_>>();
    assert!(
        !stepped.is_empty(),
        "the executor was never offered a block"
    );
    assert!(stepped.iter().all(|&chain| chain == 0), "{stepped:?}");

    assert!(matches!(vm.run(1000), qcode_vm::VmExit::InstructionLimit));
    let ran = allowances.borrow();
    assert!(!ran.is_empty(), "the executor was never offered a block");
    assert!(
        ran.iter().all(|&chain| (1..=1000).contains(&chain)),
        "{ran:?}"
    );
}
