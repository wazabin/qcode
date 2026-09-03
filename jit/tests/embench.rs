//! Embench-IoT under the VM, as a benchmark the JIT can be judged on.
//!
//! The countdown loop the other benchmark runs is a poor proxy for compiled
//! code: it touches no memory, so it never meets the case this backend
//! declines. These are real C benchmarks — 17 of Embench's 19, built
//! freestanding for x86-64 by `benchmarks/embench/build.sh`.
//!
//! Each image is entered at `main` with a stack and a sentinel return address,
//! so returning from `main` faults on the sentinel and ends the run. Embench's
//! `main` returns 0 when the benchmark verified its own result, which is read
//! out of RAX — so every run is checked, not just timed. A run that stops for
//! any other reason is a failure whatever that register holds: an unfinished
//! benchmark leaves noise there, and treating that as a pass is how a benchmark
//! comes to agree with itself while measuring nothing.
//!
//! # This currently fails, on every benchmark
//!
//! `main` calls `initialise_board` and the machine stops with "function
//! FunctionId(1) has no root block": the VM cannot yet follow a `call` into a
//! function it has only just discovered. Nothing here reaches a benchmark body.
//!
//! That is the point of adding this. Every benchmark this repository had until
//! now was a single function — the countdown loop, and hand-written kernels
//! that `-O2` inlined flat — so a gap this large in ordinary compiled code was
//! invisible. The failure is not a regression; it reproduces unchanged on the
//! commit before any of the JIT work.

use qcode_jit::Jit;
use qcode_vm::{Vm, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

/// The address a returning `main` lands on: unmapped, so the run stops there.
const SENTINEL: u64 = 0xdead_0000;
const STACK: u64 = 0x7fff_0000;
const STACK_SIZE: u64 = 0x40000;
const STACK_TOP: u64 = 0x7fff_8000;

/// Enough of ELF64 to place a freestanding static image.
fn load(image: &[u8], memory: &mut VmMemory) -> u64 {
    let half = |o: usize| u16::from_le_bytes(image[o..o + 2].try_into().unwrap());
    let word = |o: usize| u32::from_le_bytes(image[o..o + 4].try_into().unwrap());
    let long = |o: usize| u64::from_le_bytes(image[o..o + 8].try_into().unwrap());

    let entry = long(24);
    let phoff = long(32) as usize;
    let phentsize = half(54) as usize;
    for i in 0..half(56) as usize {
        let p = phoff + i * phentsize;
        if word(p) != 1 {
            continue; // PT_LOAD only
        }
        let flags = word(p + 4);
        let off = long(p + 8) as usize;
        let vaddr = long(p + 16);
        let filesz = long(p + 32) as usize;
        let memsz = long(p + 40) as usize;

        let mut bits = perm::READ | perm::INIT;
        if flags & 1 != 0 {
            bits |= perm::EXEC;
        }
        if flags & 2 != 0 {
            bits |= perm::WRITE;
        }
        // A segment's memory size exceeds its file size exactly where .bss is,
        // which the loader is responsible for zeroing.
        let mut bytes = image[off..off + filesz].to_vec();
        bytes.resize(memsz, 0);
        memory.mmu.write_unchecked(vaddr, &bytes, bits);
    }
    entry
}

struct Run {
    verified: bool,
    exit: String,
    steps: u64,
    native_bodies: u64,
    elapsed: std::time::Duration,
}

fn run(image: &[u8], jit: bool, budget: u64) -> Run {
    let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
    let ctx = source.new_context();
    let mut memory = VmMemory::new();
    let entry = load(image, &mut memory);
    memory.mmu.map(STACK, STACK_SIZE, perm::RW_INIT).unwrap();
    memory
        .mmu
        .write_unchecked(STACK_TOP, &SENTINEL.to_le_bytes(), perm::RW_INIT);

    let mut vm = Vm::at_address(ctx, entry, source, memory).expect("the entry decodes");
    let ctx = vm.context().clone();
    vm.emulator()
        .set_varnode_by_name(&ctx, "RSP", STACK_TOP)
        .expect("RSP is a register in this specification");
    if jit {
        vm.set_block_executor(Box::new(Jit::new()));
    }

    let started = std::time::Instant::now();
    let exit = vm.run(budget);
    let elapsed = started.elapsed();
    let ctx = vm.context().clone();
    // A run is only meaningful if it ran to the end: `main` returned onto the
    // sentinel. Any other exit — a fault, an unlifted instruction, the budget —
    // means the benchmark did not finish, and its result register is noise.
    let finished = matches!(
        &exit,
        qcode_vm::VmExit::Unlifted { addr, .. } if *addr == SENTINEL
    );
    Run {
        exit: format!("{exit:?}"),
        // Embench's `main` returns 0 when the benchmark verified itself.
        verified: finished && vm.emulator().read_varnode_by_name(&ctx, "EAX") == Some(0),
        steps: vm.stats.steps,
        native_bodies: vm.stats.native_bodies,
        elapsed,
    }
}

fn images() -> Vec<(String, Vec<u8>)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target/embench");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<_> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "elf"))
        .map(|e| {
            (
                e.path().file_stem().unwrap().to_string_lossy().into_owned(),
                std::fs::read(e.path()).expect("image"),
            )
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Every benchmark must verify its own result, with and without the JIT.
///
/// Ignored by default: it needs the images, which `benchmarks/embench/build.sh`
/// produces from an Embench checkout.
#[test]
#[ignore = "needs benchmarks/embench/build.sh to have been run"]
fn embench_verifies_under_both_strategies() {
    let images = images();
    assert!(!images.is_empty(), "no images; run benchmarks/embench/build.sh");

    let mut failures = Vec::new();
    for (name, image) in &images {
        let interpreted = run(image, false, 4_000_000_000);
        let jitted = run(image, true, 4_000_000_000);
        let speedup = interpreted.elapsed.as_secs_f64() / jitted.elapsed.as_secs_f64();
        eprintln!(
            "{name:16} steps={:<11} interp={:>9.1?} jit={:>9.1?} {speedup:>5.2}x  native={:<9} {}",
            interpreted.steps,
            interpreted.elapsed,
            jitted.elapsed,
            jitted.native_bodies,
            match (interpreted.verified, jitted.verified) {
                (true, true) => "ok".to_owned(),
                // Which side is wrong is the whole triage: a benchmark the
                // interpreter also gets wrong is a semantics bug, while one only
                // the JIT gets wrong is a miscompilation.
                (true, false) => format!("JIT WRONG (interpreter ok) {}", jitted.exit),
                (false, true) => format!("INTERPRETER WRONG {}", interpreted.exit),
                (false, false) => format!("BOTH WRONG {}", interpreted.exit),
            }
        );
        if !interpreted.verified {
            failures.push(format!("{name} did not verify on the interpreter"));
        }
        if !jitted.verified {
            failures.push(format!("{name} did not verify with the JIT"));
        }
        if interpreted.steps != jitted.steps {
            failures.push(format!(
                "{name} retired {} operations interpreted but {} jitted",
                interpreted.steps, jitted.steps
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
