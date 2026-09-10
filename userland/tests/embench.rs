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
use qcode_userland::bare;

mod support;

struct Run {
    verified: bool,
    exit: String,
    steps: u64,
    native_bodies: u64,
    elapsed: std::time::Duration,
}

fn run(image: &[u8], jit: bool, budget: u64) -> Run {
    let mut vm = bare::machine(image).expect("the image loads and its entry decodes");
    if jit {
        vm.set_block_executor(Box::new(Jit::new()));
    }

    let started = std::time::Instant::now();
    let exit = vm.run(budget);
    let elapsed = started.elapsed();
    // A run is only meaningful if it ran to the end: `main` returned onto the
    // sentinel. Any other exit — a fault, an unlifted instruction, the budget —
    // means the benchmark did not finish, and its result register is noise.
    let finished = bare::returned(&exit);
    Run {
        exit: format!("{exit:?}"),
        // Embench's `main` returns 0 when the benchmark verified itself.
        verified: finished && bare::register(&mut vm, "EAX") == Some(0),
        steps: vm.stats.steps,
        native_bodies: vm.stats.native_bodies,
        elapsed,
    }
}

/// Every benchmark must verify its own result, with and without the JIT.
///
/// Ignored by default: it needs the images, which `benchmarks/embench/build.sh`
/// produces from an Embench checkout.
#[test]
#[ignore = "needs benchmarks/embench/build.sh to have been run"]
fn embench_verifies_under_both_strategies() {
    let images = support::images();
    assert!(
        !images.is_empty(),
        "no images; run benchmarks/embench/build.sh"
    );

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
