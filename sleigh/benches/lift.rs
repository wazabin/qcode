//! The lifting hot path: instruction bytes → decoded instruction → QCode.
//!
//! Three stages of the same pipeline are timed on each instruction of a
//! fixed x86-64 corpus, so the cost of a stage is the difference between
//! neighbouring groups:
//!
//! - `decode`: [`FixedDecoder::decode`] alone, the sleigh side.
//! - `scratch`: plus a [`ScratchSession`] lift, the steady-state per-offset
//!   path — the session's store is reused across iterations, so this is what
//!   an isolated lift costs once the arenas are warm.
//! - `session`: plus a [`LiftSession`] lift into a fresh owned context, the
//!   persistent path — the context is made and dropped outside the timed
//!   region, so the difference from `scratch` is the cost of construction
//!   into a context that keeps the instruction.
//!
//! The corpus runs from the cheapest lowering to the busiest. The SSE and x87
//! forms are there because the x64 specification's per-lane macros make
//! them emit hundreds of p-code operations, so they weigh the per-instruction
//! cost of emission where the integer forms weigh the fixed cost of a lift.
//!
//! ```sh
//! cargo bench -p wazabin-qcode-sleigh --bench lift
//! cargo bench -p wazabin-qcode-sleigh --bench lift -- 'scratch/'        # one stage
//! cargo bench -p wazabin-qcode-sleigh --bench lift -- addps             # one instruction
//! cargo bench -p wazabin-qcode-sleigh --bench lift -- --save-baseline before
//! cargo bench -p wazabin-qcode-sleigh --bench lift -- --baseline before # compare a change
//! ```
//!
//! Timing on a loaded machine swings by several percent between runs; the
//! allocation and retained-memory counts in `tests/scratch_memory.rs` are the
//! deterministic companion when a change is about churn rather than work.

use std::{hint::black_box, time::Duration};

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use wazabin_qcode_sleigh::{
    SleighLifter,
    decode::FixedDecoder,
    session::{Host, LiftSession, ScratchSession},
};

/// Address every case is lifted at. The relative branches and the
/// `rip`-relative `lea` resolve against it.
const ADDRESS: u64 = 0x1000;

struct Case {
    name: &'static str,
    bytes: &'static [u8],
}

/// x86-64 instructions in long mode, from the cheapest lowering to the
/// busiest.
const CASES: &[Case] = &[
    Case {
        name: "nop",
        bytes: b"\x90",
    },
    Case {
        name: "push_rbp",
        bytes: b"\x55",
    },
    Case {
        name: "mov_rax_imm32",
        bytes: b"\x48\xc7\xc0\x78\x56\x34\x12",
    },
    Case {
        name: "mov_rcx_mem_rdx",
        bytes: b"\x48\x8b\x0a",
    },
    Case {
        name: "add_rax_rcx",
        bytes: b"\x48\x01\xc8",
    },
    Case {
        name: "lea_r11_rip",
        bytes: b"\x4c\x8d\x1d\x20\x00\x00\x00",
    },
    Case {
        name: "jz_rel8",
        bytes: b"\x74\x05",
    },
    Case {
        name: "call_rel32",
        bytes: b"\xe8\x10\x00\x00\x00",
    },
    Case {
        name: "ret",
        bytes: b"\xc3",
    },
    Case {
        name: "cmpxchg_rax_rcx",
        bytes: b"\x48\x0f\xb1\xc8",
    },
    Case {
        name: "rep_movsb",
        bytes: b"\xf3\xa4",
    },
    Case {
        name: "add_mem_imm32",
        bytes: b"\x48\x81\x04\x25\x10\x20\x30\x00\x05\x00\x00\x00",
    },
    Case {
        name: "mulsd_xmm0_mem",
        bytes: b"\xf2\x0f\x59\x04\x25\x10\x20\x30\x00",
    },
    Case {
        name: "fld_mem",
        bytes: b"\xd9\x05\x10\x20\x30\x00",
    },
    Case {
        name: "fmulp",
        bytes: b"\xde\xc9",
    },
    Case {
        name: "addps_xmm0_xmm1",
        bytes: b"\x0f\x58\xc1",
    },
];

fn group<'a>(
    c: &'a mut Criterion,
    name: &str,
) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group(name);
    group
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    group
}

fn bench_decode(c: &mut Criterion) {
    let spec = sleigh_precompile::x64::spec();
    let decoder = FixedDecoder::new(spec);
    let mut group = group(c, "decode");
    for case in CASES {
        group.bench_with_input(BenchmarkId::from_parameter(case.name), case, |b, case| {
            b.iter(|| {
                let instruction = decoder
                    .decode(ADDRESS, case.bytes)
                    .unwrap_or_else(|error| panic!("{}: {error}", case.name));
                black_box(instruction.len())
            })
        });
    }
    group.finish();
}

fn bench_scratch(c: &mut Criterion) {
    let spec = sleigh_precompile::x64::spec();
    let lifter = SleighLifter::new(spec);
    let mut session = ScratchSession::new(&lifter);
    let mut group = group(c, "scratch");
    for case in CASES {
        group.bench_with_input(BenchmarkId::from_parameter(case.name), case, |b, case| {
            b.iter(|| {
                let lifted = session
                    .lift(ADDRESS, case.bytes)
                    .unwrap_or_else(|error| panic!("{}: {error}", case.name));
                // Read it as a consumer would, so nothing is optimized away.
                black_box((lifted.exits().count(), lifted.blocks().count()))
            })
        });
    }
    group.finish();
}

fn bench_session(c: &mut Criterion) {
    let spec = sleigh_precompile::x64::spec();
    let lifter = SleighLifter::new(spec);
    let mut group = group(c, "session");
    for case in CASES {
        group.bench_with_input(BenchmarkId::from_parameter(case.name), case, |b, case| {
            b.iter_batched(
                || LiftSession::new(&lifter, Host::Anonymous),
                |mut session| {
                    let lifted = session
                        .lift(ADDRESS, case.bytes)
                        .unwrap_or_else(|error| panic!("{}: {error}", case.name));
                    // The session goes back out so its context is dropped
                    // outside the timed region, as it was made outside it.
                    (black_box(lifted.blocks().len()), session)
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_decode, bench_scratch, bench_session);
criterion_main!(benches);
