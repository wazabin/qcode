//! The lifting hot path: instruction bytes → decoded instruction → QCode.
//!
//! Three stages of the same pipeline are measured on each instruction of a
//! fixed x86-64 corpus, so the cost of a stage is the difference between
//! neighbouring groups:
//!
//! - `decode`: [`FixedDecoder::decode`] alone, the sleigh side.
//! - `scratch`: plus a [`ScratchSession`] lift, the steady-state per-offset
//!   path — the session has lifted once before it is measured, so this is
//!   what an isolated lift costs once the arenas are warm.
//! - `session`: plus a [`LiftSession`] lift into a fresh owned context, the
//!   persistent path — the context is made in the setup and dropped after
//!   the measurement, so the difference from `scratch` is the cost of
//!   construction into a context that keeps the instruction.
//!
//! The corpus runs from the cheapest lowering to the busiest. The SSE and x87
//! forms are there because the x64 specification's per-lane macros make
//! them emit hundreds of p-code operations, so they weigh the per-instruction
//! cost of emission where the integer forms weigh the fixed cost of a lift.
//!
//! The measure is instructions retired under callgrind, not time: it is the
//! same on a loaded workstation and on a shared runner, so a change of a
//! percent is a change in the work done, and a baseline recorded once holds.
//! It needs valgrind and the runner matching the library's version:
//!
//! ```sh
//! cargo install iai-callgrind-runner --version "$(cargo pkgid iai-callgrind | cut -d@ -f2)"
//! cargo bench -p wazabin-qcode-sleigh --bench lift
//! cargo bench -p wazabin-qcode-sleigh --bench lift -- --save-baseline=before
//! cargo bench -p wazabin-qcode-sleigh --bench lift -- --baseline=before   # compare a change
//! ```
//!
//! The allocation and retained-memory counts in `tests/scratch_memory.rs`
//! are the companion when a change is about churn rather than work.

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use wazabin_qcode_sleigh::{
    SleighLifter,
    decode::FixedDecoder,
    session::{Host, LiftSession, ScratchSession},
};

/// Address every case is lifted at. The relative branches and the
/// `rip`-relative `lea` resolve against it.
const ADDRESS: u64 = 0x1000;

/// Each benchmark runs in a process of its own, so the lifter is made once
/// per process and leaked rather than kept in a `static`.
fn lifter() -> &'static SleighLifter<'static> {
    Box::leak(Box::new(SleighLifter::new(sleigh_precompile::x64::spec())))
}

fn decoder(bytes: &'static [u8]) -> (FixedDecoder<'static>, &'static [u8]) {
    (FixedDecoder::new(sleigh_precompile::x64::spec()), bytes)
}

fn warm_scratch(bytes: &'static [u8]) -> (ScratchSession<'static, 'static>, &'static [u8]) {
    let mut session = ScratchSession::new(lifter());
    session.lift(ADDRESS, bytes).expect("the case lifts");
    (session, bytes)
}

fn fresh_session(bytes: &'static [u8]) -> (LiftSession<'static, 'static>, &'static [u8]) {
    (LiftSession::new(lifter(), Host::Anonymous), bytes)
}

/// The three stages over the corpus: one benchmark per stage, one case per
/// instruction, named as the corpus names it.
macro_rules! corpus {
    ($($name:ident = $bytes:literal),* $(,)?) => {
        #[library_benchmark]
        $(#[bench::$name(args = ($bytes,), setup = decoder)])*
        fn decode((decoder, bytes): (FixedDecoder<'static>, &'static [u8])) -> usize {
            decoder.decode(ADDRESS, bytes).expect("the case decodes").len()
        }

        #[library_benchmark]
        $(#[bench::$name(args = ($bytes,), setup = warm_scratch)])*
        fn scratch(
            (mut session, bytes): (ScratchSession<'static, 'static>, &'static [u8]),
        ) -> (usize, usize, ScratchSession<'static, 'static>) {
            let lifted = session.lift(ADDRESS, bytes).expect("the case lifts");
            // Read it as a consumer would, so nothing is optimized away, and
            // hand the session back so its store is dropped outside the
            // measurement.
            let read = black_box((lifted.exits().count(), lifted.blocks().count()));
            (read.0, read.1, session)
        }

        #[library_benchmark]
        $(#[bench::$name(args = ($bytes,), setup = fresh_session)])*
        fn session(
            (mut session, bytes): (LiftSession<'static, 'static>, &'static [u8]),
        ) -> (usize, LiftSession<'static, 'static>) {
            let lifted = session.lift(ADDRESS, bytes).expect("the case lifts");
            // The session goes back out so its context is dropped outside
            // the measurement, as it was made outside it.
            (black_box(lifted.blocks().len()), session)
        }
    };
}

corpus! {
    nop = b"\x90",
    push_rbp = b"\x55",
    mov_rax_imm32 = b"\x48\xc7\xc0\x78\x56\x34\x12",
    mov_rcx_mem_rdx = b"\x48\x8b\x0a",
    add_rax_rcx = b"\x48\x01\xc8",
    lea_r11_rip = b"\x4c\x8d\x1d\x20\x00\x00\x00",
    jz_rel8 = b"\x74\x05",
    call_rel32 = b"\xe8\x10\x00\x00\x00",
    ret = b"\xc3",
    cmpxchg_rax_rcx = b"\x48\x0f\xb1\xc8",
    rep_movsb = b"\xf3\xa4",
    add_mem_imm32 = b"\x48\x81\x04\x25\x10\x20\x30\x00\x05\x00\x00\x00",
    mulsd_xmm0_mem = b"\xf2\x0f\x59\x04\x25\x10\x20\x30\x00",
    fld_mem = b"\xd9\x05\x10\x20\x30\x00",
    fmulp = b"\xde\xc9",
    addps_xmm0_xmm1 = b"\x0f\x58\xc1",
}

library_benchmark_group!(name = lift; benchmarks = decode, scratch, session);
main!(library_benchmark_groups = lift);
