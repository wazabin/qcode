//! Total retained memory of a scratch session over many lifts.
//!
//! The scratch store recycles its host's arenas and rebuilds its context past
//! a literal budget, and its unit tests check those counters. Counters are not
//! bytes: reverse-use lists, interned varnodes and bytes, name tables, local
//! label names, registries and every retained capacity are memory too, and
//! only the allocator sees all of them. This binary installs a counting
//! allocator and measures the bytes live between lifts across a corpus that
//! varies immediates, targets, addresses and sizes and exercises calls, `rep`
//! string operations, x87 and SSE — for at least 10,000 lifts and several
//! literal-budget rebuilds — and asserts that, past the first budget cycle,
//! the live footprint stops growing.
//!
//! ```sh
//! cargo test --release -p wazabin-qcode-sleigh --test scratch_memory -- --nocapture
//! ```

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

use wazabin_qcode_sleigh::{SleighLifter, session::ScratchSession};

/// Counts the bytes currently allocated through the global allocator.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            LIVE.fetch_add(new_size, Ordering::Relaxed);
        }
        p
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn live() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// The `i`th instruction of the corpus: one of sixteen shapes, with every
/// immediate, displacement, branch target and address derived from `i` so no
/// two rounds through the shapes repeat a constant.
fn instruction(i: u32) -> (u64, Vec<u8>) {
    let imm = i.wrapping_mul(0x9e37_79b9); // distinct 32-bit immediates
    let imm8 = (i % 251) as u8;
    let le = imm.to_le_bytes();
    let wide = (u64::from(imm) << 32 | u64::from(i)).to_le_bytes();
    // Addresses spread over a 4 MiB window so labels and fall-throughs differ.
    let address = 0x40_0000 + u64::from(i) * 7 % 0x40_0000;
    let bytes = match i % 16 {
        0 => vec![0x48, 0xc7, 0xc0, le[0], le[1], le[2], le[3]], // mov rax, imm32
        1 => vec![
            0x48, 0xb8, wide[0], wide[1], wide[2], wide[3], wide[4], wide[5], wide[6], wide[7],
        ], // movabs rax, imm64
        2 => vec![0x74, imm8],                                   // jz rel8
        3 => vec![0xe9, le[0], le[1], le[2], 0x00],              // jmp rel32
        4 => vec![0xe8, le[0], le[1], le[2], 0x00],              // call rel32
        5 => vec![0xff, 0xd0],                                   // call rax
        6 => vec![0xc3],                                         // ret
        7 => vec![0xf3, 0xa4],                                   // rep movsb
        8 => vec![0xf3, 0x48, 0xab],                             // rep stosq
        9 => vec![0xd9, 0x05, le[0], le[1], le[2], 0x00],        // fld dword [disp32]
        10 => vec![0xde, 0xc9],                                  // fmulp
        11 => vec![0x0f, 0x58, 0xc1],                            // addps xmm0, xmm1
        12 => vec![0xf2, 0x0f, 0x59, 0x04, 0x25, le[0], le[1], le[2], 0x00], // mulsd xmm0, [disp32]
        13 => vec![0x66, 0x0f, 0x6f, 0xc1],                      // movdqa xmm0, xmm1
        14 => vec![0x48, 0x8d, 0x80, le[0], le[1], le[2], le[3]], // lea rax, [rax+disp32]
        _ => vec![
            0x48, 0x81, 0x04, 0x25, le[0], le[1], le[2], 0x00, imm8, 0x00, 0x00, 0x00,
        ], // add qword [disp32], imm32
    };
    (address, bytes)
}

#[test]
fn total_retained_bytes_stop_growing_after_the_first_budget_cycle() {
    const LIFTS: u32 = 12_000;
    const CHECKPOINT: u32 = 1_000;
    // Small enough that the corpus, which interns a few constants per lift,
    // rebuilds the storage several times within the run.
    const BUDGET: usize = 2_048;

    let spec = sleigh_precompile::x64::spec();
    let lifter = SleighLifter::new(spec).with_flat_control_flow();
    let mut session = ScratchSession::new(&lifter)
        .unwrap()
        .with_literal_budget(BUDGET);
    let baseline = live();

    let mut checkpoints: Vec<(u32, usize, u64, usize)> = Vec::new();
    let mut peak_in_first_cycle = 0usize;
    let mut peak_after = 0usize;
    let mut lifted = 0u32;
    for i in 0..LIFTS {
        let (address, bytes) = instruction(i);
        {
            let lifted_insn = session
                .lift(address, &bytes)
                .unwrap_or_else(|e| panic!("{bytes:02x?} at {address:#x}: {e}"));
            // Read it, as a consumer would, so nothing is optimized away.
            std::hint::black_box(lifted_insn.exits().count());
            std::hint::black_box(lifted_insn.blocks().count());
        }
        lifted += 1;
        // Between lifts: what the session retains with no instruction live.
        let retained = live() - baseline;
        if session.rebuilds() == 0 {
            peak_in_first_cycle = peak_in_first_cycle.max(retained);
        } else {
            peak_after = peak_after.max(retained);
        }
        if (i + 1) % CHECKPOINT == 0 {
            checkpoints.push((
                i + 1,
                retained,
                session.rebuilds(),
                session.interned_literals(),
            ));
        }
    }
    assert_eq!(lifted, LIFTS);

    println!(
        "{:>7} {:>14} {:>9} {:>9}",
        "lifts", "retained (B)", "rebuilds", "literals"
    );
    for (n, retained, rebuilds, literals) in &checkpoints {
        println!("{n:>7} {retained:>14} {rebuilds:>9} {literals:>9}");
    }
    println!(
        "peak retained: first budget cycle {peak_in_first_cycle} B, afterwards {peak_after} B; \
         rebuilds {}",
        session.rebuilds()
    );

    assert!(
        session.rebuilds() >= 3,
        "the corpus must cross the budget several times to test the bound ({})",
        session.rebuilds()
    );
    // Bounded: once the storage has been through one full budget cycle, no
    // later moment retains more than that cycle's peak plus a small allowance
    // for capacities that settle late (hash tables round up to powers of two,
    // and the first rebuild's clone starts from smaller ones). Measured at
    // +384 B over 12,000 lifts and 4 rebuilds; a leak of one chunk or one
    // name per lift would exceed this within the first thousand.
    let allowance = peak_in_first_cycle / 50 + 16 * 1024;
    assert!(
        peak_after <= peak_in_first_cycle + allowance,
        "retained memory kept growing: first cycle peaked at {peak_in_first_cycle} B, \
         later at {peak_after} B"
    );
    // And the last checkpoint is no larger than the first cycle's peak plus the
    // allowance either — the footprint is a bounded oscillation, not a ramp.
    let last = checkpoints.last().unwrap().1;
    assert!(
        last <= peak_in_first_cycle + allowance,
        "{last} B at the end"
    );
}

/// Retention per corpus shape, for locating what a shape leaves behind.
#[test]
#[ignore = "diagnostic: prints per-shape retained bytes"]
fn retained_bytes_per_shape() {
    use qcode::lift::ScratchStore;
    use sleigh::Decoder;
    let spec = sleigh_precompile::x64::spec();
    let lifter = SleighLifter::new(spec).with_flat_control_flow();
    let decoder = Decoder::new(spec);
    for shape in 0..16u32 {
        let mut store = ScratchStore::new(lifter.new_context()).with_literal_budget(1 << 20);
        let baseline = live();
        let mut after_first = 0;
        const N: u32 = 2_000;
        for k in 0..N {
            let (address, bytes) = instruction(shape + 16 * k);
            let instruction = decoder
                .decode_one(address, &bytes, &spec.new_context())
                .unwrap();
            store.reset();
            {
                let mut target = store.target().unwrap();
                let l = lifter.lift_into(&mut target, &instruction).unwrap();
                std::hint::black_box(l.exits().len());
            }
            drop(instruction);
            if k == 0 {
                after_first = live() - baseline;
            }
        }
        let retained = live() - baseline;
        let ctx = store.context();
        println!(
            "shape {shape:>2}: first {after_first:>8} B, after {N} lifts {retained:>9} B ({:>6} B/lift); literals {} varnodes {} bytes {} types {} functions {} arena {:?}",
            retained.saturating_sub(after_first) / (N as usize - 1),
            store.interned_literals(),
            ctx.shared.values.varnodes.len(),
            ctx.shared.values.bytes.len(),
            ctx.shared.types.published_len(),
            ctx.functions().count(),
            store.arena_stats().instructions,
        );
    }
}

/// Where the per-lift retention of one shape comes from: decoding alone,
/// flattening alone, or lowering.
#[test]
#[ignore = "diagnostic: splits retention between decode, flatten and lower"]
fn retained_bytes_by_stage() {
    use qcode::lift::ScratchStore;
    use sleigh::Decoder;
    let spec = sleigh_precompile::x64::spec();
    let lifter = SleighLifter::new(spec).with_flat_control_flow();
    let decoder = Decoder::new(spec);
    let shape: u32 = std::env::var("SHAPE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(11);
    const N: u32 = 500;
    for stage in 0..3 {
        let mut store = ScratchStore::new(lifter.new_context());
        let baseline = live();
        let mut after_first = 0;
        for k in 0..N {
            let (address, bytes) = instruction(shape + 16 * k);
            let instruction = decoder
                .decode_one(address, &bytes, &spec.new_context())
                .unwrap();
            match stage {
                0 => {}
                1 => {
                    let flat = instruction.pcode_ops().unwrap();
                    std::hint::black_box(flat.ops.len());
                }
                _ => {
                    store.reset();
                    let mut target = store.target().unwrap();
                    let l = lifter.lift_into(&mut target, &instruction).unwrap();
                    std::hint::black_box(l.exits().len());
                }
            }
            drop(instruction);
            if k == 0 {
                after_first = live() - baseline;
            }
        }
        let retained = live() - baseline;
        println!(
            "shape {shape} stage {stage}: first {after_first} B, after {N}: {retained} B ({} B/lift)",
            retained.saturating_sub(after_first) / (N as usize - 1)
        );
    }
}
