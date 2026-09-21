//! The lift cache replays exactly what a fresh lift emits.
//!
//! Every encoding of the corpus is lifted three ways — uncached, as a cache
//! miss, and as a cache hit at another address — and the hit must render to
//! the same IR as the uncached lift at that address. Addresses on both sides
//! of 4 GiB are covered, since that is where a truncated address-derived
//! constant would show.

use std::sync::Arc;

use qcode::lift::Lifted;
use wazabin_qcode_sleigh::{
    LiftError, SleighLifter,
    cache::LiftCache,
    session::{Host, LiftSession, ScratchLifted, ScratchOperand, ScratchSession},
};

/// x86-64 encodings: the lifting benchmark's corpus plus every form whose
/// lowering carries an address — relative branches and calls, `rip`-relative
/// operands, the pushed return address — and the busiest vector forms.
const CORPUS: &[(&str, &[u8])] = &[
    ("nop", b"\x90"),
    ("nop5", b"\x0f\x1f\x44\x00\x00"),
    ("nopw", b"\x66\x66\x2e\x0f\x1f\x84\x00\x00\x00\x00\x00"),
    ("push_rbp", b"\x55"),
    ("push_imm32", b"\x68\x78\x56\x34\x12"),
    ("mov_rax_imm32", b"\x48\xc7\xc0\x78\x56\x34\x12"),
    ("mov_rax_imm64", b"\x48\xb8\x00\x10\x00\x00\x01\x00\x00\x00"),
    ("mov_rcx_mem_rdx", b"\x48\x8b\x0a"),
    ("mov_mem_r8_disp_r9", b"\x4d\x89\x48\x10"),
    ("mov_rax_rip", b"\x48\x8b\x05\x10\x00\x00\x00"),
    ("lea_r11_rip", b"\x4c\x8d\x1d\x20\x00\x00\x00"),
    ("lea_rax_rip_neg", b"\x48\x8d\x05\xf0\xff\xff\xff"),
    ("add_rax_rcx", b"\x48\x01\xc8"),
    (
        "add_mem_imm32",
        b"\x48\x81\x04\x25\x10\x20\x30\x00\x05\x00\x00\x00",
    ),
    ("xor_eax_eax", b"\x31\xc0"),
    ("test_al_al", b"\x84\xc0"),
    ("cmp_rdi_imm8", b"\x48\x83\xff\x05"),
    ("shl_rax_cl", b"\x48\xd3\xe0"),
    ("div_rcx", b"\x48\xf7\xf1"),
    ("movsxd_rax_edi", b"\x48\x63\xc7"),
    ("movzx_eax_al", b"\x0f\xb6\xc0"),
    ("cmpxchg_rax_rcx", b"\x48\x0f\xb1\xc8"),
    ("jz_rel8", b"\x74\x05"),
    ("jz_rel8_back", b"\x74\xf0"),
    ("jmp_rel8_self", b"\xeb\xfe"),
    ("jmp_rel32", b"\xe9\x00\x10\x00\x00"),
    ("jrcxz", b"\xe3\x05"),
    ("loop", b"\xe2\xfc"),
    ("call_rel32", b"\xe8\x10\x00\x00\x00"),
    ("call_rel32_back", b"\xe8\xf0\xff\xff\xff"),
    ("call_rip_slot", b"\xff\x15\x10\x20\x00\x00"),
    ("call_rax", b"\xff\xd0"),
    ("call_table", b"\x41\xff\x14\xc4"),
    ("jmp_rax", b"\xff\xe0"),
    ("jmp_rip_slot", b"\xff\x25\x10\x20\x00\x00"),
    ("ret", b"\xc3"),
    ("ret_imm16", b"\xc2\x08\x00"),
    ("leave", b"\xc9"),
    ("enter", b"\xc8\x10\x00\x00"),
    ("rep_movsb", b"\xf3\xa4"),
    ("rep_stosq", b"\xf3\x48\xab"),
    ("hlt", b"\xf4"),
    ("ud2", b"\x0f\x0b"),
    ("int_80", b"\xcd\x80"),
    ("syscall", b"\x0f\x05"),
    ("cpuid", b"\x0f\xa2"),
    ("rdtsc", b"\x0f\x31"),
    ("pushfq", b"\x9c"),
    ("endbr64", b"\xf3\x0f\x1e\xfa"),
    ("mulsd_xmm0_mem", b"\xf2\x0f\x59\x04\x25\x10\x20\x30\x00"),
    ("addps_xmm0_xmm1", b"\x0f\x58\xc1"),
    ("pshufd", b"\x66\x0f\x70\xe4\x00"),
    ("pxor_xmm0_xmm0", b"\x66\x0f\xef\xc0"),
    ("movaps_rip", b"\x0f\x28\x05\x10\x00\x00\x00"),
    ("fld_mem", b"\xd9\x05\x10\x20\x30\x00"),
    ("fld_rip", b"\xdd\x05\x10\x00\x00\x00"),
    ("fmulp", b"\xde\xc9"),
    ("fxch", b"\xd9\xc9"),
];

/// Corpus entries that share a shape with an earlier one — the backward
/// `jz` and `call` with the forward ones — so they are hits even while
/// warming.
const SHAPE_REPEATS: usize = 2;

/// Corpus entries whose shape cannot be parameterized from them, and are
/// remembered by exact encoding: `jmp` to itself branches to its own entry
/// block, which no other offset does; `pshufd` masks lane selectors out of
/// its immediate.
const EXACT: usize = 2;

/// Pairs of one shape: the second differs from the first only in a
/// parameter, and is served from the first's template when the flag says
/// the shape is linear in it — including `ret 8`, whose immediate equals
/// the pop's width, `add rsp, 8` likewise, and a `jz` whose target lands on
/// the fall-through. A shape that is not, like `pshufd`'s, is remembered by
/// exact encoding, so the variant misses once and hits at every other
/// address.
const VARIANTS: &[(&str, &[u8], &[u8], bool)] = &[
    (
        "mov_rax_rbp_disp8",
        b"\x48\x8b\x45\xe8",
        b"\x48\x8b\x45\xe0",
        true,
    ),
    (
        "add_rsp_imm8",
        b"\x48\x83\xc4\x10",
        b"\x48\x83\xc4\x08",
        true,
    ),
    (
        "sub_rsp_imm32",
        b"\x48\x81\xec\x00\x01\x00\x00",
        b"\x48\x81\xec\x08\x00\x00\x00",
        true,
    ),
    ("ret_imm16", b"\xc2\x10\x00", b"\xc2\x08\x00", true),
    (
        "pshufd_imm8",
        b"\x66\x0f\x70\xe4\x1b",
        b"\x66\x0f\x70\xe4\x00",
        false,
    ),
    ("jz_rel8", b"\x74\x05", b"\x74\x00", true),
    ("jz_rel8_back", b"\x74\x05", b"\x74\xf0", true),
    (
        "call_rel32",
        b"\xe8\x10\x00\x00\x00",
        b"\xe8\xf0\xff\xff\xff",
        true,
    ),
    (
        "mov_rax_rip",
        b"\x48\x8b\x05\x10\x00\x00\x00",
        b"\x48\x8b\x05\xf0\xff\xff\xff",
        true,
    ),
    (
        "shl_rax_imm8",
        b"\x48\xc1\xe0\x03",
        b"\x48\xc1\xe0\x21",
        true,
    ),
    (
        "mov_rax_imm64",
        b"\x48\xb8\x00\x10\x00\x00\x01\x00\x00\x00",
        b"\x48\xb8\xff\xff\xff\xff\xff\xff\xff\xff",
        true,
    ),
    (
        "mov_mem_imm32",
        b"\x48\xc7\x45\xf8\x01\x00\x00\x00",
        b"\x48\xc7\x45\xd0\xff\xff\xff\xff",
        true,
    ),
];

/// Addresses on both sides of 4 GiB, none a multiple of another's stride.
const ADDRESSES: &[u64] = &[
    0x1000,
    0x40_1234,
    0x7fff_ffff_f000,
    0x1_2345_6789,
    0xffff_ffff_ffff_ff00,
];

fn lifter() -> SleighLifter<'static> {
    SleighLifter::new(sleigh_precompile::x64::spec()).with_flat_control_flow()
}

/// The IR of a scratch lift as text: every block's operations with their
/// resolved operands, and the exits. Ids restart at zero for each lift, so
/// two lifts that emit the same operations in the same order render alike —
/// except literal ids, which the shared interner numbers by first mention
/// across lifts, and the ids of other instructions' blocks, which a cached
/// lift numbers by its own scheme: those are blanked, and the resolved
/// constants and block addresses compared instead.
fn render(lifted: &ScratchLifted<'_, '_, '_, '_>) -> String {
    fn blank(text: &str, what: &str) -> String {
        let mut out = String::new();
        let mut rest = text;
        while let Some(at) = rest.find(what) {
            out.push_str(&rest[..at]);
            out.push_str(what);
            out.push_str("_)");
            rest = &rest[at + what.len()..];
            rest = &rest[rest.find(')').map_or(rest.len(), |end| end + 1)..];
        }
        out.push_str(rest);
        out
    }
    let mut out = String::new();
    for block in lifted.blocks() {
        out.push_str(&format!(
            "block @{:?} insns={}\n",
            block.address(),
            block.has_insns()
        ));
        for insn in block.instructions() {
            let operands: Vec<String> = insn
                .operands()
                .map(|operand| match operand {
                    ScratchOperand::Block(block) => format!("Block(@{:?})", block.address()),
                    other => format!("{other:?}"),
                })
                .collect();
            // Literal, block and varnode ids in a raw mnemonic are the
            // template's keys, not the store's: the resolved operands say
            // what they are.
            let mnemonic = blank(
                &blank(
                    &blank(&format!("{:?}", insn.mnemonic()), "LiteralId("),
                    "LocalBlockId(",
                ),
                "VarnodeId(",
            );
            let targets = match (insn.branch_target(), insn.cbranch_targets()) {
                (Some(target), _) => format!(" -> @{:?}", target.address()),
                (_, Some((taken, not))) => {
                    format!(" -> @{:?} | @{:?}", taken.address(), not.address())
                }
                _ => String::new(),
            };
            out.push_str(&format!(
                "  {:?} = {} {mnemonic} {operands:?}{targets} callee={:?} block=@{:?}\n",
                insn.result(),
                insn.opcode(),
                insn.callee_address(),
                insn.block().address(),
            ));
        }
    }
    for exit in lifted.exits() {
        out.push_str(&format!(
            "exit {:?} {:?} {:?} conditional={}\n",
            exit.site().result(),
            exit.arm(),
            exit.kind()
                .clone()
                .continuation()
                .map(|c| matches!(c, qcode::lift::Continuation::Next)),
            exit.is_conditional()
        ));
    }
    let plain = lifted.lifted();
    out.push_str(&format!(
        "lifted {:#x}+{} blocks={} falls_through={} calls={} entry={:?}\n",
        plain.address(),
        plain.length(),
        plain.blocks().len(),
        plain.falls_through(),
        plain.calls(),
        lifted.entry().address()
    ));
    out
}

/// Asserts two renderings are equal, reporting the first line that differs.
#[track_caller]
fn assert_same(what: &str, got: &Option<String>, expected: &Option<String>) {
    if got == expected {
        return;
    }
    let (Some(got), Some(expected)) = (got, expected) else {
        panic!("{what}: one side failed to lift: got {got:?}, expected {expected:?}");
    };
    for (line, (g, e)) in got.lines().zip(expected.lines()).enumerate() {
        assert_eq!(g, e, "{what}: first difference at line {line}");
    }
    assert_eq!(
        got.lines().count(),
        expected.lines().count(),
        "{what}: lengths differ"
    );
}

fn render_at(
    session: &mut ScratchSession<'_, '_>,
    address: u64,
    bytes: &[u8],
) -> Result<String, LiftError> {
    let lifted = session.lift(address, bytes)?;
    Ok(render(&lifted))
}

#[test]
fn a_scratch_hit_renders_as_the_uncached_lift() {
    let lifter = lifter();
    let cache = Arc::new(LiftCache::new(lifter.spec()));
    let mut plain = ScratchSession::new(&lifter);
    let mut cached = ScratchSession::new(&lifter).with_cache(Arc::clone(&cache));

    // Warm the cache at the first address, then every other address is a hit.
    for (name, bytes) in CORPUS {
        let expected = render_at(&mut plain, ADDRESSES[0], bytes);
        let got = render_at(&mut cached, ADDRESSES[0], bytes);
        assert_same(
            &format!("{name} (miss)"),
            &got.as_ref().ok().cloned(),
            &expected.as_ref().ok().cloned(),
        );
        assert_eq!(got.is_err(), expected.is_err(), "{name} (miss)");
    }
    let after_warmup = cache.stats();
    assert_eq!(
        after_warmup.hits as usize, SHAPE_REPEATS,
        "{after_warmup:?}"
    );
    assert_eq!(after_warmup.misses as usize, CORPUS.len() - SHAPE_REPEATS);
    assert_eq!(after_warmup.exact as usize, EXACT, "{after_warmup:?}");

    for &address in &ADDRESSES[1..] {
        for (name, bytes) in CORPUS {
            let expected = render_at(&mut plain, address, bytes).ok();
            let got = render_at(&mut cached, address, bytes).ok();
            assert_same(&format!("{name} at {address:#x}"), &got, &expected);
        }
    }
    let stats = cache.stats();
    let misses = CORPUS.len() - SHAPE_REPEATS;
    assert_eq!(stats.misses as usize, misses, "{stats:?}");
    assert_eq!(
        stats.uncacheable, 0,
        "every shape of the corpus is cacheable: {stats:?}"
    );
    assert_eq!(stats.exact as usize, EXACT);
    assert_eq!(stats.hits as usize, CORPUS.len() * ADDRESSES.len() - misses);
    assert_eq!(stats.entries, misses);
}

#[test]
fn a_variant_of_a_shape_is_served_from_the_first_instance() {
    let lifter = lifter();
    let cache = Arc::new(LiftCache::new(lifter.spec()));
    let mut plain = ScratchSession::new(&lifter);
    let mut cached = ScratchSession::new(&lifter).with_cache(Arc::clone(&cache));
    let mut hits = 0;
    for (name, first, variant, linear) in VARIANTS {
        let expected = render_at(&mut plain, ADDRESSES[0], first).ok();
        let lifted = cached.lift(ADDRESSES[0], first).unwrap();
        hits += usize::from(lifted.is_cached());
        assert_same(
            &format!("{name} (first)"),
            &Some(render(&lifted)),
            &expected,
        );
        for (index, &address) in ADDRESSES.iter().enumerate() {
            let expected = render_at(&mut plain, address, variant).ok();
            let lifted = cached.lift(address, variant).unwrap();
            let hit = *linear || index > 0;
            assert_eq!(lifted.is_cached(), hit, "{name} at {address:#x}");
            hits += usize::from(hit);
            assert_same(
                &format!("{name} at {address:#x}"),
                &Some(render(&lifted)),
                &expected,
            );
        }
    }
    let stats = cache.stats();
    assert_eq!(stats.hits as usize, hits, "{stats:?}");
    assert_eq!(stats.uncacheable, 0, "{stats:?}");
    // Both instances of a shape that is not linear are exact misses.
    assert_eq!(
        stats.exact as usize,
        2 * VARIANTS.iter().filter(|(_, _, _, linear)| !linear).count(),
        "{stats:?}"
    );

    // Validation agrees on every variant too, whichever instance is first.
    let cache = Arc::new(LiftCache::new(lifter.spec()).validating(true));
    let mut session = ScratchSession::new(&lifter).with_cache(Arc::clone(&cache));
    for (_, first, variant, _) in VARIANTS {
        for &address in ADDRESSES {
            session.lift(address, variant).unwrap();
            session.lift(address, first).unwrap();
        }
    }
    let stats = cache.stats();
    assert_eq!(stats.validation_failures, 0, "{stats:?}");
    assert_eq!(
        stats.hits as usize,
        2 * VARIANTS.len() * ADDRESSES.len() - stats.misses as usize
    );
}

#[test]
fn validation_agrees_on_every_hit() {
    let lifter = lifter();
    let cache = Arc::new(LiftCache::new(lifter.spec()).validating(true));
    let mut session = ScratchSession::new(&lifter).with_cache(Arc::clone(&cache));
    for &address in ADDRESSES {
        for (_, bytes) in CORPUS {
            let _ = session.lift(address, bytes);
        }
    }
    let stats = cache.stats();
    assert_eq!(stats.validation_failures, 0, "{stats:?}");
    let misses = CORPUS.len() - SHAPE_REPEATS;
    assert_eq!(
        stats.hits as usize,
        CORPUS.len() * ADDRESSES.len() - misses,
        "{stats:?}"
    );
}

/// A straight-line sequence with relative branches into and out of it and a
/// `rip`-relative load, lifted as one function.
const SEQUENCE: &[&[u8]] = &[
    b"\xf3\x0f\x1e\xfa",             // endbr64
    b"\x55",                         // push rbp
    b"\x48\x89\xe5",                 // mov rbp, rsp
    b"\x48\x8b\x05\x10\x00\x00\x00", // mov rax, [rip+0x10]
    b"\x48\x85\xc0",                 // test rax, rax
    b"\x74\x05",                     // jz +5
    b"\xe8\x00\x01\x00\x00",         // call +0x100
    b"\x0f\x58\xc1",                 // addps xmm0, xmm1
    b"\xde\xc9",                     // fmulp
    b"\x48\x8b\x05\x10\x00\x00\x00", // mov rax, [rip+0x10] again: a hit
    b"\x74\x05",                     // jz +5 again: a hit
    b"\x5d",                         // pop rbp
    b"\xc3",                         // ret
];

fn lift_sequence(session: &mut LiftSession<'_, '_>, start: u64) -> Vec<Lifted> {
    let mut address = start;
    SEQUENCE
        .iter()
        .map(|bytes| {
            let lifted = session.lift(address, bytes).unwrap();
            address = lifted.next_address();
            lifted
        })
        .collect()
}

#[test]
fn a_session_hit_builds_the_same_function_names_included() {
    for flat in [false, true] {
        let lifter = SleighLifter::new(sleigh_precompile::x64::spec());
        let lifter = if flat {
            lifter.with_flat_control_flow()
        } else {
            lifter
        };
        let cache = Arc::new(LiftCache::new(lifter.spec()));

        // Warm at one address; the sequence repeats two of its own
        // instructions, so even the warm-up has hits.
        let warm = lift_sequence(
            &mut LiftSession::new(&lifter, Host::At(0x1000)).with_cache(Arc::clone(&cache)),
            0x1000,
        );
        let plain = {
            let mut session = LiftSession::new(&lifter, Host::At(0x1000));
            let lifted = lift_sequence(&mut session, 0x1000);
            for (index, (a, b)) in lifted.iter().zip(&warm).enumerate() {
                assert_eq!(a, b, "flat={flat}, instruction {index}");
            }
            session.into_context().unwrap()
        };
        let stats = cache.stats();
        assert_eq!(stats.hits, 2, "flat={flat}: {stats:?}");
        assert_eq!(stats.uncacheable, 0, "flat={flat}: {stats:?}");

        // Everything is a hit now, at a far address, and the function reads
        // the same as one lifted without the cache — names included.
        for &start in &ADDRESSES[1..] {
            let mut cached =
                LiftSession::new(&lifter, Host::At(start)).with_cache(Arc::clone(&cache));
            let hits = lift_sequence(&mut cached, start);
            let mut plain_session = LiftSession::new(&lifter, Host::At(start));
            let misses = lift_sequence(&mut plain_session, start);
            for (index, (a, b)) in hits.iter().zip(&misses).enumerate() {
                assert_eq!(a, b, "flat={flat} at {start:#x}, instruction {index}");
            }
            assert_same(
                &format!("flat={flat} at {start:#x}"),
                &Some(cached.into_context().unwrap().to_string()),
                &Some(plain_session.into_context().unwrap().to_string()),
            );
        }
        let stats = cache.stats();
        assert_eq!(
            stats.hits as usize,
            2 + SEQUENCE.len() * (ADDRESSES.len() - 1),
            "flat={flat}: {stats:?}"
        );
        let _ = plain;
    }
}

#[test]
fn the_cache_is_shared_across_threads() {
    let lifter = lifter();
    let cache = Arc::new(LiftCache::new(lifter.spec()));
    let expected: Vec<Option<String>> = {
        let mut plain = ScratchSession::new(&lifter);
        ADDRESSES
            .iter()
            .flat_map(|&address| CORPUS.iter().map(move |(_, bytes)| (address, bytes)))
            .map(|(address, bytes)| render_at(&mut plain, address, bytes).ok())
            .collect()
    };
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let cache = Arc::clone(&cache);
            let lifter = &lifter;
            let expected = &expected;
            scope.spawn(move || {
                let mut session = ScratchSession::new(lifter).with_cache(cache);
                let got: Vec<Option<String>> = ADDRESSES
                    .iter()
                    .flat_map(|&address| CORPUS.iter().map(move |(_, bytes)| (address, bytes)))
                    .map(|(address, bytes)| render_at(&mut session, address, bytes).ok())
                    .collect();
                assert_eq!(&got, expected);
            });
        }
    });
    let stats = cache.stats();
    assert_eq!(stats.entries, CORPUS.len() - SHAPE_REPEATS);
    assert_eq!(
        stats.hits + stats.misses,
        (4 * CORPUS.len() * ADDRESSES.len()) as u64
    );
    assert!(stats.misses as usize >= CORPUS.len() - SHAPE_REPEATS);
}

#[test]
fn a_full_cache_stops_remembering() {
    let lifter = lifter();
    let cache = Arc::new(LiftCache::new(lifter.spec()).with_capacity(3));
    let mut session = ScratchSession::new(&lifter).with_cache(Arc::clone(&cache));
    for (_, bytes) in CORPUS {
        session.lift(0x1000, bytes).unwrap();
    }
    assert_eq!(cache.stats().entries, 3);
    cache.clear();
    assert_eq!(cache.stats().entries, 0);
}

#[test]
fn a_cache_refuses_another_specification() {
    let x64 = lifter();
    let cache = Arc::new(LiftCache::new(x64.spec()));
    let x86 = SleighLifter::new(sleigh_precompile::x86::spec());
    let mut session = ScratchSession::new(&x86).with_cache(cache);
    assert_eq!(
        session.lift(0x1000, b"\x90").err(),
        Some(LiftError::IncompatibleSpec)
    );
}
