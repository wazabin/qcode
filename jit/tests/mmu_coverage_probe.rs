//! Diagnostic: how much of a real program's lifted code the compiler now takes,
//! and what is left declining it.
//!
//! Not an assertion. Coverage is a number to steer by, and the thing that keeps
//! it honest is `divergence`, not a threshold here.

use qcode::value::BasicBlock;
use qcode_jit::Jit;
use qcode_vm::{Vm, VmMemory, perm};
use rustc_hash::FxHashMap;
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

const SENTINEL: u64 = 0xdead_0000;
const STACK: u64 = 0x7fff_0000;
const STACK_SIZE: u64 = 0x40000;
const STACK_TOP: u64 = 0x7fff_8000;

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
            continue;
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
        let mut bytes = image[off..off + filesz].to_vec();
        bytes.resize(memsz, 0);
        memory.mmu.write_unchecked(vaddr, &bytes, bits);
    }
    entry
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

#[test]
#[ignore = "diagnostic; needs benchmarks/embench/build.sh to have been run"]
fn report_block_coverage_over_embench() {
    let images = images();
    assert!(
        !images.is_empty(),
        "no images; run benchmarks/embench/build.sh"
    );

    let mut totals = (0u64, 0u64);
    let mut reasons: FxHashMap<String, u64> = FxHashMap::default();
    for (name, image) in &images {
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
            .expect("RSP is a register");
        // Discovery only: enough of the program to have lifted its hot code.
        vm.run(20_000_000);

        let ctx = vm.context().clone();
        let mut jit = Jit::new();
        let (mut ok, mut no) = (0u64, 0u64);
        for block in ctx.block_ids() {
            if BasicBlock::from_id(&ctx, block)
                .instruction_ids()
                .is_empty()
            {
                continue;
            }
            match jit.try_compile(&ctx, block) {
                Ok(()) => ok += 1,
                Err(reason) => {
                    no += 1;
                    *reasons.entry(reason.to_string()).or_default() += 1;
                }
            }
        }
        totals.0 += ok;
        totals.1 += no;
        let share = 100.0 * ok as f64 / (ok + no).max(1) as f64;
        eprintln!(
            "{name:16} {ok:>5} compiled / {:>5} blocks  {share:5.1}%",
            ok + no
        );
    }

    let (ok, no) = totals;
    eprintln!(
        "\ntotal {ok} of {} blocks  {:.1}%",
        ok + no,
        100.0 * ok as f64 / (ok + no).max(1) as f64
    );
    let mut ranked: Vec<_> = reasons.into_iter().collect();
    ranked.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    eprintln!("\nwhy the rest decline:");
    for (reason, count) in ranked.iter().take(15) {
        eprintln!("  {count:>6}  {reason}");
    }
}
