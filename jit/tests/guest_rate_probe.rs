//! Diagnostic: guest instructions per second, which is not what `stats.steps`
//! counts.
//!
//! `steps` is *QCode operations* retired. x86 lifts to many of those per guest
//! instruction — the flag primitives alone are several — so quoting a step rate
//! as an instruction rate overstates it by roughly an order of magnitude.
//!
//! Guest instructions are counted exactly rather than estimated: a no-op
//! `BlockExecutor` sees every block entry the VM makes, and each guest
//! instruction contributes exactly one address-carrying QCode instruction to a
//! block.
//!
//! Counted *during* the run, not from a tally of block ids afterwards. A block
//! id is not stable: a later branch into the middle of an absorbed run splits
//! it by invalidating both halves, so an id recorded early is a dead arena
//! entry by the end. Only the block being entered is certainly live.

use qcode::{
    context::Context,
    value::{BlockId, Instruction, insn::InstructionId},
};
use qcode_emulator::{EmulatorErrorKind, StandaloneEmulator};
use qcode_jit::Jit;
use qcode_vm::{BlockExecutor, Executed, Vm, VmMemory, perm};
use rustc_hash::FxHashMap;
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

const SENTINEL: u64 = 0xdead_0000;
const STACK: u64 = 0x7fff_0000;
const STACK_SIZE: u64 = 0x40000;
const STACK_TOP: u64 = 0x7fff_8000;

/// Counts the guest instructions in every block the VM enters, and declines
/// each one, so the interpreter runs the program exactly as it would with no
/// executor installed.
#[derive(Default)]
struct Counter {
    guest: u64,
    /// Per-block counts, keyed by instruction count as well as id: a block is
    /// re-lifted in place when it splits, and the old answer does not survive
    /// that.
    cache: FxHashMap<(BlockId, usize), u64>,
}

impl BlockExecutor for Counter {
    fn run_block(
        &mut self,
        ctx: &Context<'_>,
        _emu: &mut StandaloneEmulator<VmMemory>,
        block: BlockId,
        _chain: bool,
    ) -> Result<Option<Executed>, EmulatorErrorKind> {
        let ids = ctx.block(block).instruction_ids();
        let key = (block, ids.len());
        let count = *self.cache.entry(key).or_insert_with(|| {
            // *Distinct* addresses, not address-carrying instructions: the
            // lifter stamps the machine address on every QCode instruction it
            // produces for a guest instruction, not just the first, so counting
            // the stamped ones just recounts p-code operations. The
            // instructions of one guest instruction are contiguous, so a change
            // of address is a new instruction.
            //
            // The terminator is excluded, the same way the retired count
            // excludes it.
            let body = ids.split_last().map(|(_, rest)| rest).unwrap_or(&[]);
            let mut seen = None;
            let mut count = 0;
            for &id in body {
                let at = Instruction::from_id(ctx, InstructionId::new(block.func, id)).address();
                if at.is_some() && at != seen {
                    count += 1;
                    seen = at;
                }
            }
            count
        });
        self.guest += count;
        Ok(None)
    }
}

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
    // `EMBENCH_DIR` selects an alternative corpus — a build at a larger scale
    // factor, say, where one-time translation is amortised over enough
    // execution to show a steady-state rate rather than a warm-up one.
    let corpus = std::env::var("EMBENCH_DIR").unwrap_or_else(|_| "target/embench".to_owned());
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join(corpus);
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

fn prepared(image: &[u8]) -> Vm<SleighCodeSource<'static>> {
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
    vm
}

#[test]
#[ignore = "diagnostic; needs benchmarks/embench/build.sh to have been run"]
fn report_guest_instruction_rate() {
    let images = images();
    assert!(
        !images.is_empty(),
        "no images; run benchmarks/embench/build.sh"
    );

    println!(
        "{:16} {:>14} {:>12} {:>10} {:>12} {:>12}",
        "benchmark", "guest insns", "pcode/insn", "interp M/s", "jit M/s", "speedup"
    );
    let (mut total_guest, mut total_steps) = (0u128, 0u128);
    let (mut total_interp, mut total_jit) = (0.0f64, 0.0f64);

    for (name, image) in &images {
        // Interpreted, with a counting executor that declines every block.
        let mut vm = prepared(image);
        let counter = Box::new(Counter::default());
        let counted: *const Counter = &*counter;
        vm.set_block_executor(counter);
        let started = std::time::Instant::now();
        vm.run(4_000_000_000);
        let interp = started.elapsed().as_secs_f64();
        let steps = vm.stats.steps;

        // SAFETY: the VM owns the executor and is still alive; nothing else
        // holds a reference to it, and the run is over.
        let guest = unsafe { &*counted }.guest;

        // The same program under the JIT, for the rate that matters.
        let mut vm = prepared(image);
        vm.set_block_executor(Box::new(Jit::new()));
        let started = std::time::Instant::now();
        vm.run(4_000_000_000);
        let jit = started.elapsed().as_secs_f64();
        assert_eq!(
            vm.stats.steps, steps,
            "{name}: the two strategies must retire the same work"
        );

        println!(
            "{name:16} {guest:>14} {:>12.1} {:>10.1} {:>12.1} {:>11.2}x",
            steps as f64 / guest as f64,
            guest as f64 / interp / 1e6,
            guest as f64 / jit / 1e6,
            interp / jit,
        );
        total_guest += u128::from(guest);
        total_steps += u128::from(steps);
        total_interp += interp;
        total_jit += jit;
    }

    println!(
        "\n{:16} {total_guest:>14} {:>12.1} {:>10.1} {:>12.1} {:>11.2}x",
        "TOTAL",
        total_steps as f64 / total_guest as f64,
        total_guest as f64 / total_interp / 1e6,
        total_guest as f64 / total_jit / 1e6,
        total_interp / total_jit,
    );
}
