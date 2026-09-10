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
use qcode_userland::bare;
use qcode_vm::{BlockExecutor, Executed, VmMemory};
use rustc_hash::FxHashMap;

mod support;

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
        start: usize,
        _chain: bool,
    ) -> Result<Option<Executed>, EmulatorErrorKind> {
        // Only whole blocks are counted; a continuation after an interrupt is
        // the same block's instructions, already counted at its entry.
        if start != 0 {
            return Ok(None);
        }
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

fn prepared(image: &[u8]) -> bare::Machine {
    bare::machine(image).expect("the image loads and its entry decodes")
}

#[test]
#[ignore = "diagnostic; needs benchmarks/embench/build.sh to have been run"]
fn report_guest_instruction_rate() {
    let images = support::images();
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
