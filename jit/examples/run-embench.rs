//! Runs one Embench image under the VM, for profiling.
//!
//! `cargo run --release -p qcode_jit --example run-embench -- <name> [interp]`
//!
//! Exists so a profiler has a single process doing one benchmark, rather than a
//! test binary running seventeen of them twice.

use qcode::{context::Context, value::BlockId};
use qcode_emulator::{EmulatorErrorKind, StandaloneEmulator};
use qcode_jit::Jit;
use qcode_vm::{BlockExecutor, Executed, Vm, VmMemory, perm};
use rustc_hash::FxHashMap;
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

const SENTINEL: u64 = 0xdead_0000;
const STACK: u64 = 0x7fff_0000;
const STACK_SIZE: u64 = 0x40000;
const STACK_TOP: u64 = 0x7fff_8000;

/// A [`Jit`] that also records, per *execution*, why a block it could not run
/// was declined.
///
/// Counting declines at compile time answers a different question: a block
/// declined once and never executed costs nothing, while one declined once and
/// executed a million times is the whole run. Only the weighted count says
/// which is which.
struct Profiler {
    jit: Jit,
    interpreted: u64,
    reasons: FxHashMap<String, u64>,
}

impl BlockExecutor for Profiler {
    fn run_block(
        &mut self,
        ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        block: BlockId,
        chain: bool,
    ) -> Result<Option<Executed>, EmulatorErrorKind> {
        let ran = self.jit.run_block(ctx, emu, block, chain)?;
        if ran.is_none() {
            self.interpreted += 1;
            // Cached, so this is a lookup rather than a second compilation.
            if let Err(reason) = self.jit.try_compile(ctx, block) {
                *self.reasons.entry(reason.to_string()).or_default() += 1;
            }
        }
        Ok(ran)
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

fn main() {
    let mut args = std::env::args().skip(1);
    let name = args.next().unwrap_or_else(|| "matmult-int".to_owned());
    let interpret = args.next().as_deref() == Some("interp");

    // `EMBENCH_DIR` selects an alternative corpus — a build at a larger scale
    // factor, say, where one-time translation is amortised over enough
    // execution to show a steady-state rate rather than a warm-up one.
    let dir = std::env::var("EMBENCH_DIR").unwrap_or_else(|_| "target/embench".to_owned());
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join(dir)
        .join(format!("{name}.elf"));
    let image = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));

    let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
    let ctx = source.new_context();
    let mut memory = VmMemory::new();
    let entry = load(&image, &mut memory);
    memory.mmu.map(STACK, STACK_SIZE, perm::RW_INIT).unwrap();
    memory
        .mmu
        .write_unchecked(STACK_TOP, &SENTINEL.to_le_bytes(), perm::RW_INIT);
    let mut vm = Vm::at_address(ctx, entry, source, memory).expect("the entry decodes");
    let ctx = vm.context().clone();
    vm.emulator()
        .set_varnode_by_name(&ctx, "RSP", STACK_TOP)
        .expect("RSP is a register");

    let profiler: *const Profiler = if interpret {
        std::ptr::null()
    } else {
        let profiler = Box::new(Profiler {
            jit: Jit::new(),
            interpreted: 0,
            reasons: FxHashMap::default(),
        });
        let raw: *const Profiler = &*profiler;
        vm.set_block_executor(profiler);
        raw
    };

    let started = std::time::Instant::now();
    let exit = vm.run(4_000_000_000);
    let elapsed = started.elapsed();

    let stats = vm.stats.clone();
    eprintln!("{name}: {exit:?} in {elapsed:.2?}");
    eprintln!(
        "  steps={} native_bodies={} lifts={} resolves={} absorbed={}",
        stats.steps, stats.native_bodies, stats.lifts, stats.resolves, stats.absorbed
    );
    eprintln!(
        "  translation={:.2?} (fetch {:.2?} lift {:.2?} optimize {:.2?})",
        stats.translation(),
        stats.fetch,
        stats.decode_lift,
        stats.optimize
    );
    if !profiler.is_null() {
        // SAFETY: the VM still owns the executor and the run is over.
        let profiler = unsafe { &*profiler };
        let jit = &profiler.jit;
        eprintln!(
            "  jit: compiled={} declined={} native_runs={} interpreted_entries={}",
            jit.stats.compiled, jit.stats.declined, jit.stats.native_runs, profiler.interpreted
        );
        let mut ranked: Vec<_> = profiler.reasons.iter().collect();
        ranked.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
        for (reason, count) in ranked.iter().take(10) {
            let share = 100.0 * **count as f64 / profiler.interpreted.max(1) as f64;
            eprintln!("    {count:>10}  {share:5.1}%  {reason}");
        }
    }
}
