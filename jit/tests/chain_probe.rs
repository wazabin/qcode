//! Diagnostic: why compiled code hands control back to the interpreter.
//!
//! Wraps the JIT and, at every return to the interpreter, records the
//! terminator of the block the interpreter resumes at — the edge the backend
//! would not follow — so the chain breaks can be counted by kind.
use qcode::{
    context::Context,
    value::{BlockId, Instruction, insn::InstructionId},
};
use qcode_emulator::{EmulatorErrorKind, StandaloneEmulator};
use qcode_jit::{Jit, compile::Unsupported};
use qcode_vm::{BlockExecutor, Executed, Vm, VmMemory, perm};
use rustc_hash::FxHashMap;
use std::{cell::RefCell, rc::Rc};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

const SENTINEL: u64 = 0xdead_0000;
const STACK: u64 = 0x7fff_0000;
const STACK_SIZE: u64 = 0x40000;
const STACK_TOP: u64 = 0x7fff_8000;

#[derive(Default)]
struct Tally {
    /// Returns to the interpreter, by the terminator kind of the block it
    /// resumes at, and whether that block's body was ever compiled.
    breaks: FxHashMap<(&'static str, &'static str), u64>,
    declined: FxHashMap<String, u64>,
}

struct Probe {
    jit: Jit,
    tally: Rc<RefCell<Tally>>,
}

impl BlockExecutor for Probe {
    fn run_block(
        &mut self,
        ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        block: BlockId,
        start: usize,
        chain: u64,
    ) -> Result<Option<Executed>, EmulatorErrorKind> {
        let mut tally = self.tally.borrow_mut();
        let out = self.jit.run_block(ctx, emu, block, start, chain);
        let (at, state) = match &out {
            Ok(Some(run)) => (
                run.block,
                if run.body == 0 {
                    "next-declined"
                } else {
                    "ran"
                },
            ),
            Ok(None) => (block, "this-declined"),
            Err(_) => return out,
        };
        if state != "ran"
            && let Err(why) = self.jit.try_compile(ctx, at)
        {
            let key = match why {
                Unsupported::Mnemonic(m) => format!("mnemonic {m}"),
                Unsupported::Width(w) => format!("width {w}"),
                Unsupported::Access(a) => format!("access {a}"),
                Unsupported::Terminator(t) => format!("terminator {t}"),
                Unsupported::Operand(o) => format!("operand {o}"),
                Unsupported::Escapes(e) => format!("escapes {e}"),
            };
            *tally.declined.entry(key).or_default() += 1;
        }
        let term = ctx
            .block(at)
            .last_insn()
            .map(|t| {
                Instruction::from_id(ctx, InstructionId::new(at.func, t))
                    .mnemonic()
                    .opcode()
            })
            .unwrap_or("none");
        *tally.breaks.entry((term, state)).or_default() += 1;
        out
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

#[test]
#[ignore = "diagnostic"]
fn report_chain_breaks_over_embench() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target/embench");
    let mut images: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "elf"))
        .map(|e| {
            (
                e.path().file_stem().unwrap().to_string_lossy().into_owned(),
                std::fs::read(e.path()).unwrap(),
            )
        })
        .collect();
    images.sort_by(|a, b| a.0.cmp(&b.0));
    let only = std::env::var("ONLY").ok();
    for (name, image) in &images {
        if only.as_ref().is_some_and(|o| o != name) {
            continue;
        }
        let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
        let ctx = source.new_context();
        let mut memory = VmMemory::new();
        let entry = load(image, &mut memory);
        memory.mmu.map(STACK, STACK_SIZE, perm::RW_INIT).unwrap();
        memory
            .mmu
            .write_unchecked(STACK_TOP, &SENTINEL.to_le_bytes(), perm::RW_INIT);
        let mut vm = Vm::at_address(ctx, entry, source, memory).unwrap();
        let ctx = vm.context().clone();
        vm.emulator()
            .set_varnode_by_name(&ctx, "RSP", STACK_TOP)
            .unwrap();
        let tally = Rc::new(RefCell::new(Tally::default()));
        vm.set_block_executor(Box::new(Probe {
            jit: Jit::new(),
            tally: tally.clone(),
        }));
        let started = std::time::Instant::now();
        let exit = vm.run(4_000_000_000);
        let elapsed = started.elapsed();
        eprintln!(
            "{name:16} {elapsed:>8.1?} entries={} exit={exit:?}",
            vm.stats.native_bodies
        );
        eprintln!("    {}", vm.stats.report(elapsed));
        let tally = tally.borrow();
        let mut breaks: Vec<_> = tally.breaks.iter().collect();
        breaks.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
        for ((term, state), n) in breaks {
            eprintln!("    {n:>9}  {term:<12} {state}");
        }
        let mut declined: Vec<_> = tally.declined.iter().collect();
        declined.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
        for (why, n) in declined.iter().take(6) {
            eprintln!("    {n:>9}  declined: {why}");
        }
    }
}
