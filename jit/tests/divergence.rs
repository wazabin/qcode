//! Finding the first block a compiled run computes differently.
//!
//! The Embench suite reports that several benchmarks verify on the interpreter
//! and not with the JIT installed. Knowing *that* is not enough to fix one: the
//! answer is wrong many millions of operations after the mistake. This narrows
//! it to a single block.
//!
//! Both strategies are run over the same program while a record is kept, at
//! every block entry, of which block it is and what the guest's registers hold.
//! The interpreter is the reference, so the first entry where the two records
//! disagree names the block that ran wrong — and, because registers are
//! compared on the way *in*, the block named is the one that produced the bad
//! value rather than the one that tripped over it.

use qcode::{context::Context, space::MemorySpaceId, value::BlockId};
use qcode_emulator::{EmulatorErrorKind, StandaloneEmulator};
use qcode_jit::Jit;
use qcode_vm::{BlockExecutor, Executed, Vm, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

const SENTINEL: u64 = 0xdead_0000;
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
        let (off, vaddr) = (long(p + 8) as usize, long(p + 16));
        let (filesz, memsz) = (long(p + 32) as usize, long(p + 40) as usize);
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

/// One block entry: where the machine was, and what it held.
#[derive(PartialEq, Eq, Clone)]
struct Entry {
    block: BlockId,
    address: Option<u64>,
    registers: Vec<u8>,
}

/// Records every block entry, and optionally runs the block with a [`Jit`].
///
/// Declining (`Ok(None)`) is what makes the recording-only mode work: the
/// executor is consulted at every block entry, so returning "not mine" leaves
/// the interpreter to run it while the entry is still observed.
struct Recorder {
    jit: Option<Jit>,
    registers: MemorySpaceId,
    log: std::rc::Rc<std::cell::RefCell<Vec<Entry>>>,
}

impl BlockExecutor for Recorder {
    fn run_block(
        &mut self,
        ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        block: BlockId,
        chain: bool,
    ) -> Result<Option<Executed>, EmulatorErrorKind> {
        let registers = emu
            .memory
            .flat_mut()
            .read_bytes(self.registers, 0, 0x300)
            .unwrap_or_default();
        self.log.borrow_mut().push(Entry {
            block,
            address: qcode::value::BasicBlock::from_id(ctx, block).address(),
            registers,
        });
        match self.jit.as_mut() {
            // Never chained: a chained run crosses blocks without being asked
            // again, and those crossings are what has to be observed.
            Some(jit) => jit.run_block(ctx, emu, block, false && chain),
            None => Ok(None),
        }
    }
}

fn trace(image: &[u8], jit: bool, budget: u64) -> Vec<Entry> {
    let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
    let ctx = source.new_context();
    let registers = (0..ctx.space_count())
        .map(qcode::space::SpaceId::from)
        .find(|&id| {
            matches!(
                qcode::space::Space::from_id(&ctx, id).ty,
                qcode::space::SpaceType::Register
            )
        })
        .map(MemorySpaceId::Shared)
        .expect("the specification has a register space");
    let mut memory = VmMemory::new();
    let entry = load(image, &mut memory);
    memory.mmu.map(0x7fff_0000, 0x40000, perm::RW_INIT).unwrap();
    memory
        .mmu
        .write_unchecked(STACK_TOP, &SENTINEL.to_le_bytes(), perm::RW_INIT);

    let mut vm = Vm::at_address(ctx, entry, source, memory).expect("the entry decodes");
    let ctx = vm.context().clone();
    vm.emulator()
        .set_varnode_by_name(&ctx, "RSP", STACK_TOP)
        .expect("RSP is a register");
    let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    vm.set_block_executor(Box::new(Recorder {
        jit: jit.then(Jit::new),
        registers,
        log: log.clone(),
    }));
    vm.run(budget);
    let out = log.borrow().clone();
    out
}

#[test]
#[ignore = "needs benchmarks/embench/build.sh to have been run"]
fn report_first_divergent_block() {
    let name = std::env::var("B").unwrap_or_else(|_| "md5sum".into());
    let path = format!(
        "{}/../target/embench/{name}.elf",
        env!("CARGO_MANIFEST_DIR")
    );
    let image = std::fs::read(&path).expect("image; run benchmarks/embench/build.sh");
    let budget: u64 = std::env::var("BUDGET")
        .ok()
        .and_then(|b| b.parse().ok())
        .unwrap_or(50_000_000);

    let reference = trace(&image, false, budget);
    let compiled = trace(&image, true, budget);
    eprintln!(
        "{name}: {} interpreted entries, {} jitted",
        reference.len(),
        compiled.len()
    );

    for (n, (want, got)) in reference.iter().zip(&compiled).enumerate() {
        if want == got {
            continue;
        }
        eprintln!("first divergence at block entry {n}:");
        eprintln!(
            "  interpreted {:?} addr={:x?}",
            want.block, want.address
        );
        eprintln!("  jitted      {:?} addr={:x?}", got.block, got.address);
        if want.block == got.block {
            for (offset, (a, b)) in want.registers.iter().zip(&got.registers).enumerate() {
                if a != b {
                    eprintln!("  register byte {offset:#x}: interpreted {a:#04x}, jitted {b:#04x}");
                }
            }
            // The block *before* this entry is the one that computed the
            // difference.
            if n > 0 {
                eprintln!(
                    "  produced by {:?} addr={:x?}",
                    reference[n - 1].block,
                    reference[n - 1].address
                );
            }
        }
        return;
    }
    eprintln!("no divergence within the recorded entries");
}
