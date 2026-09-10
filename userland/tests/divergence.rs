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
use qcode_userland::bare;
use qcode_vm::{BlockExecutor, Executed, VmMemory};

mod support;

/// One block entry: where the machine was, and what it held.
///
/// The register file is kept as a digest rather than a copy. A run reaches
/// millions of block entries, and a few hundred bytes each would be gigabytes;
/// a mismatching digest is enough to find *where*, and the second pass goes
/// back for the bytes.
#[derive(PartialEq, Eq, Clone, Copy)]
struct Entry {
    block: BlockId,
    address: Option<u64>,
    registers: u64,
}

/// Register bytes held back for the second pass.
type Kept = std::rc::Rc<std::cell::RefCell<Vec<Vec<u8>>>>;

/// FNV-1a. Any digest would do; this one needs no dependency.
fn digest(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
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
    /// Stop recording past this many entries, so a long run cannot exhaust
    /// memory. The run itself continues.
    limit: usize,
    /// When set, the full register bytes for these entries are kept too.
    detail: Option<(usize, Kept)>,
    seen: usize,
}

impl BlockExecutor for Recorder {
    fn run_block(
        &mut self,
        ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        block: BlockId,
        start: usize,
        chain: bool,
    ) -> Result<Option<Executed>, EmulatorErrorKind> {
        let bytes = emu
            .memory
            .flat_mut()
            .read_bytes(self.registers, 0, 0x300)
            .unwrap_or_default();
        let index = self.seen;
        self.seen += 1;
        if index < self.limit {
            self.log.borrow_mut().push(Entry {
                block,
                address: qcode::value::BasicBlock::from_id(ctx, block).address(),
                registers: digest(&bytes),
            });
        }
        if let Some((around, kept)) = &self.detail
            && index + 1 >= *around
            && index <= *around
        {
            kept.borrow_mut().push(bytes);
        }
        match self.jit.as_mut() {
            // Never chained, whatever the caller allows: a chained run crosses
            // blocks without being asked again, and those crossings are exactly
            // what has to be observed. This isolates the compiled code from the
            // decision to stay in it.
            Some(jit) => {
                let _ = chain;
                jit.run_block(ctx, emu, block, start, false)
            }
            None => Ok(None),
        }
    }
}

/// Runs `image` once, recording every block entry.
///
/// `detail` names an entry index whose register bytes — and its predecessor's —
/// are kept alongside the digests, for the second pass.
fn trace(
    image: &[u8],
    jit: bool,
    budget: u64,
    detail: Option<usize>,
) -> (Vec<Entry>, Vec<Vec<u8>>) {
    let mut vm = bare::machine(image).expect("the image loads and its entry decodes");
    let ctx = vm.context().clone();
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
    let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let kept = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    vm.set_block_executor(Box::new(Recorder {
        jit: jit.then(Jit::new),
        registers,
        log: log.clone(),
        limit: MAX_ENTRIES,
        detail: detail.map(|at| (at, kept.clone())),
        seen: 0,
    }));
    vm.run(budget);
    let entries = log.borrow().clone();
    let bytes = kept.borrow().clone();
    (entries, bytes)
}

/// How many block entries are recorded before a run stops being followed.
const MAX_ENTRIES: usize = 8_000_000;

/// Reports the first block entry at which the two strategies disagree.
fn compare(name: &str, image: &[u8], budget: u64) -> bool {
    let (reference, _) = trace(image, false, budget, None);
    let (compiled, _) = trace(image, true, budget, None);

    let divergence = reference
        .iter()
        .zip(&compiled)
        .position(|(want, got)| want != got);

    let Some(n) = divergence else {
        let same = reference.len() == compiled.len();
        eprintln!(
            "{name:16} {} entries, agree{}",
            reference.len(),
            if same {
                String::new()
            } else {
                format!(" (but {} vs {} entries)", reference.len(), compiled.len())
            }
        );
        if reference.len() >= MAX_ENTRIES {
            eprintln!("                 (recording stopped at the {MAX_ENTRIES} entry cap)");
        }
        return same;
    };

    let (want, got) = (&reference[n], &compiled[n]);
    eprintln!("{name:16} DIVERGES at block entry {n}");
    eprintln!(
        "                 interpreted {:?} addr={:x?}",
        want.block, want.address
    );
    eprintln!(
        "                 jitted      {:?} addr={:x?}",
        got.block, got.address
    );
    if n > 0 {
        eprintln!(
            "                 produced by {:?} addr={:x?}",
            reference[n - 1].block,
            reference[n - 1].address
        );
    }
    if want.block == got.block {
        // Go back for the register bytes at just this entry.
        let (_, a) = trace(image, false, budget, Some(n));
        let (_, b) = trace(image, true, budget, Some(n));
        if let (Some(a), Some(b)) = (a.last(), b.last()) {
            for (offset, (x, y)) in a.iter().zip(b).enumerate() {
                if x != y {
                    eprintln!(
                        "                 register byte {offset:#05x}: interpreted {x:#04x}, jitted {y:#04x}"
                    );
                }
            }
        }
    }
    false
}

/// Every built benchmark, or just `B` when it names one.
///
/// This is the check the suite's own pass/fail cannot give: a benchmark that
/// verifies has agreed on one number at the end, while this agrees on the whole
/// register file at every block boundary.
#[test]
#[ignore = "needs benchmarks/embench/build.sh to have been run"]
fn no_program_diverges_under_the_jit() {
    let budget: u64 = std::env::var("BUDGET")
        .ok()
        .and_then(|b| b.parse().ok())
        .unwrap_or(4_000_000_000);
    let only = std::env::var("B").ok();

    let images = support::images();
    assert!(
        !images.is_empty(),
        "no images; run benchmarks/embench/build.sh"
    );

    let mut diverged = Vec::new();
    for (name, image) in images {
        if only.as_ref().is_some_and(|want| *want != name) {
            continue;
        }
        if !compare(&name, &image, budget) {
            diverged.push(name);
        }
    }
    assert!(diverged.is_empty(), "diverged under the JIT: {diverged:#?}");
}
