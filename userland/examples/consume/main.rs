//! `consume` — run an unchanged static pass over an unpack artifact, and over
//! the file it came from, and count what each sees (experiment E5, claim C4).
//!
//! ```text
//! consume /tmp/unpack/selfdecrypt                 # reads context.bin, graph.json
//! consume /tmp/unpack/selfdecrypt --elf ./selfdecrypt --out e5.json
//! ```
//!
//! The same measurement runs over two [`Context`]s:
//!
//! - **from `context.bin`**: the module the unpack run ended with, decoded as
//!   is — the original program and every generation of generated code, lifted
//!   into one module;
//! - **from the file**: a recursive-descent lift of the ELF alone, without
//!   running it — the loader maps the segments exactly as the run does, and
//!   the lifter the run uses (SLEIGH, flat control flow, one function) lifts
//!   every instruction reachable from the entry through direct branches, direct
//!   calls and fall-throughs, after which straight-line chains are absorbed as
//!   the VM absorbs them.
//!
//! On each, [`qcode_passes::resolve_addresses`] — the workspace's code-reference
//! resolver, run unchanged — annotates every literal that names lifted code,
//! and the CFG is walked with `successors()`, the query the passes' own
//! reachability pruning uses. A *resolved target* is a guest address of lifted
//! code that the IR names: the target of a CFG edge into a non-empty block, or
//! a literal the pass resolved, used by a guest instruction. Targets and blocks
//! are then placed against the generated regions of `graph.json`.
//!
//! The output, `e5.json` by default in the run directory, is
//! `{"pass", "from_context_bin", "from_file", "delta"}`; each side holds
//! `blocks, resolved_targets, targets_in_generated_regions, functions` and a
//! few more counts, and `delta` is `from_context_bin − from_file` per count.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    process,
};

use clap::Parser;
use qcode::{
    address_index::AddressIndex,
    context::Context,
    lift::{CallTarget, Continuation, ExitKind, LiftTarget},
    value::{BasicBlock, BlockId, FunctionBody, ValueId, literal::SymbolicRef},
};
use qcode_userland::{Config, Process, fs::Stdio};
use rustc_hash::FxHashSet;
use serde_json::{Value, json};
use wazabin_qcode_sleigh::{SleighLifter, decode::FixedDecoder};

/// The pass, as `e5.json` names it.
const PASS: &str = "qcode_passes::resolve_addresses + successors() CFG walk";

/// The longest x86-64 instruction, and so the widest fetch.
const MAX_INSN_LEN: usize = 16;

/// Run an unchanged static pass over context.bin and over the original file.
#[derive(Parser)]
#[command(name = "consume", version)]
struct Cli {
    /// The unpack run directory: context.bin and graph.json
    dir: PathBuf,
    /// The original ELF (default: graph.json's `program.path`)
    #[arg(long, value_name = "ELF")]
    elf: Option<PathBuf>,
    /// Where to write the result (default: DIR/e5.json)
    #[arg(long, value_name = "FILE")]
    out: Option<PathBuf>,
    /// Most instructions the static lift of the file visits
    #[arg(long, default_value_t = 1_000_000)]
    max_insns: usize,
}

fn main() {
    let cli = Cli::parse();
    if let Err(message) = run(&cli) {
        eprintln!("error: {message}");
        process::exit(1);
    }
}

fn run(cli: &Cli) -> Result<(), String> {
    let graph_path = cli.dir.join("graph.json");
    let graph: Value = serde_json::from_slice(
        &std::fs::read(&graph_path)
            .map_err(|e| format!("cannot read {}: {e}", graph_path.display()))?,
    )
    .map_err(|e| format!("{}: {e}", graph_path.display()))?;
    let regions = Regions::from_graph(&graph)?;
    let entry = graph["program"]["entry"]
        .as_str()
        .and_then(parse_hex)
        .ok_or("graph.json has no program.entry")?;

    let elf = match &cli.elf {
        Some(path) => path.clone(),
        None => PathBuf::from(
            graph["program"]["path"]
                .as_str()
                .ok_or("graph.json has no program.path; pass --elf")?,
        ),
    };

    let mut dynamic = decode_context(&cli.dir.join("context.bin"))?;
    let from_context_bin = measure(&mut dynamic, entry, &regions);

    let (mut lifted, lift) = lift_file(&elf, cli.max_insns)?;
    if lift.entry != entry {
        return Err(format!(
            "{} enters at {:#x}, the run at {entry:#x}",
            elf.display(),
            lift.entry
        ));
    }
    let from_file = measure(&mut lifted, entry, &regions);

    let delta: BTreeMap<&str, i64> = from_context_bin
        .counts()
        .into_iter()
        .zip(from_file.counts())
        .map(|((key, a), (_, b))| (key, a as i64 - b as i64))
        .collect();
    let report = json!({
        "pass": PASS,
        "elf": elf.display().to_string(),
        "entry": format!("{entry:#x}"),
        "regions": regions.0.iter().map(|(s, e)| json!({"start": format!("{s:#x}"), "end": format!("{e:#x}")})).collect::<Vec<_>>(),
        "from_context_bin": from_context_bin.to_json(),
        "from_file": from_file.to_json(),
        "file_lift": {"instructions": lift.instructions, "undecodable": lift.undecodable},
        "delta": delta,
    });

    let out = cli.out.clone().unwrap_or_else(|| cli.dir.join("e5.json"));
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())? + "\n";
    std::fs::write(&out, &text).map_err(|e| format!("cannot write {}: {e}", out.display()))?;

    println!("pass:     {PASS}");
    println!("elf:      {}", elf.display());
    println!(
        "{:<34}{:>12}{:>12}{:>12}",
        "", "context.bin", "file", "delta"
    );
    for ((key, a), (_, b)) in from_context_bin
        .counts()
        .into_iter()
        .zip(from_file.counts())
    {
        println!("{key:<34}{a:>12}{b:>12}{:>12}", a as i64 - b as i64);
    }
    println!("artifact: {}", out.display());
    Ok(())
}

/// Decodes the artifact's `context.bin`: bincode 2 of the serde [`Context`].
fn decode_context(path: &Path) -> Result<Context<'static>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    bincode::serde::decode_from_slice::<Context<'static>, _>(&bytes, bincode::config::standard())
        .map(|(ctx, _)| ctx)
        .map_err(|e| format!("cannot decode {}: {e}", path.display()))
}

fn parse_hex(text: &str) -> Option<u64> {
    u64::from_str_radix(text.strip_prefix("0x")?, 16).ok()
}

/// The generated regions of `graph.json`, as half-open ranges.
struct Regions(Vec<(u64, u64)>);

impl Regions {
    fn from_graph(graph: &Value) -> Result<Self, String> {
        let list = graph["regions"]
            .as_array()
            .ok_or("graph.json has no regions")?;
        list.iter()
            .map(|region| {
                let field = |name: &str| {
                    region[name]
                        .as_str()
                        .and_then(parse_hex)
                        .ok_or_else(|| format!("a region has no {name}"))
                };
                Ok((field("start")?, field("end")?))
            })
            .collect::<Result<_, _>>()
            .map(Self)
    }

    fn contains(&self, addr: u64) -> bool {
        self.0
            .iter()
            .any(|&(start, end)| (start..end).contains(&addr))
    }
}

/// What the static lift of the file did.
struct FileLift {
    entry: u64,
    instructions: usize,
    undecodable: usize,
}

/// Lifts the ELF at `path` statically, by recursive descent from its entry.
///
/// The image is mapped by the same loader the run uses ([`Process::new`], which
/// maps and sets up the stack but executes nothing), and every instruction goes
/// through the lifter the VM's code source uses, into one function, as the VM
/// places it. Only the lift's own exit metadata drives discovery: a direct
/// branch or call target, a fall-through, a call's continuation. After the
/// walk each addressed block absorbs its straight-line chain, as the VM does
/// once a block is lifted, so both contexts share a block granularity.
fn lift_file(path: &Path, max_insns: usize) -> Result<(Context<'static>, FileLift), String> {
    let image = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let config = Config {
        argv: vec![path.display().to_string()],
        stdio: Stdio::Captured,
        exe_path: path.display().to_string(),
        ..Config::default()
    };
    let mut process =
        Process::new(&image, config).map_err(|e| format!("cannot load {}: {e}", path.display()))?;
    let entry = process.image().entry;
    let vm = process.vm();
    let mmu = &vm.memory().mmu;

    let spec = sleigh_precompile::x64::spec();
    let lifter = SleighLifter::new(spec).with_flat_control_flow();
    let decoder = FixedDecoder::new(spec);
    let mut ctx = lifter.new_context();
    let mut index = AddressIndex::analyze(&ctx);
    let function = FunctionBody::from_addr_or_create_indexed(&mut ctx, &mut index, entry).id;

    let mut seen = FxHashSet::default();
    let mut todo = vec![entry];
    let mut lift = FileLift {
        entry,
        instructions: 0,
        undecodable: 0,
    };
    while let Some(addr) = todo.pop() {
        if lift.instructions >= max_insns || !seen.insert(addr) {
            continue;
        }
        // Shrink the window rather than fault at the end of a mapping, as the
        // VM's code source does.
        let Some(bytes) = (1..=MAX_INSN_LEN).rev().find_map(|len| {
            let mut bytes = vec![0u8; len];
            mmu.read_code(addr, &mut bytes).ok().map(|()| bytes)
        }) else {
            lift.undecodable += 1;
            continue;
        };
        let Ok(instruction) = decoder.decode(addr, &bytes) else {
            lift.undecodable += 1;
            continue;
        };
        let lifted = LiftTarget::bind_indexed(&mut ctx, &mut index, function)
            .map_err(|e| format!("{e:?}"))
            .and_then(|mut target| {
                lifter
                    .lift_into(&mut target, &instruction)
                    .map_err(|e| format!("{e:?}"))
            });
        let Ok(lifted) = lifted else {
            lift.undecodable += 1;
            continue;
        };
        lift.instructions += 1;
        for exit in lifted.exits() {
            match exit.kind() {
                ExitKind::Fallthrough => todo.push(lifted.next_address()),
                ExitKind::Branch { target } => todo.push(*target),
                ExitKind::Call {
                    callee,
                    continuation,
                } => {
                    if let CallTarget::Address(target) = callee {
                        todo.push(*target);
                    }
                    if *continuation == Continuation::Next {
                        todo.push(lifted.next_address());
                    }
                }
                ExitKind::CallInd { continuation } => {
                    if *continuation == Continuation::Next {
                        todo.push(lifted.next_address());
                    }
                }
                ExitKind::BranchInd | ExitKind::Return => {}
            }
        }
    }

    for block in ctx.block_ids() {
        if ctx.contains_block(block) && ctx.block(block).address().is_some() {
            qcode_passes::absorb_straight_line(&mut ctx, block);
        }
    }
    Ok((ctx, lift))
}

/// The counts one context yields.
#[derive(Default)]
struct Facts {
    blocks: usize,
    blocks_in_generated_regions: usize,
    instructions: usize,
    reachable_from_entry: usize,
    edges: usize,
    empty_placeholders: usize,
    code_pointers: usize,
    resolved_targets: usize,
    targets_in_generated_regions: usize,
    functions: usize,
}

impl Facts {
    fn counts(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("blocks", self.blocks),
            (
                "blocks_in_generated_regions",
                self.blocks_in_generated_regions,
            ),
            ("instructions", self.instructions),
            ("reachable_from_entry", self.reachable_from_entry),
            ("edges", self.edges),
            ("empty_placeholders", self.empty_placeholders),
            ("code_pointers", self.code_pointers),
            ("resolved_targets", self.resolved_targets),
            (
                "targets_in_generated_regions",
                self.targets_in_generated_regions,
            ),
            ("functions", self.functions),
        ]
    }

    fn to_json(&self) -> Value {
        Value::Object(
            self.counts()
                .into_iter()
                .map(|(key, n)| (key.to_owned(), json!(n)))
                .collect(),
        )
    }
}

/// Runs the pass over `ctx` and counts what it and the CFG name.
///
/// A *block* is a non-empty block with a guest address: lifted code. An empty
/// addressed block is a placeholder, counted apart: a target the lifter has
/// not lifted yet, or — in `context.bin` — a block the VM evicted when its page
/// was written, and that the run never entered again. Evicted code is not in
/// the module any more, so the entry block of a program that rewrites its own
/// page can be a placeholder, and `reachable_from_entry` then 0. Blocks are units in the sense of the
/// unpack harvest: a successor without an address (an instruction's internal
/// split) belongs to the block it came from, so the walk goes through it.
fn measure(ctx: &mut Context<'static>, entry: u64, regions: &Regions) -> Facts {
    qcode_passes::resolve_addresses(ctx);
    let ctx = &*ctx;

    let mut facts = Facts {
        functions: ctx.functions().count(),
        ..Facts::default()
    };
    let mut targets = BTreeSet::new();
    let mut instructions = BTreeSet::new();
    let mut successors: BTreeMap<BlockId, BTreeSet<BlockId>> = BTreeMap::new();
    let mut at: BTreeMap<u64, BlockId> = BTreeMap::new();

    for block in ctx.blocks() {
        let Some(addr) = block.address() else {
            continue;
        };
        if block.is_empty() {
            facts.empty_placeholders += 1;
            continue;
        }
        at.entry(addr).or_insert(block.id);
        facts.blocks += 1;
        if regions.contains(addr) {
            facts.blocks_in_generated_regions += 1;
        }

        let mut seen = FxHashSet::default();
        let mut todo = vec![block.id];
        while let Some(id) = todo.pop() {
            if !seen.insert(id) {
                continue;
            }
            let view = BasicBlock::from_id(ctx, id);
            for insn in view.instructions() {
                let Some(pc) = insn.address() else {
                    continue;
                };
                instructions.insert(pc);
                // A literal the pass resolved, used by a guest instruction:
                // hook code carries no guest address, so its constants (the
                // window bounds, say) never count.
                for operand in insn.operands() {
                    let ValueId::Literal(literal) = operand else {
                        continue;
                    };
                    let literal = &ctx.shared.values.literals[literal];
                    let named = match literal.symbolic {
                        Some(SymbolicRef::Block(b)) => Some(b),
                        Some(SymbolicRef::Function(f)) => {
                            FunctionBody::from_id(ctx, f).root().map(|root| root.id)
                        }
                        _ => None,
                    };
                    if named.is_some_and(|b| !BasicBlock::from_id(ctx, b).is_empty()) {
                        facts.code_pointers += 1;
                        targets.insert(literal.value);
                    }
                }
            }
            for (_, to) in view.successors() {
                let next = BasicBlock::from_id(ctx, to);
                match next.address() {
                    Some(_) if next.is_empty() => {}
                    Some(target) => {
                        facts.edges += 1;
                        targets.insert(target);
                        successors.entry(block.id).or_default().insert(to);
                    }
                    None => todo.push(to),
                }
            }
        }
    }

    // Reachability from the entry, over lifted blocks only.
    if let Some(&root) = at.get(&entry) {
        let mut seen = FxHashSet::default();
        let mut todo = vec![root];
        while let Some(id) = todo.pop() {
            if seen.insert(id) {
                todo.extend(successors.get(&id).into_iter().flatten().copied());
            }
        }
        facts.reachable_from_entry = seen.len();
    }

    facts.instructions = instructions.len();
    facts.resolved_targets = targets.len();
    facts.targets_in_generated_regions = targets.iter().filter(|&&t| regions.contains(t)).count();
    facts
}
