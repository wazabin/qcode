//! Instrumentation under QCode, Unicorn and icicle-emu on the Embench images.
//!
//! One row per (engine, strategy, instrumentation, image): the wall time of
//! the run, the number of times the host was entered on the hook's behalf,
//! and whether the benchmark verified itself.

mod elf;
mod qcode_backend;
#[cfg(feature = "icicle")]
mod icicle_backend;
#[cfg(feature = "unicorn")]
mod unicorn_backend;

use std::path::PathBuf;
use std::time::Duration;

/// Where the entry returns to: unmapped under QCode, a mapped `until` page
/// under Unicorn, an execute-violation under icicle.
pub const SENTINEL: u64 = qcode_userland::bare::SENTINEL;
pub const STACK: u64 = qcode_userland::bare::STACK;
pub const STACK_SIZE: u64 = qcode_userland::bare::STACK_SIZE;
pub const STACK_TOP: u64 = qcode_userland::bare::STACK_TOP;
/// Bytes watched for writes: the start of the image's first writable segment.
pub const WATCH_LEN: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Instr {
    /// No instrumentation.
    None,
    /// A block counter in a flat hook space, as IR at every block entry.
    BlockIr,
    /// The same counter kept in guest RAM, through the MMU.
    BlockRam,
    /// A host callback at every block entry.
    BlockCb,
    /// An instruction counter in a flat hook space, as IR before every
    /// guest instruction.
    InsnIr,
    /// The same counter kept in guest RAM, through the MMU.
    InsnRam,
    /// A host callback before every guest instruction.
    InsnCb,
    /// An AFL-style edge map in a bounded hook space, as IR at every block
    /// entry.
    EdgeIr,
    /// The same map kept in guest RAM, through the MMU.
    EdgeRam,
    /// Stores to a 64-byte range: the range check as IR, the host only on a
    /// hit.
    WatchIr,
    /// Stores to a 64-byte range: the host at every store, checking the
    /// range itself.
    WatchCb,
    /// Every integer comparison's operands logged to a ring buffer in a
    /// bounded hook space, as IR.
    CmpIr,
    /// The same log kept in guest RAM, through the MMU.
    CmpRam,
    /// Every integer comparison's operands handed to the host.
    CmpCb,
}

impl Instr {
    pub const ALL: [Instr; 14] = [
        Instr::None,
        Instr::BlockIr,
        Instr::BlockRam,
        Instr::BlockCb,
        Instr::InsnIr,
        Instr::InsnRam,
        Instr::InsnCb,
        Instr::EdgeIr,
        Instr::EdgeRam,
        Instr::WatchIr,
        Instr::WatchCb,
        Instr::CmpIr,
        Instr::CmpRam,
        Instr::CmpCb,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Instr::None => "none",
            Instr::BlockIr => "block-ir",
            Instr::BlockRam => "block-ram",
            Instr::BlockCb => "block-cb",
            Instr::InsnIr => "insn-ir",
            Instr::InsnRam => "insn-ram",
            Instr::InsnCb => "insn-cb",
            Instr::EdgeIr => "edge-ir",
            Instr::EdgeRam => "edge-ram",
            Instr::WatchIr => "watch-ir",
            Instr::WatchCb => "watch-cb",
            Instr::CmpIr => "cmp-ir",
            Instr::CmpRam => "cmp-ram",
            Instr::CmpCb => "cmp-cb",
        }
    }
    fn parse(s: &str) -> Option<Instr> {
        Instr::ALL.into_iter().find(|i| i.name() == s)
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Row {
    pub engine: String,
    pub instr: Instr,
    pub image: String,
    pub verified: bool,
    pub exit: String,
    pub elapsed_ns: u128,
    /// Times control left compiled or interpreted guest code for the host on
    /// the hook's behalf.
    pub host_calls: u64,
    /// The events the instrumentation counted: blocks, instructions,
    /// watched stores, comparisons — whatever the kind counts.
    pub events: u64,
    /// Sites the hook instrumented (QCode only).
    pub sites: u64,
}

pub struct Outcome {
    pub verified: bool,
    pub exit: String,
    pub elapsed: Duration,
    pub host_calls: u64,
    pub events: u64,
    pub sites: u64,
}

fn images(dir: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<_> = std::fs::read_dir(dir)
        .expect("image directory")
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "elf"))
        .map(|e| {
            (
                e.path().file_stem().unwrap().to_string_lossy().into_owned(),
                std::fs::read(e.path()).unwrap(),
            )
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut engines: Vec<String> = Vec::new();
    let mut instrs: Vec<Instr> = Vec::new();
    let mut dir = PathBuf::from("../../target/embench");
    let mut only: Option<String> = None;
    let mut repeat = 1usize;
    let mut json: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--engine" => engines.push(args.next().unwrap()),
            "--instr" => instrs.push(Instr::parse(&args.next().unwrap()).expect("instrumentation")),
            "--images" => dir = PathBuf::from(args.next().unwrap()),
            "--only" => only = Some(args.next().unwrap()),
            "--repeat" => repeat = args.next().unwrap().parse().unwrap(),
            "--json" => json = Some(PathBuf::from(args.next().unwrap())),
            other => panic!("unknown argument {other}"),
        }
    }
    if engines.is_empty() {
        engines = vec!["qcode-interp".into(), "qcode-jit".into()];
        #[cfg(feature = "unicorn")]
        engines.push("unicorn".into());
        #[cfg(feature = "icicle")]
        engines.push("icicle".into());
    }
    if instrs.is_empty() {
        instrs = Instr::ALL.to_vec();
    }
    let images = images(&dir);
    assert!(!images.is_empty(), "no images in {}", dir.display());

    let mut rows: Vec<Row> = Vec::new();
    println!(
        "{:<13} {:<9} {:<16} {:>10} {:>12} {:>12} {:>8} ok",
        "engine", "instr", "image", "time", "host-calls", "events", "sites"
    );
    for engine in &engines {
        for &instr in &instrs {
            for (name, image) in &images {
                if only.as_ref().is_some_and(|o| o != name) {
                    continue;
                }
                let mut best: Option<Outcome> = None;
                for _ in 0..repeat {
                    let outcome = match engine.as_str() {
                        "qcode-interp" => qcode_backend::run(image, false, instr),
                        "qcode-jit" => qcode_backend::run(image, true, instr),
                        #[cfg(feature = "unicorn")]
                        "unicorn" => match unicorn_backend::run(image, instr) {
                            Some(o) => o,
                            None => continue,
                        },
                        #[cfg(feature = "icicle")]
                        "icicle" => match icicle_backend::run(image, instr) {
                            Some(o) => o,
                            None => continue,
                        },
                        other => panic!("unknown engine {other}"),
                    };
                    if best.as_ref().is_none_or(|b| outcome.elapsed < b.elapsed) {
                        best = Some(outcome);
                    }
                }
                let Some(o) = best else { continue };
                println!(
                    "{:<13} {:<9} {:<16} {:>10.1?} {:>12} {:>12} {:>8} {}",
                    engine,
                    instr.name(),
                    name,
                    o.elapsed,
                    o.host_calls,
                    o.events,
                    o.sites,
                    if o.verified { "ok" } else { &o.exit }
                );
                rows.push(Row {
                    engine: engine.clone(),
                    instr,
                    image: name.clone(),
                    verified: o.verified,
                    exit: o.exit,
                    elapsed_ns: o.elapsed.as_nanos(),
                    host_calls: o.host_calls,
                    events: o.events,
                    sites: o.sites,
                });
            }
        }
    }
    if let Some(path) = json {
        std::fs::write(&path, serde_json::to_string_pretty(&rows).unwrap()).unwrap();
        eprintln!("wrote {}", path.display());
    }
}
