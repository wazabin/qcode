//! What the unpackbench baselines share, so that Unicorn and icicle differ in
//! their hooks and nothing else.
//!
//! - [`elf`] and [`kernel`]: a minimal static-ELF loader, the SysV stack and
//!   the Linux system calls the corpus needs, over the [`Guest`] trait each
//!   engine implements. Modelled on `userland/` (same stack top, same mmap
//!   base, same `MAP_SHARED` write-back), so that the windows provenance
//!   tracks land on the same guest addresses as under QCode.
//! - [`record`]: the provenance windows and the shadow layout (the unpack
//!   example's `layout.rs`, restated), and the facts a run's hooks collect.
//! - [`harvest`]: the unpack example's harvest over those facts, writing
//!   `graph.json`, `regions/` and `meta.json` in the contract's schema.
//! - [`cli`]: the `run <ELF> --out DIR ...` command line all drivers accept.
//!
//! None of this is hook code: `hook_source_lines` counts only what sits
//! between `HOOK-BEGIN` and `HOOK-END` markers ([`hook_lines`]).

pub mod cli;
pub mod elf;
pub mod harvest;
pub mod kernel;
pub mod record;

/// The x86-64 registers the loader and the system calls touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reg {
    Rax,
    Rdi,
    Rsi,
    Rdx,
    R10,
    R8,
    R9,
    Rsp,
    Rip,
    FsBase,
    GsBase,
}

/// One engine's machine, as the loader and the kernel see it.
///
/// Memory is always mapped read-write-execute: permissions are not what the
/// baselines are measured on, and the unpackers only ever tighten them.
pub trait Guest {
    fn read(&mut self, addr: u64, buf: &mut [u8]) -> bool;
    fn write(&mut self, addr: u64, data: &[u8]) -> bool;
    /// Maps `len` zeroed bytes at `addr`; both page aligned, the range free.
    fn map(&mut self, addr: u64, len: u64) -> bool;
    /// Unmaps `len` bytes at `addr`; the range is wholly mapped.
    fn unmap(&mut self, addr: u64, len: u64);
    fn reg(&mut self, reg: Reg) -> u64;
    fn set_reg(&mut self, reg: Reg, value: u64);
}

/// Counts the hook code of a driver: the non-blank, non-comment lines between
/// `// HOOK-BEGIN` and `// HOOK-END` markers of each source.
pub fn hook_lines(sources: &[&str]) -> usize {
    let mut count = 0;
    for source in sources {
        let mut inside = false;
        for line in source.lines() {
            let t = line.trim();
            if t.starts_with("// HOOK-BEGIN") || t.starts_with("# HOOK-BEGIN") {
                inside = true;
                continue;
            }
            if t.starts_with("// HOOK-END") || t.starts_with("# HOOK-END") {
                inside = false;
                continue;
            }
            let comment = t.starts_with("//") || t.starts_with("# ") || t == "#";
            if inside && !t.is_empty() && !comment {
                count += 1;
            }
        }
    }
    count
}

/// One run of an engine, as [`drive`] needs it.
pub struct Outcome {
    pub run: harvest::Run,
    /// The run proper: first guest instruction to exit.
    pub wall_ms: f64,
    /// Building the machine and loading the image.
    pub setup_ms: f64,
    /// Reads the final guest memory, for the regions' bytes.
    pub read: Box<dyn FnMut(u64, &mut [u8]) -> bool>,
    /// Set when the run hit something the engine cannot do.
    pub unsupported: Option<String>,
    /// Engine-specific counters for `meta.json`.
    pub extra: serde_json::Value,
}

/// The whole driver: parse the command line, run `reps` times after one
/// warm-up (when `reps > 1`), and write the artifact of the last run.
pub fn drive(engine: &str, hook_source_lines: usize, mut run: impl FnMut(&cli::Cli, &[u8]) -> Result<Outcome, String>) {
    let cli = cli::parse();
    let bytes = std::fs::read(&cli.elf).unwrap_or_else(|e| {
        eprintln!("error: cannot read {}: {e}", cli.elf);
        std::process::exit(1)
    });
    let base_meta = |jit: bool| {
        serde_json::json!({
            "engine": engine,
            "config": {"jit": jit, "hooks": cli.hooks, "edges": cli.edges, "budget": cli.budget},
            "hook_source_lines": hook_source_lines,
        })
    };
    let total = if cli.reps > 1 { cli.reps + 1 } else { 1 };
    let mut walls = Vec::new();
    let mut steps = Vec::new();
    let mut setups = Vec::new();
    let mut last = None;
    for i in 0..total {
        match run(&cli, &bytes) {
            Err(reason) => {
                let meta = base_meta(!cli.interp);
                harvest::write_unsupported(&cli.out, meta, &reason, reason.as_bytes()).unwrap_or_else(|e| eprintln!("error: {e}"));
                eprintln!("unsupported: {reason}");
                std::process::exit(0);
            }
            Ok(o) => {
                if let Some(reason) = &o.unsupported {
                    let mut meta = base_meta(o.run.strategy == "jit");
                    meta["stop_reason"] = serde_json::json!(o.run.stop_reason);
                    let mut err = o.run.stderr.clone();
                    err.extend_from_slice(format!("\nunsupported: {reason}\n").as_bytes());
                    harvest::write_unsupported(&cli.out, meta, reason, &err).unwrap_or_else(|e| eprintln!("error: {e}"));
                    eprintln!("unsupported: {reason}");
                    std::process::exit(0);
                }
                if total > 1 && i == 0 {
                    continue; // the warm-up
                }
                walls.push(o.wall_ms);
                setups.push(o.setup_ms);
                steps.push(o.run.steps);
                last = Some(o);
            }
        }
    }
    let mut o = last.expect("at least one run");
    let mut meta = base_meta(o.run.strategy == "jit");
    meta["reps"] = serde_json::json!(walls.len());
    meta["wall_ms"] = serde_json::json!(walls);
    meta["steps"] = serde_json::json!(steps);
    meta["setup_ms"] = serde_json::json!(setups);
    meta["wall_scope"] = serde_json::json!("run only: machine setup and image load are in setup_ms, harvest excluded");
    meta["vm"] = o.extra.take();
    let run = &o.run;
    if let Err(e) = harvest::write(&cli.out, run, &mut o.read, meta) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    let graph: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(cli.out.join("graph.json")).unwrap()).unwrap();
    println!("engine:   {engine} ({})", run.strategy);
    println!("stopped:  {}", run.stop_reason);
    println!("exit:     {:?}", run.exit);
    println!("wall:     {:.1} ms (median of {})", harvest::median(&walls), walls.len());
    println!("nodes:    {}", graph["nodes"].as_array().map_or(0, |v| v.len()));
    println!("sites:    {}", graph["sites"].as_array().map_or(0, |v| v.len()));
    for r in graph["regions"].as_array().into_iter().flatten() {
        println!("region:   {}..{} gen {} ({} bytes)", r["start"].as_str().unwrap(), r["end"].as_str().unwrap(), r["generation"], r["bytes_len"]);
    }
    for w in graph["warnings"].as_array().into_iter().flatten() {
        println!("warning:  {}", w.as_str().unwrap_or(""));
    }
    print!("{}", String::from_utf8_lossy(&run.stdout));
}
