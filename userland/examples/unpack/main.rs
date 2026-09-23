//! `unpack` — run a static x86-64 Linux ELF as a process and record how it
//! writes its own code.
//!
//! ```text
//! unpack ./selfdecrypt --out /tmp/unpack --jit --edges
//! unpack ./hello --out /tmp/unpack --budget 200000000 --no-hooks
//! ```
//!
//! The run is a real one — [`qcode_userland`]'s loader, SysV stack and
//! system calls, interpreted or compiled — and the two hooks of
//! [`unpack::hooks`] record, in compiled code and without ever stopping the
//! machine, who wrote the bytes each block runs. The `--out` directory holds
//! `context.bin`, `graph.json`, one file per generated region, and the
//! summary below.

#[path = "lib.rs"]
mod unpack;

use std::{path::PathBuf, process, time::Instant};

use clap::Parser;

use unpack::{
    artifact,
    driver::{self, Options},
};

/// Run a static x86-64 ELF as a process and record what it unpacks.
#[derive(Parser)]
#[command(name = "unpack", version)]
struct Cli {
    /// The static x86-64 Linux ELF to run
    elf: String,
    /// Directory the artifact is written to
    #[arg(long, value_name = "DIR")]
    out: Option<PathBuf>,
    /// Install the Cranelift JIT block executor
    #[arg(long)]
    jit: bool,
    /// Maximum number of p-code operations
    #[arg(long, default_value_t = 5_000_000_000)]
    budget: u64,
    /// Run without the provenance and first-entry hooks
    #[arg(long)]
    no_hooks: bool,
    /// Record the observed control-flow edge into each block as well
    #[arg(long)]
    edges: bool,
    /// A file the guest reads as its standard input
    #[arg(long, value_name = "FILE")]
    stdin: Option<PathBuf>,
    /// Resolve the guest's paths under DIR (`/` or absent: the host's own)
    #[arg(long, value_name = "DIR")]
    root: Option<PathBuf>,
    /// Arguments handed to the guest after its own name
    #[arg(trailing_var_arg = true)]
    args: Vec<String>,
}

fn main() {
    let cli = Cli::parse();
    let options = Options {
        jit: cli.jit,
        budget: cli.budget,
        hooks: !cli.no_hooks,
        edges: cli.edges,
        args: cli.args.clone(),
        root: cli.root.clone(),
        stdin: cli.stdin.as_ref().map(|path| {
            std::fs::read(path).unwrap_or_else(|e| {
                eprintln!("error: cannot read {}: {e}", path.display());
                process::exit(1);
            })
        }),
    };

    let started = Instant::now();
    let (mut process, outcome) = match driver::run_keeping(&cli.elf, &options) {
        Ok(kept) => kept,
        Err(message) => {
            eprintln!("error: {message}");
            process::exit(1);
        }
    };
    let elapsed = started.elapsed();

    println!("elf:      {}", cli.elf);
    println!("entry:    {:#x}", outcome.entry);
    println!(
        "image:    {:#x}..{:#x} (+ {:#x} of window)",
        outcome.layout.image_window().start,
        outcome.layout.image_window().end(),
        outcome.layout.mmap_window().len
    );
    println!(
        "strategy: {}",
        if options.jit { "jit" } else { "interpreter" }
    );
    println!("hooks:    {}", if options.hooks { "on" } else { "off" });
    println!("steps:    {}", outcome.steps);
    println!("wall:     {elapsed:.3?}");
    println!("stopped:  {}", outcome.stop_reason);
    match outcome.exit_status {
        Some(status) => println!("exit:     {status}"),
        None => println!("exit:     (the guest never exited)"),
    }
    println!(
        "vm:       absorbed={} native_bodies={} evicted={}",
        outcome.absorbed, outcome.native_bodies, outcome.evicted
    );
    if outcome.recorder.blocks_saturated {
        println!("note:     the first-entry log filled up; blocks went uninstrumented");
    }
    if outcome.recorder.sites_saturated {
        println!("note:     site ids saturated; the last sites share one id");
    }

    if let Some(dir) = &cli.out {
        match artifact::write(dir, &mut process, &outcome) {
            Ok(summary) => println!("{summary}"),
            Err(message) => {
                eprintln!("error: {message}");
                process::exit(1);
            }
        }
    }

    print!("{}", String::from_utf8_lossy(&outcome.stdout));
    eprint!("{}", String::from_utf8_lossy(&outcome.stderr));
}
