//! `run <ELF> --out DIR [--edges] [--no-hooks] [--stdin FILE] [--budget N]
//! [--reps N] [--interp] -- args...`
//!
//! `--reps N` times N runs after one discarded warm-up (none when N is 1),
//! each on a fresh machine; the artifact comes from the last. `--budget N`
//! caps guest *instructions* (QCode's is p-code operations), and is enforced
//! only where the engine does it for free (icicle's fuel) or when given
//! explicitly (Unicorn and Qiling pay a per-instruction hook for it).
//! `--jit` and `--root /` are accepted and ignored: the host filesystem is the
//! guest's.

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Cli {
    pub elf: String,
    pub out: PathBuf,
    pub edges: bool,
    pub hooks: bool,
    pub stdin: Vec<u8>,
    pub budget: Option<u64>,
    pub reps: usize,
    pub interp: bool,
    pub args: Vec<String>,
}

pub fn parse() -> Cli {
    let usage = "usage: run <ELF> --out DIR [--edges] [--no-hooks] [--stdin FILE] [--budget N] [--reps N] [--interp] -- args...";
    let mut it = std::env::args().skip(1);
    let mut elf = None;
    let mut out = None;
    let mut cli = Cli {
        elf: String::new(),
        out: PathBuf::new(),
        edges: false,
        hooks: true,
        stdin: Vec::new(),
        budget: None,
        reps: 1,
        interp: false,
        args: Vec::new(),
    };
    let die = |m: &str| -> ! {
        eprintln!("error: {m}\n{usage}");
        std::process::exit(2)
    };
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(it.next().unwrap_or_else(|| die("--out needs a value")))),
            "--edges" => cli.edges = true,
            "--no-hooks" => cli.hooks = false,
            "--jit" => {}
            "--interp" => cli.interp = true,
            "--root" => {
                it.next();
            }
            "--stdin" => {
                let p = it.next().unwrap_or_else(|| die("--stdin needs a value"));
                cli.stdin = std::fs::read(&p).unwrap_or_else(|e| die(&format!("cannot read {p}: {e}")));
            }
            "--budget" => {
                cli.budget = Some(it.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| die("--budget needs a number")))
            }
            "--reps" => {
                cli.reps = it.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| die("--reps needs a number"))
            }
            "--" => {
                cli.args.extend(it.by_ref());
            }
            "-h" | "--help" => {
                println!("{usage}");
                std::process::exit(0)
            }
            s if s.starts_with("--") => die(&format!("unknown option {s}")),
            s => {
                if elf.is_none() {
                    elf = Some(s.to_string())
                } else {
                    cli.args.push(s.to_string())
                }
            }
        }
    }
    cli.elf = elf.unwrap_or_else(|| die("no ELF given"));
    cli.out = out.unwrap_or_else(|| die("--out is required"));
    cli.reps = cli.reps.max(1);
    cli
}
