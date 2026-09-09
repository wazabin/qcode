//! `run-elf [--jit] [--trace] [--budget N] [--root DIR] PROG [ARGS...]`
//!
//! Runs a static x86-64 Linux executable under the emulator and exits with the
//! guest's status.

use std::path::PathBuf;
use std::process::ExitCode;

use qcode_userland::fs::Stdio;
use qcode_userland::{Config, Process, ProcessExit};

const HELP: &str = "\
usage: run-elf [--jit] [--trace] [--budget N] [--root DIR] [--env K=V]... PROG [ARGS...]

  --jit         execute with the Cranelift JIT instead of the interpreter
  --trace       print one line per system call to stderr
  --budget N    stop after N p-code operations (default: unlimited)
  --root DIR    resolve guest paths under DIR
  --env K=V     add an environment variable (default: none)
";

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let mut jit = false;
    let mut trace = false;
    let mut budget = u64::MAX;
    let mut root = None;
    let mut envp = Vec::new();
    let mut rest = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--jit" => jit = true,
            "--trace" => trace = true,
            "--budget" => match args.next().and_then(|n| n.parse().ok()) {
                Some(n) => budget = n,
                None => return usage("--budget needs a number"),
            },
            "--root" => match args.next() {
                Some(dir) => root = Some(PathBuf::from(dir)),
                None => return usage("--root needs a directory"),
            },
            "--env" => match args.next() {
                Some(kv) => envp.push(kv),
                None => return usage("--env needs K=V"),
            },
            "-h" | "--help" => {
                print!("{HELP}");
                return ExitCode::SUCCESS;
            }
            _ if arg.starts_with('-') && rest.is_empty() => {
                return usage(&format!("unknown option {arg}"));
            }
            _ => {
                rest.push(arg);
                rest.extend(args.by_ref());
            }
        }
    }
    if rest.is_empty() {
        return usage("missing PROG");
    }

    let image = match std::fs::read(&rest[0]) {
        Ok(image) => image,
        Err(e) => {
            eprintln!("run-elf: cannot read {}: {e}", rest[0]);
            return ExitCode::from(126);
        }
    };
    let exe_path = std::fs::canonicalize(&rest[0])
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| rest[0].clone());
    let config = Config {
        argv: rest,
        envp,
        jit,
        trace,
        root,
        stdio: Stdio::Host,
        exe_path,
    };
    let mut process = match Process::new(&image, config) {
        Ok(process) => process,
        Err(e) => {
            eprintln!("run-elf: {e}");
            return ExitCode::from(126);
        }
    };
    match process.run(budget) {
        ProcessExit::Exited(code) => ExitCode::from(code as u8),
        ProcessExit::Crashed(crash) => {
            eprintln!("run-elf: {crash}");
            ExitCode::from(crash.signal.map_or(125, |s| 128 + s as u8))
        }
        ProcessExit::Budget => {
            eprintln!("run-elf: budget of {budget} operations exhausted");
            ExitCode::from(124)
        }
    }
}

fn usage(message: &str) -> ExitCode {
    eprintln!("run-elf: {message}\n\n{HELP}");
    ExitCode::from(2)
}
