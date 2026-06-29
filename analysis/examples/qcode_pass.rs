//! A small UNIX-style filter that runs qcode optimization passes on textual IR.
//!
//! It reads a qcode program (the same syntax the `qcode!` macro accepts) from
//! stdin, runs one or more named passes over every function, and writes the
//! resulting IR to stdout — so it composes with pipes and redirection:
//!
//! ```text
//! qcode-pass -p loop_to_recursion < fib.qcode
//! cat fib.qcode | qcode-pass -p mem2reg -p gvn -p dce | tee out.qcode
//! qcode-pass --list                 # show every registered pass
//! qcode-pass -p gvn -i in.qcode -o out.qcode
//! ```
//!
//! Run via cargo:
//! ```text
//! cargo run -p qcode_analysis --example qcode-pass -- -p loop_to_recursion < fib.qcode
//! ```

use std::fs;
use std::io::{Read, Write};
use std::process::ExitCode;

use qcode::context::Context;
use qcode_analysis::{PipelineEnv, RegisteredPass, known_pass_names, make_pass};

struct Args {
    passes: Vec<String>,
    input: Option<String>,
    output: Option<String>,
    list: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        passes: Vec::new(),
        input: None,
        output: None,
        list: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-p" | "--pass" => {
                args.passes.push(it.next().ok_or("expected a pass name after -p")?);
            }
            "-i" | "--input" => {
                args.input = Some(it.next().ok_or("expected a path after -i")?);
            }
            "-o" | "--output" => {
                args.output = Some(it.next().ok_or("expected a path after -o")?);
            }
            "-l" | "--list" => args.list = true,
            "-h" | "--help" => {
                return Err(HELP.to_string());
            }
            other => return Err(format!("unknown argument `{other}`\n\n{HELP}")),
        }
    }
    Ok(args)
}

const HELP: &str = "\
qcode-pass — run qcode passes on textual IR (stdin -> stdout)

USAGE:
    qcode-pass -p <PASS> [-p <PASS> ...] [-i <FILE>] [-o <FILE>]
    qcode-pass --list

OPTIONS:
    -p, --pass <NAME>    Pass to run (repeatable; run in order over every function)
    -i, --input <FILE>   Read IR from FILE instead of stdin
    -o, --output <FILE>  Write IR to FILE instead of stdout
    -l, --list           List all registered passes and exit
    -h, --help           Show this help";

fn run() -> Result<(), String> {
    let args = parse_args()?;

    if args.list {
        println!("Available passes:\n  {}", known_pass_names().replace(", ", "\n  "));
        return Ok(());
    }
    if args.passes.is_empty() {
        return Err(format!("no pass selected; pass -p <NAME> (or --list)\n\n{HELP}"));
    }

    let source = match &args.input {
        Some(path) => fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?,
        None => {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .map_err(|e| format!("reading stdin: {e}"))?;
            buf
        }
    };

    let mut ctx = Context::new();
    qcode::lower::lower_str(&mut ctx, &source).map_err(|e| format!("parse/lower: {e}"))?;

    let env = PipelineEnv::headless(&mut ctx);
    for pass in &args.passes {
        let resolved =
            make_pass(pass).ok_or_else(|| format!("unknown pass `{pass}`; known: {}", known_pass_names()))?;
        match resolved {
            RegisteredPass::Function(p) => {
                for fun_id in ctx.function_ids() {
                    p.run(&mut ctx, fun_id, &env)
                        .map_err(|e| format!("pass `{pass}` failed: {e}"))?;
                }
            }
            RegisteredPass::Module(p) => {
                p.run(&mut ctx, &env)
                    .map_err(|e| format!("pass `{pass}` failed: {e}"))?;
            }
        }
    }

    let rendered = format!("{ctx}");
    match &args.output {
        Some(path) => fs::write(path, rendered).map_err(|e| format!("writing {path}: {e}"))?,
        None => {
            std::io::stdout()
                .write_all(rendered.as_bytes())
                .map_err(|e| format!("writing stdout: {e}"))?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("qcode-pass: {e}");
            ExitCode::FAILURE
        }
    }
}
