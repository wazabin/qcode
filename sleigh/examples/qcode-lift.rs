//! `qcode-lift` — decode bytes with an embedded SLEIGH specification and lift
//! them to QCode.
//!
//! Arguments come either from flags or from one JSON object whose keys are
//! the flag names; `--json` switches the output to JSON.
//!
//! ```text
//! qcode-lift --arch x64 --address 0x401000 4889d84801c8c3
//! qcode-lift --args '{"bytes":"4889d84801c8c3","arch":"x64","address":"0x401000"}' --json
//! qcode-lift --arch aarch64 --file firmware.bin --offset 0x100 --count 20 --passes
//! ```

use clap::Parser;
use qcode::{address_index::AddressIndex, value::function::FunctionBody};
use serde::{Deserialize, Serialize};
use sleigh::{CompiledSpec, Decoder};
use std::{fs, process};
use wazabin_qcode_sleigh::SleighLifter;

/// Decode bytes with an embedded SLEIGH specification and lift them to QCode.
#[derive(Parser)]
#[command(name = "qcode-lift", version)]
struct Cli {
    /// All options as one JSON object; keys are the long flag names
    #[arg(long, value_name = "JSON", conflicts_with_all = ["bytes", "arch", "address", "file", "offset", "count", "passes"])]
    args: Option<String>,
    /// Emit JSON instead of text
    #[arg(long)]
    json: bool,
    #[command(flatten)]
    opts: Opts,
}

/// The options proper: one struct for both the flags and the `--args` JSON.
#[derive(clap::Args, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct Opts {
    /// Instruction bytes as hex
    bytes: Option<String>,
    /// Architecture: x64, x86, aarch64, riscv
    #[arg(long, default_value = "x64")]
    #[serde(default = "default_arch")]
    arch: String,
    /// Address of the first instruction (decimal or 0x-prefixed)
    #[arg(long, default_value = "0x1000")]
    #[serde(default = "default_address")]
    address: String,
    /// Read the bytes from this file instead of the command line
    #[arg(long, conflicts_with = "bytes")]
    file: Option<String>,
    /// Byte offset into --file (decimal or 0x-prefixed)
    #[arg(long, default_value = "0", requires = "file")]
    offset: String,
    /// Stop after this many instructions
    #[arg(long)]
    count: Option<usize>,
    /// Run the qcode_passes cleanup (e.g. remove_dead_insns) after lifting
    #[arg(long)]
    passes: bool,
}

fn default_arch() -> String {
    "x64".to_string()
}

fn default_address() -> String {
    "0x1000".to_string()
}

#[derive(Serialize)]
struct Output {
    arch: String,
    instructions: Vec<Insn>,
    #[serde(skip_serializing_if = "Option::is_none")]
    functions: Option<Vec<Func>>,
    qcode: String,
}

#[derive(Serialize)]
struct Insn {
    address: String,
    bytes: String,
    text: String,
}

#[derive(Serialize)]
struct Func {
    name: String,
    blocks: Vec<Block>,
}

#[derive(Serialize)]
struct Block {
    name: String,
    text: String,
}

fn spec_for(arch: &str) -> Result<&'static CompiledSpec, String> {
    Ok(match arch {
        "x64" => sleigh_precompile::x64::spec(),
        "x86" => sleigh_precompile::x86::spec(),
        "aarch64" => sleigh_precompile::aarch64::spec(),
        "riscv" => sleigh_precompile::riscv::spec(),
        other => return Err(format!("unknown arch '{other}' (x64, x86, aarch64, riscv)")),
    })
}

fn parse_int(what: &str, value: &str) -> Result<u64, String> {
    let value = value.trim();
    match value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => value.parse(),
    }
    .map_err(|_| format!("invalid {what} '{value}'"))
}

fn parse_hex(value: &str) -> Result<Vec<u8>, String> {
    let value: String = value.chars().filter(|c| !c.is_whitespace()).collect();
    if value.is_empty() || value.len() % 2 != 0 {
        return Err("bytes must be a non-empty, even-length hex string".to_string());
    }
    (0..value.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&value[i..i + 2], 16)
                .map_err(|_| format!("invalid hex byte '{}'", &value[i..i + 2]))
        })
        .collect()
}

fn run(opts: &Opts) -> Result<Output, String> {
    let spec = spec_for(&opts.arch)?;
    let address = parse_int("address", &opts.address)?;
    let data = match (&opts.bytes, &opts.file) {
        (Some(hex), None) => parse_hex(hex)?,
        (None, Some(path)) => {
            let offset = parse_int("offset", &opts.offset)? as usize;
            let file = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
            file.get(offset..)
                .ok_or_else(|| format!("offset {offset:#x} is past the end of {path}"))?
                .to_vec()
        }
        (Some(_), Some(_)) => return Err("give either bytes or file, not both".to_string()),
        (None, None) => return Err("give bytes or --file".to_string()),
    };

    let decoder = Decoder::new(spec);
    let lifter = SleighLifter::new(spec);
    let mut context = lifter.new_context();
    let mut addresses = AddressIndex::analyze(&context);
    // One function gathers every instruction of the walk, as a caller lifting a
    // known body would want; without it the lifter makes one per address.
    let function = FunctionBody::make_at_addr_indexed(&mut context, &mut addresses, address, None).id;

    let limit = opts.count.unwrap_or(usize::MAX);
    let mut instructions = Vec::new();
    let mut cursor = 0usize;
    while instructions.len() < limit && cursor < data.len() {
        let at = address + cursor as u64;
        let decoded = decoder.decode_one(at, &data[cursor..], &spec.new_context());
        let instruction = match decoded {
            Ok(instruction) => instruction,
            // The first instruction is the caller's request; later failures
            // just end the walk.
            Err(e) if instructions.is_empty() => {
                return Err(format!("no instruction decodes at {at:#x}: {e:?}"));
            }
            Err(_) => break,
        };
        let len = instruction.len();
        let flat = instruction
            .pcode_ops()
            .map_err(|e| format!("SLEIGH p-code emission failed at {at:#x}: {e}"))?;
        lifter
            .lift_pcode_indexed(&mut context, &mut addresses, at, len, &flat, Some(function))
            .map_err(|e| format!("QCode lowering failed at {at:#x}: {e}"))?;
        instructions.push(Insn {
            address: format!("{at:#x}"),
            bytes: data[cursor..cursor + len]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
            text: instruction.to_string(),
        });
        cursor += len;
    }

    if opts.passes {
        let block_ids: Vec<_> = context.blocks().map(|b| b.id).collect();
        for block_id in block_ids {
            while qcode_passes::remove_dead_insns(&mut context, block_id) {}
        }
    }

    let functions = context
        .functions()
        .map(|function| Func {
            name: function.name().to_string(),
            blocks: function
                .blocks()
                .map(|block| Block {
                    name: block.name().unwrap_or_default().to_string(),
                    text: block.to_string(),
                })
                .collect(),
        })
        .collect::<Vec<_>>();

    Ok(Output {
        arch: opts.arch.clone(),
        instructions,
        qcode: context.to_string(),
        functions: if functions.is_empty() { None } else { Some(functions) },
    })
}

fn main() {
    let cli = Cli::parse();
    let opts = match &cli.args {
        Some(json) => match serde_json::from_str::<Opts>(json) {
            Ok(opts) => opts,
            Err(e) => fail(cli.json, &format!("invalid --args: {e}")),
        },
        None => cli.opts,
    };
    let output = match run(&opts) {
        Ok(output) => output,
        Err(e) => fail(cli.json, &e),
    };
    if cli.json {
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
        return;
    }
    for insn in &output.instructions {
        println!("{:<12} {:<24} {}", insn.address, insn.bytes, insn.text);
    }
    println!();
    println!("{}", output.qcode);
}

fn fail(json: bool, message: &str) -> ! {
    if json {
        eprintln!("{}", serde_json::json!({ "error": message }));
    } else {
        eprintln!("error: {message}");
    }
    process::exit(1);
}
