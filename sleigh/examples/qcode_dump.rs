//! Decode one x86-64 instruction and dump every lowering boundary.
//!
//! ```text
//! cargo run -p wazabin-qcode-sleigh --example qcode-dump -- 66f30fbcc3
//! cargo run -p wazabin-qcode-sleigh --example qcode-dump -- --address 0x1000 c4e263f5c1
//! ```

use std::{env, process};

use qcode::address_index::AddressIndex;
use sleigh::Decoder;
use sleigh_precompile::x64;
use wazabin_qcode_sleigh::SleighLifter;

const HELP: &str = "\
qcode-dump — decode an x86-64 instruction and print its SLEIGH and QCode lowering

USAGE:
    qcode-dump [--address <ADDRESS>] <HEX-BYTES>

OPTIONS:
    -a, --address <ADDRESS>  Instruction address (decimal or 0x-prefixed; default: 0x1000)
    -h, --help               Show this help
";

fn parse_hex(value: &str) -> Result<Vec<u8>, String> {
    let value: String = value.chars().filter(|ch| !ch.is_whitespace()).collect();
    if value.is_empty() || value.len() % 2 != 0 {
        return Err("hex bytes must contain a non-empty, even number of digits".to_string());
    }
    (0..value.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&value[index..index + 2], 16)
                .map_err(|_| format!("invalid hex byte '{}'", &value[index..index + 2]))
        })
        .collect()
}

fn parse_address(value: &str) -> Result<u64, String> {
    value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .map_or_else(|| value.parse(), |hex| u64::from_str_radix(hex, 16))
        .map_err(|_| format!("invalid address '{value}'"))
}

fn run() -> Result<(), String> {
    let mut address = 0x1000;
    let mut bytes = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-a" | "--address" => {
                let value = args
                    .next()
                    .ok_or_else(|| "expected an address after --address".to_string())?;
                address = parse_address(&value)?;
            }
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            _ if arg.starts_with('-') => return Err(format!("unknown option '{arg}'\n\n{HELP}")),
            _ if bytes.is_none() => bytes = Some(parse_hex(&arg)?),
            _ => return Err(format!("expected exactly one HEX-BYTES argument\n\n{HELP}")),
        }
    }
    let bytes = bytes.ok_or_else(|| format!("missing HEX-BYTES argument\n\n{HELP}"))?;

    let spec = x64::spec();
    let instruction = Decoder::new(spec)
        .decode_one(address, &bytes, &spec.new_context())
        .map_err(|error| format!("decode failed: {error}"))?;
    println!("address: {address:#x}");
    println!(
        "bytes: {}",
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    println!("instruction: {instruction}");
    println!(
        "\nSLEIGH AST:\n{}",
        instruction
            .pcode_ast()
            .map_err(|error| error.to_string())?
            .pretty_print(spec)
    );
    let flat = instruction
        .pcode_ops()
        .map_err(|error| format!("SLEIGH p-code emission failed: {error}"))?;
    println!("\nFLAT PCODE:");
    for (index, op) in flat.ops.iter().enumerate() {
        println!("{index:04}: {op:?}");
    }

    let lifter = SleighLifter::new(spec);
    let mut context = lifter.new_context();
    let mut addresses = AddressIndex::analyze(&context);
    lifter
        .lift_pcode_indexed(
            &mut context,
            &mut addresses,
            address,
            instruction.len(),
            &flat,
            None,
        )
        .map_err(|error| format!("QCode lowering failed: {error}"))?;
    println!("\nQCODE:\n{context}");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(2);
    }
}
