//! The playground's one function: lift x86-64 bytes to QCode.
//!
//! `lift(hex, address)` mirrors `qcode-lift --arch x64` without passes and
//! returns the same JSON shape: the decoded instructions and the QCode text.

use serde::Serialize;
use sleigh::Decoder;
use wasm_bindgen::prelude::*;
use wazabin_qcode_sleigh::{
    FlatPcode,
    SleighLifter,
    session::{Host, LiftSession},
};

#[derive(Serialize)]
struct Output {
    instructions: Vec<Insn>,
    qcode: String,
}

#[derive(Serialize)]
struct Insn {
    address: String,
    bytes: String,
    text: String,
}

#[derive(Serialize)]
struct Failure {
    error: String,
}

fn parse_hex(value: &str) -> Result<Vec<u8>, String> {
    let value: String = value.chars().filter(|c| !c.is_whitespace()).collect();
    if value.is_empty() || !value.len().is_multiple_of(2) {
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

fn run(hex: &str, address: u64) -> Result<Output, String> {
    let data = parse_hex(hex)?;
    let spec = sleigh_precompile::x64::spec();
    let decoder = Decoder::new(spec);
    let lifter = SleighLifter::new(spec);
    let mut session = LiftSession::new(&lifter, Host::At(address));

    let mut instructions = Vec::new();
    let mut cursor = 0usize;
    while cursor < data.len() {
        let at = address + cursor as u64;
        let instruction = match decoder.decode_one(at, &data[cursor..], &spec.new_context()) {
            Ok(instruction) => instruction,
            Err(e) if instructions.is_empty() => {
                return Err(format!("no instruction decodes at {at:#x}: {e:?}"));
            }
            Err(_) => break,
        };
        let len = instruction.len();
        let flat = FlatPcode::lower(&instruction)
            .map_err(|e| format!("SLEIGH p-code emission failed at {at:#x}: {e}"))?;
        session
            .lift_pcode(&flat)
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
    Ok(Output {
        instructions,
        qcode: session.into_context().to_string(),
    })
}

/// Lift `hex` (x86-64 bytes, whitespace ignored) placed at `address`; returns
/// `{instructions, qcode}` or `{error}` as JSON.
#[wasm_bindgen]
pub fn lift(hex: &str, address: u64) -> String {
    match run(hex, address) {
        Ok(output) => serde_json::to_string(&output).unwrap(),
        Err(error) => serde_json::to_string(&Failure { error }).unwrap(),
    }
}
