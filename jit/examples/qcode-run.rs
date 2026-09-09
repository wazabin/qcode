//! `qcode-run` — run machine code in the QCode VM.
//!
//! Arguments come either from flags or from one JSON object whose keys are
//! the flag names; `--json` switches the output to JSON.
//!
//! ```text
//! qcode-run 48c7c02a000000c3
//! qcode-run --args '{"code":"48c7c02a000000c3","arch":"x64","jit":true}' --json
//! qcode-run --map 0x20000:0x1000:rw --reg RDI=0x20000 --dump 0x20000:16 --show RAX --show RIP 48c7c02a000000c3
//! ```

use clap::Parser;
use qcode_jit::Jit;
use qcode_vm::{Vm, VmExit, VmMemory, perm};
use serde::{Deserialize, Serialize};
use sleigh::CompiledSpec;
use std::{fs, process};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

/// Run machine code in the QCode VM.
#[derive(Parser)]
#[command(name = "qcode-run", version)]
struct Cli {
    /// All options as one JSON object; keys are the long flag names
    #[arg(long, value_name = "JSON", conflicts_with_all = [
        "code", "arch", "file", "offset", "address", "entry", "map", "reg",
        "budget", "jit", "dump", "show",
    ])]
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
    /// The code to run, as hex
    code: Option<String>,
    /// Architecture: x64, x86, aarch64, riscv
    #[arg(long, default_value = "x64")]
    #[serde(default = "default_arch")]
    arch: String,
    /// Read the code from this file instead of the command line
    #[arg(long, conflicts_with = "code")]
    file: Option<String>,
    /// Byte offset into --file (decimal or 0x-prefixed)
    #[arg(long, default_value = "0", requires = "file")]
    offset: String,
    /// Address the code is loaded at (decimal or 0x-prefixed)
    #[arg(long, default_value = "0x1000")]
    #[serde(default = "default_address")]
    address: String,
    /// Address to start execution at (decimal or 0x-prefixed); default is --address
    #[arg(long)]
    entry: Option<String>,
    /// Extra memory to map, as ADDR:SIZE[:PERMS] (perms: r, rw, rwx; default rw)
    #[arg(long = "map")]
    map: Vec<String>,
    /// Initial register value, as NAME=VALUE (decimal or 0x-prefixed)
    #[arg(long = "reg")]
    reg: Vec<String>,
    /// Maximum number of instructions to execute
    #[arg(long, default_value_t = 100_000)]
    #[serde(default = "default_budget")]
    budget: u64,
    /// Install the Cranelift JIT block executor
    #[arg(long)]
    jit: bool,
    /// Memory range to print after the run, as ADDR:SIZE
    #[arg(long = "dump")]
    dump: Vec<String>,
    /// Register to print after the run; if none given, a sensible default set is used
    #[arg(long = "show")]
    show: Vec<String>,
}

fn default_arch() -> String {
    "x64".to_string()
}

fn default_address() -> String {
    "0x1000".to_string()
}

fn default_budget() -> u64 {
    100_000
}

#[derive(Serialize)]
struct Output {
    arch: String,
    entry: String,
    steps: u64,
    stopped: String,
    faults: Vec<String>,
    registers: Vec<(String, String)>,
    memory: Vec<MemoryDump>,
}

#[derive(Serialize)]
struct MemoryDump {
    address: String,
    bytes: String,
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

fn default_registers(arch: &str) -> &'static [&'static str] {
    match arch {
        "x64" => &["RAX", "RBX", "RCX", "RDX", "RSI", "RDI", "RSP", "RBP", "RIP"],
        "x86" => &["EAX", "EBX", "ECX", "EDX", "ESI", "EDI", "ESP", "EBP", "EIP"],
        "aarch64" => &["X0", "X1", "X2", "X3", "SP", "PC"],
        "riscv" => &["ra", "sp", "a0", "a1", "a2", "pc"],
        _ => &[],
    }
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
        return Err("code must be a non-empty, even-length hex string".to_string());
    }
    (0..value.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&value[i..i + 2], 16)
                .map_err(|_| format!("invalid hex byte '{}'", &value[i..i + 2]))
        })
        .collect()
}

fn parse_perms(value: &str) -> Result<u8, String> {
    match value {
        "r" => Ok(perm::MAP | perm::READ | perm::INIT),
        "rw" => Ok(perm::RW_INIT),
        "rwx" => Ok(perm::RW_INIT | perm::EXEC),
        "rx" => Ok(perm::MAP | perm::READ | perm::EXEC | perm::INIT),
        other => Err(format!("invalid perms '{other}' (r, rw, rx, rwx)")),
    }
}

struct Map {
    addr: u64,
    size: u64,
    perms: u8,
}

fn parse_map(spec: &str) -> Result<Map, String> {
    let mut parts = spec.split(':');
    let addr = parts
        .next()
        .ok_or_else(|| format!("invalid --map '{spec}' (want ADDR:SIZE[:PERMS])"))?;
    let size = parts
        .next()
        .ok_or_else(|| format!("invalid --map '{spec}' (want ADDR:SIZE[:PERMS])"))?;
    let perms = match parts.next() {
        Some(perms) => parse_perms(perms)?,
        None => perm::RW_INIT,
    };
    if parts.next().is_some() {
        return Err(format!("invalid --map '{spec}' (too many ':'-separated parts)"));
    }
    Ok(Map {
        addr: parse_int("--map address", addr)?,
        size: parse_int("--map size", size)?,
        perms,
    })
}

struct Reg {
    name: String,
    value: u64,
}

fn parse_reg(spec: &str) -> Result<Reg, String> {
    let (name, value) = spec
        .split_once('=')
        .ok_or_else(|| format!("invalid --reg '{spec}' (want NAME=VALUE)"))?;
    Ok(Reg {
        name: name.to_string(),
        value: parse_int("--reg value", value)?,
    })
}

struct Dump {
    addr: u64,
    size: u64,
}

fn parse_dump(spec: &str) -> Result<Dump, String> {
    let (addr, size) = spec
        .split_once(':')
        .ok_or_else(|| format!("invalid --dump '{spec}' (want ADDR:SIZE)"))?;
    Ok(Dump {
        addr: parse_int("--dump address", addr)?,
        size: parse_int("--dump size", size)?,
    })
}

fn fault_text(exit: &VmExit) -> Option<String> {
    match exit {
        VmExit::Fault(fault) => Some(fault.to_string()),
        VmExit::Unlifted { addr, error } => {
            Some(format!("could not lift code at {addr:#x}: {error:?}"))
        }
        _ => None,
    }
}

fn stop_text(exit: &VmExit) -> String {
    match exit {
        VmExit::InstructionLimit => "instruction limit reached".to_string(),
        VmExit::Breakpoint(addr) => format!("breakpoint at {addr:#x}"),
        VmExit::Fault(fault) => format!("fault: {fault}"),
        VmExit::Unlifted { addr, .. } => format!("could not lift code at {addr:#x}"),
        VmExit::Error(message) => format!("interpreter error: {message}"),
    }
}

fn run(opts: &Opts) -> Result<Output, String> {
    let spec = spec_for(&opts.arch)?;
    let address = parse_int("address", &opts.address)?;
    let entry = match &opts.entry {
        Some(entry) => parse_int("entry", entry)?,
        None => address,
    };
    let code = match (&opts.code, &opts.file) {
        (Some(hex), None) => parse_hex(hex)?,
        (None, Some(path)) => {
            let offset = parse_int("offset", &opts.offset)? as usize;
            let file = fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
            file.get(offset..)
                .ok_or_else(|| format!("offset {offset:#x} is past the end of {path}"))?
                .to_vec()
        }
        (Some(_), Some(_)) => return Err("give either code or --file, not both".to_string()),
        (None, None) => return Err("give code or --file".to_string()),
    };
    let maps = opts.map.iter().map(|m| parse_map(m)).collect::<Result<Vec<_>, _>>()?;
    let regs = opts.reg.iter().map(|r| parse_reg(r)).collect::<Result<Vec<_>, _>>()?;
    let dumps = opts.dump.iter().map(|d| parse_dump(d)).collect::<Result<Vec<_>, _>>()?;

    let source = SleighCodeSource::new(spec);
    let ctx = source.new_context();

    let mut memory = VmMemory::new();
    memory
        .mmu
        .write_unchecked(address, &code, perm::READ | perm::EXEC);
    for map in &maps {
        memory
            .mmu
            .map(map.addr, map.size, map.perms)
            .map_err(|e| format!("cannot map {:#x}:{:#x}: {e}", map.addr, map.size))?;
    }

    let mut vm = Vm::at_address(ctx, entry, source, memory)
        .map_err(|e| format!("no instruction decodes at {entry:#x}: {e:?}"))?;
    if opts.jit {
        vm.set_block_executor(Box::new(Jit::new()));
    }

    let ctx = vm.context().clone();
    for reg in &regs {
        vm.emulator()
            .set_varnode_by_name(&ctx, &reg.name, reg.value)
            .map_err(|e| format!("cannot set register {}: {e:?}", reg.name))?;
        if !vm
            .context()
            .get_named(&reg.name)
            .is_some_and(|id| matches!(id, qcode::value::ValueId::Varnode(_)))
        {
            return Err(format!("no such register '{}'", reg.name));
        }
    }

    let exit = vm.run(opts.budget);
    let steps = vm.stats.steps;

    let ctx = vm.context().clone();
    let show: Vec<&str> = if opts.show.is_empty() {
        default_registers(&opts.arch).to_vec()
    } else {
        opts.show.iter().map(String::as_str).collect()
    };
    let registers = show
        .iter()
        .map(|name| {
            let value = vm.emulator().read_varnode_by_name(&ctx, name);
            (
                (*name).to_string(),
                match value {
                    Some(value) => format!("{value:#x}"),
                    None => "?".to_string(),
                },
            )
        })
        .collect();

    let mut memory_dumps = Vec::new();
    for dump in &dumps {
        let mut bytes = vec![0u8; dump.size as usize];
        let bytes_text = match vm.memory().mmu.read(dump.addr, &mut bytes) {
            Ok(()) => bytes.iter().map(|b| format!("{b:02x}")).collect(),
            Err(fault) => format!("<fault: {fault}>"),
        };
        memory_dumps.push(MemoryDump {
            address: format!("{:#x}", dump.addr),
            bytes: bytes_text,
        });
    }

    Ok(Output {
        arch: opts.arch.clone(),
        entry: format!("{entry:#x}"),
        steps,
        stopped: stop_text(&exit),
        faults: fault_text(&exit).into_iter().collect(),
        registers,
        memory: memory_dumps,
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
        println!(
            "{}",
            serde_json::to_string_pretty(&JsonOutput::from(&output)).unwrap()
        );
        return;
    }
    println!("arch:  {}", output.arch);
    println!("entry: {}", output.entry);
    println!("steps: {}", output.steps);
    for (name, value) in &output.registers {
        println!("{name:<6} {value}");
    }
    println!("stopped: {}", output.stopped);
    for fault in &output.faults {
        println!("fault: {fault}");
    }
    for dump in &output.memory {
        println!("{}: {}", dump.address, dump.bytes);
    }
}

/// The JSON shape: registers and address-tagged fields as an object rather
/// than the ordered vec used for tidy text output.
#[derive(Serialize)]
struct JsonOutput {
    arch: String,
    entry: String,
    steps: u64,
    stopped: String,
    faults: Vec<String>,
    registers: std::collections::BTreeMap<String, String>,
    memory: Vec<MemoryDumpRef>,
}

#[derive(Serialize)]
struct MemoryDumpRef {
    address: String,
    bytes: String,
}

impl From<&Output> for JsonOutput {
    fn from(output: &Output) -> Self {
        Self {
            arch: output.arch.clone(),
            entry: output.entry.clone(),
            steps: output.steps,
            stopped: output.stopped.clone(),
            faults: output.faults.clone(),
            registers: output.registers.iter().cloned().collect(),
            memory: output
                .memory
                .iter()
                .map(|d| MemoryDumpRef {
                    address: d.address.clone(),
                    bytes: d.bytes.clone(),
                })
                .collect(),
        }
    }
}

fn fail(json: bool, message: &str) -> ! {
    if json {
        eprintln!("{}", serde_json::json!({ "error": message }));
    } else {
        eprintln!("error: {message}");
    }
    process::exit(1);
}
