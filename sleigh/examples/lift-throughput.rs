//! Lifting throughput over the whole `.text` of an ELF binary, every byte
//! offset (the superset) or a linear sweep, single-threaded. The lift stage
//! is a [`ScratchSession`], or with `--session` a [`LiftSession`] keeping
//! every instruction in one function, which is what a cache hit replays into.
//!
//! ```sh
//! cargo run --release -p wazabin-qcode-sleigh --example lift-throughput -- /usr/bin/ls [--linear] [--stage decode|pcode|lift] [--cache] [--validate] [--session]
//! cargo run --release -p wazabin-qcode-sleigh --example lift-throughput -- --bytes dec9 --iters 3000 --cache
//! cargo run --release -p wazabin-qcode-sleigh --example lift-throughput -- xul.dll --offset 0x400 --len 0x24603E1 --linear --stage decode
//! ```
//!
//! `--offset` and `--len` take the code from a slice of any file instead of
//! an ELF's `.text`, as disas-bench's harnesses do
//! (<https://github.com/icedland/disas-bench>: xul.dll's `.text`, decoded
//! linearly, reported in MB/s), so the decode stage compares with its table.

use std::{hint::black_box, sync::Arc, time::Instant};

use sleigh::{Decoder, LabelId, Opcode, PcodePlan, PcodeSink, Varnode};

/// A sink that keeps nothing.
#[derive(Default)]
struct Tally(usize);

impl PcodeSink for Tally {
    fn op(&mut self, opcode: Opcode, output: Option<Varnode>, inputs: &[Varnode]) {
        black_box((opcode, output, inputs));
        self.0 += 1;
    }
    fn label(&mut self, label: LabelId) {
        black_box(label);
    }
    fn branch_label(&mut self, opcode: Opcode, label: LabelId, condition: Option<Varnode>) {
        black_box((opcode, label, condition));
    }
}
use wazabin_qcode_sleigh::{
    SleighLifter,
    cache::LiftCache,
    session::{Host, LiftSession, ScratchSession},
};

fn text_section(file: &[u8]) -> (u64, Vec<u8>) {
    let u16_at = |o: usize| u16::from_le_bytes(file[o..o + 2].try_into().unwrap()) as usize;
    let u64_at = |o: usize| u64::from_le_bytes(file[o..o + 8].try_into().unwrap());
    assert_eq!(&file[..4], b"\x7fELF");
    let shoff = u64_at(0x28) as usize;
    let shentsize = u16_at(0x3a);
    let shnum = u16_at(0x3c);
    let shstrndx = u16_at(0x3e);
    let sh = |i: usize| shoff + i * shentsize;
    let strtab = u64_at(sh(shstrndx) + 0x18) as usize;
    for i in 0..shnum {
        let name = u32::from_le_bytes(file[sh(i)..sh(i) + 4].try_into().unwrap()) as usize;
        let end = file[strtab + name..].iter().position(|&b| b == 0).unwrap();
        if &file[strtab + name..strtab + name + end] == b".text" {
            let addr = u64_at(sh(i) + 0x10);
            let off = u64_at(sh(i) + 0x18) as usize;
            let size = u64_at(sh(i) + 0x20) as usize;
            return (addr, file[off..off + size].to_vec());
        }
    }
    panic!("no .text");
}

/// One stage of the pipeline: whether the bytes at an address went through.
type Stage<'a> = Box<dyn FnMut(u64, &[u8]) -> bool + 'a>;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let number = |s: &str| -> usize {
        match s.strip_prefix("0x") {
            Some(hex) => usize::from_str_radix(hex, 16).unwrap(),
            None => s.parse().unwrap(),
        }
    };
    let linear = args.iter().any(|a| a == "--linear");
    let use_cache = args.iter().any(|a| a == "--cache");
    let validate = args.iter().any(|a| a == "--validate");
    let keep = args.iter().any(|a| a == "--session");
    let stage = flag("--stage").unwrap_or_else(|| "all".into());
    let (path, base, text) = if let Some(hex) = flag("--bytes") {
        // One encoding, repeated `--iters` times at consecutive addresses.
        let iters: usize = flag("--iters")
            .map(|s| s.parse().unwrap())
            .unwrap_or(10_000);
        let one: Vec<u8> = (0..hex.len() / 2)
            .map(|k| u8::from_str_radix(&hex[2 * k..2 * k + 2], 16).unwrap())
            .collect();
        (hex, 0x1000, one.repeat(iters))
    } else {
        let path = args
            .iter()
            .find(|a| !a.starts_with("--"))
            .expect("path to an ELF")
            .clone();
        let file = std::fs::read(&path).unwrap();
        let (base, text) = match flag("--offset") {
            Some(offset) => {
                let offset = number(&offset);
                let len = flag("--len")
                    .map(|s| number(&s))
                    .unwrap_or(file.len() - offset);
                (0x1000, file[offset..offset + len].to_vec())
            }
            None => text_section(&file),
        };
        (path, base, text)
    };
    eprintln!("{path}: .text {} bytes at {base:#x}", text.len());

    let spec = sleigh_precompile::x64::spec();
    let decoder = Decoder::new(spec);
    let dctx = spec.new_context();
    let lifter = SleighLifter::new(spec).with_flat_control_flow();
    let cache = Arc::new(LiftCache::new(spec).validating(validate));
    let mut session = ScratchSession::new(&lifter);
    let mut keeping = LiftSession::new(&lifter, Host::Anonymous);
    if use_cache {
        session = session.with_cache(Arc::clone(&cache));
        keeping = keeping.with_cache(Arc::clone(&cache));
    }

    let offsets: Vec<usize> = if linear {
        let mut v = Vec::new();
        let mut off = 0;
        while off < text.len() {
            v.push(off);
            off += decoder
                .decode_one(base + off as u64, &text[off..], &dctx)
                .map(|i| i.len())
                .unwrap_or(1);
        }
        v
    } else {
        (0..text.len()).collect()
    };
    let n = offsets.len();

    let run = |name: &str, mut f: Stage<'_>| {
        let t = Instant::now();
        let mut ok = 0usize;
        for &off in &offsets {
            ok += f(base + off as u64, &text[off..]) as usize;
        }
        let dt = t.elapsed();
        println!(
            "{name:8} {n:8} offsets  {ok:8} ok  {:7.3} s  {:6.2} µs/offset  {:6.2} MB/s",
            dt.as_secs_f64(),
            dt.as_secs_f64() * 1e6 / n as f64,
            text.len() as f64 / dt.as_secs_f64() / 1e6
        );
    };

    if stage == "all" || stage == "decode" {
        run(
            "decode",
            Box::new(|a, b| {
                decoder
                    .decode_one(a, b, &dctx)
                    .map(|i| black_box(i.len()))
                    .is_ok()
            }),
        );
    }
    if stage == "all" || stage == "pcode" {
        run(
            "pcode",
            Box::new(|a, b| {
                decoder
                    .decode_one(a, b, &dctx)
                    .ok()
                    .and_then(|i| {
                        i.pcode_ops_streamed(|plan: &PcodePlan| {
                            Tally(black_box(plan.direct_branches().len()))
                        })
                        .ok()
                    })
                    .is_some()
            }),
        );
    }
    if (stage == "all" || stage == "lift") && keep {
        run(
            "session",
            Box::new(|a, b| {
                keeping
                    .lift(a, b)
                    .map(|l| black_box(l.blocks().len()))
                    .is_ok()
            }),
        );
    } else if stage == "all" || stage == "lift" {
        run(
            "lift",
            Box::new(|a, b| {
                session
                    .lift(a, b)
                    .map(|l| black_box(l.blocks().count()))
                    .is_ok()
            }),
        );
    }
    if use_cache {
        println!("{:?}", cache.stats());
    }
}
