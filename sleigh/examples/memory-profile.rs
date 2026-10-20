//! Memory of a lifted binary kept as persistent function bodies, per x86
//! instruction, by structure. The code of an ELF's `.text` is swept linearly,
//! cut into functions at the targets of its direct calls (a stand-in for the
//! functions a recursive descent finds), and lifted through one
//! [`LiftSession`] into one [`Context`], which is what a consumer keeping a
//! whole binary's IR does.
//!
//! A counting allocator gives the bytes the module holds, requested and as
//! `malloc` chunks; [`Context::body_memory`] splits the requested bytes by
//! structure, and what it does not account for is reported as the rest.
//!
//! ```sh
//! cargo run --release -p wazabin-qcode-sleigh --example memory-profile -- /usr/bin/ls [--max-func-insns N] [--limit N] [--cache] [--no-names] [--opcodes] [--by-x86 N] [--dce]
//! ```

use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering::Relaxed},
    },
    time::Instant,
};

use qcode::value::insn::Mnemonic;
use sleigh::Decoder;
use wazabin_qcode_sleigh::{
    SleighLifter,
    cache::LiftCache,
    session::{Host, LiftSession},
};

/// The system allocator, counting what is live: the bytes asked for, the
/// bytes glibc's chunks take (a header and 16-byte alignment on top), and
/// how many allocations are small.
struct Counting;

static REQUESTED: AtomicUsize = AtomicUsize::new(0);
static CHUNKS: AtomicUsize = AtomicUsize::new(0);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static SMALL: AtomicUsize = AtomicUsize::new(0);
static SMALL_BYTES: AtomicUsize = AtomicUsize::new(0);

/// What glibc's `malloc` takes for a request.
fn chunk(size: usize) -> usize {
    (size + 8).next_multiple_of(16).max(32)
}

const SMALL_LIMIT: usize = 256;

fn account(size: usize, add: bool) {
    let apply = |counter: &AtomicUsize, by: usize| {
        if add {
            counter.fetch_add(by, Relaxed);
        } else {
            counter.fetch_sub(by, Relaxed);
        }
    };
    apply(&REQUESTED, size);
    apply(&CHUNKS, chunk(size));
    apply(&COUNT, 1);
    if size <= SMALL_LIMIT {
        apply(&SMALL, 1);
        apply(&SMALL_BYTES, size);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        account(layout.size(), true);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        account(layout.size(), false);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        account(layout.size(), true);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        account(layout.size(), false);
        account(new_size, true);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

#[derive(Clone, Copy)]
struct Live {
    requested: usize,
    chunks: usize,
    count: usize,
    small: usize,
    small_bytes: usize,
}

fn live() -> Live {
    Live {
        requested: REQUESTED.load(Relaxed),
        chunks: CHUNKS.load(Relaxed),
        count: COUNT.load(Relaxed),
        small: SMALL.load(Relaxed),
        small_bytes: SMALL_BYTES.load(Relaxed),
    }
}

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

fn rss_kb() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse().ok())
        })
        .unwrap_or(0)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let has = |name: &str| args.iter().any(|a| a == name);
    let func_insns: usize = flag("--max-func-insns").map_or(usize::MAX, |s| s.parse().unwrap());
    let limit: usize = flag("--limit").map_or(usize::MAX, |s| s.parse().unwrap());
    let by_x86: usize = flag("--by-x86").map_or(0, |s| s.parse().unwrap());
    let path = args.first().expect("path to an ELF").clone();
    let file = std::fs::read(&path).unwrap();
    let (base, text) = text_section(&file);
    drop(file);

    let spec = sleigh_precompile::x64::spec();
    let decoder = Decoder::new(spec);
    let dctx = spec.new_context();
    let lifter = SleighLifter::new(spec);
    let cache = has("--cache").then(|| Arc::new(LiftCache::new(spec)));

    // The instruction boundaries of a linear sweep.
    let mut offsets: Vec<(usize, usize)> = Vec::new();
    let mut off = 0;
    while off < text.len() && offsets.len() < limit {
        let len = decoder
            .decode_one(base + off as u64, &text[off..], &dctx)
            .map(|i| i.len())
            .unwrap_or(1)
            .max(1);
        offsets.push((off, len));
        off += len;
    }
    let n = offsets.len();
    eprintln!("{path}: {n} x86 instructions, {} bytes", off);

    let before = live();
    let mut session = LiftSession::new(&lifter, Host::Anonymous);
    if let Some(cache) = &cache {
        session = session.with_cache(Arc::clone(cache));
    }
    if has("--no-names") {
        session = session.without_debug_names();
    }
    let mut per_x86: HashMap<String, (usize, usize)> = HashMap::new();
    let mut failures = 0usize;
    let mut errors: HashMap<String, usize> = HashMap::new();
    let start = Instant::now();
    // Functions start where something calls: the entries a recursive
    // descent over the direct calls finds, and none a linear sweep makes up.
    let mut entries: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for &(off, _) in &offsets {
        let at = base + off as u64;
        if text[off] != 0xe8 {
            continue;
        }
        if let Ok(i) = decoder.decode_one(at, &text[off..], &dctx) {
            let target = at
                .wrapping_add(i.len() as u64)
                .wrapping_add(
                    i32::from_le_bytes(text[off + 1..off + 5].try_into().unwrap()) as i64 as u64,
                );
            entries.insert(target);
        }
    }
    let mut functions_at: Vec<&[(usize, usize)]> = Vec::new();
    let mut from = 0;
    for (k, &(off, _)) in offsets.iter().enumerate() {
        if k > from && (entries.contains(&(base + off as u64)) || k - from >= func_insns) {
            functions_at.push(&offsets[from..k]);
            from = k;
        }
    }
    functions_at.push(&offsets[from..]);
    for chunk in functions_at {
        session.set_host(Host::At(base + chunk[0].0 as u64));
        for &(off, _) in chunk {
            let at = base + off as u64;
            match session.lift(at, &text[off..]) {
                Ok(lifted) => {
                    if by_x86 > 0 {
                        let ctx = session.context();
                        let insns: usize = lifted
                            .blocks()
                            .iter()
                            .map(|&b| ctx.block_ref(b).instructions().count())
                            .sum();
                        let name = decoder
                            .decode_one(at, &text[off..], &dctx)
                            .ok()
                            .and_then(|i| i.display().ok())
                            .and_then(|d| d.split_whitespace().next().map(str::to_owned))
                            .unwrap_or_else(|| "?".into());
                        let e = per_x86.entry(name).or_default();
                        e.0 += 1;
                        e.1 += insns;
                    }
                }
                Err(e) => {
                    failures += 1;
                    let text = e.to_string();
                    *errors
                        .entry(text.chars().take(80).collect::<String>())
                        .or_default() += 1;
                }
            }
        }
    }
    let lift_time = start.elapsed();
    let with_session = live();
    let mut ctx = session.into_context().unwrap();
    // A consumer keeps the module: whatever the lift reserved past what it
    // used goes back.
    ctx.shrink_bodies_to_fit();
    drop(cache);
    let after = live();

    let per = |bytes: usize| bytes as f64 / n as f64;
    println!(
        "lifted {n} x86 instructions ({failures} failed) in {:.2} s ({:.2} µs each)",
        lift_time.as_secs_f64(),
        lift_time.as_secs_f64() * 1e6 / n as f64
    );
    let mut errors: Vec<_> = errors.into_iter().collect();
    errors.sort_by_key(|e| std::cmp::Reverse(e.1));
    for (e, c) in errors.iter().take(5) {
        println!("  failed {c}x: {e}");
    }
    println!("VmRSS {} MB", rss_kb() / 1024);

    let functions = ctx.function_ids().len();
    let mut blocks = 0usize;
    let mut insns = 0usize;
    let mut named = 0usize;
    let mut opcodes: HashMap<String, usize> = HashMap::new();
    for f in ctx.functions() {
        for block in f.blocks() {
            blocks += 1;
            for insn in block.instructions() {
                insns += 1;
                named += insn.name().is_some() as usize;
                if has("--opcodes") {
                    let label = match insn.mnemonic() {
                        Mnemonic::Binop(b) => format!("binop {}", b.op),
                        Mnemonic::Unop(u) => format!("unop {}", u.op),
                        m => m.opcode().to_owned(),
                    };
                    *opcodes.entry(label).or_default() += 1;
                }
            }
        }
    }
    println!(
        "{functions} functions, {blocks} blocks ({:.3}/x86), {insns} instructions ({:.2}/x86), {named} named ({:.2}/x86)",
        blocks as f64 / n as f64,
        insns as f64 / n as f64,
        named as f64 / n as f64,
    );

    let delta = |after: Live, before: Live| Live {
        requested: after.requested - before.requested,
        chunks: after.chunks - before.chunks,
        count: after.count - before.count,
        small: after.small - before.small,
        small_bytes: after.small_bytes - before.small_bytes,
    };
    let session_total = delta(with_session, before);
    let module = delta(after, before);
    println!("\nbytes per x86 instruction");
    println!(
        "  session before shrink  requested {:8.1}  malloc chunks {:8.1}",
        per(session_total.requested),
        per(session_total.chunks)
    );
    println!(
        "  module after shrink    requested {:8.1}  malloc chunks {:8.1}  ({:.2} allocations, {:.2} of them <= {SMALL_LIMIT} B, {:.1} B/x86 in those)",
        per(module.requested),
        per(module.chunks),
        module.count as f64 / n as f64,
        module.small as f64 / n as f64,
        per(module.small_bytes),
    );
    let m = ctx.body_memory();
    let rows = [
        ("instruction records", m.instructions),
        ("instruction arena index", m.instruction_index),
        ("block records", m.blocks),
        ("block heap", m.block_heap),
        ("block roster", m.roster),
        ("block params", m.params),
        ("cfg edges", m.edges),
        ("use edges + shared heads", m.uses),
        ("name table", m.names),
        ("instruction addrs", m.instruction_addrs),
        ("temps", m.temps),
    ];
    println!("\n  by structure (requested):");
    for (name, bytes) in rows {
        println!("    {name:26} {:8.1}", per(bytes));
    }
    println!("    {:26} {:8.1}", "accounted", per(m.total()));
    println!(
        "    {:26} {:8.1}",
        "shared values, mnemonic heap, rest",
        per(module.requested.saturating_sub(m.total()))
    );
    println!(
        "    {:26} {:8.1}",
        "malloc header and padding",
        per(module.chunks - module.requested)
    );
    println!(
        "  per QCode instruction: {:.1} B requested",
        module.requested as f64 / insns as f64
    );

    if has("--opcodes") {
        let mut v: Vec<_> = opcodes.into_iter().collect();
        v.sort_by_key(|e| std::cmp::Reverse(e.1));
        println!("\nQCode instructions by opcode (per x86 instruction)");
        for (op, c) in v.iter().take(40) {
            println!(
                "  {op:20} {:7.3}  {:5.1}%",
                *c as f64 / n as f64,
                *c as f64 * 100.0 / insns as f64
            );
        }
    }
    if by_x86 > 0 {
        let mut v: Vec<_> = per_x86.into_iter().collect();
        v.sort_by_key(|e| std::cmp::Reverse(e.1.1));
        println!("\nQCode instructions by x86 mnemonic: top {by_x86} by total");
        println!(
            "  {:10} {:>9} {:>11} {:>9}",
            "mnemonic", "count", "qcode total", "per insn"
        );
        for (name, (count, total)) in v.iter().take(by_x86) {
            println!(
                "  {name:10} {count:9} {total:11} {:9.2}",
                *total as f64 / *count as f64
            );
        }
    }
}
