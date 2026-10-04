//! The cost of editing a big function: the code of an ELF's `.text` lifted
//! into one function, then every block cleaned of dead instructions with
//! [`remove_dead_insns`], then every other block deleted. One function means
//! the operands every instruction names (a register, a constant) have use
//! lists as long as the function, which is what the edits must not walk.
//!
//! ```sh
//! cargo run --release -p wazabin-qcode-sleigh --example edit-bench -- /usr/bin/ls [--insns 100000]
//! ```

use std::time::Instant;

use qcode::value::QCodeMut;
use qcode_passes::remove_dead_insns;
use sleigh::Decoder;
use wazabin_qcode_sleigh::{
    SleighLifter,
    session::{Host, LiftSession},
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let limit: usize = args
        .iter()
        .position(|a| a == "--insns")
        .map_or(100_000, |i| args[i + 1].parse().unwrap());
    let file = std::fs::read(&args[0]).unwrap();
    let (base, text) = text_section(&file);
    let spec = sleigh_precompile::x64::spec();
    let decoder = Decoder::new(spec);
    let dctx = spec.new_context();
    let lifter = SleighLifter::new(spec).with_flat_control_flow();

    let start = Instant::now();
    let mut session = LiftSession::new(&lifter, Host::At(base));
    let function = session.function();
    let (mut off, mut lifted) = (0, 0);
    while off < text.len() && lifted < limit {
        let len = decoder
            .decode_one(base + off as u64, &text[off..], &dctx)
            .map(|i| i.len())
            .unwrap_or(1)
            .max(1);
        if session.lift(base + off as u64, &text[off..]).is_ok() {
            lifted += 1;
        }
        off += len;
    }
    let mut ctx = session.into_context().unwrap();
    println!(
        "lifted {lifted} x86 instructions in {:.2} s",
        start.elapsed().as_secs_f64()
    );

    let blocks = ctx.function_ref(function).block_ids();
    let insns = ctx.function_ref(function).instruction_ids().len();
    println!("{} blocks, {insns} QCode instructions", blocks.len());

    let start = Instant::now();
    let changed = blocks
        .iter()
        .filter(|&&block| remove_dead_insns(&mut ctx, block))
        .count();
    let left = ctx.function_ref(function).instruction_ids().len();
    println!(
        "dce: {changed} blocks changed, {} instructions removed, {:.3} s",
        insns - left,
        start.elapsed().as_secs_f64()
    );

    let start = Instant::now();
    let mut deleted = 0;
    for &block in blocks.iter().step_by(2) {
        ctx.delete_block(block);
        deleted += 1;
    }
    println!(
        "delete_block: {deleted} blocks, {:.3} s",
        start.elapsed().as_secs_f64()
    );
}
