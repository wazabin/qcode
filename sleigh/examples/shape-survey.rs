//! Survey of the shapes of a binary's encodings: how many distinct keys a
//! shape-level cache would hold against an exact one, and a cross-check of
//! the decoder's shape mask against perturbation — a byte whose change alters
//! the constructor path must be in the mask.
use std::collections::HashMap;

use sleigh::Decoder;

fn text_section(file: &[u8]) -> (u64, Vec<u8>) {
    let u16_at = |o: usize| u16::from_le_bytes(file[o..o + 2].try_into().unwrap()) as usize;
    let u64_at = |o: usize| u64::from_le_bytes(file[o..o + 8].try_into().unwrap());
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
            let off = u64_at(sh(i) + 0x18) as usize;
            let size = u64_at(sh(i) + 0x20) as usize;
            return (u64_at(sh(i) + 0x10), file[off..off + size].to_vec());
        }
    }
    panic!("no .text");
}

/// A shape as a survey key: its mask, its masked bytes and its exclusions.
type ShapeKey = (Vec<u8>, Vec<u8>, Vec<sleigh::Exclusion>);

fn main() {
    let path = std::env::args().nth(1).expect("path");
    let superset = std::env::args().any(|a| a == "--superset");
    let file = std::fs::read(&path).unwrap();
    let (base, text) = text_section(&file);
    let spec = sleigh_precompile::x64::spec();
    let decoder = Decoder::new(spec);
    let dctx = spec.new_context();
    let path_of = |a: u64, b: &[u8]| -> Option<(usize, Vec<(String, usize)>)> {
        let i = decoder.decode_one(a, b, &dctx).ok()?;
        Some((
            i.len(),
            i.constructor_matches()
                .map(|m| (m.table().name().to_string(), m.index()))
                .collect(),
        ))
    };
    let mut encodings: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut total = 0usize;
    let mut off = 0;
    while off < text.len() {
        let a = base + off as u64;
        match decoder.decode_one(a, &text[off..], &dctx) {
            Ok(i) => {
                *encodings.entry(i.bytes().to_vec()).or_default() += 1;
                total += 1;
                off += if superset { 1 } else { i.len() };
            }
            Err(_) => off += 1,
        }
    }
    let a = 0x40_0000u64;
    let mut shapes: HashMap<ShapeKey, usize> = HashMap::new();
    let mut with_exclusions = 0usize;
    let mut neither_example: Option<(Vec<u8>, Vec<u8>)> = None;
    let mut overruns = 0usize;
    let mut violations = 0usize;
    let mut params_hist = [0usize; 8];
    let mut param_bits = 0usize;
    let mut unmasked_nonparam_bits = 0usize;
    for (bytes, count) in &encodings {
        let (insn, shape) = decoder.decode_one_shaped(a, bytes, &dctx).unwrap();
        assert_eq!(insn.len(), shape.len());
        assert_eq!(shape.len(), bytes.len());
        if shape.overruns() {
            overruns += 1;
        }
        params_hist[shape.params().len().min(7)] += 1;
        let mut param_mask = vec![0u8; bytes.len()];
        for p in shape.params() {
            for bit in p.bit as usize..p.bit as usize + p.width as usize {
                param_mask[bit / 8] |= 1 << (bit % 8);
                param_bits += 1;
            }
        }
        let neither: usize = shape
            .mask()
            .iter()
            .zip(&param_mask)
            .map(|(mask, param)| (!(mask | param)).count_ones() as usize)
            .sum();
        unmasked_nonparam_bits += neither;
        if neither > 0 && neither_example.is_none() {
            neither_example = Some((bytes.clone(), shape.mask().to_vec()));
        }
        if !shape.exclusions().is_empty() {
            with_exclusions += 1;
        }
        // Cross-check: flipping any bit outside the mask must keep the path.
        let reference = path_of(a, bytes).unwrap();
        for i in 0..bytes.len() {
            let free = !shape.mask()[i];
            if free == 0 {
                continue;
            }
            for x in [0xffu8, 0x55, 0xaa, 0x01, 0x80, 0x0f] {
                let flip = x & free;
                if flip == 0 {
                    continue;
                }
                let mut v = bytes.clone();
                v[i] ^= flip;
                if !shape.admits(&v) {
                    continue;
                }
                if path_of(a, &v).as_ref() != Some(&reference) {
                    violations += 1;
                    if violations <= 10 {
                        println!(
                            "VIOLATION {} byte {i} flip {flip:02x} mask {:02x} -> {}",
                            hex(bytes),
                            shape.mask()[i],
                            insn.display().unwrap_or_default()
                        );
                    }
                    break;
                }
            }
        }
        let mut masked = vec![0u8; bytes.len()];
        shape.masked(bytes, &mut masked);
        *shapes
            .entry((shape.mask().to_vec(), masked, shape.exclusions().to_vec()))
            .or_default() += count;
    }
    println!(
        "{path}: {total} lifts, {} distinct encodings, {} distinct shapes, {overruns} overruns, {violations} violations",
        encodings.len(),
        shapes.len()
    );
    println!(
        "params per encoding: {params_hist:?}; param bits {param_bits}, bits neither masked nor parametric {unmasked_nonparam_bits}; {with_exclusions} encodings with exclusions"
    );
    if let Some((bytes, mask)) = neither_example {
        println!("  neither example: {} mask {}", hex(&bytes), hex(&mask));
    }
    // Candidate masks a lookup before decoding would try, per leading byte.
    let mut masks_by_first: HashMap<u8, Vec<Vec<u8>>> = HashMap::new();
    for (mask, masked, _) in shapes.keys() {
        let list = masks_by_first.entry(masked[0]).or_default();
        if !list.contains(mask) {
            list.push(mask.clone());
        }
    }
    let max = masks_by_first.values().map(|v| v.len()).max().unwrap_or(0);
    let weighted: usize = shapes
        .iter()
        .map(|((_, masked, _), c)| masks_by_first[&masked[0]].len() * c)
        .sum();
    let mut by_count: Vec<_> = masks_by_first.iter().map(|(b, v)| (v.len(), *b)).collect();
    by_count.sort_unstable_by_key(|(n, _)| std::cmp::Reverse(*n));
    println!(
        "candidate masks per leading byte: max {max}, mean per lift {:.1}; worst {:?}",
        weighted as f64 / total as f64,
        &by_count[..by_count.len().min(8)]
    );
    let mut top: Vec<_> = shapes.iter().collect();
    top.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for ((mask, masked, exclusions), count) in top.iter().take(12) {
        println!(
            "  {count:6}  {}  mask {}  excl {}",
            hex(masked),
            hex(mask),
            exclusions.len()
        );
    }
}

fn hex(b: &[u8]) -> String {
    b.iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}
