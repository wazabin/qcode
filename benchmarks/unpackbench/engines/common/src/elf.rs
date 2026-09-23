//! The least ELF64 loading a static x86-64 Linux program needs: `PT_LOAD`
//! segments, the entry point, and where the program headers sit for
//! `AT_PHDR`. `ET_DYN` images (a static-pie, or `ld.so` run as the program)
//! land at `userland/`'s `PIE_BASE`.

use crate::Guest;

pub const PAGE: u64 = 0x1000;
/// Where `userland/src/loader.rs` puts an `ET_DYN` image.
pub const PIE_BASE: u64 = 0x5555_5555_0000;

pub fn page_down(a: u64) -> u64 {
    a & !(PAGE - 1)
}
pub fn page_up(a: u64) -> u64 {
    a.wrapping_add(PAGE - 1) & !(PAGE - 1)
}

pub struct Segment {
    pub vaddr: u64,
    pub offset: u64,
    pub filesz: u64,
    pub memsz: u64,
}

pub struct Image {
    pub entry: u64,
    pub base: u64,
    pub phdr: u64,
    pub phnum: u16,
    pub phentsize: u16,
    pub segments: Vec<Segment>,
    pub interp: bool,
}

fn u16_at(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}
fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn u64_at(b: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(o..o + 8)?.try_into().ok()?))
}

pub fn parse(bytes: &[u8]) -> Result<Image, String> {
    if bytes.get(..4) != Some(b"\x7fELF") || bytes.get(4) != Some(&2) {
        return Err("not an ELF64 image".into());
    }
    let bad = || "truncated ELF header".to_string();
    if u16_at(bytes, 0x12).ok_or_else(bad)? != 62 {
        return Err("not an x86-64 ELF".into());
    }
    let kind = u16_at(bytes, 0x10).ok_or_else(bad)?;
    let base = match kind {
        2 => 0,
        3 => PIE_BASE,
        k => return Err(format!("unsupported ELF type {k}")),
    };
    let entry = u64_at(bytes, 0x18).ok_or_else(bad)? + base;
    let phoff = u64_at(bytes, 0x20).ok_or_else(bad)? as usize;
    let phentsize = u16_at(bytes, 0x36).ok_or_else(bad)?;
    let phnum = u16_at(bytes, 0x38).ok_or_else(bad)?;
    let mut segments = Vec::new();
    let mut phdr = None;
    let mut interp = false;
    for i in 0..phnum as usize {
        let p = phoff + i * phentsize as usize;
        let kind = u32_at(bytes, p).ok_or_else(bad)?;
        let offset = u64_at(bytes, p + 8).ok_or_else(bad)?;
        let vaddr = u64_at(bytes, p + 16).ok_or_else(bad)? + base;
        let filesz = u64_at(bytes, p + 32).ok_or_else(bad)?;
        let memsz = u64_at(bytes, p + 40).ok_or_else(bad)?;
        match kind {
            1 => segments.push(Segment { vaddr, offset, filesz, memsz }),
            3 => interp = true,
            6 => phdr = Some(vaddr),
            _ => {}
        }
    }
    // Without PT_PHDR, the table is where the segment holding offset
    // `phoff` maps it.
    let phdr = phdr
        .or_else(|| {
            segments.iter().find_map(|s| {
                let o = phoff as u64;
                (o >= s.offset && o < s.offset + s.filesz).then(|| s.vaddr + (o - s.offset))
            })
        })
        .unwrap_or(0);
    Ok(Image { entry, base, phdr, phnum, phentsize, segments, interp })
}

/// Maps the segments and copies their file bytes; returns `(lo, hi)`, the
/// page-aligned span the image covers, which is where `brk` starts.
pub fn load(image: &Image, bytes: &[u8], guest: &mut dyn Guest, mapped: &mut crate::kernel::Ranges) -> (u64, u64) {
    let mut lo = u64::MAX;
    let mut hi = 0;
    for seg in &image.segments {
        let start = page_down(seg.vaddr);
        let end = page_up(seg.vaddr + seg.memsz.max(1));
        for (a, b) in mapped.gaps(start, end) {
            assert!(guest.map(a, b - a), "cannot map segment {a:#x}..{b:#x}");
            mapped.insert(a, b);
        }
        lo = lo.min(start);
        hi = hi.max(end);
    }
    for seg in &image.segments {
        let from = seg.offset as usize;
        let to = (seg.offset + seg.filesz) as usize;
        if let Some(data) = bytes.get(from..to.min(bytes.len())) {
            guest.write(seg.vaddr, data);
        }
    }
    (lo, hi)
}
