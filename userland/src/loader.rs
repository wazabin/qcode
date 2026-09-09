//! Places a static ELF64 x86-64 executable into guest memory.
//!
//! `PT_LOAD` segments come from [`wazabin_binary::elf::ElfBinary`]; the pieces
//! it does not expose — the file type (`ET_EXEC` versus `ET_DYN`), the program
//! header table's location for `AT_PHDR`, and `PT_TLS` — are read from the
//! headers here.
//!
//! Mappings are page-granular: each segment's `[vaddr, vaddr + memsz)` is
//! rounded out to pages and mapped zero-filled with the segment's permissions,
//! then the file-backed bytes are written over it. Anything past `filesz` is
//! thereby `.bss`, zeroed. Where two segments share a page the file bytes of
//! each keep their own segment's permissions (permissions are per byte in this
//! MMU), which is stricter than Linux rather than looser.

use qcode_vm::{Mmu, PAGE_SIZE, perm};
use wazabin_binary::{Arch, elf::ElfBinary};

/// Where a position-independent executable is placed. Linux picks a random
/// address near here; the environment is deterministic.
pub const PIE_BASE: u64 = 0x5555_5555_0000;

const ET_EXEC: u16 = 2;
const ET_DYN: u16 = 3;
const PT_LOAD: u32 = 1;
const PT_PHDR: u32 = 6;
const PT_TLS: u32 = 7;

/// A `PT_TLS` segment, for a guest that sets up thread-local storage itself.
#[derive(Debug, Clone, Copy)]
pub struct Tls {
    pub vaddr: u64,
    pub filesz: u64,
    pub memsz: u64,
    pub align: u64,
}

/// One mapped segment, after relocation.
#[derive(Debug, Clone, Copy)]
pub struct Mapped {
    pub start: u64,
    pub end: u64,
    pub perm: u8,
}

/// What the loader learned about an image, for the auxiliary vector and the
/// heap.
#[derive(Debug, Clone)]
pub struct LoadedImage {
    /// The relocated entry point.
    pub entry: u64,
    /// Load bias: `PIE_BASE` for `ET_DYN`, zero for `ET_EXEC`.
    pub base: u64,
    /// The guest address of the program header table, if it is mapped.
    pub phdr: Option<u64>,
    pub phnum: u16,
    pub phentsize: u16,
    /// The first page after the highest segment: where the heap starts.
    pub brk: u64,
    pub tls: Option<Tls>,
    pub segments: Vec<Mapped>,
}

#[derive(Debug)]
pub enum LoadError {
    Parse(String),
    NotX86_64,
    Truncated,
    UnsupportedType(u16),
    Map(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Parse(e) => write!(f, "ELF parse error: {e}"),
            LoadError::NotX86_64 => write!(f, "not an x86-64 ELF"),
            LoadError::Truncated => write!(f, "ELF header is truncated"),
            LoadError::UnsupportedType(t) => {
                write!(f, "unsupported ELF type {t} (need ET_EXEC or ET_DYN)")
            }
            LoadError::Map(e) => write!(f, "cannot map segment: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

pub fn page_down(addr: u64) -> u64 {
    addr & !(PAGE_SIZE - 1)
}

pub fn page_up(addr: u64) -> u64 {
    (addr + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

/// Raw program header fields the loader needs.
struct Phdr {
    kind: u32,
    offset: u64,
    vaddr: u64,
    filesz: u64,
    memsz: u64,
    align: u64,
}

fn headers(image: &[u8]) -> Result<(u16, u64, u16, u16, Vec<Phdr>), LoadError> {
    if image.len() < 64 {
        return Err(LoadError::Truncated);
    }
    let half = |o: usize| u16::from_le_bytes(image[o..o + 2].try_into().unwrap());
    let word = |o: usize| u32::from_le_bytes(image[o..o + 4].try_into().unwrap());
    let long = |o: usize| u64::from_le_bytes(image[o..o + 8].try_into().unwrap());
    let e_type = half(16);
    let phoff = long(32);
    let phentsize = half(54);
    let phnum = half(56);
    let mut phdrs = Vec::with_capacity(phnum as usize);
    for i in 0..phnum as usize {
        let p = phoff as usize + i * phentsize as usize;
        if p + 56 > image.len() {
            return Err(LoadError::Truncated);
        }
        phdrs.push(Phdr {
            kind: word(p),
            offset: long(p + 8),
            vaddr: long(p + 16),
            filesz: long(p + 32),
            memsz: long(p + 40),
            align: long(p + 48),
        });
    }
    Ok((e_type, phoff, phentsize, phnum, phdrs))
}

/// Loads `image` into `mmu`.
pub fn load(image: &[u8], mmu: &mut Mmu) -> Result<LoadedImage, LoadError> {
    let elf = ElfBinary::parse(image).map_err(|e| LoadError::Parse(e.to_string()))?;
    if elf.architecture != Arch::X86_64 {
        return Err(LoadError::NotX86_64);
    }
    let (e_type, phoff, phentsize, phnum, phdrs) = headers(image)?;
    let base = match e_type {
        ET_EXEC => 0,
        ET_DYN => PIE_BASE,
        other => return Err(LoadError::UnsupportedType(other)),
    };

    let mut segments = Vec::new();
    let mut brk = 0;
    // Pages first, so a later segment sharing a page with an earlier one does
    // not zero the earlier one's bytes.
    for seg in &elf.segments {
        let start = page_down(seg.start + base);
        let end = page_up(seg.start + base + seg.mem_size);
        let mut bits = perm::READ | perm::INIT;
        if seg.executable {
            bits |= perm::EXEC;
        }
        if seg.writable {
            bits |= perm::WRITE;
        }
        for page in (start..end).step_by(PAGE_SIZE as usize) {
            // A page shared with a previous segment keeps its contents; only
            // fresh pages are zeroed.
            if mmu.permissions(page) & perm::MAP == 0 {
                mmu.map(page, PAGE_SIZE, bits)
                    .map_err(|e| LoadError::Map(e.to_string()))?;
            }
        }
        segments.push(Mapped {
            start,
            end,
            perm: bits,
        });
        brk = brk.max(end);
    }
    for (seg, mapped) in elf.segments.iter().zip(&segments) {
        if !seg.data.is_empty() {
            mmu.write_unchecked(seg.start + base, &seg.data, mapped.perm);
        }
    }

    // AT_PHDR: PT_PHDR names it outright; otherwise the table lives inside the
    // PT_LOAD that covers its file offset.
    let phdr = phdrs
        .iter()
        .find(|p| p.kind == PT_PHDR)
        .map(|p| p.vaddr + base)
        .or_else(|| {
            phdrs
                .iter()
                .filter(|p| p.kind == PT_LOAD)
                .find(|p| phoff >= p.offset && phoff < p.offset + p.filesz)
                .map(|p| p.vaddr + (phoff - p.offset) + base)
        });
    let tls = phdrs.iter().find(|p| p.kind == PT_TLS).map(|p| Tls {
        vaddr: p.vaddr + base,
        filesz: p.filesz,
        memsz: p.memsz,
        align: p.align,
    });

    Ok(LoadedImage {
        entry: elf.analysis.entrypoint + base,
        base,
        phdr,
        phnum,
        phentsize,
        brk,
        tls,
        segments,
    })
}
