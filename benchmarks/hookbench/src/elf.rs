//! The least ELF64 loading the freestanding Embench images need: the entry
//! point and the `PT_LOAD` segments.

pub struct Segment {
    pub vaddr: u64,
    pub data: Vec<u8>,
    pub memsz: u64,
    pub flags: u32,
}

pub struct Image {
    pub entry: u64,
    pub segments: Vec<Segment>,
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap())
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

pub fn parse(bytes: &[u8]) -> Image {
    assert_eq!(&bytes[..4], b"\x7fELF", "not an ELF image");
    assert_eq!(bytes[4], 2, "not ELF64");
    let entry = u64_at(bytes, 0x18);
    let phoff = u64_at(bytes, 0x20) as usize;
    let phentsize = u16_at(bytes, 0x36) as usize;
    let phnum = u16_at(bytes, 0x38) as usize;
    let mut segments = Vec::new();
    for i in 0..phnum {
        let p = phoff + i * phentsize;
        if u32_at(bytes, p) != 1 {
            continue;
        }
        let flags = u32_at(bytes, p + 4);
        let offset = u64_at(bytes, p + 8) as usize;
        let vaddr = u64_at(bytes, p + 16);
        let filesz = u64_at(bytes, p + 32) as usize;
        let memsz = u64_at(bytes, p + 40);
        segments.push(Segment {
            vaddr,
            data: bytes[offset..offset + filesz].to_vec(),
            memsz,
            flags,
        });
    }
    Image { entry, segments }
}

/// The first writable segment: where the write watch points.
pub fn first_rw(image: &Image) -> Option<u64> {
    image.segments.iter().find(|s| s.flags & 2 != 0).map(|s| s.vaddr)
}

pub const PAGE: u64 = 0x1000;
pub fn page_down(a: u64) -> u64 {
    a & !(PAGE - 1)
}
pub fn page_up(a: u64) -> u64 {
    (a + PAGE - 1) & !(PAGE - 1)
}
