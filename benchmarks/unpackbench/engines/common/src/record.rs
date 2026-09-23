//! The provenance windows, the shadow's layout, and the facts a run's hooks
//! collect. The windows are the unpack example's (`layout.rs`): the image
//! plus 6 MiB, and the first ~2 MiB of the mmap arena at `MMAP_BASE`.
//!
//! The shadow is two bytes (a little-endian site id) per tracked guest byte,
//! behind a sink that absorbs stores outside both windows, exactly as
//! `unpack.shadow` is laid out, so an engine that stamps from compiled code
//! (icicle) and one that stamps from a callback (Unicorn) share one buffer
//! format and one harvest.

use crate::kernel::MMAP_BASE;

pub const PAGE: u64 = 4096;
/// `MAX_FLAT_SPACE` in `vm/src/flat.rs`, which sizes the unpack example's
/// shadow; restated so the windows here have the same extent.
const QCODE_SPACE_MAX: u64 = 1 << 24;
pub const IMAGE_WINDOW_LEN: u64 = 6 << 20;
pub const MMAP_WINDOW_LEN: u64 = ((QCODE_SPACE_MAX - 16 - 64) / 2 - IMAGE_WINDOW_LEN) & !(PAGE - 1);

/// Where a store outside both windows stamps: wide enough for the widest
/// store (32 bytes, 64 shadow bytes) with room to spare.
pub const SINK_SIZE: u64 = 128;
pub const IMAGE_SHADOW_OFF: u64 = SINK_SIZE;
pub const MMAP_SHADOW_OFF: u64 = IMAGE_SHADOW_OFF + 2 * IMAGE_WINDOW_LEN;
/// Past the last shadow byte, so a wide store at the top of a window stays
/// inside the buffer.
pub const SLACK: u64 = 128;
pub const SHADOW_LEN: usize = (MMAP_SHADOW_OFF + 2 * MMAP_WINDOW_LEN + SLACK) as usize;

/// How many blocks the first-entry log can hold before it gives up.
pub const MAX_BLOCKS: u64 = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub start: u64,
    pub len: u64,
    /// Byte offset of the first guest byte's shadow.
    pub shadow: u64,
}

impl Window {
    pub fn end(&self) -> u64 {
        self.start + self.len
    }
    pub fn contains(&self, a: u64) -> bool {
        a >= self.start && a < self.end()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub image: Window,
    pub mmap: Window,
}

impl Layout {
    pub fn new(image_lo: u64, image_hi: u64) -> Result<Self, String> {
        if image_hi - image_lo > IMAGE_WINDOW_LEN {
            return Err(format!(
                "the image spans {} bytes, more than the {IMAGE_WINDOW_LEN}-byte provenance window",
                image_hi - image_lo
            ));
        }
        Ok(Self::at(image_lo, MMAP_BASE))
    }

    /// The windows for an image at `image_lo` and an mmap arena at
    /// `mmap_base` (Qiling's arena is elsewhere).
    pub fn at(image_lo: u64, mmap_base: u64) -> Self {
        Self {
            image: Window { start: image_lo, len: IMAGE_WINDOW_LEN, shadow: IMAGE_SHADOW_OFF },
            mmap: Window { start: mmap_base, len: MMAP_WINDOW_LEN, shadow: MMAP_SHADOW_OFF },
        }
    }

    pub fn windows(&self) -> [Window; 2] {
        [self.image, self.mmap]
    }

    pub fn in_window(&self, a: u64) -> bool {
        self.image.contains(a) || self.mmap.contains(a)
    }

    /// The shadow byte offset of `a`, or the sink's (0).
    #[inline]
    pub fn shadow_of(&self, a: u64) -> u64 {
        if self.image.contains(a) {
            self.image.shadow + 2 * (a - self.image.start)
        } else if self.mmap.contains(a) {
            self.mmap.shadow + 2 * (a - self.mmap.start)
        } else {
            0
        }
    }

    /// The site ids of `start..end`, if one window covers all of it.
    pub fn ids(&self, shadow: &[u16], start: u64, end: u64) -> Option<Vec<u16>> {
        if end <= start {
            return None;
        }
        let w = self.windows().into_iter().find(|w| w.contains(start))?;
        if end > w.end() {
            return None;
        }
        let from = ((w.shadow + 2 * (start - w.start)) / 2) as usize;
        Some(shadow[from..from + (end - start) as usize].to_vec())
    }
}

/// A store site: the pc of a storing instruction, first-seen order; its id is
/// its index plus one (0 means "never written").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Site {
    pub pc: u64,
    pub size: usize,
}

/// A block the entry hook saw, by the index the log refers to it with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    pub addr: u64,
    pub end: u64,
}

/// Everything a run's hooks leave for the harvest.
#[derive(Debug, Default, Clone)]
pub struct Record {
    pub sites: Vec<Site>,
    pub blocks: Vec<Block>,
    /// First entries in order: `(block index, predecessor index)`; the
    /// predecessor only with `--edges`.
    pub log: Vec<(u32, Option<u32>)>,
    /// Two bytes per tracked guest byte, [`SHADOW_LEN`] bytes long.
    pub shadow: Vec<u16>,
    pub sites_saturated: bool,
    pub blocks_saturated: bool,
}
