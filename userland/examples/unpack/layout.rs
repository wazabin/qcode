//! Where the hooks keep their state, and which guest bytes they describe.
//!
//! Nothing here is guest memory. The three tables the compiled hooks of
//! [`super::hooks`] read and write are *bounded flat state spaces*
//! ([`Vm::state_space`](qcode_vm::Vm::state_space)): host buffers of a fixed
//! length that compiled code indexes by a computed offset off a base pointer,
//! with one bound check and no TLB, no permissions and no guest visibility.
//!
//! ```text
//! unpack.shadow    two bytes of provenance per tracked guest byte, plus a sink
//! unpack.entries   a cursor, the last block entered, and the first-entry log
//! unpack.visited   one byte per instrumented block
//! ```
//!
//! # Why the window is two ranges
//!
//! A flat space may hold at most `1 << 24` bytes (`MAX_FLAT_SPACE` in
//! `vm/src/flat.rs`), so the shadow can describe about eight megabytes of
//! guest address space — and [`qcode_userland`] spreads the interesting
//! bytes over two places far apart: the image and its `brk` heap sit where
//! the ELF was linked (`0x400000` for a non-PIE), while a hint-less `mmap`
//! is served from `0x7f00_0000_0000` upward (`MMAP_BASE` in
//! `userland/src/process.rs`). One contiguous window over both would be 139
//! terabytes wide.
//!
//! So provenance tracks two windows — [`Layout::image_window`], the image
//! plus room for the heap, and [`Layout::mmap_window`], where anonymous
//! mappings land — and the hook selects between their two shadow ranges
//! branchlessly. Everything else (the stack, a mapping placed past the mmap
//! window) lands in the [sink](SINK_OFF) and is untracked; the harvest says
//! so rather than guessing.

/// The largest a flat space may be: `MAX_FLAT_SPACE` in `vm/src/flat.rs`.
///
/// Not re-exported by the VM, so it is restated here, and `tests/unpack.rs`
/// checks that the shadow really fits inside it.
pub const STATE_SPACE_MAX: u64 = 1 << 24;

/// Guest page size, for rounding the windows.
pub const PAGE: u64 = 4096;

/// The space holding two bytes of provenance per tracked guest byte.
pub const SHADOW_SPACE: &str = "unpack.shadow";
/// The space holding the first-entry log.
pub const ENTRIES_SPACE: &str = "unpack.entries";
/// The space holding one byte per instrumented block.
pub const VISITED_SPACE: &str = "unpack.visited";

// ---- unpack.shadow

/// Where a store outside both windows stamps its site id.
///
/// At offset zero on purpose: the branchless select of
/// [`ProvenanceHook`](super::hooks::ProvenanceHook) ands each candidate
/// offset with the mask of its window test and ors the results, so "in
/// neither window" falls out as offset zero without a third term.
pub const SINK_OFF: u64 = 0;
/// How long the sink is: wider than any single shadow stamp.
pub const SINK_SIZE: u64 = 16;

/// Bytes left past the last shadow byte so that a wide store at the very top
/// of a window cannot reach past the space's bound.
///
/// A store stamps `2 * size` bytes; 64 covers every width the x86-64
/// specification lifts, up to a 32-byte store.
pub const SLACK: u64 = 64;

/// Where the image window's shadow starts.
pub const IMAGE_SHADOW_OFF: u64 = SINK_OFF + SINK_SIZE;

/// How much guest memory the image window covers: the image itself and the
/// room `brk` grows into above it.
pub const IMAGE_WINDOW_LEN: u64 = 6 << 20;

/// Where the mmap window's shadow starts.
pub const MMAP_SHADOW_OFF: u64 = IMAGE_SHADOW_OFF + 2 * IMAGE_WINDOW_LEN;

/// Where [`qcode_userland`] serves a hint-less `mmap` from: `MMAP_BASE` in
/// `userland/src/process.rs`, which is private to that crate.
pub const MMAP_WINDOW_BASE: u64 = 0x7f00_0000_0000;

/// How much of the mmap arena the shadow can still describe: whatever the
/// flat-space cap leaves once the image window, the sink and the slack are
/// paid for, rounded down to a page.
pub const MMAP_WINDOW_LEN: u64 =
    ((STATE_SPACE_MAX - SINK_SIZE - SLACK) / 2 - IMAGE_WINDOW_LEN) & !(PAGE - 1);

/// The length of [`SHADOW_SPACE`].
pub const SHADOW_LEN: usize = (MMAP_SHADOW_OFF + 2 * MMAP_WINDOW_LEN + SLACK) as usize;

/// The largest site id two shadow bytes can hold; id 0 means "never
/// written", so the id of site *index* `i` is `i + 1`.
pub const SITE_ID_MAX: u16 = u16::MAX;

// ---- unpack.entries and unpack.visited

/// How many blocks the entry hook can instrument before it gives up.
pub const MAX_BLOCKS: u64 = 1 << 17;

/// The cursor: how many first entries the log holds, as a `u64`.
pub const CURSOR_OFF: u64 = 0;
/// The index of the block entered most recently, as a `u32`; maintained
/// only with `--edges`.
pub const LAST_OFF: u64 = 8;
/// Where the log's slots start.
pub const SLOTS_OFF: u64 = 16;
/// One slot: the block index, then the index of the block that ran before
/// it.
pub const SLOT_SIZE: u64 = 8;

/// The length of [`ENTRIES_SPACE`]: one slot more than there are blocks, so
/// that the slot at the cursor is always writable.
pub const ENTRIES_LEN: usize = (SLOTS_OFF + SLOT_SIZE * (MAX_BLOCKS + 1)) as usize;

/// The length of [`VISITED_SPACE`].
pub const VISITED_LEN: usize = MAX_BLOCKS as usize;

/// A guest byte range provenance tracks, and where its shadow lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    /// The first guest byte the window covers.
    pub start: u64,
    /// How many bytes it covers.
    pub len: u64,
    /// The offset in [`SHADOW_SPACE`] of the first byte's shadow.
    pub shadow: u64,
}

impl Window {
    /// One past the last guest byte.
    pub fn end(&self) -> u64 {
        self.start + self.len
    }

    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.start && addr < self.end()
    }

    /// The shadow offset of `addr`, if the window covers it.
    pub fn shadow_of(&self, addr: u64) -> Option<u64> {
        self.contains(addr)
            .then(|| self.shadow + (addr - self.start) * 2)
    }
}

/// The two windows of a loaded image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    image: Window,
    mmap: Window,
}

/// Why an image has no layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// The image spans more than the image window can describe.
    ImageTooWide { span: u64 },
    /// The image was placed where the mmap window is.
    ImageOverlapsMmap { image_lo: u64 },
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutError::ImageTooWide { span } => write!(
                f,
                "the image spans {span} bytes, more than the {IMAGE_WINDOW_LEN}-byte \
                 provenance window"
            ),
            LayoutError::ImageOverlapsMmap { image_lo } => write!(
                f,
                "the image at {image_lo:#x} overlaps the mmap window at \
                 {MMAP_WINDOW_BASE:#x}"
            ),
        }
    }
}

impl std::error::Error for LayoutError {}

impl Layout {
    /// The layout of an image mapped over `image_lo..image_hi`.
    pub fn new(image_lo: u64, image_hi: u64) -> Result<Self, LayoutError> {
        let span = image_hi.saturating_sub(image_lo);
        if span > IMAGE_WINDOW_LEN {
            return Err(LayoutError::ImageTooWide { span });
        }
        let image = Window {
            start: image_lo,
            len: IMAGE_WINDOW_LEN,
            shadow: IMAGE_SHADOW_OFF,
        };
        let mmap = Window {
            start: MMAP_WINDOW_BASE,
            len: MMAP_WINDOW_LEN,
            shadow: MMAP_SHADOW_OFF,
        };
        if image.start < mmap.end() && mmap.start < image.end() {
            return Err(LayoutError::ImageOverlapsMmap { image_lo });
        }
        Ok(Self { image, mmap })
    }

    /// The image and the heap that grows above it.
    pub fn image_window(&self) -> Window {
        self.image
    }

    /// Where [`qcode_userland`]'s anonymous mappings land.
    pub fn mmap_window(&self) -> Window {
        self.mmap
    }

    /// Whether provenance tracks `addr`.
    pub fn in_window(&self, addr: u64) -> bool {
        self.image.contains(addr) || self.mmap.contains(addr)
    }

    /// The offset in [`SHADOW_SPACE`] of `addr`'s two provenance bytes.
    pub fn shadow_of(&self, addr: u64) -> Option<u64> {
        self.image
            .shadow_of(addr)
            .or_else(|| self.mmap.shadow_of(addr))
    }

    /// The longest prefix of `start..end` that one window covers, as
    /// `(shadow offset, length in guest bytes)`.
    ///
    /// A range that straddles a window edge is reported up to the edge: the
    /// shadow is contiguous within a window and nowhere else, so a reader
    /// walks window by window.
    pub fn shadow_run(&self, start: u64, end: u64) -> Option<(u64, u64)> {
        let window = [self.image, self.mmap]
            .into_iter()
            .find(|window| window.contains(start))?;
        let hi = end.min(window.end());
        if hi <= start {
            return None;
        }
        Some((window.shadow_of(start)?, hi - start))
    }
}
