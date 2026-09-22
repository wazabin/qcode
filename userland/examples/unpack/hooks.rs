//! The two compiled hooks: provenance per guest store, and a first-entry log
//! per block.
//!
//! Both are rewrites of the lifted code, so the interpreter and the JIT run
//! them alike, and — this is the whole point — **neither ever stops the
//! run**. [`qcode_userland`] owns the machine's interrupts: `Task::handle_interrupt`
//! crashes the task on any explicit `vm.interrupt(code)` it does not
//! recognise, so a hook that wants to live inside a real process has to do
//! all its bookkeeping in compiled code and let the host read the result
//! afterwards.
//!
//! That rules out control flow as well as exits. A gate of the shape "if
//! this is the first entry, record it" would split every block in three and
//! put a branch on the hot path; instead both hooks compute what they want
//! branchlessly and write it to a [bounded state space](super::layout):
//!
//! - [`ProvenanceHook`] stamps the id of the storing site into the two
//!   shadow bytes of every guest byte a store writes, selecting between the
//!   two tracked windows and a sink with masks rather than branches.
//! - [`EntryHook`] appends the block's index to a log at the cursor and
//!   advances the cursor by `1 - visited[k]`, so an already-visited block
//!   rewrites the slot it will overwrite again and the log ends up holding
//!   exactly the first entries, in order.
//!
//! # A hook must not instrument itself
//!
//! A block is offered to the injectors again every time absorption grows it,
//! and `BlockView::since` means the offer only names what is new — so the
//! guard costs one field access per site. It is still needed: only lifted
//! guest code carries a guest address, and emitted code carries none, so
//! filtering sites down to the ones whose instruction has an address is what
//! keeps a second pass from instrumenting the first pass's work.

use std::{cell::RefCell, rc::Rc};

use qcode::value::{Instruction, ValueId, insn::IntBinop};
use qcode_vm::{BlockView, Emitter, Hook, Site};

use super::layout::{self, Layout};

/// A guest store the provenance hook instrumented.
///
/// Its position in [`Recorder::sites`] is the id stamped into the shadow,
/// minus one; see [`Recorder::site_of`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiteRecord {
    /// The guest address of the storing instruction.
    pub pc: Option<u64>,
    /// The width in bytes of the guest store.
    pub size: usize,
}

/// A block the entry hook instrumented.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockRecord {
    /// The block's guest address.
    pub addr: Option<u64>,
}

/// What the hooks know that the state spaces do not.
///
/// The spaces hold ids; what an id *means* lives here, shared between the
/// two hooks and [`super::driver`] through an `Rc<RefCell<_>>` because a
/// [`Hook`] must be `'static`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Recorder {
    /// Every instrumented store site, in instrumentation order.
    pub sites: Vec<SiteRecord>,
    /// Every instrumented block, in instrumentation order; the index is the
    /// `k` the hook writes into the log.
    pub blocks: Vec<BlockRecord>,
    /// Whether some site had to share the saturating id
    /// [`SITE_ID_MAX`](layout::SITE_ID_MAX).
    pub sites_saturated: bool,
    /// Whether some block went uninstrumented because the log is full.
    pub blocks_saturated: bool,
}

impl Recorder {
    /// The site a shadow id names, if it names one.
    pub fn site_of(&self, id: u16) -> Option<&SiteRecord> {
        if id == 0 {
            return None;
        }
        self.sites.get(usize::from(id) - 1)
    }
}

/// Whether `site`'s anchor came from a guest instruction; see the
/// [module documentation](self).
fn is_guest(block: &BlockView<'_>, site: &Site) -> bool {
    Instruction::from_id(block.ctx, site.anchor())
        .address()
        .is_some()
}

/// Stamps the id of the storing site into the shadow bytes of everything a
/// guest store writes.
///
/// # The branchless select
///
/// Provenance covers two guest ranges that are terabytes apart (see
/// [`super::layout`]), and a store may be in either or in neither. Branching
/// on that would double the block count and put two unpredictable branches on
/// the hot path, so the destination is arithmetic:
///
/// ```text
/// offI = p - image.start;  mI = 0 - (u64)(offI < image.len)
/// offM = p - mmap.start;   mM = 0 - (u64)(offM < mmap.len)
/// dst  = ((2 * offI + image.shadow) & mI) | ((2 * offM + mmap.shadow) & mM)
/// ```
///
/// The windows are disjoint, so at most one mask is set; when neither is,
/// `dst` is zero, which is [`layout::SINK_OFF`] — a scratch slot wider than
/// any stamp. Stack traffic and anything mapped past the mmap window land
/// there and are simply untracked, which the harvest reports rather than
/// guesses at.
///
/// Sixteen QCode operations for a store of four bytes or fewer, no control
/// flow, and no exit: a sample that decrypts a megabyte pays a store per
/// store rather than a VM exit per store.
pub struct ProvenanceHook {
    recorder: Rc<RefCell<Recorder>>,
    layout: Layout,
}

impl ProvenanceHook {
    pub fn new(recorder: Rc<RefCell<Recorder>>, layout: Layout) -> Self {
        Self { recorder, layout }
    }

    /// Registers a site and returns the id to stamp for it.
    fn take_id(&mut self, pc: Option<u64>, size: usize) -> u16 {
        let mut recorder = self.recorder.borrow_mut();
        let index = recorder.sites.len();
        let id = if index >= usize::from(layout::SITE_ID_MAX) {
            recorder.sites_saturated = true;
            layout::SITE_ID_MAX
        } else {
            (index + 1) as u16
        };
        recorder.sites.push(SiteRecord { pc, size });
        id
    }
}

impl Hook for ProvenanceHook {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        block
            .stores()
            .into_iter()
            .filter(|site| is_guest(block, site))
            .collect()
    }

    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let Some((ptr, size, _)) = emit.store_operands() else {
            return;
        };
        let id = self.take_id(emit.address(), size);
        let space = emit.state_space(layout::SHADOW_SPACE);

        let p = emit.zext(ptr, 8);
        let image = selected(emit, p, self.layout.image_window());
        let mmap = selected(emit, p, self.layout.mmap_window());
        let dst = emit.binop(IntBinop::Or, image, mmap);

        // Two shadow bytes per guest byte, in chunks of at most eight so the
        // constant holding the splatted id is a width every backend lowers.
        let total = 2 * size as u64;
        let mut off = 0;
        while off < total {
            let chunk = [8u64, 4, 2]
                .into_iter()
                .find(|c| off + c <= total)
                .unwrap_or(2);
            let mut splat = 0u64;
            for lane in 0..chunk / 2 {
                splat |= u64::from(id) << (16 * lane);
            }
            let value = emit.constant(splat, chunk as usize);
            let at = if off == 0 {
                dst
            } else {
                let delta = emit.constant(off, 8);
                emit.binop(IntBinop::Add, dst, delta)
            };
            emit.store_to(space, value, at);
            off += chunk;
        }
    }
}

/// `(2 * (p - window.start) + window.shadow)` when `p` is in `window`, and
/// zero — the sink — when it is not.
fn selected(emit: &mut Emitter<'_>, p: ValueId, window: layout::Window) -> ValueId {
    let start = emit.constant(window.start, 8);
    let off = emit.binop(IntBinop::Sub, p, start);
    let len = emit.constant(window.len, 8);
    let inside = emit.binop(IntBinop::Less, off, len);

    let one = emit.constant(1, 8);
    let scaled = emit.binop(IntBinop::ShiftLeft, off, one);
    let base = emit.constant(window.shadow, 8);
    let shadow = emit.binop(IntBinop::Add, scaled, base);

    let wide = emit.zext(inside, 8);
    let zero = emit.constant(0, 8);
    let mask = emit.binop(IntBinop::Sub, zero, wide);
    emit.binop(IntBinop::And, shadow, mask)
}

/// Appends a block's index to the first-entry log, without stopping the run
/// and without a branch.
///
/// The log is a cursor followed by slots. Every entry of block `k` writes
/// `k` into the slot *at* the cursor and then advances the cursor by
/// `1 - visited[k]`, raising `visited[k]` on the way — so the first entry
/// keeps its slot and every later one scribbles over a slot that the next
/// first entry will claim anyway. The log therefore ends up holding exactly
/// the blocks that ran, once each, in the order they first ran.
///
/// Ten QCode operations, or fourteen with `--edges`, and no exit.
///
/// # `--edges`
///
/// The header also carries `last`, the index of the block entered most
/// recently, stored unconditionally at every entry. A slot records the
/// `last` it read *before* overwriting it, so a first entry's slot names the
/// block that ran immediately before it — the observed control-flow edge
/// into it. The pairs for entries that did not stick are overwritten with
/// their slot.
pub struct EntryHook {
    recorder: Rc<RefCell<Recorder>>,
    edges: bool,
}

impl EntryHook {
    pub fn new(recorder: Rc<RefCell<Recorder>>, edges: bool) -> Self {
        Self { recorder, edges }
    }

    /// Registers a block and returns its index, or `None` when the log is
    /// full.
    fn take_index(&mut self, addr: Option<u64>) -> Option<u64> {
        let mut recorder = self.recorder.borrow_mut();
        let k = recorder.blocks.len() as u64;
        if k >= layout::MAX_BLOCKS {
            recorder.blocks_saturated = true;
            return None;
        }
        recorder.blocks.push(BlockRecord { addr });
        Some(k)
    }
}

impl Hook for EntryHook {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        if self.recorder.borrow().blocks_saturated {
            return Vec::new();
        }
        block.entry().into_iter().collect()
    }

    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let Some(k) = self.take_index(emit.address()) else {
            return;
        };
        let entries = emit.state_space(layout::ENTRIES_SPACE);
        let visited = emit.state_space(layout::VISITED_SPACE);

        let seat = emit.constant(k, 8);
        let flag = emit.load_from(visited, seat, 1);

        let cursor = emit.constant(layout::CURSOR_OFF, 8);
        let at = emit.load_from(entries, cursor, 8);
        let scale = emit.constant(layout::SLOT_SIZE.trailing_zeros().into(), 8);
        let scaled = emit.binop(IntBinop::ShiftLeft, at, scale);
        let slots = emit.constant(layout::SLOTS_OFF, 8);
        let slot = emit.binop(IntBinop::Add, scaled, slots);
        let index = emit.constant(k, 4);
        emit.store_to(entries, index, slot);

        if self.edges {
            let last = emit.constant(layout::LAST_OFF, 8);
            let pred = emit.load_from(entries, last, 4);
            let four = emit.constant(4, 8);
            let second = emit.binop(IntBinop::Add, slot, four);
            emit.store_to(entries, pred, second);
            emit.store_to(entries, index, last);
        }

        // `1 - visited[k]`: one the first time the block runs, zero after,
        // so the slot just written is kept exactly once.
        let wide = emit.zext(flag, 8);
        let one = emit.constant(1, 8);
        let step = emit.binop(IntBinop::Sub, one, wide);
        let next = emit.binop(IntBinop::Add, at, step);
        emit.store_to(entries, next, cursor);

        let raised = emit.constant(1, 1);
        emit.store_to(visited, raised, seat);
    }
}

/// One first entry, as the log recorded it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogEntry {
    /// The block's index in [`Recorder::blocks`].
    pub k: u32,
    /// The block that ran immediately before it, with `--edges`.
    pub pred: Option<u32>,
}

/// Reads the first-entry log back out of [`layout::ENTRIES_SPACE`].
///
/// The cursor counts the entries that stuck; the slots past it hold whatever
/// the last already-visited block wrote and are not read.
pub fn read_log(
    flat: &qcode_vm::flat::FlatSpaces,
    entries: qcode::space::SpaceId,
    edges: bool,
) -> Vec<LogEntry> {
    let space = qcode::space::MemorySpaceId::Shared(entries);
    let Ok(cursor) = flat.read_u128(space, layout::CURSOR_OFF, 8) else {
        return Vec::new();
    };
    let count = (cursor as u64).min(layout::MAX_BLOCKS);
    let Ok(bytes) = flat.read_bytes(
        space,
        layout::SLOTS_OFF,
        (count * layout::SLOT_SIZE) as usize,
    ) else {
        return Vec::new();
    };
    bytes
        .chunks_exact(layout::SLOT_SIZE as usize)
        .enumerate()
        .map(|(position, slot)| LogEntry {
            k: u32::from_le_bytes([slot[0], slot[1], slot[2], slot[3]]),
            // Nothing ran before the first block, and `last` reads as the
            // zero the space was born with, which is a block index: the
            // first slot's second half is not a predecessor.
            pred: (edges && position > 0)
                .then(|| u32::from_le_bytes([slot[4], slot[5], slot[6], slot[7]])),
        })
        .collect()
}
