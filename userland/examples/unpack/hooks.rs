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

/// Loads the running task's id from the scheduler's cell, as a four-byte
/// value.
///
/// The scheduler keeps this cell holding the pid of the task on the machine,
/// updated on every switch and on fork/exec (see
/// [`Process::set_task_id_space`](qcode_userland::Process::set_task_id_space)),
/// so a record a hook makes is stamped with the task that made it and two
/// tasks' work either side of a cooperative switch is never conflated.
fn current_task(emit: &mut Emitter<'_>) -> ValueId {
    let space = emit.state_space(layout::TASK_SPACE);
    let zero = emit.constant(0, 8);
    emit.load_from(space, zero, 4)
}

/// The task lane of the running task: `min(task - TASK_BASE, MAX_TASK_LANES -
/// 1)`, computed branchlessly so the per-task visited map is one sized space
/// reached at an injected index rather than a swapped one.
fn task_lane(emit: &mut Emitter<'_>, task: ValueId) -> ValueId {
    let task = emit.zext(task, 8);
    let base = emit.constant(layout::TASK_BASE, 8);
    let raw = emit.binop(IntBinop::Sub, task, base);
    let max = emit.constant(layout::MAX_TASK_LANES, 8);
    let inside = emit.binop(IntBinop::Less, raw, max);
    let wide = emit.zext(inside, 8);
    let zero = emit.constant(0, 8);
    let mask = emit.binop(IntBinop::Sub, zero, wide);
    let ones = emit.constant(u64::MAX, 8);
    let notmask = emit.binop(IntBinop::Xor, mask, ones);
    let ceil = emit.constant(layout::MAX_TASK_LANES - 1, 8);
    let lo = emit.binop(IntBinop::And, raw, mask);
    let hi = emit.binop(IntBinop::And, ceil, notmask);
    emit.binop(IntBinop::Or, lo, hi)
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

        // Stamp the running task into the site's slot of the site-task table,
        // so the host can say which task wrote the code a site produced —
        // the writer half of a cross-task provenance edge. One store at a
        // constant offset, no branch.
        let task = current_task(emit);
        let sitetask = emit.state_space(layout::SITETASK_SPACE);
        let slot = emit.constant(u64::from(id) * 4, 8);
        emit.store_to(sitetask, task, slot);

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

        // The running task and its lane. `visited` is laned per task, so a
        // block entered by two tasks logs an entry for each; the log's
        // `visited[lane][k]` is one sized space reached at an injected lane.
        let task = current_task(emit);
        let lane = task_lane(emit, task);
        let stride = emit.constant(layout::MAX_BLOCKS, 8);
        let base = emit.binop(IntBinop::Mul, lane, stride);
        let k_wide = emit.constant(k, 8);
        let seat = emit.binop(IntBinop::Add, base, k_wide);
        let flag = emit.load_from(visited, seat, 1);

        let cursor = emit.constant(layout::CURSOR_OFF, 8);
        let at = emit.load_from(entries, cursor, 8);
        let scale = emit.constant(layout::SLOT_SIZE.trailing_zeros().into(), 8);
        let scaled = emit.binop(IntBinop::ShiftLeft, at, scale);
        let slots = emit.constant(layout::SLOTS_OFF, 8);
        let slot = emit.binop(IntBinop::Add, scaled, slots);
        let index = emit.constant(k, 4);
        emit.store_to(entries, index, slot);
        let task_delta = emit.constant(layout::SLOT_TASK_OFF, 8);
        let task_off = emit.binop(IntBinop::Add, slot, task_delta);
        emit.store_to(entries, task, task_off);

        if self.edges {
            let last = emit.constant(layout::LAST_OFF, 8);
            let last_task = emit.constant(layout::LAST_TASK_OFF, 8);
            let pred = emit.load_from(entries, last, 4);
            let pred_task = emit.load_from(entries, last_task, 4);
            let pred_k_delta = emit.constant(layout::SLOT_PRED_K_OFF, 8);
            let pk = emit.binop(IntBinop::Add, slot, pred_k_delta);
            let pred_task_delta = emit.constant(layout::SLOT_PRED_TASK_OFF, 8);
            let pt = emit.binop(IntBinop::Add, slot, pred_task_delta);
            emit.store_to(entries, pred, pk);
            emit.store_to(entries, pred_task, pt);
            emit.store_to(entries, index, last);
            emit.store_to(entries, task, last_task);
        }

        // `1 - visited[lane][k]`: one the first time this task runs the block,
        // zero after, so the slot just written is kept exactly once.
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
    /// The task that entered it.
    pub task: u32,
    /// The block that ran immediately before it, with `--edges`.
    pub pred: Option<u32>,
    /// The task that ran that predecessor, with `--edges`: when it differs
    /// from [`task`](Self::task) the predecessor is a cooperative switch and
    /// not a control-flow edge.
    pub pred_task: Option<u32>,
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
    let u32_at = |slot: &[u8], off: usize| {
        u32::from_le_bytes([slot[off], slot[off + 1], slot[off + 2], slot[off + 3]])
    };
    bytes
        .chunks_exact(layout::SLOT_SIZE as usize)
        .enumerate()
        .map(|(position, slot)| LogEntry {
            k: u32_at(slot, layout::SLOT_K_OFF as usize),
            task: u32_at(slot, layout::SLOT_TASK_OFF as usize),
            // Nothing ran before the first block, and `last` reads as the
            // zero the space was born with, which is a block index: the
            // first slot's predecessor half is not a predecessor.
            pred: (edges && position > 0).then(|| u32_at(slot, layout::SLOT_PRED_K_OFF as usize)),
            pred_task: (edges && position > 0)
                .then(|| u32_at(slot, layout::SLOT_PRED_TASK_OFF as usize)),
        })
        .collect()
}

/// Reads the last task to execute each store site out of
/// [`layout::SITETASK_SPACE`], as `site_tasks[i]` for the site of id `i + 1`.
///
/// A site the run never reached reads as the zero the space was born with,
/// reported as `None`; the pid of a real task is never zero.
pub fn read_site_tasks(
    flat: &qcode_vm::flat::FlatSpaces,
    sitetask: qcode::space::SpaceId,
    sites: usize,
) -> Vec<Option<u32>> {
    let space = qcode::space::MemorySpaceId::Shared(sitetask);
    let Ok(bytes) = flat.read_bytes(space, 4, sites * 4) else {
        return vec![None; sites];
    };
    bytes
        .chunks_exact(4)
        .map(|w| {
            let v = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            (v != 0).then_some(v)
        })
        .collect()
}
