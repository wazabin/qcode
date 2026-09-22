//! Memoized lifting: the QCode of an encoding's shape, replayed at another
//! address with other operand values.
//!
//! Machine code repeats. Over the superset of every byte offset of a binary
//! a few hundred thousand distinct encodings account for millions of lifts,
//! and a linear disassembly is more than half repeats too. Lowering one is
//! expensive twice over — SLEIGH expands the constructor's semantics, then
//! the builder types, interns and links every operation — and both costs are
//! per operation, so the vector forms whose per-lane macros run to hundreds
//! of operations pay them hundreds of times.
//!
//! Encodings repeat more still once their immediates and displacements are
//! set aside: `mov rax, [rbp - 0x18]` and `mov rax, [rbp - 0x20]` lower to
//! the same operations around different constants. The decoder reports,
//! with an instruction, its [`Shape`]: which bits chose the constructors,
//! and which fields it read only for their value — the parameters. A
//! [`LiftCache`] remembers what a shape lowered to, as a [`Template`]: the
//! instruction's blocks, temporaries and operations with every operand
//! renumbered relative to the template, and every constant held as the
//! value it had at the captured instance plus the address delta, when it
//! moves with the address, plus the delta of each parameter it moves with.
//! Replaying a template into a construction is a walk over that record — a
//! block, a temporary or an operation pushed per entry, with its type and
//! constants resolved once per replay rather than inferred per operation —
//! so a hit costs the IR and nothing around it.
//!
//! Registers are parameters too. `mov rax, [rdi + 8]` and `mov rbx,
//! [rsi + 8]` are one shape whose register fields — the fields the decoder
//! read as an index into an `attach variables` table — differ; a
//! [`RegisterField`](sleigh::RegisterField) is left out of the mask, and
//! the template holds every varnode an operation names as a slot that is
//! either fixed or the register some field's value picks from a table, or
//! a lane of it: the byte at the bottom of it, the low dword. On xul.dll's
//! ten million instructions that is 13 500 shapes in place of 88 000.
//!
//! # Exactness
//!
//! SLEIGH is the only source of semantics here; the cache decides nothing
//! about an instruction. What it must decide is how every value of the
//! lowered IR depends on the address and on each parameter, and it measures
//! that on the IR rather than parsing the encoding. On a miss the lift is
//! captured with one slot per constant use, and the instruction is lifted
//! again into a private store, usually twice: a near probe at a distant
//! address with a low free bit of every parameter flipped, each in a
//! different bit so no two move by the same amount, and a far probe with
//! every free bit of every parameter flipped. Each probe's capture must
//! have the same structure as the first, and is compared with it slot by
//! slot: a value of the near probe moved by the sum of some of the probe's
//! deltas, in its width, and by no other sum, moves with exactly those
//! sources — the address, or parameters — and the far probe must move it
//! by exactly the sum of theirs, so that a lane selector masked out of an
//! immediate, which moves by one when its low bit flips and not at all
//! when a high one does, is caught. When a near probe's move fits two
//! sums, or a perturbation lifts to another structure, the miss is probed
//! one thing at a time instead: the address alone, then each parameter
//! by its lowest free bit and by every free bit. The probe distance is
//! larger than 4 GiB so that a value the specification truncates to 32
//! bits fails the comparison instead of matching by accident.
//!
//! The near probe flips a free bit of every register field as well, and a
//! varnode slot that changed must be the register — or the same lane of
//! the register — that exactly one field's table binds to its old and new
//! values. Fields bound to different registers flip different bits, so
//! that a byte of one register cannot pass for a byte of another; fields
//! bound to one register flip alike and stay bound alike. Which is the
//! catch with registers: a lift's structure can turn on two of them
//! coinciding — `mov eax, eax` lowers to no move at all — so a template
//! records which of the registers its instance named were one register,
//! among the fields' and the ones the lift names on its own, and serves
//! only instances that coincide the same way; a shape holds one template
//! per way. [`LiftCache::validating`] lifts every hit for real as well and
//! compares, for measuring these claims over a corpus.
//!
//! A value that moves by anything else marks the shape uncacheable, so it is
//! lifted for real every time and never probed again. Not every shape is
//! linear in its parameters, though, and not every instance can be
//! perturbed without changing the lift's structure — a branch to itself
//! names its own entry block. Such a miss is remembered by its exact
//! encoding instead, as an entry whose mask covers every bit, and serves
//! that encoding alone.
//!
//! # Lookup
//!
//! Entries are keyed by the lowering flag, the decode context and the
//! shape's masked bytes, and found by walking a trie over the instruction
//! stream: each node holds, per mask byte some shape applies there, its
//! children by masked value, and a leaf — at the shape's last masked byte,
//! the rest deciding nothing — holds a shape's entry with the patterns it
//! excludes: the more specific candidates the decoder passed over, which
//! an encoding of the shape never matches. A lookup needs no decode: it
//! walks the bytes, tries every mask at a node, and passes over a leaf
//! whose exclusions reject the stream. The tries are sharded by the
//! leading byte under the bits every shape's mask keeps of it, so that
//! threads mostly lock apart. A session decoding under another context or
//! lowering calls differently never sees another's entries, and the cache
//! is bound to one specification and refuses a lifter of another.
//!
//! A template records debug names, so a replay into a target that
//! [names](qcode::lift::LiftTarget::without_debug_names) its values names
//! them as a fresh lift would.

use std::{
    borrow::Cow,
    cell::RefCell,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering},
    },
};

use qcode::{
    address_index::AddressIndex,
    context::Context,
    lift::{
        CallTarget, Construction, Continuation, Exit, ExitArm, ExitKind, LiftTarget, Lifted,
        ScratchStore,
    },
    space::{LocalMemorySpaceId, MemorySpaceId},
    types::{TypeId, TypeRepr},
    value::{
        BasicBlock, BlockId, FunctionBody, FunctionId, Instruction, InstructionId, LiteralId,
        LocalBlockId, LocalInsnId, LocalTempId, LocalTempSpaceId, LocalValueId, Temp, TempId,
        TempRef, TempSpace, TempSpaceId, Varnode, VarnodeId,
        insn::{Callee, Mnemonic},
        view::ModuleView,
    },
};
use rustc_hash::FxHashMap as HashMap;
use sleigh::{
    CompiledSpec, ContextBytes, Exclusion, FieldTableId, Instruction as Decoded, ParamField, Shape,
    SpecFingerprint,
};
use smallvec::SmallVec;

use crate::{LiftError, SleighLifter, decode::FixedDecoder};

/// How far from the instruction's address the probe lift is made. Past
/// 4 GiB, so a 32-bit truncation of an address-derived value cannot pass for
/// an offset; odd in every byte, so no alignment of the two addresses
/// coincides.
const PROBE_DISTANCE: u64 = 0x1_0305_0709_0b0d;

/// The longest key prefix: a flag byte and the context bytes.
const MAX_PREFIX: usize = 32;

/// The most parameters a template carries. A shape with more is not cached.
pub(crate) const MAX_PARAMS: usize = 8;

/// The most register parameters a template carries.
pub(crate) const MAX_REGS: usize = 6;

/// One shard per leading byte of the instruction, under the shard mask.
const SHARDS: usize = 256;

/// Counters of a [`LiftCache`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Lifts served by replaying a template.
    pub hits: u64,
    /// Lifts of a shape not yet in the cache; each was lifted for real and
    /// probed.
    pub misses: u64,
    /// Lifts made to probe misses, over and above the miss's own lift: one
    /// with the address moved and every parameter's lowest free bit
    /// flipped, one with every free bit flipped when a parameter has more
    /// than one, and — when a probe is ambiguous or changes the lift's
    /// structure — one at a time.
    pub probes: u64,
    /// Lifts of a shape the probe found uncacheable; each was lifted for
    /// real. Counted per lift, not per shape.
    pub uncacheable: u64,
    /// Misses whose shape could not be parameterized from that instance —
    /// a constant not linear in a field, a branch target no perturbation
    /// keeps off the fall-through — and were remembered by their exact
    /// encoding instead.
    pub exact: u64,
    /// Hits whose validation lift disagreed with the template. Only counted
    /// when [validating](LiftCache::validating); each such hit was discarded,
    /// lifted for real, and its entry evicted.
    pub validation_failures: u64,
    /// Shapes held, as templates or as uncacheable.
    pub entries: usize,
}

enum Kind {
    /// The shape's templates: one per way its registers coincide.
    Templates(Vec<Arc<Template>>),
    Uncacheable,
}

/// The most templates a shape holds.
const MAX_TEMPLATES: usize = 8;

/// What a leaf of the trie holds: a shape's entry and the patterns no
/// encoding of the shape matches.
struct Entry {
    /// The shape's instructions' length: past the leaf's depth, its bytes
    /// are all unmasked.
    len: usize,
    exclusions: Box<[Exclusion]>,
    kind: Kind,
}

impl Entry {
    /// Whether `bytes`, a stream agreeing with the shape's masked bytes,
    /// starts an instruction of the shape. An exclusion the stream is too
    /// short to test does not match, as it would not for the decoder
    /// reading the same stream.
    fn admits(&self, bytes: &[u8]) -> bool {
        bytes.len() >= self.len && !self.exclusions.iter().any(|e| e.matches(bytes))
    }
}

/// How deep in the trie a shape's leaf sits: past its last masked byte,
/// the bytes decide nothing. A more specific shape whose masked bytes
/// reach further is still found, since the shorter shape's exclusions
/// refuse its instructions and the walk goes on.
fn keyed(mask: &[u8]) -> usize {
    mask.iter()
        .rposition(|byte| *byte != 0)
        .map_or(0, |i| i + 1)
}

/// A node of the lookup trie, at one byte of the instruction stream.
#[derive(Default)]
struct Node {
    /// Per mask byte some shape applies at this byte, the children by the
    /// byte's masked value. Almost always one arm, held in the node.
    arms: SmallVec<[Arm; 1]>,
    /// The shape whose instructions end here.
    leaf: Option<Entry>,
}

/// The children of a [`Node`] under one mask: the masked values, sorted
/// and held with the node for the usual fan-out, apart from the nodes, so
/// a walk searches one run of bytes rather than hashing or striding over
/// nodes.
#[derive(Default)]
struct Arm {
    mask: u8,
    values: SmallVec<[u8; 14]>,
    children: Vec<Node>,
}

impl Arm {
    fn child(&self, value: u8) -> Option<&Node> {
        let index = if self.values.len() <= 8 {
            self.values.iter().position(|&v| v == value)?
        } else {
            self.values.binary_search(&value).ok()?
        };
        Some(&self.children[index])
    }
}

impl Node {
    /// The entry of the shape `bytes` starts, searching from `depth`. With
    /// `exact`, `bytes` is one instruction of that length and only a leaf
    /// of that length counts; an exclusion reaching past it cannot have
    /// matched, or the decoder would have taken the longer candidate.
    fn find(&self, bytes: &[u8], depth: usize, exact: Option<usize>) -> Option<&Entry> {
        if let Some(entry) = &self.leaf
            && exact.is_none_or(|len| len == entry.len)
            && entry.admits(bytes)
        {
            return Some(entry);
        }
        if exact.is_some_and(|len| depth >= len) {
            return None;
        }
        let byte = *bytes.get(depth)?;
        for arm in &self.arms {
            if let Some(child) = arm.child(byte & arm.mask)
                && let Some(entry) = child.find(bytes, depth + 1, exact)
            {
                return Some(entry);
            }
        }
        None
    }

    /// The node at the end of the path of `mask` and `masked`, made if
    /// missing.
    fn walk_mut(&mut self, mask: &[u8], masked: &[u8]) -> &mut Node {
        let mut node = self;
        for (&mask, &value) in mask.iter().zip(masked) {
            let arm = match node.arms.iter().position(|a| a.mask == mask) {
                Some(arm) => arm,
                None => {
                    node.arms.push(Arm {
                        mask,
                        ..Arm::default()
                    });
                    node.arms.len() - 1
                }
            };
            let arm = &mut node.arms[arm];
            let index = match arm.values.binary_search(&value) {
                Ok(index) => index,
                Err(index) => {
                    arm.values.insert(index, value);
                    arm.children.insert(index, Node::default());
                    index
                }
            };
            node = &mut arm.children[index];
        }
        node
    }

    /// The node at the end of the path of `mask` and `masked`, if present.
    fn get_mut(&mut self, mask: &[u8], masked: &[u8]) -> Option<&mut Node> {
        let mut node = self;
        for (&mask, &value) in mask.iter().zip(masked) {
            let arm = node.arms.iter_mut().find(|a| a.mask == mask)?;
            let index = arm.values.binary_search(&value).ok()?;
            node = &mut arm.children[index];
        }
        Some(node)
    }
}

/// The answer of [`LiftCache::find`].
pub(crate) enum Lookup {
    /// The shape's template, instantiated for the instruction.
    Hit(Arc<Template>, Instance),
    /// The shape was probed and cannot be cached, or the cache does not
    /// apply: lift it, and do not remember it.
    Uncacheable,
    /// The shape has not been seen: lift it through [`LiftCache::miss`].
    Unknown,
}

/// The tries of one leading byte, by key prefix.
type Shard = RwLock<HashMap<Box<[u8]>, Node>>;

/// A view of a register table: the `size` bytes at `delta` into each
/// register it binds.
type View = (FieldTableId, u64, usize);

/// A cache of lifted instructions keyed by the shape of their encoding. See the [module
/// documentation](self).
///
/// It is shared between the sessions of one lifter — across threads, behind
/// an [`Arc`] — through [`ScratchSession::with_cache`](crate::session::ScratchSession::with_cache)
/// and [`LiftSession::with_cache`](crate::session::LiftSession::with_cache).
pub struct LiftCache {
    fingerprint: SpecFingerprint,
    shards: Box<[Shard]>,
    /// Which bits of the leading byte pick the shard: the bits every
    /// entry's mask keeps, so an entry and every instruction of its shape
    /// land in one shard. It narrows as shapes with a freer leading byte
    /// arrive, which empties the cache; on a real specification that
    /// happens a few times at the start and never again.
    shard_mask: AtomicU8,
    /// Per view of an `attach variables` table — a slice of each register
    /// it binds — the varnodes, once resolved.
    tables: Mutex<HashMap<View, Arc<TableView>>>,
    capacity: usize,
    entries: AtomicUsize,
    validating: bool,
    hits: AtomicU64,
    misses: AtomicU64,
    probes: AtomicU64,
    uncacheable: AtomicU64,
    exact: AtomicU64,
    validation_failures: AtomicU64,
}

impl std::fmt::Debug for LiftCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiftCache")
            .field("capacity", &self.capacity)
            .field("validating", &self.validating)
            .field("stats", &self.stats())
            .finish()
    }
}

impl LiftCache {
    /// The default number of shapes a cache holds before it stops
    /// remembering new ones.
    pub const DEFAULT_CAPACITY: usize = 1 << 20;

    /// An empty cache for lifters of `spec`.
    pub fn new(spec: &CompiledSpec) -> Self {
        Self {
            fingerprint: spec.fingerprint(),
            shards: (0..SHARDS)
                .map(|_| RwLock::new(HashMap::default()))
                .collect(),
            shard_mask: AtomicU8::new(0xff),
            tables: Mutex::new(HashMap::default()),
            capacity: Self::DEFAULT_CAPACITY,
            entries: AtomicUsize::new(0),
            validating: false,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            probes: AtomicU64::new(0),
            uncacheable: AtomicU64::new(0),
            exact: AtomicU64::new(0),
            validation_failures: AtomicU64::new(0),
        }
    }

    /// Holds at most `entries` shapes; once full, further shapes are lifted
    /// for real and not remembered. Memory is bounded by the entries'
    /// templates, which are roughly the size of the IR they stand for.
    pub fn with_capacity(mut self, entries: usize) -> Self {
        self.capacity = entries;
        self
    }

    /// Lifts every hit for real as well and compares the two, discarding
    /// the replay and evicting the entry when they differ. For measuring
    /// the cache's exactness over a corpus; it costs every hit a full lift.
    pub fn validating(mut self, validating: bool) -> Self {
        self.validating = validating;
        self
    }

    pub fn is_validating(&self) -> bool {
        self.validating
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            probes: self.probes.load(Ordering::Relaxed),
            uncacheable: self.uncacheable.load(Ordering::Relaxed),
            exact: self.exact.load(Ordering::Relaxed),
            validation_failures: self.validation_failures.load(Ordering::Relaxed),
            entries: self.entries.load(Ordering::Relaxed),
        }
    }

    /// Forgets every shape; the counters stay.
    pub fn clear(&self) {
        for shard in &self.shards {
            shard.write().unwrap().clear();
        }
        self.entries.store(0, Ordering::Relaxed);
    }

    /// The shard of an instruction whose leading byte is `first`.
    fn shard(&self, first: u8) -> &Shard {
        &self.shards[usize::from(first & self.shard_mask.load(Ordering::Relaxed))]
    }

    /// Narrows the shard mask to `mask` — the bits a new entry's leading
    /// byte keeps — moving every subtree of a leading byte to the shard it
    /// now belongs in. A lookup racing this may miss; nothing worse.
    fn narrow_shards(&self, mask: u8) {
        let mut guards: Vec<_> = self.shards.iter().map(|s| s.write().unwrap()).collect();
        if self.shard_mask.load(Ordering::Relaxed) == mask {
            return;
        }
        // A root's arm holds, per masked leading byte, that byte's subtree.
        struct Move {
            to: usize,
            prefix: Box<[u8]>,
            mask: u8,
            value: u8,
            node: Node,
        }
        let mut moves: Vec<Move> = Vec::new();
        for (index, shard) in guards.iter_mut().enumerate() {
            for (prefix, root) in shard.iter_mut() {
                for arm in &mut root.arms {
                    let mut i = 0;
                    while i < arm.values.len() {
                        let to = usize::from(arm.values[i] & mask);
                        if to == index {
                            i += 1;
                            continue;
                        }
                        moves.push(Move {
                            to,
                            prefix: prefix.clone(),
                            mask: arm.mask,
                            value: arm.values.remove(i),
                            node: arm.children.remove(i),
                        });
                    }
                }
            }
        }
        for Move {
            to,
            prefix,
            mask,
            value,
            node,
        } in moves
        {
            let root = guards[to].entry(prefix).or_default();
            let arm = match root.arms.iter().position(|a| a.mask == mask) {
                Some(arm) => arm,
                None => {
                    root.arms.push(Arm {
                        mask,
                        ..Arm::default()
                    });
                    root.arms.len() - 1
                }
            };
            let arm = &mut root.arms[arm];
            let index = arm
                .values
                .binary_search(&value)
                .expect_err("a subtree lands where none was");
            arm.values.insert(index, value);
            arm.children.insert(index, node);
        }
        self.shard_mask.store(mask, Ordering::Relaxed);
    }

    fn check(&self, lifter: &SleighLifter<'_>) -> Result<(), LiftError> {
        if lifter.spec().fingerprint() != self.fingerprint {
            return Err(LiftError::IncompatibleSpec);
        }
        Ok(())
    }

    /// The entry of the shape the instruction at the front of `bytes` has,
    /// instantiated at `address`. See [`Node::find`] for `exact`.
    fn lookup(
        &self,
        prefix: &[u8],
        address: u64,
        bytes: &[u8],
        exact: Option<usize>,
    ) -> Option<Found> {
        let first = *bytes.first()?;
        let shard = self.shard(first).read().unwrap();
        let entry = shard.get(prefix)?.find(bytes, 0, exact)?;
        Some(match &entry.kind {
            Kind::Templates(templates) => {
                let (template, instance) = templates
                    .iter()
                    .find_map(|t| t.instance(address, bytes).map(|i| (t, i)))?;
                Found::Template(Arc::clone(template), instance)
            }
            Kind::Uncacheable => Found::Uncacheable,
        })
    }

    fn insert(&self, prefix: &[u8], mask: &[u8], masked: &[u8], entry: Entry) {
        if self.entries.load(Ordering::Relaxed) >= self.capacity {
            return;
        }
        let (Some(&first), Some(&kept)) = (masked.first(), mask.first()) else {
            return;
        };
        let narrowed = self.shard_mask.load(Ordering::Relaxed) & kept;
        if narrowed != self.shard_mask.load(Ordering::Relaxed) {
            self.narrow_shards(narrowed);
        }
        let depth = keyed(mask);
        let mut shard = self.shard(first).write().unwrap();
        let leaf = &mut shard
            .entry(Box::from(prefix))
            .or_default()
            .walk_mut(&mask[..depth], &masked[..depth])
            .leaf;
        match leaf {
            None => {
                *leaf = Some(entry);
                self.entries.fetch_add(1, Ordering::Relaxed);
            }
            // The shape is known: a new way its registers coincide joins
            // the entry, once.
            Some(Entry {
                kind: Kind::Templates(known),
                ..
            }) => {
                if let Kind::Templates(new) = entry.kind
                    && let Some(template) = new.into_iter().next()
                {
                    let dup = known
                        .iter()
                        .any(|k| k.ties == template.ties && k.varnodes == template.varnodes);
                    if known.len() < MAX_TEMPLATES && !dup {
                        known.push(template);
                    }
                }
            }
            Some(_) => {}
        }
    }

    fn evict(&self, prefix: &[u8], template: &Template) {
        let keys = &template.keys;
        let Some(&first) = keys.masked.first() else {
            return;
        };
        let mut shard = self.shard(first).write().unwrap();
        let depth = keyed(&keys.mask);
        if let Some(node) = shard
            .get_mut(prefix)
            .and_then(|root| root.get_mut(&keys.mask[..depth], &keys.masked[..depth]))
            && let Some(Entry {
                kind: Kind::Templates(templates),
                ..
            }) = &mut node.leaf
        {
            templates.retain(|t| !std::ptr::eq(&**t, template));
            if templates.is_empty() {
                node.leaf = None;
                self.entries.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    /// Lifts `instruction` into `target` through the cache: a replay when
    /// its shape is known, a real lift — probed and remembered — otherwise.
    /// `decoder` decoded the instruction and decodes its probes; `shape` is
    /// the instruction's, when that decode reported it.
    pub(crate) fn lower(
        &self,
        lifter: &SleighLifter<'_>,
        target: &mut LiftTarget<'_, 'static>,
        instruction: &Decoded<'_, '_>,
        shape: Option<Shape>,
        decoder: &FixedDecoder<'_>,
        flat: bool,
        scratch: &mut ReplayScratch,
    ) -> Result<Lifted, LiftError> {
        match self.find(lifter, instruction, decoder, flat)? {
            Lookup::Hit(template, instance) => template.replay(target, &instance, scratch),
            Lookup::Uncacheable => lifter.lower(target, instruction, flat),
            Lookup::Unknown => self.miss(lifter, target, instruction, shape, decoder, flat),
        }
    }

    /// Whether a lookup before decoding can answer: a validating cache
    /// decodes every instruction to check its hit, so it never does.
    pub(crate) fn answers_undecoded(&self) -> bool {
        !self.validating
    }

    /// What the cache holds for `instruction`. A hit is counted here, and
    /// validated when the cache validates; a hit that fails validation is
    /// evicted and reported uncacheable. Whatever the answer, the
    /// instruction is not lifted.
    pub(crate) fn find(
        &self,
        lifter: &SleighLifter<'_>,
        instruction: &Decoded<'_, '_>,
        decoder: &FixedDecoder<'_>,
        flat: bool,
    ) -> Result<Lookup, LiftError> {
        self.check(lifter)?;
        let mut buffer = [0u8; MAX_PREFIX];
        let Some(prefix) = prefix(&mut buffer, flat, decoder.context()) else {
            return Ok(Lookup::Uncacheable);
        };
        let bytes = instruction.bytes();
        let found = self.lookup(prefix, instruction.address(), bytes, Some(bytes.len()));
        Ok(match found {
            Some(Found::Template(template, instance)) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                if self.validating
                    && !self.validate(lifter, &template, &instance, instruction, decoder, flat)?
                {
                    self.validation_failures.fetch_add(1, Ordering::Relaxed);
                    self.evict(prefix, &template);
                    return Ok(Lookup::Uncacheable);
                }
                Lookup::Hit(template, instance)
            }
            Some(Found::Uncacheable) => {
                self.uncacheable.fetch_add(1, Ordering::Relaxed);
                Lookup::Uncacheable
            }
            None => Lookup::Unknown,
        })
    }

    /// [`find`](Self::find) before decoding: the template of the shape of
    /// the instruction at the front of `bytes`, instantiated at `address`,
    /// when the cache holds one. Valid encodings are prefix-free — a decoder
    /// reads only the bytes it needs, so no encoding is a proper prefix of
    /// another under one context — so the walk stops at the first shape the
    /// stream fits. A validating cache never answers, so that every hit is
    /// decoded and checked; nor does one whose entry says uncacheable, which
    /// the decoded path counts.
    pub(crate) fn find_undecoded(
        &self,
        lifter: &SleighLifter<'_>,
        decoder: &FixedDecoder<'_>,
        flat: bool,
        address: u64,
        bytes: &[u8],
    ) -> Result<Option<(Arc<Template>, Instance)>, LiftError> {
        self.check(lifter)?;
        if self.validating {
            return Ok(None);
        }
        let mut buffer = [0u8; MAX_PREFIX];
        let Some(prefix) = prefix(&mut buffer, flat, decoder.context()) else {
            return Ok(None);
        };
        Ok(match self.lookup(prefix, address, bytes, None) {
            Some(Found::Template(template, instance)) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some((template, instance))
            }
            _ => None,
        })
    }

    /// Lifts an instruction [`find`](Self::find) did not know into `target`,
    /// probes its shape, and remembers it. `shape` is the instruction's when
    /// its decode reported one; otherwise it is decoded again for it.
    pub(crate) fn miss(
        &self,
        lifter: &SleighLifter<'_>,
        target: &mut LiftTarget<'_, 'static>,
        instruction: &Decoded<'_, '_>,
        shape: Option<Shape>,
        decoder: &FixedDecoder<'_>,
        flat: bool,
    ) -> Result<Lifted, LiftError> {
        self.misses.fetch_add(1, Ordering::Relaxed);
        let marks = Marks::of(target);
        let lifted = lifter.lower(target, instruction, flat)?;
        let address = instruction.address();
        let bytes = instruction.bytes();
        // A decode reading past the instruction's own bytes — a delay slot —
        // has no shape the key can hold: it overruns, or from the bytes
        // alone it fails.
        let shape = match shape {
            Some(shape) => shape,
            None => match decoder.decode_shaped(address, bytes) {
                Ok((_, shape)) => shape,
                Err(_) => return Ok(lifted),
            },
        };
        if shape.overruns() || shape.len() != bytes.len() {
            return Ok(lifted);
        }
        let mut buffer = [0u8; MAX_PREFIX];
        let Some(prefix) = prefix(&mut buffer, flat, decoder.context()) else {
            return Ok(lifted);
        };
        let mut masked = vec![0u8; bytes.len()];
        shape.masked(bytes, &mut masked);
        let attempt = |params: Vec<ParamField>,
                       regs: Vec<RegParam>,
                       tables: Vec<RegField>,
                       mask: &[u8],
                       masked: &[u8]| {
            let keys = Keys {
                mask: mask.into(),
                masked: masked.into(),
                base_params: params.iter().map(|p| p.value(bytes)).collect(),
                params: params.into_boxed_slice(),
                base_regs: regs.iter().map(|r| r.value(bytes)).collect(),
                regs: regs.into_boxed_slice(),
            };
            let mut template =
                Template::capture(target.context(), target.addresses(), marks, &lifted, keys)?;
            template.reg_fields = tables;
            let registers = Registers {
                cache: self,
                spec: lifter.spec(),
                ctx: target.context(),
            };
            let probing = Probing {
                lifter,
                probes: &self.probes,
                decoder,
                registers: &registers,
                address,
                bytes,
                shape: &shape,
                flat,
            };
            with_probe_store(lifter, |store| template.measure(&probing, store))?;
            template.dedupe_literals();
            template.dedupe_varnodes();
            template.record_ties(&registers);
            template.bake_names(target.context());
            Ok(template)
        };
        let captured = parameters(&shape)
            .and_then(|params| {
                let (regs, tables) = register_params(&shape)?;
                attempt(params, regs, tables, shape.mask(), &masked)
            })
            .or_else(|refusal| match refusal {
                Refusal::Parameters(reason) => {
                    debug_uncacheable(instruction, "is remembered exactly", reason);
                    self.exact.fetch_add(1, Ordering::Relaxed);
                    attempt(
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        &vec![0xff; bytes.len()],
                        bytes,
                    )
                }
                other => Err(other),
            });
        let (mask, masked, kind) = match captured {
            Ok(template) => {
                let (mask, masked) = (template.keys.mask.clone(), template.keys.masked.clone());
                (mask, masked, Kind::Templates(vec![Arc::new(template)]))
            }
            Err(Refusal::Shape(reason) | Refusal::Parameters(reason)) => {
                debug_uncacheable(instruction, "is uncacheable", reason);
                (shape.mask().into(), masked.into(), Kind::Uncacheable)
            }
        };
        let entry = Entry {
            len: bytes.len(),
            exclusions: shape.exclusions().into(),
            kind,
        };
        self.insert(prefix, &mask, &masked, entry);
        Ok(lifted)
    }

    /// Whether a fresh lift of `instruction` agrees with the template at
    /// `instance`.
    fn validate(
        &self,
        lifter: &SleighLifter<'_>,
        template: &Template,
        instance: &Instance,
        instruction: &Decoded<'_, '_>,
        decoder: &FixedDecoder<'_>,
        flat: bool,
    ) -> Result<bool, LiftError> {
        let address = instruction.address();
        with_probe_store(lifter, |store| {
            let decoded = decoder.decode(address, instruction.bytes())?;
            store.reset();
            let mut target = store.target()?.without_debug_names();
            let marks = Marks::of(&target);
            let lifted = lifter.lower(&mut target, &decoded, flat)?;
            let fresh = Template::capture(
                target.context(),
                target.addresses(),
                marks,
                &lifted,
                Keys::default(),
            );
            let Ok(mut fresh) = fresh else {
                debug_uncacheable(
                    instruction,
                    "failed validation",
                    "the fresh lift has no template",
                );
                return Ok(false);
            };
            fresh.canonicalize();
            match template.instantiate(instance).disagreement(&fresh) {
                None => Ok(true),
                Some(reason) => {
                    debug_uncacheable(instruction, "failed validation", reason);
                    Ok(false)
                }
            }
        })
    }
}

/// The answer of [`LiftCache::lookup`].
enum Found {
    Template(Arc<Template>, Instance),
    Uncacheable,
}

/// Why a miss made no template.
enum Refusal {
    /// The shape cannot be cached; every instance would fail alike.
    Shape(&'static str),
    /// The shape's parameters cannot be carried from this instance; the
    /// exact encoding still can be.
    Parameters(&'static str),
}

/// One instruction of a shape: the address a template is replayed at and
/// how far its parameters are from the captured instance's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Instance {
    address: u64,
    /// The address less the template's base.
    delta: u64,
    /// Per parameter, the value less the template's.
    params: [i64; MAX_PARAMS],
    /// Per register parameter, the field's value: the index into its
    /// tables.
    regs: [u8; MAX_REGS],
}

/// A register parameter of a template: a field of the shape read as the
/// index of a register, by however many tables — the same bits name a
/// 32-bit register to one constructor and its 64-bit one to another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RegParam {
    bit: u32,
    width: u8,
}

impl RegParam {
    fn end(&self) -> usize {
        self.bit as usize + usize::from(self.width)
    }

    /// The field's value in `bytes`, an instruction of the shape.
    fn value(&self, bytes: &[u8]) -> u8 {
        (self.bit as usize..self.end())
            .enumerate()
            .fold(0u8, |value, (i, bit)| {
                value | ((bytes[bit / 8] >> (bit % 8)) & 1) << i
            })
    }
}

/// One view a register parameter's value picks: the register of `size`
/// bytes `delta` into the register that table `table` binds to the value —
/// the register itself, or a lane of it — as the lifter's varnodes, `None`
/// where nothing is bound or declared there.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RegTable {
    param: u8,
    table: FieldTableId,
    delta: u64,
    size: usize,
    view: Arc<TableView>,
}

/// The registers one view of a table binds, by the field's value.
#[derive(Debug, PartialEq, Eq)]
struct TableView {
    varnodes: Box<[Option<VarnodeId>]>,
    /// Each register's name as a load of it is named: lowercase. Resolved
    /// once here rather than lowercased at every replay.
    names: Box<[Option<Box<str>>]>,
}

/// Resolves registers while a template is measured: the specification's
/// geometry, the context's varnodes.
#[derive(Clone, Copy)]
struct Registers<'a> {
    cache: &'a LiftCache,
    spec: &'a CompiledSpec,
    ctx: &'a Context<'static>,
}

impl Registers<'_> {
    /// Where a varnode of the context is.
    fn geometry(&self, varnode: VarnodeId) -> sleigh::Varnode {
        let varnode = Varnode::from_id(self.ctx, varnode);
        sleigh::Varnode::new(varnode.space().id, varnode.address() as u64, varnode.size())
    }

    /// The register table `table` binds to `value`.
    fn register(&self, table: FieldTableId, value: u8) -> Option<sleigh::RegisterId> {
        self.spec
            .attached_registers(table)
            .get(usize::from(value))
            .copied()
            .flatten()
    }

    /// The varnodes of the `size` bytes `delta` into each register of
    /// `table`, by value, with their names; resolved once per view.
    fn table(&self, table: FieldTableId, delta: u64, size: usize) -> Arc<TableView> {
        let mut tables = self.cache.tables.lock().unwrap();
        Arc::clone(tables.entry((table, delta, size)).or_insert_with(|| {
            let varnodes: Box<[Option<VarnodeId>]> = self
                .spec
                .attached_registers(table)
                .iter()
                .map(|register| {
                    let whole = self.spec.register_varnode((*register)?)?;
                    let part = sleigh::Varnode::new(whole.space, whole.offset + delta, size);
                    let register = self.spec.register_at(part)?;
                    self.ctx.shared.registers.get(&register).copied()
                })
                .collect();
            let names = varnodes
                .iter()
                .map(|varnode| {
                    Varnode::from_id(self.ctx, (*varnode)?)
                        .name()
                        .map(|name| name.to_lowercase().into_boxed_str())
                })
                .collect();
            Arc::new(TableView { varnodes, names })
        }))
    }
}

/// A varnode an operation names: the architecture's, or the register a
/// parameter's value picks from a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VarnodeSlot {
    Fixed(VarnodeId),
    /// An index into the template's tables.
    Register(u32),
}

/// An operation's debug name: as recorded, or that of the register in one
/// of its varnode slots, which is how the emitter names a register's load.
#[derive(Debug, Clone, PartialEq, Eq)]
enum OpName {
    Fixed(Box<str>),
    Varnode(u32),
}

/// A constant of the template: its value at the captured instance, plus
/// the address delta when it moves with the address, plus the delta of
/// every parameter it moves with, in `size` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Affine {
    value: u64,
    size: usize,
    relative: bool,
    /// A bit per parameter.
    params: u8,
}

impl Affine {
    fn fixed(value: u64, size: usize) -> Self {
        Self {
            value,
            size,
            relative: false,
            params: 0,
        }
    }

    fn mask(size: usize) -> u64 {
        if size >= 8 {
            u64::MAX
        } else {
            (1u64 << (8 * size)) - 1
        }
    }

    fn at(self, instance: &Instance) -> u64 {
        let mut value = self.value;
        if self.relative {
            value = value.wrapping_add(instance.delta);
        }
        let mut params = self.params;
        while params != 0 {
            let index = params.trailing_zeros() as usize;
            value = value.wrapping_add(instance.params[index] as u64);
            params &= params - 1;
        }
        value & Self::mask(self.size)
    }

    fn resolved(self, instance: &Instance) -> Self {
        Self::fixed(self.at(instance), self.size)
    }
}

/// The bits of `param` the shape's mask leaves free, lowest first.
fn free_bits<'a>(param: &ParamField, mask: &'a [u8]) -> impl Iterator<Item = usize> + 'a {
    let start = param.bit as usize;
    (start..start + usize::from(param.width)).filter(move |bit| {
        mask.get(bit / 8)
            .is_none_or(|byte| byte & (1 << (bit % 8)) == 0)
    })
}

/// The parameters of `shape` a template carries: each field with a bit the
/// mask leaves free, once. Two such fields sharing bits would be perturbed
/// together, and are refused.
fn parameters(shape: &Shape) -> Result<Vec<ParamField>, Refusal> {
    let mut params: Vec<ParamField> = Vec::new();
    let end = |p: &ParamField| p.bit as usize + usize::from(p.width);
    for &param in shape.params() {
        if free_bits(&param, shape.mask()).next().is_none() || params.contains(&param) {
            continue;
        }
        if params
            .iter()
            .any(|known| (param.bit as usize) < end(known) && (known.bit as usize) < end(&param))
        {
            return Err(Refusal::Parameters("two parameter fields share bits"));
        }
        params.push(param);
    }
    if params.len() > MAX_PARAMS {
        return Err(Refusal::Parameters(
            "more parameters than a template carries",
        ));
    }
    Ok(params)
}

/// The bits of a run the shape's mask leaves free, lowest first.
fn free_bits_of(bit: u32, width: u8, mask: &[u8]) -> impl Iterator<Item = usize> + '_ {
    let start = bit as usize;
    (start..start + usize::from(width)).filter(move |bit| {
        mask.get(bit / 8)
            .is_none_or(|byte| byte & (1 << (bit % 8)) == 0)
    })
}

/// The register parameters of `shape` a template carries — each run of
/// bits with one the mask leaves free, whichever tables it indexes — and,
/// per table a parameter indexes, which parameter. A field sharing bits
/// with another field, register or integer, would be perturbed with it,
/// and is refused.
fn register_params(shape: &Shape) -> Result<(Vec<RegParam>, Vec<RegField>), Refusal> {
    let mut params: Vec<RegParam> = Vec::new();
    let mut tables: Vec<RegField> = Vec::new();
    let overlaps =
        |a_bit: usize, a_end: usize, b_bit: usize, b_end: usize| a_bit < b_end && b_bit < a_end;
    for field in shape.registers() {
        let param = RegParam {
            bit: field.bit,
            width: field.width,
        };
        if free_bits_of(param.bit, param.width, shape.mask())
            .next()
            .is_none()
        {
            continue;
        }
        let index = match params.iter().position(|known| *known == param) {
            Some(index) => index,
            None => {
                if params.iter().any(|known| {
                    overlaps(
                        param.bit as usize,
                        param.end(),
                        known.bit as usize,
                        known.end(),
                    )
                }) || shape.params().iter().any(|int| {
                    overlaps(
                        param.bit as usize,
                        param.end(),
                        int.bit as usize,
                        int.bit as usize + usize::from(int.width),
                    )
                }) {
                    return Err(Refusal::Parameters(
                        "a register field shares bits with another field",
                    ));
                }
                params.push(param);
                params.len() - 1
            }
        };
        if !tables.contains(&(index as u8, field.table)) {
            tables.push((index as u8, field.table));
        }
    }
    if params.len() > MAX_REGS {
        return Err(Refusal::Parameters(
            "more register parameters than a template carries",
        ));
    }
    Ok((params, tables))
}

/// A table a register parameter indexes: the parameter, and the table.
type RegField = (u8, FieldTableId);

/// The lookup key's prefix: the lowering flag and the decode context.
/// `None` when it does not fit, which no real specification causes.
fn prefix<'k>(
    buffer: &'k mut [u8; MAX_PREFIX],
    flat: bool,
    context: &ContextBytes,
) -> Option<&'k [u8]> {
    let context = context.as_bytes();
    let len = 1 + context.len();
    if len > MAX_PREFIX {
        return None;
    }
    buffer[0] = flat as u8;
    buffer[1..len].copy_from_slice(context);
    Some(&buffer[..len])
}

/// `QCODE_CACHE_DEBUG=1`: reports every miss that made no template and why.
fn debug_uncacheable(instruction: &Decoded<'_, '_>, what: &str, reason: &str) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ENABLED.get_or_init(|| std::env::var_os("QCODE_CACHE_DEBUG").is_some()) {
        eprintln!(
            "lift cache: {} ({}) {what}: {reason}",
            instruction,
            instruction
                .bytes()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        );
    }
}

thread_local! {
    /// One private scratch store per specification per thread, for probe
    /// and validation lifts.
    static PROBE_STORES: RefCell<Vec<(SpecFingerprint, ScratchStore)>> = const { RefCell::new(Vec::new()) };
}

fn with_probe_store<R>(lifter: &SleighLifter<'_>, f: impl FnOnce(&mut ScratchStore) -> R) -> R {
    PROBE_STORES.with(|stores| {
        let mut stores = stores.borrow_mut();
        let fingerprint = lifter.spec().fingerprint();
        let index = match stores.iter().position(|(fp, _)| *fp == fingerprint) {
            Some(index) => index,
            None => {
                stores.push((fingerprint, ScratchStore::new(lifter.new_context())));
                stores.len() - 1
            }
        };
        f(&mut stores[index].1)
    })
}

/// A type an instruction's value can have, independent of any context's
/// type ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TypeKey {
    Int(usize),
    Bool,
    SpaceAddress(usize, MemorySpaceId),
}

impl TypeKey {
    fn of(ctx: &Context<'_>, id: TypeId) -> Option<Self> {
        Some(match ctx.shared.types.get(id).repr() {
            TypeRepr::Int { size } => Self::Int(size),
            TypeRepr::Bool => Self::Bool,
            TypeRepr::SpaceAddress { size, space } => Self::SpaceAddress(size, space),
            _ => return None,
        })
    }

    fn resolve(self, ctx: &Context<'_>) -> TypeId {
        let types = &ctx.shared.types;
        match self {
            Self::Int(size) => types.get_or_make_int(size),
            Self::Bool => types.get_or_make_bool(),
            Self::SpaceAddress(size, space) => types.get_or_make_space_address(size, space),
        }
    }
}

/// An interned constant of the template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Literal {
    affine: Affine,
    /// The literal's type: a comparison's `false` is a `bool`, not an `i8`.
    ty: TypeKey,
}

impl Literal {
    /// Whether the constant is the same at every instance.
    fn is_fixed(&self) -> bool {
        !self.affine.relative && self.affine.params == 0
    }

    /// The literal at `instance`, interned in `ctx`.
    fn resolve(&self, ctx: &Context<'static>, instance: &Instance) -> LocalValueId {
        LocalValueId::Literal(ctx.shared.values.get_or_make_typed_literal(
            self.affine.at(instance),
            self.ty.resolve(ctx),
            self.affine.size,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TempSpaceKey {
    word_size: usize,
    addr_size: usize,
}

/// A temporary of the template, in one of the template's own spaces.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TempKey {
    space: u32,
    address: i64,
    size: usize,
}

/// How a block of the template was named by the emitter, so a replay names
/// it for its own address.
#[derive(Debug, Clone, PartialEq, Eq)]
enum BlockName {
    None,
    Label(u32),
    Fallthrough(u32),
    Other(Box<str>),
}

impl BlockName {
    fn parse(name: Option<&str>, address: u64) -> Self {
        let Some(name) = name else {
            return Self::None;
        };
        let label = format!("pcode_{address:x}_");
        let fallthrough = format!("pcode_fallthrough_{address:x}_");
        if let Some(index) = name.strip_prefix(&fallthrough).and_then(|i| i.parse().ok()) {
            return Self::Fallthrough(index);
        }
        if let Some(index) = name.strip_prefix(&label).and_then(|i| i.parse().ok()) {
            return Self::Label(index);
        }
        Self::Other(name.into())
    }

    fn render(&self, address: u64) -> Option<String> {
        match self {
            Self::None => None,
            Self::Label(index) => Some(format!("pcode_{address:x}_{index}")),
            Self::Fallthrough(index) => Some(format!("pcode_fallthrough_{address:x}_{index}")),
            Self::Other(name) => Some(name.to_string()),
        }
    }
}

/// The name an emitter gave a value before its body made it unique. The
/// emitter names a load after its register; a body makes a taken name free
/// by suffixing `_<n>`. Replaying the base lets the target body number it
/// as a fresh lift there would.
fn base_name(ctx: &Context<'static>, name: &str) -> Box<str> {
    if let Some((base, digits)) = name.rsplit_once('_')
        && !digits.is_empty()
        && digits.bytes().all(|b| b.is_ascii_digit())
        && (ctx.get_named(base).is_some() || ctx.get_named(&base.to_uppercase()).is_some())
    {
        return base.into();
    }
    name.into()
}

/// One operation of the template. Its mnemonic's operands are template
/// indices dressed as [`LocalValueId`]s: `Literal(k)` is the template's
/// `k`th literal, `Instruction(k)` its `k`th operation, `Temp(k)` its `k`th
/// temporary, `BasicBlock(k)` its `k`th block — the instruction's own blocks
/// first, then the external ones — and a call's `Callee::Minted(k)` its
/// `k`th callee, and `Varnode(k)` its `k`th [varnode slot](VarnodeSlot).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Op {
    block: u32,
    ty: u32,
    mnemonic: Mnemonic,
    name: Option<OpName>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExitTarget {
    Fallthrough,
    Branch(Affine),
    BranchInd,
    Call {
        callee: CallKey,
        continuation: Option<u32>,
    },
    CallInd {
        continuation: Option<u32>,
    },
    Return,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CallKey {
    Address(Affine),
    Named(Box<str>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExitRecord {
    site: u32,
    arm: ExitArm,
    target: ExitTarget,
}

/// What locates a template in the cache and instantiates it: the shape's
/// mask and masked bytes, and its parameters with their values at the
/// captured instance.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Keys {
    mask: Box<[u8]>,
    masked: Box<[u8]>,
    params: Box<[ParamField]>,
    base_params: Box<[i64]>,
    regs: Box<[RegParam]>,
    base_regs: Box<[u8]>,
}

/// The lowered IR of one shape, relative to the instance it was captured
/// from. See the [module documentation](self).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Template {
    /// The address the values were captured at.
    base: u64,
    length: usize,
    keys: Keys,
    /// The instruction's own blocks after the entry, with how they were
    /// named.
    blocks: Vec<BlockName>,
    /// Addresses of the blocks of other instructions the IR names, in
    /// address order at the captured instance, plus the fall-through, which
    /// the emitter resolves whether or not the IR names it.
    externals: Vec<Affine>,
    /// The order the emitter resolves the externals in, as indices into
    /// `externals` — the order their placeholders are made in when none
    /// exists yet, which a replay keeps so it issues the same block ids.
    external_order: Vec<u32>,
    callees: Vec<Affine>,
    /// The temporary spaces the lift made, in order.
    temp_spaces: Vec<TempSpaceKey>,
    /// The temporaries the lift made, in order: every one, since a body
    /// numbers them by creation and a replay must number them alike, not
    /// only those an operation names.
    temps: Vec<TempKey>,
    literals: Vec<Literal>,
    types: Vec<TypeKey>,
    /// Per table a register parameter indexes, the parameter: the views a
    /// probe may find a varnode following.
    reg_fields: Vec<RegField>,
    /// The views the register slots follow.
    reg_tables: Vec<RegTable>,
    /// Per pair of registers the lift names — the whole register a field
    /// binds, through each of its tables, or one a varnode slot names
    /// outright — whether they were one register at the captured
    /// instance: a lift's structure can turn on it (a move between one
    /// register is nothing), so the template serves only instances that
    /// agree.
    ties: Vec<(VarnodeSlot, VarnodeSlot, bool)>,
    /// The varnodes the operations name, one slot per use until
    /// [`dedupe_varnodes`](Self::dedupe_varnodes).
    varnodes: Vec<VarnodeSlot>,
    ops: Vec<Op>,
    exits: Vec<ExitRecord>,
}

/// What a template's context-independent keys resolve to in one session's
/// context: its types, and its constants that are the same at every
/// instance, which are most of them. Interned values never move, so the
/// ids hold for the context's life.
#[derive(Debug)]
struct Resolved {
    /// Keeps the template alive, so its address is not reused for another.
    _template: Arc<Template>,
    types: Box<[TypeId]>,
    /// Per literal, its id when the literal is fixed.
    literals: Box<[Option<LocalValueId>]>,
}

/// Where a lift's additions to its body start.
#[derive(Debug, Clone, Copy)]
struct Marks {
    temp_spaces: usize,
    temps: usize,
}

impl Marks {
    fn of(target: &LiftTarget<'_, 'static>) -> Self {
        let body = target.context().body(target.function());
        Self {
            temp_spaces: body.temp_space_count(),
            temps: body.temp_count(),
        }
    }
}

/// Assigns template indices to the ids of one lift, in first-seen order.
#[derive(Default)]
struct Numbering {
    /// The instruction's operations by id, ascending; an operation's
    /// position is its index.
    insns: Vec<LocalInsnId>,
    /// The first temporary the lift made; those after it are numbered from
    /// there.
    temps_from: usize,
    temps: usize,
    blocks: Vec<(LocalBlockId, u32)>,
    types: Vec<TypeId>,
}

impl Numbering {
    fn insn(&self, id: LocalInsnId) -> Option<u32> {
        self.insns.binary_search(&id).ok().map(|i| i as u32)
    }

    fn temp(&self, id: LocalTempId) -> Option<u32> {
        let raw = usize::from(id);
        (self.temps_from..self.temps_from + self.temps)
            .contains(&raw)
            .then(|| (raw - self.temps_from) as u32)
    }

    fn block(&self, id: LocalBlockId) -> Option<u32> {
        self.blocks.iter().find(|(b, _)| *b == id).map(|(_, i)| *i)
    }
}

impl Template {
    /// Records the IR `lifted` describes in `ctx`, or why a template cannot
    /// hold it.
    fn capture(
        ctx: &Context<'static>,
        addresses: &AddressIndex,
        marks: Marks,
        lifted: &Lifted,
        keys: Keys,
    ) -> Result<Self, Refusal> {
        let base = lifted.address();
        let next = base.wrapping_add(lifted.length() as u64);
        let func = lifted.entry().func;
        let body = ctx.body(func);
        let view = ModuleView::new(ctx);
        let mut numbering = Numbering {
            insns: Vec::new(),
            temps_from: marks.temps,
            temps: 0,
            blocks: Vec::new(),
            types: Vec::new(),
        };
        let mut template = Self {
            base,
            length: lifted.length(),
            keys,
            blocks: Vec::new(),
            externals: Vec::new(),
            external_order: Vec::new(),
            callees: Vec::new(),
            temp_spaces: Vec::new(),
            temps: Vec::new(),
            literals: Vec::new(),
            types: Vec::new(),
            reg_fields: Vec::new(),
            reg_tables: Vec::new(),
            ties: Vec::new(),
            varnodes: Vec::new(),
            ops: Vec::new(),
            exits: Vec::new(),
        };

        // The instruction's blocks, in the order they were opened; the entry
        // is block 0 and needs no name of its own.
        for (index, &block) in lifted.blocks().iter().enumerate() {
            numbering.blocks.push((block.local, index as u32));
            if index > 0 {
                let name = BasicBlock::from_id(ctx, block).name();
                template.blocks.push(BlockName::parse(name, base));
            }
        }

        // The operations, in the order they were issued — which is the order
        // the emitter made them, so every operand names an earlier one.
        let mut order: Vec<(LocalInsnId, u32)> = Vec::new();
        for (index, &block) in lifted.blocks().iter().enumerate() {
            for insn in body.insn_ids(block.local) {
                order.push((insn, index as u32));
            }
        }
        order.sort_unstable_by_key(|(insn, _)| usize::from(*insn));
        numbering.insns = order.iter().map(|(insn, _)| *insn).collect();

        // The temporary spaces and temporaries the lift appended, in order.
        for space in marks.temp_spaces..body.temp_space_count() {
            let space = body.temp_space(TempSpaceId::new(func, LocalTempSpaceId::from(space)));
            if space.name().is_some() {
                return Err(Refusal::Shape("a named temporary space"));
            }
            template.temp_spaces.push(TempSpaceKey {
                word_size: space.word_size(),
                addr_size: space.addr_size(),
            });
        }
        for temp in marks.temps..body.temp_count() {
            let local = LocalTempId::from(temp);
            numbering.temps += 1;
            let temp = TempRef::new(view, TempId::new(func, local));
            if temp.name().is_some() || temp.label().is_some() {
                return Err(Refusal::Shape("a named or labeled temporary"));
            }
            let space = usize::from(temp.space().id.local);
            if space < marks.temp_spaces {
                return Err(Refusal::Shape(
                    "a temporary in an earlier instruction's space",
                ));
            }
            template.temps.push(TempKey {
                space: (space - marks.temp_spaces) as u32,
                address: temp.address(),
                size: temp.size(),
            });
        }

        // The blocks of other instructions, in the order the operations
        // first name them, then the fall-through: the order every lift of
        // the shape lists them in, so probes pair up slot by slot.
        let mut external_blocks: Vec<LocalBlockId> = Vec::new();
        let mut external = |block: LocalBlockId| {
            if numbering.block(block).is_none() && !external_blocks.contains(&block) {
                external_blocks.push(block);
            }
        };
        for (insn, _) in &order {
            let mnemonic = ctx.instruction(InstructionId::new(func, *insn)).mnemonic();
            match mnemonic {
                Mnemonic::Branch(branch) => external(branch.target),
                Mnemonic::CBranch(cbranch) => {
                    external(cbranch.success_block);
                    external(cbranch.failure_block);
                }
                _ => {}
            }
        }
        // The fall-through placeholder is resolved by every lift, named by
        // the IR or not.
        let next_block = addresses
            .block_at(next)
            .ok_or(Refusal::Shape("the fall-through has no block"))?;
        if next_block.func != func {
            return Err(Refusal::Shape(
                "the fall-through belongs to another function",
            ));
        }
        external(next_block.local);
        let internal = lifted.blocks().len() as u32;
        for (index, &block) in external_blocks.iter().enumerate() {
            let address = ctx
                .block(BlockId::new(func, block))
                .address()
                .ok_or(Refusal::Shape(
                    "a block of another instruction has no address",
                ))?;
            template.externals.push(Affine::fixed(address, 8));
            numbering.blocks.push((block, internal + index as u32));
        }
        let mut creation: Vec<(LocalBlockId, u32)> = external_blocks
            .iter()
            .enumerate()
            .map(|(index, &block)| (block, index as u32))
            .collect();
        creation.sort_unstable_by_key(|(block, _)| usize::from(*block));
        template.external_order = creation.iter().map(|(_, index)| *index).collect();

        for (position, (insn, block)) in order.into_iter().enumerate() {
            let position = position as u32;
            let id = InstructionId::new(func, insn);
            let reference = Instruction::from_id(ctx, id);
            let mut failed = None;
            let first_slot = template.varnodes.len();
            let mut map = |operand: LocalValueId| -> LocalValueId {
                match operand {
                    // One slot per use: two uses of one value may move
                    // differently, and only a probe can tell.
                    LocalValueId::Literal(literal) => {
                        let value = ctx.get_literal_value(literal);
                        let type_id = ctx.shared.values.literals[literal].type_id;
                        let Some(ty) = TypeKey::of(ctx, type_id) else {
                            failed = Some(Refusal::Shape(
                                "a constant of a type a template cannot hold",
                            ));
                            return operand;
                        };
                        let size = ctx.shared.types.size_of(type_id);
                        template.literals.push(Literal {
                            affine: Affine::fixed(value, size),
                            ty,
                        });
                        LocalValueId::Literal(LiteralId::from(template.literals.len() - 1))
                    }
                    LocalValueId::Instruction(other) => match numbering.insn(other) {
                        // A use may only name an operation issued before it.
                        Some(index) if index < position => {
                            LocalValueId::Instruction(LocalInsnId::from(index as usize))
                        }
                        _ => {
                            failed = Some(Refusal::Shape("an operand names a later operation"));
                            operand
                        }
                    },
                    LocalValueId::Temp(temp) => match numbering.temp(temp) {
                        Some(index) => LocalValueId::Temp(LocalTempId::from(index as usize)),
                        None => {
                            failed = Some(Refusal::Shape(
                                "an operand names an earlier instruction's temporary",
                            ));
                            operand
                        }
                    },
                    // One slot per use, like a constant: a probe tells
                    // which register field each follows.
                    LocalValueId::Varnode(varnode) => {
                        template.varnodes.push(VarnodeSlot::Fixed(varnode));
                        LocalValueId::Varnode(VarnodeId::from(template.varnodes.len() - 1))
                    }
                    _ => {
                        failed = Some(Refusal::Shape(
                            "an operand of a kind a template cannot hold",
                        ));
                        operand
                    }
                }
            };
            let mut mnemonic = reference.mnemonic().clone().map_operands(&mut map);
            if let Some(reason) = failed {
                return Err(reason);
            }
            // A load is named after its register; the name then follows
            // the slot, not the record.
            let name = reference.name().map(|name| {
                let base = base_name(ctx, name);
                let slot = (first_slot..template.varnodes.len()).find(|&slot| {
                    let VarnodeSlot::Fixed(varnode) = template.varnodes[slot] else {
                        return false;
                    };
                    Varnode::from_id(ctx, varnode)
                        .name()
                        .is_some_and(|register| register.to_lowercase() == *base)
                });
                match slot {
                    Some(slot) => OpName::Varnode(slot as u32),
                    None => OpName::Fixed(base),
                }
            });
            let space_index = |space: &mut LocalMemorySpaceId| -> Result<(), Refusal> {
                if let LocalMemorySpaceId::Temp(local) = space {
                    let raw = usize::from(*local);
                    if raw < marks.temp_spaces {
                        return Err(Refusal::Shape(
                            "an operation in an earlier instruction's space",
                        ));
                    }
                    *local = LocalTempSpaceId::from(raw - marks.temp_spaces);
                }
                Ok(())
            };
            match &mut mnemonic {
                Mnemonic::Load(load) => space_index(&mut load.space)?,
                Mnemonic::Store(store) => space_index(&mut store.space)?,
                _ => {}
            }
            // Block targets are not operands; a call's callee is not either.
            let block_index = |target: LocalBlockId| {
                LocalBlockId::from(
                    numbering.block(target).expect("a target block of the lift") as usize
                )
            };
            match &mut mnemonic {
                Mnemonic::Branch(branch) => branch.target = block_index(branch.target),
                Mnemonic::CBranch(cbranch) => {
                    cbranch.success_block = block_index(cbranch.success_block);
                    cbranch.failure_block = block_index(cbranch.failure_block);
                }
                Mnemonic::Switch(_) => return Err(Refusal::Shape("a switch")),
                Mnemonic::Call(call) => call.target = template.callee_key(ctx, call.target)?,
                Mnemonic::TailCall(call) => call.target = template.callee_key(ctx, call.target)?,
                Mnemonic::Apply(_) => return Err(Refusal::Shape("an apply")),
                _ => {}
            }
            let type_id = reference.type_id();
            let next = numbering.types.len() as u32;
            let ty = match numbering.types.iter().position(|&t| t == type_id) {
                Some(index) => index as u32,
                None => {
                    template.types.push(
                        TypeKey::of(ctx, type_id)
                            .ok_or(Refusal::Shape("a type a template cannot hold"))?,
                    );
                    numbering.types.push(type_id);
                    next
                }
            };
            template.ops.push(Op {
                block,
                ty,
                mnemonic,
                name,
            });
        }

        for exit in lifted.exits() {
            let site = numbering
                .insn(exit.site().local)
                .ok_or(Refusal::Shape("an exit site outside the instruction"))?;
            let continuation = |continuation: Continuation| -> Result<Option<u32>, Refusal> {
                match continuation {
                    Continuation::Next => Ok(None),
                    Continuation::Block(block) => numbering
                        .block(block.local)
                        .map(Some)
                        .ok_or(Refusal::Shape("a continuation outside the instruction")),
                }
            };
            let target = match exit.kind() {
                ExitKind::Fallthrough => ExitTarget::Fallthrough,
                ExitKind::Branch { target } => ExitTarget::Branch(Affine::fixed(*target, 8)),
                ExitKind::BranchInd => ExitTarget::BranchInd,
                ExitKind::Call {
                    callee,
                    continuation: c,
                } => ExitTarget::Call {
                    callee: match callee {
                        CallTarget::Address(address) => {
                            CallKey::Address(Affine::fixed(*address, 8))
                        }
                        CallTarget::Named(name) => CallKey::Named(name.clone()),
                    },
                    continuation: continuation(*c)?,
                },
                ExitKind::CallInd { continuation: c } => ExitTarget::CallInd {
                    continuation: continuation(*c)?,
                },
                ExitKind::Return => ExitTarget::Return,
            };
            template.exits.push(ExitRecord {
                site,
                arm: exit.arm(),
                target,
            });
        }
        Ok(template)
    }

    /// The instance of this template that the instruction at `address`,
    /// starting `bytes`, is; `None` when a register its fields pick has no
    /// varnode, which a fresh lift refuses too.
    fn instance(&self, address: u64, bytes: &[u8]) -> Option<Instance> {
        let mut params = [0i64; MAX_PARAMS];
        for ((slot, param), base) in params
            .iter_mut()
            .zip(&self.keys.params)
            .zip(&self.keys.base_params)
        {
            *slot = param.value(bytes).wrapping_sub(*base);
        }
        let mut regs = [0u8; MAX_REGS];
        for (slot, param) in regs.iter_mut().zip(&self.keys.regs) {
            *slot = param.value(bytes);
        }
        if self
            .varnodes
            .iter()
            .any(|&slot| self.varnode_with(slot, &regs).is_none())
        {
            return None;
        }
        // A register the lifter has no varnode for coincides with nothing
        // it lifts; the slots above refuse the instance if it matters.
        if self.ties.iter().any(|&(a, b, tied)| {
            match (self.varnode_with(a, &regs), self.varnode_with(b, &regs)) {
                (Some(a), Some(b)) => (a == b) != tied,
                _ => false,
            }
        }) {
            return None;
        }
        Some(Instance {
            address,
            delta: address.wrapping_sub(self.base),
            params,
            regs,
        })
    }

    /// The varnode of `slot` when the register parameters have the values
    /// `regs`.
    fn varnode_with(&self, slot: VarnodeSlot, regs: &[u8]) -> Option<VarnodeId> {
        match slot {
            VarnodeSlot::Fixed(varnode) => Some(varnode),
            VarnodeSlot::Register(table) => {
                let table = &self.reg_tables[table as usize];
                table
                    .view
                    .varnodes
                    .get(usize::from(regs[usize::from(table.param)]))
                    .copied()
                    .flatten()
            }
        }
    }

    /// The varnode of the template's `k`th slot at `instance`.
    pub(crate) fn varnode_at(&self, k: usize, instance: &Instance) -> VarnodeId {
        self.varnode_with(self.varnodes[k], &instance.regs)
            .expect("an instance resolves every register")
    }

    /// Whether every varnode slot of `probed` — the same instruction
    /// lifted with the register parameters at `regs` — is what this
    /// template says.
    fn varnodes_agree(&self, probed: &Self, regs: &[u8]) -> bool {
        self.varnodes.len() == probed.varnodes.len()
            && self
                .varnodes
                .iter()
                .zip(&probed.varnodes)
                .all(|(mine, theirs)| {
                    let VarnodeSlot::Fixed(theirs) = *theirs else {
                        return false;
                    };
                    self.varnode_with(*mine, regs) == Some(theirs)
                })
    }

    /// Marks every fixed varnode slot that `probed` — the same instruction
    /// lifted with the register parameters at `regs` instead of the base
    /// values — shows changed as following the one table of a perturbed
    /// parameter that maps the base value to the slot's varnode and the
    /// new value to the probe's. A changed slot no table explains, or that
    /// two explain alike, is a mismatch.
    fn classify_varnodes(
        &mut self,
        probed: &Self,
        regs: &[u8],
        registers: &Registers<'_>,
    ) -> Result<(), Mismatch> {
        if self.varnodes.len() != probed.varnodes.len() {
            return Err(Mismatch::Structure(
                "the two lifts differ in their varnodes",
            ));
        }
        for k in 0..self.varnodes.len() {
            let VarnodeSlot::Fixed(theirs) = probed.varnodes[k] else {
                return Err(Mismatch::Structure("a probe was classified"));
            };
            let mine = self.varnodes[k];
            if self.varnode_with(mine, regs) == Some(theirs) {
                continue;
            }
            let VarnodeSlot::Fixed(was) = mine else {
                return Err(Mismatch::Value(
                    "a register operand does not follow its field",
                ));
            };
            let index = self.view_of(was, theirs, regs, registers)?;
            let table = &self.reg_tables[index];
            let at = |value: u8| {
                table
                    .view
                    .varnodes
                    .get(usize::from(value))
                    .copied()
                    .flatten()
            };
            if at(self.keys.base_regs[usize::from(table.param)]) != Some(was)
                || at(regs[usize::from(table.param)]) != Some(theirs)
            {
                return Err(Mismatch::Value(
                    "a register operand does not follow its view",
                ));
            }
            self.varnodes[k] = VarnodeSlot::Register(index as u32);
        }
        Ok(())
    }

    /// The view — made if new — of a perturbed register field under which
    /// a slot that named `was` at the base values names `theirs` at `regs`:
    /// `was` lies at some offset in the register the field bound before,
    /// and `theirs` at the same offset in the one it binds now.
    fn view_of(
        &mut self,
        was: VarnodeId,
        theirs: VarnodeId,
        regs: &[u8],
        registers: &Registers<'_>,
    ) -> Result<usize, Mismatch> {
        let (g_was, g_theirs) = (registers.geometry(was), registers.geometry(theirs));
        if g_was.size != g_theirs.size || g_was.space != g_theirs.space {
            return Err(Mismatch::Value(
                "a register operand does not follow a register field",
            ));
        }
        let size = g_was.size;
        let mut found: Option<(u8, FieldTableId, u64)> = None;
        for &(param, table) in &self.reg_fields {
            let p = usize::from(param);
            let base = self.keys.base_regs[p];
            if regs[p] == base {
                continue;
            }
            let whole = |value: u8| {
                registers
                    .register(table, value)
                    .and_then(|r| registers.spec.register_varnode(r))
            };
            let (Some(before), Some(after)) = (whole(base), whole(regs[p])) else {
                continue;
            };
            if before.space != g_was.space
                || g_was.offset < before.offset
                || g_was.offset + size as u64 > before.offset + before.size as u64
            {
                continue;
            }
            let delta = g_was.offset - before.offset;
            if after.space != g_theirs.space
                || after.offset + delta != g_theirs.offset
                || (delta as usize) + size > after.size
            {
                continue;
            }
            // Two fields bound to one register at the base values explain
            // a slot alike, and the template serves only instances where
            // they still are; two bound to different registers do not.
            match found {
                Some((known, _, _))
                    if !self.tied_fields(usize::from(known), usize::from(param), registers) =>
                {
                    return Err(Mismatch::Ambiguous);
                }
                Some(_) => {}
                None => found = Some((param, table, delta)),
            }
        }
        let Some((param, table, delta)) = found else {
            return Err(Mismatch::Value(
                "a register operand does not follow a register field",
            ));
        };
        let known = self.reg_tables.iter().position(|t| {
            t.param == param && t.table == table && t.delta == delta && t.size == size
        });
        Ok(known.unwrap_or_else(|| {
            self.reg_tables.push(RegTable {
                param,
                table,
                delta,
                size,
                view: registers.table(table, delta, size),
            });
            self.reg_tables.len() - 1
        }))
    }

    /// Which registers the fields bind at `values` coincide — with each
    /// other, and with the registers the lift names on its own — as a
    /// probe must keep it: a lift can turn on the coincidence.
    fn reg_pattern(&self, values: &[u8], registers: &Registers<'_>) -> Vec<bool> {
        let bound: Vec<Option<sleigh::RegisterId>> = self
            .reg_fields
            .iter()
            .map(|&(param, table)| registers.register(table, values[usize::from(param)]))
            .collect();
        let at_base: Vec<Option<sleigh::RegisterId>> = self
            .reg_fields
            .iter()
            .map(|&(param, table)| {
                registers.register(table, self.keys.base_regs[usize::from(param)])
            })
            .collect();
        let implicit: Vec<sleigh::RegisterId> = self
            .varnodes
            .iter()
            .filter_map(|slot| match slot {
                VarnodeSlot::Fixed(varnode) => {
                    registers.spec.register_at(registers.geometry(*varnode))
                }
                VarnodeSlot::Register(_) => None,
            })
            .filter(|register| !at_base.contains(&Some(*register)))
            .collect();
        let mut pattern = Vec::new();
        for i in 0..bound.len() {
            for j in i + 1..bound.len() {
                pattern.push(bound[i] == bound[j]);
            }
            pattern.push(bound[i].is_some_and(|r| implicit.contains(&r)));
        }
        pattern
    }

    /// The coincidences at the base values, for
    /// [`instance`](Self::instance) to hold instances to: among the whole
    /// registers the fields bind, and between those and the registers the
    /// slots name outright.
    fn record_ties(&mut self, registers: &Registers<'_>) {
        let mut views: Vec<VarnodeSlot> = Vec::new();
        for index in 0..self.reg_fields.len() {
            let (param, table) = self.reg_fields[index];
            let Some(size) = registers
                .spec
                .attached_registers(table)
                .iter()
                .find_map(|r| registers.spec.register_varnode((*r)?))
                .map(|whole| whole.size)
            else {
                continue;
            };
            let view = self
                .reg_tables
                .iter()
                .position(|t| {
                    t.param == param && t.table == table && t.delta == 0 && t.size == size
                })
                .unwrap_or_else(|| {
                    self.reg_tables.push(RegTable {
                        param,
                        table,
                        delta: 0,
                        size,
                        view: registers.table(table, 0, size),
                    });
                    self.reg_tables.len() - 1
                });
            views.push(VarnodeSlot::Register(view as u32));
        }
        let fields = views.len();
        for slot in &self.varnodes {
            if matches!(slot, VarnodeSlot::Fixed(_)) && !views.contains(slot) {
                views.push(*slot);
            }
        }
        // Only pairs that can coincide at some instance are worth holding
        // instances to: a field never binds a flag.
        let could_coincide = |a: VarnodeSlot, b: VarnodeSlot| match (a, b) {
            (VarnodeSlot::Register(a), VarnodeSlot::Register(b)) => {
                let (a, b) = (&self.reg_tables[a as usize], &self.reg_tables[b as usize]);
                a.view
                    .varnodes
                    .iter()
                    .flatten()
                    .any(|v| b.view.varnodes.contains(&Some(*v)))
            }
            (VarnodeSlot::Register(a), VarnodeSlot::Fixed(v))
            | (VarnodeSlot::Fixed(v), VarnodeSlot::Register(a)) => {
                self.reg_tables[a as usize].view.varnodes.contains(&Some(v))
            }
            (VarnodeSlot::Fixed(_), VarnodeSlot::Fixed(_)) => false,
        };
        let base = &self.keys.base_regs;
        let mut ties = Vec::new();
        for i in 0..fields {
            for j in i + 1..views.len() {
                let (Some(a), Some(b)) = (
                    self.varnode_with(views[i], base),
                    self.varnode_with(views[j], base),
                ) else {
                    continue;
                };
                if could_coincide(views[i], views[j]) {
                    ties.push((views[i], views[j], a == b));
                }
            }
        }
        self.ties = ties;
    }

    /// A result naming the template's own keys, for views over the template
    /// at `instance` in `func`: block `k` is the template's `k`th block —
    /// the instruction's own, then the external ones — and instruction `k`
    /// its `k`th operation.
    pub(crate) fn lifted_at(&self, instance: &Instance, func: FunctionId) -> Lifted {
        let block = |k: usize| BlockId::new(func, LocalBlockId::from(k));
        let blocks = (0..=self.blocks.len()).map(block);
        let exits = self.exits.iter().map(|exit| {
            let continuation = |c: Option<u32>| match c {
                None => Continuation::Next,
                Some(k) => Continuation::Block(block(k as usize)),
            };
            Exit::new(
                InstructionId::new(func, LocalInsnId::from(exit.site as usize)),
                exit.arm,
                exit.target.kind(instance, continuation),
            )
        });
        Lifted::new(instance.address, self.length, block(0), blocks, exits)
    }

    /// The address of the template's `k`th block at `instance`: the
    /// instruction's own for the entry, none for its other blocks, another
    /// instruction's for an external one.
    pub(crate) fn block_address(&self, k: usize, instance: &Instance) -> Option<u64> {
        if k == 0 {
            Some(instance.address)
        } else if k <= self.blocks.len() {
            None
        } else {
            self.externals
                .get(k - 1 - self.blocks.len())
                .map(|affine| affine.at(instance))
        }
    }

    pub(crate) fn block_has_ops(&self, k: usize) -> bool {
        self.ops.iter().any(|op| op.block as usize == k)
    }

    /// The range of operation indices to scan for block `k`'s operations.
    pub(crate) fn ops_of(&self, k: usize) -> std::ops::Range<usize> {
        let first = self.ops.iter().position(|op| op.block as usize == k);
        let last = self.ops.iter().rposition(|op| op.block as usize == k);
        match (first, last) {
            (Some(first), Some(last)) => first..last + 1,
            _ => 0..0,
        }
    }

    pub(crate) fn op_block(&self, k: usize) -> usize {
        self.ops[k].block as usize
    }

    pub(crate) fn op_mnemonic(&self, k: usize) -> &Mnemonic {
        &self.ops[k].mnemonic
    }

    /// The `k`th literal's value and width at `instance`.
    pub(crate) fn literal_at(&self, k: usize, instance: &Instance) -> (u64, usize) {
        let affine = self.literals[k].affine;
        (affine.at(instance), affine.size)
    }

    /// The address of the template's `slot`th callee at `instance`.
    pub(crate) fn callee_address(&self, slot: usize, instance: &Instance) -> Option<u64> {
        self.callees.get(slot).map(|affine| affine.at(instance))
    }

    /// The template's slot for a callee of the captured IR: a function the
    /// construction resolved by address.
    fn callee_key(&mut self, ctx: &Context<'static>, callee: Callee) -> Result<Callee, Refusal> {
        let Callee::Real(function) = callee else {
            return Err(Refusal::Shape("a callee still minted"));
        };
        let address = FunctionBody::from_id(ctx, function)
            .address()
            .ok_or(Refusal::Shape("a callee without an address"))?;
        let key = Affine::fixed(address, 8);
        let slot = match self.callees.iter().position(|k| *k == key) {
            Some(slot) => slot,
            None => {
                self.callees.push(key);
                self.callees.len() - 1
            }
        };
        Ok(Callee::Minted(slot as u32))
    }

    /// Every value of the template that can move, in a fixed order: the
    /// literals, the externals, the callees, then the exits' targets.
    fn affines(&self) -> impl Iterator<Item = &Affine> {
        self.literals
            .iter()
            .map(|literal| &literal.affine)
            .chain(&self.externals)
            .chain(&self.callees)
            .chain(self.exits.iter().filter_map(|exit| exit.target.affine()))
    }

    fn affines_mut(&mut self) -> impl Iterator<Item = &mut Affine> {
        self.literals
            .iter_mut()
            .map(|literal| &mut literal.affine)
            .chain(&mut self.externals)
            .chain(&mut self.callees)
            .chain(
                self.exits
                    .iter_mut()
                    .filter_map(|exit| exit.target.affine_mut()),
            )
    }

    /// Whether two templates describe the same IR apart from the values
    /// that can move and debug names, or what differs.
    fn same_structure(&self, other: &Self) -> Result<(), &'static str> {
        if self.length != other.length {
            return Err("the two lifts differ in length");
        }
        if self.blocks.len() != other.blocks.len() {
            return Err("the two lifts differ in their blocks");
        }
        // Not the order the placeholders were made in: a block that existed
        // before one lift and not before the other was made in one and found
        // in the other, and the replay makes what it does not find.
        if self.externals.len() != other.externals.len() {
            return Err("the two lifts differ in the blocks they reach");
        }
        if self.callees.len() != other.callees.len() {
            return Err("the two lifts differ in their callees");
        }
        if self.temp_spaces != other.temp_spaces || self.temps != other.temps {
            return Err("the two lifts differ in their temporaries");
        }
        if self.types != other.types {
            return Err("the two lifts differ in their types");
        }
        if self.varnodes.len() != other.varnodes.len() {
            return Err("the two lifts differ in their varnodes");
        }
        if self.literals.len() != other.literals.len()
            || !self
                .literals
                .iter()
                .zip(&other.literals)
                .all(|(a, b)| a.ty == b.ty && a.affine.size == b.affine.size)
        {
            return Err("the two lifts differ in their constants");
        }
        if self.exits.len() != other.exits.len()
            || !self
                .exits
                .iter()
                .zip(&other.exits)
                .all(|(a, b)| a.site == b.site && a.arm == b.arm && a.target.same_kind(&b.target))
        {
            return Err("the two lifts differ in their exits");
        }
        if self.ops.len() != other.ops.len() {
            return Err("the two lifts differ in their number of operations");
        }
        if !self
            .ops
            .iter()
            .zip(&other.ops)
            .all(|(a, b)| a.block == b.block && a.ty == b.ty && a.mnemonic == b.mnemonic)
        {
            return Err("the two lifts differ in an operation");
        }
        Ok(())
    }

    /// How two canonical templates of the same instance differ in the IR
    /// they describe, debug names aside, if they do.
    fn disagreement(&self, other: &Self) -> Option<&'static str> {
        if self.base != other.base {
            return Some("the two lifts differ in address");
        }
        if let Err(reason) = self.same_structure(other) {
            return Some(reason);
        }
        if self.varnodes != other.varnodes {
            return Some("the two lifts differ in a register");
        }
        self.affines()
            .zip(other.affines())
            .any(|(a, b)| a != b)
            .then_some("the two lifts differ in a value")
    }

    /// Measures how every value of the template moves: lifts the same
    /// bytes at the probe address, then once per parameter with that field
    /// perturbed in a bit the mask leaves free, and compares slot by slot.
    /// A perturbation that lifts to another structure — a branch target
    /// landing on the fall-through, say — is retried in another bit.
    fn measure(
        &mut self,
        probing: &Probing<'_, '_>,
        store: &mut ScratchStore,
    ) -> Result<(), Refusal> {
        let Probing {
            lifter,
            probes,
            decoder,
            registers,
            address,
            bytes,
            shape,
            flat,
        } = *probing;
        let probe = |store: &mut ScratchStore, decoded: &Decoded<'_, '_>| {
            probes.fetch_add(1, Ordering::Relaxed);
            store.reset();
            // Debug names are not compared, and minting them costs.
            let mut target = store
                .target()
                .map_err(|_| Refusal::Shape("a probe had no target"))?
                .without_debug_names();
            let marks = Marks::of(&target);
            let lifted = lifter
                .lower(&mut target, decoded, flat)
                .map_err(|_| Refusal::Shape("a probe did not lift"))?;
            Template::capture(
                target.context(),
                target.addresses(),
                marks,
                &lifted,
                Keys::default(),
            )
        };

        let flip = |bits: &[usize]| {
            let mut perturbed = bytes.to_vec();
            for &bit in bits {
                perturbed[bit / 8] ^= 1 << (bit % 8);
            }
            perturbed
        };
        let params = self.keys.params.clone();
        let free: Vec<Vec<usize>> = params
            .iter()
            .map(|param| free_bits(param, shape.mask()).collect())
            .collect();

        // Everything at once, when the sums tell the parts apart.
        let perturbing = Perturbing {
            address,
            bytes,
            params: &params,
            free: &free,
            probe: &probe,
            flip: &flip,
            decoder,
            registers,
            shape,
        };
        if self.measure_together(store, &perturbing)? {
            return Ok(());
        }

        // One thing at a time.
        let decoded = decoder
            .decode(address.wrapping_add(PROBE_DISTANCE), bytes)
            .map_err(|_| Refusal::Shape("a probe did not decode"))?;
        let probed = probe(store, &decoded)?;
        self.compare(&probed, PROBE_DISTANCE, |affine| affine.relative = true)
            .map_err(|mismatch| Refusal::Shape(mismatch.reason()))?;

        for (index, param) in params.iter().enumerate() {
            let base = param.value(bytes);
            let free = &free[index];

            // Which values move with the field: the lowest free bit flipped,
            // the smallest step, so a constant of any width sees it.
            let mut first = None;
            for &bit in free {
                let perturbed = flip(&[bit]);
                let Some(decoded) = decode_alike(decoder, shape, bytes.len(), address, &perturbed)
                else {
                    continue;
                };
                let probed = probe(store, &decoded)?;
                let delta = param.value(&perturbed).wrapping_sub(base) as u64;
                match self.compare(&probed, delta, |affine| affine.params |= 1 << index) {
                    Ok(()) => {
                        first = Some(perturbed);
                        break;
                    }
                    Err(Mismatch::Structure(_)) => {}
                    Err(mismatch) => return Err(Refusal::Parameters(mismatch.reason())),
                }
            }
            let Some(first) = first else {
                return Err(Refusal::Parameters(
                    "no perturbation of a parameter lifts to the same structure",
                ));
            };

            // That they move linearly: a far step — every free bit flipped,
            // or the highest — must move exactly those values, by exactly
            // as much. A lane selector masked out of an immediate moves by
            // one when its low bit flips and not at all when a high one does.
            let far = [flip(free), flip(&free[free.len() - 1..])];
            let mut verified = free.len() < 2;
            for perturbed in far.iter().filter(|p| **p != first) {
                let Some(decoded) = decode_alike(decoder, shape, bytes.len(), address, perturbed)
                else {
                    continue;
                };
                let probed = probe(store, &decoded)?;
                let delta = param.value(perturbed).wrapping_sub(base) as u64;
                match self.verify(&probed, delta, index) {
                    Ok(()) => {
                        verified = true;
                        break;
                    }
                    Err(Mismatch::Structure(_)) => {}
                    Err(mismatch) => return Err(Refusal::Parameters(mismatch.reason())),
                }
            }
            if !verified {
                return Err(Refusal::Parameters(
                    "no far perturbation of a parameter lifts to the same structure",
                ));
            }
        }

        // Each register field on its own — with the fields bound to the
        // same register as it, so they stay so: the first perturbation the
        // shape admits that lifts to the same structure says which slots
        // follow it. Another value of the field is another register of the
        // same size, so one suffices.
        let regs = self.keys.regs.clone();
        let mut done = vec![false; regs.len()];
        for index in 0..regs.len() {
            if done[index] {
                continue;
            }
            let group: Vec<usize> = (0..regs.len())
                .filter(|&other| other == index || self.tied_fields(index, other, registers))
                .collect();
            let mut classified = false;
            for k in 0..8 {
                let Some(bits) = self.register_bits_at(&regs, &group, k, &[], &perturbing) else {
                    continue;
                };
                let perturbed = flip(&bits);
                let Some(decoded) = decode_alike(decoder, shape, bytes.len(), address, &perturbed)
                else {
                    continue;
                };
                let probed = probe(store, &decoded)?;
                let values: Vec<u8> = regs.iter().map(|param| param.value(&perturbed)).collect();
                match self.classify_varnodes(&probed, &values, registers) {
                    Ok(()) => {
                        classified = true;
                        break;
                    }
                    Err(Mismatch::Structure(_)) => {}
                    Err(mismatch) => return Err(Refusal::Parameters(mismatch.reason())),
                }
            }
            if !classified {
                return Err(Refusal::Parameters(
                    "no perturbation of a register field lifts to the same structure",
                ));
            }
            for &member in &group {
                done[member] = true;
            }
        }
        Ok(())
    }

    /// Whether register fields `a` and `b` bind one register at the base
    /// values through some of their views.
    fn tied_fields(&self, a: usize, b: usize, registers: &Registers<'_>) -> bool {
        let bound = |param: usize| {
            self.reg_fields
                .iter()
                .filter(move |(p, _)| usize::from(*p) == param)
                .filter_map(move |&(_, table)| {
                    registers.register(table, self.keys.base_regs[param])
                })
        };
        bound(a).any(|r| bound(b).any(|q| q == r))
    }

    /// `base`, plus a free bit of each register field in `group` — the
    /// `k`th, round robin — when the shape admits the flip and it keeps the
    /// registers coinciding as at the base values.
    fn register_bits_at(
        &self,
        regs: &[RegParam],
        group: &[usize],
        k: usize,
        base: &[usize],
        perturbing: &Perturbing<'_, '_>,
    ) -> Option<Vec<usize>> {
        // Fields bound to one register move alike, to stay so; fields
        // bound to different ones move differently, so that a slot inside
        // one register cannot pass for inside another.
        let mut classes: Vec<usize> = Vec::new();
        let mut bits = base.to_vec();
        for &i in group {
            let class = match group
                .iter()
                .take_while(|&&j| j != i)
                .position(|&j| self.tied_fields(i, j, perturbing.registers))
            {
                Some(index) => classes[index],
                None => classes.iter().max().map_or(0, |c| c + 1),
            };
            classes.push(class);
            let free: Vec<usize> =
                free_bits_of(regs[i].bit, regs[i].width, perturbing.shape.mask()).collect();
            bits.push(free[(k + class) % free.len()]);
        }
        let flipped = (perturbing.flip)(&bits);
        if !perturbing.shape.admits(&flipped) {
            return None;
        }
        let values: Vec<u8> = regs.iter().map(|param| param.value(&flipped)).collect();
        (self.reg_pattern(&values, perturbing.registers)
            == self.reg_pattern(&self.keys.base_regs, perturbing.registers))
        .then_some(bits)
    }

    /// [`register_bits_at`](Self::register_bits_at) in the first round
    /// that works.
    fn register_bits(
        &self,
        regs: &[RegParam],
        group: Vec<usize>,
        base: &[usize],
        perturbing: &Perturbing<'_, '_>,
    ) -> Option<Vec<usize>> {
        (0..8).find_map(|k| self.register_bits_at(regs, &group, k, base, perturbing))
    }

    /// [`measure`](Self::measure) in two probes instead of two per
    /// parameter: one with the address moved and every parameter's lowest
    /// free bit flipped, and one with every free bit of every parameter
    /// flipped. A slot of the first is classified by the subset of the
    /// probe's deltas that sums to its move in its width; the second must
    /// then move every slot by exactly the sum its subset says. Returns
    /// `false`, leaving the template unmarked, when a probe lifts to
    /// another structure or a slot's move has no unique subset — a byte
    /// constant cannot tell the address from an immediate when their deltas
    /// coincide in its low byte — and `measure` goes on one thing at a time.
    fn measure_together(
        &mut self,
        store: &mut ScratchStore,
        perturbing: &Perturbing<'_, '_>,
    ) -> Result<bool, Refusal> {
        let Perturbing {
            address,
            bytes,
            params,
            free,
            probe,
            flip,
            decoder,
            registers,
            shape,
        } = *perturbing;
        // Parameter `i` in its `i`th free bit, so two parameters' deltas
        // differ — two lowest bits both move by one, and a slot moving by
        // one could belong to either.
        let mut near_bits: Vec<usize> = free
            .iter()
            .enumerate()
            .map(|(i, f)| f[i.min(f.len() - 1)])
            .collect();
        // Every register field too, each in a free bit, such that the
        // shape admits the result and the registers coincide as before.
        let regs = self.keys.regs.clone();
        if !regs.is_empty() {
            let Some(bits) =
                self.register_bits(&regs, (0..regs.len()).collect(), &near_bits, perturbing)
            else {
                return Ok(false);
            };
            near_bits = bits;
        }
        let near = flip(&near_bits);
        let Some(decoded) = decode_alike(
            decoder,
            shape,
            bytes.len(),
            address.wrapping_add(PROBE_DISTANCE),
            &near,
        ) else {
            return Ok(false);
        };
        let probed = probe(store, &decoded)?;
        if self.same_structure(&probed).is_err() {
            return Ok(false);
        }
        let near_regs: Vec<u8> = regs.iter().map(|param| param.value(&near)).collect();
        match self.classify_varnodes(&probed, &near_regs, registers) {
            Ok(()) => {}
            Err(Mismatch::Structure(_) | Mismatch::Ambiguous) => return Ok(false),
            Err(Mismatch::Value(reason)) => return Err(Refusal::Parameters(reason)),
        }
        // Source 0 is the address; source `i + 1` is parameter `i`.
        let mut deltas = [0u64; MAX_PARAMS + 1];
        deltas[0] = PROBE_DISTANCE;
        for (i, param) in params.iter().enumerate() {
            deltas[i + 1] = param.value(&near).wrapping_sub(param.value(bytes)) as u64;
        }
        let sources = params.len() + 1;
        let mut marks: Vec<(bool, u8)> = Vec::with_capacity(self.literals.len());
        for (mine, theirs) in self.affines().zip(probed.affines()) {
            let mask = Affine::mask(mine.size);
            let moved = theirs.value.wrapping_sub(mine.value) & mask;
            let mut found = None;
            for subset in 0u32..(1 << sources) {
                let sum = (0..sources)
                    .filter(|&i| subset & (1 << i) != 0)
                    .fold(0u64, |acc, i| acc.wrapping_add(deltas[i]));
                if sum & mask == moved {
                    if found.is_some() {
                        return Ok(false);
                    }
                    found = Some(subset);
                }
            }
            let Some(subset) = found else {
                return Ok(false);
            };
            marks.push((subset & 1 != 0, (subset >> 1) as u8));
        }

        if free.iter().any(|f| f.len() > 1) {
            // Every free bit, or the highest of each, as `measure` tries.
            let all: Vec<usize> = free.iter().flatten().copied().collect();
            let highest: Vec<usize> = free.iter().map(|f| f[f.len() - 1]).collect();
            let candidates = [flip(&all), flip(&highest)];
            let Some((far, decoded)) =
                candidates
                    .iter()
                    .filter(|far| **far != near)
                    .find_map(|far| {
                        decode_alike(decoder, shape, bytes.len(), address, far)
                            .map(|decoded| (far, decoded))
                    })
            else {
                return Ok(false);
            };
            let probed = probe(store, &decoded)?;
            if self.same_structure(&probed).is_err()
                || !self.varnodes_agree(&probed, &self.keys.base_regs)
            {
                return Ok(false);
            }
            for (i, param) in params.iter().enumerate() {
                deltas[i + 1] = param.value(far).wrapping_sub(param.value(bytes)) as u64;
            }
            for ((mine, theirs), &(_, moving)) in self.affines().zip(probed.affines()).zip(&marks) {
                let expected = (0..params.len())
                    .filter(|&i| moving & (1 << i) != 0)
                    .fold(mine.value, |acc, i| acc.wrapping_add(deltas[i + 1]));
                if theirs.value != expected & Affine::mask(mine.size) {
                    return Err(Refusal::Parameters(
                        "a constant is not linear in a parameter",
                    ));
                }
            }
        }

        for (affine, &(relative, moving)) in self.affines_mut().zip(&marks) {
            affine.relative = relative;
            affine.params = moving;
        }
        Ok(true)
    }

    /// Checks `probed` — the same instruction lifted with parameter
    /// `index` changed by `delta` — against what [`compare`](Self::compare)
    /// found: every value marked as moving with the parameter moved by
    /// exactly `delta` in its width, and no other value moved.
    fn verify(&self, probed: &Self, delta: u64, index: usize) -> Result<(), Mismatch> {
        self.same_structure(probed).map_err(Mismatch::Structure)?;
        if !self.varnodes_agree(probed, &self.keys.base_regs) {
            return Err(Mismatch::Value("a register operand moved with a parameter"));
        }
        for (mine, theirs) in self.affines().zip(probed.affines()) {
            let mask = Affine::mask(mine.size);
            let expected = if mine.params & (1 << index) != 0 {
                mine.value.wrapping_add(delta) & mask
            } else {
                mine.value
            };
            if theirs.value != expected {
                return Err(Mismatch::Value("a constant is not linear in a parameter"));
            }
        }
        Ok(())
    }

    /// Marks, with `moved`, every value of the template that `probed` — the
    /// same instruction lifted with one thing changed by `delta` — shows
    /// moving by exactly `delta` in its width; an unmoved one stays;
    /// anything else is a mismatch.
    fn compare(
        &mut self,
        probed: &Self,
        delta: u64,
        moved: impl Fn(&mut Affine),
    ) -> Result<(), Mismatch> {
        self.same_structure(probed).map_err(Mismatch::Structure)?;
        if !self.varnodes_agree(probed, &self.keys.base_regs) {
            return Err(Mismatch::Value("a register operand moved with a probe"));
        }
        for (mine, theirs) in self.affines_mut().zip(probed.affines()) {
            let mask = Affine::mask(mine.size);
            if theirs.value == mine.value {
                if delta & mask == 0 {
                    return Err(Mismatch::Value(
                        "a probe's delta is invisible in a constant",
                    ));
                }
                continue;
            }
            if theirs.value.wrapping_sub(mine.value) & mask == delta & mask {
                moved(mine);
            } else {
                return Err(Mismatch::Value(
                    "a constant is neither fixed nor moving with a probe",
                ));
            }
        }
        Ok(())
    }

    /// Merges literal slots that hold the same value, moving alike, and
    /// renumbers the operations.
    fn dedupe_literals(&mut self) {
        let mut unique: Vec<Literal> = Vec::with_capacity(self.literals.len());
        let remap: Vec<usize> = self
            .literals
            .iter()
            .map(|literal| match unique.iter().position(|u| u == literal) {
                Some(index) => index,
                None => {
                    unique.push(*literal);
                    unique.len() - 1
                }
            })
            .collect();
        for op in &mut self.ops {
            op.mnemonic = op.mnemonic.clone().map_operands(|operand| match operand {
                LocalValueId::Literal(k) => {
                    LocalValueId::Literal(LiteralId::from(remap[usize::from(k)]))
                }
                other => other,
            });
        }
        self.literals = unique;
    }

    /// Merges varnode slots that name the same varnode, following the same
    /// field, and renumbers the operations.
    fn dedupe_varnodes(&mut self) {
        let mut unique: Vec<VarnodeSlot> = Vec::with_capacity(self.varnodes.len());
        let remap: Vec<usize> = self
            .varnodes
            .iter()
            .map(|slot| match unique.iter().position(|u| u == slot) {
                Some(index) => index,
                None => {
                    unique.push(*slot);
                    unique.len() - 1
                }
            })
            .collect();
        for op in &mut self.ops {
            op.mnemonic = op.mnemonic.clone().map_operands(|operand| match operand {
                LocalValueId::Varnode(k) => {
                    LocalValueId::Varnode(VarnodeId::from(remap[usize::from(k)]))
                }
                other => other,
            });
            if let Some(OpName::Varnode(k)) = &mut op.name {
                *k = remap[*k as usize] as u32;
            }
        }
        self.varnodes = unique;
    }

    /// This template with its values as they are at `instance`, in
    /// canonical form.
    fn instantiate(&self, instance: &Instance) -> Self {
        let mut resolved = self.clone();
        resolved.base = instance.address;
        resolved.keys = Keys::default();
        for affine in resolved.affines_mut() {
            *affine = affine.resolved(instance);
        }
        for slot in &mut resolved.varnodes {
            *slot = VarnodeSlot::Fixed(
                self.varnode_with(*slot, &instance.regs)
                    .expect("an instance resolves every register"),
            );
        }
        resolved.reg_tables.clear();
        resolved.canonicalize();
        resolved
    }

    /// Puts a template whose values are all fixed in a form two lifts of
    /// the same IR share: literals merged by value, externals and callees
    /// in address order without repeats — two that moved differently can
    /// coincide at an instance, or land on the instruction itself, whose
    /// block is the entry — and the operations renumbered to match.
    fn canonicalize(&mut self) {
        self.dedupe_literals();
        self.dedupe_varnodes();
        let internal = 1 + self.blocks.len();
        let mut addresses: Vec<u64> = self
            .externals
            .iter()
            .map(|a| a.value)
            .filter(|&a| a != self.base)
            .collect();
        addresses.sort_unstable();
        addresses.dedup();
        let block_remap: Vec<u32> = self
            .externals
            .iter()
            .map(|a| match addresses.binary_search(&a.value) {
                Ok(index) => (internal + index) as u32,
                Err(_) => 0,
            })
            .collect();
        let block = |id: LocalBlockId| -> LocalBlockId {
            let k = usize::from(id);
            if k < internal {
                id
            } else {
                LocalBlockId::from(block_remap[k - internal] as usize)
            }
        };
        let mut callees: Vec<u64> = self.callees.iter().map(|a| a.value).collect();
        callees.sort_unstable();
        callees.dedup();
        let callee_remap: Vec<u32> = self
            .callees
            .iter()
            .map(|a| callees.binary_search(&a.value).expect("its own address") as u32)
            .collect();
        for op in &mut self.ops {
            match &mut op.mnemonic {
                Mnemonic::Branch(branch) => branch.target = block(branch.target),
                Mnemonic::CBranch(cbranch) => {
                    cbranch.success_block = block(cbranch.success_block);
                    cbranch.failure_block = block(cbranch.failure_block);
                }
                Mnemonic::Call(call) => {
                    call.target =
                        Callee::Minted(callee_remap[call.target.minted().unwrap() as usize])
                }
                Mnemonic::TailCall(call) => {
                    call.target =
                        Callee::Minted(callee_remap[call.target.minted().unwrap() as usize])
                }
                _ => {}
            }
        }
        for exit in &mut self.exits {
            let continuation = match &mut exit.target {
                ExitTarget::Call { continuation, .. } | ExitTarget::CallInd { continuation } => {
                    continuation
                }
                _ => continue,
            };
            if let Some(k) = continuation {
                *k = usize::from(block(LocalBlockId::from(*k as usize))) as u32;
            }
        }
        self.externals = addresses.iter().map(|&a| Affine::fixed(a, 8)).collect();
        self.external_order = (0..self.externals.len() as u32).collect();
        self.callees = callees.iter().map(|&a| Affine::fixed(a, 8)).collect();
    }

    /// Lifts the instruction this template is at `instance` into `target`,
    /// from the record alone. `scratch` is the replay's working memory.
    pub(crate) fn replay(
        self: &Arc<Self>,
        target: &mut LiftTarget<'_, 'static>,
        instance: &Instance,
        scratch: &mut ReplayScratch,
    ) -> Result<Lifted, LiftError> {
        let mut construction = target.begin(instance.address, self.length)?;
        self.emit(&mut construction, instance, scratch)?;
        Ok(construction.commit()?)
    }

    /// The debug name of the `k`th operation at `instance`: as recorded,
    /// or that of the register in its varnode slot, which
    /// [`bake_names`](Self::bake_names) resolved when the slot is fixed.
    fn name_at(&self, k: usize, instance: &Instance) -> Option<&str> {
        match self.ops[k].name.as_ref()? {
            OpName::Fixed(name) => Some(name),
            OpName::Varnode(slot) => match self.varnodes[*slot as usize] {
                VarnodeSlot::Register(table) => {
                    let table = &self.reg_tables[table as usize];
                    table.view.names[usize::from(instance.regs[usize::from(table.param)])]
                        .as_deref()
                }
                VarnodeSlot::Fixed(_) => {
                    debug_assert!(false, "a fixed slot's name was not baked");
                    None
                }
            },
        }
    }

    /// Resolves the name of every operation named after a fixed varnode
    /// slot, once the probes have settled which slots are fixed, so a
    /// replay does not look the register up and lowercase it every time.
    fn bake_names(&mut self, ctx: &Context<'static>) {
        for op in &mut self.ops {
            if let Some(OpName::Varnode(slot)) = op.name
                && let VarnodeSlot::Fixed(varnode) = self.varnodes[slot as usize]
            {
                op.name = Varnode::from_id(ctx, varnode)
                    .name()
                    .map(|name| OpName::Fixed(name.to_lowercase().into_boxed_str()));
            }
        }
    }

    fn emit(
        self: &Arc<Self>,
        construction: &mut Construction<'_, '_, 'static>,
        instance: &Instance,
        scratch: &mut ReplayScratch,
    ) -> Result<(), LiftError> {
        let address = instance.address;
        let ReplayScratch {
            blocks,
            externals,
            callees,
            types,
            resolved,
            varnodes,
            literals,
            spaces,
            temps,
            insns,
        } = scratch;
        blocks.clear();
        externals.clear();
        callees.clear();
        types.clear();
        varnodes.clear();
        literals.clear();
        spaces.clear();
        temps.clear();
        insns.clear();
        // Everything the construction resolves against the module comes
        // before the emitter borrows it: the blocks of other instructions and
        // the callees, exactly as the emitter resolves its plan.
        blocks.push(construction.entry());
        externals.resize(self.externals.len(), None);
        for &index in &self.external_order {
            let external = self.externals[index as usize].at(instance);
            externals[index as usize] = Some(construction.block_at(external)?);
        }
        for affine in &self.callees {
            callees.push(construction.callee_at(affine.at(instance))?);
        }

        let ctx = construction.context();
        let resolved = resolved
            .entry(Arc::as_ptr(self) as usize)
            .or_insert_with(|| Resolved {
                _template: Arc::clone(self),
                types: self.types.iter().map(|key| key.resolve(ctx)).collect(),
                literals: self
                    .literals
                    .iter()
                    .map(|literal| literal.is_fixed().then(|| literal.resolve(ctx, instance)))
                    .collect(),
            });
        types.extend_from_slice(&resolved.types);
        literals.extend(
            self.literals
                .iter()
                .zip(&resolved.literals)
                .map(|(literal, fixed)| fixed.unwrap_or_else(|| literal.resolve(ctx, instance))),
        );
        varnodes.extend((0..self.varnodes.len()).map(|k| self.varnode_at(k, instance)));

        let mut emitter = construction.emitter();
        emitter.set_address(address);
        for name in &self.blocks {
            let block = match name.render(address) {
                Some(name) if emitter.naming() => emitter.get_or_make_local_label(Cow::Owned(name)),
                _ => emitter.push_anonymous_block(),
            };
            emitter.block(block);
            blocks.push(block);
        }
        blocks.extend(
            externals
                .iter()
                .map(|b| b.expect("every external is ordered")),
        );
        for space in &self.temp_spaces {
            spaces.push(
                emitter
                    .push_temp_space(TempSpace::new(None, space.word_size, space.addr_size))
                    .local,
            );
        }
        for temp in &self.temps {
            temps.push(
                emitter
                    .push_temp(Temp::new(
                        temp.address,
                        temp.size,
                        spaces[temp.space as usize],
                    ))
                    .local,
            );
        }

        let mut current = 0u32;
        for (k, op) in self.ops.iter().enumerate() {
            if op.block != current {
                current = op.block;
                emitter.switch_to_block(blocks[current as usize]);
            }
            let mut mnemonic = op.mnemonic.clone().map_operands(|operand| match operand {
                LocalValueId::Literal(index) => literals[usize::from(index)],
                LocalValueId::Instruction(index) => {
                    LocalValueId::Instruction(insns[usize::from(index)].local)
                }
                LocalValueId::Temp(index) => LocalValueId::Temp(temps[usize::from(index)]),
                LocalValueId::Varnode(index) => LocalValueId::Varnode(varnodes[usize::from(index)]),
                other => other,
            });
            match &mut mnemonic {
                Mnemonic::Load(load) => {
                    if let LocalMemorySpaceId::Temp(space) = &mut load.space {
                        *space = spaces[usize::from(*space)];
                    }
                }
                Mnemonic::Store(store) => {
                    if let LocalMemorySpaceId::Temp(space) = &mut store.space {
                        *space = spaces[usize::from(*space)];
                    }
                }
                Mnemonic::Branch(branch) => {
                    branch.target = blocks[usize::from(branch.target)].local
                }
                Mnemonic::CBranch(cbranch) => {
                    cbranch.success_block = blocks[usize::from(cbranch.success_block)].local;
                    cbranch.failure_block = blocks[usize::from(cbranch.failure_block)].local;
                }
                Mnemonic::Call(call) => {
                    call.target = callees[call.target.minted().unwrap() as usize]
                }
                Mnemonic::TailCall(call) => {
                    call.target = callees[call.target.minted().unwrap() as usize]
                }
                _ => {}
            }
            let id = emitter
                .push_mnemonic_with_type_named(
                    mnemonic,
                    types[op.ty as usize],
                    self.name_at(k, instance),
                )
                .id;
            insns.push(id);
        }

        for exit in &self.exits {
            let continuation = |c: Option<u32>| match c {
                None => Continuation::Next,
                Some(block) => Continuation::Block(blocks[block as usize]),
            };
            emitter.exit(
                insns[exit.site as usize],
                exit.arm,
                exit.target.kind(instance, continuation),
            );
        }
        Ok(())
    }
}

/// A replay's working memory: what it resolves before emitting, kept by
/// the session between hits so a hit allocates none of it.
#[derive(Debug, Default)]
pub struct ReplayScratch {
    /// The entry, the instruction's own blocks, then the externals.
    blocks: Vec<BlockId>,
    externals: Vec<Option<BlockId>>,
    callees: Vec<Callee>,
    types: Vec<TypeId>,
    /// Per template replayed, by address, what its keys resolve to in the
    /// session's context.
    resolved: HashMap<usize, Resolved>,
    varnodes: Vec<VarnodeId>,
    literals: Vec<LocalValueId>,
    spaces: Vec<LocalTempSpaceId>,
    temps: Vec<LocalTempId>,
    insns: Vec<InstructionId>,
}

/// What a miss probes: the instruction, how it is decoded and lifted, and
/// its shape.
#[derive(Clone, Copy)]
struct Probing<'a, 'spec> {
    lifter: &'a SleighLifter<'spec>,
    probes: &'a AtomicU64,
    decoder: &'a FixedDecoder<'spec>,
    registers: &'a Registers<'a>,
    address: u64,
    bytes: &'a [u8],
    shape: &'a Shape,
    flat: bool,
}

/// Lifts a decoded instruction into a store and captures it.
type Probe<'a> = dyn Fn(&mut ScratchStore, &Decoded<'_, '_>) -> Result<Template, Refusal> + 'a;

/// A perturbation of an instruction of `shape`, `len` bytes long, decoded
/// at `at` — where it will be lifted — when the shape admits it and it
/// decodes to the same length.
fn decode_alike<'spec, 'b>(
    decoder: &FixedDecoder<'spec>,
    shape: &Shape,
    len: usize,
    at: u64,
    perturbed: &'b [u8],
) -> Option<Decoded<'spec, 'b>> {
    if !shape.admits(perturbed) {
        return None;
    }
    decoder
        .decode(at, perturbed)
        .ok()
        .filter(|candidate| candidate.len() == len)
}

/// A miss's instruction and the ways [`measure`](Template::measure)
/// perturbs it.
#[derive(Clone, Copy)]
struct Perturbing<'a, 'spec> {
    address: u64,
    bytes: &'a [u8],
    params: &'a [ParamField],
    /// Per parameter, its free bits, lowest first.
    free: &'a [Vec<usize>],
    /// Lifts bytes at an address and captures them.
    probe: &'a Probe<'a>,
    /// The instruction with those bits flipped.
    flip: &'a dyn Fn(&[usize]) -> Vec<u8>,
    decoder: &'a FixedDecoder<'spec>,
    registers: &'a Registers<'a>,
    shape: &'a Shape,
}

/// How a probe's lift differs from the captured one.
enum Mismatch {
    /// A probe moved two things a slot could follow either of; one at a
    /// time tells.
    Ambiguous,
    /// In its structure: the probe is not an instance of the same template.
    Structure(&'static str),
    /// In a value that moved by something other than the probe's delta.
    Value(&'static str),
}

impl Mismatch {
    fn reason(&self) -> &'static str {
        match self {
            Self::Ambiguous => "a probe moved two things a value could follow",
            Self::Structure(reason) | Self::Value(reason) => reason,
        }
    }
}

impl ExitTarget {
    fn affine(&self) -> Option<&Affine> {
        match self {
            Self::Branch(affine)
            | Self::Call {
                callee: CallKey::Address(affine),
                ..
            } => Some(affine),
            _ => None,
        }
    }

    fn affine_mut(&mut self) -> Option<&mut Affine> {
        match self {
            Self::Branch(affine)
            | Self::Call {
                callee: CallKey::Address(affine),
                ..
            } => Some(affine),
            _ => None,
        }
    }

    /// Whether two targets are alike apart from the addresses they hold.
    fn same_kind(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Branch(_), Self::Branch(_)) => true,
            (
                Self::Call {
                    callee: a,
                    continuation: c,
                },
                Self::Call {
                    callee: b,
                    continuation: d,
                },
            ) => {
                c == d
                    && match (a, b) {
                        (CallKey::Address(_), CallKey::Address(_)) => true,
                        (CallKey::Named(a), CallKey::Named(b)) => a == b,
                        _ => false,
                    }
            }
            _ => self == other,
        }
    }

    /// The exit kind this target is at `instance`, with `continuation`
    /// resolving the template's block indices.
    fn kind(
        &self,
        instance: &Instance,
        continuation: impl Fn(Option<u32>) -> Continuation,
    ) -> ExitKind {
        match self {
            Self::Fallthrough => ExitKind::Fallthrough,
            Self::Branch(affine) => ExitKind::Branch {
                target: affine.at(instance),
            },
            Self::BranchInd => ExitKind::BranchInd,
            Self::Call {
                callee,
                continuation: c,
            } => ExitKind::Call {
                callee: match callee {
                    CallKey::Address(affine) => CallTarget::Address(affine.at(instance)),
                    CallKey::Named(name) => CallTarget::Named(name.clone()),
                },
                continuation: continuation(*c),
            },
            Self::CallInd { continuation: c } => ExitKind::CallInd {
                continuation: continuation(*c),
            },
            Self::Return => ExitKind::Return,
        }
    }
}
