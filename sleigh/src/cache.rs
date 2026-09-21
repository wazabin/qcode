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
//! bits fails the comparison instead of matching by accident. [`LiftCache::validating`] lifts every hit for real as well and
//! compares, for measuring that claim over a corpus.
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
//! children by masked value, and a leaf holds a shape's entry with the
//! patterns it excludes — the more specific candidates the decoder passed
//! over, which an encoding of the shape never matches. A lookup needs no
//! decode: it walks the bytes, tries every mask at a node, and passes over a
//! leaf whose exclusions reject the stream. A session decoding under
//! another context or lowering calls differently never sees another's
//! entries, and the cache is bound to one specification and refuses a
//! lifter of another.
//!
//! A template records debug names, so a replay into a target that
//! [names](qcode::lift::LiftTarget::without_debug_names) its values names
//! them as a fresh lift would.

use std::{
    borrow::Cow,
    cell::RefCell,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, AtomicUsize, Ordering},
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
        TempRef, TempSpace, TempSpaceId,
        insn::{Callee, Mnemonic},
        view::ModuleView,
    },
};
use rustc_hash::FxHashMap as HashMap;
use sleigh::{
    CompiledSpec, ContextBytes, Exclusion, Instruction as Decoded, ParamField, Shape,
    SpecFingerprint,
};

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

/// One shard per leading byte of the instruction.
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
    Template(Arc<Template>),
    Uncacheable,
}

/// What a leaf of the trie holds: a shape's entry and the patterns no
/// encoding of the shape matches.
struct Entry {
    exclusions: Box<[Exclusion]>,
    kind: Kind,
}

impl Entry {
    /// Whether `bytes`, a stream agreeing with the shape's masked bytes over
    /// its first `len`, starts an instruction of the shape. An exclusion
    /// the stream is too short to test does not match, as it would not for
    /// the decoder reading the same stream.
    fn admits(&self, bytes: &[u8], len: usize) -> bool {
        bytes.len() >= len && !self.exclusions.iter().any(|e| e.matches(bytes))
    }
}

/// A node of the lookup trie, at one byte of the instruction stream.
#[derive(Default)]
struct Node {
    /// Per mask byte some shape applies at this byte, the children by the
    /// byte's masked value. Almost always one.
    arms: Vec<(u8, HashMap<u8, Node>)>,
    /// The shape whose instructions end here.
    leaf: Option<Entry>,
}

impl Node {
    /// The entry of the shape `bytes` starts, searching from `depth`. With
    /// `exact`, `bytes` is one instruction of that length and only a leaf
    /// at that depth counts; an exclusion reaching past it cannot have
    /// matched, or the decoder would have taken the longer candidate.
    fn find(&self, bytes: &[u8], depth: usize, exact: Option<usize>) -> Option<&Entry> {
        if let Some(entry) = &self.leaf
            && exact.is_none_or(|len| len == depth)
            && entry.admits(bytes, depth)
        {
            return Some(entry);
        }
        if exact.is_some_and(|len| depth >= len) {
            return None;
        }
        let byte = *bytes.get(depth)?;
        for (mask, children) in &self.arms {
            if let Some(child) = children.get(&(byte & mask))
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
            let arm = match node.arms.iter().position(|(m, _)| *m == mask) {
                Some(arm) => arm,
                None => {
                    node.arms.push((mask, HashMap::default()));
                    node.arms.len() - 1
                }
            };
            node = node.arms[arm].1.entry(value).or_default();
        }
        node
    }

    /// The node at the end of the path of `mask` and `masked`, if present.
    fn get_mut(&mut self, mask: &[u8], masked: &[u8]) -> Option<&mut Node> {
        let mut node = self;
        for (&mask, &value) in mask.iter().zip(masked) {
            let arm = node.arms.iter().position(|(m, _)| *m == mask)?;
            node = node.arms[arm].1.get_mut(&value)?;
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

/// A cache of lifted instructions keyed by the shape of their encoding. See the [module
/// documentation](self).
///
/// It is shared between the sessions of one lifter — across threads, behind
/// an [`Arc`] — through [`ScratchSession::with_cache`](crate::session::ScratchSession::with_cache)
/// and [`LiftSession::with_cache`](crate::session::LiftSession::with_cache).
pub struct LiftCache {
    fingerprint: SpecFingerprint,
    shards: Box<[Shard]>,
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
        let shard = self.shards[usize::from(first)].read().unwrap();
        let entry = shard.get(prefix)?.find(bytes, 0, exact)?;
        Some(match &entry.kind {
            Kind::Template(template) => {
                let instance = template.instance(address, bytes);
                Found::Template(Arc::clone(template), instance)
            }
            Kind::Uncacheable => Found::Uncacheable,
        })
    }

    fn insert(&self, prefix: &[u8], mask: &[u8], masked: &[u8], entry: Entry) {
        if self.entries.load(Ordering::Relaxed) >= self.capacity {
            return;
        }
        let Some(&first) = masked.first() else {
            return;
        };
        let mut shard = self.shards[usize::from(first)].write().unwrap();
        let leaf = &mut shard
            .entry(Box::from(prefix))
            .or_default()
            .walk_mut(mask, masked)
            .leaf;
        if leaf.is_none() {
            *leaf = Some(entry);
            self.entries.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn evict(&self, prefix: &[u8], template: &Template) {
        let keys = &template.keys;
        let Some(&first) = keys.masked.first() else {
            return;
        };
        let mut shard = self.shards[usize::from(first)].write().unwrap();
        if let Some(node) = shard
            .get_mut(prefix)
            .and_then(|root| root.get_mut(&keys.mask, &keys.masked))
            && node.leaf.take().is_some()
        {
            self.entries.fetch_sub(1, Ordering::Relaxed);
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
    ) -> Result<Lifted, LiftError> {
        match self.find(lifter, instruction, decoder, flat)? {
            Lookup::Hit(template, instance) => template.replay(target, &instance),
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
        let attempt = |params: Vec<ParamField>, mask: &[u8], masked: &[u8]| {
            let keys = Keys {
                mask: mask.into(),
                masked: masked.into(),
                base_params: params.iter().map(|p| p.value(bytes)).collect(),
                params: params.into_boxed_slice(),
            };
            let mut template =
                Template::capture(target.context(), target.addresses(), marks, &lifted, keys)?;
            let probing = Probing {
                lifter,
                probes: &self.probes,
                decoder,
                address,
                bytes,
                shape: &shape,
                flat,
            };
            with_probe_store(lifter, |store| template.measure(&probing, store))?;
            template.dedupe_literals();
            Ok(template)
        };
        let captured = parameters(&shape)
            .and_then(|params| attempt(params, shape.mask(), &masked))
            .or_else(|refusal| match refusal {
                Refusal::Parameters(reason) => {
                    debug_uncacheable(instruction, "is remembered exactly", reason);
                    self.exact.fetch_add(1, Ordering::Relaxed);
                    attempt(Vec::new(), &vec![0xff; bytes.len()], bytes)
                }
                other => Err(other),
            });
        let (mask, masked, kind) = match captured {
            Ok(template) => {
                let (mask, masked) = (template.keys.mask.clone(), template.keys.masked.clone());
                (mask, masked, Kind::Template(Arc::new(template)))
            }
            Err(Refusal::Shape(reason) | Refusal::Parameters(reason)) => {
                debug_uncacheable(instruction, "is uncacheable", reason);
                (shape.mask().into(), masked.into(), Kind::Uncacheable)
            }
        };
        let entry = Entry {
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
/// `k`th callee. Varnodes are the architecture's and stay as they are.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Op {
    block: u32,
    ty: u32,
    mnemonic: Mnemonic,
    name: Option<Box<str>>,
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
    ops: Vec<Op>,
    exits: Vec<ExitRecord>,
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
                    LocalValueId::Varnode(_) => operand,
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
                name: reference.name().map(|name| base_name(ctx, name)),
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
    /// starting `bytes`, is.
    fn instance(&self, address: u64, bytes: &[u8]) -> Instance {
        let mut params = [0i64; MAX_PARAMS];
        for ((slot, param), base) in params
            .iter_mut()
            .zip(&self.keys.params)
            .zip(&self.keys.base_params)
        {
            *slot = param.value(bytes).wrapping_sub(*base);
        }
        Instance {
            address,
            delta: address.wrapping_sub(self.base),
            params,
        }
    }

    /// A result naming the template's own keys, for views over the template
    /// at `instance` in `func`: block `k` is the template's `k`th block —
    /// the instruction's own, then the external ones — and instruction `k`
    /// its `k`th operation.
    pub(crate) fn lifted_at(&self, instance: &Instance, func: FunctionId) -> Lifted {
        let block = |k: usize| BlockId::new(func, LocalBlockId::from(k));
        let blocks: Vec<BlockId> = (0..=self.blocks.len()).map(block).collect();
        let exits = self
            .exits
            .iter()
            .map(|exit| {
                let continuation = |c: Option<u32>| match c {
                    None => Continuation::Next,
                    Some(k) => Continuation::Block(block(k as usize)),
                };
                Exit::new(
                    InstructionId::new(func, LocalInsnId::from(exit.site as usize)),
                    exit.arm,
                    exit.target.kind(instance, continuation),
                )
            })
            .collect();
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
                    Err(Mismatch::Value(reason)) => return Err(Refusal::Parameters(reason)),
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
                    Err(Mismatch::Value(reason)) => return Err(Refusal::Parameters(reason)),
                }
            }
            if !verified {
                return Err(Refusal::Parameters(
                    "no far perturbation of a parameter lifts to the same structure",
                ));
            }
        }
        Ok(())
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
            shape,
        } = *perturbing;
        // Parameter `i` in its `i`th free bit, so two parameters' deltas
        // differ — two lowest bits both move by one, and a slot moving by
        // one could belong to either.
        let near_bits: Vec<usize> = free
            .iter()
            .enumerate()
            .map(|(i, f)| f[i.min(f.len() - 1)])
            .collect();
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
            if self.same_structure(&probed).is_err() {
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

    /// This template with its values as they are at `instance`, in
    /// canonical form.
    fn instantiate(&self, instance: &Instance) -> Self {
        let mut resolved = self.clone();
        resolved.base = instance.address;
        resolved.keys = Keys::default();
        for affine in resolved.affines_mut() {
            *affine = affine.resolved(instance);
        }
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

    /// Emits the template's IR into `target` as the instruction at
    /// `instance`.
    /// Lifts the instruction this template is at `instance` into `target`,
    /// from the record alone.
    pub(crate) fn replay(
        &self,
        target: &mut LiftTarget<'_, 'static>,
        instance: &Instance,
    ) -> Result<Lifted, LiftError> {
        let mut construction = target.begin(instance.address, self.length)?;
        self.emit(&mut construction, instance)?;
        Ok(construction.commit()?)
    }

    fn emit(
        &self,
        construction: &mut Construction<'_, '_, 'static>,
        instance: &Instance,
    ) -> Result<(), LiftError> {
        let address = instance.address;
        // Everything the construction resolves against the module comes
        // before the emitter borrows it: the blocks of other instructions and
        // the callees, exactly as the emitter resolves its plan.
        let mut blocks: Vec<BlockId> =
            Vec::with_capacity(1 + self.blocks.len() + self.externals.len());
        blocks.push(construction.entry());
        let mut externals: Vec<Option<BlockId>> = vec![None; self.externals.len()];
        for &index in &self.external_order {
            let external = self.externals[index as usize].at(instance);
            externals[index as usize] = Some(construction.block_at(external)?);
        }
        let externals: Vec<BlockId> = externals
            .into_iter()
            .map(|b| b.expect("every external is ordered"))
            .collect();
        let mut callees = Vec::with_capacity(self.callees.len());
        for affine in &self.callees {
            callees.push(construction.callee_at(affine.at(instance))?);
        }

        let ctx = construction.context();
        let types: Vec<TypeId> = self.types.iter().map(|key| key.resolve(ctx)).collect();
        let literals: Vec<LocalValueId> = self
            .literals
            .iter()
            .map(|literal| {
                let ty = literal.ty.resolve(ctx);
                LocalValueId::Literal(ctx.shared.values.get_or_make_typed_literal(
                    literal.affine.at(instance),
                    ty,
                    literal.affine.size,
                ))
            })
            .collect();

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
        blocks.extend(externals);
        let spaces: Vec<LocalTempSpaceId> = self
            .temp_spaces
            .iter()
            .map(|space| {
                emitter
                    .push_temp_space(TempSpace::new(None, space.word_size, space.addr_size))
                    .local
            })
            .collect();
        let temps: Vec<LocalTempId> = self
            .temps
            .iter()
            .map(|temp| {
                emitter
                    .push_temp(Temp::new(
                        temp.address,
                        temp.size,
                        spaces[temp.space as usize],
                    ))
                    .local
            })
            .collect();

        let mut insns: Vec<InstructionId> = Vec::with_capacity(self.ops.len());
        let mut current = 0u32;
        for op in &self.ops {
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
            let name = op.name.as_deref().map(|name| Cow::Owned(name.to_owned()));
            let id = emitter
                .push_mnemonic_with_type_named(mnemonic, types[op.ty as usize], name)
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

/// What a miss probes: the instruction, how it is decoded and lifted, and
/// its shape.
#[derive(Clone, Copy)]
struct Probing<'a, 'spec> {
    lifter: &'a SleighLifter<'spec>,
    probes: &'a AtomicU64,
    decoder: &'a FixedDecoder<'spec>,
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
    shape: &'a Shape,
}

/// How a probe's lift differs from the captured one.
enum Mismatch {
    /// In its structure: the probe is not an instance of the same template.
    Structure(&'static str),
    /// In a value that moved by something other than the probe's delta.
    Value(&'static str),
}

impl Mismatch {
    fn reason(&self) -> &'static str {
        match self {
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
