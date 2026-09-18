//! Memoized lifting: the QCode of an encoding, replayed at another address.
//!
//! Machine code repeats. Over the superset of every byte offset of a binary
//! a few hundred thousand distinct encodings account for millions of lifts,
//! and a linear disassembly is more than half repeats too. Lowering one is
//! expensive twice over — SLEIGH expands the constructor's semantics, then
//! the builder types, interns and links every operation — and both costs are
//! per operation, so the vector forms whose per-lane macros run to hundreds
//! of operations pay them hundreds of times.
//!
//! A [`LiftCache`] remembers what an encoding lowered to, as a [`Template`]:
//! the instruction's blocks, temporaries and operations with every operand
//! renumbered relative to the template, and every address-derived constant
//! held relative to the address it was lifted at. Replaying a template into
//! a construction is a walk over that record — a block, a temporary or an
//! operation pushed per entry, with its type and constants resolved once per
//! replay rather than inferred per operation — so a hit costs the IR and
//! nothing around it.
//!
//! # Exactness
//!
//! SLEIGH is the only source of semantics here; the cache decides nothing
//! about an instruction. What it must decide is which constants of the
//! lowered IR depend on the address, and it measures that rather than
//! parsing the encoding. The p-code of an instruction is the same at every
//! address except for the constants the specification computes from
//! `inst_start` and `inst_next`, so on a miss the emitter records the
//! constants of the p-code it lowers, the same p-code is streamed again at
//! a distant probe address, and the two are compared constant by constant:
//! one equal in both is absolute, one that moved by exactly the distance
//! between the addresses is relative, and anything else marks the encoding
//! uncacheable, so it is lifted for real every time and never probed again.
//! Each literal of the IR is then classified by the p-code constant it
//! came from; see [`Classifier`]. A relative constant is therefore one the
//! specification computes as `address + k` in the width of the constant,
//! which is what every `inst_next`- and `inst_start`-derived value of a
//! 64-bit specification is. The probe distance is larger than 4 GiB so that
//! a value the specification truncates to 32 bits fails the comparison
//! instead of matching by accident. [`LiftCache::validating`] lifts every
//! hit for real as well and compares, for measuring that claim over a
//! corpus.
//!
//! The key is the encoding together with the decode context and the
//! control-flow lowering, so a session decoding under another context or
//! lowering calls differently never sees another's entries. The cache is
//! bound to one specification and refuses a lifter of another.
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
use rustc_hash::{FxHashMap as HashMap, FxHasher};
use sleigh::{CompiledSpec, ContextBytes, Instruction as Decoded, PcodePlan, SpecFingerprint};

use crate::{ConstSink, LiftError, SleighLifter, decode::FixedDecoder};

/// How far from the instruction's address the probe lift is made. Past
/// 4 GiB, so a 32-bit truncation of an address-derived value cannot pass for
/// an offset; odd in every byte, so no alignment of the two addresses
/// coincides.
const PROBE_DISTANCE: u64 = 0x1_0305_0709_0b0d;

/// The longest key: a flag byte, the context bytes and the instruction's.
const MAX_KEY: usize = 64;

const SHARDS: usize = 32;

/// Counters of a [`LiftCache`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Lifts served by replaying a template.
    pub hits: u64,
    /// Lifts of an encoding not yet in the cache; each was lifted for real
    /// and probed.
    pub misses: u64,
    /// Lifts of an encoding the probe found uncacheable; each was lifted
    /// for real. Counted per lift, not per encoding.
    pub uncacheable: u64,
    /// Hits whose validation lift disagreed with the template. Only counted
    /// when [validating](LiftCache::validating); each such hit was discarded,
    /// lifted for real, and its entry evicted.
    pub validation_failures: u64,
    /// Encodings held, as templates or as uncacheable.
    pub entries: usize,
}

enum Entry {
    Template(Arc<Template>),
    Uncacheable,
}

/// The answer of [`LiftCache::find`].
pub(crate) enum Lookup {
    /// The encoding's template.
    Hit(Arc<Template>),
    /// The encoding was probed and cannot be cached, or the cache does
    /// not apply: lift it, and do not remember it.
    Uncacheable,
    /// The encoding has not been seen: lift it through
    /// [`LiftCache::miss`].
    Unknown,
}

type Shard = RwLock<HashMap<Box<[u8]>, Entry>>;

/// A cache of lifted instructions keyed by their encoding. See the [module
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
    uncacheable: AtomicU64,
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
    /// The default number of encodings a cache holds before it stops
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
            uncacheable: AtomicU64::new(0),
            validation_failures: AtomicU64::new(0),
        }
    }

    /// Holds at most `entries` encodings; once full, further encodings are
    /// lifted for real and not remembered. Memory is bounded by the entries'
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
            uncacheable: self.uncacheable.load(Ordering::Relaxed),
            validation_failures: self.validation_failures.load(Ordering::Relaxed),
            entries: self.entries.load(Ordering::Relaxed),
        }
    }

    /// Forgets every encoding; the counters stay.
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

    fn shard(&self, key: &[u8]) -> &Shard {
        use std::hash::Hasher;
        let mut hasher = FxHasher::default();
        hasher.write(key);
        &self.shards[hasher.finish() as usize % SHARDS]
    }

    fn lookup(&self, key: &[u8]) -> Option<Option<Arc<Template>>> {
        let shard = self.shard(key).read().unwrap();
        match shard.get(key)? {
            Entry::Template(template) => Some(Some(Arc::clone(template))),
            Entry::Uncacheable => Some(None),
        }
    }

    fn insert(&self, key: &[u8], entry: Entry) {
        if self.entries.load(Ordering::Relaxed) >= self.capacity {
            return;
        }
        let mut shard = self.shard(key).write().unwrap();
        if let std::collections::hash_map::Entry::Vacant(slot) = shard.entry(Box::from(key)) {
            slot.insert(entry);
            self.entries.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn evict(&self, key: &[u8]) {
        if self.shard(key).write().unwrap().remove(key).is_some() {
            self.entries.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Lifts `instruction` into `target` through the cache: a replay when
    /// its encoding is known, a real lift — probed and remembered —
    /// otherwise. `decoder` decoded the instruction and decodes its probe.
    pub(crate) fn lower(
        &self,
        lifter: &SleighLifter<'_>,
        target: &mut LiftTarget<'_, 'static>,
        instruction: &Decoded<'_, '_>,
        decoder: &FixedDecoder<'_>,
        flat: bool,
    ) -> Result<Lifted, LiftError> {
        match self.find(lifter, instruction, decoder, flat)? {
            Lookup::Hit(template) => template.replay(target, instruction.address()),
            Lookup::Uncacheable => lifter.lower(target, instruction, flat),
            Lookup::Unknown => self.miss(lifter, target, instruction, decoder, flat),
        }
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
        let mut buffer = [0u8; MAX_KEY];
        let Some(key) = key(&mut buffer, flat, decoder.context(), instruction.bytes()) else {
            return Ok(Lookup::Uncacheable);
        };
        Ok(match self.lookup(key) {
            Some(Some(template)) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                if self.validating
                    && !self.validate(lifter, &template, instruction, decoder, flat)?
                {
                    self.validation_failures.fetch_add(1, Ordering::Relaxed);
                    self.evict(key);
                    return Ok(Lookup::Uncacheable);
                }
                Lookup::Hit(template)
            }
            Some(None) => {
                self.uncacheable.fetch_add(1, Ordering::Relaxed);
                Lookup::Uncacheable
            }
            None => Lookup::Unknown,
        })
    }

    /// [`find`](Self::find) before decoding: the template of the encoding at
    /// the front of `bytes`, and its length, when one of the `lengths` the
    /// caller has seen for such bytes keys a template. Valid encodings are
    /// prefix-free — a decoder reads only the bytes it needs, so no
    /// encoding is a proper prefix of another under one context — which is
    /// why at most one length can match and the shortest is tried first.
    /// A validating cache never answers, so that every hit is decoded and
    /// checked; nor does one whose entry says uncacheable, which the decoded
    /// path counts.
    pub(crate) fn find_undecoded(
        &self,
        lifter: &SleighLifter<'_>,
        decoder: &FixedDecoder<'_>,
        flat: bool,
        bytes: &[u8],
        lengths: u16,
    ) -> Result<Option<(Arc<Template>, usize)>, LiftError> {
        self.check(lifter)?;
        if self.validating {
            return Ok(None);
        }
        let mut buffer = [0u8; MAX_KEY];
        for length in 1..=15usize {
            if lengths & (1 << length) == 0 || bytes.len() < length {
                continue;
            }
            let Some(key) = key(&mut buffer, flat, decoder.context(), &bytes[..length]) else {
                continue;
            };
            if let Some(Some(template)) = self.lookup(key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(Some((template, length)));
            }
        }
        Ok(None)
    }

    /// Lifts an instruction [`find`](Self::find) did not know into `target`,
    /// probes it, and remembers it.
    pub(crate) fn miss(
        &self,
        lifter: &SleighLifter<'_>,
        target: &mut LiftTarget<'_, 'static>,
        instruction: &Decoded<'_, '_>,
        decoder: &FixedDecoder<'_>,
        flat: bool,
    ) -> Result<Lifted, LiftError> {
        self.misses.fetch_add(1, Ordering::Relaxed);
        let marks = Marks::of(target);
        let (lifted, consts) = lifter.lower_recording(target, instruction, flat)?;
        let entry = match Classifier::probe(instruction, decoder, &consts).and_then(|classifier| {
            Template::capture(
                target.context(),
                target.addresses(),
                marks,
                &lifted,
                &classifier,
            )
        }) {
            Ok(template) => Entry::Template(Arc::new(template)),
            Err(reason) => {
                debug_uncacheable(instruction, reason);
                Entry::Uncacheable
            }
        };
        let mut buffer = [0u8; MAX_KEY];
        if let Some(key) = key(&mut buffer, flat, decoder.context(), instruction.bytes()) {
            self.insert(key, entry);
        }
        Ok(lifted)
    }

    /// Whether a fresh lift of `instruction` agrees with the template.
    fn validate(
        &self,
        lifter: &SleighLifter<'_>,
        template: &Template,
        instruction: &Decoded<'_, '_>,
        decoder: &FixedDecoder<'_>,
        flat: bool,
    ) -> Result<bool, LiftError> {
        let address = instruction.address();
        with_probe_store(lifter, |store| {
            let decoded = decoder.decode(address, instruction.bytes())?;
            store.reset();
            let mut target = store.target()?;
            let marks = Marks::of(&target);
            let lifted = lifter.lower(&mut target, &decoded, flat)?;
            let fresh = Template::capture(
                target.context(),
                target.addresses(),
                marks,
                &lifted,
                &Classifier::absolute(),
            );
            Ok(fresh.is_ok_and(|fresh| template.instantiate(address).agrees_with(&fresh)))
        })
    }
}

/// Which constants of a lift the address made, measured on the p-code.
///
/// The p-code of an instruction is the same at every address except for
/// the constants the specification computes from `inst_start` and
/// `inst_next`, so streaming it at a second, distant address and comparing
/// constant by constant tells which ones moved — and by how much: exactly
/// the distance, in the constant's width, or the encoding is not cached.
/// A QCode literal is then classified by the p-code constant it came from:
/// the one of its value and width, or of its value alone when the builder
/// resized it, or the one it is the low bytes of when the builder took a
/// sub-range; a value that matches no constant is the builder's own and
/// does not move. A value both moving and fixed constants share cannot be
/// told apart once interned, and is refused.
struct Classifier {
    /// By value and width, then by value alone: whether the constant moves,
    /// or `None` when constants of that value disagree.
    by_value_and_size: HashMap<(u64, usize), Option<bool>>,
    by_value: HashMap<u64, Option<bool>>,
    /// The p-code constants and whether each moves, for sub-range matches.
    consts: Vec<(u64, usize, bool)>,
}

impl Classifier {
    /// A classifier for a lift compared against itself: nothing moves.
    fn absolute() -> Self {
        Self {
            by_value_and_size: HashMap::default(),
            by_value: HashMap::default(),
            consts: Vec::new(),
        }
    }

    /// Streams `instruction`'s p-code at the probe address and compares its
    /// constants with `consts`, those of the real lift, in stream order.
    fn probe(
        instruction: &Decoded<'_, '_>,
        decoder: &FixedDecoder<'_>,
        consts: &[(u64, usize)],
    ) -> Result<Self, &'static str> {
        let address = instruction.address();
        let probe_address = address.wrapping_add(PROBE_DISTANCE);
        let decoded = decoder
            .decode(probe_address, instruction.bytes())
            .map_err(|_| "the probe did not decode")?;
        let probed = decoded
            .pcode_ops_streamed(|_plan: &PcodePlan| ConstSink::default())
            .map_err(|_| "the probe did not stream")?;
        if probed.consts.len() != consts.len() {
            return Err("the probe's p-code differs in its constants");
        }
        let mut classifier = Self::absolute();
        for (&(value, size), &(moved, moved_size)) in consts.iter().zip(&probed.consts) {
            if size != moved_size {
                return Err("a constant changed width at the probe address");
            }
            let mask = Literal::mask(size);
            let relative = if moved == value {
                false
            } else if moved.wrapping_sub(value) & mask == PROBE_DISTANCE & mask {
                true
            } else {
                return Err("a constant is neither fixed nor moving with the address");
            };
            let note = |slot: &mut Option<bool>, first: bool| {
                if first {
                    *slot = Some(relative);
                } else if *slot != Some(relative) {
                    *slot = None;
                }
            };
            match classifier.by_value_and_size.entry((value, size)) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(Some(relative));
                }
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    note(slot.get_mut(), false);
                }
            }
            match classifier.by_value.entry(value) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(Some(relative));
                }
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    note(slot.get_mut(), false);
                }
            }
            classifier.consts.push((value, size, relative));
        }
        Ok(classifier)
    }

    /// Whether the literal `value` of `size` bytes moves with the address.
    fn classify(&self, value: u64, size: usize) -> Result<bool, &'static str> {
        let ambiguous = "a constant is both fixed and moving";
        if let Some(&relative) = self.by_value_and_size.get(&(value, size)) {
            return relative.ok_or(ambiguous);
        }
        if let Some(&relative) = self.by_value.get(&value) {
            return relative.ok_or(ambiguous);
        }
        // The low bytes of a moving constant move with it; higher bytes
        // carry, and are not cached. A sub-range of a fixed constant is
        // fixed.
        let mut found = None;
        for &(whole, whole_size, relative) in &self.consts {
            for offset in 0..whole_size.saturating_sub(size) + 1 {
                if size <= whole_size && (whole >> (8 * offset)) & Literal::mask(size) == value {
                    let this = match (relative, offset) {
                        (false, _) => false,
                        (true, 0) => true,
                        (true, _) => return Err("the high bytes of a moving constant"),
                    };
                    match found {
                        None => found = Some(this),
                        Some(other) if other != this => return Err(ambiguous),
                        _ => {}
                    }
                }
            }
        }
        Ok(found.unwrap_or(false))
    }
}

/// The lookup key of an instruction: the lowering flag, the decode context
/// and the bytes. `None` when it does not fit, which no real instruction
/// causes.
fn key<'k>(
    buffer: &'k mut [u8; MAX_KEY],
    flat: bool,
    context: &ContextBytes,
    bytes: &[u8],
) -> Option<&'k [u8]> {
    let context = context.as_bytes();
    let len = 1 + context.len() + bytes.len();
    if len > MAX_KEY {
        return None;
    }
    buffer[0] = flat as u8;
    buffer[1..1 + context.len()].copy_from_slice(context);
    buffer[1 + context.len()..len].copy_from_slice(bytes);
    Some(&buffer[..len])
}

/// `QCODE_CACHE_DEBUG=1`: reports every encoding found uncacheable and why.
fn debug_uncacheable(instruction: &Decoded<'_, '_>, reason: &str) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ENABLED.get_or_init(|| std::env::var_os("QCODE_CACHE_DEBUG").is_some()) {
        eprintln!(
            "lift cache: {} ({}) is uncacheable: {reason}",
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
    /// The value at the template's base address.
    value: u64,
    size: usize,
    /// The literal's type: a comparison's `false` is a `bool`, not an `i8`.
    ty: TypeKey,
    /// Whether the value moves with the address.
    relative: bool,
}

impl Literal {
    fn mask(size: usize) -> u64 {
        if size >= 8 {
            u64::MAX
        } else {
            (1u64 << (8 * size)) - 1
        }
    }

    fn at(self, delta: u64) -> u64 {
        if self.relative {
            self.value.wrapping_add(delta) & Self::mask(self.size)
        } else {
            self.value
        }
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
/// `k`th callee. Varnodes are the architecture's and stay as they are.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Op {
    block: u32,
    ty: u32,
    mnemonic: Mnemonic,
    name: Option<Box<str>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CalleeKey {
    Address(u64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExitTarget {
    Fallthrough,
    Branch(u64),
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
    /// Relative to the template's base.
    Address(u64),
    Named(Box<str>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExitRecord {
    site: u32,
    arm: ExitArm,
    target: ExitTarget,
}

/// The lowered IR of one encoding, relative to the address it was lifted
/// at. See the [module documentation](self).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Template {
    /// The address the values were captured at.
    base: u64,
    length: usize,
    /// The instruction's own blocks after the entry, with how they were
    /// named.
    blocks: Vec<BlockName>,
    /// Addresses of the blocks of other instructions the IR names, relative
    /// to the base and in address order, plus the fall-through, which the
    /// emitter resolves whether or not the IR names it.
    externals: Vec<u64>,
    /// The order the emitter resolves the externals in, as indices into
    /// `externals` — the order their placeholders are made in when none
    /// exists yet, which a replay keeps so it issues the same block ids.
    external_order: Vec<u32>,
    callees: Vec<CalleeKey>,
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
    insns: HashMap<LocalInsnId, u32>,
    temps: HashMap<LocalTempId, u32>,
    literals: HashMap<LiteralId, u32>,
    blocks: HashMap<LocalBlockId, u32>,
    types: HashMap<TypeId, u32>,
}

impl Template {
    /// Records the IR `lifted` describes in `ctx`, or why a template cannot
    /// hold it.
    fn capture(
        ctx: &Context<'static>,
        addresses: &AddressIndex,
        marks: Marks,
        lifted: &Lifted,
        classifier: &Classifier,
    ) -> Result<Self, &'static str> {
        let base = lifted.address();
        let func = lifted.entry().func;
        let body = ctx.body(func);
        let view = ModuleView::new(ctx);
        let mut numbering = Numbering::default();
        let mut template = Self {
            base,
            length: lifted.length(),
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
            numbering.blocks.insert(block.local, index as u32);
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
        for (position, (insn, _)) in order.iter().enumerate() {
            numbering.insns.insert(*insn, position as u32);
        }

        // The temporary spaces and temporaries the lift appended, in order.
        for space in marks.temp_spaces..body.temp_space_count() {
            let space = body.temp_space(TempSpaceId::new(func, LocalTempSpaceId::from(space)));
            if space.name().is_some() {
                return Err("a named temporary space");
            }
            template.temp_spaces.push(TempSpaceKey {
                word_size: space.word_size(),
                addr_size: space.addr_size(),
            });
        }
        for temp in marks.temps..body.temp_count() {
            let local = LocalTempId::from(temp);
            numbering.temps.insert(local, (temp - marks.temps) as u32);
            let temp = TempRef::new(view, TempId::new(func, local));
            if temp.name().is_some() || temp.label().is_some() {
                return Err("a named or labeled temporary");
            }
            let space = usize::from(temp.space().id.local);
            if space < marks.temp_spaces {
                return Err("a temporary in an earlier instruction's space");
            }
            template.temps.push(TempKey {
                space: (space - marks.temp_spaces) as u32,
                address: temp.address(),
                size: temp.size(),
            });
        }

        // The blocks of other instructions, in the order their placeholders
        // were made — their id order.
        let mut external_blocks: Vec<LocalBlockId> = Vec::new();
        let mut external = |block: LocalBlockId| {
            if !numbering.blocks.contains_key(&block) && !external_blocks.contains(&block) {
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
        let next = addresses
            .block_at(base.wrapping_add(lifted.length() as u64))
            .ok_or("the fall-through has no block")?;
        if next.func != func {
            return Err("the fall-through belongs to another function");
        }
        external(next.local);
        let mut by_address: Vec<(u64, LocalBlockId)> = Vec::with_capacity(external_blocks.len());
        for &block in &external_blocks {
            let address = ctx
                .block(BlockId::new(func, block))
                .address()
                .ok_or("a block of another instruction has no address")?;
            by_address.push((address.wrapping_sub(base), block));
        }
        by_address.sort_unstable();
        template.externals = by_address.iter().map(|(relative, _)| *relative).collect();
        let internal = lifted.blocks().len() as u32;
        for (index, (_, block)) in by_address.iter().enumerate() {
            numbering.blocks.insert(*block, internal + index as u32);
        }
        let mut creation: Vec<(LocalBlockId, u32)> = by_address
            .iter()
            .enumerate()
            .map(|(index, (_, block))| (*block, index as u32))
            .collect();
        creation.sort_unstable_by_key(|(block, _)| usize::from(*block));
        template.external_order = creation.iter().map(|(_, index)| *index).collect();

        for (insn, block) in order {
            let id = InstructionId::new(func, insn);
            let reference = Instruction::from_id(ctx, id);
            let mut failed = None;
            let mut map = |operand: LocalValueId| -> LocalValueId {
                match operand {
                    LocalValueId::Literal(literal) => {
                        let next = numbering.literals.len() as u32;
                        let index = match numbering.literals.get(&literal) {
                            Some(&index) => index,
                            None => {
                                let value = ctx.get_literal_value(literal);
                                let type_id = ctx.shared.values.literals[literal].type_id;
                                let Some(ty) = TypeKey::of(ctx, type_id) else {
                                    failed = Some("a constant of a type a template cannot hold");
                                    return operand;
                                };
                                let size = ctx.shared.types.size_of(type_id);
                                let relative = match classifier.classify(value, size) {
                                    Ok(relative) => relative,
                                    Err(reason) => {
                                        failed = Some(reason);
                                        return operand;
                                    }
                                };
                                template.literals.push(Literal {
                                    value,
                                    size,
                                    ty,
                                    relative,
                                });
                                numbering.literals.insert(literal, next);
                                next
                            }
                        };
                        LocalValueId::Literal(LiteralId::from(index as usize))
                    }
                    LocalValueId::Instruction(other) => match numbering.insns.get(&other) {
                        // A use may only name an operation issued before it.
                        Some(&index) if index < numbering.insns[&insn] => {
                            LocalValueId::Instruction(LocalInsnId::from(index as usize))
                        }
                        _ => {
                            failed = Some("an operand names a later operation");
                            operand
                        }
                    },
                    LocalValueId::Temp(temp) => match numbering.temps.get(&temp) {
                        Some(&index) => LocalValueId::Temp(LocalTempId::from(index as usize)),
                        None => {
                            failed = Some("an operand names an earlier instruction's temporary");
                            operand
                        }
                    },
                    LocalValueId::Varnode(_) => operand,
                    _ => {
                        failed = Some("an operand of a kind a template cannot hold");
                        operand
                    }
                }
            };
            let mut mnemonic = reference.mnemonic().clone().map_operands(&mut map);
            if let Some(reason) = failed {
                return Err(reason);
            }
            let space_index = |space: &mut LocalMemorySpaceId| -> Result<(), &'static str> {
                if let LocalMemorySpaceId::Temp(local) = space {
                    let raw = usize::from(*local);
                    if raw < marks.temp_spaces {
                        return Err("an operation in an earlier instruction's space");
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
            let block_index =
                |target: LocalBlockId| LocalBlockId::from(numbering.blocks[&target] as usize);
            match &mut mnemonic {
                Mnemonic::Branch(branch) => branch.target = block_index(branch.target),
                Mnemonic::CBranch(cbranch) => {
                    cbranch.success_block = block_index(cbranch.success_block);
                    cbranch.failure_block = block_index(cbranch.failure_block);
                }
                Mnemonic::Switch(_) => return Err("a switch"),
                Mnemonic::Call(call) => call.target = template.callee_key(ctx, call.target)?,
                Mnemonic::TailCall(call) => call.target = template.callee_key(ctx, call.target)?,
                Mnemonic::Apply(_) => return Err("an apply"),
                _ => {}
            }
            let type_id = reference.type_id();
            let next = numbering.types.len() as u32;
            let ty = match numbering.types.get(&type_id) {
                Some(&index) => index,
                None => {
                    template
                        .types
                        .push(TypeKey::of(ctx, type_id).ok_or("a type a template cannot hold")?);
                    numbering.types.insert(type_id, next);
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
            let site = *numbering
                .insns
                .get(&exit.site().local)
                .ok_or("an exit site outside the instruction")?;
            let continuation = |continuation: Continuation| -> Result<Option<u32>, &'static str> {
                match continuation {
                    Continuation::Next => Ok(None),
                    Continuation::Block(block) => numbering
                        .blocks
                        .get(&block.local)
                        .copied()
                        .map(Some)
                        .ok_or("a continuation outside the instruction"),
                }
            };
            let target = match exit.kind() {
                ExitKind::Fallthrough => ExitTarget::Fallthrough,
                ExitKind::Branch { target } => ExitTarget::Branch(target.wrapping_sub(base)),
                ExitKind::BranchInd => ExitTarget::BranchInd,
                ExitKind::Call {
                    callee,
                    continuation: c,
                } => ExitTarget::Call {
                    callee: match callee {
                        CallTarget::Address(address) => {
                            CallKey::Address(address.wrapping_sub(base))
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

    /// A result naming the template's own keys, for views over the template
    /// at `address` in `func`: block `k` is the template's `k`th block —
    /// the instruction's own, then the external ones — and instruction `k`
    /// its `k`th operation.
    pub(crate) fn lifted_at(&self, address: u64, func: FunctionId) -> Lifted {
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
                let kind = match &exit.target {
                    ExitTarget::Fallthrough => ExitKind::Fallthrough,
                    ExitTarget::Branch(relative) => ExitKind::Branch {
                        target: address.wrapping_add(*relative),
                    },
                    ExitTarget::BranchInd => ExitKind::BranchInd,
                    ExitTarget::Call {
                        callee,
                        continuation: c,
                    } => ExitKind::Call {
                        callee: match callee {
                            CallKey::Address(relative) => {
                                CallTarget::Address(address.wrapping_add(*relative))
                            }
                            CallKey::Named(name) => CallTarget::Named(name.clone()),
                        },
                        continuation: continuation(*c),
                    },
                    ExitTarget::CallInd { continuation: c } => ExitKind::CallInd {
                        continuation: continuation(*c),
                    },
                    ExitTarget::Return => ExitKind::Return,
                };
                Exit::new(
                    InstructionId::new(func, LocalInsnId::from(exit.site as usize)),
                    exit.arm,
                    kind,
                )
            })
            .collect();
        Lifted::new(address, self.length, block(0), blocks, exits)
    }

    /// The address of the template's `k`th block at `address`: the
    /// instruction's own for the entry, none for its other blocks, another
    /// instruction's for an external one.
    pub(crate) fn block_address(&self, k: usize, address: u64) -> Option<u64> {
        if k == 0 {
            Some(address)
        } else if k <= self.blocks.len() {
            None
        } else {
            self.externals
                .get(k - 1 - self.blocks.len())
                .map(|relative| address.wrapping_add(*relative))
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

    /// The `k`th literal's value and width at `address`.
    pub(crate) fn literal_at(&self, k: usize, address: u64) -> (u64, usize) {
        let literal = self.literals[k];
        (literal.at(address.wrapping_sub(self.base)), literal.size)
    }

    /// The address of the template's `slot`th callee at `address`.
    pub(crate) fn callee_address(&self, slot: usize, address: u64) -> Option<u64> {
        self.callees
            .get(slot)
            .map(|CalleeKey::Address(relative)| address.wrapping_add(*relative))
    }

    /// The template's slot for a callee of the captured IR: a function the
    /// construction resolved by address.
    fn callee_key(
        &mut self,
        ctx: &Context<'static>,
        callee: Callee,
    ) -> Result<Callee, &'static str> {
        let Callee::Real(function) = callee else {
            return Err("a callee still minted");
        };
        let address = FunctionBody::from_id(ctx, function)
            .address()
            .ok_or("a callee without an address")?;
        let key = CalleeKey::Address(address.wrapping_sub(self.base));
        let slot = match self.callees.iter().position(|k| *k == key) {
            Some(slot) => slot,
            None => {
                self.callees.push(key);
                self.callees.len() - 1
            }
        };
        Ok(Callee::Minted(slot as u32))
    }

    /// Whether two templates describe the same IR apart from their literal
    /// values and debug names, or what differs.
    fn agrees_with_except_literals(&self, other: &Self) -> Result<(), &'static str> {
        if self.length != other.length {
            return Err("the two lifts differ in length");
        }
        if self.blocks.len() != other.blocks.len() {
            return Err("the two lifts differ in their blocks");
        }
        // Not the order the placeholders were made in: a block that existed
        // before one lift and not before the other was made in one and found
        // in the other, and the replay makes what it does not find.
        if self.externals != other.externals {
            return Err("the two lifts differ in the blocks they reach");
        }
        if self.callees != other.callees {
            return Err("the two lifts differ in their callees");
        }
        if self.temp_spaces != other.temp_spaces || self.temps != other.temps {
            return Err("the two lifts differ in their temporaries");
        }
        if self.types != other.types {
            return Err("the two lifts differ in their types");
        }
        if self.exits != other.exits {
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

    /// Whether two templates captured at the same address describe the same
    /// IR, debug names aside.
    fn agrees_with(&self, other: &Self) -> bool {
        self.base == other.base
            && self.agrees_with_except_literals(other).is_ok()
            && self.literals.len() == other.literals.len()
            && self
                .literals
                .iter()
                .zip(&other.literals)
                .all(|(a, b)| a.value == b.value && a.size == b.size && a.ty == b.ty)
    }

    /// This template with its values as they are at `address`.
    fn instantiate(&self, address: u64) -> Self {
        let delta = address.wrapping_sub(self.base);
        let mut instance = self.clone();
        instance.base = address;
        for literal in &mut instance.literals {
            literal.value = literal.at(delta);
            literal.relative = false;
        }
        instance
    }

    /// Emits the template's IR into `target` as the instruction at
    /// `address`.
    fn replay(
        &self,
        target: &mut LiftTarget<'_, 'static>,
        address: u64,
    ) -> Result<Lifted, LiftError> {
        let delta = address.wrapping_sub(self.base);
        let mut construction = target.begin(address, self.length)?;
        self.emit(&mut construction, address, delta)?;
        Ok(construction.commit()?)
    }

    fn emit(
        &self,
        construction: &mut Construction<'_, '_, 'static>,
        address: u64,
        delta: u64,
    ) -> Result<(), LiftError> {
        // Everything the construction resolves against the module comes
        // before the emitter borrows it: the blocks of other instructions and
        // the callees, exactly as the emitter resolves its plan.
        let mut blocks: Vec<BlockId> =
            Vec::with_capacity(1 + self.blocks.len() + self.externals.len());
        blocks.push(construction.entry());
        let mut externals: Vec<Option<BlockId>> = vec![None; self.externals.len()];
        for &index in &self.external_order {
            let relative = self.externals[index as usize];
            externals[index as usize] =
                Some(construction.block_at(address.wrapping_add(relative))?);
        }
        let externals: Vec<BlockId> = externals
            .into_iter()
            .map(|b| b.expect("every external is ordered"))
            .collect();
        let mut callees = Vec::with_capacity(self.callees.len());
        for key in &self.callees {
            let CalleeKey::Address(relative) = key;
            callees.push(construction.callee_at(address.wrapping_add(*relative))?);
        }

        let ctx = construction.context();
        let types: Vec<TypeId> = self.types.iter().map(|key| key.resolve(ctx)).collect();
        let literals: Vec<LocalValueId> = self
            .literals
            .iter()
            .map(|literal| {
                let ty = literal.ty.resolve(ctx);
                LocalValueId::Literal(ctx.shared.values.get_or_make_typed_literal(
                    literal.at(delta),
                    ty,
                    literal.size,
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
            let kind = match &exit.target {
                ExitTarget::Fallthrough => ExitKind::Fallthrough,
                ExitTarget::Branch(relative) => ExitKind::Branch {
                    target: address.wrapping_add(*relative),
                },
                ExitTarget::BranchInd => ExitKind::BranchInd,
                ExitTarget::Call {
                    callee,
                    continuation: c,
                } => ExitKind::Call {
                    callee: match callee {
                        CallKey::Address(relative) => {
                            CallTarget::Address(address.wrapping_add(*relative))
                        }
                        CallKey::Named(name) => CallTarget::Named(name.clone()),
                    },
                    continuation: continuation(*c),
                },
                ExitTarget::CallInd { continuation: c } => ExitKind::CallInd {
                    continuation: continuation(*c),
                },
                ExitTarget::Return => ExitKind::Return,
            };
            emitter.exit(insns[exit.site as usize], exit.arm, kind);
        }
        Ok(())
    }
}
