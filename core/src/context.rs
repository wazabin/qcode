//! The central arena for all IR state: [`Context`].

use std::{borrow::Cow, fmt::Display};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{
    assumption::{Certainty, KnownContradiction, PassName, Proposition, Truth, Violation},
    error::{Error, ErrorTy, Result},
    pass_scope,
    space::{Space, SpaceId},
    types::TypeManager,
    value::{
        BasicBlock, Function, FunctionId, FunctionRef, Instruction, ValueId,
        block::{BlockId, BlockMutRef, BlockRef, EdgeData, EdgeId, EdgeMutRef, EdgeRef},
        insn::{InstructionId, InstructionRef, Mnemonic, PCodeOpId},
        literal::{LiteralId, LiteralRef},
        registry::ValueRegistry,
        varnode::{Varnode, VarnodeId, VarnodeRef, register::RegisterId},
    },
};
use jstd::{
    graph::Graph,
    registry::{self, Registry},
};

/// The central arena that owns all IR state.
///
/// `Context` is the single source of truth for every value (instructions,
/// varnodes, literals, blocks, functions), every memory space, and the
/// bidirectional maps that let you look up values by name or by machine
/// address.
///
/// # Usage
///
/// Create a context with [`Context::new`] and pass `&mut` references to a
/// [`Builder`](crate::builder::Builder) when constructing IR, or to analysis
/// passes when transforming it.
///
/// ```rust,ignore
/// use qcode_core::context::Context;
///
/// let mut ctx = Context::new();
/// // ctx.default_space is the RAM space created by new()
/// ```
///
/// # Lifetime parameter `'str`
///
/// The `'str` lifetime is the lifetime of interned string data used for names
/// and space identifiers. When names are owned (e.g. generated names), they
/// are stored as `Cow::Owned`; when they are borrowed from source data they are
/// `Cow::Borrowed` and must outlive the context.
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct Context<'str> {
    pub default_space: SpaceId,

    /// A mapping of space ids to their corresponding [`Space`]s.
    pub(crate) spaces: Registry<SpaceId, Space>,

    /// A mapping of pcode ops to their names
    pub pcode_ops: Registry<PCodeOpId, Box<str>>,

    /// A mapping of names to spaces
    pub named_spaces: HashMap<Box<str>, SpaceId>,

    /// Global reverse name map for module-scoped values (functions, varnodes,
    /// spaces, p-code ops, byte blobs), used to keep their name hints unique and
    /// resolve them by name. Block/instruction/param names are **not** here — they
    /// live in each [`Function`](crate::value::Function)'s own [`NameTable`], so
    /// those namespaces stay independent across functions (see [`NameTable`]).
    name_map: NameTable<'str>,

    /// Reverse mapping from addresses to value IDs
    address_map: HashMap<u64, ValueId>,

    /// A mapping of register IDs to their corresponding value IDs
    pub registers: HashMap<RegisterId, VarnodeId>,

    /// The values available in the context, indexed by their ID
    pub values: ValueRegistry<'str>,

    /// Type registry: owns all [`Type`] objects and hands out [`TypeId`]s.
    pub types: TypeManager,

    /// Initialized memory of the loaded binary (read-only data, code, …),
    /// populated by the lifter. Lets analysis passes read constants such as
    /// jump-table entries straight out of `.rodata` without the loader-side
    /// `BinaryFormat`. Empty for synthetically-built contexts.
    pub memory_image: crate::memory_image::MemoryImage,

    /// The binary format's primary entrypoint, when the loader supplied one.
    /// Analysis passes use this for narrow loader-shaped recognizers such as
    /// CRT startup recovery without depending on a binary-format crate.
    #[serde(default)]
    primary_entrypoint: Option<u64>,

    /// Code addresses discovered by lifting or analysis but not yet lifted.
    /// `qcode_analysis` cannot call the lifter (one-way crate dependency), so
    /// passes that resolve new targets (e.g. the jump-table pass) record them
    /// here; the `lift_new_addresses` pass drains them and lifts the code into
    /// the (clean) IR. Rides through clone (so it survives checkpoint+replay
    /// rounds) and serialization.
    #[serde(default)]
    discoveries: crate::discovery::DiscoveryQueue,

    /// The operating system of the loaded binary, stamped by the loader from the
    /// binary format (PE → Windows, ELF → Linux). Platform-gated passes — e.g.
    /// TEB seeding, which only applies to Windows — read it. `Unknown` for
    /// synthetic contexts.
    #[serde(default)]
    target_os: TargetOs,

    /// Entry addresses of functions the user asked to skip optimizing (via the
    /// `--ignore` flag). Such functions are still lifted, but every per-function
    /// analysis pass skips them. Rides through clone so it survives the
    /// checkpoint+replay rounds, and through serialization so a saved session
    /// keeps honoring the request.
    #[serde(default)]
    ignored_functions: HashSet<u64>,
}

/// The operating system of a loaded binary, inferred from its container format.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TargetOs {
    #[default]
    Unknown,
    Windows,
    Linux,
}

impl<'str> Context<'str> {
    /// Creates a new, empty context with a single default RAM space.
    ///
    /// The default space has a word size of 1 byte and an address size of 8
    /// bytes (suitable for 64-bit architectures). Its [`SpaceId`] is stored in
    /// [`Context::default_space`].
    pub fn new() -> Self {
        let mut ctx = Self::default();
        // SPACE_CONST = SpaceId(0): virtual space for constant/immediate values
        ctx.spaces.push(Space::new(Some("const"), 1, 8));
        // default RAM space (SpaceId(1)); temp spaces start at SpaceId(2)
        let default_space = Space::new(Some("ram"), 1, 8);
        ctx.default_space = ctx.spaces.push(default_space);
        ctx
    }

    /// Returns the [`SpaceId`] for the named space, or `None` if it has not
    /// been registered.
    pub fn try_get_space(&self, name: &str) -> Option<SpaceId> {
        self.named_spaces.get(name).copied()
    }

    /// Resolve a space by name for textual lowering: an already-registered named
    /// space, the default space when its name matches (the default `ram` space is
    /// not in `named_spaces`), or a freshly-registered RAM space otherwise. Used
    /// by the canonical `load(space:size, ptr)` / `store(...)` lowering.
    pub fn get_or_make_named_space(&mut self, name: &str) -> SpaceId {
        if let Some(id) = self.try_get_space(name) {
            return id;
        }
        // Named temp spaces (e.g. the per-varnode spaces minted by
        // `make_named_temp_space`) carry a name but are not in `named_spaces`, so
        // scan the registry by name before minting a fresh one. This keeps a
        // canonical `load(V0:4, V0)` bound to the same space as varnode `V0`.
        if let Some(found) = self
            .spaces
            .iter()
            .find(|s| s.name.as_deref() == Some(name))
            .map(|s| s.id)
        {
            return found;
        }
        let default = &self.spaces[self.default_space];
        let space = Space::new(Some(name), default.word_size, default.addr_size);
        self.add_space(space)
    }

    /// Creates a new temporary address space and returns its ID.
    pub fn make_temp_space(&mut self) -> SpaceId {
        let default_space = &self.spaces[self.default_space];
        let mut space = Space::new(None, default_space.word_size, default_space.addr_size);
        space.ty = crate::space::SpaceType::Temporary;
        self.spaces.push(space)
    }

    /// Adds a space to the context, registering its name, and returns its ID.
    pub fn add_space(&mut self, space: Space) -> SpaceId {
        let name_key: Option<Box<str>> = space.name.clone();
        let id = self.spaces.push(space);
        if let Some(name) = name_key {
            self.named_spaces.insert(name, id);
        }
        id
    }

    /// Returns the number of spaces registered in this context.
    pub fn space_count(&self) -> usize {
        self.spaces.len()
    }

    pub fn set_primary_entrypoint(&mut self, entrypoint: Option<u64>) {
        self.primary_entrypoint = entrypoint;
    }

    pub fn primary_entrypoint(&self) -> Option<u64> {
        self.primary_entrypoint
    }

    /// Record the set of function entry addresses whose optimization the user
    /// asked to skip (`--ignore`). Per-function passes consult
    /// [`Context::is_function_ignored`] and skip these functions.
    pub fn set_ignored_functions(&mut self, addrs: HashSet<u64>) {
        self.ignored_functions = addrs;
    }

    /// The function entry addresses whose optimization is being skipped.
    pub fn ignored_functions(&self) -> &HashSet<u64> {
        &self.ignored_functions
    }

    /// Whether the function at `addr` was marked ignored (`--ignore`). A `None`
    /// address (synthetic functions with no entry) is never ignored.
    pub fn is_function_ignored(&self, addr: Option<u64>) -> bool {
        addr.is_some_and(|a| self.ignored_functions.contains(&a))
    }

    /// Records the loaded binary's operating system (set by the loader from the
    /// container format).
    pub fn set_target_os(&mut self, os: TargetOs) {
        self.target_os = os;
    }

    /// The loaded binary's operating system, or [`TargetOs::Unknown`].
    pub fn target_os(&self) -> TargetOs {
        self.target_os
    }

    /// Replaces the spaces registry wholesale. Intended for initialization from a pre-built spec.
    pub fn load_spaces(&mut self, spaces: registry::Registry<SpaceId, Space>) {
        self.spaces = spaces;
    }

    /// Creates a new named temporary address space and returns its ID.
    pub fn make_named_temp_space(&mut self, name: impl Into<Box<str>>) -> SpaceId {
        let default_space = &self.spaces[self.default_space];
        let (word_size, addr_size) = (default_space.word_size, default_space.addr_size);
        self.spaces.push(Space {
            name: Some(name.into()),
            word_size,
            addr_size,
            ty: crate::space::SpaceType::Temporary,
        })
    }

    /// Read `n` bytes of initialized binary memory at virtual address `addr`,
    /// or `None` if any byte is unmapped. See [`MemoryImage::read_bytes`].
    ///
    /// [`MemoryImage::read_bytes`]: crate::memory_image::MemoryImage::read_bytes
    pub fn read_bytes(&self, addr: u64, n: usize) -> Option<Vec<u8>> {
        self.memory_image.read_bytes(addr, n)
    }

    /// Read a little-endian unsigned integer of `size` bytes from initialized
    /// binary memory at `addr`. See [`MemoryImage::read_uint`].
    ///
    /// [`MemoryImage::read_uint`]: crate::memory_image::MemoryImage::read_uint
    pub fn read_uint(&self, addr: u64, size: usize) -> Option<u64> {
        self.memory_image.read_uint(addr, size)
    }

    /// True if `addr` lies in an executable region of the loaded binary.
    pub fn is_executable_addr(&self, addr: u64) -> bool {
        self.memory_image.is_executable(addr)
    }

    /// True only if `addr` is in a region *known* to be writable (protections
    /// established and the segment writable). Passes that fold a value out of
    /// initialized memory use this to refuse mutable memory — e.g. a GOT slot the
    /// dynamic linker overwrites at load time, whose file bytes are the lazy PLT
    /// resolver stub, not the real target. See
    /// [`MemoryImage::is_known_writable`](crate::memory_image::MemoryImage::is_known_writable).
    pub fn is_known_writable_addr(&self, addr: u64) -> bool {
        self.memory_image.is_known_writable(addr)
    }

    /// Mark the binary's memory protections as established (the
    /// `memory_protections` pass has run), so executability checks narrow from the
    /// permissive default to the real per-segment flags.
    pub fn mark_protections_known(&mut self) {
        self.memory_image.mark_protections_known();
    }

    /// The lifter's pre-decode executability gate, modeling executability as a
    /// [`Proposition::ExecutableMemory`]. Returns whether `addr` should be lifted:
    ///
    /// - protections not yet established → optimistic default r/x (`true`); the
    ///   memory image may be empty (the lifter reads bytes from the binary format,
    ///   not the image), so it is *not* consulted in this case;
    /// - protections known and the region is executable → `true`;
    /// - protections known and the region is non-executable (or unmapped) →
    ///   `false` (skip), recording the proven fact
    ///   `ExecutableMemory{start, end} = false` for the whole containing segment
    ///   of a mapped-but-non-executable target.
    ///
    /// The proposition is keyed by the containing segment, not the individual
    /// address, so repeated skips in the same non-executable region collapse to a
    /// single truth-map entry rather than one per byte.
    ///
    /// A *known* value for the containing region (a proven fact, or a user
    /// override seeded as known) wins over the raw segment flags, so the user can
    /// force a region executable or non-executable from the Assumptions panel.
    pub fn assume_executable(&mut self, addr: u64) -> bool {
        let bounds = self.memory_image.segment_bounds(addr);
        if let Some((start, end)) = bounds
            && let Some(known) = self.known(Proposition::ExecutableMemory { start, end })
        {
            return known;
        }
        if !self.memory_image.protections_known() || self.memory_image.is_executable(addr) {
            return true;
        }
        if let Some((start, end)) = bounds {
            self.set_known(Proposition::ExecutableMemory { start, end }, false);
        }
        false
    }

    /// Record a discovered code address (typed: a new function or a block within
    /// an existing function) for the `lift_new_addresses` pass to lift.
    pub fn discover(&mut self, discovery: crate::discovery::Discovery) -> bool {
        self.discoveries.insert(discovery)
    }

    /// Convenience for the common case: the jump-table pass resolved a branch in
    /// the function at `func_entry` to `target`, a block within that function.
    ///
    /// `source_block` is the address of the block ending in the indirect branch,
    /// so the lifter can connect a real CFG edge from it to `target` in the clean
    /// IR (the resolution is otherwise only reflected in the disposable optimized
    /// clone, which would leave the target an orphan that function-splitting and
    /// reachability cannot follow).
    pub fn discover_code(&mut self, func_entry: u64, source_block: u64, target: u64) {
        self.discoveries.insert(
            crate::discovery::Discovery::block(target, func_entry)
                .with_edge_kind(crate::discovery::EdgeKind::JumpTableTarget)
                .from_block_addr(source_block)
                .with_provenance(crate::discovery::DiscoveryProvenance::Optimization {
                    pass: "handle_jump_tables".to_string(),
                    assumption: None,
                }),
        );
    }

    /// Remove and return every pending discovery, leaving the queue empty.
    pub fn drain_discoveries(&mut self) -> Vec<crate::discovery::Discovery> {
        self.discoveries.drain()
    }

    /// Iterate pending discoveries without consuming them.
    pub fn discoveries(&self) -> impl Iterator<Item = &crate::discovery::Discovery> + '_ {
        self.discoveries.iter()
    }

    /// True if there are no pending discoveries.
    pub fn has_no_discoveries(&self) -> bool {
        self.discoveries.is_empty()
    }

    /// Every code address lifted in this context, as portable [`CodeSeed`]s. Used
    /// to export a "code map" that pre-seeds a later run of the same binary.
    ///
    /// [`CodeSeed`]: crate::discovery::CodeSeed
    pub fn lifted_code_seeds(&self) -> Vec<crate::discovery::CodeSeed> {
        self.discoveries.lifted_seeds()
    }

    /// Enqueue exported [`CodeSeed`]s as pending discoveries so the lifter reaches
    /// them in its first pass. Call before lifting begins; seeds whose key already
    /// has a terminal outcome are ignored by the queue.
    ///
    /// [`CodeSeed`]: crate::discovery::CodeSeed
    pub fn seed_code(&mut self, seeds: impl IntoIterator<Item = crate::discovery::CodeSeed>) {
        for seed in seeds {
            self.discoveries.insert(seed.into_discovery());
        }
    }

    pub fn mark_discovery_lifted(&mut self, key: crate::discovery::DiscoveryKey) {
        self.discoveries.mark_lifted(key);
    }

    pub fn mark_discovery_failed(
        &mut self,
        key: crate::discovery::DiscoveryKey,
        reason: impl Into<String>,
    ) {
        self.discoveries.mark_failed(key, reason);
    }

    pub fn mark_discovery_skipped(
        &mut self,
        key: crate::discovery::DiscoveryKey,
        reason: impl Into<String>,
    ) {
        self.discoveries.mark_skipped(key, reason);
    }

    /// Returns the [`BlockId`] for a block at `addr`, creating one if needed.
    ///
    /// The newly created block is named after the address in hex and registered
    /// in the address map.
    pub fn get_or_make_block(&mut self, addr: u64, func: FunctionId) -> BlockId {
        match BasicBlock::from_addr(self, addr) {
            Some(block) => block.id,
            None => BasicBlock::make(self, func).with_address(addr).id,
        }
    }

    /// The forced rendering mode for a `Bytes` blob, or
    /// [`BytesDisplay::Auto`](crate::value::BytesDisplay::Auto) if unset.
    pub fn bytes_display(&self, id: crate::value::BytesId) -> crate::value::BytesDisplay {
        self.values
            .bytes_display
            .get(&id)
            .copied()
            .unwrap_or_default()
    }

    /// Force how a `Bytes` blob renders as a `b"..."` literal everywhere.
    /// Setting [`BytesDisplay::Auto`](crate::value::BytesDisplay::Auto) clears
    /// any existing override.
    pub fn set_bytes_display(
        &mut self,
        id: crate::value::BytesId,
        mode: crate::value::BytesDisplay,
    ) {
        if mode == crate::value::BytesDisplay::Auto {
            self.values.bytes_display.remove(&id);
        } else {
            self.values.bytes_display.insert(id, mode);
        }
    }

    /// Returns a list of all live blocks in the context (across all functions).
    pub fn block_ids(&self) -> Vec<BlockId> {
        self.functions().flat_map(|f| f.block_ids()).collect()
    }

    /// Returns a list of all live instructions in the context (across all functions).
    pub fn instruction_ids(&self) -> Vec<InstructionId> {
        self.functions().flat_map(|f| f.instruction_ids()).collect()
    }

    /// Returns a list of all functions in the context
    pub fn function_ids(&self) -> Vec<FunctionId> {
        self.values.functions.iter().map(|f| f.id).collect()
    }

    /// Mints a fresh, uniquely-named anonymous function and returns its id.
    ///
    /// A block must be born into some function's arena; this hands out a host
    /// for standalone blocks (tests, the raw-hex/bare-block lift paths, and the
    /// pyqcode API that build a block without an enclosing function).
    pub fn anon_function(&mut self) -> FunctionId {
        let name = self.get_unique_name(std::borrow::Cow::Borrowed("anon"));
        crate::value::Function::make(self, name)
            .expect("unique anon function name")
            .id
    }

    /// `(total_slots, tombstones)` across every function's instruction arena.
    /// Instruction storage is now per-function; this sums the arenas for the
    /// whole-program fragmentation probe.
    pub fn instruction_arena_stats(&self) -> (usize, usize) {
        let mut total = 0;
        let mut dead = 0;
        for f in self.values.functions.iter() {
            total += f.insns.len();
            dead += f.insns.iter().filter(|i| i.deleted).count();
        }
        (total, dead)
    }

    /// Iterates over all the (live) instructions in the context.
    pub fn instructions(&self) -> impl Iterator<Item = InstructionRef<'str, '_>> + '_ {
        self.instruction_ids()
            .into_iter()
            .map(move |id| Instruction::from_id(self, id))
    }

    /// Iterates over all the (live) blocks in the context.
    pub fn blocks(&self) -> impl Iterator<Item = BlockRef<'str, '_>> + '_ {
        self.block_ids()
            .into_iter()
            .map(move |id| BlockRef::from_id(self, id))
    }

    /// Iterates over all the functions in the context
    pub fn functions(&self) -> FunctionIter<'str, '_> {
        FunctionIter {
            ctx: self,
            inner: self.values.functions.iter(),
        }
    }

    /// Iterates over all the functions in the context
    /// alias for `functions()`
    pub fn iter(&self) -> FunctionIter<'str, '_> {
        self.functions()
    }

    /// Check a function *out* of the module: move its [`Function`] out of the
    /// registry, leaving an empty [`Function::sentinel`] in its slot, and return
    /// the owned function. The caller then owns it exclusively and can mutate it in
    /// isolation from the rest of the module — the primitive a pass driver uses to
    /// hand a function to a worker (see `PARALLEL_PASSES.md`). The id and every
    /// other function's stable address are untouched (segmented registry storage).
    ///
    /// While a function is checked out, reading it back through this context
    /// observes the sentinel, so a checked-out function must always be reinstalled
    /// with [`checkin_function`](Self::checkin_function) before anything else looks
    /// at that slot.
    pub fn checkout_function(&mut self, id: FunctionId) -> Function<'str> {
        self.values.functions.replace(id, Function::sentinel())
    }

    /// Reinstall a function previously taken with
    /// [`checkout_function`](Self::checkout_function), discarding the sentinel that
    /// held its slot. The function's id must be the one it was checked out under.
    pub fn checkin_function(&mut self, id: FunctionId, fun: Function<'str>) {
        self.values.functions.replace(id, fun);
    }

    pub fn varnodes(&self) -> impl Iterator<Item = VarnodeRef<'str, '_>> + '_ {
        self.values
            .varnodes
            .iter()
            .map(|v| Varnode::from_id(self, v.id))
    }

    /// Number of varnodes in the context. The varnode registry is append-only, so
    /// this is monotonic and an unchanged value means an unchanged varnode set —
    /// used to validate caches keyed on the register/varnode layout (e.g. the
    /// alias [`RegisterBase`](../../qcode_analysis/alias/struct.RegisterBase.html)).
    pub fn varnode_count(&self) -> usize {
        self.values.varnodes.len()
    }

    /// Adds a directed edge in the CFG from `from` to `to`, returning its id.
    pub fn add_cfg_edge(&mut self, from: BlockId, to: BlockId) -> EdgeId {
        // The edge is stored in `from`'s edge arena. It is usually intra-function,
        // but a thunk/tail-call `Branch` targets another function's entry block —
        // a legitimate cross-function edge. Composite `EdgeId` routing lets both
        // incident blocks reference it regardless of which arena holds it.
        let edge_id = self.values.push_edge(from.func, EdgeData { from, to });
        BasicBlock::from_id_mut(self, from).add_edge(edge_id);
        BasicBlock::from_id_mut(self, to).add_edge(edge_id);
        edge_id
    }

    /// Removes a CFG edge, unlinking it from both incident blocks' edge sets.
    ///
    /// The backing [`EdgeData`] slot in the append-only registry is left in
    /// place (dangling), consistent with how removed instructions are handled;
    /// per-block traversal reads the block edge sets, which this updates.
    pub fn remove_cfg_edge(&mut self, edge_id: EdgeId) {
        let &EdgeData { from, to } = self.values.edge(edge_id);
        BasicBlock::from_id_mut(self, from).remove_edge(edge_id);
        BasicBlock::from_id_mut(self, to).remove_edge(edge_id);
    }

    /// Assumes `prop` is true. Returns `false` (and records nothing) if the
    /// proposition is already assumed or known false; returns `true` if it was
    /// recorded or already held with the same polarity (idempotent). The
    /// recording pass is taken from [`pass_scope`](crate::pass_scope).
    pub fn assume_true(&mut self, prop: Proposition) -> bool {
        self.assume(prop, true)
    }

    /// Assumes `prop` is false. Mirror of [`assume_true`](Self::assume_true).
    pub fn assume_false(&mut self, prop: Proposition) -> bool {
        self.assume(prop, false)
    }

    fn assume(&mut self, prop: Proposition, value: bool) -> bool {
        match self.values.truths.get(&prop) {
            Some(t) => t.value == value,
            None => {
                self.values.truths.insert(
                    prop,
                    Truth {
                        value,
                        certainty: Certainty::Assumed,
                        pass: PassName(pass_scope::current_pass()),
                    },
                );
                true
            }
        }
    }

    /// Records `prop = value` as proven, overriding any assumption. If this
    /// contradicts an existing assumption, a [`Violation`] is recorded — the
    /// checkpoint+replay driver's signal to discard this working copy.
    /// Contradicting an existing *known* fact is a logic error.
    ///
    /// Returns `true` if the fact is *novel* (no prior truth, or it overturned
    /// an assumption): the driver replays when a round produced novel facts.
    pub fn set_known(&mut self, prop: Proposition, value: bool) -> bool {
        let pass = PassName(pass_scope::current_pass());
        let novel = match self.values.truths.get(&prop) {
            Some(prior) => {
                // Proving the opposite of an already-*known* fact (e.g. a user
                // override the analysis disproves) is not a replay signal: record
                // it as a hard contradiction and keep the original known value so
                // the driver can surface an error and terminate.
                if prior.certainty == Certainty::Known && prior.value != value {
                    self.values.known_contradictions.push(KnownContradiction {
                        prop,
                        known: prior.value,
                        proven: value,
                        known_pass: prior.pass,
                        proven_pass: pass,
                    });
                    return false;
                }
                if prior.certainty == Certainty::Assumed && prior.value != value {
                    self.values.violations.push(Violation {
                        prop,
                        assumed: prior.value,
                        assuming_pass: prior.pass,
                        asserting_pass: pass,
                    });
                    true
                } else {
                    false
                }
            }
            None => true,
        };
        self.values.truths.insert(
            prop,
            Truth {
                value,
                certainty: Certainty::Known,
                pass,
            },
        );
        novel
    }

    /// Seeds a proven fact carried over from an earlier checkpoint+replay
    /// round. Unlike [`set_known`](Self::set_known) this is not "novel": it
    /// must not retrigger a replay, and seeding over an existing entry is a
    /// logic error (seed before any pass runs).
    ///
    /// `pass` is the identity of the pass that originally proved the fact (as
    /// harvested from [`known_facts`](Self::known_facts)), preserved across the
    /// round boundary so the converged context still names the proving pass
    /// rather than the re-seeding driver.
    pub fn seed_known(&mut self, prop: Proposition, value: bool, pass: PassName) {
        let prior = self.values.truths.insert(
            prop,
            Truth {
                value,
                certainty: Certainty::Known,
                pass,
            },
        );
        debug_assert!(prior.is_none(), "seeding {prop:?} over an existing truth");
    }

    /// The recorded [`Truth`] of `prop`, if any.
    pub fn truth(&self, prop: Proposition) -> Option<Truth> {
        self.values.truths.get(&prop).copied()
    }

    /// The proven value of `prop`: `Some` only for *known* entries.
    pub fn known(&self, prop: Proposition) -> Option<bool> {
        self.truth(prop)
            .filter(|t| t.certainty == Certainty::Known)
            .map(|t| t.value)
    }

    /// Iterates over every recorded truth (assumed and known).
    pub fn truths(&self) -> impl Iterator<Item = (Proposition, Truth)> + '_ {
        self.values.truths.iter().map(|(&p, &t)| (p, t))
    }

    /// Iterates over the proven facts, for the replay driver to harvest into
    /// the next round's [`seed_known`](Self::seed_known) calls.
    pub fn known_facts(&self) -> impl Iterator<Item = (Proposition, bool, PassName)> + '_ {
        self.truths()
            .filter(|(_, t)| t.certainty == Certainty::Known)
            .map(|(p, t)| (p, t.value, t.pass))
    }

    /// The violations recorded this round (proven facts that contradicted an
    /// assumption). Non-empty means derived IR may be wrong: replay.
    pub fn violations(&self) -> &[Violation] {
        &self.values.violations
    }

    /// Facts proven this round that contradicted an existing *known* fact (e.g. a
    /// user override the analysis disproved). Non-empty means the analysis cannot
    /// honor the forced value; the driver surfaces this as a hard error.
    pub fn known_contradictions(&self) -> &[KnownContradiction] {
        &self.values.known_contradictions
    }

    /// Returns the raw `u64` backing value of the literal `id`.
    pub fn get_literal_value(&self, id: LiteralId) -> u64 {
        self.values.literals[id].value
    }

    /// Returns an immutable reference to the instruction identified by `id`.
    pub fn get_insn(&self, id: InstructionId) -> InstructionRef<'str, '_> {
        InstructionRef::from_id(self, id)
    }

    /// Returns an immutable reference to the varnode mapped to the named
    /// register `id`.
    pub fn get_register(&self, id: RegisterId) -> VarnodeRef<'str, '_> {
        Varnode::from_id(self, self.registers[&id])
    }

    /// Creates a [`Value`] representing an integer constant of the given byte width.
    pub fn get_const(&self, value: u64, size: usize) -> LiteralRef<'str, '_> {
        let type_id = self.types.get_or_make_int(size);
        let id = self.values.get_or_make_typed_literal(value, type_id, size);
        LiteralRef::new(self, id)
    }

    /// Creates a `bool`-typed constant (`true`/`false`), byte-stored with value
    /// `1`/`0`. This is the only way to mint a `bool` literal.
    pub fn get_bool_const(&self, value: bool) -> LiteralRef<'str, '_> {
        let type_id = self.types.get_or_make_bool();
        let id = self
            .values
            .get_or_make_typed_literal(u64::from(value), type_id, 1);
        LiteralRef::new(self, id)
    }

    /// Creates a typed constant literal.
    ///
    /// Unlike [`get_const`](Self::get_const) this accepts an arbitrary [`TypeId`],
    /// allowing StackAddress constants (e.g. the stack base) to preserve their
    /// type through constant folding.
    pub fn get_typed_const(
        &mut self,
        value: u64,
        type_id: crate::types::TypeId,
    ) -> LiteralRef<'str, '_> {
        let size = self.types.size_of(type_id);
        let id = self.values.get_or_make_typed_literal(value, type_id, size);
        LiteralRef::new(self, id)
    }

    /// Creates an opaque byte-blob constant from a little-endian, memory-order
    /// byte vector.
    ///
    /// The blob is typed as an `Array(i8, data.len())`. Unlike numeric literals,
    /// byte blobs are **not interned**: every call produces a fresh
    /// [`BytesId`](crate::value::BytesId). Use this for constants wider than a
    /// `u64` (SSE/AVX pools, wide stack/memory reads, coalesced constant stores).
    pub fn get_bytes(&mut self, data: Vec<u8>) -> crate::value::BytesRef<'str, '_> {
        let i8_ty = self.types.get_or_make_int(1);
        let type_id = self.types.get_or_make_array(i8_ty, data.len());
        let id = self
            .values
            .bytes
            .push(crate::value::Bytes { data, type_id });
        crate::value::BytesRef::new(self, id)
    }

    /// Returns the [`TypeId`] of any [`ValueId`] in this context.
    ///
    /// Varnodes are typed as `Int(varnode.size())`. Blocks, functions, and other
    /// non-data values return `Int(0)`.
    pub fn type_of(&self, id: ValueId) -> crate::types::TypeId {
        match id {
            ValueId::Literal(lid) => self.values.literals[lid].type_id,
            ValueId::Bytes(bid) => self.values.bytes[bid].type_id,
            ValueId::Instruction(iid) => self.values.instruction(iid).type_id,
            ValueId::BlockParam(pid) => self.values.block_param(pid).type_id,
            ValueId::Varnode(vid) => {
                if let Some(&ty) = self.values.varnode_types.get(&vid) {
                    return ty;
                }
                let size = self.values.varnodes[vid].size_bytes();
                self.types.get_or_make_int(size)
            }
            // Exhaustive on purpose: a new ValueId variant must decide its type
            // here rather than silently inheriting the zero-width sentinel.
            ValueId::BasicBlock(_) | ValueId::Function(_) => self.types.get_or_make_int(0),
        }
    }

    /// Returns the stored [`TypeId`] for value kinds that carry one directly.
    ///
    /// Unlike [`Context::type_of`], this never interns fallback integer types,
    /// so it works from immutable formatting and parsing paths. Varnodes,
    /// blocks, and functions return `None`.
    pub fn stored_type_of(&self, id: ValueId) -> Option<crate::types::TypeId> {
        match id {
            ValueId::Literal(lid) => Some(self.values.literals[lid].type_id),
            ValueId::Bytes(bid) => Some(self.values.bytes[bid].type_id),
            ValueId::Instruction(iid) => Some(self.values.instruction(iid).type_id),
            ValueId::BlockParam(pid) => Some(self.values.block_param(pid).type_id),
            ValueId::Varnode(vid) => self.values.varnode_types.get(&vid).copied(),
            ValueId::BasicBlock(_) | ValueId::Function(_) => None,
        }
    }

    /// Gives `varnode` a global type override, replacing the default
    /// `Int(size)`. Used to type ambient register globals — e.g. the `FS_OFFSET`
    /// segment base as `PtrTo<TEB>` — so every use across all functions reads the
    /// richer type. Pass a type whose size matches the varnode's width.
    pub fn set_varnode_type(&mut self, varnode: VarnodeId, type_id: crate::types::TypeId) {
        self.values.varnode_types.insert(varnode, type_id);
    }

    /// Return all instructions that use `value` as an operand.
    ///
    /// For an SSA value (instruction result or block param) this is the complete
    /// user set, read from its owning function. For a shared value
    /// (literal/bytes/varnode) it is `&[]` — those have no owning function and
    /// their uses are tracked per using-function; use
    /// [`users_across_functions`](Self::users_across_functions) to find them.
    pub fn users(&self, value: impl Into<ValueId>) -> &[InstructionId] {
        self.values.users_of(value.into())
    }

    /// Every instruction across all functions that uses `value` as an operand.
    /// Unlike [`users`](Self::users) this scans every function, so it answers a
    /// shared value (literal/bytes/varnode) whose uses span functions. Off the
    /// hot path (allocates); prefer [`users`](Self::users) for an SSA value.
    pub fn users_across_functions(&self, value: impl Into<ValueId>) -> Vec<InstructionId> {
        let value = value.into();
        self.functions()
            .flat_map(|f| f.users_of(value).to_vec())
            .collect()
    }

    /// Replace every use of `old` with `new` across all instructions that
    /// reference `old`, and update the users reverse map accordingly.
    pub fn replace_all_uses_with(&mut self, old: impl Into<ValueId>, new: impl Into<ValueId>) {
        let old = old.into();
        let new = new.into();
        if old == new {
            return;
        }
        let users: Vec<InstructionId> = self.values.users_of(old).to_vec();
        // `old`'s user list lives in its owning function's map; every user we are
        // rewriting is in that same function, so `new`'s new uses are recorded
        // there too. `old` is always an SSA def in practice (audited); a shared
        // value has no owning function and no locatable user list here.
        let Some(func) = old.owning_function() else {
            debug_assert!(
                users.is_empty(),
                "replace_all_uses_with on a shared value that has users"
            );
            return;
        };
        for user_id in users {
            Instruction::from_id_mut(self, user_id)
                .mnemonic_mut()
                .replace_value(old, new);
            self.values.functions[func]
                .users
                .entry(new)
                .or_default()
                .push(user_id);
        }
        self.values.functions[func].users.remove(&old);
    }

    /// Replaces one instruction's mnemonic and keeps the reverse use map in sync.
    ///
    /// This is for transforms that change an instruction in place without
    /// changing its identity, parent block, address, or result type.
    pub fn replace_instruction_mnemonic(&mut self, id: InstructionId, mnemonic: Mnemonic) {
        // Every operand's use is recorded in this instruction's own function map.
        let func = id.func;
        let old_args = self.values.instruction(id).mnemonic().args();
        for arg in old_args {
            let mut remove_arg = false;
            if let Some(users) = self.values.functions[func].users.get_mut(&arg) {
                users.retain(|&user| user != id);
                remove_arg = users.is_empty();
            }
            if remove_arg {
                self.values.functions[func].users.remove(&arg);
            }
        }

        // Drop this instruction's old call edge (if it was a direct call) before
        // overwriting the mnemonic; the new one's edge is recorded below.
        if let Some(target) = self.values.instruction(id).mnemonic().call_target()
            && let Some(sites) = self.values.call_sites.get_mut(&target)
        {
            sites.retain(|&site| site != id);
        }

        *Instruction::from_id_mut(self, id).mnemonic_mut() = mnemonic;

        for arg in self.values.instruction(id).mnemonic().args() {
            self.values.functions[func]
                .users
                .entry(arg)
                .or_default()
                .push(id);
        }
        if let Some(target) = self.values.instruction(id).mnemonic().call_target() {
            self.values.call_sites.entry(target).or_default().push(id);
        }
    }

    /// Removes an instruction from its parent block and unlinks all associated state:
    /// removes the instruction from the block's instruction list, clears `parent`,
    /// removes its name from the name map, clears the name field, and prunes it from
    /// the `users` reverse map for all its operands.
    ///
    /// If the instruction has no parent block the block-list and parent steps are
    /// skipped, but name and users cleanup still runs.
    pub fn remove_instruction(&mut self, id: InstructionId) {
        let parent = self.values.instruction(id).parent;
        let name = self.values.instruction(id).name.clone();

        if let Some(block_id) = parent {
            self.values
                .block_mut(block_id)
                .instructions
                .retain(|&i| i != id);

            // If this instruction was a terminator instruction in a basic block,
            // remove cfg edges
            if Instruction::from_id(&*self, id).mnemonic().is_terminator() {
                let mut edges_to_remove = HashSet::default();
                for edge in BasicBlock::from_id(&*self, block_id).successors() {
                    edges_to_remove.insert(edge.0);
                }

                for edge_id in edges_to_remove {
                    self.remove_cfg_edge(edge_id);
                }
            }
        }

        self.values.instruction_mut(id).parent = None;

        if let Some(ref n) = name {
            // The instruction's name lives in its own function's name table.
            self.values.functions[id.func].names.forget(n.as_ref());
        }
        self.values.instruction_mut(id).name = None;

        self.values.remove_instructions(&HashSet::from_iter([id]));
    }

    /// Associates `addr` with `id` in the address map.
    ///
    /// Returns `Err(Error::DuplicateAddress(addr))` if the address is already
    /// mapped. Callers that have already checked (e.g. via
    /// [`get_at_addr`](Self::get_at_addr)) may safely `.expect(...)` the result.
    pub(crate) fn set_address(&mut self, addr: u64, id: ValueId) -> crate::error::Result<()> {
        if let Some(existing) = self.address_map.insert(addr, id) {
            // The only case where duplicates are allowed are for a function and its root block sharing an address
            // In that case, the function address should be kept.

            match (existing, id) {
                // A function and its entry block share an address by design.
                // The function always takes priority in the address map,
                // regardless of whether the root link has been established yet
                // (make_at_addr registers the address before calling make_root).
                (ValueId::Function(func_id), ValueId::BasicBlock(block_id))
                | (ValueId::BasicBlock(block_id), ValueId::Function(func_id)) => {
                    let mut function = Function::from_id_mut(self, func_id);
                    function.ensure_root(block_id)?;
                    self.address_map.insert(addr, ValueId::Function(func_id));
                    return Ok(());
                }

                _ => {}
            }

            Err(Error::spanless(ErrorTy::DuplicateAddress(addr, existing)))
        } else {
            Ok(())
        }
    }

    /// Changes the name of a value, in the name table that owns its kind
    /// (function-local for block/instruction/param, global otherwise).
    pub fn update_name(
        &mut self,
        name: Cow<'str, str>,
        id: ValueId,
        old_name: Option<&str>,
    ) -> Result<()> {
        match id.name_scope_function() {
            Some(func) => self.values.functions[func]
                .names
                .register(name, id, old_name),
            None => self.name_map.register(name, id, old_name),
        }
    }

    /// Resolve `name` in the table that owns `id`'s kind (function-local for
    /// block/instruction/param, global otherwise). Used by the rename path to
    /// check for a conflict in the correct namespace, and by passes that mint a
    /// unique name for a known SSA value.
    pub fn get_named_in_scope(&self, id: ValueId, name: &str) -> Option<ValueId> {
        match id.name_scope_function() {
            Some(func) => self.values.functions[func].names.get(name),
            None => self.name_map.get(name),
        }
    }

    /// Remove `name` from the name map, keeping the [`get_unique_name`] suffix
    /// hint exact: if `name` is a generated `base_<n>` suffix, lower `base`'s hint
    /// so the freed suffix is reconsidered on the next call (a naive first-free
    /// scan would reuse it, and the hint must not skip it). Un-suffixed names are
    /// Attempts to get a value ID by its *global* name (function/varnode/space/
    /// p-code/bytes). Block/instruction/param names are function-scoped and are
    /// resolved through their owning [`Function`] (see [`NameTable`]); this
    /// returns `None` for them.
    pub fn get_named(&self, name: &str) -> Option<ValueId> {
        self.name_map.get(name)
    }

    /// Gets a value ID at a given address
    pub fn get_at_addr(&self, addr: &u64) -> Option<ValueId> {
        self.address_map.get(addr).copied()
    }

    /// Gets a unique **global** name (functions, varnodes, spaces, …), appending
    /// a numeric suffix until free. For a block/instruction/param name, use
    /// [`get_unique_name_in`](Self::get_unique_name_in) so uniqueness is checked
    /// against the owning function's table.
    pub fn get_unique_name(&mut self, name: Cow<'str, str>) -> Cow<'str, str> {
        self.name_map.unique(name)
    }

    /// Gets a unique name within `func`'s function-local name table (for block,
    /// instruction, and block-param names). Two functions may thus reuse the same
    /// name independently.
    pub fn get_unique_name_in(&mut self, func: FunctionId, name: Cow<'str, str>) -> Cow<'str, str> {
        self.values.functions[func].names.unique(name)
    }
}

/// A name → value reverse map with amortized unique-name minting.
///
/// The context keeps one **global** table for module-scoped values (functions,
/// varnodes, spaces, p-code ops, byte blobs); each [`Function`](crate::value::Function)
/// keeps its **own** table for its block/instruction/param names. Keeping those
/// namespaces independent is a prerequisite for running function passes in
/// parallel: a worker mints names against its function's table with no global
/// lock and no cross-function collisions. Two functions may each name a block
/// `loop` — they render correctly because a value's own `name` field is the
/// source of truth; this table only enforces uniqueness and resolves by name.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct NameTable<'str> {
    /// name → the value that holds it.
    map: HashMap<Cow<'str, str>, ValueId>,
    /// Per-base "next suffix to try" lower-bound hints for [`unique`](Self::unique),
    /// so probing resumes instead of rescanning from `0`. A derived cache: rides
    /// through `clone` but is not serialized (see [`Context::get_unique_name`]).
    #[serde(skip)]
    suffix_hint: HashMap<String, u32>,
}

impl<'str> NameTable<'str> {
    /// The value currently holding `name`, if any.
    pub fn get(&self, name: &str) -> Option<ValueId> {
        self.map.get(name).copied()
    }

    /// Whether `name` is taken.
    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// Register `name` for `id`, forgetting `old_name` first. Errors if `name`
    /// is already taken (callers pre-check via [`get`](Self::get), so this only
    /// fires defensively).
    pub fn register(
        &mut self,
        name: Cow<'str, str>,
        id: ValueId,
        old_name: Option<&str>,
    ) -> Result<()> {
        if let Some(old_name) = old_name {
            self.forget(old_name);
        }
        match self.map.insert(name.clone(), id) {
            Some(_) => Err(Error::spanless(ErrorTy::DuplicateName(name.to_string()))),
            None => Ok(()),
        }
    }

    /// Remove `name`, keeping the [`unique`](Self::unique) suffix hint exact: if
    /// `name` is a generated `base_<n>` suffix, lower `base`'s hint so the freed
    /// suffix is reconsidered next time.
    pub fn forget(&mut self, name: &str) {
        self.map.remove(name);
        if let Some((base, suffix)) = split_generated_suffix(name)
            && let Some(hint) = self.suffix_hint.get_mut(base)
        {
            *hint = (*hint).min(suffix);
        }
    }

    /// A free name derived from `name`: the bare name if untaken, else the first
    /// free `name_<n>`. Resumes suffix probing from a cached lower bound so
    /// minting many like-named values stays ~O(1) amortized; the chosen suffix is
    /// identical to a naive first-free scan from `1`.
    pub fn unique(&mut self, name: Cow<'str, str>) -> Cow<'str, str> {
        use std::fmt::Write as _;

        if !self.map.contains_key(&name) {
            return name;
        }
        let base: &str = &name;
        let mut suffix = self.suffix_hint.get(base).copied().unwrap_or(1).max(1);
        let mut unique_name = format!("{base}_{suffix}");
        while self.map.contains_key(unique_name.as_str()) {
            suffix += 1;
            unique_name.clear();
            let _ = write!(unique_name, "{base}_{suffix}");
        }
        self.suffix_hint.insert(base.to_string(), suffix);
        Cow::Owned(unique_name)
    }
}

/// Split a generated unique name into its base and numeric suffix, i.e. the
/// inverse of the `format!("{base}_{suffix}")` in [`NameTable::unique`]:
/// `"tmp_7"` → `Some(("tmp", 7))`. Returns `None` for names with no `_<digits>`
/// tail (a bare base, or a name whose tail is empty/non-numeric/overflows).
fn split_generated_suffix(name: &str) -> Option<(&str, u32)> {
    let (base, digits) = name.rsplit_once('_')?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((base, digits.parse().ok()?))
}

impl Display for Context<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.functions().try_for_each(|fun| fun.fmt(f))?;

        self.blocks()
            .filter(|block| block.parent().is_none())
            .try_for_each(|block| block.fmt(f))
    }
}

impl<'str> Graph for Context<'str> {
    type NodeId = BlockId;

    type EdgeId = EdgeId;

    // Fixed-seed hasher so a block's incident-edge set iterates in a
    // deterministic order. This is what makes `predecessors()`/`successors()`
    // (and thus borderline mem2reg promotions) reproducible across runs; the
    // std default (`RandomState`) reseeds per process and leaks that
    // nondeterminism into the lifted IR.
    type Hasher = jstd::graph::FxBuildHasher;

    type Node<'a>
        = BlockRef<'str, 'a>
    where
        Self: 'a;

    type Edge<'a>
        = EdgeRef<'str, 'a>
    where
        Self: 'a;

    type NodeMut<'a>
        = BlockMutRef<'str, 'a>
    where
        Self: 'a;

    type EdgeMut<'a>
        = EdgeMutRef<'str, 'a>
    where
        Self: 'a;

    fn get_node(&self, id: Self::NodeId) -> Option<Self::Node<'_>> {
        Some(BasicBlock::from_id(self, id))
    }

    fn get_node_mut(&mut self, id: Self::NodeId) -> Option<Self::NodeMut<'_>> {
        Some(BasicBlock::from_id_mut(self, id))
    }

    fn get_edge(&self, id: Self::EdgeId) -> Option<Self::Edge<'_>> {
        Some(EdgeRef::new(self, id))
    }

    fn get_edge_mut(&mut self, id: Self::EdgeId) -> Option<Self::EdgeMut<'_>> {
        Some(EdgeMutRef::new(self, id))
    }

    fn nodes(&self) -> impl Iterator<Item = Self::Node<'_>> + '_ {
        self.block_ids()
            .into_iter()
            .map(move |id| BasicBlock::from_id(self, id))
    }

    fn edges(&self) -> impl Iterator<Item = Self::Edge<'_>> + '_ {
        let ids: Vec<EdgeId> = self.functions().flat_map(|f| f.edge_ids()).collect();
        ids.into_iter().map(move |id| EdgeRef::new(self, id))
    }
}

pub struct FunctionIter<'str, 'ctx> {
    ctx: &'ctx Context<'str>,
    inner: registry::Iter<'ctx, FunctionId, Function<'str>>,
}

impl<'str, 'ctx> Iterator for FunctionIter<'str, 'ctx> {
    type Item = FunctionRef<'str, 'ctx>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|f| FunctionRef::from_id(self.ctx, f.id))
    }
}

impl<'str, 'ctx> IntoIterator for &'ctx Context<'str> {
    type Item = FunctionRef<'str, 'ctx>;
    type IntoIter = FunctionIter<'str, 'ctx>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{
        BasicBlock, Function, ValueId,
        insn::{Binary, Binop, Call, IntBinop, Load, Mnemonic},
    };
    use qcode_macro::qcode;

    fn make_fn_with_blocks(ctx: &mut Context<'static>, name: &'static str, n: usize) -> FunctionId {
        // The function must exist before its blocks so they are born into its arena.
        let f = Function::make(ctx, name.into()).unwrap().id;
        for _ in 0..n {
            BasicBlock::make(ctx, f);
        }
        f
    }

    #[test]
    fn functions_iter_yields_all_functions() {
        let mut ctx = Context::new();
        make_fn_with_blocks(&mut ctx, "alpha", 1);
        make_fn_with_blocks(&mut ctx, "beta", 1);

        let names: Vec<_> = ctx.functions().map(|f| f.name().to_string()).collect();
        assert!(names.contains(&"alpha".to_string()));
        assert!(names.contains(&"beta".to_string()));
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn checkout_checkin_round_trips_a_function() {
        let mut ctx = Context::new();
        let alpha = make_fn_with_blocks(&mut ctx, "alpha", 2);
        let beta = make_fn_with_blocks(&mut ctx, "beta", 1);

        // Check `alpha` out: the slot now holds an empty sentinel, and we own the
        // real function with all its blocks.
        let fun = ctx.checkout_function(alpha);
        assert_eq!(fun.name, "alpha");
        assert_eq!(fun.roster.len(), 2);
        assert_eq!(FunctionRef::from_id(&ctx, alpha).name(), "");
        assert_eq!(FunctionRef::from_id(&ctx, alpha).blocks().count(), 0);
        // A sibling is entirely undisturbed while `alpha` is out.
        assert_eq!(FunctionRef::from_id(&ctx, beta).name(), "beta");
        assert_eq!(FunctionRef::from_id(&ctx, beta).blocks().count(), 1);

        // Check it back in: the function is whole again.
        ctx.checkin_function(alpha, fun);
        assert_eq!(FunctionRef::from_id(&ctx, alpha).name(), "alpha");
        assert_eq!(FunctionRef::from_id(&ctx, alpha).blocks().count(), 2);
        assert_eq!(FunctionRef::from_id(&ctx, beta).name(), "beta");
    }

    #[test]
    fn checked_host_reads_match_module_reads() {
        use crate::value::{
            FunctionId, FunctionRef,
            block::BlockId,
            util::base_ref::{BaseRef, HostRef},
        };

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn foo:
                <bb1>
                    if i8 1 goto <bb2> else goto <bb3>;
                <bb2>
                    goto <bb3>;
                <bb3>
                    return at 0;
            "
        );
        let fid = Function::from_name(&ctx, "foo").unwrap().id();
        let fid = ValueId::as_function(fid).unwrap();

        // A structural snapshot read entirely through a `HostRef` — function name,
        // and per (address-then-index ordered) block: name, successor block names,
        // instruction opcodes, and param count. Both hosts route through the same
        // ref code, so equal snapshots prove the `Checked` routing.
        type Snap = (String, Vec<(String, Vec<String>, Vec<String>, usize)>);
        fn snapshot(host: HostRef, fid: FunctionId) -> Snap {
            let f = FunctionRef::new(host, fid);
            let blocks = f
                .blocks()
                .map(|b| {
                    let name = b.name().unwrap_or("?").to_string();
                    let mut succ: Vec<String> = b
                        .successors()
                        .map(|(_, s)| {
                            BaseRef::<HostRef, BlockId>::new(host, s)
                                .name()
                                .unwrap_or("?")
                                .to_string()
                        })
                        .collect();
                    succ.sort();
                    let ops: Vec<String> =
                        b.instructions().map(|i| i.opcode().to_string()).collect();
                    (name, succ, ops, b.num_params())
                })
                .collect();
            (f.name().to_string(), blocks)
        }

        let module_snap = snapshot(HostRef::Module(&ctx), fid);
        assert!(!module_snap.1.is_empty(), "sanity: foo has blocks");

        // Check the function out; its registry slot is now an empty sentinel, so a
        // plain module read sees nothing — only `Checked` routing recovers it.
        let fun = ctx.checkout_function(fid);
        assert_eq!(FunctionRef::from_id(&ctx, fid).blocks().count(), 0);

        let checked = HostRef::Checked {
            fun: &fun,
            shared: &ctx,
            id: fid,
        };
        let checked_snap = snapshot(checked, fid);
        assert_eq!(
            module_snap, checked_snap,
            "reads through a Checked host must match the pre-checkout module reads"
        );

        ctx.checkin_function(fid, fun);
        assert_eq!(FunctionRef::from_id(&ctx, fid).blocks().count(), 3);
    }

    #[test]
    fn checked_host_mut_matches_module_mut() {
        use crate::value::{
            BlockParam, FunctionId, FunctionRef, InstructionId, Renameable,
            block::BlockId,
            block_param::BlockParamId,
            util::{
                base_ref::BaseRef,
                host_mut::{CheckedOut, HostMut},
            },
        };

        fn build(mut ctx: &mut Context<'static>) -> (FunctionId, BlockId, BlockId, InstructionId) {
            qcode!(
                ctx,
                "
                varnode i64 x;
                fn foo:
                    <entry>
                        %a = load(x:8, &x);
                        %b = load(x:8, &x);
                        goto <bb1>;
                    <bb1>
                        return at %a;
                "
            );
            let fid = foo;
            let entry = FunctionRef::from_id(ctx, fid).root().unwrap().id;
            let bb1 = FunctionRef::from_id(ctx, fid)
                .blocks()
                .map(|b| b.id)
                .find(|&b| b != entry)
                .unwrap();
            let insns = BasicBlock::from_id(ctx, entry).instruction_ids().to_vec();
            (fid, entry, bb1, insns[0])
        }

        // Give `bb1` a parameter to resize; identical setup on both paths.
        fn add_param(ctx: &mut Context<'static>, bb1: BlockId) -> BlockParamId {
            BasicBlock::from_id_mut(ctx, bb1).push_param(8).id
        }

        // Structural snapshot: per block, (name, comment, param sizes, opcodes,
        // sorted successor names).
        type MSnap = Vec<(String, Option<String>, Vec<usize>, Vec<String>, Vec<String>)>;
        fn snap(ctx: &Context, fid: FunctionId) -> MSnap {
            FunctionRef::from_id(ctx, fid)
                .blocks()
                .map(|b| {
                    let name = b.name().unwrap_or("?").to_string();
                    let comment = b.comment().map(str::to_string);
                    let params: Vec<usize> = b.params().map(|p| p.size()).collect();
                    let ops: Vec<String> =
                        b.instructions().map(|i| i.opcode().to_string()).collect();
                    let mut succ: Vec<String> = b
                        .successors()
                        .map(|(_, s)| {
                            BasicBlock::from_id(ctx, s)
                                .name()
                                .unwrap_or("?")
                                .to_string()
                        })
                        .collect();
                    succ.sort();
                    (name, comment, params, ops, succ)
                })
                .collect()
        }

        // ---- (a) mutate on the module directly (the reference behaviour) ------
        let mut ctx_a = Context::new();
        let (fid, entry, bb1, a) = build(&mut ctx_a);
        let param = add_param(&mut ctx_a, bb1);
        let b = BasicBlock::from_id(&ctx_a, entry).instruction_ids()[1];
        BasicBlock::from_id_mut(&mut ctx_a, entry).set_comment(Some("c".into()));
        BasicBlock::from_id_mut(&mut ctx_a, entry)
            .rename("start".into())
            .unwrap();
        let e = ctx_a.add_cfg_edge(entry, bb1);
        ctx_a.remove_cfg_edge(e);
        ctx_a.replace_all_uses_with(ValueId::Instruction(a), ValueId::Instruction(b));
        ctx_a.remove_instruction(a);
        BlockParam::from_id_mut(&mut ctx_a, param).set_size(4);
        let snap_a = snap(&ctx_a, fid);

        // ---- (b) the same mutations via a checked-out host -------------------
        let mut ctx_b = Context::new();
        let (fid_b, entry_b, bb1_b, a_b) = build(&mut ctx_b);
        let param_b = add_param(&mut ctx_b, bb1_b);
        let b_b = BasicBlock::from_id(&ctx_b, entry_b).instruction_ids()[1];

        let mut fun = ctx_b.checkout_function(fid_b);
        {
            let mut host = CheckedOut::new(&mut fun, fid_b, &ctx_b);
            let mut r = BaseRef::new(host.reborrow(), entry_b);
            r.set_comment(Some("c".into()));
            let mut r = BaseRef::new(host.reborrow(), entry_b);
            r.rename("start".into()).unwrap();
            let e = host.add_cfg_edge(entry_b, bb1_b);
            host.remove_cfg_edge(e);
            host.replace_all_uses_with(ValueId::Instruction(a_b), ValueId::Instruction(b_b));
            host.remove_instruction(a_b);
            let mut r = BaseRef::new(host.reborrow(), param_b);
            r.set_size(4);
        }
        ctx_b.checkin_function(fid_b, fun);
        let snap_b = snap(&ctx_b, fid_b);

        assert_eq!(
            snap_a, snap_b,
            "mutations through a checked-out host must match the module-path mutations"
        );
    }

    #[test]
    fn into_iterator_for_context_matches_functions() {
        let mut ctx = Context::new();
        make_fn_with_blocks(&mut ctx, "f1", 1);
        make_fn_with_blocks(&mut ctx, "f2", 1);

        let via_method: Vec<_> = ctx.functions().map(|f| f.id()).collect();
        let via_into: Vec<_> = (&ctx).into_iter().map(|f| f.id()).collect();
        assert_eq!(via_method, via_into);
    }

    #[test]
    fn blocks_iter_yields_all_blocks() {
        let mut ctx = Context::new();
        make_fn_with_blocks(&mut ctx, "g", 3);

        let count = ctx.blocks().count();
        assert_eq!(count, 3);
    }

    #[test]
    fn instructions_iter_yields_all_instructions() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 ptr;

            <block>
                store(ptr:8, &ptr <- i64 0x1234);
                return at ptr;
            "
        );

        let count = ctx.instructions().count();
        assert!(count >= 1, "expected at least one instruction, got {count}");
    }

    #[test]
    fn remove_instruction_removes_from_block() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            <block>
                %a = load(x:8, &x);
                %b = load(x:8, &x);
                return at %a;
            "
        );
        let block_ref = BasicBlock::from_id(&ctx, block);
        let ids: Vec<_> = block_ref.instruction_ids().to_vec();
        let load_a = ids[0];
        let original_len = ids.len();

        ctx.remove_instruction(load_a);

        let remaining: Vec<_> = BasicBlock::from_id(&ctx, block).instruction_ids().to_vec();
        assert_eq!(remaining.len(), original_len - 1);
        assert!(!remaining.contains(&load_a));
    }

    #[test]
    fn remove_instruction_clears_parent() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            <block>
                %a = load(x:8, &x);
                return at %a;
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];

        ctx.remove_instruction(load_id);

        assert!(
            ctx.get_insn(load_id).parent().is_none(),
            "parent should be None after removal"
        );
    }

    #[test]
    fn remove_instruction_frees_name() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            <block>
                %a = load(x:8, &x);
                return at %a;
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];
        // Instruction names are function-scoped, so resolve in the owner's table.
        assert!(
            ctx.get_named_in_scope(load_id.into(), "a").is_some(),
            "name should be in map before removal"
        );

        ctx.remove_instruction(load_id);

        assert!(
            ctx.get_named_in_scope(load_id.into(), "a").is_none(),
            "name should be gone after removal"
        );
        assert!(
            ctx.get_insn(load_id).name().is_none(),
            "instruction name field should be cleared"
        );
    }

    #[test]
    fn remove_instruction_frees_name_for_reuse() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            <block>
                %a = load(x:8, &x);
                return at %a;
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];

        ctx.remove_instruction(load_id);

        // Building another instruction named %a should succeed now.
        qcode!(
            ctx,
            "
            varnode i64 y;
            <block2>
                %a = load(y:8, &y);
                return at %a;
            "
        );
        let a2 = BasicBlock::from_id(&ctx, block2).instruction_ids()[0];
        assert!(
            ctx.get_named_in_scope(a2.into(), "a").is_some(),
            "name should be reusable after removal"
        );
    }

    #[test]
    fn remove_instruction_updates_users_map() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            <block>
                %a = load(x:8, &x);
                %b = %a + i64 1;
                return at %b;
            "
        );
        let ids: Vec<_> = BasicBlock::from_id(&ctx, block).instruction_ids().to_vec();
        let load_id = ids[0];
        let add_id = ids[1];

        assert!(
            ctx.users(load_id).contains(&add_id),
            "add should be a user of load before removal"
        );

        ctx.remove_instruction(add_id);

        assert!(
            ctx.users(load_id).is_empty(),
            "load should have no users after add is removed"
        );
    }

    #[test]
    fn removed_instruction_is_tombstoned_and_not_iterated() {
        // Registry IDs are stable, so a removed instruction stays in the arena — but it
        // must be tombstoned and skipped by `ctx.instructions()`, so the stale operands
        // it still carries (e.g. its `Load.ptr`) never pollute a whole-program scan.
        // Regression: a deleted ram load kept showing up in the alias pass's pointer
        // scan, faking a "pointer used in two spaces" invariant break.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            <block>
                %a = load(x:8, &x);
                %dead = %a + i64 1;
                return at i64 0;
            "
        );
        let ids: Vec<_> = BasicBlock::from_id(&ctx, block).instruction_ids().to_vec();
        let dead_id = ids[1]; // %dead, unused

        assert!(
            ctx.instructions().any(|i| i.id == dead_id),
            "the instruction is iterated while live"
        );

        ctx.remove_instruction(dead_id);

        assert!(
            ctx.get_insn(dead_id).is_deleted(),
            "a removed instruction must be tombstoned"
        );
        assert!(
            !ctx.instructions().any(|i| i.id == dead_id),
            "a deleted instruction must not be yielded by ctx.instructions()"
        );
    }

    #[test]
    fn replace_instruction_mnemonic_rewrites_callind_users() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 ptr;
            <block>
                call [ptr];
            "
        );
        let call_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];
        let ptr = match ctx.get_insn(call_id).mnemonic() {
            Mnemonic::CallInd(call) => call.ptr,
            other => panic!("expected CallInd, got {other:?}"),
        };
        // `ptr` is a shared varnode, so query its uses across functions.
        assert_eq!(ctx.users_across_functions(ptr), vec![call_id]);

        let target = Function::make(&mut ctx, "target".into()).unwrap().id;
        ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target,
                args: vec![],
                clobbers: vec![],
            }),
        );

        assert!(
            ctx.users_across_functions(ptr).is_empty(),
            "old indirect pointer should no longer list the rewritten call"
        );
        assert!(matches!(
            ctx.get_insn(call_id).mnemonic(),
            Mnemonic::Call(Call {
                target: actual,
                args,
                ..
            }) if *actual == target && args.is_empty()
        ));

        // Rewriting the indirect call into a direct one records the call edge.
        assert_eq!(ctx.values.call_sites_of(target), &[call_id]);
    }

    #[test]
    fn call_sites_track_direct_calls_through_replace_and_remove() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 ptr;
            <block>
                call [ptr];
            "
        );
        let call_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];
        let target = Function::make(&mut ctx, "target".into()).unwrap().id;

        // Indirect calls have no static target, so nothing is recorded yet.
        assert!(ctx.values.call_sites_of(target).is_empty());

        ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target,
                args: vec![],
                clobbers: vec![],
            }),
        );
        assert_eq!(ctx.values.call_sites_of(target), &[call_id]);

        // Removing the instruction prunes its call edge.
        ctx.remove_instruction(call_id);
        assert!(ctx.values.call_sites_of(target).is_empty());
    }

    #[test]
    fn replace_instruction_mnemonic_moves_operand_users() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            varnode i64 y;
            <block>
                %a = load(x:8, x);
                return at %a;
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];
        let old_ptr = ValueId::Varnode(x);
        let new_ptr = ValueId::Varnode(y);
        // Varnodes are shared values, so query their uses across functions.
        assert_eq!(ctx.users_across_functions(old_ptr), vec![load_id]);
        assert!(ctx.users_across_functions(new_ptr).is_empty());

        ctx.replace_instruction_mnemonic(
            load_id,
            Mnemonic::Load(Load {
                space: ctx.default_space,
                ptr: new_ptr,
                size: 8,
            }),
        );

        assert!(ctx.users_across_functions(old_ptr).is_empty());
        assert_eq!(ctx.users_across_functions(new_ptr), vec![load_id]);
    }

    #[test]
    fn replace_instruction_mnemonic_tracks_repeated_operands() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            varnode i64 y;
            <block>
                %a = load(x:8, x);
                return at %a;
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];
        let old_ptr = ValueId::Varnode(x);
        let new_arg = ValueId::Varnode(y);

        ctx.replace_instruction_mnemonic(
            load_id,
            Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::Add),
                lhs: new_arg,
                rhs: new_arg,
            }),
        );

        assert!(ctx.users_across_functions(old_ptr).is_empty());
        assert_eq!(
            ctx.users_across_functions(new_arg),
            vec![load_id, load_id],
            "a mnemonic using the same operand twice should record both uses"
        );
    }

    #[test]
    fn remove_instruction_unparented_noop() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            <block>
                %a = load(x:8, &x);
                return at %a;
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];

        // Manually detach from block without using remove_instruction,
        // simulating an instruction with no parent.
        ctx.values.instruction_mut(load_id).parent = None;

        // Should not panic even though parent is None.
        ctx.remove_instruction(load_id);

        assert!(ctx.get_named("a").is_none());
    }

    #[test]
    fn add_cfg_edge_returns_id_and_remove_unlinks_both_blocks() {
        let mut ctx = Context::new();
        let a = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .id;
        let b = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .id;

        let edge = ctx.add_cfg_edge(a, b);
        assert_eq!(
            BasicBlock::from_id(&ctx, a)
                .successors()
                .collect::<Vec<_>>(),
            vec![(edge, b)]
        );
        assert_eq!(
            BasicBlock::from_id(&ctx, b)
                .predecessors()
                .collect::<Vec<_>>(),
            vec![(edge, a)]
        );

        ctx.remove_cfg_edge(edge);
        assert!(BasicBlock::from_id(&ctx, a).successors().next().is_none());
        assert!(BasicBlock::from_id(&ctx, b).predecessors().next().is_none());
    }

    #[test]
    fn truth_map_tracks_four_states_and_conflicts() {
        let mut ctx = Context::new();
        let callee = Function::make(&mut ctx, "callee".into()).unwrap().id;
        let prop = Proposition::FunctionReturns(callee);

        // First assume wins; same polarity is idempotent; opposite fails.
        assert!(ctx.assume_true(prop));
        assert!(ctx.assume_true(prop));
        assert!(!ctx.assume_false(prop));
        assert_eq!(ctx.known(prop), None, "assumed is not known");

        // The truth map is part of the arena, so it snapshots with a clone.
        let snapshot = ctx.clone();

        // Proving the opposite overturns the assumption and records the
        // violation with both pass names.
        let scope = pass_scope::enter("verifier");
        assert!(ctx.set_known(prop, false), "overturning is novel");
        drop(scope);
        assert_eq!(ctx.known(prop), Some(false));
        let [v] = ctx.violations() else {
            panic!("expected one violation")
        };
        assert_eq!(v.prop, prop);
        assert!(v.assumed);
        assert_eq!(v.asserting_pass, "verifier");

        // Re-proving the same value is not novel.
        assert!(!ctx.set_known(prop, false));

        // The independent snapshot is unaffected.
        assert!(snapshot.violations().is_empty());
        assert_eq!(snapshot.known(prop), None);

        // An assume against a known fact fails; with it, succeeds.
        assert!(!ctx.assume_true(prop));
        assert!(ctx.assume_false(prop));
    }

    #[test]
    fn seeded_facts_are_not_novel() {
        let mut ctx = Context::new();
        let callee = Function::make(&mut ctx, "exit".into()).unwrap().id;
        let prop = Proposition::FunctionReturns(callee);

        ctx.seed_known(prop, false, PassName("seed"));
        assert_eq!(ctx.known(prop), Some(false));
        assert!(!ctx.assume_true(prop), "seeded fact blocks opposite assume");
        assert!(
            !ctx.set_known(prop, false),
            "re-proving a seed is not novel"
        );
        assert!(ctx.violations().is_empty());
    }

    #[test]
    fn discovered_code_records_and_survives_round_trip() {
        let mut ctx = Context::new();
        ctx.discover_code(0x1000, 0x10f0, 0x1100);
        ctx.discover_code(0x1000, 0x10f0, 0x1200);
        ctx.discover_code(0x1000, 0x10f0, 0x1100); // duplicate target is deduped

        let targets: Vec<u64> = ctx.discoveries().map(|d| d.target).collect();
        assert_eq!(targets, vec![0x1100, 0x1200]);

        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&ctx, config).expect("encode");
        let (restored, _): (Context<'static>, usize) =
            bincode::serde::decode_from_slice(&bytes, config).expect("decode");
        assert_eq!(
            restored.discoveries().map(|d| d.target).collect::<Vec<_>>(),
            targets
        );
    }

    #[test]
    fn assume_executable_narrows_once_protections_known() {
        let mut ctx = Context::new();
        ctx.memory_image
            .add_segment(0x1000, vec![0u8; 4], true, false); // code
        ctx.memory_image
            .add_segment(0x2000, vec![0u8; 4], false, true); // data

        // Default r/x while protections unknown: everything is permissive, even
        // unmapped (the lifter reads bytes from the format, not the image).
        assert!(ctx.assume_executable(0x1000));
        assert!(ctx.assume_executable(0x2000));
        assert!(ctx.assume_executable(0x9999));

        ctx.mark_protections_known();
        assert!(ctx.assume_executable(0x1000), "code region stays liftable");
        assert!(
            !ctx.assume_executable(0x2000),
            "data region is skipped once protections are known"
        );
        assert!(
            !ctx.assume_executable(0x9999),
            "unmapped is skipped once known"
        );
        // The skip records the proven fact for the whole containing segment.
        assert_eq!(
            ctx.known(Proposition::ExecutableMemory {
                start: 0x2000,
                end: 0x2004,
            }),
            Some(false),
        );
    }

    #[test]
    fn assume_executable_honors_region_override() {
        let mut ctx = Context::new();
        ctx.memory_image
            .add_segment(0x1000, vec![0u8; 4], true, false); // code
        ctx.memory_image
            .add_segment(0x2000, vec![0u8; 4], false, true); // data
        ctx.mark_protections_known();

        // Force the data region executable and the code region non-executable.
        ctx.seed_known(
            Proposition::ExecutableMemory {
                start: 0x2000,
                end: 0x2004,
            },
            true,
            PassName("override"),
        );
        ctx.seed_known(
            Proposition::ExecutableMemory {
                start: 0x1000,
                end: 0x1004,
            },
            false,
            PassName("override"),
        );

        assert!(
            ctx.assume_executable(0x2000),
            "override wins over the non-executable segment flag"
        );
        assert!(
            !ctx.assume_executable(0x1000),
            "override wins over the executable segment flag"
        );
    }

    #[test]
    fn context_survives_bincode_round_trip() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 ptr;
            <block>
                %a = load(ptr:8, &ptr);
                %b = %a + i64 0x10;
                store(ptr:8, &ptr <- i64 0x1234);
                return at %b;
            "
        );

        // A SpaceAddress type exercises the custom TypeManager serialization.
        let some_space = ctx.make_named_temp_space("scratch");
        let sa = ctx.types.get_or_make_space_address(8, some_space);
        let sa_size = ctx.types.size_of(sa);

        let blocks_before = ctx.block_ids().len();
        let insns_before = ctx.instruction_ids().len();
        let funcs_before = ctx.function_ids().len();

        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&ctx, config).expect("encode");
        let (restored, _): (Context<'static>, usize) =
            bincode::serde::decode_from_slice(&bytes, config).expect("decode");

        assert_eq!(restored.block_ids().len(), blocks_before);
        assert_eq!(restored.instruction_ids().len(), insns_before);
        assert_eq!(restored.function_ids().len(), funcs_before);
        // The SpaceAddress type round-trips: same id, same size, same space.
        assert_eq!(restored.types.size_of(sa), sa_size);
        assert_eq!(restored.types.space_of(sa), Some(some_space));
    }

    #[test]
    fn get_unique_name_resumes_probe_and_reuses_freed_suffixes() {
        use crate::value::VarnodeId;

        let mut ctx = Context::new();
        let id = ValueId::Varnode(VarnodeId::from(0usize));

        // Mirror real callers: take the deduplicated name, then bind it.
        fn take(ctx: &mut Context<'static>, id: ValueId, base: &str) -> String {
            let name = ctx
                .get_unique_name(Cow::Owned(base.to_string()))
                .to_string();
            ctx.update_name(Cow::Owned(name.clone()), id, None).unwrap();
            name
        }

        // Suffixes are handed out in ascending order (bare name first).
        assert_eq!(take(&mut ctx, id, "tmp"), "tmp");
        assert_eq!(take(&mut ctx, id, "tmp"), "tmp_1");
        assert_eq!(take(&mut ctx, id, "tmp"), "tmp_2");
        assert_eq!(take(&mut ctx, id, "tmp"), "tmp_3");

        // A distinct base is unaffected by tmp's hint.
        assert_eq!(take(&mut ctx, id, "x"), "x");
        assert_eq!(take(&mut ctx, id, "x"), "x_1");

        // Freeing tmp_1 must make the next tmp reuse it, exactly as a naive
        // first-free scan would — the resume hint must not skip the hole.
        ctx.update_name(Cow::Borrowed("relocated"), id, Some("tmp_1"))
            .unwrap();
        assert_eq!(take(&mut ctx, id, "tmp"), "tmp_1");
        // ...then continue past the still-taken suffixes.
        assert_eq!(take(&mut ctx, id, "tmp"), "tmp_4");
    }
}
