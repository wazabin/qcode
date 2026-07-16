//! The central arena for all IR state: [`Context`].

use crate::value::QCodeMut;
use std::{borrow::Cow, fmt::Display};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{
    assumption::{Certainty, KnownContradiction, PassName, Proposition, Truth, Violation},
    error::{Error, ErrorTy, Result},
    pass_scope,
    space::{LocalMemorySpaceId, MemorySpaceId, Space, SpaceId},
    types::TypeManager,
    value::{
        BasicBlock, BlockParamRef, FunctionBody, FunctionId, FunctionRef, Instruction, ModuleView,
        QCodeView, TempId, TempSpaceId, ValueId,
        block::{BlockId, BlockRef, EdgeData, EdgeId},
        block_param::{BlockParam, BlockParamId},
        insn::{InstructionId, InstructionRef, Mnemonic, PCodeOpId},
        literal::{LiteralId, LiteralRef},
        registry::ValueRegistry,
        varnode::{Varnode, VarnodeId, VarnodeRef, register::RegisterId},
    },
};
use jstd::registry::{self, Registry};

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
/// // ctx.shared.default_space is the RAM space created by new()
/// ```
///
/// # Lifetime parameter `'str`
///
/// The `'str` lifetime is the lifetime of interned string data used for names
/// and space identifiers. When names are owned (e.g. generated names), they
/// are stored as `Cow::Owned`; when they are borrowed from source data they are
/// `Cow::Borrowed` and must outlive the context.
#[derive(Default, Clone, serde::Serialize)]
pub struct Context<'str> {
    /// Module-shared IR state: everything that is **not** per-function interface
    /// or body storage (regimes 1–3 of the context-split design — architecture,
    /// interners, module maps, truths). Reached today behind `&mut Context`;
    /// [`Context::split()`](Self) (stage 5b-ii c.2) will hand it out as a frozen
    /// `&Shared` view while the bodies registry is borrowed mutably.
    pub shared: Shared<'str>,

    /// Per-function *interface* storage — the caller-reasoning surface (name,
    /// address, kind, external-ness, signature) held in lockstep with
    /// [`bodies`](Self::bodies) under the same [`FunctionId`] space. Never checked
    /// out: a co-checked-out callee answers interface queries from here.
    #[serde(default)]
    pub interfaces: Registry<FunctionId, crate::value::function::FunctionInterface<'str>>,

    /// Per-function *body* storage. Each function owns its instruction/block/param/
    /// edge arenas; the composite-ID accessors ([`Context::instruction`] etc.)
    /// route through here. A checked-out function's body is moved out of its slot
    /// (leaving an empty body); its [`interface`](Self::interfaces) stays put, so
    /// callers always read the real interface.
    pub bodies: Registry<FunctionId, FunctionBody<'str>>,
}

#[derive(serde::Deserialize)]
struct ContextWire<'str> {
    shared: Shared<'str>,
    #[serde(default)]
    interfaces: Registry<FunctionId, crate::value::function::FunctionInterface<'str>>,
    bodies: Registry<FunctionId, FunctionBody<'str>>,
}

impl<'de, 'str> serde::Deserialize<'de> for Context<'str> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let ContextWire {
            shared,
            interfaces,
            mut bodies,
        } = ContextWire::deserialize(deserializer)?;
        if interfaces.len() != bodies.len() {
            return Err(serde::de::Error::custom(
                "function body/interface registries drifted",
            ));
        }
        for mut body in bodies.iter_mut() {
            let id = body.id;
            body.rehydrate_id(id);
        }
        Ok(Self {
            shared,
            interfaces,
            bodies,
        })
    }
}

/// Module-shared IR state: regimes 1–3 of the context-split design (see
/// `docs/plans/context-split/00-overview.md`). Holds the frozen architecture
/// (spaces, registers, memory image), the append-interned value arenas
/// (literals, bytes, varnodes, types) inside [`values`](Self::values), and the
/// phase-mutable module maps (names, truths, discoveries, call
/// sites). Everything here is reachable through a frozen `&Shared` view; nothing
/// per-function-body lives here.
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct Shared<'str> {
    pub default_space: SpaceId,

    /// A mapping of space ids to their corresponding [`Space`]s.
    pub(crate) spaces: Registry<SpaceId, Space>,

    /// A mapping of pcode ops to their names
    pub pcode_ops: Registry<PCodeOpId, Box<str>>,

    /// A mapping of names to spaces
    pub named_spaces: HashMap<Box<str>, SpaceId>,

    /// Global reverse name map for module-scoped values (functions, varnodes,
    /// spaces, p-code ops, byte blobs), used to keep their name hints unique and
    /// resolve them by name. Block/instruction/param/Temp names are **not** here — they
    /// live in each [`FunctionBody`](crate::value::FunctionBody)'s own [`NameTable`], so
    /// those namespaces stay independent across functions (see [`NameTable`]).
    pub(crate) name_map: NameTable<'str>,

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
    pub(crate) primary_entrypoint: Option<u64>,

    /// Code addresses discovered by lifting or analysis but not yet lifted.
    /// `qcode_analysis` cannot call the lifter (one-way crate dependency), so
    /// passes that resolve new targets (e.g. the jump-table pass) record them
    /// here; the `lift_new_addresses` pass drains them and lifts the code into
    /// the (clean) IR. Rides through clone (so it survives checkpoint+replay
    /// rounds) and serialization.
    #[serde(default)]
    pub(crate) discoveries: crate::discovery::DiscoveryQueue,

    /// The operating system of the loaded binary, stamped by the loader from the
    /// binary format (PE → Windows, ELF → Linux). Platform-gated passes — e.g.
    /// TEB seeding, which only applies to Windows — read it. `Unknown` for
    /// synthetic contexts.
    #[serde(default)]
    pub(crate) target_os: TargetOs,

    /// Entry addresses of functions the user asked to skip optimizing (via the
    /// `--ignore` flag). Such functions are still lifted, but every per-function
    /// analysis pass skips them. Rides through clone so it survives the
    /// checkpoint+replay rounds, and through serialization so a saved session
    /// keeps honoring the request.
    #[serde(default)]
    pub(crate) ignored_functions: HashSet<u64>,
}

impl<'str> Shared<'str> {
    /// The value id currently bound to the module-global `name`, if any.
    /// Shared-only mirror of [`Context::get_named`].
    pub fn get_named(&self, name: &str) -> Option<ValueId> {
        self.name_map.get(name)
    }

    /// The varnode `id`. Shared-only accessor (varnodes live in the interners).
    pub fn varnode(&self, id: VarnodeId) -> &crate::value::Varnode<'str> {
        &self.values.varnodes[id]
    }

    /// The space `id`. Shared-only accessor (spaces are frozen architecture).
    pub fn space(&self, id: SpaceId) -> &Space {
        &self.spaces[id]
    }

    /// An interned integer constant of the given byte width, as a [`ValueId`].
    /// Shared-only mirror of [`Context::get_const`] returning the id directly
    /// (the `LiteralRef` wrapper needs a whole `&Context`).
    pub fn get_const(&self, value: u64, size: usize) -> ValueId {
        let type_id = self.types.get_or_make_int(size);
        ValueId::Literal(self.values.get_or_make_typed_literal(value, type_id, size))
    }

    /// A `bool`-typed constant (`true`/`false`), byte-stored. Shared-only mirror
    /// of [`Context::get_bool_const`] returning the id directly.
    pub fn get_bool_const(&self, value: bool) -> ValueId {
        let type_id = self.types.get_or_make_bool();
        ValueId::Literal(
            self.values
                .get_or_make_typed_literal(u64::from(value), type_id, 1),
        )
    }

    /// A typed constant literal. Shared-only mirror of
    /// [`Context::get_typed_const`] returning the id directly.
    pub fn get_typed_const(&self, value: u64, type_id: crate::types::TypeId) -> ValueId {
        let size = self.types.size_of(type_id);
        ValueId::Literal(self.values.get_or_make_typed_literal(value, type_id, size))
    }

    /// An opaque `Array(i8, len)` byte-blob constant, as a [`ValueId`].
    /// Shared-only mirror of [`Context::get_bytes`] returning the id directly.
    pub fn get_bytes(&self, data: Vec<u8>) -> ValueId {
        let i8_ty = self.types.get_or_make_int(1);
        let type_id = self.types.get_or_make_array(i8_ty, data.len());
        ValueId::Bytes(
            self.values
                .bytes
                .push(crate::value::Bytes { data, type_id }),
        )
    }

    /// The forced rendering mode for a `Bytes` blob, or
    /// [`BytesDisplay::Auto`](crate::value::BytesDisplay::Auto) if unset.
    /// Shared-only mirror of [`Context::bytes_display`] (the override map lives in
    /// the interners), for the `&Shared`-backed [`BytesRef`](crate::value::BytesRef).
    pub fn bytes_display(&self, id: crate::value::BytesId) -> crate::value::BytesDisplay {
        self.values
            .bytes_display
            .get(&id)
            .copied()
            .unwrap_or_default()
    }

    /// Like [`get_bytes`](Self::get_bytes) but with an explicit array/sequence
    /// [`TypeId`]. Shared-only mirror of [`Context::get_typed_bytes`] returning
    /// the id directly.
    pub fn get_typed_bytes(&self, data: Vec<u8>, type_id: crate::types::TypeId) -> ValueId {
        ValueId::Bytes(
            self.values
                .bytes
                .push(crate::value::Bytes { data, type_id }),
        )
    }

    /// The recorded [`Truth`] of `prop`, if any. Shared-only
    /// mirror of [`Context::truth`] (truths live in the phase-mutable shared
    /// maps), for `&Shared`-served pass reads.
    pub fn truth(&self, prop: Proposition) -> Option<Truth> {
        self.values.truths.get(&prop).copied()
    }

    /// Iterate every varnode as a [`VarnodeRef`]. Shared-only mirror of
    /// [`Context::varnodes`] (varnodes live in the interners).
    pub fn varnodes(&self) -> impl Iterator<Item = crate::value::VarnodeRef<'str, '_>> + '_ {
        self.values
            .varnodes
            .iter()
            .map(move |v| crate::value::Varnode::from_id(self, v.id))
    }

    /// Number of varnodes. Shared-only mirror of [`Context::varnode_count`];
    /// append-only, so an unchanged value means an unchanged varnode set.
    pub fn varnode_count(&self) -> usize {
        self.values.varnodes.len()
    }

    /// The stored [`TypeId`] of a **shared-leaf** value (literal, bytes, or
    /// varnode-with-override). Shared-only mirror of [`Context::stored_type_of`]:
    /// instruction/block-param/block/function ids live in function bodies and are
    /// out of a `&Shared`'s reach, so they return `None` here (callers route those
    /// through the body). Matches the actual call pattern, where only shared-leaf
    /// ids are passed to the shared path.
    pub fn stored_type_of(&self, id: ValueId) -> Option<crate::types::TypeId> {
        match id {
            ValueId::Literal(lid) => Some(self.values.literals[lid].type_id),
            ValueId::Bytes(bid) => Some(self.values.bytes[bid].type_id),
            ValueId::Varnode(vid) => self.values.varnode_types.get(&vid).copied(),
            ValueId::Instruction(_)
            | ValueId::BlockParam(_)
            | ValueId::BasicBlock(_)
            | ValueId::Temp(_)
            | ValueId::Function(_) => None,
        }
    }
}

/// The operating system of a loaded binary, inferred from its container format.
/// The enum now lives in the leaf `binfmt` crate (next to the container
/// parsers); re-exported here so `qcode::context::TargetOs` keeps resolving.
pub use binfmt::TargetOs;

impl<'str> Context<'str> {
    /// Creates a new, empty context with a single default RAM space.
    ///
    /// The default space has a word size of 1 byte and an address size of 8
    /// bytes (suitable for 64-bit architectures). Its [`SpaceId`] is stored in
    /// [`Context::default_space`].
    pub fn new() -> Self {
        let mut ctx = Self::default();
        // SPACE_CONST = SpaceId(0): virtual space for constant/immediate values
        ctx.shared.spaces.push(Space::new(Some("const"), 1, 8));
        // default RAM space (SpaceId(1)); temp spaces start at SpaceId(2)
        let default_space = Space::new(Some("ram"), 1, 8);
        ctx.shared.default_space = ctx.shared.spaces.push(default_space);
        ctx
    }

    /// Returns the [`SpaceId`] for the named space, or `None` if it has not
    /// been registered.
    pub fn try_get_space(&self, name: &str) -> Option<SpaceId> {
        self.shared.named_spaces.get(name).copied()
    }

    /// Resolve a space by name for textual lowering: an already-registered named
    /// space, the default space when its name matches (the default `ram` space is
    /// not in `named_spaces`), or a freshly-registered RAM space otherwise. Used
    /// by the canonical `load(space:size, ptr)` / `store(...)` lowering.
    pub fn get_or_make_named_space(&mut self, name: &str) -> SpaceId {
        if let Some(id) = self.try_get_space(name) {
            return id;
        }
        let default_id = self.shared.default_space;
        if self.shared.spaces[default_id].name.as_deref() == Some(name) {
            return default_id;
        }
        let default = &self.shared.spaces[self.shared.default_space];
        let space = Space::new(Some(name), default.word_size, default.addr_size);
        self.add_space(space)
    }

    /// Adds a space to the context, registering its name, and returns its ID.
    pub fn add_space(&mut self, space: Space) -> SpaceId {
        let name_key: Option<Box<str>> = space.name.clone();
        let id = self.shared.spaces.push(space);
        if let Some(name) = name_key {
            self.shared.named_spaces.insert(name, id);
        }
        id
    }

    /// Returns the number of spaces registered in this context.
    pub fn space_count(&self) -> usize {
        self.shared.spaces.len()
    }

    pub fn set_primary_entrypoint(&mut self, entrypoint: Option<u64>) {
        self.shared.primary_entrypoint = entrypoint;
    }

    pub fn primary_entrypoint(&self) -> Option<u64> {
        self.shared.primary_entrypoint
    }

    /// Record the set of function entry addresses whose optimization the user
    /// asked to skip (`--ignore`). Per-function passes consult
    /// [`Context::is_function_ignored`] and skip these functions.
    pub fn set_ignored_functions(&mut self, addrs: HashSet<u64>) {
        self.shared.ignored_functions = addrs;
    }

    /// The function entry addresses whose optimization is being skipped.
    pub fn ignored_functions(&self) -> &HashSet<u64> {
        &self.shared.ignored_functions
    }

    /// Whether the function at `addr` was marked ignored (`--ignore`). A `None`
    /// address (synthetic functions with no entry) is never ignored.
    pub fn is_function_ignored(&self, addr: Option<u64>) -> bool {
        addr.is_some_and(|a| self.shared.ignored_functions.contains(&a))
    }

    /// Records the loaded binary's operating system (set by the loader from the
    /// container format).
    pub fn set_target_os(&mut self, os: TargetOs) {
        self.shared.target_os = os;
    }

    /// The loaded binary's operating system, or [`TargetOs::Unknown`].
    pub fn target_os(&self) -> TargetOs {
        self.shared.target_os
    }

    /// Replaces the spaces registry wholesale. Intended for initialization from a pre-built spec.
    pub fn load_spaces(&mut self, spaces: registry::Registry<SpaceId, Space>) {
        self.shared.spaces = spaces;
    }

    /// Read `n` bytes of initialized binary memory at virtual address `addr`,
    /// or `None` if any byte is unmapped. See [`MemoryImage::read_bytes`].
    ///
    /// [`MemoryImage::read_bytes`]: crate::memory_image::MemoryImage::read_bytes
    pub fn read_bytes(&self, addr: u64, n: usize) -> Option<Vec<u8>> {
        self.shared.memory_image.read_bytes(addr, n)
    }

    /// Read a little-endian unsigned integer of `size` bytes from initialized
    /// binary memory at `addr`. See [`MemoryImage::read_uint`].
    ///
    /// [`MemoryImage::read_uint`]: crate::memory_image::MemoryImage::read_uint
    pub fn read_uint(&self, addr: u64, size: usize) -> Option<u64> {
        self.shared.memory_image.read_uint(addr, size)
    }

    /// True if `addr` lies in an executable region of the loaded binary.
    pub fn is_executable_addr(&self, addr: u64) -> bool {
        self.shared.memory_image.is_executable(addr)
    }

    /// True only if `addr` is in a region *known* to be writable (protections
    /// established and the segment writable). Passes that fold a value out of
    /// initialized memory use this to refuse mutable memory — e.g. a GOT slot the
    /// dynamic linker overwrites at load time, whose file bytes are the lazy PLT
    /// resolver stub, not the real target. See
    /// [`MemoryImage::is_known_writable`](crate::memory_image::MemoryImage::is_known_writable).
    pub fn is_known_writable_addr(&self, addr: u64) -> bool {
        self.shared.memory_image.is_known_writable(addr)
    }

    /// Mark the binary's memory protections as established (the
    /// `memory_protections` pass has run), so executability checks narrow from the
    /// permissive default to the real per-segment flags.
    pub fn mark_protections_known(&mut self) {
        self.shared.memory_image.mark_protections_known();
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
        let bounds = self.shared.memory_image.segment_bounds(addr);
        if let Some((start, end)) = bounds
            && let Some(known) = self.known(Proposition::ExecutableMemory { start, end })
        {
            return known;
        }
        if !self.shared.memory_image.protections_known()
            || self.shared.memory_image.is_executable(addr)
        {
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
        self.shared.discoveries.insert(discovery)
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
        self.shared.discoveries.insert(
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
        self.shared.discoveries.drain()
    }

    /// Iterate pending discoveries without consuming them.
    pub fn discoveries(&self) -> impl Iterator<Item = &crate::discovery::Discovery> + '_ {
        self.shared.discoveries.iter()
    }

    /// True if there are no pending discoveries.
    pub fn has_no_discoveries(&self) -> bool {
        self.shared.discoveries.is_empty()
    }

    /// Every code address lifted in this context, as portable [`CodeSeed`]s. Used
    /// to export a "code map" that pre-seeds a later run of the same binary.
    ///
    /// [`CodeSeed`]: crate::discovery::CodeSeed
    pub fn lifted_code_seeds(&self) -> Vec<crate::discovery::CodeSeed> {
        self.shared.discoveries.lifted_seeds()
    }

    /// Enqueue exported [`CodeSeed`]s as pending discoveries so the lifter reaches
    /// them in its first pass. Call before lifting begins; seeds whose key already
    /// has a terminal outcome are ignored by the queue.
    ///
    /// [`CodeSeed`]: crate::discovery::CodeSeed
    pub fn seed_code(&mut self, seeds: impl IntoIterator<Item = crate::discovery::CodeSeed>) {
        for seed in seeds {
            self.shared.discoveries.insert(seed.into_discovery());
        }
    }

    pub fn mark_discovery_lifted(&mut self, key: crate::discovery::DiscoveryKey) {
        self.shared.discoveries.mark_lifted(key);
    }

    pub fn mark_discovery_failed(
        &mut self,
        key: crate::discovery::DiscoveryKey,
        reason: impl Into<String>,
    ) {
        self.shared.discoveries.mark_failed(key, reason);
    }

    pub fn mark_discovery_skipped(
        &mut self,
        key: crate::discovery::DiscoveryKey,
        reason: impl Into<String>,
    ) {
        self.shared.discoveries.mark_skipped(key, reason);
    }

    /// Returns the [`BlockId`] for a block at `addr`, creating one if needed.
    ///
    /// The newly created block is named after the address in hex and registered
    /// in the address map.
    pub fn get_or_make_block(&mut self, addr: u64, func: FunctionId) -> BlockId {
        let mut addresses = crate::address_index::AddressIndex::analyze(self);
        self.get_or_make_block_indexed(&mut addresses, addr, func)
    }

    /// Indexed construction variant of [`get_or_make_block`](Self::get_or_make_block).
    /// The caller owns `addresses` for the duration of its lifting/lowering
    /// operation and threads it through every address-bearing mutation.
    #[track_caller]
    pub fn get_or_make_block_indexed(
        &mut self,
        addresses: &mut crate::address_index::AddressIndex,
        addr: u64,
        func: FunctionId,
    ) -> BlockId {
        use crate::address_index::AddressTarget;

        if let Some(AddressTarget::Function(owner)) = addresses.get(addr) {
            assert_eq!(
                owner, func,
                "cannot create a block at an address owned by another function"
            );
        }
        let existing = match addresses.get(addr) {
            Some(AddressTarget::Block(block)) => Some(block),
            Some(AddressTarget::Function(function)) => FunctionBody::from_id(self, function)
                .root()
                .map(|root| root.id),
            None => None,
        };
        match existing {
            Some(block) => {
                if block.func != func {
                    let stored = FunctionBody::from_id(self, block.func);
                    let requested = FunctionBody::from_id(self, func);
                    // Blocks are stored in per-function arenas now; ownership is
                    // encoded by the qualified block id rather than a field on
                    // `BasicBlock`.
                    let parent = Some(block.func);
                    let caller = std::panic::Location::caller();
                    let detail = format!(
                        "cannot reuse a block stored in another function arena: block={block:?} address=0x{addr:x}; stored={:?} name={:?} entry={:?} parent={parent:?}; requested={:?} name={:?} entry={:?}; caller={caller}",
                        block.func,
                        stored.name(),
                        stored.address(),
                        func,
                        requested.name(),
                        requested.address(),
                    );
                    log::error!(
                        target: "qcode::arena",
                        "{detail}\nbacktrace:\n{}",
                        std::backtrace::Backtrace::force_capture()
                    );
                    panic!("{detail}");
                }
                block
            }
            None => {
                BasicBlock::make(self, func)
                    .with_address_indexed(addresses, addr)
                    .id
            }
        }
    }

    /// Borrows one function body and creates the concrete body-local builder
    /// positioned at `block`.
    pub fn builder(&mut self, block: BlockId) -> crate::builder::Builder<'str, '_> {
        let body = &mut self.bodies[block.func];
        crate::builder::Builder::new(body, &self.shared, &self.interfaces, block)
    }

    /// Test/API convenience for preparing a machine-address block before
    /// narrowing construction to its body-local builder.
    pub fn builder_at(&mut self, address: u64) -> crate::builder::Builder<'str, '_> {
        use crate::address_index::AddressTarget;

        let mut addresses = crate::address_index::AddressIndex::analyze(self);
        let block = match addresses.get(address) {
            Some(AddressTarget::Function(function)) => self.bodies[function]
                .root_id()
                .map(|local| BlockId::new(function, local))
                .unwrap_or_else(|| {
                    self.get_or_make_block_indexed(&mut addresses, address, function)
                }),
            Some(AddressTarget::Block(block)) => block,
            None => {
                let function = FunctionBody::make(self, Cow::Owned(format!("blk_{address:x}")))
                    .expect("anonymous host function")
                    .id;
                self.get_or_make_block_indexed(&mut addresses, address, function)
            }
        };
        let mut builder = self.builder(block);
        builder.set_address(address);
        builder
    }

    /// The forced rendering mode for a `Bytes` blob, or
    /// [`BytesDisplay::Auto`](crate::value::BytesDisplay::Auto) if unset.
    pub fn bytes_display(&self, id: crate::value::BytesId) -> crate::value::BytesDisplay {
        self.shared
            .values
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
            self.shared.values.bytes_display.remove(&id);
        } else {
            self.shared.values.bytes_display.insert(id, mode);
        }
    }

    /// Returns a list of all live blocks in the context (across all functions).
    pub fn block_ids(&self) -> Vec<BlockId> {
        self.functions().flat_map(|f| f.block_ids()).collect()
    }

    /// Returns all live instructions across all functions in stable logical-ID
    /// order. Function arenas iterate in dense physical order, so this explicit
    /// sort preserves the observable whole-context order across compaction.
    pub fn instruction_ids(&self) -> Vec<InstructionId> {
        let mut ids: Vec<_> = self.functions().flat_map(|f| f.instruction_ids()).collect();
        ids.sort_unstable();
        ids
    }

    /// Returns a list of all functions in the context.
    pub fn function_ids(&self) -> Vec<FunctionId> {
        self.interfaces.iter().map(|i| i.id).collect()
    }

    /// Mints a fresh, uniquely-named anonymous function and returns its id.
    ///
    /// A block must be born into some function's arena; this hands out a host
    /// for standalone blocks (tests, the raw-hex/bare-block lift paths, and the
    /// pyqcode API that build a block without an enclosing function).
    pub fn anon_function(&mut self) -> FunctionId {
        let name = self.get_unique_name(std::borrow::Cow::Borrowed("anon"));
        crate::value::FunctionBody::make(self, name)
            .expect("unique anon function name")
            .id
    }

    /// `(issued_ids, removed_ids)` across every function's instruction arena.
    pub fn instruction_arena_stats(&self) -> (usize, usize) {
        let mut total = 0;
        let mut dead = 0;
        for f in self.bodies.iter() {
            total += f.insns.issued_len();
            dead += f.insns.issued_len() - f.insns.len();
        }
        (total, dead)
    }

    /// Aggregate issued/live/dead and structural capacity for every body arena.
    ///
    /// This is the stable reporting surface used by the Stage 7 before/after
    /// probe. Keeping the aggregation here avoids exposing arena internals to
    /// measurement binaries.
    pub fn body_arena_stats(&self) -> crate::value::BodyArenaStats {
        let mut total = crate::value::BodyArenaStats::default();
        for body in self.bodies.iter() {
            total.add_assign(body.arena_stats());
        }
        total
    }

    /// Releases body-arena capacity retained from peak analysis churn in every
    /// function (see [`FunctionBody::shrink_to_fit`](crate::value::FunctionBody::shrink_to_fit)).
    ///
    /// Purely an allocator hint: IDs, ordering, and rendered IR are unchanged.
    /// Called once at explicit end-of-mutation boundaries such as pipeline
    /// convergence; nothing depends on it running.
    pub fn shrink_bodies_to_fit(&mut self) {
        for mut body in self.bodies.iter_mut() {
            body.shrink_to_fit();
        }
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
            inner: self.bodies.iter(),
        }
    }

    /// Iterates over all the functions in the context
    /// alias for `functions()`
    pub fn iter(&self) -> FunctionIter<'str, '_> {
        self.functions()
    }

    pub fn varnodes(&self) -> impl Iterator<Item = VarnodeRef<'str, '_>> + '_ {
        self.shared.varnodes()
    }

    /// Number of varnodes in the context. The varnode registry is append-only, so
    /// this is monotonic and an unchanged value means an unchanged varnode set —
    /// used to validate caches keyed on the register/varnode layout (e.g. the
    /// alias [`RegisterBase`](../../qcode_analysis/alias/struct.RegisterBase.html)).
    pub fn varnode_count(&self) -> usize {
        self.shared.varnode_count()
    }

    /// Removes a CFG edge, unlinking it from both incident blocks' edge sets and
    /// physically dropping its payload. The module-path (function-qualified)
    /// spelling of [`FunctionBody::remove_cfg_edge`].
    pub fn remove_cfg_edge(&mut self, func: FunctionId, edge_id: EdgeId) {
        self.bodies[func].remove_cfg_edge(edge_id);
    }

    /// Relocate every block in `olds` into `target`'s own arena. The originals
    /// remain owned by their source functions until deletion; only the clones are
    /// rostered in `target`, so ownership and storage never diverge. This is the storage
    /// mover [`split_function_at`](Self::split_function_at) uses to make a split-off
    /// tail self-stored.
    ///
    /// A pure storage move: the resulting IR is semantically identical. Every
    /// relocated block is deep-cloned into `target` (preserving instruction types,
    /// machine addresses, and labels), all intra-set value/block references are
    /// remapped to the clones, the incident CFG edges are rebuilt between the new
    /// blocks (and their unmoved neighbours), the block addresses and the function
    /// root are re-pointed, and the originals are deleted. `target`'s reverse-use
    /// map is rebuilt from its live instructions afterwards.
    ///
    /// Assumes the relocated set is closed (the caller strips every cross-function
    /// CFG edge and rewrites foreign terminator targets to `TailCall`s first): every
    /// reference from a relocated block resolves to another relocated block, an
    /// unmoved block of `target`, or a shared value; a reference into a *third*
    /// function is a bug upstream, and debug builds assert against it.
    pub fn rehome_owned_blocks(
        &mut self,
        addresses: &mut crate::address_index::AddressIndex,
        target: FunctionId,
        olds: &[BlockId],
    ) -> HashMap<BlockId, BlockId> {
        // Body-local temporary values and spaces move with blocks that reference
        // them. Collect the exact dependency closure first: operand/origin temps,
        // explicit load/store spaces, and pointer provenance carried by types.
        let mut needed_temps: HashSet<TempId> = HashSet::default();
        let mut needed_temp_spaces: HashSet<TempSpaceId> = HashSet::default();
        for &old in olds {
            for &param_local in &self.block(old).params {
                let param = self.block_param(BlockParamId::new(old.func, param_local));
                if let Some(crate::value::LocalValueId::Temp(temp)) = param.origin {
                    needed_temps.insert(TempId::new(old.func, temp));
                }
                if let Some(MemorySpaceId::Temp(space)) = self.shared.types.space_of(param.type_id)
                {
                    needed_temp_spaces.insert(space);
                }
            }
            for &insn_local in &self.block(old).instructions {
                let insn = self.instruction(InstructionId::new(old.func, insn_local));
                for arg in insn.mnemonic().args() {
                    if let crate::value::LocalValueId::Temp(temp) = arg {
                        needed_temps.insert(TempId::new(old.func, temp));
                    }
                }
                let explicit_space = match insn.mnemonic() {
                    Mnemonic::Load(load) => Some(load.space),
                    Mnemonic::Store(store) => Some(store.space),
                    _ => None,
                };
                if let Some(LocalMemorySpaceId::Temp(space)) = explicit_space {
                    needed_temp_spaces.insert(TempSpaceId::new(old.func, space));
                }
                if let Some(MemorySpaceId::Temp(space)) = self.shared.types.space_of(insn.type_id) {
                    needed_temp_spaces.insert(space);
                }
            }
        }
        for &temp in &needed_temps {
            let data = &self.bodies[temp.func].temps[temp.local];
            needed_temp_spaces.insert(TempSpaceId::new(temp.func, data.space));
        }

        let mut needed_temp_spaces: Vec<_> = needed_temp_spaces.into_iter().collect();
        needed_temp_spaces.sort_unstable();
        let mut temp_space_map: HashMap<TempSpaceId, TempSpaceId> = HashMap::default();
        for old in needed_temp_spaces {
            if old.func == target {
                continue;
            }
            let space = self.bodies[old.func].temp_spaces[old.local].clone();
            let new = self.bodies[target].push_temp_space(space);
            temp_space_map.insert(old, new);
        }

        let mut needed_temps: Vec<_> = needed_temps.into_iter().collect();
        needed_temps.sort_unstable();
        let mut value_map: HashMap<ValueId, ValueId> = HashMap::default();
        for old in needed_temps {
            if old.func == target {
                continue;
            }
            let mut temp = self.bodies[old.func].temps[old.local].clone();
            temp.space = temp_space_map[&TempSpaceId::new(old.func, temp.space)].local;
            if let Some(name) = temp.name.take() {
                temp.name = Some(self.bodies[target].names.unique(name));
            }
            let new = self.bodies[target].push_temp(temp);
            value_map.insert(ValueId::Temp(old), ValueId::Temp(new));
        }

        // Phase 1: structurally clone every block into `target`, accumulating the
        // remaining old -> new value and block maps.
        let mut block_map: HashMap<BlockId, BlockId> = HashMap::default();
        for &old in olds {
            let new = BasicBlock::clone_block_into(self, old, target, &mut value_map);
            block_map.insert(old, new);
        }

        // Phase 2: with the full map known, remap the clones' operands and block
        // targets (this resolves forward references between relocated blocks). The
        // cloned terminators still hold their source block's local targets, so the
        // remap needs the *old* arena (`old.func`) to qualify them before lookup.
        for (&old, &new) in &block_map {
            let old_params = self.block(old).params.clone();
            let new_params = self.block(new).params.clone();
            for (old_local, new_local) in old_params.into_iter().zip(new_params) {
                let old_param = BlockParamId::new(old.func, old_local);
                let new_param = BlockParamId::new(new.func, new_local);
                let type_id = remap_rehomed_type(
                    self,
                    self.block_param(new_param).type_id,
                    target,
                    &temp_space_map,
                );
                self.block_param_mut(new_param).type_id = type_id;
                let Some(origin) = self.block_param(new_param).origin else {
                    continue;
                };
                let qualified = origin.qualify(old.func);
                let remapped = value_map.get(&qualified).copied().unwrap_or(qualified);
                debug_assert!(
                    remapped.owning_function().is_none_or(|f| f == target),
                    "rehome: relocated block param {old_param:?} has an origin in another \
                     function ({qualified:?}); the relocated set is not closed",
                );
                self.block_param_mut(new_param).origin = Some(remapped.localize(new.func));
            }

            let insns = self.block(new).instructions.clone();
            for insn_local in insns {
                let insn_id = InstructionId::new(new.func, insn_local);
                let type_id = remap_rehomed_type(
                    self,
                    self.instruction(insn_id).type_id,
                    target,
                    &temp_space_map,
                );
                self.instruction_mut(insn_id).type_id = type_id;
                let mut mnemonic = self.instruction(insn_id).mnemonic().clone();
                let mut pairs = Vec::new();
                for arg in mnemonic.args() {
                    // The clone still holds its *source* arena's bare-local operands,
                    // so qualify with `old.func` to look them up and re-localize the
                    // mapped replacement against the clone's own arena (`new.func`).
                    let qualified = arg.qualify(old.func);
                    if let Some(&new_val) = value_map.get(&qualified) {
                        pairs.push((arg, new_val.localize(new.func)));
                    } else {
                        // An operand not in the map must resolve to `target` itself
                        // (an unmoved own block) or to a shared value — never into a
                        // third function. A cross-function data dependence would mean
                        // the split left the set non-closed (a bug upstream).
                        debug_assert!(
                            qualified.owning_function().is_none_or(|f| f == target),
                            "rehome: relocated block references a value in another \
                             function ({qualified:?}); the relocated set is not closed",
                        );
                    }
                }
                crate::value::block::substitute_operands(&mut mnemonic, &pairs);
                remap_rehomed_memory_space(&mut mnemonic, old.func, target, &temp_space_map);
                remap_block_targets(&mut mnemonic, old.func, new.func, &block_map);
                *self.instruction_mut(insn_id).mnemonic_mut() = mnemonic;
            }
        }

        // Phase 3: rebuild every CFG edge incident to a relocated block, retargeting
        // the moved endpoint(s) to the clone. Collect the incident edge ids first
        // (an edge between two relocated blocks appears in both edge sets — the set
        // dedups it).
        // A block's edge set holds bare body-local `EdgeId`s; recover the storing
        // function from the incident block itself (its own `id.func`).
        let mut incident: HashSet<(FunctionId, EdgeId)> = HashSet::default();
        for &old in olds {
            incident.extend(self.block(old).edges.iter().map(|&e| (old.func, e)));
        }
        let mut incident: Vec<_> = incident.into_iter().collect();
        incident.sort_unstable();
        for (edge_func, edge) in incident {
            let EdgeData { from, to } = *self.edge(edge_func, edge);
            let from = BlockId::new(edge_func, from);
            let to = BlockId::new(edge_func, to);
            let new_from = block_map.get(&from).copied().unwrap_or(from);
            let new_to = block_map.get(&to).copied().unwrap_or(to);
            self.add_cfg_edge(new_from, new_to);
        }

        // Phase 4: move each block's machine address onto its clone.
        for &old in olds {
            let Some(addr) = self.block(old).address else {
                continue;
            };
            let new = block_map[&old];
            let extra = self.block(old).extra_addresses.clone();
            self.block_mut(new).extra_addresses = extra;
            self.block_mut(new).address = Some(addr);
        }

        // Phase 5: delete the originals (unlinks their old edges, physically
        // removes their instructions and physical block payloads).
        for &old in olds {
            BasicBlock::from_id_mut(self, old).delete();
        }

        // Phase 6: rebuild `target`'s reverse-use map from its live instructions,
        // since phase 2 rewrote operands in place.
        self.rebuild_users(target);
        addresses.refresh(self);
        block_map
    }

    /// The function registered at `block`'s machine address, if any. Registration
    /// is the entry-boundary signal even while the function is a rootless stub:
    /// Path A never adopts a foreign-storage block merely because addresses match.
    fn function_registered_at_block(
        &self,
        addresses: &crate::address_index::AddressIndex,
        block: BlockId,
    ) -> Option<FunctionId> {
        self.block(block)
            .address
            .and_then(|addr| addresses.function_at(addr))
    }

    /// Blocks reachable from `block` along CFG edges, stopping at any *other*
    /// function's entry (the tail-call boundary). `block` itself is always included.
    /// The walk is owner-agnostic: it crosses blocks regardless of which function
    /// currently owns them (an absorbed tail is owned by the function that absorbed
    /// it, not by `g`), exactly like the settle's `claimed_from`. `g` is the function
    /// the tail is being reclaimed into, so `g`'s own entry (which is `block`) is not
    /// a boundary. Deterministically ordered (by machine address, then id) so the
    /// storage relocation that follows assigns ids reproducibly.
    fn split_tail(
        &self,
        addresses: &crate::address_index::AddressIndex,
        block: BlockId,
        g: FunctionId,
    ) -> Vec<BlockId> {
        let mut seen: HashSet<BlockId> = HashSet::default();
        seen.insert(block);
        let mut queue = vec![block];
        while let Some(b) = queue.pop() {
            let succs: Vec<BlockId> = BasicBlock::from_id(self, b)
                .successors()
                .map(|(_, s)| s)
                .collect();
            for s in succs {
                if seen.contains(&s) {
                    continue;
                }
                // A different function's entry is a tail-call boundary — never
                // crossed. `g`'s own entry is `block` (already seen), so this stops
                // only at *foreign* entries.
                if let Some(entry_func) = self.function_registered_at_block(addresses, s)
                    && entry_func != g
                {
                    continue;
                }
                seen.insert(s);
                queue.push(s);
            }
        }
        let mut tail: Vec<BlockId> = seen.into_iter().collect();
        tail.sort_unstable_by_key(|&b| (self.block(b).address, b.local, b.func));
        tail
    }

    /// Split at `block`, returning the function `G` whose entry is `block`. This is
    /// the strict-locality construction verb (context-split ruling 2): a control
    /// transfer that lands mid-function is modelled as a *function split* — never a
    /// foreign block reference.
    ///
    /// Concretely it: (i) reuses the function already registered at `block`'s address
    /// (a stub minted by a `call`, which may already have adopted `block` as its
    /// root) or mints a conventional `fn_<addr>` (synthesized interface, unknown ABI
    /// — the optimization pipeline derives its purity/clobber/ABI facts later);
    /// (ii) extracts the tail reachable from `block`, stopping at other function
    /// entries ([`split_tail`](Self::split_tail)), and reassigns it to `G` (an
    /// absorbed tail may currently be owned by the function that absorbed it);
    /// (iii) rewrites every terminator that statically targeted `block` — in the
    /// absorbing function and in any already-lifted caller — into a function-level
    /// [`TailCall`](crate::value::insn::TailCall) (`G` for an unconditional `Branch`;
    /// a fresh intra-function trampoline block ending in a `TailCall` for a
    /// conditional `CBranch` arm), strips every cross-function CFG edge incident to
    /// the moved tail, and rewrites any foreign back-edge out of the tail the same
    /// way; (iv) relocates the tail into `G`'s own arena
    /// ([`rehome_owned_blocks`](Self::rehome_owned_blocks)) so `G` is self-stored.
    /// Afterwards no foreign block reference and no cross-function edge survives.
    ///
    /// `block` must carry a machine address.
    pub fn split_function_at(&mut self, block: BlockId) -> FunctionId {
        let mut addresses = crate::address_index::AddressIndex::analyze(self);
        self.split_function_at_indexed(&mut addresses, block)
    }

    /// Indexed construction variant of
    /// [`split_function_at`](Self::split_function_at).
    pub fn split_function_at_indexed(
        &mut self,
        addresses: &mut crate::address_index::AddressIndex,
        block: BlockId,
    ) -> FunctionId {
        use crate::value::insn::{Branch, CBranch, Callee, TailCall};

        let addr = self
            .block(block)
            .address
            .expect("split_function_at: block has no machine address");

        // G: reuse an existing function at this address (a call-minted stub that
        // may carry a symbol name), else mint a conventional one. A block already
        // stored elsewhere at this address is not adopted; relocation below creates
        // and roots a self-stored clone.
        let g = match addresses.function_at(addr) {
            Some(existing) => existing,
            None => FunctionBody::make_at_addr_indexed(self, addresses, addr, None).id,
        };

        // The tail is computed on the pre-split CFG (cross-function edges intact) so
        // the reach walk is exact — matching the settle's `claimed_from`.
        let tail = self.split_tail(addresses, block, g);
        let tail_set: HashSet<BlockId> = tail.iter().copied().collect();

        // Every function that currently owns a tail block loses those blocks; record
        // them so their `instruction_addrs` can be rebuilt afterwards.
        let mut prev_owners: HashSet<FunctionId> = HashSet::default();
        for &b in &tail {
            // Ownership is derived from the storing arena (`b.func`).
            prev_owners.insert(b.func);
        }

        // Treat the tail as G-owned while computing boundary rewrites, without
        // ever adopting its foreign-storage blocks into G's roster/root. The
        // physical move below is the only supported ownership transition.
        let effective_owner = |_ctx: &Context, candidate: BlockId| {
            if tail_set.contains(&candidate) {
                Some(g)
            } else {
                // Ownership is derived from the storing arena.
                Some(candidate.func)
            }
        };

        // Resolve a static terminator target to the foreign function whose *entry* it
        // is, from the perspective of `owner`.
        let foreign_entry =
            |ctx: &Context, target: BlockId, owner: FunctionId| -> Option<FunctionId> {
                let callee = if target == block {
                    g
                } else {
                    ctx.function_registered_at_block(addresses, target)?
                };
                (callee != owner).then_some(callee)
            };

        // Collect terminator rewrites: (a) any terminator that statically targets
        // `block` (G's new entry) — the origin's own branch into the tail and any
        // already-lifted caller; (b) any terminator in the moved tail whose target
        // is now a foreign entry (a boundary tail-call, or a back-edge into the
        // origin's retained entry). Both must become function-level `TailCall`s.
        let mut tail_calls: Vec<(InstructionId, FunctionId)> = Vec::new();
        let mut cond_calls: Vec<(InstructionId, BlockId, FunctionId)> = Vec::new();
        let relevant: Vec<BlockId> = self.block_ids();
        for b in relevant {
            let Some(owner) = effective_owner(self, b) else {
                continue;
            };
            let Some((term_id, mnemonic)) = BasicBlock::from_id(self, b)
                .instructions()
                .last()
                .map(|t| (t.id, t.mnemonic().clone()))
            else {
                continue;
            };
            // Terminator targets are bare body-local indices in the block's own
            // arena (`b.func`); qualify to recover the full `BlockId`.
            match mnemonic {
                Mnemonic::Branch(Branch { target, .. }) => {
                    if let Some(callee) = foreign_entry(self, BlockId::new(b.func, target), owner) {
                        tail_calls.push((term_id, callee));
                    }
                }
                Mnemonic::CBranch(CBranch {
                    success_block,
                    failure_block,
                    ..
                }) => {
                    if let Some(callee) =
                        foreign_entry(self, BlockId::new(b.func, success_block), owner)
                    {
                        cond_calls.push((term_id, b, callee));
                    }
                    if let Some(callee) =
                        foreign_entry(self, BlockId::new(b.func, failure_block), owner)
                    {
                        cond_calls.push((term_id, b, callee));
                    }
                }
                _ => {}
            }
        }

        for (insn, callee) in tail_calls {
            self.replace_instruction_mnemonic(
                insn,
                Mnemonic::TailCall(TailCall {
                    target: Callee::Real(callee),
                    args: vec![],
                }),
            );
        }
        for (insn, owner_block, callee) in cond_calls {
            // Ownership is derived from the storing arena.
            let owner = owner_block.func;
            let tramp = BasicBlock::make(self, owner).id;
            (self).builder(tramp).push_tail_call(callee);
            self.add_cfg_edge(owner_block, tramp);

            let Mnemonic::CBranch(mut cb) = self.instruction(insn).mnemonic().clone() else {
                continue;
            };
            // The CBranch and its targets share the terminator's arena (`insn.func`);
            // qualify the local targets to compare, localize `tramp` on the way in.
            if foreign_entry(self, BlockId::new(insn.func, cb.success_block), owner) == Some(callee)
            {
                cb.success_block = tramp.localize(insn.func);
            }
            if foreign_entry(self, BlockId::new(insn.func, cb.failure_block), owner) == Some(callee)
            {
                cb.failure_block = tramp.localize(insn.func);
            }
            self.replace_instruction_mnemonic(insn, Mnemonic::CBranch(cb));
        }

        // Strip every cross-function CFG edge incident to a moved tail block; the
        // reach walk already stopped at these boundaries, so removing them cannot
        // change ownership — it only closes each function's graph over its own
        // blocks (a precondition of the storage relocation below).
        let mut stale: HashSet<(FunctionId, EdgeId)> = HashSet::default();
        for &b in &tail {
            for edge in self.block(b).edges.iter().copied() {
                let &EdgeData { from, to } = self.edge(b.func, edge);
                let from = BlockId::new(b.func, from);
                let to = BlockId::new(b.func, to);
                let cross = effective_owner(self, from) != effective_owner(self, to);
                let touches_tail = tail_set.contains(&from) || tail_set.contains(&to);
                if cross && touches_tail {
                    stale.insert((b.func, edge));
                }
            }
        }
        let mut stale: Vec<_> = stale.into_iter().collect();
        stale.sort_unstable();
        for (func, edge) in stale {
            self.remove_cfg_edge(func, edge);
        }

        // Storage move: relocate the tail into G's own arena (self-stored). The set
        // is now closed (all cross-function edges stripped, foreign targets rewritten
        // to `TailCall`s), so the relocation's closure assumptions hold.
        let moved = self.rehome_owned_blocks(addresses, g, &tail);
        self.bodies[g].set_root_id(Some(moved[&block].local));

        // Rebuild `instruction_addrs` on G and on every function that lost blocks.
        self.recompute_instruction_addrs(g);
        for owner in prev_owners {
            if owner != g {
                self.recompute_instruction_addrs(owner);
            }
        }

        g
    }

    /// Rebuild `func`'s `instruction_addrs` from the machine addresses of the
    /// instructions in its current blocks.
    fn recompute_instruction_addrs(&mut self, func: FunctionId) {
        let blocks = FunctionBody::from_id(self, func).block_ids();
        let mut addrs = std::collections::BTreeSet::new();
        for b in blocks {
            for insn in BasicBlock::from_id(self, b).instructions() {
                if let Some(a) = insn.address() {
                    addrs.insert(a);
                }
            }
        }
        self.bodies[func].instruction_addrs = addrs;
    }

    /// Rebuild `func`'s reverse-use map (`users`) from scratch by scanning its live
    /// instructions' operands. Mirrors the per-operand recording in
    /// [`Context::push_insn`](crate::context::Context::push_insn).
    fn rebuild_users(&mut self, func: FunctionId) {
        let live: Vec<InstructionId> = FunctionBody::from_id(self, func).instruction_ids();
        let users = &mut self.bodies[func].users;
        users.clear();
        for id in live {
            let args = self.bodies[func].insns[id.local].mnemonic().args();
            let users = &mut self.bodies[func].users;
            for arg in args {
                users.entry(arg).or_default().push(id.localize(func));
            }
        }
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
        match self.shared.values.truths.get(&prop) {
            Some(t) => t.value == value,
            None => {
                self.shared.values.truths.insert(
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
        let novel = match self.shared.values.truths.get(&prop) {
            Some(prior) => {
                // Proving the opposite of an already-*known* fact (e.g. a user
                // override the analysis disproves) is not a replay signal: record
                // it as a hard contradiction and keep the original known value so
                // the driver can surface an error and terminate.
                if prior.certainty == Certainty::Known && prior.value != value {
                    self.shared
                        .values
                        .known_contradictions
                        .push(KnownContradiction {
                            prop,
                            known: prior.value,
                            proven: value,
                            known_pass: prior.pass,
                            proven_pass: pass,
                        });
                    return false;
                }
                if prior.certainty == Certainty::Assumed && prior.value != value {
                    self.shared.values.violations.push(Violation {
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
        self.shared.values.truths.insert(
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
        let prior = self.shared.values.truths.insert(
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
        self.shared.values.truths.get(&prop).copied()
    }

    /// The proven value of `prop`: `Some` only for *known* entries.
    pub fn known(&self, prop: Proposition) -> Option<bool> {
        self.truth(prop)
            .filter(|t| t.certainty == Certainty::Known)
            .map(|t| t.value)
    }

    /// Iterates over every recorded truth (assumed and known).
    pub fn truths(&self) -> impl Iterator<Item = (Proposition, Truth)> + '_ {
        self.shared.values.truths.iter().map(|(&p, &t)| (p, t))
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
        &self.shared.values.violations
    }

    /// Facts proven this round that contradicted an existing *known* fact (e.g. a
    /// user override the analysis disproved). Non-empty means the analysis cannot
    /// honor the forced value; the driver surfaces this as a hard error.
    pub fn known_contradictions(&self) -> &[KnownContradiction] {
        &self.shared.values.known_contradictions
    }

    /// Returns the raw `u64` backing value of the literal `id`.
    pub fn get_literal_value(&self, id: LiteralId) -> u64 {
        self.shared.values.literals[id].value
    }

    /// Returns an immutable reference to the instruction identified by `id`.
    pub fn get_insn(&self, id: InstructionId) -> InstructionRef<'str, '_> {
        InstructionRef::from_id(self, id)
    }

    /// The function *body* `fid` (context-split stage 5a bridging accessor).
    ///
    /// Names the owning function explicitly so IR reads route through the body's
    /// function-local raw accessors — `ctx.body(fid).block(id)` in place of the
    /// globally routed `BasicBlock::from_id(ctx, id)`. This is the module-scope
    /// (`&Context`) obtain-form; a function pass reaches the same body accessors
    /// through its checked-out host. After the stage-4 `func`-strip only this
    /// obtain step changes (the caller already holds `&FunctionBody`); the
    /// `.block(id)` call on the result is unchanged.
    pub fn body(&self, fid: FunctionId) -> &crate::value::FunctionBody<'str> {
        &self.bodies[fid]
    }

    /// The function *body* `fid`, mutably (see [`Context::body`]).
    pub fn body_mut(&mut self, fid: FunctionId) -> &mut crate::value::FunctionBody<'str> {
        &mut self.bodies[fid]
    }

    // ----- Composite-id arena routing (moved off `ValueRegistry` in the
    // context-split reshape: function bodies now live in `Context.bodies`, so the
    // accessors that route a `(FunctionId, Local)` id to its arena are inherent on
    // `Context`). Each reads/writes `self.bodies[id.func]`. -----

    /// Appends an instruction to `func`'s body and records all its operands in the
    /// `users` map.
    ///
    /// # Immutability invariant
    ///
    /// Instructions are considered immutable after this call. If you alter the
    /// operands of an instruction after insertion the `users` map will be stale.
    /// Rewrite operands through [`replace_all_uses_with`](Self::replace_all_uses_with)
    /// instead.
    pub fn push_insn(&mut self, func: FunctionId, insn: Instruction<'str>) -> InstructionId {
        let args = insn.mnemonic().args();
        let local = self.bodies[func].insns.push(insn);
        let id = InstructionId::new(func, local);
        for arg in args {
            self.bodies[func]
                .users
                .entry(arg)
                .or_default()
                .push(id.localize(func));
        }
        id
    }

    /// Borrows the instruction `id`, routing through its owning function's arena.
    pub fn instruction(&self, id: InstructionId) -> &Instruction<'str> {
        &self.bodies[id.func].insns[id.local]
    }

    /// Mutably borrows the instruction `id`.
    pub fn instruction_mut(&mut self, id: InstructionId) -> &mut Instruction<'str> {
        &mut self.bodies[id.func].insns[id.local]
    }

    /// Whether `id` currently names a live instruction payload.
    pub fn contains_instruction(&self, id: InstructionId) -> bool {
        Into::<usize>::into(id.func) < self.bodies.len()
            && self.bodies[id.func].insns.contains(id.local)
    }

    /// Borrows the basic block `id`.
    pub fn block(&self, id: BlockId) -> &BasicBlock<'str> {
        &self.bodies[id.func].blocks[id.local]
    }

    /// Mutably borrows the basic block `id`.
    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock<'str> {
        &mut self.bodies[id.func].blocks[id.local]
    }

    /// Whether `id` currently names a live block payload.
    pub fn contains_block(&self, id: BlockId) -> bool {
        Into::<usize>::into(id.func) < self.bodies.len()
            && self.bodies[id.func].blocks.contains(id.local)
    }

    /// Borrows the block parameter `id`.
    pub fn block_param(&self, id: BlockParamId) -> &BlockParam<'str> {
        &self.bodies[id.func].params[id.local]
    }

    /// Mutably borrows the block parameter `id`.
    pub fn block_param_mut(&mut self, id: BlockParamId) -> &mut BlockParam<'str> {
        &mut self.bodies[id.func].params[id.local]
    }

    /// Whether `id` currently names a live block-parameter payload.
    pub fn contains_block_param(&self, id: BlockParamId) -> bool {
        Into::<usize>::into(id.func) < self.bodies.len()
            && self.bodies[id.func].params.contains(id.local)
    }

    /// Borrows the CFG edge `id`, stored in function `func`'s edge arena.
    pub fn edge(&self, func: FunctionId, id: EdgeId) -> &EdgeData {
        &self.bodies[func].edges[id]
    }

    /// Mutably borrows the CFG edge `id`, stored in function `func`'s edge arena.
    pub fn edge_mut(&mut self, func: FunctionId, id: EdgeId) -> &mut EdgeData {
        &mut self.bodies[func].edges[id]
    }

    /// Returns the instructions that use `value` as an operand, read from
    /// `value`'s owning function. For an SSA def (instruction/param) that is the
    /// complete user set (all uses are intra-function). For a shared value
    /// (literal/bytes/varnode) there is no single owner, so this returns `&[]`.
    pub fn users_of(&self, value: ValueId) -> Vec<InstructionId> {
        match value.owning_function() {
            Some(func) => self.bodies[func].users_of(value),
            None => Vec::new(),
        }
    }

    pub fn push_block(&mut self, func: FunctionId, block: BasicBlock<'str>) -> BlockId {
        let local = self.bodies[func].blocks.push(block);
        let id = BlockId::new(func, local);
        // A block is born owned by the function whose arena stores it.
        self.bodies[func].roster.push(local);
        id
    }

    pub fn push_block_param(&mut self, func: FunctionId, param: BlockParam<'str>) -> BlockParamId {
        let local = self.bodies[func].params.push(param);
        BlockParamId::new(func, local)
    }

    pub fn push_edge(&mut self, func: FunctionId, edge: EdgeData) -> EdgeId {
        self.bodies[func].edges.push(edge)
    }

    /// Push a function's interface and body in lockstep, returning the shared
    /// [`FunctionId`]. Both registries must always grow together.
    pub fn push_function(
        &mut self,
        interface: crate::value::function::FunctionInterface<'str>,
        body: FunctionBody<'str>,
    ) -> FunctionId {
        let expected = FunctionId::from(self.bodies.len());
        assert_eq!(
            body.id(),
            expected,
            "function body id does not match its registry slot"
        );
        let id = self.bodies.push(body);
        let iid = self.interfaces.push(interface);
        debug_assert_eq!(
            Into::<usize>::into(id),
            Into::<usize>::into(iid),
            "function body/interface registries drifted"
        );
        id
    }

    /// Returns an immutable reference to the varnode mapped to the named
    /// register `id`.
    pub fn get_register(&self, id: RegisterId) -> VarnodeRef<'str, '_> {
        Varnode::from_id(self, self.shared.registers[&id])
    }

    /// Creates a [`Value`] representing an integer constant of the given byte width.
    pub fn get_const(&self, value: u64, size: usize) -> LiteralRef<'str, '_> {
        let type_id = self.shared.types.get_or_make_int(size);
        let id = self
            .shared
            .values
            .get_or_make_typed_literal(value, type_id, size);
        LiteralRef::from_id(self, id)
    }

    /// Creates a `bool`-typed constant (`true`/`false`), byte-stored with value
    /// `1`/`0`. This is the only way to mint a `bool` literal.
    pub fn get_bool_const(&self, value: bool) -> LiteralRef<'str, '_> {
        let type_id = self.shared.types.get_or_make_bool();
        let id = self
            .shared
            .values
            .get_or_make_typed_literal(u64::from(value), type_id, 1);
        LiteralRef::from_id(self, id)
    }

    /// Creates a typed constant literal.
    ///
    /// Unlike [`get_const`](Self::get_const) this accepts an arbitrary [`TypeId`],
    /// allowing StackAddress constants (e.g. the stack base) to preserve their
    /// type through constant folding.
    pub fn get_typed_const(
        &self,
        value: u64,
        type_id: crate::types::TypeId,
    ) -> LiteralRef<'str, '_> {
        let size = self.shared.types.size_of(type_id);
        let id = self
            .shared
            .values
            .get_or_make_typed_literal(value, type_id, size);
        LiteralRef::from_id(self, id)
    }

    /// Creates an opaque byte-blob constant from a little-endian, memory-order
    /// byte vector.
    ///
    /// The blob is typed as an `Array(i8, data.len())`. Unlike numeric literals,
    /// byte blobs are **not interned**: every call produces a fresh
    /// [`BytesId`](crate::value::BytesId). Use this for constants wider than a
    /// `u64` (SSE/AVX pools, wide stack/memory reads, coalesced constant stores).
    pub fn get_bytes(&self, data: Vec<u8>) -> crate::value::BytesRef<'str, '_> {
        let i8_ty = self.shared.types.get_or_make_int(1);
        let type_id = self.shared.types.get_or_make_array(i8_ty, data.len());
        self.get_typed_bytes(data, type_id)
    }

    /// Like [`get_bytes`](Self::get_bytes) but stamps the blob with an explicit
    /// array/sequence [`TypeId`] instead of the default `Array(i8, len)`. Mints
    /// through the `&self` append path (no post-hoc `type_id` write), so a
    /// checked-out function pass reading through a [`BodyView`] can materialize a
    /// typed constant array without mutable access to the shared registry.
    pub fn get_typed_bytes(
        &self,
        data: Vec<u8>,
        type_id: crate::types::TypeId,
    ) -> crate::value::BytesRef<'str, '_> {
        let id = self
            .shared
            .values
            .bytes
            .push(crate::value::Bytes { data, type_id });
        crate::value::BytesRef::from_id(self, id)
    }

    /// Returns the [`TypeId`] of any [`ValueId`] in this context.
    ///
    /// Varnodes are typed as `Int(varnode.size())`. Blocks, functions, and other
    /// non-data values return `Int(0)`.
    pub fn type_of(&self, id: ValueId) -> crate::types::TypeId {
        match id {
            ValueId::Literal(lid) => self.shared.values.literals[lid].type_id,
            ValueId::Bytes(bid) => self.shared.values.bytes[bid].type_id,
            ValueId::Instruction(iid) => self.instruction(iid).type_id,
            ValueId::BlockParam(pid) => self.block_param(pid).type_id,
            ValueId::Varnode(vid) => {
                if let Some(&ty) = self.shared.values.varnode_types.get(&vid) {
                    return ty;
                }
                let size = self.shared.values.varnodes[vid].size_bytes();
                self.shared.types.get_or_make_int(size)
            }
            ValueId::Temp(id) => self
                .shared
                .types
                .get_or_make_int(self.bodies[id.func].temps[id.local].size),
            // Exhaustive on purpose: a new ValueId variant must decide its type
            // here rather than silently inheriting the zero-width fallback.
            ValueId::BasicBlock(_) | ValueId::Function(_) => self.shared.types.get_or_make_int(0),
        }
    }

    /// Returns the stored [`TypeId`] for value kinds that carry one directly.
    ///
    /// Unlike [`Context::type_of`], this never interns fallback integer types,
    /// so it works from immutable formatting and parsing paths. Varnodes,
    /// blocks, and functions return `None`.
    pub fn stored_type_of(&self, id: ValueId) -> Option<crate::types::TypeId> {
        match id {
            ValueId::Literal(lid) => Some(self.shared.values.literals[lid].type_id),
            ValueId::Bytes(bid) => Some(self.shared.values.bytes[bid].type_id),
            ValueId::Instruction(iid) => Some(self.instruction(iid).type_id),
            ValueId::BlockParam(pid) => Some(self.block_param(pid).type_id),
            ValueId::Varnode(vid) => self.shared.values.varnode_types.get(&vid).copied(),
            ValueId::Temp(_) => None,
            ValueId::BasicBlock(_) | ValueId::Function(_) => None,
        }
    }

    /// Gives `varnode` a global type override, replacing the default
    /// `Int(size)`. Used to type ambient register globals — e.g. the `FS_OFFSET`
    /// segment base as `PtrTo<TEB>` — so every use across all functions reads the
    /// richer type. Pass a type whose size matches the varnode's width.
    pub fn set_varnode_type(&mut self, varnode: VarnodeId, type_id: crate::types::TypeId) {
        self.shared.values.varnode_types.insert(varnode, type_id);
    }

    /// Return all instructions that use `value` as an operand.
    ///
    /// For an SSA value (instruction result or block param) this is the complete
    /// user set, read from its owning function. For a shared value
    /// (literal/bytes/varnode) it is `&[]` — those have no owning function and
    /// their uses are tracked per using-function; use
    /// [`users_across_functions`](Self::users_across_functions) to find them.
    pub fn users(&self, value: impl Into<ValueId>) -> Vec<InstructionId> {
        self.users_of(value.into())
    }

    /// Every instruction across all functions that uses `value` as an operand.
    /// Unlike [`users`](Self::users) this scans every function, so it answers a
    /// shared value (literal/bytes/varnode) whose uses span functions. Off the
    /// hot path (allocates); prefer [`users`](Self::users) for an SSA value.
    pub fn users_across_functions(&self, value: impl Into<ValueId>) -> Vec<InstructionId> {
        let value = value.into();
        if value.owning_function().is_some() {
            self.users_of(value)
        } else {
            self.functions().flat_map(|f| f.users_of(value)).collect()
        }
    }

    // ---- module read/mint surface (context-split stage 5b-ii Pin A) ----------
    //
    // Module-scope read accessors and type-minting verbs, mirrored on the
    // checked-out `BodyMut` pass host, so the module walker and the
    // module-scope GVN sub-passes read/mint over `&mut Context` directly.
    // `function{,_mut}` alias the existing `body{,_mut}`.

    /// A `Copy` read view over the whole module (for the mutation refs' reads).
    pub fn view(&self) -> ModuleView<'_, 'str> {
        ModuleView::new(self)
    }
    /// The module's shared IR state ([`Shared`]) — the module-path twin of
    /// [`ModuleView::shared`]/[`BodyMut::shr`], so a `&mut Context` module walker and
    /// a checked-out pass spell shared-data reads identically (context-split
    /// stage 5b-ii item #1).
    pub fn shr(&self) -> &Shared<'str> {
        &self.shared
    }
    /// The owning function's storage (read). Alias of [`body`](Self::body).
    pub fn function(&self, f: FunctionId) -> &FunctionBody<'str> {
        &self.bodies[f]
    }
    /// The owning function's storage (write). Alias of [`body_mut`](Self::body_mut).
    pub fn function_mut(&mut self, f: FunctionId) -> &mut FunctionBody<'str> {
        &mut self.bodies[f]
    }

    /// A read [`BlockRef`](crate::value::BlockRef) over `id`, module-routed.
    pub fn block_ref(&self, id: BlockId) -> BlockRef<'str, '_, ModuleView<'_, 'str>> {
        self.view().block_ref(id)
    }
    /// A read [`InstructionRef`] over `id`, module-routed.
    pub fn insn_ref(&self, id: InstructionId) -> InstructionRef<'str, '_, ModuleView<'_, 'str>> {
        self.view().insn_ref(id)
    }
    /// A read [`BlockParamRef`](crate::value::BlockParamRef) over `id`.
    pub fn param_ref(&self, id: BlockParamId) -> BlockParamRef<'str, '_, ModuleView<'_, 'str>> {
        self.view().param_ref(id)
    }
    /// A read [`FunctionRef`] over `id`, module-routed.
    pub fn function_ref(&self, id: FunctionId) -> FunctionRef<'str, '_, ModuleView<'_, 'str>> {
        self.view().function_ref(id)
    }

    /// Mint an `Int(size)`-typed instruction with `mnemonic` into `func`'s arena.
    pub fn push_mnemonic(
        &mut self,
        func: FunctionId,
        mnemonic: Mnemonic,
        size: usize,
    ) -> InstructionId {
        let type_id = self.shared.types.get_or_make_int(size);
        self.push_insn(func, Instruction::new(type_id, mnemonic))
    }

    /// Mint an instruction with `mnemonic` and explicit result `type_id` into
    /// `func`'s arena.
    pub fn push_mnemonic_with_type(
        &mut self,
        func: FunctionId,
        mnemonic: Mnemonic,
        type_id: crate::types::TypeId,
    ) -> InstructionId {
        self.push_insn(func, Instruction::new(type_id, mnemonic))
    }

    /// Mint a fresh empty block into `func`'s arena, owned (arena membership) and
    /// rostered. The module-scope mint of a fresh empty block.
    pub fn make_block(&mut self, func: FunctionId) -> BlockId {
        self.push_block(func, BasicBlock::detached())
    }

    /// Register `name` for `id` in the table that owns its kind (function-local
    /// for block/insn/param/Temp, global otherwise).
    pub fn register_local_name(
        &mut self,
        id: ValueId,
        name: Cow<'str, str>,
        old_name: Option<&str>,
    ) -> Result<()> {
        let existing = match id.name_scope_function() {
            Some(func) => self
                .function(func)
                .names
                .get(&name)
                .map(|id| id.qualify(func)),
            None => self.get_named(&name),
        };
        if let Some(existing) = existing {
            return if existing == id {
                Ok(())
            } else {
                Err(Error::spanless(ErrorTy::DuplicateName(name.to_string())))
            };
        }
        match id.name_scope_function() {
            Some(func) => self
                .function_mut(func)
                .names
                .register(name, id.localize(func), old_name),
            None => self.update_name(name, id, old_name),
        }
    }

    /// Registers an address in a caller-owned construction index.
    pub(crate) fn set_address_indexed(
        &mut self,
        addresses: &mut crate::address_index::AddressIndex,
        addr: u64,
        id: ValueId,
    ) -> crate::error::Result<()> {
        let target = match id {
            ValueId::Function(id) => crate::address_index::AddressTarget::Function(id),
            ValueId::BasicBlock(id) => crate::address_index::AddressTarget::Block(id),
            _ => unreachable!("only functions and blocks have module addresses"),
        };
        addresses.register(self, addr, target)
    }

    /// Changes the name of a value, in the name table that owns its kind
    /// (function-local for block/instruction/param/Temp, global otherwise).
    pub fn update_name(
        &mut self,
        name: Cow<'str, str>,
        id: ValueId,
        old_name: Option<&str>,
    ) -> Result<()> {
        match id.name_scope_function() {
            Some(func) => self.bodies[func]
                .names
                .register(name, id.localize(func), old_name),
            None => self.shared.name_map.register(name, id, old_name),
        }
    }

    /// Resolve `name` in the table that owns `id`'s kind (function-local for
    /// block/instruction/param/Temp, global otherwise). Used by the rename path to
    /// check for a conflict in the correct namespace, and by passes that mint a
    /// unique name for a known SSA value.
    pub fn get_named_in_scope(&self, id: ValueId, name: &str) -> Option<ValueId> {
        match id.name_scope_function() {
            Some(func) => self.bodies[func].names.get(name).map(|id| id.qualify(func)),
            None => self.shared.name_map.get(name),
        }
    }

    /// Remove `name` from the name map, keeping the [`get_unique_name`] suffix
    /// hint exact: if `name` is a generated `base_<n>` suffix, lower `base`'s hint
    /// so the freed suffix is reconsidered on the next call (a naive first-free
    /// scan would reuse it, and the hint must not skip it). Un-suffixed names are
    /// Attempts to get a value ID by its *global* name (function/varnode/space/
    /// p-code/bytes). Block/instruction/param/Temp names are function-scoped and are
    /// resolved through their owning [`FunctionBody`] (see [`NameTable`]); this
    /// returns `None` for them.
    pub fn get_named(&self, name: &str) -> Option<ValueId> {
        self.shared.name_map.get(name)
    }

    /// Gets a unique **global** name (functions, varnodes, spaces, …), appending
    /// a numeric suffix until free. For a block/instruction/param/Temp name, use
    /// [`get_unique_name_in`](Self::get_unique_name_in) so uniqueness is checked
    /// against the owning function's table.
    pub fn get_unique_name(&mut self, name: Cow<'str, str>) -> Cow<'str, str> {
        self.shared.name_map.unique(name)
    }

    /// Gets a unique name within `func`'s function-local name table (for block,
    /// instruction, block-param, and Temp names). Two functions may thus reuse the same
    /// name independently.
    pub fn get_unique_name_in(&mut self, func: FunctionId, name: Cow<'str, str>) -> Cow<'str, str> {
        self.bodies[func].names.unique(name)
    }
}

/// A name → value reverse map with amortized unique-name minting.
///
/// The context keeps one **global** table for module-scoped values (functions,
/// varnodes, spaces, p-code ops, byte blobs); each [`FunctionBody`](crate::value::FunctionBody)
/// keeps its **own** table for its block/instruction/param/Temp names. Keeping those
/// namespaces independent is a prerequisite for running function passes in
/// parallel: a worker mints names against its function's table with no global
/// lock and no cross-function collisions. Two functions may each name a block
/// `loop` — they render correctly because a value's own `name` field is the
/// source of truth; this table only enforces uniqueness and resolves by name.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct NameTable<'str, Id = ValueId> {
    /// name → the value that holds it.
    map: HashMap<Cow<'str, str>, Id>,
    /// Per-base "next suffix to try" lower-bound hints for [`unique`](Self::unique),
    /// so probing resumes instead of rescanning from `0`. A derived cache: rides
    /// through `clone` but is not serialized (see [`Context::get_unique_name`]).
    #[serde(skip)]
    suffix_hint: HashMap<String, u32>,
}

impl<Id> Default for NameTable<'_, Id> {
    fn default() -> Self {
        Self {
            map: HashMap::default(),
            suffix_hint: HashMap::default(),
        }
    }
}

impl<'str, Id: Copy + Eq> NameTable<'str, Id> {
    pub(crate) fn entries(&self) -> impl Iterator<Item = (&str, Id)> + '_ {
        self.map.iter().map(|(name, &value)| (name.as_ref(), value))
    }

    /// The value currently holding `name`, if any.
    pub fn get(&self, name: &str) -> Option<Id> {
        self.map.get(name).copied()
    }

    /// Whether `name` is taken.
    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// Register `name` for `id`, forgetting `old_name` first. Errors if `name`
    /// is already taken (callers pre-check via [`get`](Self::get), so this only
    /// fires defensively).
    pub fn register(&mut self, name: Cow<'str, str>, id: Id, old_name: Option<&str>) -> Result<()> {
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

/// Rebind pointer provenance carried by a result/parameter type when its
/// temporary space was cloned into another function arena.
fn remap_rehomed_type(
    ctx: &Context<'_>,
    type_id: crate::types::TypeId,
    target: FunctionId,
    temp_space_map: &HashMap<TempSpaceId, TempSpaceId>,
) -> crate::types::TypeId {
    let Some(MemorySpaceId::Temp(old_space)) = ctx.shared.types.space_of(type_id) else {
        return type_id;
    };
    let Some(&new_space) = temp_space_map.get(&old_space) else {
        debug_assert_eq!(
            old_space.func, target,
            "rehome: result type references unmapped foreign temporary space {old_space:?}"
        );
        return type_id;
    };
    ctx.shared.types.get_or_make_space_address(
        ctx.shared.types.size_of(type_id),
        MemorySpaceId::Temp(new_space),
    )
}

/// Rebind the explicit memory space stored by load/store mnemonics. Operand
/// remapping does not see this field because it is not a `LocalValueId`.
fn remap_rehomed_memory_space(
    mnemonic: &mut Mnemonic,
    old_func: FunctionId,
    target: FunctionId,
    temp_space_map: &HashMap<TempSpaceId, TempSpaceId>,
) {
    let remap = |space: &mut LocalMemorySpaceId| {
        let LocalMemorySpaceId::Temp(old_local) = *space else {
            return;
        };
        let old = TempSpaceId::new(old_func, old_local);
        if let Some(&new) = temp_space_map.get(&old) {
            *space = LocalMemorySpaceId::Temp(new.local);
        } else {
            debug_assert_eq!(
                old_func, target,
                "rehome: mnemonic references unmapped foreign temporary space {old:?}"
            );
        }
    };
    match mnemonic {
        Mnemonic::Load(load) => remap(&mut load.space),
        Mnemonic::Store(store) => remap(&mut store.space),
        _ => {}
    }
}

/// Retarget a terminator's static block targets through `block_map` (used by
/// [`Context::rehome_owned_blocks`] to point relocated branches at the clones).
/// Value operands are handled separately via [`Mnemonic::replace_value`]; this
/// only rewrites the block targets, which are not value operands.
///
/// Targets are stored as bare body-local indices. A freshly cloned instruction
/// still holds its *source* block's local index (`old_func`-relative, strict IR
/// locality ⇒ a terminator's target shares its arena); this qualifies with
/// `old_func`, looks the full [`BlockId`] up in `block_map`, and re-localizes the
/// mapped clone against its new arena `new_func`.
fn remap_block_targets(
    mnemonic: &mut Mnemonic,
    old_func: FunctionId,
    new_func: FunctionId,
    block_map: &HashMap<BlockId, BlockId>,
) {
    let remap = |b: &mut crate::value::LocalBlockId| {
        if let Some(&new) = block_map.get(&BlockId::new(old_func, *b)) {
            *b = new.localize(new_func);
        }
    };
    match mnemonic {
        Mnemonic::Branch(branch) => remap(&mut branch.target),
        Mnemonic::CBranch(cbranch) => {
            remap(&mut cbranch.success_block);
            remap(&mut cbranch.failure_block);
        }
        _ => {}
    }
}

impl Display for Context<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.functions().try_for_each(|fun| fun.fmt(f))?;

        self.blocks()
            .filter(|block| block.parent().is_none())
            .try_for_each(|block| block.fmt(f))
    }
}

pub struct FunctionIter<'str, 'ctx> {
    ctx: &'ctx Context<'str>,
    inner: registry::Iter<'ctx, FunctionId, FunctionBody<'str>>,
}

impl<'str, 'ctx> Iterator for FunctionIter<'str, 'ctx> {
    type Item = FunctionRef<'str, 'ctx>;

    fn next(&mut self) -> Option<Self::Item> {
        let ctx = self.ctx;
        self.inner.next().map(|f| FunctionRef::from_id(ctx, f.id))
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
        BasicBlock, FunctionBody, ValueId,
        insn::{Binary, Binop, Call, Callee, IntBinop, Load, Mnemonic},
    };
    use qcode_macro::qcode;

    fn make_fn_with_blocks(ctx: &mut Context<'static>, name: &'static str, n: usize) -> FunctionId {
        // The function must exist before its blocks so they are born into its arena.
        let f = FunctionBody::make(ctx, name.into()).unwrap().id;
        for _ in 0..n {
            BasicBlock::make(ctx, f);
        }
        f
    }

    #[test]
    #[should_panic(expected = "cannot reuse a block stored in another function arena")]
    fn get_or_make_block_rejects_foreign_storage_at_address() {
        let mut ctx = Context::new();
        let a = FunctionBody::make(&mut ctx, "address_owner".into())
            .unwrap()
            .id;
        let b = FunctionBody::make(&mut ctx, "address_requester".into())
            .unwrap()
            .id;
        BasicBlock::make(&mut ctx, a).with_address(0x1000);

        ctx.get_or_make_block(0x1000, b);
    }

    #[test]
    #[should_panic(expected = "cannot create a block at an address owned by another function")]
    fn get_or_make_block_rejects_foreign_function_address_without_root() {
        let mut ctx = Context::new();
        FunctionBody::make_at_addr(&mut ctx, 0x1000, None);
        let requester = FunctionBody::make(&mut ctx, "address_requester".into())
            .unwrap()
            .id;

        ctx.get_or_make_block(0x1000, requester);
    }

    #[test]
    fn functions_iter_yields_all_functions() {
        let mut ctx = Context::new();
        let alpha = make_fn_with_blocks(&mut ctx, "alpha", 1);
        let beta = make_fn_with_blocks(&mut ctx, "beta", 1);

        let names: Vec<_> = ctx.functions().map(|f| f.name().to_string()).collect();
        assert!(names.contains(&"alpha".to_string()));
        assert!(names.contains(&"beta".to_string()));
        assert_eq!(names.len(), 2);
        assert_eq!(ctx.function_ids(), vec![alpha, beta]);
        assert_eq!(ctx.function_ids().len(), ctx.interfaces.len());
    }

    #[test]
    fn body_view_reads_match_module_reads() {
        use crate::value::{BodyView, FunctionId, FunctionRef, ModuleView, QCodeView};

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
        let fid = FunctionBody::from_name(&ctx, "foo").unwrap().id();
        let fid = ValueId::as_function(fid).unwrap();

        // A structural snapshot read entirely through a `QCodeView` — function name,
        // and per (address-then-index ordered) block: name, successor block names,
        // instruction opcodes, and param count. Both hosts route through the same
        // ref code, so equal snapshots prove the `Checked` routing.
        type Snap = (String, Vec<(String, Vec<String>, Vec<String>, usize)>);
        fn snapshot<'a, 'str: 'a>(view: impl QCodeView<'a, 'str>, fid: FunctionId) -> Snap {
            let f = FunctionRef::new(view, fid);
            let blocks = f
                .blocks()
                .map(|b| {
                    let name = b.name().unwrap_or("?").to_string();
                    let mut succ: Vec<String> = b
                        .successors()
                        .map(|(_, s)| BlockRef::new(view, s).name().unwrap_or("?").to_string())
                        .collect();
                    succ.sort();
                    let ops: Vec<String> =
                        b.instructions().map(|i| i.opcode().to_string()).collect();
                    (name, succ, ops, b.num_params())
                })
                .collect();
            (f.name().to_string(), blocks)
        }

        let module_snap = snapshot(ModuleView::new(&ctx), fid);
        assert!(!module_snap.1.is_empty(), "sanity: foo has blocks");

        // A `BodyView` over the body borrowed in place must read identically to
        // the module path — both route through the same ref code.
        let checked = BodyView::new(&ctx.bodies[fid], &ctx.shared, &ctx.interfaces);
        let checked_snap = snapshot(checked, fid);
        assert_eq!(
            module_snap, checked_snap,
            "reads through BodyView must match the module reads"
        );
    }

    #[test]
    fn body_mut_mut_matches_module_mut() {
        use crate::value::{
            BlockParam, FunctionId, FunctionRef, InstructionId, Renameable,
            block::BlockId,
            block_param::BlockParamId,
            util::{base_ref::BaseRef, body_mut::BodyMut},
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
            let insns = BasicBlock::from_id(ctx, entry).instruction_ids();
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
        ctx_a.remove_cfg_edge(entry.func, e);
        ctx_a.replace_instruction(a, ValueId::Instruction(b));
        BlockParam::from_id_mut(&mut ctx_a, param).set_size(4);
        let snap_a = snap(&ctx_a, fid);

        // ---- (b) the same mutations via a checked-out host -------------------
        let mut ctx_b = Context::new();
        let (fid_b, entry_b, bb1_b, a_b) = build(&mut ctx_b);
        let param_b = add_param(&mut ctx_b, bb1_b);
        let b_b = BasicBlock::from_id(&ctx_b, entry_b).instruction_ids()[1];

        {
            let mut host = BodyMut::new(&mut ctx_b.bodies[fid_b], &ctx_b.shared, &ctx_b.interfaces);
            let mut r = BaseRef::new(host.reborrow(), entry_b);
            r.set_comment(Some("c".into()));
            let mut r = BaseRef::new(host.reborrow(), entry_b);
            r.rename("start".into()).unwrap();
            let e = host.add_cfg_edge(entry_b, bb1_b);
            host.remove_cfg_edge(e);
            host.replace_instruction(a_b, ValueId::Instruction(b_b));
            let mut r = BaseRef::new(host.reborrow(), param_b);
            r.set_size(4);
        }
        let snap_b = snap(&ctx_b, fid_b);

        assert_eq!(
            snap_a, snap_b,
            "mutations through a pass-scoped host must match the module-path mutations"
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
    fn move_insn_before_preserves_id_and_supports_arbitrary_anchors() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <source>
                    %a = i64 0x1 + i64 0x2;
                    %free = i64 0x5 + i64 0x6;
                    goto <target>;
                <target>
                    %b = i64 0x3 + i64 0x4;
                    %consumer = %a + %b;
                    return %consumer;
            "
        );

        assert!(ctx.users(a).contains(&consumer));
        ctx.move_insn_before(a, b);

        assert!(ctx.contains_instruction(a), "moving keeps the ID live");
        assert_eq!(ctx.get_insn(a).parent().map(|block| block.id), Some(target));
        assert!(
            !BasicBlock::from_id(&ctx, source)
                .instruction_ids()
                .contains(&a)
        );
        assert_eq!(
            BasicBlock::from_id(&ctx, target).instruction_ids()[..3],
            [a, b, consumer]
        );
        assert!(
            ctx.users(a).contains(&consumer),
            "moving preserves use-map entries"
        );

        // The anchor may be any instruction, including one in the same block.
        ctx.move_insn_before(b, a);
        assert_eq!(
            BasicBlock::from_id(&ctx, target).instruction_ids()[..3],
            [b, a, consumer]
        );

        // A terminator is also a valid destination anchor.
        let return_id = *BasicBlock::from_id(&ctx, target)
            .instruction_ids()
            .last()
            .unwrap();
        ctx.move_insn_before(free, return_id);
        assert_eq!(
            BasicBlock::from_id(&ctx, target).instruction_ids()[..4],
            [b, a, consumer, free]
        );
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
        let ids = block_ref.instruction_ids();
        let load_a = ids[0];
        let original_len = ids.len();

        ctx.remove_instruction(load_a);

        let remaining = BasicBlock::from_id(&ctx, block).instruction_ids();
        assert_eq!(remaining.len(), original_len - 1);
        assert!(!remaining.contains(&load_a));
    }

    #[test]
    fn remove_instruction_drops_payload() {
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

        assert!(!ctx.contains_instruction(load_id));
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
        assert!(!ctx.contains_instruction(load_id));
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
        let ids = BasicBlock::from_id(&ctx, block).instruction_ids();
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
    fn removed_instruction_is_absent_and_not_iterated() {
        // Stable IDs survive payload compaction, while the removed payload itself must
        // disappear so stale operands never pollute a whole-program scan.
        // Regression: a removed ram load kept showing up in the alias pass's pointer
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
        let ids = BasicBlock::from_id(&ctx, block).instruction_ids();
        let dead_id = ids[1]; // %dead, unused

        assert!(
            ctx.instructions().any(|i| i.id == dead_id),
            "the instruction is iterated while live"
        );

        ctx.remove_instruction(dead_id);

        assert!(!ctx.contains_instruction(dead_id));
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
            Mnemonic::CallInd(call) => call.ptr.qualify(call_id.func),
            other => panic!("expected CallInd, got {other:?}"),
        };
        // `ptr` is a shared varnode, so query its uses across functions.
        assert_eq!(ctx.users_across_functions(ptr), vec![call_id]);

        let target = FunctionBody::make(&mut ctx, "target".into()).unwrap().id;
        ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: Callee::Real(target),
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
            }) if *actual == Callee::Real(target) && args.is_empty()
        ));
    }

    #[test]
    fn users_across_functions_keeps_ssa_users_in_the_owning_function() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <f_entry>
                    %fx = i64 1 + i64 2;
                    %fuse = %fx + i64 3;
                    return at %fuse;
            fn g:
                <g_entry>
                    %gx = i64 4 + i64 5;
                    %guse = %gx + i64 6;
                    return at %guse;
            "
        );
        let f_ids = BasicBlock::from_id(&ctx, f_entry).instruction_ids();
        let g_ids = BasicBlock::from_id(&ctx, g_entry).instruction_ids();
        assert_eq!(
            f_ids[0].local, g_ids[0].local,
            "precondition: arena-local ids collide"
        );
        assert_eq!(
            ctx.users_across_functions(ValueId::Instruction(f_ids[0])),
            vec![f_ids[1]],
            "an SSA query must not pick up the same local key from another function"
        );
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
                space: ctx.shared.default_space.into(),
                ptr: new_ptr.localize(load_id.func),
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
                lhs: new_arg.localize(load_id.func),
                rhs: new_arg.localize(load_id.func),
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
        ctx.instruction_mut(load_id).parent = None;

        // Should not panic even though parent is None.
        ctx.remove_instruction(load_id);

        assert!(ctx.get_named("a").is_none());
    }

    #[test]
    fn add_cfg_edge_returns_id_and_remove_unlinks_both_blocks() {
        let mut ctx = Context::new();
        // CFG edges are intra-function (strict IR locality): both blocks in one func.
        let f = ctx.anon_function();
        let a = BasicBlock::make(&mut ctx, f).id;
        let b = BasicBlock::make(&mut ctx, f).id;
        let c = BasicBlock::make(&mut ctx, f).id;

        let edge = ctx.add_cfg_edge(a, b);
        let surviving_edge = ctx.add_cfg_edge(b, c);
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

        ctx.remove_cfg_edge(a.func, edge);
        assert!(BasicBlock::from_id(&ctx, a).successors().next().is_none());
        assert!(BasicBlock::from_id(&ctx, b).predecessors().next().is_none());
        assert!(!ctx.bodies[a.func].edges.contains(edge));
        let surviving = ctx.edge(a.func, surviving_edge);
        assert_eq!(
            surviving.from, b.local,
            "swap removal must preserve the source"
        );
        assert_eq!(
            surviving.to, c.local,
            "swap removal must preserve the target"
        );
        assert_eq!(ctx.bodies[a.func].edges.len(), 1);

        let self_edge = ctx.add_cfg_edge(a, a);
        ctx.remove_cfg_edge(a.func, self_edge);
        assert!(!ctx.bodies[a.func].edges.contains(self_edge));
        assert!(ctx.block(a).edges.is_empty());

        let parallel_a = ctx.add_cfg_edge(a, b);
        let parallel_b = ctx.add_cfg_edge(a, b);
        ctx.remove_cfg_edge(a.func, parallel_a);
        assert!(!ctx.bodies[a.func].edges.contains(parallel_a));
        assert!(ctx.bodies[a.func].edges.contains(parallel_b));
        assert_eq!(
            BasicBlock::from_id(&ctx, a)
                .successors()
                .collect::<Vec<_>>(),
            vec![(parallel_b, b)],
        );
    }

    #[test]
    fn truth_map_tracks_four_states_and_conflicts() {
        let mut ctx = Context::new();
        let callee = FunctionBody::make(&mut ctx, "callee".into()).unwrap().id;
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
        let callee = FunctionBody::make(&mut ctx, "exit".into()).unwrap().id;
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
        ctx.shared
            .memory_image
            .add_segment(0x1000, vec![0u8; 4], true, false); // code
        ctx.shared
            .memory_image
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
        ctx.shared
            .memory_image
            .add_segment(0x1000, vec![0u8; 4], true, false); // code
        ctx.shared
            .memory_image
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
        let some_space = ctx.get_or_make_named_space("scratch");
        let sa = ctx.shared.types.get_or_make_space_address(8, some_space);
        let sa_size = ctx.shared.types.size_of(sa);

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
        for function_id in restored.function_ids() {
            assert_eq!(restored.bodies[function_id].id(), function_id);
        }
        // The SpaceAddress type round-trips: same id, same size, same space.
        assert_eq!(restored.shared.types.size_of(sa), sa_size);
        assert_eq!(
            restored.shared.types.space_of(sa),
            Some(crate::space::MemorySpaceId::Shared(some_space))
        );
    }

    #[test]
    fn compact_edge_arena_preserves_ids_across_round_trip() {
        let mut ctx = Context::new();
        let function = ctx.anon_function();
        let a = BasicBlock::make(&mut ctx, function).id;
        let b = BasicBlock::make(&mut ctx, function).id;
        let c = BasicBlock::make(&mut ctx, function).id;
        let d = BasicBlock::make(&mut ctx, function).id;
        let first = ctx.add_cfg_edge(a, b);
        let removed = ctx.add_cfg_edge(b, c);
        let last = ctx.add_cfg_edge(c, d);
        ctx.remove_cfg_edge(function, removed);

        let physical_order: Vec<_> = ctx.bodies[function]
            .edges
            .iter()
            .map(|edge| edge.id)
            .collect();
        assert_eq!(physical_order, vec![first, last]);

        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&ctx, config).expect("encode");
        let (mut restored, _): (Context<'static>, usize) =
            bincode::serde::decode_from_slice(&bytes, config).expect("decode");

        assert!(!restored.bodies[function].edges.contains(removed));
        assert_eq!(
            restored.bodies[function]
                .edges
                .iter()
                .map(|edge| edge.id)
                .collect::<Vec<_>>(),
            physical_order,
        );
        assert_eq!(restored.edge(function, first).to, b.local);
        assert_eq!(restored.edge(function, last).from, c.local);

        let fresh = restored.add_cfg_edge(a, d);
        assert!(fresh > last);
        assert_ne!(fresh, removed, "removed edge IDs must never be reused");
    }

    #[test]
    fn compact_instruction_arena_preserves_ids_across_round_trip() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                %first = i64 1 + i64 2;
                %removed = i64 3 + i64 4;
                return at %first;
            "
        );
        let ids = BasicBlock::from_id(&ctx, block).instruction_ids();
        let first = ids[0];
        let removed = ids[1];
        let last = ids[2];
        ctx.remove_instruction(removed);

        let physical_order: Vec<_> = ctx.bodies[first.func]
            .insns
            .iter()
            .map(|insn| insn.id)
            .collect();
        assert_eq!(physical_order, vec![first.local, last.local]);

        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&ctx, config).expect("encode");
        let (mut restored, _): (Context<'static>, usize) =
            bincode::serde::decode_from_slice(&bytes, config).expect("decode");

        assert!(!restored.contains_instruction(removed));
        assert_eq!(
            restored.bodies[first.func]
                .insns
                .iter()
                .map(|insn| insn.id)
                .collect::<Vec<_>>(),
            physical_order,
        );
        assert!(restored.contains_instruction(first));
        assert!(restored.contains_instruction(last));

        let template = restored.instruction(last).clone();
        let fresh = restored.push_insn(first.func, template);
        assert!(fresh.local > last.local);
        assert_ne!(
            fresh, removed,
            "removed instruction IDs must never be reused"
        );
    }

    #[test]
    fn compact_param_arena_preserves_ids_across_round_trip() {
        let mut ctx = Context::new();
        let function = ctx.anon_function();
        let block = BasicBlock::make(&mut ctx, function).id;
        let first = BasicBlock::from_id_mut(&mut ctx, block).push_param(8).id;
        let removed = BasicBlock::from_id_mut(&mut ctx, block).push_param(8).id;
        let last = BasicBlock::from_id_mut(&mut ctx, block).push_param(8).id;

        ctx.block_mut(block).params.remove(1);
        ctx.block_param_mut(last).index = 1;
        ctx.remove_block_param(removed);

        let physical_order: Vec<_> = ctx.bodies[function]
            .params
            .iter()
            .map(|param| param.id)
            .collect();
        assert_eq!(physical_order, vec![first.local, last.local]);
        assert_eq!(ctx.block_param(first).index, 0);
        assert_eq!(ctx.block_param(last).index, 1);

        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&ctx, config).expect("encode");
        let (mut restored, _): (Context<'static>, usize) =
            bincode::serde::decode_from_slice(&bytes, config).expect("decode");

        assert!(!restored.contains_block_param(removed));
        assert_eq!(
            restored.bodies[function]
                .params
                .iter()
                .map(|param| param.id)
                .collect::<Vec<_>>(),
            physical_order,
        );
        assert!(restored.contains_block_param(first));
        assert!(restored.contains_block_param(last));

        let fresh = BasicBlock::from_id_mut(&mut restored, block)
            .push_param(8)
            .id;
        assert!(fresh.local > last.local);
        assert_ne!(fresh, removed, "removed parameter IDs must never be reused");
    }

    #[test]
    fn compact_block_arena_preserves_ids_across_round_trip() {
        let mut ctx = Context::new();
        let function = ctx.anon_function();
        let first = BasicBlock::make(&mut ctx, function).id;
        let removed = BasicBlock::make(&mut ctx, function).id;
        let last = BasicBlock::make(&mut ctx, function).id;
        FunctionBody::from_id_mut(&mut ctx, function)
            .set_root(first)
            .expect("set root");

        ctx.delete_block(removed);

        let physical_order: Vec<_> = ctx.bodies[function]
            .blocks
            .iter()
            .map(|block| block.id)
            .collect();
        assert_eq!(physical_order, vec![first.local, last.local]);
        assert_eq!(ctx.block_ids(), vec![first, last]);

        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&ctx, config).expect("encode");
        let (mut restored, _): (Context<'static>, usize) =
            bincode::serde::decode_from_slice(&bytes, config).expect("decode");

        assert!(!restored.contains_block(removed));
        assert_eq!(
            restored.bodies[function]
                .blocks
                .iter()
                .map(|block| block.id)
                .collect::<Vec<_>>(),
            physical_order,
        );
        assert!(restored.contains_block(first));
        assert!(restored.contains_block(last));
        assert_eq!(
            FunctionBody::from_id(&restored, function)
                .root()
                .map(|block| block.id),
            Some(first),
        );

        let fresh = BasicBlock::make(&mut restored, function).id;
        assert!(fresh.local > last.local);
        assert_ne!(fresh, removed, "removed block IDs must never be reused");
    }

    #[test]
    fn deleting_root_clears_function_root() {
        let mut ctx = Context::new();
        let function = ctx.anon_function();
        let root = BasicBlock::make(&mut ctx, function).id;
        FunctionBody::from_id_mut(&mut ctx, function)
            .set_root(root)
            .expect("set root");

        ctx.delete_block(root);

        assert!(!ctx.contains_block(root));
        assert!(FunctionBody::from_id(&ctx, function).root().is_none());
        assert!(ctx.block_ids().is_empty());
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

    // --- split_function_at (strict-local construction verb, ruling 2) ---

    mod split_function_at {
        use super::*;

        use crate::value::insn::{Callee, Mnemonic, TailCall};
        use crate::value::{BasicBlock, FunctionBody, Instruction, Value};
        use std::borrow::Cow;

        fn block_at(ctx: &mut Context<'static>, func: FunctionId, addr: u64) -> BlockId {
            BasicBlock::make(ctx, func).with_address(addr).id
        }

        fn branch_at(ctx: &mut Context<'static>, block: BlockId, target: BlockId, addr: u64) {
            let id = (ctx).builder(block).push_branch(target).id;
            Instruction::from_id_mut(ctx, id).set_address(addr);
        }

        fn cbranch_at(
            ctx: &mut Context<'static>,
            block: BlockId,
            success: BlockId,
            failure: BlockId,
            addr: u64,
        ) {
            let cond = ctx.get_const(1, 1).id();
            let id = (ctx).builder(block).push_cbranch(cond, success, failure).id;
            Instruction::from_id_mut(ctx, id).set_address(addr);
        }

        fn return_at(ctx: &mut Context<'static>, block: BlockId, addr: u64) {
            let zero = ctx.get_const(0, 8).id();
            let id = (ctx).builder(block).push_return(zero).id;
            Instruction::from_id_mut(ctx, id).set_address(addr);
        }

        fn block_at_addr(ctx: &Context, func: FunctionId, addr: u64) -> BlockId {
            FunctionBody::from_id(ctx, func)
                .block_ids()
                .into_iter()
                .find(|b| ctx.block(*b).address == Some(addr))
                .unwrap_or_else(|| panic!("{func:?} has no block at {addr:#x}"))
        }

        fn addrs(ctx: &Context, func: FunctionId) -> Vec<u64> {
            let mut got: Vec<u64> = FunctionBody::from_id(ctx, func)
                .block_ids()
                .into_iter()
                .filter_map(|b| ctx.block(b).address)
                .collect();
            got.sort_unstable();
            got
        }

        /// F@0x1000 (`jmp 0x2000`) absorbed the body later found to be its own
        /// function at 0x2000 (`0x2000: jmp 0x2005 ; 0x2005: ret`). Splitting at the
        /// 0x2000 block reuses the stub `G`, moves 0x2000+0x2005 into `G` self-stored,
        /// leaves only the thunk in `F`, and turns the thunk's branch into a `TailCall`.
        #[test]
        fn splits_absorbed_body_reusing_the_stub() {
            let mut ctx = Context::new();
            let f = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("thunk"))).id;
            let b0 = block_at(&mut ctx, f, 0x1000);
            let b1 = block_at(&mut ctx, f, 0x2000);
            let b2 = block_at(&mut ctx, f, 0x2005);
            branch_at(&mut ctx, b0, b1, 0x1000);
            branch_at(&mut ctx, b1, b2, 0x2000);
            return_at(&mut ctx, b2, 0x2005);
            {
                let mut func = FunctionBody::from_id_mut(&mut ctx, f);
                func.set_root(b0).unwrap();
            }
            // A later `call 0x2000` minted the stub.
            let g = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some(Cow::Borrowed("real"))).id;

            let split_g = ctx.split_function_at(b1);
            assert_eq!(
                split_g, g,
                "the split must reuse the existing stub at 0x2000"
            );

            assert_eq!(addrs(&ctx, f), vec![0x1000]);
            assert_eq!(addrs(&ctx, g), vec![0x2000, 0x2005]);
            let g_entry = block_at_addr(&ctx, g, 0x2000);
            assert_eq!(ctx.bodies[g].root_id(), Some(g_entry.local));

            // Every G block is self-stored.
            for b in FunctionBody::from_id(&ctx, g).block_ids() {
                assert_eq!(b.func, g);
            }

            // The thunk's branch into the tail became a TailCall(G); its edge is gone.
            let f_entry = block_at_addr(&ctx, f, 0x1000);
            assert_eq!(BasicBlock::from_id(&ctx, f_entry).successors().count(), 0);
            let term = BasicBlock::from_id(&ctx, f_entry)
                .instructions()
                .last()
                .map(|i| i.mnemonic().clone());
            assert!(
                matches!(term, Some(Mnemonic::TailCall(TailCall { target, .. })) if target == Callee::Real(g)),
                "thunk branch must become TailCall(G), got {term:?}",
            );
        }

        #[test]
        fn split_rehomes_temporary_values_spaces_and_pointer_types() {
            let mut ctx = Context::new();
            let f = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("f"))).id;
            let entry = block_at(&mut ctx, f, 0x1000);
            let tail = block_at(&mut ctx, f, 0x2000);
            branch_at(&mut ctx, entry, tail, 0x1000);
            FunctionBody::from_id_mut(&mut ctx, f)
                .set_root(entry)
                .unwrap();

            let temp = ctx
                .builder(tail)
                .make_named_temp(Cow::Borrowed("scratch"), 8);
            ctx.builder(entry)
                .make_named_temp(Cow::Borrowed("unused"), 4);
            let temp_space = ctx.bodies[f].temps[temp.local].space;
            let load = {
                let mut builder = ctx.builder(tail);
                let ValueId::Instruction(load) = builder
                    .push_load::<false>(
                        ValueId::Temp(temp),
                        8,
                        LocalMemorySpaceId::Temp(temp_space),
                    )
                    .id()
                else {
                    unreachable!()
                };
                builder.push_return(ValueId::Instruction(load));
                load
            };
            let pointer_type = ctx
                .shared
                .types
                .get_or_make_space_address(8, MemorySpaceId::Temp(TempSpaceId::new(f, temp_space)));
            ctx.instruction_mut(load).type_id = pointer_type;

            let g =
                FunctionBody::make_at_addr(&mut ctx, 0x2000, Some(Cow::Borrowed("discovered"))).id;
            assert_eq!(ctx.split_function_at(tail), g);

            let diagnostics = crate::verify_body_arena_integrity(&ctx);
            assert!(diagnostics.is_empty(), "{diagnostics:#?}");
            assert_eq!(ctx.bodies[g].temp_spaces.len(), 1);
            assert_eq!(ctx.bodies[g].temps.len(), 1);
            assert_eq!(ctx.bodies[f].temps.len(), 2, "source arenas remain intact");

            let moved_load = FunctionBody::from_id(&ctx, g)
                .blocks()
                .flat_map(|block| block.instructions())
                .find(|insn| matches!(insn.mnemonic(), Mnemonic::Load(_)))
                .expect("load moved with the split");
            let Mnemonic::Load(moved) = moved_load.mnemonic() else {
                unreachable!()
            };
            let LocalMemorySpaceId::Temp(moved_space) = moved.space else {
                panic!("load lost temporary-space provenance")
            };
            assert!(matches!(moved.ptr, crate::value::LocalValueId::Temp(_)));
            assert!(usize::from(moved_space) < ctx.bodies[g].temp_spaces.len());
            assert_eq!(
                ctx.shared.types.space_of(moved_load.type_id()),
                Some(MemorySpaceId::Temp(TempSpaceId::new(g, moved_space)))
            );

            // This is the path that previously panicked in `function_fingerprint`.
            let rendered = FunctionBody::from_id(&ctx, g).to_string();
            assert!(rendered.contains("scratch"));
        }

        #[test]
        fn split_stops_at_a_foreign_rootless_stub_address() {
            let mut ctx = Context::new();
            let f = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("f"))).id;
            let entry = block_at(&mut ctx, f, 0x1000);
            let split = block_at(&mut ctx, f, 0x2000);
            let foreign_entry = block_at(&mut ctx, f, 0x3000);
            let foreign_body = block_at(&mut ctx, f, 0x3005);
            branch_at(&mut ctx, entry, split, 0x1000);
            branch_at(&mut ctx, split, foreign_entry, 0x2000);
            branch_at(&mut ctx, foreign_entry, foreign_body, 0x3000);
            return_at(&mut ctx, foreign_body, 0x3005);
            FunctionBody::from_id_mut(&mut ctx, f)
                .set_root(entry)
                .unwrap();

            let g = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some(Cow::Borrowed("g"))).id;
            let h = FunctionBody::make_at_addr(&mut ctx, 0x3000, Some(Cow::Borrowed("h"))).id;
            assert!(FunctionBody::from_id(&ctx, g).root().is_none());
            assert!(FunctionBody::from_id(&ctx, h).root().is_none());

            assert_eq!(ctx.split_function_at(split), g);
            assert_eq!(addrs(&ctx, g), vec![0x2000]);
            assert_eq!(addrs(&ctx, f), vec![0x1000, 0x3000, 0x3005]);
            assert!(FunctionBody::from_id(&ctx, h).root().is_none());

            let g_entry = block_at_addr(&ctx, g, 0x2000);
            let term = BasicBlock::from_id(&ctx, g_entry)
                .instructions()
                .last()
                .map(|i| i.mnemonic().clone());
            assert!(
                matches!(term, Some(Mnemonic::TailCall(TailCall { target, .. })) if target == Callee::Real(h)),
                "split tail must stop and tail-call rootless stub H, got {term:?}",
            );
        }

        #[test]
        fn split_rehomes_block_param_origin_into_destination_arena() {
            let mut ctx = Context::new();
            let f = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("f"))).id;
            let entry = block_at(&mut ctx, f, 0x1000);
            let tail = block_at(&mut ctx, f, 0x2000);
            let param = BasicBlock::from_id_mut(&mut ctx, tail).push_param(8).id;
            crate::value::BlockParam::from_id_mut(&mut ctx, param)
                .set_origin(ValueId::BlockParam(param));

            let arg = ctx.get_const(7, 8).id();
            let branch = ctx.builder(entry).push_branch_with_args(tail, vec![arg]).id;
            Instruction::from_id_mut(&mut ctx, branch).set_address(0x1000);
            let ret = ctx.builder(tail).push_return(ValueId::BlockParam(param)).id;
            Instruction::from_id_mut(&mut ctx, ret).set_address(0x2000);
            FunctionBody::from_id_mut(&mut ctx, f)
                .set_root(entry)
                .unwrap();

            let g = ctx.split_function_at(tail);
            let new_tail = block_at_addr(&ctx, g, 0x2000);
            let new_param = BasicBlock::from_id(&ctx, new_tail).params().next().unwrap();
            assert_eq!(new_param.origin(), Some(ValueId::BlockParam(new_param.id)));
        }

        /// A conditional arm into the split block is routed through a fresh
        /// intra-function trampoline ending in a `TailCall`; the fall-through arm is
        /// untouched and no foreign block reference survives.
        #[test]
        fn conditional_arm_into_split_block_uses_a_trampoline() {
            let mut ctx = Context::new();
            let f = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("f"))).id;
            let entry = block_at(&mut ctx, f, 0x1000);
            let cont = block_at(&mut ctx, f, 0x1008);
            let tail = block_at(&mut ctx, f, 0x2000);
            cbranch_at(&mut ctx, entry, tail, cont, 0x1000);
            return_at(&mut ctx, cont, 0x1008);
            return_at(&mut ctx, tail, 0x2000);
            FunctionBody::from_id_mut(&mut ctx, f)
                .set_root(entry)
                .unwrap();

            let g = ctx.split_function_at(tail);

            let entry = block_at_addr(&ctx, f, 0x1000);
            let cont = block_at_addr(&ctx, f, 0x1008);
            assert_eq!(addrs(&ctx, g), vec![0x2000]);

            let Mnemonic::CBranch(cb) = BasicBlock::from_id(&ctx, entry)
                .instructions()
                .last()
                .unwrap()
                .mnemonic()
                .clone()
            else {
                panic!("entry must still end in a cbranch");
            };
            assert_eq!(cb.failure_block, cont.local, "fall-through arm untouched");
            let tramp = BlockId::new(entry.func, cb.success_block);
            assert_eq!(
                BasicBlock::from_id(&ctx, tramp).parent().map(|f| f.id),
                Some(f),
                "trampoline lives in F",
            );
            let term = BasicBlock::from_id(&ctx, tramp)
                .instructions()
                .last()
                .map(|i| i.mnemonic().clone());
            assert!(
                matches!(term, Some(Mnemonic::TailCall(TailCall { target, .. })) if target == Callee::Real(g)),
                "trampoline must tail-call G, got {term:?}",
            );
            // Every successor of entry is intra-F.
            for (_, s) in BasicBlock::from_id(&ctx, entry).successors() {
                assert_eq!(BasicBlock::from_id(&ctx, s).parent().map(|f| f.id), Some(f));
            }
        }

        /// With no pre-existing stub at the landing address, the split mints a
        /// conventional `fn_<addr>` and moves the tail into it self-stored.
        #[test]
        fn mints_a_conventional_function_when_no_stub_exists() {
            let mut ctx = Context::new();
            let f = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("f"))).id;
            let entry = block_at(&mut ctx, f, 0x1000);
            let mid = block_at(&mut ctx, f, 0x1008);
            branch_at(&mut ctx, entry, mid, 0x1000);
            return_at(&mut ctx, mid, 0x1008);
            FunctionBody::from_id_mut(&mut ctx, f)
                .set_root(entry)
                .unwrap();

            let g = ctx.split_function_at(mid);
            assert_eq!(FunctionBody::from_id(&ctx, g).name(), "fn_1008");
            assert_eq!(addrs(&ctx, f), vec![0x1000]);
            assert_eq!(addrs(&ctx, g), vec![0x1008]);
            let addresses = crate::address_index::AddressIndex::analyze(&ctx);
            assert_eq!(addresses.function_at(0x1008), Some(g));
            for b in FunctionBody::from_id(&ctx, g).block_ids() {
                assert_eq!(b.func, g);
            }
        }
    }
}
