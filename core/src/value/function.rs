use jstd::{
    Identifier,
    registry::{Identified, Registry},
    stable_arena::StableArena,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    borrow::Cow,
    collections::BTreeSet,
    fmt::{Display, Formatter},
    marker::PhantomData,
};

mod footprint;
#[cfg(test)]
mod insn_order;
#[cfg(test)]
mod use_edges;
pub use footprint::{Footprint, RamBase, RamField, RamLocations, RamObject, RamRegion};

mod signature;
pub use signature::{
    ArgMemKind, ExternArg, ExternArgmem, ExternInterface, ExternSlot, FunctionSignature, ParamAttrs,
};

use crate::{
    context::Context,
    error::{Error, ErrorTy, Result},
    value::{
        BasicBlock, BlockId, BlockRef, Instruction, InstructionId, LocalValueId, ModuleView,
        QCodeView, Temp, TempId, TempSpace, TempSpaceId, Value, ValueId, VarnodeId,
        block::cfg::{EdgeId, LocalBlockId},
        block::{EdgeData, InsnList},
        block_param::{BlockParam, BlockParamId, LocalParamId},
        insn::{LocalInsnId, Mnemonic},
        uses::{Use, UseArena, UseId, WithUsers},
        util::{
            base_ref::{BaseRef, WithCtx, WithCtxMut},
            named::{Named, Renameable, update_context_name},
        },
    },
};

#[derive(Identifier)]
pub struct FunctionId(u32);

/// Everything a *caller* reasons about a function: its name, address, semantic
/// kind, external-ness, and ABI/analysis signature. This is the caller-reasoning
/// surface (ruling 1 of the context-split design): it is precisely the data a
/// function pass may read about *another* function. Its counterpart is the
/// function *body* (arenas, roster, users, local names) — everything only the
/// function's own passes touch.
///
/// Interfaces are stored in their own
/// [`Context::interfaces`](crate::context::Context::interfaces)
/// registry, held in lockstep with the function bodies under the same
/// [`FunctionId`] and never checked out — so a caller always reads the real
/// interface even while a callee's body is checked out to a worker.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct FunctionInterface<'str> {
    /// The function's name.
    pub name: Cow<'str, str>,

    /// Optional entry address (from binary).
    ///
    /// Crate-private: an entry address is what an
    /// [`AddressIndex`](crate::address_index::AddressIndex) lists and the
    /// module's [shape revision](crate::context::Context::revision) counts, so
    /// only the mutators that tick that clock write it. Read it through
    /// [`address`](Self::address).
    pub(crate) address: Option<u64>,

    /// Whether this is an external (imported) function.
    ///
    /// External functions have no lifted body — they are stubs for calls that
    /// go outside the binary (e.g. PLT thunks for shared-library functions).
    /// The recursive disassembler will not attempt to lift their body.
    pub is_external: bool,

    /// Optional ABI description used by alias analysis.
    pub signature: Option<FunctionSignature>,

    /// What semantic class this function belongs to.
    #[serde(default)]
    pub kind: FunctionKind,

    /// Call-graph-closed implicit *effect summary* of this function, per the
    /// argpromote v2 design (`ARGPROMOTE_REGISTERS_V2.md`). Solved by the
    /// effect-analysis pass and read by the materialize/regpure passes, the
    /// emulator, alias analysis, and the verifier.
    ///
    /// Serialized into the `.harbinger` wire shape so that rewritten regpure
    /// call sites and materialized interfaces stay in sync with the snapshot.
    /// Older snapshots that predate this field load as
    /// [`RegisterChannelState::Unsolved`] via `#[serde(default)]`.
    #[serde(default)]
    pub effects: FunctionEffects,

    /// For a PE import brought in by ordinal only, the ordinal it was imported
    /// at. Recorded before the stub is renamed to its real export name.
    #[serde(default)]
    pub import_ordinal: Option<u16>,
}

/// A function's full effect summary, one component per side-effect channel:
/// the register-lifecycle state and the memory write-space verdict. Serialized
/// as part of [`FunctionInterface`].
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FunctionEffects {
    /// Register-channel lifecycle summary (argpromote v2).
    #[serde(default)]
    pub register: RegisterChannelState,
    /// Memory-channel effect summary: the coarse written-space verdict plus the
    /// precise RAM footprint.
    #[serde(default)]
    pub memory: MemoryChannelState,
}

impl FunctionEffects {
    /// The materialized register interface mapping, if the register channel has
    /// been materialized. Delegates to [`RegisterChannelState::materialized`].
    pub fn materialized(&self) -> Option<&RegisterInterfaceMap> {
        self.register.materialized()
    }

    /// Whether the register channel is solved. Delegates to
    /// [`RegisterChannelState::is_solved`].
    pub fn is_solved(&self) -> bool {
        self.register.is_solved()
    }
}

/// The state of a function's register-channel effect summary (argpromote v2).
///
/// Purity has moved from a function flag (`pure_reg`) to per-call-site tags, but
/// the *interface mapping* a materialized function exposes still lives on the
/// function — the emulator's implicit call convention, alias analysis, and the
/// verifier all consume it. This enum records how far the summary has advanced.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RegisterChannelState {
    /// Not yet solved by the effect-analysis pass (the default / post-load
    /// state).
    #[default]
    Unsolved,
    /// ⊤ — unknowable: the function contains an unresolved indirect call, calls
    /// a ⊤ function, or is a prototype-less external. Its register effects stay
    /// modelled conservatively (clobbers-all) at every call site.
    Top,
    /// Solved to a finite effect set but the interface is not yet materialized
    /// (no by-value params / return pack added). Call sites still bind
    /// implicitly, but the solved read/write register sets are precise: a call
    /// to this function reads at most `reads` and writes at most `writes`.
    Solved(RegisterEffectSets),
    /// Materialized: the function carries by-value register params and a return
    /// pack, and this mapping records which register each interface slot binds.
    /// Consumed by the dual binding convention.
    Materialized(RegisterInterfaceMap),
}

impl RegisterChannelState {
    /// The materialized interface mapping, if this function has been
    /// materialized.
    pub fn materialized(&self) -> Option<&RegisterInterfaceMap> {
        match self {
            RegisterChannelState::Materialized(map) => Some(map),
            _ => None,
        }
    }

    /// Whether the summary is solved (either not-yet- or already-materialized),
    /// i.e. its register effects are known precisely rather than ⊤.
    pub fn is_solved(&self) -> bool {
        matches!(
            self,
            RegisterChannelState::Solved(_) | RegisterChannelState::Materialized(_)
        )
    }
}

/// The memory-channel component of a function's effects: the coarse
/// written-space tri-state plus the precise RAM [`Footprint`] the same solve
/// derived.
///
/// The two components are independently ⊤: `coarse` is deliberately laxer, so a
/// function whose footprint defies classification (`precise == None`) usually
/// still has a bounded space set.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryChannelState {
    /// The coarse set of non-register spaces this function may (transitively)
    /// write. Subsumes the retired `written_spaces` + `written_spaces_stamped`
    /// signature pair.
    #[serde(default)]
    pub coarse: WrittenSpacesState,
    /// The exhaustive outward memory footprint this function may touch, or
    /// `None` for ⊤ — inexpressible in the lattice (an unclassifiable access, a
    /// non-lockstep interface, budget saturation, or an unrebasable call edge).
    ///
    /// Persisted so that a memory-channel effect delta can compare *addresses*,
    /// not merely space granularity: two solves that both write `{ram}` at
    /// different addresses must not compare `Equal`, since `Equal` is the one
    /// verdict that licenses stopping invalidation propagation.
    ///
    /// `#[serde(default)]` (→ `None`, i.e. ⊤) so snapshots written before the
    /// footprint was persisted still load, conservatively.
    #[serde(default)]
    pub precise: Option<Footprint>,

    /// The materialized memory interface: where each by-value memory input is
    /// bound from and each write-set output replayed to, once the RAM channel
    /// has functionalized this function. `None` while the memory channel is not
    /// materialized (the default and, today, the only state any pass sets).
    ///
    /// The memory analogue of
    /// [`RegisterChannelState::Materialized`].
    /// Unlike the register channel this is a field rather than a lattice state,
    /// because `coarse` and `precise` are independently ⊤ and materialization is
    /// orthogonal to both.
    ///
    /// `#[serde(default)]` (→ `None`) so snapshots predating the memory
    /// interface load unchanged.
    #[serde(default)]
    pub materialized: Option<MemoryInterfaceMap>,
}

impl MemoryChannelState {
    /// The materialized memory interface, if the memory channel has been
    /// materialized.
    pub fn materialized(&self) -> Option<&MemoryInterfaceMap> {
        self.materialized.as_ref()
    }
}

/// Owned tri-state of a function's coarse written-space verdict, subsuming the
/// old `written_spaces: Option<Vec<SpaceId>>` + `written_spaces_stamped: bool`
/// pair. The borrowing view [`WrittenSpaces`] is derived from this.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WrittenSpacesState {
    /// Analysis has never recorded a verdict — a freshly minted function.
    /// (Was `written_spaces_stamped == false`.)
    #[default]
    Unstamped,
    /// Recorded, but unbounded (⊤): the function may write any space.
    /// (Was stamped with `written_spaces == None`.)
    Unbounded,
    /// A recorded exact witnessed bound: a space not listed is never written.
    /// (Was stamped with `written_spaces == Some(sorted)`.)
    Bounded(Vec<crate::space::SpaceId>),
}

/// The solved (transitive) register effect of a function whose interface is
/// *not* materialized: the registers a call to it may read / write, callee
/// effects included. Sorted, deduplicated varnode lists — the persistable form
/// of the register channel's solved lattice value.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RegisterEffectSets {
    /// Registers a call may read (sorted).
    #[serde(alias = "loads")]
    pub reads: Vec<VarnodeId>,
    /// Registers a call may write (sorted).
    #[serde(alias = "stores")]
    pub writes: Vec<VarnodeId>,
}

/// A register whose value on return is *computed* rather than carried in the
/// return pack: the linked function is this function's **projection** for that
/// one output — a pure function of its inputs, returning what the register
/// would have held.
///
/// Recording the projection is what lets the register leave both
/// [`outputs`](RegisterInterfaceMap::outputs) and, once nothing else reads it,
/// [`inputs`](RegisterInterfaceMap::inputs) without the fact being lost. The
/// motivating case is the stack pointer: every function "returns" `SP + k`,
/// which is a true statement about the machine code and no part of what the
/// function means. Leaving it in the pack put `RSP` in every signature; deleting
/// it outright would discard a real effect. A projection does neither.
///
/// The projection is an ordinary function and says what it reads through its
/// *own* interface — this record deliberately does not restate the binding, so
/// there is nothing here to desync from the function it names.
///
/// Entries are disjoint from `outputs`: a register is either packed or derived,
/// never both.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DerivedOutput {
    /// The register this projection computes.
    pub register: VarnodeId,
    /// The projection: a pure function whose return value is `register`'s value
    /// on return from the parent.
    pub projection: crate::value::insn::Callee,
}

/// The ordered, machine-readable register interface of a *materialized*
/// function: which register each by-value input parameter binds, and which
/// register each return-pack slot stores back. Slot `i` of `inputs` is the
/// `i`-th register param; slot `i` of `outputs` is the `i`-th pack field.
///
/// Both the register channel's own rewrite (regpure calls) and the emulator's
/// implicit (zero-arg) convention read this: implicitly, param `i` is seeded
/// from `inputs[i]` at entry and pack slot `i` is stored back to `outputs[i]`
/// on return.
///
/// A third category sits alongside those two: a register that is neither an
/// input nor packed, because it is *derived* — see
/// [`projections`](Self::projections).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RegisterInterfaceMap {
    /// Register bound by each by-value input parameter, in parameter order.
    pub inputs: Vec<VarnodeId>,
    /// Register written back by each return-pack slot, in pack order. Ordered
    /// returns-first: slots `..returns` carry real computed values, the rest
    /// are clobbers (undefined — poison — at a rewritten call site).
    pub outputs: Vec<VarnodeId>,
    /// How many leading `outputs` slots are return values (the rest are
    /// clobbers). A bodied function replays every slot as a real store, so its
    /// `returns == outputs.len()`; a prototyped external returns only its ABI
    /// return register(s) and clobbers the caller-saved tail.
    pub returns: usize,
    /// Registers whose returned value is computed by a linked projection rather
    /// than carried in the pack (see [`DerivedOutput`]). Unordered and disjoint
    /// from `outputs`; a consumer that needs such a register's value evaluates
    /// its projection instead of reading a pack field.
    ///
    /// `#[serde(default)]` (→ empty) covers self-describing formats only. A
    /// `.harbinger` session payload is *positional* bincode under a hard version
    /// lock with no migration path, so adding this field changed the payload
    /// layout and required a `harbinger_session::session::FORMAT_VERSION` bump —
    /// old sessions are rejected, not defaulted.
    #[serde(default)]
    pub projections: Vec<DerivedOutput>,
}

/// Where one materialized interface input is bound from, or one write-set
/// output replayed to, at a call site that does not pass it explicitly.
///
/// This is the memory channel's analogue of [`RegisterInterfaceMap`]'s bare
/// [`VarnodeId`]: a register input needs no descriptor beyond the register
/// itself, but a memory input has to say *which address* the caller reads. It
/// generalizes [`ExternSlot`], which
/// describes the same thing for prototyped externals only.
///
/// Every slot is `mem[base + offset]` of `size` bytes: a memory input is, by
/// construction, a dereference. The interface says *where*, in terms the caller
/// can evaluate — never in terms of the callee's body.
///
/// Deliberately **non-recursive**: a base that must itself be loaded is the
/// "re-dereference whose address is loaded at runtime" case the RAM channel
/// already rejects as unmodellable — such a base records
/// [`SlotBase::Unmappable`] rather than being described.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InterfaceSlot {
    pub base: SlotBase,
    pub offset: i64,
    pub size: usize,
}

/// What an [`InterfaceSlot`]'s address is relative to.
///
/// Deliberately **register-free**. A materialized function is on its way to
/// being a pure function of its arguments, and its interface should carry no
/// notion of a register file: the register channel has already turned every
/// register input into a by-value argument, so a base that *was* a register is
/// simply the argument bound to it. The property that matters is that a caller
/// can express the address — hence the vocabulary is "an argument you pass" or
/// "an address that is the same everywhere".
///
/// The address-less cases are kept apart on purpose: an absolute address and a
/// base we failed to describe are both address-less, but only the first is
/// bindable. Collapsing them would let a consumer load from a bogus absolute
/// address for a base it never resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SlotBase {
    /// The callee's `i`-th positional argument: `mem[args[i] + offset]`.
    ///
    /// Bindable at any site that passes that argument — which is *site*-relative,
    /// and honestly so: a pointer the caller supplies cannot be reconstructed
    /// without a caller. A caller-less function is still materialized; it simply
    /// binds nowhere.
    Arg(usize),
    /// An absolute address (a global): `mem[addr + offset]`. Bindable anywhere,
    /// including at an implicit or indirect site, since the address is the same
    /// in every caller.
    Global(u64),
    /// The base could not be expressed. **Not bindable**: a consumer must refuse
    /// such a slot rather than treat it as absolute.
    Unmappable,
}

impl InterfaceSlot {
    /// Whether this slot's address is expressible at a call site at all.
    ///
    /// A [`SlotBase::Arg`] slot additionally needs the site to actually pass
    /// that argument; this reports only the address-independent half.
    pub fn is_bindable(&self) -> bool {
        !matches!(self.base, SlotBase::Unmappable)
    }
}

/// The ordered, machine-readable *memory* interface of a function whose memory
/// channel has been materialized: where each by-value memory input parameter is
/// loaded from, and where each memory write-set output is replayed to.
///
/// The memory analogue of [`RegisterInterfaceMap`]. Memory input parameters
/// follow the register inputs in root-parameter order, so `inputs[i]` describes
/// root param `register_inputs.len() + i`.
///
/// Read by the emulator's implicit binding convention (evaluate each input
/// slot's address in the *caller's* state at the call, replay each output slot
/// on return) and by the decompiler.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryInterfaceMap {
    /// Where each by-value memory input parameter is bound from, in parameter
    /// order (after the register inputs).
    pub inputs: Vec<InterfaceSlot>,
    /// Where each memory write-set output slot is replayed to, in pack order
    /// (after the register outputs).
    pub outputs: Vec<InterfaceSlot>,
}

/// A function *body*: arenas, roster, root, use edges, local names. The
/// caller-reasoning surface lives separately in [`FunctionInterface`], stored in
/// [`Context::interfaces`](crate::context::Context::interfaces)
/// under the same [`FunctionId`].
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct FunctionBody<'str> {
    /// Immutable identity of this body in the lockstep function registries, or
    /// `None` while the body is *detached* (freshly minted by a pass, not yet
    /// installed under a registry key).
    ///
    /// This field is deliberately absent from the serialized body wire shape.
    /// Bodies are serialized and deserialized as part of [`Context`], whose
    /// custom deserializer restores the registry key here. Standalone body
    /// deserialization therefore does not establish a usable identity. A detached
    /// body is never serialized (bodies are installed at the mint barrier before
    /// any save), so the `None` state never reaches the wire.
    #[serde(skip)]
    id: Option<FunctionId>,

    /// The entry block (dominates all other blocks in this function).
    /// Private: read via [`FunctionBody::root_id`], write via
    /// [`FunctionBody::set_root_id`] (stage 6a §11).
    root: Option<LocalBlockId>,

    /// Instruction storage for this function. Function-scoped: the composite
    /// [`InstructionId`](crate::value::InstructionId) `{ func, local }` indexes
    /// here via `local`. Live payloads are dense; logical IDs are monotonic and
    /// never reused after physical removal.
    pub(crate) insns: StableArena<LocalInsnId, Instruction<'str>>,

    /// Basic-block storage for this function. A block is born here and keeps its
    /// `id.func` for life; live payloads are dense while logical IDs remain
    /// stable and are never reused.
    pub(crate) blocks: StableArena<LocalBlockId, BasicBlock<'str>>,

    /// Body-local ids of the blocks this function owns, in order. Path A forbids
    /// cross-arena ownership, so every entry indexes this function's `blocks`
    /// arena. Kept in sync with each block's `parent`.
    #[serde(default)]
    pub(crate) roster: Vec<LocalBlockId>,

    /// Block-parameter storage for this function.
    pub(crate) params: StableArena<LocalParamId, BlockParam<'str>>,

    /// CFG-edge storage for this function. Keyed by the plain body-local
    /// [`EdgeId`](crate::value::block::EdgeId) (stage 4).
    pub(crate) edges: StableArena<EdgeId, EdgeData>,

    /// Append-only function-local temporary-space storage. Producers migrate
    /// here in later plan-10 commits; the arena is intentionally empty until
    /// then.
    pub(crate) temp_spaces: Registry<crate::value::LocalTempSpaceId, TempSpace>,

    /// Append-only function-local temporary-value storage.
    pub(crate) temps: Registry<crate::value::LocalTempId, Temp<'str>>,

    /// Addresses of every machine instruction lifted into this function, in
    /// ascending order. Recorded during recursive disassembly and preserved
    /// across optimization (which merges blocks and rewrites the IR), so the
    /// raw disassembly view can be reconstructed regardless of CFG changes.
    pub instruction_addrs: BTreeSet<u64>,

    /// Function-local name table for this function's block, instruction,
    /// block-param, and Temp names (ruling 1 of the parallel-passes plan).
    /// Keeping these out of the global [`name_map`](crate::context::Context)
    /// lets two functions name values independently — a prerequisite for
    /// parallel function passes.
    /// A value's own `name` field is the source of truth for rendering; this only
    /// enforces uniqueness and resolves names within the function.
    #[serde(default)]
    pub(crate) names: crate::context::NameTable<'str, LocalValueId>,

    /// The operand relation read backwards: one [`Use`] edge per operand
    /// occurrence in this body, threaded into a singly linked list per used
    /// value (see [`crate::value::uses`]). A value this body owns — an
    /// instruction, block parameter, block or temporary — holds the head of
    /// its own list ([`WithUsers`]); the heads for shared values live in
    /// [`shared_first_use`](Self::shared_first_use). By the SSA ownership
    /// invariant every user of a local value is in this body, so a local
    /// value's list is complete; a shared value's list holds only this body's
    /// uses, which is all any pass needs.
    ///
    /// Derived bookkeeping: skipped by serde and rebuilt from the operands
    /// ([`rebuild_uses`](Self::rebuild_uses)). Kept in step by the verbs that
    /// create, delete and rewrite instructions
    /// ([`push_insn`](Self::push_insn), [`remove_instruction`](Self::remove_instruction),
    /// [`replace_uses_where`](Self::replace_uses_where),
    /// [`replace_instruction_mnemonic`](Self::replace_instruction_mnemonic), …),
    /// which are the only ways an installed instruction's operands change.
    #[serde(skip)]
    pub(crate) uses: UseArena,

    /// Use-list heads for the shared values this body uses — literals, bytes,
    /// varnodes, functions, poison — which cannot carry a head themselves:
    /// use tracking is per body, and bodies are mutated independently. Sparse:
    /// a value with no use in this body has no entry.
    #[serde(skip)]
    pub(crate) shared_first_use: FxHashMap<LocalValueId, UseId>,

    /// The module's shape clock (see
    /// [`Context::revision`](crate::context::Context::revision)), ticked by
    /// every change to which addresses this body's blocks carry. Linked when
    /// the body is installed; a detached body ticks a clock of its own.
    #[serde(skip)]
    clock: crate::context::ShapeClock,
}

/// Aggregate storage statistics for one kind of function-body entity.
///
/// `structural_bytes` counts the payload capacity reserved by the current body
/// arenas. It deliberately excludes allocations owned by payload fields (for
/// example mnemonic operands and block vectors); the Stage 7 probe measures
/// those with allocator accounting in a separate process.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BodyArenaKindStats {
    pub issued: usize,
    pub live: usize,
    pub dead: usize,
    pub capacity: usize,
    pub structural_bytes: usize,
}

/// Aggregate statistics for all four arenas across function bodies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BodyArenaStats {
    pub instructions: BodyArenaKindStats,
    pub blocks: BodyArenaKindStats,
    pub params: BodyArenaKindStats,
    pub edges: BodyArenaKindStats,
}

impl BodyArenaKindStats {
    fn stable_arena<Id: jstd::registry::Identifier, T>(arena: &StableArena<Id, T>) -> Self {
        let issued = arena.issued_len();
        let live = arena.len();
        Self {
            issued,
            live,
            dead: issued - live,
            capacity: arena.capacity(),
            structural_bytes: arena.structural_bytes(),
        }
    }

    fn add_assign(&mut self, other: Self) {
        self.issued += other.issued;
        self.live += other.live;
        self.dead += other.dead;
        self.capacity += other.capacity;
        self.structural_bytes += other.structural_bytes;
    }
}

impl BodyArenaStats {
    pub(crate) fn add_assign(&mut self, other: Self) {
        self.instructions.add_assign(other.instructions);
        self.blocks.add_assign(other.blocks);
        self.params.add_assign(other.params);
        self.edges.add_assign(other.edges);
    }
}

/// Tri-state view of a function's `written_spaces` verdict, distinguishing a
/// never-computed fresh mint from a deliberately recorded ⊤. See
/// [`FunctionSignature::written_spaces`](super::function::FunctionSignature) and
/// [`FunctionRef::written_spaces_state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrittenSpaces<'a> {
    /// Analysis has never recorded a verdict — a freshly minted function.
    /// Treated conservatively (may write any space) but distinct from a
    /// recorded ⊤: it is a candidate for (re-)seeding, not a stale bound.
    Unstamped,
    /// Recorded, but unbounded (⊤): the function may write any space.
    Unbounded,
    /// A recorded exact witnessed bound: a space not listed is never written.
    Bounded(&'a [crate::space::SpaceId]),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FunctionKind {
    #[default]
    Machine,
    Lambda,
}

impl<'str> FunctionInterface<'str> {
    /// The function's entry address, if it has one.
    pub fn address(&self) -> Option<u64> {
        self.address
    }

    /// A fresh interface named `name`, with default (empty) signature/kind.
    pub fn new(name: Cow<'str, str>) -> Self {
        Self {
            name,
            address: None,
            is_external: false,
            signature: None,
            kind: FunctionKind::Machine,
            effects: FunctionEffects::default(),
            import_ordinal: None,
        }
    }

    /// The inferred pointer attributes for positional argument `index`, or
    /// `None` when this function has no analyzed attributes.
    pub fn param_attr(&self, index: usize) -> Option<ParamAttrs> {
        self.signature
            .as_ref()
            .and_then(|s| s.param_attrs.as_ref())
            .and_then(|attrs| attrs.get(index))
            .copied()
    }
}

/// The instructions of a block, in order, walked along their links from
/// both ends (see [`FunctionBody::insn_ids`]).
pub struct InsnIds<'a, 'str> {
    body: &'a FunctionBody<'str>,
    front: Option<LocalInsnId>,
    back: Option<LocalInsnId>,
    remaining: usize,
}

impl Iterator for InsnIds<'_, '_> {
    type Item = LocalInsnId;

    fn next(&mut self) -> Option<LocalInsnId> {
        if self.remaining == 0 {
            return None;
        }
        let local = self.front?;
        self.remaining -= 1;
        self.front = self.body.insns[local].next;
        Some(local)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl DoubleEndedIterator for InsnIds<'_, '_> {
    fn next_back(&mut self) -> Option<LocalInsnId> {
        if self.remaining == 0 {
            return None;
        }
        let local = self.back?;
        self.remaining -= 1;
        self.back = self.body.insns[local].prev;
        Some(local)
    }
}

impl ExactSizeIterator for InsnIds<'_, '_> {}

/// The users of a value, walked along its use list (see
/// [`FunctionBody::local_users_of`]).
pub struct Users<'a, 'str> {
    body: &'a FunctionBody<'str>,
    at: Option<UseId>,
}

impl Iterator for Users<'_, '_> {
    type Item = LocalInsnId;

    fn next(&mut self) -> Option<LocalInsnId> {
        let edge = &self.body.uses[self.at?];
        self.at = edge.next;
        Some(edge.user)
    }
}

impl<'str> FunctionBody<'str> {
    /// Reports the current arena footprint and logical liveness.
    pub fn arena_stats(&self) -> BodyArenaStats {
        BodyArenaStats {
            instructions: BodyArenaKindStats::stable_arena(&self.insns),
            blocks: BodyArenaKindStats::stable_arena(&self.blocks),
            params: BodyArenaKindStats::stable_arena(&self.params),
            edges: BodyArenaKindStats::stable_arena(&self.edges),
        }
    }

    /// Releases structural capacity retained from peak analysis churn.
    ///
    /// Covers the four body arenas plus the block-owned instruction/parameter/
    /// edge collections, the roster, and the use edges. IDs, liveness,
    /// ordering, and every semantic invariant are unchanged — this is an
    /// allocator hint for explicit end-of-mutation boundaries, never a
    /// correctness barrier.
    pub fn shrink_to_fit(&mut self) {
        self.insns.shrink_to_fit();
        self.blocks.shrink_to_fit();
        self.params.shrink_to_fit();
        self.edges.shrink_to_fit();
        self.roster.shrink_to_fit();
        for mut block in self.blocks.iter_mut() {
            block.params.shrink_to_fit();
            block.edges.shrink_to_fit();
        }
        self.uses.shrink_to_fit();
        self.shared_first_use.shrink_to_fit();
    }

    /// Install a registry ID onto a freshly [`detached`](Self::detached) body at
    /// the mint barrier. Panics if the body already carries an id.
    pub fn install_id(&mut self, id: FunctionId) {
        assert!(self.id.is_none(), "body already installed");
        self.id = Some(id);
    }

    /// Resolve one pass-local callee slot throughout this detached or installed
    /// body. Returns the number of call-like instructions patched.
    pub fn resolve_minted_callee(&mut self, slot: u32, real: FunctionId) -> usize {
        let mut patched = 0;
        for mut insn in self.insns.iter_mut() {
            patched += usize::from(insn.mnemonic_mut().resolve_minted_callee(slot, real));
        }
        patched
    }

    /// Resolve every pass-local callee slot in this body against the installed
    /// mapping (`installed[k]` is the real function for slot `k`) in one arena
    /// walk. Returns the number of call-like instructions patched, or the first
    /// slot with no installed function.
    pub fn resolve_minted_callees(
        &mut self,
        installed: &[FunctionId],
    ) -> std::result::Result<usize, u32> {
        let mut patched = 0;
        for mut insn in self.insns.iter_mut() {
            let mnemonic = insn.mnemonic_mut();
            let Some(slot) = mnemonic.minted_callee_slot() else {
                continue;
            };
            let Some(&real) = installed.get(slot as usize) else {
                return Err(slot);
            };
            mnemonic.resolve_minted_callee(slot, real);
            patched += 1;
        }
        Ok(patched)
    }

    /// An empty function *body* carrying the identity `id`. Used for bodies
    /// installed under a known registry key at creation
    /// ([`make`](Self::make)-family constructors). Pass-minted bodies instead use
    /// [`detached`](Self::detached) + [`install_id`](Self::install_id).
    pub fn empty_with_id(id: FunctionId) -> Self {
        Self {
            id: Some(id),
            root: None,
            insns: StableArena::default(),
            blocks: StableArena::default(),
            roster: Vec::new(),
            params: StableArena::default(),
            edges: StableArena::default(),
            temp_spaces: Registry::default(),
            temps: Registry::default(),
            instruction_addrs: BTreeSet::new(),
            names: crate::context::NameTable::default(),
            uses: UseArena::default(),
            shared_first_use: FxHashMap::default(),
            clock: crate::context::ShapeClock::default(),
        }
    }

    /// An empty *detached* function body: no root, empty arenas, and **no**
    /// registry identity yet ([`id`](Self::id) panics until
    /// [`install_id`](Self::install_id) runs at the mint barrier). The interface
    /// lives separately in
    /// [`Context::interfaces`](crate::context::Context::interfaces). This is how a
    /// pass mints a function; the id is stamped by
    /// [`install_id`](Self::install_id) at the mint barrier.
    pub fn detached() -> Self {
        Self {
            id: None,
            root: None,
            insns: StableArena::default(),
            blocks: StableArena::default(),
            roster: Vec::new(),
            params: StableArena::default(),
            edges: StableArena::default(),
            temp_spaces: Registry::default(),
            temps: Registry::default(),
            instruction_addrs: BTreeSet::new(),
            names: crate::context::NameTable::default(),
            uses: UseArena::default(),
            shared_first_use: FxHashMap::default(),
            clock: crate::context::ShapeClock::default(),
        }
    }

    /// This body's immutable function identity. Panics on a detached body (one
    /// minted but not yet installed) — the loud, release-active tripwire against
    /// laundering an owner ID through an uninstalled body.
    pub fn id(&self) -> FunctionId {
        self.id.expect("detached body: no registry id yet")
    }

    /// This body's registry identity, or `None` while detached. The honest
    /// accessor for the install barrier and verifiers.
    pub fn try_id(&self) -> Option<FunctionId> {
        self.id
    }

    /// Restore the skipped identity field from the body's registry key after
    /// deserialization. Serialized function bodies remain wire-compatible with
    /// sessions written before the identity became intrinsic.
    pub(crate) fn rehydrate_id(&mut self, id: FunctionId) {
        self.id = Some(id);
    }

    /// This body's instructions that use `value` as an operand, as body-local
    /// ids, in no particular order. An instruction using `value` twice is
    /// yielded twice. Allocation-free; empty for a value owned by another
    /// function.
    pub fn local_users_of(&self, value: ValueId) -> Users<'_, 'str> {
        let head = match value.owning_function() {
            Some(owner) if owner != self.id() => None,
            _ => self.first_use_of(value.strip_func()),
        };
        Users {
            body: self,
            at: head,
        }
    }

    /// Whether any instruction in this body uses `value`.
    ///
    /// The question `users_of(v).is_empty()` asks, without the allocation it
    /// takes to answer it that way. Dead-code elimination asks it once per
    /// instruction per round, which made building those vectors the single
    /// largest cost of lifting a block.
    pub fn has_users(&self, value: ValueId) -> bool {
        if value
            .owning_function()
            .is_some_and(|owner| owner != self.id())
        {
            return false;
        }
        self.first_use_of(value.strip_func()).is_some()
    }

    /// This body's qualified instruction IDs that use `value`, in no
    /// particular order. A value owned by another function has no users in
    /// this body, even if its local index collides with one of this body's
    /// values.
    pub fn users_of(&self, value: ValueId) -> Vec<InstructionId> {
        let func = self.id();
        self.local_users_of(value)
            .map(|local| InstructionId::new(func, local))
            .collect()
    }

    /// This function's body-local entry block id, if any (raw accessor).
    pub fn root_id(&self) -> Option<LocalBlockId> {
        self.root
    }

    /// Sets this function's body-local entry block id directly, without rostering /
    /// address bookkeeping of [`FunctionMutRef::set_root`]. Routing target for
    /// the raw `.root = …` field writes whose callers have already rostered the
    /// block (stage 6a §11).
    pub fn set_root_id(&mut self, root: Option<LocalBlockId>) {
        self.root = root;
    }

    // ---- function-local raw arena accessors (context-split stage 5a) --------
    //
    // Resolve a composite id against *this* body by its `local` half alone,
    // ignoring `id.func`. Under strict IR locality a body only ever stores its
    // own values, so `id.func` is always this function's id; naming the body
    // explicitly (`ctx.body(fid).block(id)`) instead of routing through
    // `id.func` (`BasicBlock::from_id(ctx, id)`) is what makes the stage-4
    // `func`-strip mechanical — after it, `id` *is* the local index and these
    // bodies are unchanged. These return raw `&`/`&mut` arena values; for the
    // wrapper-ref surface (`successors()`, `name()`, …) use the `*_ref`
    // constructors on a [`QCodeView`] or a [`FunctionRef`].

    /// The block `id`, by its function-local index (see the note above).
    pub fn block(&self, id: BlockId) -> &BasicBlock<'str> {
        assert_eq!(id.func, self.id(), "block belongs to another function");
        &self.blocks[id.local]
    }
    /// The block `id`, mutably.
    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock<'str> {
        assert_eq!(id.func, self.id(), "block belongs to another function");
        &mut self.blocks[id.local]
    }

    /// Whether `id` currently names a live block payload in this body.
    pub fn contains_block(&self, id: BlockId) -> bool {
        id.func == self.id() && self.blocks.contains(id.local)
    }
    /// The instruction `id`, by its function-local index.
    pub fn insn(&self, id: InstructionId) -> &Instruction<'str> {
        assert_eq!(
            id.func,
            self.id(),
            "instruction belongs to another function"
        );
        &self.insns[id.local]
    }
    /// The instruction `id`, mutably.
    pub fn insn_mut(&mut self, id: InstructionId) -> &mut Instruction<'str> {
        assert_eq!(
            id.func,
            self.id(),
            "instruction belongs to another function"
        );
        &mut self.insns[id.local]
    }

    /// Whether `id` currently names a live instruction payload in this body.
    pub fn contains_instruction(&self, id: InstructionId) -> bool {
        id.func == self.id() && self.insns.contains(id.local)
    }
    /// The block parameter `id`, by its function-local index.
    pub fn block_param(&self, id: BlockParamId) -> &BlockParam<'str> {
        assert_eq!(
            id.func,
            self.id(),
            "block parameter belongs to another function"
        );
        &self.params[id.local]
    }
    /// The block parameter `id`, mutably.
    pub fn block_param_mut(&mut self, id: BlockParamId) -> &mut BlockParam<'str> {
        assert_eq!(
            id.func,
            self.id(),
            "block parameter belongs to another function"
        );
        &mut self.params[id.local]
    }

    /// Whether `id` currently names a live block-parameter payload in this body.
    pub fn contains_block_param(&self, id: BlockParamId) -> bool {
        id.func == self.id() && self.params.contains(id.local)
    }

    /// The result type of a **body-local** operand, resolved without a registry
    /// identity. The shared arms (`Literal`/`Bytes`/`Varnode`/`Function`) route
    /// through `shared`; the arena arms (`Instruction`/`BlockParam`/`Temp`/
    /// `BasicBlock`) index this body's own arenas by their bare local index. This
    /// is the id-less twin of [`QCodeView::type_of`] — usable on a detached body.
    pub fn local_type_of(
        &self,
        shared: &crate::context::Shared<'str>,
        id: crate::value::LocalValueId,
    ) -> crate::types::TypeId {
        use crate::value::LocalValueId;
        match id {
            LocalValueId::Literal(id) => shared.values.literals[id].type_id,
            LocalValueId::Bytes(id) => shared.values.bytes[id].type_id,
            LocalValueId::Instruction(local) => self.insns[local].type_id,
            LocalValueId::BlockParam(local) => self.params[local].type_id,
            LocalValueId::Varnode(id) => shared
                .values
                .varnode_types
                .get(&id)
                .copied()
                .unwrap_or_else(|| {
                    shared
                        .types
                        .get_or_make_int(shared.values.varnodes[id].size_bytes())
                }),
            LocalValueId::Temp(local) => shared.types.get_or_make_int(self.temps[local].size),
            LocalValueId::Poison(id) => shared.values.poisons[id].type_id,
            LocalValueId::BasicBlock(_) | LocalValueId::Function(_) => {
                shared.types.get_or_make_int(0)
            }
        }
    }

    /// The stored type of a **body-local** operand, or `None` where the operand
    /// carries no stored type (untyped varnode, temp, block, function). The
    /// id-less twin of [`QCodeView::stored_type_of`].
    pub fn local_stored_type_of(
        &self,
        shared: &crate::context::Shared<'str>,
        id: crate::value::LocalValueId,
    ) -> Option<crate::types::TypeId> {
        use crate::value::LocalValueId;
        match id {
            LocalValueId::Literal(id) => Some(shared.values.literals[id].type_id),
            LocalValueId::Bytes(id) => Some(shared.values.bytes[id].type_id),
            LocalValueId::Instruction(local) => Some(self.insns[local].type_id),
            LocalValueId::BlockParam(local) => Some(self.params[local].type_id),
            LocalValueId::Varnode(id) => shared.values.varnode_types.get(&id).copied(),
            LocalValueId::Poison(id) => Some(shared.values.poisons[id].type_id),
            LocalValueId::Temp(_) | LocalValueId::BasicBlock(_) | LocalValueId::Function(_) => None,
        }
    }

    /// Appends a body-local temporary space and returns its qualified ID.
    pub fn push_temp_space(&mut self, space: TempSpace) -> TempSpaceId {
        TempSpaceId::new(self.id(), self.temp_spaces.push(space))
    }

    /// Appends a body-local temporary value and returns its qualified ID.
    pub fn push_temp(&mut self, mut temp: Temp<'str>) -> TempId {
        temp.first_use = None;
        assert!(
            usize::from(temp.space) < self.temp_spaces.len(),
            "temporary references a missing local space"
        );
        let name = temp.name.clone();
        if let Some(name) = &name {
            assert!(
                !self.names.contains(name),
                "temporary name {name:?} is already registered in this function"
            );
        }
        let local = self.temps.push(temp);
        if let Some(name) = name {
            self.names
                .register(name, LocalValueId::Temp(local), None)
                .expect("temporary name was checked before insertion");
        }
        TempId::new(self.id(), local)
    }

    /// Empties the body and starts a new epoch of its arenas: every id is
    /// invalid, and will be issued again, from zero.
    ///
    /// Only a [`ScratchStore`](crate::lift::ScratchStore) may call this — the
    /// one owner that can vouch no id of the previous epoch survives.
    pub(crate) fn start_epoch(&mut self) {
        self.touch_shape();
        self.insns.clear();
        self.blocks.clear();
        self.params.clear();
        self.edges.clear();
        self.roster.clear();
        self.root = None;
        self.temps.truncate(0);
        self.temp_spaces.truncate(0);
        self.instruction_addrs.clear();
        self.names.clear();
        self.uses.clear();
        self.shared_first_use.clear();
    }

    /// Drops the temporaries and temporary spaces appended since the body had
    /// `temps` and `temp_spaces` of them, with their names.
    ///
    /// For a construction taking back an instruction it could not finish:
    /// nothing may still refer to the dropped temporaries.
    pub(crate) fn take_back_temps(&mut self, temps: usize, temp_spaces: usize) {
        for raw in temps..self.temps.len() {
            if let Some(name) = &self.temps[crate::value::LocalTempId::from(raw)].name {
                self.names.forget(name);
            }
        }
        self.temps.truncate(temps);
        self.temp_spaces.truncate(temp_spaces);
    }

    /// Resolves a qualified temporary-space ID against this body.
    #[track_caller]
    pub fn temp_space(&self, id: TempSpaceId) -> &TempSpace {
        assert_eq!(
            id.func,
            self.id(),
            "temporary space belongs to another function"
        );
        debug_assert!(
            self.contains_temp_space(id),
            "missing temporary space {id:?} in function {:?} (arena length {})",
            self.id(),
            self.temp_spaces.len()
        );
        &self.temp_spaces[id.local]
    }

    /// Iterates over every temporary space owned by this body, in id order.
    pub fn temp_spaces(&self) -> impl Iterator<Item = (TempSpaceId, &TempSpace)> + '_ {
        let func = self.id();
        self.temp_spaces
            .iter()
            .map(move |space| (TempSpaceId::new(func, space.id), space.inner))
    }

    /// Whether `id` names a temporary space in this body.
    pub fn contains_temp_space(&self, id: TempSpaceId) -> bool {
        id.func == self.id() && usize::from(id.local) < self.temp_spaces.len()
    }

    /// Resolves a qualified temporary-value ID against this body.
    #[track_caller]
    pub fn temp(&self, id: TempId) -> &Temp<'str> {
        assert_eq!(id.func, self.id(), "temporary belongs to another function");
        debug_assert!(
            self.contains_temp(id),
            "missing temporary {id:?} in function {:?} (arena length {})",
            self.id(),
            self.temps.len()
        );
        &self.temps[id.local]
    }

    /// Whether `id` names a temporary value in this body.
    pub fn contains_temp(&self, id: TempId) -> bool {
        id.func == self.id() && usize::from(id.local) < self.temps.len()
    }

    /// Physically removes a block parameter and its local bookkeeping.
    /// Positional block and edge-argument rewrites belong to the caller. Those
    /// rewrites may occur later in the same transformation, so outstanding uses
    /// are allowed while the transformation is in progress.
    pub fn remove_block_param(&mut self, id: BlockParamId) {
        assert!(
            self.contains_block_param(id),
            "cannot remove stale param {id:?}"
        );
        let name = self.params[id.local].name.clone();
        if let Some(name) = name {
            self.names.forget(name.as_ref());
        }
        self.drop_uses_of(LocalValueId::BlockParam(id.local));
        self.params.remove(id.local);
    }
    /// The CFG edge `id`, by its function-local index.
    pub fn edge(&self, id: EdgeId) -> &EdgeData {
        &self.edges[id]
    }

    // ---- structural mutation verbs (context-split stage 5b-ii(b)) -----------
    //
    // The single-homed IR mutation surface for a function *body* (design ruling
    // 6). Each verb operates directly on this body's own arenas, reading shared
    // data (types for minting) through an explicit `&Context` where needed. These
    // are the algorithm bodies formerly living on the checked-out mutation path
    // (`value::util::body_mut`), ported here with the routing indirection dropped:
    // `self.function_mut(f)` collapses to `self`, `self.view()` to `self`'s
    // own arena accessors. The owning [`FunctionId`] comes from [`id`](Self::id).

    /// Push a fresh instruction into this body's arena, recording a use edge
    /// for each of its operands.
    pub fn push_insn(&mut self, insn: Instruction<'str>) -> InstructionId {
        InstructionId::new(self.id(), self.push_insn_local(insn))
    }

    /// Push a fresh instruction into this body's arena, recording a use edge
    /// for each of its operands, and return its **body-local** id. The id-less
    /// twin of [`push_insn`](Self::push_insn), usable on a detached body.
    pub fn push_insn_local(&mut self, mut insn: Instruction<'str>) -> LocalInsnId {
        // A caller may clone a linked instruction as its template; the copy is
        // a new value in no block, used by nothing, whatever the original's
        // links said.
        insn.parent = None;
        insn.prev = None;
        insn.next = None;
        insn.first_use = None;
        let local = self.insns.push(insn);
        self.add_operand_uses(local);
        local
    }

    /// Push a fresh block into this body's arena and onto its ownership roster.
    /// Ownership is derived from the storing arena: the returned id's `func` is
    /// this body's own id.
    pub fn push_block(&mut self, block: BasicBlock<'str>) -> BlockId {
        let func = self.id();
        BlockId::new(func, self.push_block_local(block))
    }

    /// Push a fresh block into this body's arena and roster, returning its
    /// **body-local** id. The id-less twin of [`push_block`](Self::push_block),
    /// usable on a detached (uninstalled) body.
    pub fn push_block_local(&mut self, mut block: BasicBlock<'str>) -> LocalBlockId {
        block.first_use = None;
        let local = self.blocks.push(block);
        self.roster.push(local);
        local
    }

    /// Counts a change to this body's address-bearing shape on the module's
    /// clock.
    pub(crate) fn touch_shape(&mut self) {
        self.clock.tick();
    }

    /// Makes this body tick `clock`: the module's, once the body is installed
    /// in it.
    pub(crate) fn link_clock(&mut self, clock: crate::context::ShapeClock) {
        self.clock = clock;
    }

    /// Mint a fresh empty block, owned by this function (arena membership) and
    /// rostered.
    pub fn make_block(&mut self) -> BlockId {
        self.push_block(BasicBlock::detached())
    }

    /// Mint a fresh empty block, returning its **body-local** id. The id-less twin
    /// of [`make_block`](Self::make_block), usable on a detached body.
    pub fn make_block_local(&mut self) -> LocalBlockId {
        self.push_block_local(BasicBlock::detached())
    }

    /// This body's block `block`, by its function-local index (id-free read,
    /// usable on a detached body).
    pub fn block_local(&self, block: LocalBlockId) -> &BasicBlock<'str> {
        &self.blocks[block]
    }

    /// This body's instruction mnemonic, by its function-local index (id-free
    /// read, usable on a detached body).
    pub fn mnemonic_local(&self, insn: LocalInsnId) -> &Mnemonic {
        self.insns[insn].mnemonic()
    }

    /// Push a fresh block parameter into this body's arena.
    pub fn push_block_param(&mut self, mut param: BlockParam<'str>) -> BlockParamId {
        param.first_use = None;
        let local = self.params.push(param);
        BlockParamId::new(self.id(), local)
    }

    /// Push a fresh block parameter into this body's arena and wire it into
    /// `block`'s parameter list, returning its **body-local** id. The id-less twin
    /// of [`push_block_param`](Self::push_block_param), usable on a detached body.
    pub fn push_block_param_local(
        &mut self,
        block: LocalBlockId,
        mut param: BlockParam<'str>,
    ) -> LocalParamId {
        param.first_use = None;
        let local = self.params.push(param);
        self.blocks[block].params.push(local);
        local
    }

    /// Append an already-created instruction to the end of `block`, setting its
    /// parent (id-free; the mutation twin of [`BaseRef::push_insn`], usable on a
    /// detached body).
    pub fn append_insn_local(&mut self, block: LocalBlockId, insn: LocalInsnId) {
        self.link_last(block, insn);
    }

    // ---- the instruction list of a block ------------------------------------
    //
    // A block's instruction order is a doubly linked list threaded through
    // the instructions (`Instruction::prev`/`next`), the block holding its
    // ends and length (`InsnList`). Linking before an instruction, after
    // one, at the end, and unlinking cost the same however long the block
    // is; walking it costs its length. These four verbs are the only ones
    // that touch the links, and keep `parent` in step: linked means in a
    // block, unlinked means in none. Linking an instruction that is in a
    // block moves it: the verbs unlink first, so no list is ever left with
    // a member whose links lead elsewhere.

    /// The instructions of `block`, in order.
    pub fn insn_ids(&self, block: LocalBlockId) -> InsnIds<'_, 'str> {
        let list = self.blocks[block].instructions;
        InsnIds {
            body: self,
            front: list.first,
            back: list.last,
            remaining: list.len,
        }
    }

    /// Links `insn` at the end of `block`, taking it out of the block it was
    /// in, if any.
    pub fn link_last(&mut self, block: LocalBlockId, insn: LocalInsnId) {
        self.unlink(insn);
        let last = self.blocks[block].instructions.last;
        {
            let i = &mut self.insns[insn];
            i.parent = Some(block);
            i.prev = last;
            i.next = None;
        }
        match last {
            Some(last) => self.insns[last].next = Some(insn),
            None => self.blocks[block].instructions.first = Some(insn),
        }
        let list = &mut self.blocks[block].instructions;
        list.last = Some(insn);
        list.len += 1;
    }

    /// Links `insn` immediately before `before`, which must be in `block`,
    /// taking `insn` out of the block it was in, if any.
    pub fn link_before(&mut self, block: LocalBlockId, before: LocalInsnId, insn: LocalInsnId) {
        if insn == before {
            return;
        }
        self.unlink(insn);
        debug_assert_eq!(
            self.insns[before].parent,
            Some(block),
            "{before:?} is not in {block:?}"
        );
        let prev = self.insns[before].prev;
        {
            let i = &mut self.insns[insn];
            i.parent = Some(block);
            i.prev = prev;
            i.next = Some(before);
        }
        self.insns[before].prev = Some(insn);
        match prev {
            Some(prev) => self.insns[prev].next = Some(insn),
            None => self.blocks[block].instructions.first = Some(insn),
        }
        self.blocks[block].instructions.len += 1;
    }

    /// Links `insn` immediately after `after`, which must be in `block`.
    pub fn link_after(&mut self, block: LocalBlockId, after: LocalInsnId, insn: LocalInsnId) {
        debug_assert_eq!(
            self.insns[after].parent,
            Some(block),
            "{after:?} is not in {block:?}"
        );
        match self.insns[after].next {
            Some(next) => self.link_before(block, next, insn),
            None => self.link_last(block, insn),
        }
    }

    /// Takes `insn` out of its block's list, leaving it in no block. The
    /// instruction itself stays in the arena, with its uses.
    pub fn unlink(&mut self, insn: LocalInsnId) {
        let (block, prev, next) = {
            let i = &mut self.insns[insn];
            let Some(block) = i.parent.take() else {
                return;
            };
            (block, i.prev.take(), i.next.take())
        };
        match prev {
            Some(prev) => self.insns[prev].next = next,
            None => self.blocks[block].instructions.first = next,
        }
        match next {
            Some(next) => self.insns[next].prev = prev,
            None => self.blocks[block].instructions.last = prev,
        }
        self.blocks[block].instructions.len -= 1;
    }

    /// Links `insn` at position `index` of `block`, walking to it.
    pub fn insert_insn_at(&mut self, block: LocalBlockId, index: usize, insn: LocalInsnId) {
        match self.insn_ids(block).nth(index) {
            Some(at) => self.link_before(block, at, insn),
            None => {
                assert_eq!(
                    index, self.blocks[block].instructions.len,
                    "index past the end"
                );
                self.link_last(block, insn);
            }
        }
    }

    /// Empties `block`'s list, returning its instructions in order, each
    /// now in no block.
    pub fn take_insns(&mut self, block: LocalBlockId) -> Vec<LocalInsnId> {
        let ids: Vec<LocalInsnId> = self.insn_ids(block).collect();
        for &id in &ids {
            let i = &mut self.insns[id];
            i.parent = None;
            i.prev = None;
            i.next = None;
        }
        self.blocks[block].instructions = InsnList::default();
        ids
    }

    /// Mint an `Int(size)`-typed instruction with `mnemonic` (the type is minted
    /// in `shared`'s interner through its `&self` path).
    pub fn push_mnemonic(
        &mut self,
        shared: &crate::context::Shared<'str>,
        mnemonic: Mnemonic,
        size: usize,
    ) -> InstructionId {
        let type_id = shared.types.get_or_make_int(size);
        let insn = Instruction::new(type_id, mnemonic);
        self.push_insn(insn)
    }

    /// Mint an instruction with `mnemonic` and an explicit result `type_id`.
    pub fn push_mnemonic_with_type(
        &mut self,
        mnemonic: Mnemonic,
        type_id: crate::types::TypeId,
    ) -> InstructionId {
        let insn = Instruction::new(type_id, mnemonic);
        self.push_insn(insn)
    }

    /// Mint an instruction with `mnemonic` and an explicit result `type_id`,
    /// returning its **body-local** id. The id-less twin of
    /// [`push_mnemonic_with_type`](Self::push_mnemonic_with_type).
    pub fn push_mnemonic_with_type_local(
        &mut self,
        mnemonic: Mnemonic,
        type_id: crate::types::TypeId,
    ) -> LocalInsnId {
        self.push_insn_local(Instruction::new(type_id, mnemonic))
    }

    /// Insert `insn` immediately before `before` in `block`. Panics if `before`
    /// is not in `block`.
    pub fn insert_insn_before(
        &mut self,
        block: BlockId,
        before: InstructionId,
        insn: InstructionId,
    ) {
        assert_eq!(
            self.insn(before).parent,
            Some(block.local),
            "before not in block"
        );
        self.link_before(
            block.local,
            before.localize(block.func),
            insn.localize(block.func),
        );
    }

    /// Move the live, non-terminator instruction `insn` immediately before the
    /// live instruction `before`, inferring the destination block from
    /// `before`. The moved instruction keeps its ID, payload, name, and use
    /// edges. Supports both cross-block motion and reordering within one
    /// block.
    pub fn move_insn_before(&mut self, insn: InstructionId, before: InstructionId) {
        let id = self.id();
        assert_eq!(insn.func, id, "instruction belongs to another function");
        assert_eq!(
            before.func, id,
            "anchor instruction belongs to another function"
        );
        if insn == before {
            return;
        }
        assert!(
            !self.insn(insn).mnemonic().is_terminator(),
            "moving a terminator requires updating its CFG edges"
        );

        assert!(
            self.insn(insn).parent.is_some(),
            "moved instruction must belong to a block"
        );
        let target = self
            .insn(before)
            .parent
            .expect("anchor instruction must belong to a block");
        self.unlink(insn.local);
        self.link_before(target, before.local, insn.local);
    }

    /// Add a directed CFG edge `from -> to`, stored in this body's edge arena and
    /// linked into both incident blocks' edge sets.
    pub fn add_cfg_edge(&mut self, from: BlockId, to: BlockId) -> EdgeId {
        self.add_cfg_edge_local(from.local, to.local)
    }

    /// Add a directed CFG edge `from -> to` over **body-local** block ids (id-free;
    /// the twin of [`add_cfg_edge`](Self::add_cfg_edge), usable on a detached body).
    pub fn add_cfg_edge_local(&mut self, from: LocalBlockId, to: LocalBlockId) -> EdgeId {
        let edge_id = self.edges.push(EdgeData { from, to });
        self.blocks[from].edges.insert(edge_id);
        self.blocks[to].edges.insert(edge_id);
        edge_id
    }

    /// Remove CFG edge `edge_id`, unlinking it from both incident blocks and
    /// physically dropping its payload.
    pub fn remove_cfg_edge(&mut self, edge_id: EdgeId) {
        let EdgeData { from, to } = *self.edge(edge_id);
        let func = self.id();
        self.block_mut(BlockId::new(func, from))
            .edges
            .remove(&edge_id);
        self.block_mut(BlockId::new(func, to))
            .edges
            .remove(&edge_id);
        self.edges.remove(edge_id);
    }

    /// Replace every use of `old` in this body with `new`, moving the use
    /// edges along. For a value this body owns that is every use there is;
    /// for a shared value it is this body's uses of it.
    pub fn replace_all_uses_with(&mut self, old: ValueId, new: ValueId) {
        self.replace_uses_where(old, new, |_, _, _| true);
    }

    /// Replace the uses of `old` in this body that `select` picks with `new`,
    /// moving their use edges along. `select` sees the using instruction and
    /// the operand's index in it, so it can keep to a block, or to one of two
    /// occurrences in `%x = add %old, %old`, which are distinct edges.
    /// Allocation-free: it walks `old`'s use list once. The lists of `old`
    /// and `new` are in flux during the walk, so `select` should read the
    /// instructions, not the users, of either.
    ///
    /// `old` and `new` must each be a value this body owns or a shared value.
    pub fn replace_uses_where(
        &mut self,
        old: ValueId,
        new: ValueId,
        mut select: impl FnMut(&FunctionBody<'str>, InstructionId, usize) -> bool,
    ) {
        if old == new {
            return;
        }
        let func = self.id();
        if let Some(old_owner) = old.owning_function() {
            assert_eq!(
                old_owner, func,
                "cannot replace uses of a value owned by another function"
            );
        }
        if let Some(new_owner) = new.owning_function() {
            assert_eq!(
                new_owner, func,
                "cannot replace uses with a value owned by another function"
            );
        }
        let old = old.strip_func();
        let new = new.strip_func();
        // The heads are written back once at the end, not once per edge
        // moved: the walk starts at `old`'s front, where every moved edge
        // comes off, and every one goes onto `new`'s front.
        let new_has_home = self.has_use_home(new);
        let mut new_head = self.first_use_of(new);
        let mut old_head = self.first_use_of(old);
        let mut prev: Option<UseId> = None;
        let mut at = old_head;
        while let Some(edge) = at {
            let Use {
                user,
                operand_index,
                next,
                ..
            } = self.uses[edge];
            at = next;
            if !select(
                self,
                InstructionId::new(func, user),
                usize::from(operand_index),
            ) {
                prev = Some(edge);
                continue;
            }
            match prev {
                None => old_head = next,
                Some(prev) => self.uses[prev].next = next,
            }
            self.insns[user]
                .mnemonic_mut()
                .set_operand(usize::from(operand_index), new);
            if new_has_home {
                let moved = &mut self.uses[edge];
                moved.value = new;
                moved.next = new_head;
                new_head = Some(edge);
            } else {
                // A dangling operand records no use.
                self.uses.remove(edge);
            }
        }
        if self.has_use_home(old) {
            self.set_first_use_of(old, old_head);
        }
        if new_has_home {
            self.set_first_use_of(new, new_head);
        }
    }

    /// Sets operand `index` of instruction `id` to `new`, moving its use edge
    /// along: the way to rewrite one operand of an installed instruction.
    /// `new` must be a value this body owns or a shared value.
    pub fn replace_operand(&mut self, id: InstructionId, index: usize, new: ValueId) {
        assert_eq!(
            id.func,
            self.id(),
            "instruction belongs to another function"
        );
        if let Some(new_owner) = new.owning_function() {
            assert_eq!(
                new_owner,
                self.id(),
                "cannot use a value owned by another function"
            );
        }
        let old = self.insns[id.local]
            .mnemonic()
            .operand(index)
            .unwrap_or_else(|| panic!("{id:?} has no operand {index}"));
        self.replace_use(old, id.local, index, new.strip_func());
    }

    /// Replace every use of instruction `id` with `new`, then remove `id` —
    /// the standard "rewrite to a cheaper value" epilogue
    /// ([`replace_all_uses_with`](Self::replace_all_uses_with) +
    /// [`remove_instruction`](Self::remove_instruction)).
    pub fn replace_instruction(&mut self, id: InstructionId, new: ValueId) {
        // Replacing an instruction with itself is a contradiction: the use
        // forwarding is a no-op, so removing `id` would delete a value that is
        // still referenced. Leave it in place.
        if new == ValueId::Instruction(id) {
            return;
        }
        self.replace_all_uses_with(ValueId::Instruction(id), new);
        self.remove_instruction(id);
    }

    /// Physically removes a set of instructions after taking down their use
    /// edges. Call after removing them from their parent blocks and unlinking
    /// any CFG edges owned by terminators.
    pub fn remove_instructions(&mut self, dead: &FxHashSet<LocalInsnId>) {
        // Removed in id order, so the arena's physical order — which the
        // dense walks observe — does not depend on the set's iteration order.
        let mut ids: Vec<_> = dead.iter().copied().collect();
        ids.sort_unstable();
        for &id in &ids {
            assert!(
                self.insns.contains(id),
                "cannot remove stale instruction {id:?}"
            );
        }
        self.remove_uses_of_dead_instructions(dead);
        for id in ids {
            self.insns.remove(id);
        }
    }

    /// Remove instruction `id` from its block, unlink its outgoing CFG edges if a
    /// terminator, clear its name, take down its use edges, and physically
    /// drop its payload.
    pub fn remove_instruction(&mut self, id: InstructionId) {
        assert_eq!(
            id.func,
            self.id(),
            "instruction belongs to another function"
        );
        let (parent, name, is_terminator) = {
            let insn = self.insn(id);
            (
                insn.parent.map(|l| BlockId::new(self.id(), l)),
                insn.name.clone(),
                insn.mnemonic().is_terminator(),
            )
        };

        if let Some(block_id) = parent {
            self.unlink(id.local);
            if is_terminator {
                let mut succ: Vec<EdgeId> = {
                    let block = self.block(block_id);
                    block
                        .edges
                        .iter()
                        .copied()
                        .filter(|&e| self.edge(e).from == block_id.local)
                        .collect()
                };
                succ.sort_unstable();
                for edge_id in succ {
                    self.remove_cfg_edge(edge_id);
                }
            }
        }

        if let Some(n) = name {
            self.names.forget(n.as_ref());
        }
        self.remove_operand_uses(id.local);
        self.drop_uses_of(LocalValueId::Instruction(id.local));
        self.insns.remove(id.local);
    }

    /// Removes several non-terminator instructions of one block at once.
    ///
    /// Each comes out of the block's list in constant time either way; what
    /// this saves over [`remove_instruction`](Self::remove_instruction) is
    /// the use lists, where each operand's list is walked once rather than
    /// once per dead user. Lifting an absorbed guest basic block deletes
    /// hundreds of instructions sharing a few operands, and that product was
    /// a real share of translation time.
    ///
    /// Terminators are rejected rather than handled: removing one has to tear
    /// down CFG edges too, and no caller of this deletes one — dead-code
    /// elimination will not touch a terminator, and store forwarding removes
    /// only loads and stores.
    pub fn remove_block_instructions(&mut self, block_id: BlockId, dead: &FxHashSet<LocalInsnId>) {
        assert_eq!(
            block_id.func,
            self.id(),
            "block belongs to another function"
        );
        if dead.is_empty() {
            return;
        }

        let mut names = Vec::new();
        for &id in dead {
            let insn = &self.insns[id];
            assert!(
                !insn.mnemonic().is_terminator(),
                "bulk removal does not unlink CFG edges; {id:?} is a terminator"
            );
            if let Some(name) = insn.name.clone() {
                names.push(name);
            }
        }

        for &id in dead {
            self.unlink(id);
        }
        self.purge_instructions(dead, names);
    }

    /// Forgets `dead`'s names, takes down their use edges, and drops their
    /// payloads.
    ///
    /// The shared tail of removing instructions in bulk. It does not touch any
    /// block's instruction list — the caller has already unlinked them.
    fn purge_instructions(&mut self, dead: &FxHashSet<LocalInsnId>, names: Vec<Cow<'str, str>>) {
        for name in names {
            self.names.forget(name.as_ref());
        }
        self.remove_uses_of_dead_instructions(dead);
        for &id in dead {
            self.insns.remove(id);
        }
    }

    /// Rehome `remove`'s outgoing CFG edges onto `keep`. The direct edge and
    /// `keep`'s forwarding terminator have already been removed by the caller.
    /// Moves `insn` and everything after it in its block — the terminator
    /// included — into a fresh block, and returns that block.
    ///
    /// The original block keeps its identity, its address, its parameters and
    /// its incoming edges, and is left *unterminated*: the caller ends it,
    /// typically with a branch to the new block or a conditional branch that
    /// reaches the new block one way or another. The outgoing edges follow the
    /// terminator to the new block. Values defined before the split stay
    /// visible to the instructions after it, as SSA allows across blocks.
    ///
    /// Unlike an address split this moves code rather than discarding it, so
    /// it is for rewriting a block in place — inserting a conditional detour —
    /// not for establishing a new branch target in the guest.
    pub fn split_block_before(&mut self, block: BlockId, insn: InstructionId) -> BlockId {
        assert_eq!(block.func, self.id(), "block belongs to another function");
        assert_eq!(
            insn.func,
            self.id(),
            "instruction belongs to another function"
        );
        assert_eq!(
            self.insn(insn).parent,
            Some(block.local),
            "split point is not in the block"
        );
        let tail = self.make_block();
        let mut moved: Vec<LocalInsnId> = Vec::new();
        let mut at = Some(insn.local);
        while let Some(local) = at {
            at = self.insns[local].next;
            moved.push(local);
        }
        for &local in &moved {
            self.unlink(local);
            self.link_last(tail.local, local);
        }
        self.rehome_outgoing_edges(tail, block);
        tail
    }

    pub fn rehome_outgoing_edges(&mut self, keep: BlockId, remove: BlockId) {
        let outgoing: Vec<EdgeId> = {
            let block = self.block(remove);
            block
                .edges
                .iter()
                .copied()
                .filter(|&e| self.edge(e).from == remove.local)
                .collect()
        };
        for eid in outgoing {
            self.edges[eid].from = keep.local;
            self.block_mut(keep).edges.insert(eid);
            self.block_mut(remove).edges.remove(&eid);
        }
    }

    /// Replace an instruction's mnemonic in place, keeping the use edges in
    /// step: the old operands' uses come down, the new operands' go up.
    pub fn replace_instruction_mnemonic(&mut self, id: InstructionId, mnemonic: Mnemonic) {
        assert_eq!(
            id.func,
            self.id(),
            "instruction belongs to another function"
        );
        self.replace_instruction_mnemonic_local(id.local, mnemonic);
    }

    /// Replace an instruction's mnemonic in place, keeping the use edges in
    /// step, over a **body-local** instruction id (id-free; the twin of
    /// [`replace_instruction_mnemonic`](Self::replace_instruction_mnemonic),
    /// usable on a detached body).
    pub fn replace_instruction_mnemonic_local(&mut self, id: LocalInsnId, mnemonic: Mnemonic) {
        self.remove_operand_uses(id);
        *self.insns[id].mnemonic_mut() = mnemonic;
        self.add_operand_uses(id);
    }

    // ---- use edges ---------------------------------------------------------
    //
    // Every operand occurrence in this body is one `Use` edge, threaded into a
    // singly linked list per used value whose head the value holds (or, for a
    // shared value, `shared_first_use` holds). The verbs below are the only
    // ones that touch the edges; the public verbs that create, delete and
    // rewrite instructions are built on them, and an installed instruction's
    // operands never change any other way. Prepending keeps every edge
    // operation constant-time except for finding one edge in a list, which
    // costs the value's fanout — small, in practice.

    /// Whether this body has storage for `value`, and so a place for its
    /// use-list head. Always true for a shared value.
    fn has_use_home(&self, value: LocalValueId) -> bool {
        match value {
            LocalValueId::Instruction(id) => self.insns.contains(id),
            LocalValueId::BlockParam(id) => self.params.contains(id),
            LocalValueId::BasicBlock(id) => self.blocks.contains(id),
            LocalValueId::Temp(id) => usize::from(id) < self.temps.len(),
            LocalValueId::Literal(_)
            | LocalValueId::Bytes(_)
            | LocalValueId::Varnode(_)
            | LocalValueId::Function(_)
            | LocalValueId::Poison(_) => true,
        }
    }

    /// The head of `value`'s use list: `None` when nothing in this body uses
    /// it, or when this body has no storage for it.
    pub(crate) fn first_use_of(&self, value: LocalValueId) -> Option<UseId> {
        match value {
            LocalValueId::Instruction(id) => self.insns.get(id).and_then(|i| i.first_use()),
            LocalValueId::BlockParam(id) => self.params.get(id).and_then(|p| p.first_use()),
            LocalValueId::BasicBlock(id) => self.blocks.get(id).and_then(|b| b.first_use()),
            LocalValueId::Temp(id) => (usize::from(id) < self.temps.len())
                .then(|| self.temps[id].first_use())
                .flatten(),
            LocalValueId::Literal(_)
            | LocalValueId::Bytes(_)
            | LocalValueId::Varnode(_)
            | LocalValueId::Function(_)
            | LocalValueId::Poison(_) => self.shared_first_use.get(&value).copied(),
        }
    }

    /// Sets the head of `value`'s use list. Panics when this body has no
    /// storage for `value`: check [`has_use_home`](Self::has_use_home) first.
    fn set_first_use_of(&mut self, value: LocalValueId, head: Option<UseId>) {
        match value {
            LocalValueId::Instruction(id) => *self.insns[id].first_use_mut() = head,
            LocalValueId::BlockParam(id) => *self.params[id].first_use_mut() = head,
            LocalValueId::BasicBlock(id) => *self.blocks[id].first_use_mut() = head,
            LocalValueId::Temp(id) => *self.temps[id].first_use_mut() = head,
            LocalValueId::Literal(_)
            | LocalValueId::Bytes(_)
            | LocalValueId::Varnode(_)
            | LocalValueId::Function(_)
            | LocalValueId::Poison(_) => match head {
                Some(head) => {
                    self.shared_first_use.insert(value, head);
                }
                None => {
                    self.shared_first_use.remove(&value);
                }
            },
        }
    }

    /// Records that `user`'s operand `operand_index` names `value`, at the
    /// front of `value`'s use list.
    ///
    /// `None`, and nothing recorded, when this body has no storage for
    /// `value`: the operand is dangling, which the integrity check reports,
    /// and which a clone into another function's arenas leaves behind until
    /// the caller remaps the operands and rebuilds the edges.
    pub(crate) fn add_use(
        &mut self,
        value: LocalValueId,
        user: LocalInsnId,
        operand_index: usize,
    ) -> Option<UseId> {
        if !self.has_use_home(value) {
            return None;
        }
        let operand_index = u16::try_from(operand_index).expect("operand index fits a use edge");
        let next = self.first_use_of(value);
        let edge = self.uses.push(Use {
            value,
            user,
            operand_index,
            next,
        });
        self.set_first_use_of(value, Some(edge));
        Some(edge)
    }

    /// Finds the edge recording that `user`'s operand `operand_index` names
    /// `value`, with the edge before it in `value`'s list (`None` at the
    /// head).
    fn find_use(
        &self,
        value: LocalValueId,
        user: LocalInsnId,
        operand_index: usize,
    ) -> Option<(Option<UseId>, UseId)> {
        let mut prev = None;
        let mut at = self.first_use_of(value);
        while let Some(edge) = at {
            let Use {
                user: u,
                operand_index: i,
                next,
                ..
            } = self.uses[edge];
            if u == user && usize::from(i) == operand_index {
                return Some((prev, edge));
            }
            prev = Some(edge);
            at = next;
        }
        None
    }

    /// Takes down the edge recording that `user`'s operand `operand_index`
    /// names `value`. Returns whether there was one.
    pub(crate) fn remove_use(
        &mut self,
        value: LocalValueId,
        user: LocalInsnId,
        operand_index: usize,
    ) -> bool {
        let Some((prev, edge)) = self.find_use(value, user, operand_index) else {
            return false;
        };
        self.unlink_use(value, prev, edge);
        self.uses.remove(edge);
        true
    }

    /// Rewrites `user`'s operand `operand_index` from `old` to `new`, moving
    /// its edge from `old`'s list to `new`'s.
    pub(crate) fn replace_use(
        &mut self,
        old: LocalValueId,
        user: LocalInsnId,
        operand_index: usize,
        new: LocalValueId,
    ) {
        debug_assert_eq!(
            self.insns[user].mnemonic().operand(operand_index),
            Some(old),
            "operand {operand_index} of {user:?} is not {old:?}"
        );
        if old == new {
            return;
        }
        self.insns[user]
            .mnemonic_mut()
            .set_operand(operand_index, new);
        match self.find_use(old, user, operand_index) {
            Some((prev, edge)) => {
                self.unlink_use(old, prev, edge);
                self.relink_use(edge, new);
            }
            // A dangling operand had no edge to move; it may have a home now.
            None => {
                self.add_use(new, user, operand_index);
            }
        }
    }

    /// Unlinks `edge` from `value`'s list, `prev` being the edge before it
    /// (`None` at the head). The edge stays allocated for the caller to free
    /// or relink.
    fn unlink_use(&mut self, value: LocalValueId, prev: Option<UseId>, edge: UseId) {
        let next = self.uses[edge].next;
        match prev {
            None => self.set_first_use_of(value, next),
            Some(prev) => self.uses[prev].next = next,
        }
    }

    /// Points the unlinked `edge` at `new` and prepends it to `new`'s list —
    /// or frees it, when this body has no storage for `new`.
    fn relink_use(&mut self, edge: UseId, new: LocalValueId) {
        if !self.has_use_home(new) {
            self.uses.remove(edge);
            return;
        }
        let head = self.first_use_of(new);
        let use_ = &mut self.uses[edge];
        use_.value = new;
        use_.next = head;
        self.set_first_use_of(new, Some(edge));
    }

    /// Records a use edge for each operand of `insn`: on creation, and after
    /// its mnemonic is swapped.
    fn add_operand_uses(&mut self, insn: LocalInsnId) {
        // Inline for up to two operands, which covers most instructions.
        let operands = self.insns[insn].mnemonic().args();
        for (index, value) in operands.into_iter().enumerate() {
            self.add_use(value, insn, index);
        }
    }

    /// Takes down the use edges of every operand of `insn`: before it is
    /// deleted, or its mnemonic swapped.
    fn remove_operand_uses(&mut self, insn: LocalInsnId) {
        let operands = self.insns[insn].mnemonic().args();
        for (index, value) in operands.into_iter().enumerate() {
            self.remove_use(value, insn, index);
        }
    }

    /// Frees every edge in `value`'s use list, for a value about to lose its
    /// storage. Its users, if any are left, keep naming it: they are dangling
    /// operands, allowed while a transformation is in progress and reported
    /// by the integrity check once it is not.
    fn drop_uses_of(&mut self, value: LocalValueId) {
        let mut at = self.first_use_of(value);
        while let Some(edge) = at {
            at = self.uses.remove(edge).next;
        }
        if self.has_use_home(value) {
            self.set_first_use_of(value, None);
        }
    }

    /// Takes down every use edge in and out of the instructions in `dead`,
    /// which are about to be deleted.
    ///
    /// The bulk path: rather than finding each dead instruction's edge in
    /// each of its operands' lists — a walk per dead user — this walks the
    /// list of each operand any dead instruction names once, freeing the
    /// edges whose user is dead. A lifted block deletes hundreds of
    /// instructions sharing a few operands, and that product mattered. An
    /// operand that is itself dead is not walked: its whole list goes with
    /// it.
    pub(crate) fn remove_uses_of_dead_instructions(&mut self, dead: &FxHashSet<LocalInsnId>) {
        let mut operands: FxHashSet<LocalValueId> = FxHashSet::default();
        for &id in dead {
            self.insns[id].mnemonic().for_each_operand(|value| {
                if let LocalValueId::Instruction(user) = value
                    && dead.contains(&user)
                {
                    return;
                }
                operands.insert(value);
            });
        }
        for value in operands {
            // The head is written back once, not once per edge unlinked
            // at the front — which, the lists being prepended to, is where
            // the edges of the newest users are.
            let mut head = self.first_use_of(value);
            let mut prev: Option<UseId> = None;
            let mut at = head;
            while let Some(edge) = at {
                let Use { user, next, .. } = self.uses[edge];
                at = next;
                if dead.contains(&user) {
                    match prev {
                        None => head = next,
                        Some(prev) => self.uses[prev].next = next,
                    }
                    self.uses.remove(edge);
                } else {
                    prev = Some(edge);
                }
            }
            self.set_first_use_of(value, head);
        }
        for &id in dead {
            self.drop_uses_of(LocalValueId::Instruction(id));
        }
    }

    /// Rebuilds every use edge from the operands of this body's live
    /// instructions: after deserialization, which does not carry the edges,
    /// and after a relocation that rewrote operands wholesale.
    pub(crate) fn rebuild_uses(&mut self) {
        self.uses.clear();
        self.shared_first_use.clear();
        for mut insn in self.insns.iter_mut() {
            *insn.first_use_mut() = None;
        }
        for mut param in self.params.iter_mut() {
            *param.first_use_mut() = None;
        }
        for mut block in self.blocks.iter_mut() {
            *block.first_use_mut() = None;
        }
        for mut temp in self.temps.iter_mut() {
            *temp.first_use_mut() = None;
        }
        let live: Vec<LocalInsnId> = self.insns.iter().map(|insn| insn.id).collect();
        for insn in live {
            self.add_operand_uses(insn);
        }
    }

    /// Every live use edge in this body, for the integrity check.
    pub(crate) fn use_edges(&self) -> impl Iterator<Item = Identified<UseId, &Use>> + '_ {
        self.uses.iter()
    }

    /// Set `block`'s name and register it in this body's local name table, over a
    /// **body-local** block id (id-free; the twin of
    /// [`BaseRef::rename_local`](crate::value::util::base_ref::BaseRef::rename_local)
    /// restricted to the id-free local parts, usable on a detached body). Errors
    /// only on a duplicate name.
    pub fn rename_block_local(&mut self, block: LocalBlockId, name: Cow<'str, str>) -> Result<()> {
        let target = LocalValueId::BasicBlock(block);
        if let Some(existing) = self.names.get(&name) {
            return if existing == target {
                Ok(())
            } else {
                Err(Error::spanless(ErrorTy::DuplicateName(name.to_string())))
            };
        }
        let old_name = self.blocks[block].local_name().map(str::to_owned);
        self.names
            .register(name.clone(), target, old_name.as_deref())?;
        self.blocks[block].set_name(Some(name));
        Ok(())
    }

    /// Drop `block` from this body's ownership roster. Ownership is derived from
    /// the storing arena (`block.func`).
    pub fn unroster_block(&mut self, block: BlockId) {
        self.roster.retain(|&b| b != block.localize(block.func));
    }

    /// Remove `block` from this body: unlink every incident CFG edge, remove its
    /// instructions and params, clear ownership metadata, then drop its payload.
    /// Empties `block` of code, keeping the block itself.
    ///
    /// Only the *outgoing* edges go, because those are owned by the terminator
    /// being removed; the incoming ones belong to other blocks' terminators,
    /// which still name this block and must keep resolving to it. That is the
    /// point of clearing rather than deleting: every branch already targeting
    /// this block stays valid while its contents are rebuilt.
    pub fn clear_block_instructions(&mut self, block: BlockId) {
        assert_eq!(block.func, self.id(), "block belongs to another function");
        let mut outgoing: Vec<EdgeId> = self
            .block(block)
            .edges
            .iter()
            .copied()
            .filter(|&edge| self.edges[edge].from == block.local)
            .collect();
        outgoing.sort_unstable();
        for edge in outgoing {
            self.remove_cfg_edge(edge);
        }
        // The list is emptied in one move and the instructions purged as a
        // set, each operand's user list pruned once: the blocks this clears
        // are absorbed guest basic blocks, thousands of instructions long,
        // sharing a few operands.
        let insns = self.take_insns(block.local);
        let dead: FxHashSet<LocalInsnId> = insns.iter().copied().collect();
        let names: Vec<Cow<'str, str>> = insns
            .iter()
            .filter_map(|&local| self.insns[local].name.clone())
            .collect();
        self.purge_instructions(&dead, names);
    }

    pub fn delete_block(&mut self, block: BlockId) {
        assert_eq!(block.func, self.id(), "block belongs to another function");
        let mut edges: Vec<EdgeId> = self.block(block).edges.iter().copied().collect();
        edges.sort_unstable();
        for edge in edges {
            self.remove_cfg_edge(edge);
        }
        // Each removal unlinks the instruction, so the walk snapshots first.
        let insns: Vec<LocalInsnId> = self.insn_ids(block.local).collect();
        for insn in insns {
            self.remove_instruction(InstructionId::new(block.func, insn));
        }
        let params: Vec<BlockParamId> = self
            .block(block)
            .params
            .iter()
            .map(|&local| BlockParamId::new(self.id(), local))
            .collect();
        for param in params {
            self.remove_block_param(param);
        }
        let name = self.block(block).local_name().map(str::to_owned);
        self.unroster_block(block);
        if self.root == Some(block.local) {
            self.root = None;
        }
        if let Some(name) = name {
            self.names.forget(&name);
        }
        let addressed = {
            let block = &self.blocks[block.local];
            block.address.is_some() || !block.extra_addresses.is_empty()
        };
        self.blocks.remove(block.local);
        if addressed {
            self.touch_shape();
        }
    }

    /// Absorb `other` into `keep`: drop `keep`'s terminal branch, append `other`'s
    /// instructions, rehome its outgoing edges, and remove it. `edge_ab` is the
    /// direct edge `keep -> other`.
    pub fn absorb_block(&mut self, keep: BlockId, other: BlockId, edge_ab: EdgeId) {
        assert_eq!(
            keep.func, other.func,
            "cannot absorb across function arenas"
        );
        let (branch_id, branch_args) = self
            .block(keep)
            .last_insn()
            .and_then(
                |local| match self.insn(InstructionId::new(keep.func, local)).mnemonic() {
                    Mnemonic::Branch(branch) if BlockId::new(keep.func, branch.target) == other => {
                        Some((InstructionId::new(keep.func, local), branch.args.clone()))
                    }
                    _ => None,
                },
            )
            .expect("absorbed block must be reached by keep's terminal branch");
        let other_params: Vec<_> = self
            .block(other)
            .params
            .iter()
            .map(|&local| BlockParamId::new(other.func, local))
            .collect();
        if !other_params.is_empty() {
            assert_eq!(
                other_params.len(),
                branch_args.len(),
                "cannot absorb block with {} params through branch with {} args",
                other_params.len(),
                branch_args.len()
            );
            for (param, arg) in other_params.iter().copied().zip(branch_args) {
                self.replace_all_uses_with(ValueId::BlockParam(param), arg.qualify(keep.func));
            }
        }
        self.remove_cfg_edge(edge_ab);
        self.remove_instruction(branch_id);
        for local in self.take_insns(other.local) {
            self.link_last(keep.local, local);
        }
        self.rehome_outgoing_edges(keep, other);
        let (b_addr, b_extra, b_name) = {
            let b = self.block(other);
            (
                b.address,
                b.extra_addresses.clone(),
                b.local_name().map(str::to_owned),
            )
        };
        for param in other_params {
            self.remove_block_param(param);
        }
        self.unroster_block(other);
        if self.root == Some(other.local) {
            self.root = Some(keep.local);
        }
        if let Some(name) = b_name {
            self.names.forget(&name);
        }
        self.blocks.remove(other.local);
        // The addresses `other` carried move to `keep`: an index naming
        // `other` for them is behind now, and one that never listed `other`
        // has nothing to catch up on.
        if b_addr.is_some() || !b_extra.is_empty() {
            self.touch_shape();
        }
        if let Some(addr) = b_addr {
            self.block_mut(keep).extra_addresses.push(addr);
        }
        self.block_mut(keep).extra_addresses.extend(b_extra);
    }

    /// Register `name` for `id` in this body's local name table (block/instruction/
    /// param). A global-scoped `id` reads `shared` for the duplicate check but
    /// cannot be *registered* through a body (its shared table is read-only here);
    /// no body verb reaches that arm.
    pub fn register_local_name(
        &mut self,
        shared: &crate::context::Shared<'str>,
        id: ValueId,
        name: Cow<'str, str>,
        old_name: Option<&str>,
    ) -> Result<()> {
        if id.name_scope_function().is_none() {
            return match shared.get_named(&name) {
                Some(existing) if existing == id => Ok(()),
                Some(_) => Err(Error::spanless(ErrorTy::DuplicateName(name.to_string()))),
                None => unimplemented!(
                    "a function body cannot register a global name (shared is read-only)"
                ),
            };
        }
        self.register_body_name(id, name, old_name)
    }

    /// Register `name` for the function-scoped `id` (block/instruction/param/Temp)
    /// in this body's local name table. The shared-arm-free canon behind
    /// [`register_local_name`](Self::register_local_name); panics on a
    /// global-scoped `id`. Errors only on a duplicate name.
    pub fn register_body_name(
        &mut self,
        id: ValueId,
        name: Cow<'str, str>,
        old_name: Option<&str>,
    ) -> Result<()> {
        assert!(
            id.name_scope_function().is_some(),
            "register_body_name on a global-scoped value {id:?}"
        );
        if let Some(existing) = self.names.get(&name).map(|id| id.qualify(self.id())) {
            return if existing == id {
                Ok(())
            } else {
                Err(Error::spanless(ErrorTy::DuplicateName(name.to_string())))
            };
        }
        self.names.register(name, id.localize(self.id()), old_name)
    }

    /// Gets a reference to a function from its ID
    pub fn from_id<'ctx>(ctx: &'ctx Context<'str>, id: FunctionId) -> FunctionRef<'str, 'ctx> {
        FunctionRef::new(ModuleView::new(ctx), id)
    }

    /// Gets a mutable reference to a function from its ID
    pub fn from_id_mut<'ctx>(
        ctx: &'ctx mut Context<'str>,
        id: FunctionId,
    ) -> FunctionMutRef<'str, 'ctx> {
        FunctionMutRef::new(ctx, id)
    }

    /// Gets a reference to a function by name
    pub fn from_name<'ctx>(
        ctx: &'ctx Context<'str>,
        name: &str,
    ) -> Option<FunctionRef<'str, 'ctx>> {
        ctx.get_named(name)
            .and_then(ValueId::as_function)
            .map(|id| FunctionBody::from_id(ctx, id))
    }

    /// Create a new function
    pub fn make<'ctx>(
        ctx: &'ctx mut Context<'str>,
        name: Cow<'str, str>,
    ) -> Result<FunctionMutRef<'str, 'ctx>> {
        let id = FunctionId::from(ctx.bodies.len());
        let pushed = ctx.push_function(
            FunctionInterface::new(name.clone()),
            FunctionBody::empty_with_id(id),
        );
        debug_assert_eq!(pushed, id);
        ctx.update_name(name, id.into(), None)?;
        Ok(Self::from_id_mut(ctx, id))
    }

    /// Create a new pure value-level lambda function.
    pub fn make_lambda<'ctx>(
        ctx: &'ctx mut Context<'str>,
        name: Cow<'str, str>,
    ) -> Result<FunctionMutRef<'str, 'ctx>> {
        let mut function = Self::make(ctx, name)?;
        function.interface_mut().kind = FunctionKind::Lambda;
        function.set_is_pure(true);
        function.set_register_effects(RegisterChannelState::Materialized(
            RegisterInterfaceMap::default(),
        ));
        Ok(function)
    }

    /// Create a new function at a given address, generating a name if necessary.
    pub fn make_at_addr<'ctx>(
        ctx: &'ctx mut Context<'str>,
        address: u64,
        name: Option<Cow<'str, str>>,
    ) -> FunctionMutRef<'str, 'ctx> {
        let mut addresses = crate::address_index::AddressIndex::analyze(ctx);
        Self::make_at_addr_indexed(ctx, &mut addresses, address, name)
    }

    /// Indexed construction variant of [`make_at_addr`](Self::make_at_addr).
    pub fn make_at_addr_indexed<'ctx>(
        ctx: &'ctx mut Context<'str>,
        addresses: &mut crate::address_index::AddressIndex,
        address: u64,
        name: Option<Cow<'str, str>>,
    ) -> FunctionMutRef<'str, 'ctx> {
        // Symbol names are not unique in a linked binary (two static functions
        // of the same name from different translation units); take the first
        // free `name_<n>` rather than refusing the function.
        let name = name.unwrap_or_else(|| Cow::Owned(format!("fn_{address:x}")));
        let name = ctx.shared.name_map.unique(name);
        let id = FunctionId::from(ctx.bodies.len());
        // A current index stays current: the function and its address both
        // go into it here.
        let current = addresses.is_current(ctx);
        let pushed = ctx.push_function(
            FunctionInterface::new(name.clone()),
            FunctionBody::empty_with_id(id),
        );
        debug_assert_eq!(pushed, id);

        let function = Self::from_id_mut(ctx, id)
            .with_name(name)
            .expect("Function name is not unique")
            .with_address_indexed(addresses, address)
            .expect("Function address is not unique");
        if current {
            addresses.mark_current(function.ctx);
        }
        function
    }

    /// Like [`FunctionBody::make_at_addr`] but marks the result as external.
    ///
    /// External functions have no lifted body; the recursive disassembler will
    /// not try to explore them.
    pub fn make_external<'ctx>(
        ctx: &'ctx mut Context<'str>,
        address: u64,
        name: Option<Cow<'str, str>>,
    ) -> FunctionMutRef<'str, 'ctx> {
        let mut f = Self::make_at_addr(ctx, address, name);
        f.interface_mut().is_external = true;
        f
    }

    /// Indexed construction variant of [`make_external`](Self::make_external).
    pub fn make_external_indexed<'ctx>(
        ctx: &'ctx mut Context<'str>,
        addresses: &mut crate::address_index::AddressIndex,
        address: u64,
        name: Option<Cow<'str, str>>,
    ) -> FunctionMutRef<'str, 'ctx> {
        let mut function = Self::make_at_addr_indexed(ctx, addresses, address, name);
        function.interface_mut().is_external = true;
        function
    }

    /// Returns the [`FunctionId`] for `addr`, creating a named stub if absent.
    pub fn from_addr_or_create<'ctx>(
        ctx: &'ctx mut Context<'str>,
        address: u64,
    ) -> FunctionMutRef<'str, 'ctx> {
        let mut addresses = crate::address_index::AddressIndex::analyze(ctx);
        Self::from_addr_or_create_indexed(ctx, &mut addresses, address)
    }

    /// Indexed construction variant of
    /// [`from_addr_or_create`](Self::from_addr_or_create).
    pub fn from_addr_or_create_indexed<'ctx>(
        ctx: &'ctx mut Context<'str>,
        addresses: &mut crate::address_index::AddressIndex,
        address: u64,
    ) -> FunctionMutRef<'str, 'ctx> {
        match addresses.function_at(address) {
            Some(id) => Self::from_id_mut(ctx, id),
            None => Self::make_at_addr_indexed(ctx, addresses, address, None),
        }
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx, R> FunctionRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx FunctionBody<'str> {
        self.view.function(self.id)
    }

    /// This function's published interface (never checked out; always read from
    /// the shared registry).
    fn interface(&'s self) -> &'ctx FunctionInterface<'str> {
        self.view.interface(self.id)
    }

    fn size(&self) -> usize {
        0
    }

    /// The function interface's entry address.
    pub fn address(&'s self) -> Option<u64> {
        self.interface().address
    }

    /// Whether the function interface marks this function external.
    pub fn is_external(&'s self) -> bool {
        self.interface().is_external
    }

    /// The ordinal this import was brought in at, for a PE import resolved from
    /// an ordinal-only entry. `None` for named imports and local functions.
    pub fn import_ordinal(&'s self) -> Option<u16> {
        self.interface().import_ordinal
    }

    /// A reference to the function interface's signature, if any.
    pub fn signature(&'s self) -> Option<&'ctx FunctionSignature> {
        self.interface().signature.as_ref()
    }

    /// This function's instructions that use `value` as an operand. See
    /// [`FunctionBody::users_of`]; this is the function-scoped read every pass wants
    /// for an SSA value (all its users are intra-function).
    pub fn users_of(&'s self, value: ValueId) -> Vec<InstructionId> {
        let func = self.id;
        if value.owning_function().is_some_and(|owner| owner != func) {
            return Vec::new();
        }
        self.inner().users_of(value)
    }

    /// This function's users of `value`, as body-local ids, in no particular
    /// order; an instruction using `value` twice comes twice.
    ///
    /// Walked rather than built: a pass that reads the list once per
    /// instruction should not allocate one per instruction to do it. Qualify
    /// with this function's id when a whole [`InstructionId`] is needed.
    pub fn local_users_of(&'s self, value: ValueId) -> Users<'ctx, 'str> {
        self.inner().local_users_of(value)
    }

    /// Whether this function uses `value` at all, without building the user
    /// list to ask. See [`FunctionBody::has_users`].
    pub fn has_users(&'s self, value: ValueId) -> bool {
        let func = self.id;
        if value.owning_function().is_some_and(|owner| owner != func) {
            return false;
        }
        self.inner().has_users(value)
    }

    /// Resolve a block/instruction/param/Temp `name` within this function's local name
    /// table (see `FunctionBody::names`). `None` if this function has no such name.
    pub fn local_named(&'s self, name: &str) -> Option<ValueId> {
        self.inner().names.get(name).map(|id| id.qualify(self.id))
    }

    /// The inferred pointer attributes for positional argument `index`, or `None`
    /// when this function has no analyzed attributes (treat conservatively: the
    /// argument escapes and may be written through). See
    /// [`FunctionSignature::param_attrs`].
    pub fn param_attr(&'s self, index: usize) -> Option<ParamAttrs> {
        self.interface().param_attr(index)
    }

    /// The full per-parameter attribute vector, if analyzed.
    pub fn param_attrs(&'s self) -> Option<&'ctx [ParamAttrs]> {
        self.interface()
            .signature
            .as_ref()
            .and_then(|s| s.param_attrs.as_deref())
    }

    /// The non-register memory spaces this function may (transitively) write, as
    /// set by analysis. `Some(spaces)` is exact (a space not listed is never
    /// written); `None` conflates "unstamped" and "stamped unbounded" — both are
    /// treated conservatively (may write any space) by consumers. For the
    /// tri-state distinction use [`written_spaces_state`](Self::written_spaces_state).
    /// See `FunctionSignature::written_spaces`.
    pub fn written_spaces(&'s self) -> Option<&'ctx [crate::space::SpaceId]> {
        match &self.interface().effects.memory.coarse {
            WrittenSpacesState::Bounded(spaces) => Some(spaces),
            _ => None,
        }
    }

    /// The tri-state `written_spaces` verdict, distinguishing a never-stamped
    /// fresh mint ([`WrittenSpaces::Unstamped`]) from a deliberately recorded
    /// ⊤ ([`WrittenSpaces::Unbounded`]). See `FunctionSignature::written_spaces`.
    pub fn written_spaces_state(&'s self) -> WrittenSpaces<'ctx> {
        match &self.interface().effects.memory.coarse {
            WrittenSpacesState::Unstamped => WrittenSpaces::Unstamped,
            WrittenSpacesState::Unbounded => WrittenSpaces::Unbounded,
            WrittenSpacesState::Bounded(spaces) => WrittenSpaces::Bounded(spaces),
        }
    }

    /// Whether this function's register interface has been materialized (argpromote
    /// v2) — i.e. its [`effects`](FunctionInterface::effects) are
    /// [`RegisterChannelState::Materialized`]. Legacy name for the register-channel
    /// "functionalized" predicate.
    pub fn is_reg_materialized(&'s self) -> bool {
        matches!(
            self.interface().effects.register,
            RegisterChannelState::Materialized(_)
        )
    }

    /// This function's call-graph-closed register [`FunctionEffects`] summary.
    /// [`RegisterChannelState::Unsolved`] until the effect-analysis pass runs (and
    /// after a snapshot load). See [`FunctionInterface::effects`].
    pub fn effects(&'s self) -> &'ctx FunctionEffects {
        &self.interface().effects
    }

    /// Whether argpromote has functionalized *every* side-effect channel of this
    /// function — it is a deterministic pure function of its by-value params,
    /// touching no caller-visible memory or registers. Strictly stronger than
    /// [`is_reg_materialized`](Self::is_reg_materialized). See [`FunctionSignature::is_pure`].
    pub fn is_pure(&'s self) -> bool {
        self.interface()
            .signature
            .as_ref()
            .is_some_and(|s| s.is_pure)
    }

    /// Whether this is a pure value-level lambda rather than a machine function.
    pub fn is_lambda(&'s self) -> bool {
        self.interface().kind == FunctionKind::Lambda
    }

    pub fn kind(&'s self) -> FunctionKind {
        self.interface().kind
    }

    /// The C-prototype-derived external call interface, if `external_sigs`
    /// planned one. Read by `argpromote_external` to rewrite call sites. See
    /// [`FunctionSignature::extern_interface`].
    pub fn extern_interface(&'s self) -> Option<&'ctx crate::value::ExternInterface> {
        self.interface()
            .signature
            .as_ref()
            .and_then(|s| s.extern_interface.as_ref())
    }

    /// The C-prototype-derived argmem summary for a prototyped external, or `None`
    /// when this function is not a prototyped external. See
    /// [`FunctionSignature::argmem`].
    pub fn argmem(&'s self) -> Option<&'ctx crate::value::ExternArgmem> {
        self.interface()
            .signature
            .as_ref()
            .and_then(|s| s.argmem.as_ref())
    }

    /// The display name for the call-site argument bound to input `index`: the
    /// name of the callee's root block param at `index`, or — for a bodyless
    /// external with no root block — the C-prototype argument name recorded in
    /// its [`extern_interface`](Self::extern_interface). `None` when there is no
    /// input at `index` or it is unnamed.
    pub fn input_arg_name(&'s self, index: usize) -> Option<String> {
        // The root block param at `index` is the interface element a call
        // argument actually binds to, named after its register by
        // `argpromote_registers` or `stack_<addr>` by mem2reg's
        // `block_param_name_for_var`. Prefer it: it is the source of truth and is
        // populated even for `pure_reg` functions.
        if let Some(root) = self.root()
            && let Some(name) = root
                .params()
                .nth(index)
                .and_then(|p| p.name().map(str::to_owned))
        {
            return Some(name);
        }

        // Fall back to the C-prototype-derived external call interface, the
        // source of truth for bodyless externals which have no root block.
        self.extern_interface()
            .and_then(|iface| iface.args.get(index))
            .and_then(|a| a.name.as_ref().map(|n| n.to_string()))
    }

    /// Whether this function performs an unresolved/dynamic stack read (or
    /// forwards a stack pointer into one). See
    /// [`FunctionSignature::reads_unbounded_stack`].
    pub fn reads_unbounded_stack(&'s self) -> bool {
        self.interface()
            .signature
            .as_ref()
            .is_some_and(|s| s.reads_unbounded_stack)
    }

    /// Whether this function hands a pointer into its own frame to a callee that
    /// may read it unboundedly. See
    /// [`FunctionSignature::frame_escapes_to_unbounded`].
    pub fn frame_escapes_to_unbounded(&'s self) -> bool {
        self.interface()
            .signature
            .as_ref()
            .is_some_and(|s| s.frame_escapes_to_unbounded)
    }

    /// The function interface's name.
    pub fn name(&'s self) -> &'ctx str {
        self.interface().name.as_ref()
    }

    /// The addresses of every machine instruction lifted into this function, in
    /// ascending order. Unlike [`blocks`](Self::blocks), this is stable across
    /// optimization, so it drives the raw disassembly view.
    pub fn instruction_addrs(&'s self) -> impl Iterator<Item = u64> + 'ctx {
        self.inner().instruction_addrs.iter().copied()
    }

    /// Whether this function contains at least one [`Map`](Mnemonic::Map)
    /// instruction — a lane-wise array map operation. Surfaced as an advanced
    /// filter in the function list.
    pub fn has_map(&'s self) -> bool {
        self.blocks().any(|block| {
            block
                .instructions()
                .any(|insn| matches!(insn.mnemonic(), Mnemonic::Map(_)))
        })
    }

    /// Whether this function contains at least one [`Scan`](Mnemonic::Scan)
    /// instruction — a lane-wise prefix-fold array operation. Surfaced as an
    /// advanced filter in the function list, alongside [`has_map`](Self::has_map).
    pub fn has_scan(&'s self) -> bool {
        self.blocks().any(|block| {
            block
                .instructions()
                .any(|insn| matches!(insn.mnemonic(), Mnemonic::Scan(_)))
        })
    }

    /// The root block of this function, if it exists.
    pub fn root(&'s self) -> Option<BlockRef<'str, 'ctx, R>> {
        self.inner()
            .root
            .map(|local| BlockRef::new(self.view, BlockId::new(self.id, local)))
    }

    /// An iterator over the (live) blocks belonging to this function.
    pub fn blocks(&'s self) -> impl Iterator<Item = BlockRef<'str, 'ctx, R>> + 's {
        let view = self.view;
        let mut ids = self.block_ids();
        // Total order: primarily by machine address, but break ties by the
        // function-local index. Address-less blocks (e.g. fallthrough splits,
        // whose `address()` is `None`) must still order deterministically.
        ids.sort_by_key(|&id| (BlockRef::new(view, id).address(), id.local));
        ids.into_iter().map(move |id| BlockRef::new(view, id))
    }

    /// The composite ids of this function's live blocks, in roster order.
    pub fn block_ids(&'s self) -> Vec<BlockId> {
        let func = self.id;
        self.inner()
            .roster
            .iter()
            .copied()
            .map(|local| BlockId::new(func, local))
            .collect()
    }

    /// The composite IDs of this function's live instructions, in dense physical
    /// order — including any currently detached (`parent == None`).
    pub fn instruction_ids(&'s self) -> Vec<InstructionId> {
        let func = self.id;
        self.inner()
            .insns
            .iter()
            .map(|i| InstructionId::new(func, i.id))
            .collect()
    }

    /// The IDs of every live CFG edge in this function's edge arena, in dense
    /// physical order.
    pub fn edge_ids(&'s self) -> Vec<crate::value::block::EdgeId> {
        self.inner().edges.iter().map(|e| e.id).collect()
    }

    /// Iterates over the (live) blocks in this function in arena order (i.e. not
    /// sorted by address, unlike [`blocks`](Self::blocks)).
    pub fn iter(&'s self) -> BlockIter<'str, 'ctx, R> {
        BlockIter {
            view: self.view,
            inner: self.block_ids().into_iter(),
            marker: PhantomData,
        }
    }

    fn fmt(&'s self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_external() {
            return writeln!(f, "extern fn {};", self.name());
        }
        let keyword = match self.kind() {
            FunctionKind::Machine => "fn",
            FunctionKind::Lambda => "lambda",
        };
        writeln!(f, "{keyword} {}:", self.name())?;
        for block in self.blocks() {
            block.fmt(f)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct FunctionRef<'str, 'ctx, R = ModuleView<'ctx, 'str>> {
    pub id: FunctionId,
    pub(in crate::value) view: R,
    marker: PhantomData<&'ctx &'str ()>,
}

impl<'str, 'ctx, R> FunctionRef<'str, 'ctx, R> {
    pub fn new(view: R, id: FunctionId) -> Self {
        Self {
            id,
            view,
            marker: PhantomData,
        }
    }

    pub fn id(&self) -> ValueId {
        self.id.into()
    }
}

impl<'str, 'ctx> FunctionRef<'str, 'ctx> {
    pub fn from_id(ctx: &'ctx Context<'str>, id: FunctionId) -> Self {
        Self::new(ModuleView::new(ctx), id)
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 'ctx, 'str> for FunctionRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'ctx Context<'str> {
        // Module-scope-only escape hatch: shared-only reads go through
        // `host().shr()`; only whole-module walks (callees/callers) reach here,
        // and those panic on a checked-out host by design (context-split Pin B).
        self.view.context()
    }
}

impl<'str: 'ctx, 'ctx, R> Named for FunctionRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn name(&self) -> Option<&str> {
        Some(self.view.interface(self.id).name.as_ref())
    }
}

impl<'str: 'ctx, 'ctx, R> Display for FunctionRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        FunctionRef::fmt(self, f)
    }
}

impl<'str: 'ctx, 'ctx, R> Value<'str, 'ctx> for FunctionRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        FunctionRef::size(self)
    }
}

pub struct BlockIter<'str, 'ctx, R = ModuleView<'ctx, 'str>> {
    view: R,
    inner: std::vec::IntoIter<BlockId>,
    marker: PhantomData<&'ctx &'str ()>,
}

impl<'str: 'ctx, 'ctx, R> Iterator for BlockIter<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    type Item = BlockRef<'str, 'ctx, R>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|id| BlockRef::new(self.view, id))
    }
}

impl<'str: 'ctx, 'ctx, R> IntoIterator for &FunctionRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    type Item = BlockRef<'str, 'ctx, R>;
    type IntoIter = BlockIter<'str, 'ctx, R>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub type FunctionMutRef<'str, 'ctx> = BaseRef<&'ctx mut Context<'str>, FunctionId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 's, 'str> for FunctionMutRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'s Context<'str> {
        self.ctx
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtxMut<'s, 'str> for FunctionMutRef<'str, 'ctx> {
    fn ctx_mut(&'s mut self) -> &'s mut Context<'str> {
        self.ctx
    }
}

impl Display for FunctionMutRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<'ctx, 'str> Value<'str, 'ctx> for FunctionMutRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        self.as_ref().size()
    }
}

impl Named for FunctionMutRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        Some(self.ctx.interfaces[self.id].name.as_ref())
    }
}

impl<'str, 'ctx> Renameable<'str, 'ctx> for FunctionMutRef<'str, 'ctx> {
    fn rename(&mut self, name: Cow<'str, str>) -> Result<()> {
        let id = self.id();
        let old_name = self.ctx.interfaces[self.id].name.as_ref().to_owned();
        update_context_name(id, self.ctx, name.clone(), Some(old_name.as_ref()))?;
        self.ctx.interfaces[self.id].name = name;
        Ok(())
    }
}

impl<'str, 'ctx> FunctionMutRef<'str, 'ctx> {
    pub fn as_ref(&self) -> FunctionRef<'str, '_> {
        FunctionRef::new(ModuleView::new(self.ctx), self.id)
    }

    fn inner(&self) -> &FunctionBody<'str> {
        self.ctx.function(self.id)
    }

    fn interface(&self) -> &FunctionInterface<'str> {
        &self.ctx.interfaces[self.id]
    }

    fn address(&self) -> Option<u64> {
        self.interface().address
    }

    pub fn name(&self) -> &str {
        self.interface().name.as_ref()
    }

    pub fn blocks(&self) -> impl Iterator<Item = BlockRef<'str, '_>> {
        self.as_ref().blocks().collect::<Vec<_>>().into_iter()
    }

    pub fn root(&self) -> Option<BlockRef<'str, '_>> {
        self.as_ref().root()
    }

    pub(crate) fn inner_mut(&mut self) -> &mut FunctionBody<'str> {
        &mut self.ctx.bodies[self.id]
    }

    /// This function's published interface (mutable). Interface writes are
    /// module-scope only; this is the write path for the setters below.
    pub(crate) fn interface_mut(&mut self) -> &mut FunctionInterface<'str> {
        &mut self.ctx.interfaces[self.id]
    }

    fn set_address(&mut self, address: u64) -> Result<()> {
        let mut addresses = crate::address_index::AddressIndex::analyze(&*self.ctx);
        self.set_address_indexed(&mut addresses, address)
    }

    fn set_address_indexed(
        &mut self,
        addresses: &mut crate::address_index::AddressIndex,
        address: u64,
    ) -> Result<()> {
        let current = addresses.is_current(self.ctx);
        let old_address = self.interface().address;
        self.interface_mut().address = Some(address);
        self.ctx.touch_shape();
        let result = self
            .ctx
            .set_address_indexed(addresses, address, self.id.into());
        if result.is_err() {
            self.interface_mut().address = old_address;
        }
        // Registered or restored, the index reflects the interface either way.
        if current {
            addresses.mark_current(self.ctx);
        }
        result
    }

    fn with_address_indexed(
        mut self,
        addresses: &mut crate::address_index::AddressIndex,
        address: u64,
    ) -> Result<Self> {
        self.set_address_indexed(addresses, address)?;
        Ok(self)
    }

    /// Sets a block as the root of this function.
    /// This will also add the block to the function's block list if it's not already present.
    /// This will also set the address of the function/block to the address of the root block/function if both addresses are unset.
    /// Panics if the function already has an address that doesn't match the root block's address.
    pub fn set_root(&mut self, id: BlockId) -> Result<()> {
        assert_eq!(
            id.func, self.id,
            "cannot root a function at a block stored in another function arena"
        );
        self.add_block(id);
        self.inner_mut().root = Some(id.localize(self.id));

        let block_addr = BasicBlock::from_id(&*self.ctx, id).address();
        let self_addr = self.address();

        match (self_addr, block_addr) {
            (Some(fn_addr), Some(block_addr)) if fn_addr != block_addr => {
                return Err(Error::spanless(ErrorTy::FunctionRootAddressMismatch {
                    fn_addr,
                    block_addr,
                }));
            }
            (None, Some(addr)) => {
                self.set_address(addr)
                    .expect("This address should be valid");
            }
            (Some(addr), None) => {
                BasicBlock::from_id_mut(self.ctx, id)
                    .set_address(addr)
                    .expect("This address should be valid");
            }
            _ => {}
        }
        Ok(())
    }

    pub fn make_root(&mut self) -> BlockRef<'str, '_> {
        let func = self.id;
        let root = BasicBlock::make(self.ctx, func).id;
        self.set_root(root).expect("We just created the block");
        BasicBlock::from_id(&*self.ctx, root)
    }

    pub fn ensure_root(&mut self, id: BlockId) -> Result<()> {
        assert_eq!(
            id.func, self.id,
            "cannot ensure a function root from another function arena"
        );
        if let Some(root) = self.inner().root {
            if root != id.localize(self.id) {
                return Err(Error::spanless(ErrorTy::FunctionRootMismatch {
                    expected: BlockId::new(self.id, root),
                    actual: id,
                }));
            }
            Ok(())
        } else {
            self.set_root(id)
        }
    }

    pub fn set_external(&mut self, is_external: bool) {
        self.interface_mut().is_external = is_external;
        assert!(
            self.inner().blocks.is_empty(),
            "External functions should not have blocks"
        );
    }

    /// Record the ordinal a by-ordinal PE import was brought in at. Set by the
    /// `resolve_ordinals` pass before it renames the stub, so the by-ordinal
    /// origin survives the rename.
    pub fn set_import_ordinal(&mut self, ordinal: Option<u16>) {
        self.interface_mut().import_ordinal = ordinal;
    }

    pub fn set_kind(&mut self, kind: FunctionKind) {
        self.interface_mut().kind = kind;
        if kind == FunctionKind::Lambda {
            self.set_is_pure(true);
            self.set_register_effects(RegisterChannelState::Materialized(
                RegisterInterfaceMap::default(),
            ));
        }
    }

    pub fn set_signature(&mut self, sig: FunctionSignature) {
        self.ctx.interfaces[self.id].signature = Some(sig);
    }

    /// Records the inferred per-parameter pointer attributes on this function.
    /// See [`FunctionSignature::param_attrs`].
    pub fn set_param_attrs(&mut self, attrs: Vec<ParamAttrs>) {
        self.interface_mut()
            .signature
            .get_or_insert_default()
            .param_attrs = Some(attrs);
    }

    /// Drops any inferred per-parameter attributes (e.g. after a signature
    /// rewrite changed the parameter list, invalidating the index alignment).
    pub fn clear_param_attrs(&mut self) {
        if let Some(sig) = self.interface_mut().signature.as_mut() {
            sig.param_attrs = None;
        }
    }

    /// Records the analysis-computed set of non-register spaces this function may
    /// write. This is always a deliberate stamp: `Some(spaces)` is a bounded
    /// witnessed set, `None` records *stamped unbounded* (⊤) — never clears the
    /// stamp back to unstamped. See `FunctionSignature::written_spaces` and
    /// [`FunctionRef::written_spaces_state`].
    pub fn set_written_spaces(&mut self, spaces: Option<Vec<crate::space::SpaceId>>) {
        let coarse = match spaces {
            Some(spaces) => WrittenSpacesState::Bounded(spaces),
            None => WrittenSpacesState::Unbounded,
        };
        // Coarse-only setter: every other component of the channel is left
        // exactly as it was.
        let precise = self.interface_mut().effects.memory.precise.take();
        self.set_memory_solved(coarse, precise);
    }

    /// Records the C-prototype-derived external call interface on this function.
    /// See [`FunctionSignature::extern_interface`]; set by `external_sigs`,
    /// consumed by `argpromote_external`.
    pub fn set_extern_interface(&mut self, iface: crate::value::ExternInterface) {
        self.interface_mut()
            .signature
            .get_or_insert_default()
            .extern_interface = Some(iface);
    }

    /// Records the C-prototype-derived argmem summary on this external. See
    /// [`FunctionSignature::argmem`]; set by `external_sigs`, read by the RAM
    /// effect channel's `external_leaf`.
    pub fn set_argmem(&mut self, argmem: crate::value::ExternArgmem) {
        self.interface_mut()
            .signature
            .get_or_insert_default()
            .argmem = Some(argmem);
    }

    /// Records this function's register-channel effect state, preserving the
    /// memory channel (read-modify-write). See [`FunctionInterface::effects`].
    pub fn set_register_effects(&mut self, register: RegisterChannelState) {
        self.interface_mut().effects.register = register;
    }

    /// Records this function's memory-channel effect state, preserving the
    /// register channel (read-modify-write). See [`FunctionInterface::effects`].
    ///
    /// Replaces **every** component of the memory channel. The channel has two
    /// independent writers — the effect solve owns `coarse`/`precise`, the RAM
    /// channel's rewrite owns `materialized` — so a caller that computes only
    /// one writer's components must not build a whole state and pass it here:
    /// the other writer's field would be silently lost. Use
    /// [`set_memory_solved`](Self::set_memory_solved) or
    /// [`set_memory_interface`](Self::set_memory_interface) instead.
    pub fn set_memory_effects(&mut self, memory: MemoryChannelState) {
        self.interface_mut().effects.memory = memory;
    }

    /// Records the *solved* components of the memory channel — the coarse
    /// written-space verdict and the precise footprint the same solve derived —
    /// leaving the materialized interface untouched.
    ///
    /// The effect solve does not compute the interface, so it must not clear it.
    pub fn set_memory_solved(&mut self, coarse: WrittenSpacesState, precise: Option<Footprint>) {
        let memory = &mut self.interface_mut().effects.memory;
        memory.coarse = coarse;
        memory.precise = precise;
    }

    /// Records the materialized memory interface, leaving the solved components
    /// untouched. `None` marks the memory channel as not materialized.
    pub fn set_memory_interface(&mut self, materialized: Option<MemoryInterfaceMap>) {
        self.interface_mut().effects.memory.materialized = materialized;
    }

    /// Marks this function as fully functionalized over *every* side-effect
    /// channel — a deterministic pure function of its params. See
    /// [`FunctionSignature::is_pure`].
    pub fn set_is_pure(&mut self, value: bool) {
        self.interface_mut()
            .signature
            .get_or_insert_default()
            .is_pure = value;
    }

    /// Records whether this function performs an unresolved/dynamic stack read.
    /// See [`FunctionSignature::reads_unbounded_stack`].
    pub fn set_reads_unbounded_stack(&mut self, value: bool) {
        self.interface_mut()
            .signature
            .get_or_insert_default()
            .reads_unbounded_stack = value;
    }

    /// Records whether this function hands a pointer into its own frame to a
    /// callee that may read it unboundedly. See
    /// [`FunctionSignature::frame_escapes_to_unbounded`].
    pub fn set_frame_escapes_to_unbounded(&mut self, value: bool) {
        self.interface_mut()
            .signature
            .get_or_insert_default()
            .frame_escapes_to_unbounded = value;
    }

    /// Records the address of a machine instruction lifted into this function.
    pub fn add_instruction_addr(&mut self, addr: u64) {
        self.inner_mut().instruction_addrs.insert(addr);
    }

    /// Associates `block` with `function` by setting the block's `parent` field.
    ///
    /// With per-function block arenas, membership *is* arena ownership: a block
    /// lives in the arena of the function it was born into (`id.func`), and that
    /// must equal `self.id`. Ownership is derived from the arena, so this only
    /// ensures the roster lists the block; it no longer moves storage between
    /// functions.
    pub fn add_block(&mut self, id: BlockId) {
        assert_eq!(
            id.func, self.id,
            "cannot add a block stored in another function arena"
        );
        let local = id.localize(self.id);
        // Ensure the roster lists it exactly once (a freshly `make`d block is
        // auto-rostered, so this is usually a no-op).
        if !self.inner().roster.contains(&local) {
            self.inner_mut().roster.push(local);
        }
    }
}

#[cfg(test)]
mod tests {
    use wazabin_qcode_macro::qcode;

    use super::*;

    fn foreign_block_fixture() -> (Context<'static>, FunctionId, BlockId) {
        let mut ctx = Context::new();
        let owner = FunctionBody::make(&mut ctx, "block_owner".into())
            .unwrap()
            .id;
        let destination = FunctionBody::make(&mut ctx, "block_destination".into())
            .unwrap()
            .id;
        let block = BasicBlock::make(&mut ctx, owner).id;
        (ctx, destination, block)
    }

    #[test]
    fn raw_root_and_roster_are_local_while_refs_qualify_per_function() {
        let mut ctx = Context::new();
        let a = FunctionBody::make(&mut ctx, "local_root_a".into())
            .unwrap()
            .id;
        let b = FunctionBody::make(&mut ctx, "local_root_b".into())
            .unwrap()
            .id;
        let a_root = BasicBlock::make(&mut ctx, a).id;
        let b_root = BasicBlock::make(&mut ctx, b).id;
        assert_eq!(a_root.local, b_root.local, "arena-local ids should collide");
        FunctionBody::from_id_mut(&mut ctx, a)
            .set_root(a_root)
            .unwrap();
        FunctionBody::from_id_mut(&mut ctx, b)
            .set_root(b_root)
            .unwrap();

        assert_eq!(ctx.bodies[a].root_id(), Some(a_root.local));
        assert_eq!(ctx.bodies[b].root_id(), Some(b_root.local));
        assert_eq!(ctx.bodies[a].roster, vec![a_root.local]);
        assert_eq!(ctx.bodies[b].roster, vec![b_root.local]);
        assert_eq!(
            FunctionBody::from_id(&ctx, a).root().map(|root| root.id),
            Some(a_root)
        );
        assert_eq!(
            FunctionBody::from_id(&ctx, b).root().map(|root| root.id),
            Some(b_root)
        );
    }

    #[test]
    #[should_panic(expected = "cannot add a block stored in another function arena")]
    fn add_block_rejects_foreign_storage() {
        let (mut ctx, destination, block) = foreign_block_fixture();
        FunctionBody::from_id_mut(&mut ctx, destination).add_block(block);
    }

    #[test]
    #[should_panic(expected = "cannot root a function at a block stored in another function arena")]
    fn set_root_rejects_foreign_storage() {
        let (mut ctx, destination, block) = foreign_block_fixture();
        FunctionBody::from_id_mut(&mut ctx, destination)
            .set_root(block)
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "cannot ensure a function root from another function arena")]
    fn ensure_root_rejects_foreign_storage() {
        let (mut ctx, destination, block) = foreign_block_fixture();
        FunctionBody::from_id_mut(&mut ctx, destination)
            .ensure_root(block)
            .unwrap();
    }

    #[test]
    fn function_ref_users_of_rejects_foreign_owned_values() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn users_a:
                <a_entry>
                    %a_def = i64 1 + i64 2;
                    %a_user = %a_def + i64 3;
                    return at %a_user;

            fn users_b:
                <b_entry>
                    %b_def = i64 1 + i64 2;
                    %b_user = %b_def + i64 3;
                    return at %b_user;
            "
        );

        let a_ids = FunctionRef::from_id(&ctx, users_a)
            .root()
            .unwrap()
            .iter_instruction_ids()
            .collect::<Vec<_>>();
        let a_def = ValueId::Instruction(a_ids[0]);
        assert_eq!(
            FunctionRef::from_id(&ctx, users_a).users_of(a_def),
            vec![a_ids[1]]
        );
        assert!(
            FunctionRef::from_id(&ctx, users_b)
                .users_of(a_def)
                .is_empty()
        );

        let one = ctx.get_const(1, 8).id();
        assert!(!FunctionRef::from_id(&ctx, users_b).users_of(one).is_empty());
    }

    fn colliding_body_ids() -> (
        Context<'static>,
        FunctionId,
        FunctionId,
        BlockId,
        BlockId,
        InstructionId,
        InstructionId,
        BlockParamId,
        BlockParamId,
    ) {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn raw_a:
                <a_entry @a:i64>
                    %a_def = i64 1 + i64 2;
                    return at %a_def;
            fn raw_b:
                <b_entry @b:i64>
                    %b_def = i64 1 + i64 2;
                    return at %b_def;
            "
        );
        let a_root = FunctionRef::from_id(&ctx, raw_a).root().unwrap();
        let b_root = FunctionRef::from_id(&ctx, raw_b).root().unwrap();
        let a_block = a_root.id;
        let b_block = b_root.id;
        let a_insn = a_root.first_instruction().unwrap();
        let b_insn = b_root.first_instruction().unwrap();
        let a_param = a_root.params().next().unwrap().id;
        let b_param = b_root.params().next().unwrap().id;
        assert_eq!(a_block.local, b_block.local);
        assert_eq!(a_insn.local, b_insn.local);
        assert_eq!(a_param.local, b_param.local);
        (
            ctx, raw_a, raw_b, a_block, b_block, a_insn, b_insn, a_param, b_param,
        )
    }

    /// `replace_instruction(id, id)` must be a no-op: forwarding uses to itself
    /// does nothing, so deleting `id` would strand its still-live users. A pass
    /// that resolves an instruction to itself must leave it in place.
    #[test]
    fn replace_instruction_with_itself_is_a_noop() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <entry @a:i32>
                %x = @a + 1;
                %y = %x + 2;
                return %y;
            "
        );
        // `%x` is used by `%y`; find both.
        let root = FunctionBody::from_id(&ctx, f).root().unwrap().id;
        let insns: Vec<InstructionId> = BasicBlock::from_id(&ctx, root)
            .iter_instruction_ids()
            .collect();
        let x = insns[0];
        let users_before = ctx.bodies[f].users_of(ValueId::Instruction(x));
        assert!(!users_before.is_empty(), "x should have a user (%y)");

        // Replace x with itself — must not delete x or disturb its users.
        ctx.bodies[f].replace_instruction(x, ValueId::Instruction(x));

        assert!(
            ctx.bodies[f].insns.contains(x.local),
            "x must survive a self-replacement"
        );
        assert_eq!(
            ctx.bodies[f].users_of(ValueId::Instruction(x)),
            users_before,
            "x's users must be unchanged"
        );
    }

    #[test]
    fn body_users_of_rejects_foreign_owned_values() {
        let (ctx, a, b, _, _, a_insn, _, _, _) = colliding_body_ids();
        assert!(
            ctx.bodies[b]
                .users_of(ValueId::Instruction(a_insn))
                .is_empty()
        );
        assert!(
            !ctx.bodies[a]
                .users_of(ValueId::Instruction(a_insn))
                .is_empty()
        );
    }

    #[test]
    #[should_panic(expected = "cannot replace uses of a value owned by another function")]
    fn body_replace_uses_rejects_foreign_old() {
        let (mut ctx, _, b, _, _, a_insn, b_insn, _, _) = colliding_body_ids();
        ctx.bodies[b]
            .replace_all_uses_with(ValueId::Instruction(a_insn), ValueId::Instruction(b_insn));
    }

    #[test]
    #[should_panic(expected = "cannot replace uses with a value owned by another function")]
    fn body_replace_uses_rejects_foreign_new() {
        let (mut ctx, _, b, _, _, a_insn, b_insn, _, _) = colliding_body_ids();
        ctx.bodies[b]
            .replace_all_uses_with(ValueId::Instruction(b_insn), ValueId::Instruction(a_insn));
    }

    #[test]
    #[should_panic(expected = "block belongs to another function")]
    fn body_block_access_rejects_colliding_foreign_id() {
        let (ctx, _, b, a_block, _, _, _, _, _) = colliding_body_ids();
        let _ = ctx.bodies[b].block(a_block);
    }

    #[test]
    #[should_panic(expected = "instruction belongs to another function")]
    fn body_insn_access_rejects_colliding_foreign_id() {
        let (ctx, _, b, _, _, a_insn, _, _, _) = colliding_body_ids();
        let _ = ctx.bodies[b].insn(a_insn);
    }

    #[test]
    #[should_panic(expected = "block parameter belongs to another function")]
    fn body_param_access_rejects_colliding_foreign_id() {
        let (ctx, _, b, _, _, _, _, a_param, _) = colliding_body_ids();
        let _ = ctx.bodies[b].block_param(a_param);
    }

    // The `push_block` foreign-parent asserts are gone (stage 2): block ownership
    // is derived from the storing arena, so a block pushed into body `b` is owned
    // by `b` by construction — a foreign parent is unrepresentable.

    #[test]
    fn make_function_creates_function_with_correct_name_root_address() {
        let mut ctx = Context::new();
        let f = FunctionBody::make(&mut ctx, "main".into()).unwrap();
        assert_eq!(f.name(), "main");
    }

    #[test]
    fn get_function_by_name_returns_correct_function() {
        let mut ctx = Context::new();
        let id = FunctionBody::make(&mut ctx, "foo".into()).unwrap().id();
        let f = FunctionBody::from_name(&ctx, "foo").unwrap();
        assert_eq!(f.id(), id);
        assert_eq!(f.name(), "foo");
    }

    #[test]
    fn get_function_by_name_returns_none_if_not_found() {
        let ctx = Context::new();
        assert!(FunctionBody::from_name(&ctx, "nonexistent").is_none());
    }

    #[test]
    fn get_function_by_addr_returns_correct_function() {
        let mut ctx = Context::new();
        let id = FunctionBody::make_at_addr(&mut ctx, 0x2000, None).id();
        let addresses = crate::address_index::AddressIndex::analyze(&ctx);
        let f = FunctionBody::from_id(&ctx, addresses.function_at(0x2000).unwrap());
        assert_eq!(f.id(), id);
        assert_eq!(f.address(), Some(0x2000));
        assert_eq!(f.name(), "fn_2000");
    }

    #[test]
    fn get_function_by_addr_returns_none_if_missing() {
        let ctx = Context::new();
        let addresses = crate::address_index::AddressIndex::analyze(&ctx);
        assert!(addresses.function_at(0xdeadbeef).is_none());
    }

    #[test]
    fn add_block_via_function_mut_ref_updates_blocks_list() {
        let mut ctx = Context::new();
        let baz_id = FunctionBody::make(&mut ctx, "baz".into()).unwrap().id;
        let root = BasicBlock::make(&mut ctx, baz_id).id;
        let extra = BasicBlock::make(&mut ctx, baz_id).id;

        let mut baz = FunctionBody::from_id_mut(&mut ctx, baz_id);
        baz.add_block(root);
        baz.add_block(extra);

        let block_ids: Vec<_> = baz.blocks().map(|b| b.id).collect();
        assert!(block_ids.contains(&root));
        assert!(block_ids.contains(&extra));
    }

    #[test]
    fn display_shows_function_name_and_block_contents() {
        let mut ctx = Context::new();
        FunctionBody::make(&mut ctx, "display_test".into()).unwrap();

        let f = FunctionBody::from_name(&ctx, "display_test").unwrap();

        let s = f.to_string();
        assert!(s.contains("fn display_test:"));
    }

    #[test]
    fn iter_yields_all_blocks() {
        let mut ctx = Context::new();
        let f_id = FunctionBody::make(&mut ctx, "iter_fn".into()).unwrap().id;
        let root = BasicBlock::make(&mut ctx, f_id).id;
        let extra = BasicBlock::make(&mut ctx, f_id).id;
        let mut f = FunctionBody::from_id_mut(&mut ctx, f_id);
        f.add_block(root);
        f.add_block(extra);

        let f = FunctionBody::from_name(&ctx, "iter_fn").unwrap();
        let ids: Vec<_> = f.iter().map(|b| b.id).collect();
        assert!(ids.contains(&root));
        assert!(ids.contains(&extra));
    }

    #[test]
    fn into_iterator_for_function_ref_matches_iter() {
        let mut ctx = Context::new();
        let f_id = FunctionBody::make(&mut ctx, "into_iter_fn".into())
            .unwrap()
            .id;
        let b1 = BasicBlock::make(&mut ctx, f_id).id;
        let b2 = BasicBlock::make(&mut ctx, f_id).id;
        let mut f = FunctionBody::from_id_mut(&mut ctx, f_id);
        f.add_block(b1);
        f.add_block(b2);

        let f = FunctionBody::from_name(&ctx, "into_iter_fn").unwrap();
        let mut via_iter: Vec<usize> = f.iter().map(|b| usize::from(b.id.local)).collect();
        let mut via_into: Vec<usize> = (&f).into_iter().map(|b| usize::from(b.id.local)).collect();
        via_iter.sort();
        via_into.sort();
        assert_eq!(via_iter, via_into);
    }

    #[test]
    fn qcode_fn_single_block_populates_function() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn simple:
                <entry>
                    return at 0;
            "
        );

        let f = FunctionBody::from_name(&ctx, "simple").unwrap();
        assert_eq!(f.name(), "simple");
        assert!(f.root().is_some());
        assert_eq!(f.root().unwrap().name().unwrap(), "entry");
        assert_eq!(f.blocks().count(), 1);
    }

    #[test]
    fn qcode_fn_multi_block_populates_all_blocks() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn multiblock:
                <bb1>
                    if i8 1 goto <bb2> else goto <bb3>;

                <bb2>
                    goto <bb3>;

                <bb3>
                    return at 0;
            "
        );

        let f = FunctionBody::from_name(&ctx, "multiblock").unwrap();
        assert_eq!(f.root().unwrap().name().unwrap(), "bb1");
        let block_names: Vec<_> = f.blocks().filter_map(|b| b.name()).collect();
        assert!(block_names.contains(&"bb1"), "missing bb1");
        assert!(block_names.contains(&"bb2"), "missing bb2");
        assert!(block_names.contains(&"bb3"), "missing bb3");
        assert_eq!(f.blocks().count(), 3);
    }

    #[test]
    fn qcode_fn_id_variable_is_set() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn myfn:
                <start>
                    return at 0;
            "
        );

        let by_name = FunctionBody::from_name(&ctx, "myfn").unwrap();
        assert_eq!(by_name.name(), "myfn");
    }

    /// Split construction lets a rootless function temporarily win an address
    /// occupied by a block that has not yet been rehomed into its arena.
    #[test]
    fn indexed_address_registration_keeps_foreign_block_rootless() {
        let mut ctx = Context::new();

        // Simulate a branch-target block created at 0x1000 before the function
        // stub exists (as happens with tail-jumps to sibling functions).
        let block_id = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .id;
        let mut addresses = crate::address_index::AddressIndex::analyze(&ctx);
        addresses
            .register(
                &mut ctx,
                0x1000,
                crate::address_index::AddressTarget::Block(block_id),
            )
            .unwrap();

        let fn_id = FunctionBody::make(&mut ctx, "fn_1000".into()).unwrap().id;
        addresses
            .register(
                &mut ctx,
                0x1000,
                crate::address_index::AddressTarget::Function(fn_id),
            )
            .unwrap();

        assert_eq!(addresses.function_at(0x1000), Some(fn_id));
        assert_eq!(addresses.block_at(0x1000), None);
        assert!(FunctionBody::from_id(&ctx, fn_id).root().is_none());
        assert_ne!(block_id.func, fn_id);
    }
}

#[cfg(test)]
mod memory_interface_tests {
    use super::*;

    fn slot() -> InterfaceSlot {
        InterfaceSlot {
            base: SlotBase::Arg(0),
            offset: 8,
            size: 8,
        }
    }

    /// The interface survives a round-trip through the snapshot wire format.
    ///
    /// Back-compat is *not* tested here and is not provided: the payload is
    /// bincode under a hard version lock (`session.rs` `FORMAT_VERSION`, bumped
    /// for this field), so snapshots written before it are rejected outright
    /// rather than defaulted.
    #[test]
    fn memory_interface_round_trips_through_the_wire_format() {
        let state = MemoryChannelState {
            materialized: Some(MemoryInterfaceMap {
                inputs: vec![slot()],
                outputs: vec![InterfaceSlot {
                    base: SlotBase::Global(0x2000),
                    offset: 0,
                    size: 4,
                }],
            }),
            ..MemoryChannelState::default()
        };
        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&state, config).expect("encode memory state");
        let (decoded, _): (MemoryChannelState, _) =
            bincode::serde::decode_from_slice(&bytes, config).expect("decode memory state");
        assert_eq!(decoded, state);
    }

    /// A default (unmaterialized) channel reports no interface.
    #[test]
    fn default_memory_state_is_not_materialized() {
        assert_eq!(MemoryChannelState::default().materialized(), None);
    }

    /// The coarse-only setter must not disturb the other two components: a
    /// re-stamp of the written-space set is not a re-materialization.
    #[test]
    fn stamping_written_spaces_preserves_the_materialized_interface() {
        let mut ctx = Context::new();
        let fid = FunctionBody::make(&mut ctx, "keeps_interface".into())
            .unwrap()
            .id;
        let map = MemoryInterfaceMap {
            inputs: vec![slot()],
            outputs: vec![],
        };
        let mut body = FunctionBody::from_id_mut(&mut ctx, fid);
        body.set_memory_effects(MemoryChannelState {
            materialized: Some(map.clone()),
            ..MemoryChannelState::default()
        });
        body.set_written_spaces(None);

        let effects = FunctionBody::from_id(&ctx, fid).effects().memory.clone();
        assert_eq!(effects.materialized(), Some(&map));
        assert_eq!(effects.coarse, WrittenSpacesState::Unbounded);
    }

    /// `Unmappable` and `Global` are both address-less bases, but only the
    /// second is bindable. A consumer that collapsed them would synthesize a
    /// load from a bogus absolute address for a base it never resolved.
    #[test]
    fn an_unmappable_base_is_distinct_from_a_global_and_is_not_bindable() {
        let unmappable = InterfaceSlot {
            base: SlotBase::Unmappable,
            offset: 0,
            size: 8,
        };
        let global = InterfaceSlot {
            base: SlotBase::Global(0),
            offset: 0,
            size: 8,
        };
        assert_ne!(unmappable, global);
        assert!(!unmappable.is_bindable());
        assert!(global.is_bindable());
        assert!(
            InterfaceSlot {
                base: SlotBase::Arg(0),
                offset: -8,
                size: 8,
            }
            .is_bindable()
        );
    }
}
