use jstd::{Identifier, registry::Registry, stable_arena::StableArena};
use rustc_hash::FxHashMap;
use std::{
    borrow::Cow,
    collections::BTreeSet,
    fmt::{Display, Formatter},
    marker::PhantomData,
};

mod signature;
pub use signature::{FunctionSignature, ParamAttrs};

use crate::{
    context::Context,
    error::{Error, ErrorTy, Result},
    value::{
        BasicBlock, BlockId, BlockRef, Instruction, InstructionId, LocalValueId, ModuleView,
        QCodeView, Temp, TempId, TempSpace, TempSpaceId, Value, ValueId, Varnode, VarnodeId,
        block::EdgeData,
        block::cfg::{EdgeId, LocalBlockId},
        block_param::{BlockParam, BlockParamId, LocalParamId},
        insn::{LocalInsnId, Mnemonic},
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
    pub address: Option<u64>,

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
}

/// A function *body*: arenas, roster, root, reverse use-def, local names. The
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

    /// Reverse use-def map, scoped to this function: for each [`ValueId`] the
    /// list of *this function's* instructions that use it as an operand. By the
    /// SSA ownership invariant every user of an instruction/param value is
    /// intra-function, so an SSA def's users all live here. Shared values
    /// (literals, varnodes) may be used by many functions; each records only its
    /// own uses, which is all any pass needs (no pass queries a shared value's
    /// users program-wide). Kept in sync by
    /// [`push_insn`](crate::context::Context::push_insn),
    /// [`remove_instructions`](crate::context::Context::remove_instructions),
    /// [`Context::replace_all_uses_with`](crate::context::Context::replace_all_uses_with),
    /// and [`Context::replace_instruction_mnemonic`](crate::context::Context::replace_instruction_mnemonic).
    ///
    /// Keyed by the body-local [`LocalValueId`] form of each used value (the
    /// owning func is this body's, so it is stripped — see
    /// [`ValueId::strip_func`]); the value list stays composite
    /// [`InstructionId`]s.
    #[serde(default)]
    pub(crate) users: FxHashMap<LocalValueId, Vec<LocalInsnId>>,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FunctionKind {
    #[default]
    Machine,
    Lambda,
}

impl<'str> FunctionInterface<'str> {
    /// A fresh interface named `name`, with default (empty) signature/kind.
    pub fn new(name: Cow<'str, str>) -> Self {
        Self {
            name,
            address: None,
            is_external: false,
            signature: None,
            kind: FunctionKind::Machine,
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
    /// edge collections, the roster, and the reverse-use map. IDs, liveness,
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
            block.instructions.shrink_to_fit();
            block.params.shrink_to_fit();
            block.edges.shrink_to_fit();
        }
        for insns in self.users.values_mut() {
            insns.shrink_to_fit();
        }
        self.users.shrink_to_fit();
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
            users: FxHashMap::default(),
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
            users: FxHashMap::default(),
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

    /// This function's instructions that use `value` as an operand (see
    /// [`users`](Self::users)). Empty for a value this function never uses.
    pub(crate) fn local_users_of(&self, value: ValueId) -> &[LocalInsnId] {
        self.users
            .get(&value.strip_func())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// This body's qualified instruction IDs that use `value`. A value owned by
    /// another function has no users in this body, even if its local index
    /// collides with one of this body's values.
    pub fn users_of(&self, value: ValueId) -> Vec<InstructionId> {
        if value
            .owning_function()
            .is_some_and(|owner| owner != self.id())
        {
            return Vec::new();
        }
        self.local_users_of(value)
            .iter()
            .map(|&local| InstructionId::new(self.id(), local))
            .collect()
    }

    /// Iterate this function's recorded `(value, users)` reverse-use entries, with
    /// keys in their stored body-local form (qualify via the owning func at the
    /// [`FunctionRef`] wrapper). Read-only; used by the users-map consistency verifier.
    pub fn user_map_entries(&self) -> impl Iterator<Item = (LocalValueId, &[LocalInsnId])> {
        self.users.iter().map(|(v, u)| (*v, u.as_slice()))
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
            LocalValueId::Temp(_) | LocalValueId::BasicBlock(_) | LocalValueId::Function(_) => None,
        }
    }

    /// Appends a body-local temporary space and returns its qualified ID.
    pub fn push_temp_space(&mut self, space: TempSpace) -> TempSpaceId {
        TempSpaceId::new(self.id(), self.temp_spaces.push(space))
    }

    /// Appends a body-local temporary value and returns its qualified ID.
    pub fn push_temp(&mut self, temp: Temp<'str>) -> TempId {
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
        let key = ValueId::BlockParam(id).strip_func();
        let name = self.params[id.local].name.clone();
        if let Some(name) = name {
            self.names.forget(name.as_ref());
        }
        self.users.remove(&key);
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

    /// Push a fresh instruction into this body's arena, recording each operand's
    /// use in the reverse-use map.
    pub fn push_insn(&mut self, insn: Instruction<'str>) -> InstructionId {
        InstructionId::new(self.id(), self.push_insn_local(insn))
    }

    /// Push a fresh instruction into this body's arena, recording each operand's
    /// use in the reverse-use map, and return its **body-local** id. The id-less
    /// twin of [`push_insn`](Self::push_insn), usable on a detached body.
    pub fn push_insn_local(&mut self, insn: Instruction<'str>) -> LocalInsnId {
        let args: Vec<LocalValueId> = insn.mnemonic().args().into_iter().collect();
        let local = self.insns.push(insn);
        for arg in args {
            self.users.entry(arg).or_default().push(local);
        }
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
    pub fn push_block_local(&mut self, block: BasicBlock<'str>) -> LocalBlockId {
        let local = self.blocks.push(block);
        self.roster.push(local);
        local
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
    pub fn push_block_param(&mut self, param: BlockParam<'str>) -> BlockParamId {
        let local = self.params.push(param);
        BlockParamId::new(self.id(), local)
    }

    /// Push a fresh block parameter into this body's arena and wire it into
    /// `block`'s parameter list, returning its **body-local** id. The id-less twin
    /// of [`push_block_param`](Self::push_block_param), usable on a detached body.
    pub fn push_block_param_local(
        &mut self,
        block: LocalBlockId,
        param: BlockParam<'str>,
    ) -> LocalParamId {
        let local = self.params.push(param);
        self.blocks[block].params.push(local);
        local
    }

    /// Append an already-created instruction to the end of `block`, setting its
    /// parent (id-free; the mutation twin of [`BaseRef::push_insn`], usable on a
    /// detached body).
    pub fn append_insn_local(&mut self, block: LocalBlockId, insn: LocalInsnId) {
        self.insns[insn].parent = Some(block);
        self.blocks[block].instructions.push(insn);
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
        let index = self
            .block(block)
            .instructions
            .iter()
            .position(|&local| InstructionId::new(block.func, local) == before)
            .expect("before not in block");
        self.insn_mut(insn).parent = Some(block.local);
        self.block_mut(block)
            .instructions
            .insert(index, insn.localize(block.func));
    }

    /// Move the live, non-terminator instruction `insn` immediately before the
    /// live instruction `before`, inferring the destination block from
    /// `before`. The moved instruction keeps its ID, payload, name, and use-map
    /// entries. Supports both cross-block motion and reordering within one
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

        let source = self
            .insn(insn)
            .parent
            .map(|local| BlockId::new(id, local))
            .expect("moved instruction must belong to a block");
        let target = self
            .insn(before)
            .parent
            .map(|local| BlockId::new(id, local))
            .expect("anchor instruction must belong to a block");
        let source_index = self
            .block(source)
            .instructions
            .iter()
            .position(|&local| local == insn.local)
            .expect("moved instruction missing from its parent block");
        let before_index = self
            .block(target)
            .instructions
            .iter()
            .position(|&local| local == before.local)
            .expect("anchor instruction missing from its parent block");
        let insert_index = if source == target && source_index < before_index {
            before_index - 1
        } else {
            before_index
        };

        self.block_mut(source).instructions.remove(source_index);
        self.block_mut(target)
            .instructions
            .insert(insert_index, insn.local);
        self.insn_mut(insn).parent = Some(target.local);
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

    /// Replace every use of `old` with `new` across this body's instructions and
    /// update the reverse use-map (SSA defs only; `old` is intra-function).
    pub fn replace_all_uses_with(&mut self, old: ValueId, new: ValueId) {
        if old == new {
            return;
        }
        let Some(func) = old.owning_function() else {
            return;
        };
        assert_eq!(
            func,
            self.id(),
            "cannot replace uses of a value owned by another function"
        );
        if let Some(new_owner) = new.owning_function() {
            assert_eq!(
                new_owner,
                self.id(),
                "cannot replace uses with a value owned by another function"
            );
        }
        let users = self.users_of(old);
        let old = old.localize(func);
        let new = new.localize(func);
        for user in users {
            self.insn_mut(user).mnemonic_mut().replace_value(old, new);
            self.users.entry(new).or_default().push(user.localize(func));
        }
        self.users.remove(&old);
    }

    /// Remove instruction `id` from its block, unlink its outgoing CFG edges if a
    /// terminator, clear its name, prune its operand use-lists, and physically
    /// drop its payload.
    pub fn remove_instruction(&mut self, id: InstructionId) {
        assert_eq!(
            id.func,
            self.id(),
            "instruction belongs to another function"
        );
        let func = self.id();
        let (parent, name, is_terminator, args) = {
            let insn = self.insn(id);
            (
                insn.parent.map(|l| BlockId::new(self.id(), l)),
                insn.name.clone(),
                insn.mnemonic().is_terminator(),
                insn.mnemonic().args().into_iter().collect::<Vec<_>>(),
            )
        };

        if let Some(block_id) = parent {
            self.block_mut(block_id)
                .instructions
                .retain(|&local| local != id.localize(block_id.func));
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
        for arg in args {
            let remove_key = if let Some(users) = self.users.get_mut(&arg) {
                users.retain(|&local| local != id.localize(func));
                users.is_empty()
            } else {
                false
            };
            if remove_key {
                self.users.remove(&arg);
            }
        }
        self.users.remove(&ValueId::Instruction(id).strip_func());
        self.insns.remove(id.local);
    }

    /// Rehome `remove`'s outgoing CFG edges onto `keep`. The direct edge and
    /// `keep`'s forwarding terminator have already been removed by the caller.
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

    /// Replace an instruction's mnemonic in place, keeping the reverse use-map in
    /// sync.
    pub fn replace_instruction_mnemonic(&mut self, id: InstructionId, mnemonic: Mnemonic) {
        assert_eq!(
            id.func,
            self.id(),
            "instruction belongs to another function"
        );
        self.replace_instruction_mnemonic_local(id.local, mnemonic);
    }

    /// Replace an instruction's mnemonic in place, keeping the reverse use-map in
    /// sync, over a **body-local** instruction id (id-free; the twin of
    /// [`replace_instruction_mnemonic`](Self::replace_instruction_mnemonic),
    /// usable on a detached body).
    pub fn replace_instruction_mnemonic_local(&mut self, id: LocalInsnId, mnemonic: Mnemonic) {
        let old_args = self.insns[id]
            .mnemonic()
            .args()
            .into_iter()
            .collect::<Vec<_>>();
        for arg in old_args {
            let now_empty = if let Some(users) = self.users.get_mut(&arg) {
                users.retain(|&local| local != id);
                users.is_empty()
            } else {
                false
            };
            if now_empty {
                self.users.remove(&arg);
            }
        }
        *self.insns[id].mnemonic_mut() = mnemonic;
        let new_args = self.insns[id]
            .mnemonic()
            .args()
            .into_iter()
            .collect::<Vec<_>>();
        for arg in new_args {
            self.users.entry(arg).or_default().push(id);
        }
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
    pub fn delete_block(&mut self, block: BlockId) {
        assert_eq!(block.func, self.id(), "block belongs to another function");
        let mut edges: Vec<EdgeId> = self.block(block).edges.iter().copied().collect();
        edges.sort_unstable();
        for edge in edges {
            self.remove_cfg_edge(edge);
        }
        let insns: Vec<InstructionId> = self
            .block(block)
            .instructions
            .iter()
            .map(|&local| InstructionId::new(self.id(), local))
            .collect();
        for insn in insns {
            self.remove_instruction(insn);
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
        self.blocks.remove(block.local);
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
            .instructions
            .last()
            .and_then(
                |&local| match self.insn(InstructionId::new(keep.func, local)).mnemonic() {
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
        let b_insns = std::mem::take(&mut self.block_mut(other).instructions);
        for &local in &b_insns {
            self.insn_mut(InstructionId::new(other.func, local)).parent = Some(keep.local);
        }
        self.block_mut(keep).instructions.extend(b_insns);
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
        let existing = match id.name_scope_function() {
            Some(_) => self.names.get(&name).map(|id| id.qualify(self.id())),
            None => shared.get_named(&name),
        };
        if let Some(existing) = existing {
            return if existing == id {
                Ok(())
            } else {
                Err(Error::spanless(ErrorTy::DuplicateName(name.to_string())))
            };
        }
        match id.name_scope_function() {
            Some(_) => self.names.register(name, id.localize(self.id()), old_name),
            None => {
                unimplemented!(
                    "a function body cannot register a global name (shared is read-only)"
                )
            }
        }
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
        function.set_pure_reg(true);
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
        let name = name.unwrap_or_else(|| Cow::Owned(format!("fn_{address:x}")));
        let id = FunctionId::from(ctx.bodies.len());
        let pushed = ctx.push_function(
            FunctionInterface::new(name.clone()),
            FunctionBody::empty_with_id(id),
        );
        debug_assert_eq!(pushed, id);

        Self::from_id_mut(ctx, id)
            .with_name(name)
            .expect("Function name is not unique")
            .with_address_indexed(addresses, address)
            .expect("Function address is not unique")
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

    /// Iterate this function's recorded `(value, users)` reverse-use entries
    /// (see [`FunctionBody::user_map_entries`]).
    pub fn user_map_entries(&'s self) -> impl Iterator<Item = (ValueId, Vec<InstructionId>)> + 's {
        let func = self.id;
        self.inner().user_map_entries().map(move |(v, u)| {
            (
                v.qualify(func),
                u.iter()
                    .map(|&local| InstructionId::new(func, local))
                    .collect(),
            )
        })
    }

    /// Resolve a block/instruction/param/Temp `name` within this function's local name
    /// table (see [`FunctionBody::names`]). `None` if this function has no such name.
    pub fn local_named(&'s self, name: &str) -> Option<ValueId> {
        self.inner().names.get(name).map(|id| id.qualify(self.id))
    }

    /// Whether this function's full register effect is captured by its call
    /// interface: it reads no registers (only its explicit args) and writes
    /// exactly its [`clobbered_regs`](Self::clobbered_regs). See
    /// [`FunctionSignature::externally_resolved`].
    pub fn is_externally_resolved(&'s self) -> bool {
        self.interface()
            .signature
            .as_ref()
            .is_some_and(|s| s.externally_resolved)
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

    /// Registers concretely written by this function, as set by analysis.
    pub fn clobbered_regs(&'s self) -> Option<&'ctx [VarnodeId]> {
        self.interface()
            .signature
            .as_ref()
            .and_then(|s| s.clobbered.as_deref())
    }

    /// The non-register memory spaces this function may (transitively) write, as
    /// set by analysis. `Some(spaces)` is exact (a space not listed is never
    /// written); `None` means unknown/unbounded. See
    /// [`FunctionSignature::written_spaces`].
    pub fn written_spaces(&'s self) -> Option<&'ctx [crate::space::SpaceId]> {
        self.interface()
            .signature
            .as_ref()
            .and_then(|s| s.written_spaces.as_deref())
    }

    /// Whether `argpromote_registers` has functionalized this function's register
    /// side effects into a pure value function. See
    /// [`FunctionSignature::pure_reg`].
    pub fn is_pure_reg(&'s self) -> bool {
        self.interface()
            .signature
            .as_ref()
            .is_some_and(|s| s.pure_reg)
    }

    /// Whether argpromote has functionalized *every* side-effect channel of this
    /// function — it is a deterministic pure function of its by-value params,
    /// touching no caller-visible memory or registers. Strictly stronger than
    /// [`is_pure_reg`](Self::is_pure_reg). See [`FunctionSignature::is_pure`].
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

    /// Registers read before written (function inputs), as inferred by analysis.
    ///
    /// Legacy ABI input-register list. It is **`None` for `pure_reg` functions**
    /// (argpromote never populates it); their by-value root block params are the
    /// source of truth for the call interface, so prefer the params (e.g.
    /// [`input_arg_name`](Self::input_arg_name)). Retained only for the
    /// conventional/external calling-convention path (`summaries`).
    #[deprecated(
        note = "legacy ABI register list; None for pure_reg functions. Use the root block params \
                as the call interface; this remains only for the conventional/external path."
    )]
    pub fn input_regs(&'s self) -> Option<&'ctx [VarnodeId]> {
        self.interface()
            .signature
            .as_ref()
            .and_then(|s| s.inputs.as_deref())
    }

    /// The display name for the call-site argument bound to input `index`: the
    /// register name for a register input, or a synthesized `stack_<addr>` slot
    /// name for a stack-passed input (whose varnode is a nameless stack-space
    /// offset carrier). Mirrors mem2reg's `block_param_name_for_var` so a call
    /// argument reads with the same name as the callee's promoted stack
    /// parameter. `None` when there is no input at `index`.
    pub fn input_arg_name(&'s self, index: usize) -> Option<String> {
        // The root block param at `index` is the interface element a call
        // argument actually binds to, named after its register by
        // `argpromote_registers` or `stack_<addr>` by mem2reg's
        // `block_param_name_for_var`. Prefer it: it is the source of truth and is
        // populated even for `pure_reg` functions, whose ABI register list
        // (`input_regs`) is never filled in.
        if let Some(root) = self.root()
            && let Some(name) = root
                .params()
                .nth(index)
                .and_then(|p| p.name().map(str::to_owned))
        {
            return Some(name);
        }

        // Explicit per-argument names recorded by name-derived passes (e.g. an
        // external callee's C-prototype parameters, plus a synthesized
        // `return_address` slot). The source of truth for bodyless externals,
        // which have neither a root block nor an inferred input-register list.
        if let Some(name) = self
            .interface()
            .signature
            .as_ref()
            .and_then(|s| s.input_names.as_ref())
            .and_then(|names| names.get(index))
            .and_then(|n| n.as_deref())
        {
            return Some(name.to_owned());
        }

        // Fall back to the inferred input-register list: a register name, or a
        // synthesized `stack_<addr>` slot name for a stack-passed input.
        // Intentional use of the legacy list — only reached when the param has no
        // name (conventional functions, never `pure_reg`).
        #[allow(deprecated)]
        let input = self.input_regs()?.get(index).copied()?;
        let vn = Varnode::from_id(self.view.shared(), input);
        if let Some(name) = vn.name() {
            return Some(name.to_owned());
        }
        let space = vn.space();
        if space.name.as_deref() == Some("stack") {
            return Some(format!("stack_{:x}", vn.address() as u64));
        }
        None
    }

    /// Registers saved and restored unchanged (preserved across calls), as
    /// inferred by analysis.
    pub fn saved_regs(&'s self) -> Option<&'ctx [VarnodeId]> {
        self.interface()
            .signature
            .as_ref()
            .and_then(|s| s.saved.as_deref())
    }

    /// The net change this function applies to the stack pointer between entry
    /// and return, as inferred by analysis. See [`FunctionSignature::stack_delta`].
    pub fn stack_delta(&'s self) -> Option<i64> {
        self.interface()
            .signature
            .as_ref()
            .and_then(|s| s.stack_delta)
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
        let old_address = self.interface().address;
        self.interface_mut().address = Some(address);
        if let Err(error) = self
            .ctx
            .set_address_indexed(addresses, address, self.id.into())
        {
            self.interface_mut().address = old_address;
            return Err(error);
        }
        Ok(())
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

    pub fn set_kind(&mut self, kind: FunctionKind) {
        self.interface_mut().kind = kind;
        if kind == FunctionKind::Lambda {
            self.set_is_pure(true);
            self.set_pure_reg(true);
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

    /// Records the analysis-computed clobbered register set on this function.
    pub fn set_clobbered_regs(&mut self, regs: Vec<VarnodeId>) {
        self.interface_mut()
            .signature
            .get_or_insert_default()
            .clobbered = Some(regs);
    }

    /// Records the analysis-computed set of non-register spaces this function may
    /// write (`None` = unknown/unbounded). See
    /// [`FunctionSignature::written_spaces`].
    pub fn set_written_spaces(&mut self, spaces: Option<Vec<crate::space::SpaceId>>) {
        self.interface_mut()
            .signature
            .get_or_insert_default()
            .written_spaces = spaces;
    }

    /// Marks this function's register effect as fully captured by its call
    /// interface — reads no registers, writes exactly its clobbered set. See
    /// [`FunctionSignature::externally_resolved`].
    pub fn set_externally_resolved(&mut self, value: bool) {
        self.interface_mut()
            .signature
            .get_or_insert_default()
            .externally_resolved = value;
    }

    /// Records the analysis-inferred input (live-in) register set on this function.
    ///
    /// Legacy ABI input-register list — see [`input_regs`](Self::input_regs).
    /// Not set for `pure_reg` functions, whose param interface supersedes it.
    #[deprecated(
        note = "legacy ABI register list; None for pure_reg functions. Use the root block params \
                as the call interface; this remains only for the conventional/external path."
    )]
    pub fn set_input_regs(&mut self, regs: Vec<VarnodeId>) {
        self.interface_mut()
            .signature
            .get_or_insert_default()
            .inputs = Some(regs);
    }

    /// Records display names for this function's positional call arguments, one
    /// per `Call.args` slot. Consulted by [`input_arg_name`](Self::input_arg_name)
    /// for callees (chiefly externals) whose argument names come from a C
    /// prototype rather than a register or promoted stack param.
    pub fn set_input_arg_names(&mut self, names: Vec<Option<Box<str>>>) {
        self.interface_mut()
            .signature
            .get_or_insert_default()
            .input_names = Some(names);
    }

    /// Marks this function as fully functionalized over its register channel.
    /// See [`FunctionSignature::pure_reg`].
    pub fn set_pure_reg(&mut self, value: bool) {
        self.interface_mut()
            .signature
            .get_or_insert_default()
            .pure_reg = value;
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

    /// Records the analysis-inferred saved (preserved) register set on this function.
    pub fn set_saved_regs(&mut self, regs: Vec<VarnodeId>) {
        self.interface_mut().signature.get_or_insert_default().saved = Some(regs);
    }

    /// Records the output (return-value) register set on this function.
    pub fn set_output_regs(&mut self, regs: Vec<VarnodeId>) {
        self.interface_mut()
            .signature
            .get_or_insert_default()
            .outputs = Some(regs);
    }

    /// Records the analysis-inferred net stack-pointer delta on this function.
    pub fn set_stack_delta(&mut self, delta: i64) {
        self.interface_mut()
            .signature
            .get_or_insert_default()
            .stack_delta = Some(delta);
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
    use qcode_macro::qcode;

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
            .instruction_ids();
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
        let a_insn = a_root.instruction_ids()[0];
        let b_insn = b_root.instruction_ids()[0];
        let a_param = a_root.params().next().unwrap().id;
        let b_param = b_root.params().next().unwrap().id;
        assert_eq!(a_block.local, b_block.local);
        assert_eq!(a_insn.local, b_insn.local);
        assert_eq!(a_param.local, b_param.local);
        (
            ctx, raw_a, raw_b, a_block, b_block, a_insn, b_insn, a_param, b_param,
        )
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
