//! The central arena for all IR state: [`Context`].

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fmt::Display,
};

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

    /// Reverse mapping of name hints to value IDs, used to ensure that name hints are unique
    name_map: HashMap<Cow<'str, str>, ValueId>,

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
    pub fn get_or_make_block(&mut self, addr: u64) -> BlockId {
        match BasicBlock::from_addr(self, addr) {
            Some(block) => block.id,
            None => BasicBlock::make(self).with_address(addr).id,
        }
    }

    /// Returns a list of all blocks in the context
    pub fn block_ids(&self) -> Vec<BlockId> {
        self.values.basic_blocks.iter().map(|b| b.id).collect()
    }

    /// Returns a list of all instructions in the context
    pub fn instruction_ids(&self) -> Vec<InstructionId> {
        self.values.instructions.iter().map(|i| i.id).collect()
    }

    /// Returns a list of all functions in the context
    pub fn function_ids(&self) -> Vec<FunctionId> {
        self.values.functions.iter().map(|f| f.id).collect()
    }

    /// Iterates over all the instructions in the context
    pub fn instructions(&self) -> impl Iterator<Item = InstructionRef<'str, '_>> + '_ {
        self.values
            .instructions
            .iter()
            .map(|i| Instruction::from_id(self, i.id))
    }

    /// Iterates over all the blocks in the context
    pub fn blocks(&self) -> impl Iterator<Item = BlockRef<'str, '_>> + '_ {
        self.values
            .basic_blocks
            .iter()
            .map(|b| BlockRef::from_id(self, b.id))
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

    pub fn varnodes(&self) -> impl Iterator<Item = VarnodeRef<'str, '_>> + '_ {
        self.values
            .varnodes
            .iter()
            .map(|v| Varnode::from_id(self, v.id))
    }

    /// Adds a directed edge in the CFG from `from` to `to`, returning its id.
    pub fn add_cfg_edge(&mut self, from: BlockId, to: BlockId) -> EdgeId {
        let edge_id = self.values.edges.push(EdgeData { from, to });
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
        let EdgeData { from, to } = self.values.edges[edge_id];
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
        InstructionRef::new(self, id)
    }

    /// Returns an immutable reference to the varnode mapped to the named
    /// register `id`.
    pub fn get_register(&self, id: RegisterId) -> VarnodeRef<'str, '_> {
        Varnode::from_id(self, self.registers[&id])
    }

    /// Creates a [`Value`] representing an integer constant of the given byte width.
    pub fn get_const(&mut self, value: u64, size: usize) -> LiteralRef<'str, '_> {
        let type_id = self.types.get_or_make_int(size);
        let id = self.values.get_or_make_typed_literal(value, type_id, size);
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

    /// Returns the [`TypeId`] of any [`ValueId`] in this context.
    ///
    /// Varnodes are typed as `Int(varnode.size())`. Blocks, functions, and other
    /// non-data values return `Int(0)`.
    pub fn type_of(&mut self, id: ValueId) -> crate::types::TypeId {
        match id {
            ValueId::Literal(lid) => self.values.literals[lid].type_id,
            ValueId::Instruction(iid) => self.values.instructions[iid].type_id,
            ValueId::BlockParam(pid) => self.values.block_params[pid].type_id,
            ValueId::Varnode(vid) => {
                let size = self.values.varnodes[vid].size_bytes();
                self.types.get_or_make_int(size)
            }
            // Exhaustive on purpose: a new ValueId variant must decide its type
            // here rather than silently inheriting the zero-width sentinel.
            ValueId::BasicBlock(_) | ValueId::Function(_) => self.types.get_or_make_int(0),
        }
    }

    /// Return all instructions that use `value` as an operand.
    pub fn users(&self, value: impl Into<ValueId>) -> &[InstructionId] {
        self.values.users_of(value.into())
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
        for user_id in users {
            Instruction::from_id_mut(self, user_id)
                .mnemonic_mut()
                .replace_value(old, new);
            self.values.users.entry(new).or_default().push(user_id);
        }
        self.values.users.remove(&old);
    }

    /// Replaces one instruction's mnemonic and keeps the reverse use map in sync.
    ///
    /// This is for transforms that change an instruction in place without
    /// changing its identity, parent block, address, or result type.
    pub fn replace_instruction_mnemonic(&mut self, id: InstructionId, mnemonic: Mnemonic) {
        let old_args = self.values.instructions[id].mnemonic().args();
        for arg in old_args {
            let mut remove_arg = false;
            if let Some(users) = self.values.users.get_mut(&arg) {
                users.retain(|&user| user != id);
                remove_arg = users.is_empty();
            }
            if remove_arg {
                self.values.users.remove(&arg);
            }
        }

        // Drop this instruction's old call edge (if it was a direct call) before
        // overwriting the mnemonic; the new one's edge is recorded below.
        if let Some(target) = self.values.instructions[id].mnemonic().call_target()
            && let Some(sites) = self.values.call_sites.get_mut(&target)
        {
            sites.retain(|&site| site != id);
        }

        *Instruction::from_id_mut(self, id).mnemonic_mut() = mnemonic;

        for arg in self.values.instructions[id].mnemonic().args() {
            self.values.users.entry(arg).or_default().push(id);
        }
        if let Some(target) = self.values.instructions[id].mnemonic().call_target() {
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
        let parent = self.values.instructions[id].parent;
        let name = self.values.instructions[id].name.clone();

        if let Some(block_id) = parent {
            self.values.basic_blocks[block_id]
                .instructions
                .retain(|&i| i != id);

            // If this instruction was a terminator instruction in a basic block,
            // remove cfg edges
            if Instruction::from_id(self, id).mnemonic().is_terminator() {
                let mut edges_to_remove = HashSet::new();
                for edge in BasicBlock::from_id(self, block_id).successors() {
                    edges_to_remove.insert(edge.0);
                }

                for edge_id in edges_to_remove {
                    self.remove_cfg_edge(edge_id);
                }
            }
        }

        self.values.instructions[id].parent = None;

        if let Some(ref n) = name {
            self.name_map.remove(n.as_ref());
        }
        self.values.instructions[id].name = None;

        self.values.remove_instructions(&HashSet::from([id]));
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

    /// Changes the name of a value
    pub fn update_name(
        &mut self,
        name: Cow<'str, str>,
        id: ValueId,
        old_name: Option<&str>,
    ) -> Result<()> {
        if let Some(old_name) = old_name {
            self.name_map.remove(old_name);
        }
        match self.name_map.insert(name.clone(), id) {
            Some(_) => Err(Error::spanless(ErrorTy::DuplicateName(name.to_string()))),
            None => Ok(()),
        }
    }

    /// Attempts to get a value ID by its name.
    /// Returns `None` if no value with the given name exists.
    pub fn get_named(&self, name: &str) -> Option<ValueId> {
        self.name_map.get(name).copied()
    }

    /// Gets a value ID at a given address
    pub fn get_at_addr(&self, addr: &u64) -> Option<ValueId> {
        self.address_map.get(addr).copied()
    }

    /// Gets a unique name for a value, generating one
    /// if necessary by appending a numeric suffix to the provided name
    /// until an unused name is found.
    pub fn get_unique_name(&self, name: Cow<'str, str>) -> Cow<'str, str> {
        // If the name is already taken, we don't want to overwrite it, since that would make debugging harder
        // instead we add a numeric suffix to the name until we find an unused name
        let mut suffix = 0;

        let mut unique_name = Cow::Owned(name.to_string());

        while self.name_map.contains_key(&unique_name) {
            suffix += 1;
            unique_name = Cow::Owned(format!("{}_{suffix}", name));
        }
        unique_name
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

impl<'str> Graph for Context<'str> {
    type NodeId = BlockId;

    type EdgeId = EdgeId;

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
        self.values
            .basic_blocks
            .iter()
            .map(|block| BasicBlock::from_id(self, block.id))
    }

    fn edges(&self) -> impl Iterator<Item = Self::Edge<'_>> + '_ {
        self.values
            .edges
            .iter()
            .map(|edge| EdgeRef::new(self, edge.id))
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
        let block_ids: Vec<BlockId> = (0..n).map(|_| BasicBlock::make(ctx).id).collect();
        let mut f = Function::make(ctx, name.into()).unwrap();
        for id in block_ids {
            f.add_block(id);
        }
        f.id
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
                store(&ptr, i64 0x1234);
                return [ptr];
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
                %a = load(i64, &x);
                %b = load(i64, &x);
                return [%a];
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
                %a = load(i64, &x);
                return [%a];
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
                %a = load(i64, &x);
                return [%a];
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];
        assert!(
            ctx.get_named("a").is_some(),
            "name should be in map before removal"
        );

        ctx.remove_instruction(load_id);

        assert!(
            ctx.get_named("a").is_none(),
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
                %a = load(i64, &x);
                return [%a];
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
                %a = load(i64, &y);
                return [%a];
            "
        );
        assert!(
            ctx.get_named("a").is_some(),
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
                %a = load(i64, &x);
                %b = %a + i64 1;
                return [%b];
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
        assert_eq!(ctx.users(ptr), &[call_id]);

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
            ctx.users(ptr).is_empty(),
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
                %a = load(i64, x);
                return [%a];
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];
        let old_ptr = ValueId::Varnode(x);
        let new_ptr = ValueId::Varnode(y);
        assert_eq!(ctx.users(old_ptr), &[load_id]);
        assert!(ctx.users(new_ptr).is_empty());

        ctx.replace_instruction_mnemonic(
            load_id,
            Mnemonic::Load(Load {
                space: ctx.default_space,
                ptr: new_ptr,
                size: 8,
            }),
        );

        assert!(ctx.users(old_ptr).is_empty());
        assert_eq!(ctx.users(new_ptr), &[load_id]);
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
                %a = load(i64, x);
                return [%a];
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

        assert!(ctx.users(old_ptr).is_empty());
        assert_eq!(
            ctx.users(new_arg),
            &[load_id, load_id],
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
                %a = load(i64, &x);
                return [%a];
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];

        // Manually detach from block without using remove_instruction,
        // simulating an instruction with no parent.
        ctx.values.instructions[load_id].parent = None;

        // Should not panic even though parent is None.
        ctx.remove_instruction(load_id);

        assert!(ctx.get_named("a").is_none());
    }

    #[test]
    fn add_cfg_edge_returns_id_and_remove_unlinks_both_blocks() {
        let mut ctx = Context::new();
        let a = BasicBlock::make(&mut ctx).id;
        let b = BasicBlock::make(&mut ctx).id;

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
        ctx.memory_image.add_segment(0x1000, vec![0u8; 4], true); // code
        ctx.memory_image.add_segment(0x2000, vec![0u8; 4], false); // data

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
        ctx.memory_image.add_segment(0x1000, vec![0u8; 4], true); // code
        ctx.memory_image.add_segment(0x2000, vec![0u8; 4], false); // data
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
                %a = load(i64, &ptr);
                %b = %a + i64 0x10;
                store(&ptr, i64 0x1234);
                return [%b];
            "
        );

        // A StackAddress type exercises the custom TypeManager serialization.
        let stack_space = ctx.make_named_temp_space("stack");
        let sa = ctx.types.get_or_make_stack_address(8, Some(stack_space));
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
        // The StackAddress type round-trips: same id, same size, still a stack address.
        assert_eq!(restored.types.size_of(sa), sa_size);
        assert!(restored.types.is_stack_address(sa));
    }
}
