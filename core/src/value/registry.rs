use crate::{
    assumption::{KnownContradiction, Proposition, Truth, Violation},
    types::TypeId,
    value::{
        ValueId,
        block::{BasicBlock, BlockId, EdgeData, EdgeId},
        block_param::{BlockParam, BlockParamId},
        bytes::{Bytes, BytesId},
        function::{Function, FunctionId},
        insn::{Instruction, InstructionId},
        literal::{Literal, LiteralId},
        varnode::{Varnode, VarnodeId},
    },
};
// NOTE (IR-ownership refactor): instruction/block/param/edge *storage* now lives
// in each `Function` (see `Function::insns/blocks/params/edges`). This registry
// keeps only the global value arenas plus the cross-function maps (`users`,
// `call_sites`, `synthetic_callees`). Composite IDs route through the owning
// function via the `instruction()/block()/block_param()/edge()` accessors below.
use jstd::registry::Registry;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::BTreeSet;

/// Central storage arena for all IR values in a [`Context`](crate::context::Context).
///
/// Each field is a typed arena ([`Registry`]) keyed by the corresponding ID
/// type. Values are append-only: once pushed, their ID is stable for the
/// lifetime of the registry and their data is never moved.
///
/// # Invariants
///
/// - **`push_insn` is final.** Instructions are immutable after insertion.
///   The owning function's `users` map is populated at push time from the
///   instruction's operands and is not updated if operands are later altered via
///   interior mutation. Use
///   [`Context::replace_all_uses_with`](crate::context::Context::replace_all_uses_with)
///   to rewrite operands while keeping `users` consistent.
///
/// - **`users` is managed internally.** The reverse use-def map now lives in
///   each [`Function`] (function-scoped; see [`Function::users`]). Do not mutate
///   it directly. Read it through
///   [`FunctionRef::users_of`](crate::value::FunctionRef::users_of) /
///   [`Context::users`](crate::context::Context::users), and remove dead
///   instructions via [`remove_instructions`](Self::remove_instructions).
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ValueRegistry<'str> {
    /// Literal (constant) storage.
    pub literals: Registry<LiteralId, Literal>,

    /// Opaque byte-blob constant storage (constants wider than a `u64`).
    #[serde(default)]
    pub bytes: Registry<BytesId, Bytes>,

    /// User-forced rendering overrides for `Bytes` blobs (e.g. from the GUI
    /// Strings pane). Absent entries render under [`BytesDisplay::Auto`].
    #[serde(default)]
    pub(crate) bytes_display: HashMap<BytesId, crate::value::BytesDisplay>,

    /// Varnode storage.
    pub varnodes: Registry<VarnodeId, Varnode<'str>>,

    /// Per-varnode type overrides. A varnode is normally typed `Int(size)`; an
    /// entry here gives it a richer *global* type instead (e.g. the `FS_OFFSET`
    /// register typed `PtrTo<TEB>` by the TEB-seeding pass). Consulted by
    /// [`Context::type_of`](crate::context::Context::type_of) /
    /// [`stored_type_of`](crate::context::Context::stored_type_of). Set via
    /// [`Context::set_varnode_type`](crate::context::Context::set_varnode_type).
    #[serde(default)]
    pub(crate) varnode_types: HashMap<VarnodeId, TypeId>,

    /// Function storage. Each function owns its instruction/block/param/edge
    /// arenas; the composite-ID accessors ([`instruction`](Self::instruction)
    /// etc.) route through here.
    pub functions: Registry<FunctionId, Function<'str>>,

    /// Truth map of the assumption system: what each [`Proposition`] is
    /// currently assumed or known to be (see [`crate::assumption`]). Accessed
    /// through [`Context::assume_true`](crate::context::Context::assume_true)
    /// and friends.
    pub(crate) truths: HashMap<Proposition, Truth>,

    /// Proven facts that contradicted an assumption this round; non-empty means
    /// the checkpoint+replay driver must discard this working copy and replay.
    pub(crate) violations: Vec<Violation>,

    /// Proven facts that contradicted an existing *known* fact this round (e.g. a
    /// user override the analysis disproved). A hard error for the driver, not a
    /// replay signal. Transient per round, so not serialized.
    #[serde(default, skip)]
    pub(crate) known_contradictions: Vec<KnownContradiction>,

    /// Reverse call graph: for each callee [`FunctionId`], the direct-call sites
    /// (instructions) that target it. Kept in sync alongside `users` by
    /// [`push_insn`](Self::push_insn),
    /// [`remove_instructions`](Self::remove_instructions), and
    /// [`Context::replace_instruction_mnemonic`](crate::context::Context::replace_instruction_mnemonic).
    /// Indirect calls have no static target and are not recorded here.
    #[serde(default)]
    pub(crate) call_sites: HashMap<FunctionId, Vec<InstructionId>>,

    /// Synthetic forward call-graph edges that are not backed by a direct `Call`
    /// instruction: `caller FunctionId → set of callee entry addresses`. Used for
    /// relationships a pass recovers but the IR can't express as a direct call —
    /// e.g. `entry → main`, where `main` is passed to `__libc_start_main` as a
    /// pointer argument rather than called. Keyed by address so the edge resolves
    /// once a function exists at the callee, independent of when it materializes.
    /// Merged into [`FunctionRef::callees`](crate::value::FunctionRef::callees).
    #[serde(default)]
    pub(crate) synthetic_callees: HashMap<FunctionId, BTreeSet<u64>>,

    /// Intern cache for non-symbolic literals: `(masked_value, TypeId) → LiteralId`.
    literal_cache: HashMap<(u64, TypeId), LiteralId>,
}

impl<'str> ValueRegistry<'str> {
    /// Returns a canonical [`LiteralId`] for the given typed constant.
    ///
    /// The value is masked to `type_id`'s size before lookup. Symbolic literals
    /// (created via [`push_literal`](Self::push_literal)) are not included in
    /// the intern cache and will not alias with constants produced here.
    ///
    /// Call [`Context::get_const`] for the common `Int(size)` case; use this
    /// method directly when you need to preserve a non-`Int` type (e.g.
    /// [`StackAddress`](crate::types::StackAddress)) through folding.
    pub fn get_or_make_typed_literal(
        &mut self,
        mut value: u64,
        type_id: TypeId,
        size: usize,
    ) -> LiteralId {
        value &= if size >= 8 {
            u64::MAX
        } else {
            (1u64 << (size * 8)) - 1
        };

        if let Some(&id) = self.literal_cache.get(&(value, type_id)) {
            return id;
        }

        let id = self.literals.push(Literal {
            value,
            type_id,
            symbolic: None,
        });
        self.literal_cache.insert((value, type_id), id);
        id
    }

    /// Pushes a [`Literal`] with arbitrary fields (e.g. with a symbolic ref)
    /// without interning. Use [`get_or_make_literal`](Self::get_or_make_literal)
    /// for plain integer constants.
    pub fn push_literal(&mut self, literal: Literal) -> LiteralId {
        self.literals.push(literal)
    }

    /// Appends an instruction and records all its operands in the `users` map.
    ///
    /// # Immutability invariant
    ///
    /// Instructions are considered immutable after this call. If you alter the
    /// operands of an instruction after insertion the `users` map will be
    /// stale. Rewrite operands through
    /// [`Context::replace_all_uses_with`](crate::context::Context::replace_all_uses_with)
    /// instead.
    pub fn push_insn(&mut self, func: FunctionId, insn: Instruction<'str>) -> InstructionId {
        let args = insn.mnemonic().args();
        let call_target = insn.mnemonic().call_target();
        let local = self.functions[func].insns.push(insn);
        let id = InstructionId::new(func, local);
        // Record each operand's use in *this* function's map. For SSA operands
        // (instruction/param) the operand's own function is `func` by the SSA
        // ownership invariant; for shared operands (literal/varnode) the entry
        // lives with the using function, which is all any reader needs.
        for arg in args {
            self.functions[func].users.entry(arg).or_default().push(id);
        }
        // call_sites is the one remaining global reverse map (Stage 2): the
        // parallel driver (Stage 6) will rebuild it from a per-function
        // outgoing-call diff at check-in, at which point these direct writes move
        // there. Until then it is maintained inline.
        if let Some(target) = call_target {
            self.call_sites.entry(target).or_default().push(id);
        }
        id
    }

    /// Borrows the instruction `id`, routing through its owning function's arena.
    pub fn instruction(&self, id: InstructionId) -> &Instruction<'str> {
        &self.functions[id.func].insns[id.local]
    }

    /// Mutably borrows the instruction `id`.
    pub fn instruction_mut(&mut self, id: InstructionId) -> &mut Instruction<'str> {
        &mut self.functions[id.func].insns[id.local]
    }

    /// Borrows the basic block `id`.
    pub fn block(&self, id: BlockId) -> &BasicBlock<'str> {
        &self.functions[id.func].blocks[id.local]
    }

    /// Mutably borrows the basic block `id`.
    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock<'str> {
        &mut self.functions[id.func].blocks[id.local]
    }

    /// Borrows the block parameter `id`.
    pub fn block_param(&self, id: BlockParamId) -> &BlockParam<'str> {
        &self.functions[id.func].params[id.local]
    }

    /// Mutably borrows the block parameter `id`.
    pub fn block_param_mut(&mut self, id: BlockParamId) -> &mut BlockParam<'str> {
        &mut self.functions[id.func].params[id.local]
    }

    /// Borrows the CFG edge `id`.
    pub fn edge(&self, id: EdgeId) -> &EdgeData {
        &self.functions[id.func].edges[id.local]
    }

    /// Mutably borrows the CFG edge `id`.
    pub fn edge_mut(&mut self, id: EdgeId) -> &mut EdgeData {
        &mut self.functions[id.func].edges[id.local]
    }

    /// Returns the instructions that use `value` as an operand, read from
    /// `value`'s owning function. For an SSA def (instruction/param) that is the
    /// complete user set (all uses are intra-function). For a shared value
    /// (literal/bytes/varnode) there is no single owner, so this returns `&[]`;
    /// scan [`Context::functions`](crate::context::Context::functions) with
    /// [`Function::users_of`] to find a shared value's uses across functions.
    pub fn users_of(&self, value: ValueId) -> &[InstructionId] {
        match value.owning_function() {
            Some(func) => self.functions[func].users_of(value),
            None => &[],
        }
    }

    /// Returns the direct-call sites (instructions) targeting `callee`.
    pub fn call_sites_of(&self, callee: FunctionId) -> &[InstructionId] {
        self.call_sites
            .get(&callee)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Records a synthetic forward call-graph edge `caller → callee_addr` (see
    /// [`synthetic_callees`](Self::synthetic_callees)). Returns `true` if the
    /// edge was newly added, so callers can drive a fixpoint without spinning.
    pub fn add_synthetic_callee(&mut self, caller: FunctionId, callee_addr: u64) -> bool {
        self.synthetic_callees
            .entry(caller)
            .or_default()
            .insert(callee_addr)
    }

    /// Returns the synthetic callee entry addresses recorded for `caller`.
    pub fn synthetic_callees_of(&self, caller: FunctionId) -> impl Iterator<Item = u64> + '_ {
        self.synthetic_callees
            .get(&caller)
            .into_iter()
            .flatten()
            .copied()
    }

    /// Removes a set of dead instructions from the use-def map.
    ///
    /// For every instruction in `dead`, each of its operands will have all
    /// members of `dead` pruned from their user lists. Call this after
    /// removing dead instructions from their basic blocks.
    pub fn remove_instructions(&mut self, dead: &HashSet<InstructionId>) {
        // Collect the distinct operands and call targets referenced by any dead
        // instruction first, then prune each affected list exactly once. The
        // retain condition (`!dead.contains(..)`) is independent of which dead
        // instruction referenced the operand, so pruning per distinct operand
        // yields the same result as pruning per (dead, operand) pair — but a
        // value shared by K dead users has its list scanned once instead of K
        // times. This is the dominant cost when large blocks are cleared.
        // An operand's user entry lives in the *using* instruction's function
        // map, so the affected map is keyed by the dead instruction's own func.
        let mut affected_args: HashSet<(FunctionId, ValueId)> = HashSet::default();
        let mut affected_targets: HashSet<FunctionId> = HashSet::default();
        for &id in dead {
            let mnemonic = self.instruction(id).mnemonic();
            affected_args.extend(mnemonic.args().into_iter().map(|arg| (id.func, arg)));
            if let Some(target) = mnemonic.call_target() {
                affected_targets.insert(target);
            }
            // Tombstone it. Registry IDs are stable indices and cannot be reclaimed, so
            // the entry stays in the arena; marking it deleted keeps `Context::instructions`
            // (and any whole-program scan built on it) from yielding the stale operands it
            // still carries.
            self.instruction_mut(id).deleted = true;
        }
        for (func, arg) in affected_args {
            if let Some(users) = self.functions[func].users.get_mut(&arg) {
                users.retain(|u| !dead.contains(u));
            }
        }
        for target in affected_targets {
            if let Some(sites) = self.call_sites.get_mut(&target) {
                sites.retain(|s| !dead.contains(s));
            }
        }
    }

    pub fn push_block(&mut self, func: FunctionId, block: BasicBlock<'str>) -> BlockId {
        let local = self.functions[func].blocks.push(block);
        let id = BlockId::new(func, local);
        // A block is born owned by the function whose arena stores it.
        self.functions[func].roster.push(id);
        id
    }

    /// Removes `id` from its current owner's roster, if present. Storage (the
    /// arena slot) is untouched. Used by reattribution and block deletion.
    pub fn unroster_block(&mut self, id: BlockId) {
        let owner = self.block(id).parent;
        if let Some(f) = owner {
            self.functions[f].roster.retain(|&b| b != id);
        }
        // Defensive: also drop from the storage function's roster in case
        // ownership and storage diverged and both listed it.
        self.functions[id.func].roster.retain(|&b| b != id);
    }

    pub fn push_block_param(&mut self, func: FunctionId, param: BlockParam<'str>) -> BlockParamId {
        let local = self.functions[func].params.push(param);
        BlockParamId::new(func, local)
    }

    pub fn push_edge(&mut self, func: FunctionId, edge: EdgeData) -> EdgeId {
        let local = self.functions[func].edges.push(edge);
        EdgeId::new(func, local)
    }

    pub fn push_varnode(&mut self, varnode: Varnode<'str>) -> VarnodeId {
        self.varnodes.push(varnode)
    }

    pub fn push_function(&mut self, f: Function<'str>) -> FunctionId {
        self.functions.push(f)
    }
}
