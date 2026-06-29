use jstd::graph::analysis::compute_dominators;
use qcode::space::SpaceType;
#[cfg(test)]
use qcode::value::block::BlockRef;
use qcode::value::{
    BasicBlock, BlockId, BlockParam, BlockParamId, Function, FunctionId, Instruction, Value,
    ValueId, ValueRef, Varnode, VarnodeId,
    insn::{Branch, CBranch, InstructionId, InstructionRef, Load, Mnemonic, Range, Store, Zext},
};
use qcode::{
    builder::Builder,
    context::Context,
    value::{FunctionRef, block::BlockRef},
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::borrow::Cow;

use crate::AliasResult;
use crate::gvn::affine::{Numbering, precompute_forms};
use crate::stack::frame::{frame_offset, incoming_sp_param};

/// Returns `true` if any variables were promoted.
pub fn mem2reg(ctx: &mut Context, function_id: FunctionId, aliases: &AliasResult) -> bool {
    mem2reg_framed(ctx, function_id, aliases, None)
}

/// [`mem2reg`] with the incoming stack-pointer parameter (`@SP`) supplied, so
/// canonical `@SP ± N` stack slots are recognised alongside legacy `@stack_base`
/// literals. `sp_param` is `None` when the caller has no stack-pointer context
/// (most unit tests), leaving only the literal path active.
pub fn mem2reg_framed(
    ctx: &mut Context,
    function_id: FunctionId,
    aliases: &AliasResult,
    sp_param: Option<ValueId>,
) -> bool {
    Mem2Reg::new(ctx, function_id, aliases, sp_param).run()
}

struct Mem2Reg<'ctx, 'str> {
    ctx: &'ctx mut Context<'str>,
    function_id: FunctionId,
    root_id: Option<BlockId>,
    aliases: &'ctx AliasResult,
    /// Affine decomposition of every value, used to resolve `@SP ± N` slot
    /// offsets. Position-independent, computed once up front.
    numbering: Numbering,
    /// The incoming stack-pointer parameter, when known. Slots are `@SP ± N`
    /// relative to it; `None` falls back to `@stack_base`-literal recognition.
    sp_param: Option<ValueId>,
}

impl<'ctx, 'str> Mem2Reg<'ctx, 'str> {
    fn new(
        ctx: &'ctx mut Context<'str>,
        function_id: FunctionId,
        aliases: &'ctx AliasResult,
        sp_param: Option<ValueId>,
    ) -> Self {
        let root_id = Function::from_id(&*ctx, function_id).root().map(|b| b.id);
        let numbering = precompute_forms(&*ctx, function_id);
        Self {
            ctx,
            function_id,
            root_id,
            aliases,
            numbering,
            sp_param,
        }
    }

    /// The signed byte offset of stack-slot pointer `ptr` from the entry stack
    /// pointer `@SP` — the canonical slot key. `None` when there is no incoming
    /// stack-pointer param or `ptr` is not an `@SP ± N` slot.
    fn slot_offset(&self, ptr: ValueId) -> Option<i64> {
        let sp = self.sp_param?;
        frame_offset(self.ctx, &self.numbering, sp, ptr)
    }

    /// Whether `ptr` is `@SP`-derived but *not* a fixed slot offset — a
    /// dynamically indexed (`@SP + reg`) or realigned (`(@SP & -mask) + k`) stack
    /// pointer that may alias any slot. Only meaningful once the incoming
    /// stack-pointer param is known; the legacy `@stack_base` path relies on
    /// [`is_stack_typed`] instead.
    fn is_dynamic_sp_deref(&self, ptr: ValueId) -> bool {
        self.sp_param
            .is_some_and(|sp| self.numbering.affine_mentions(ptr, sp))
    }

    fn run(&mut self) -> bool {
        // `live_in_blocks` is an O(blocks × insns) fixpoint; cache it per var so
        // the collection and block-param phases share a single computation.
        let mut live_in_cache: HashMap<ValueId, HashSet<BlockId>> = HashMap::default();

        let Promotable {
            vars,
            sizes,
            sliced,
        } = self.collect_promotable_vars(&mut live_in_cache);
        if vars.is_empty() {
            return false;
        }

        let root_id = self.root_id();
        let dom = compute_dominators(&*self.ctx, root_id);
        let frontier = dom.dominator_frontier().clone();
        let InsertedBlockParams {
            by_block: var_params,
            changed,
            excluded,
        } = self.insert_block_params(&vars, &sizes, &sliced, &frontier, &mut live_in_cache);

        // Drop vars whose promotion was declined (implicit-edge join); their
        // memory accesses stay in place, so renaming and store removal must not
        // touch them.
        let vars: HashSet<ValueId> = if excluded.is_empty() {
            vars
        } else {
            vars.difference(&excluded).copied().collect()
        };
        if vars.is_empty() {
            return changed;
        }

        let register_clobbers =
            register_clobber_index(self.ctx, self.function_id, &vars, self.aliases);

        let mut state = RenameState::new(&var_params, &vars, &sliced, register_clobbers, changed);
        self.decide_values_start_from(root_id, &mut state);

        let RenameState {
            consumed_stores,
            dead_stores,
            preserved_stores,
            changed,
            ..
        } = state;
        changed
            | self.remove_promoted_stores(&vars, &consumed_stores, &dead_stores, &preserved_stores)
    }

    fn root_id(&self) -> BlockId {
        self.root_id.expect("function has no root")
    }

    fn resize_forwarded_load_value(
        &mut self,
        block: BlockId,
        before: InstructionId,
        value: ValueId,
        load_size: usize,
    ) -> ValueId {
        let value_size = ValueRef::new(value, self.ctx).size();
        if value_size == load_size {
            return value;
        }

        if let ValueId::Literal(id) = value {
            let literal = self.ctx.values.literals[id].clone();
            if literal.symbolic.is_none() {
                return self.ctx.get_const(literal.value, load_size).id();
            }
        }

        let mnemonic = if value_size < load_size {
            Mnemonic::Zext(Zext {
                src: value,
                size: load_size,
            })
        } else {
            Mnemonic::Range(Range {
                src: value,
                start: 0,
                size: load_size,
            })
        };
        let new_id = InstructionRef::from_mnemonic(self.ctx, mnemonic, load_size).id;
        BasicBlock::from_id_mut(self.ctx, block).insert_insn_before(before, new_id);
        ValueId::Instruction(new_id)
    }
}

/// A total, hash-independent order over `ValueId`, used to canonicalize the order
/// in which mem2reg promotes variables (and thus the order of the block params it
/// creates). Keyed by `(variant, inner index)`; both halves are stable across runs.
fn value_id_order_key(v: ValueId) -> (u8, usize) {
    match v {
        ValueId::Literal(id) => (0, id.into()),
        ValueId::Instruction(id) => (1, id.into()),
        ValueId::BasicBlock(id) => (2, id.into()),
        ValueId::BlockParam(id) => (3, id.into()),
        ValueId::Varnode(id) => (4, id.into()),
        ValueId::Function(id) => (5, id.into()),
        // `ValueId` is `#[non_exhaustive]`; keep any future variant last but stable.
        _ => (u8::MAX, 0),
    }
}
/// Whether `function_id` performs any load or store through a *computed stack
/// frame pointer* — an `@SP`-rooted value whose offset from the incoming stack
/// pointer is not a fixed constant (`@SP + reg`, or a realigned
/// `(@SP & -mask) + k`). A computed RAM pointer loaded from a stack-passed
/// argument is not enough: that reads behind the argument value, not arbitrary
/// bytes of the caller's frame.
pub(crate) fn has_dynamic_stack_pointer_deref(
    ctx: &Context,
    function_id: FunctionId,
    stack_ptr: VarnodeId,
) -> bool {
    let Some(sp) = incoming_sp_param(ctx, function_id, stack_ptr) else {
        return false;
    };
    let numbering = precompute_forms(ctx, function_id);
    for block in Function::from_id(ctx, function_id).blocks() {
        for insn in block.iter() {
            let Some(access) = MemoryAccess::from_mnemonic(insn.mnemonic()) else {
                continue;
            };
            if frame_offset(ctx, &numbering, sp, access.ptr).is_none()
                && numbering.affine_mentions(access.ptr, sp)
            {
                return true;
            }
        }
    }
    false
}

fn register_varnode(ctx: &Context, value: ValueId) -> Option<VarnodeId> {
    let ValueId::Varnode(vn_id) = value else {
        return None;
    };
    matches!(Varnode::from_id(ctx, vn_id).space().ty, SpaceType::Register).then_some(vn_id)
}

fn wider_register_store_contains(ctx: &Context, store: ValueId, var: ValueId) -> bool {
    let (Some(store), Some(var)) = (register_varnode(ctx, store), register_varnode(ctx, var))
    else {
        return false;
    };
    let (store, var) = (Varnode::from_id(ctx, store), Varnode::from_id(ctx, var));
    if store.space().id != var.space().id || store.size() <= var.size() {
        return false;
    }
    let store_start = store.address() as i128;
    let var_start = var.address() as i128;
    let store_end = store_start + store.size() as i128;
    let var_end = var_start + var.size() as i128;
    store_start <= var_start && var_end <= store_end
}

/// Whether register store `store` fully covers narrower register var `var` and
/// shares its start address — so `var`'s bytes are exactly the low `var.size`
/// bytes of the stored value (e.g. `AL`/`AX` within an `EAX` store). For such a
/// low-aligned containment the stored value can be *sliced* into `var` with a
/// plain low-byte truncation (`Range { start: 0 }`), the resize the renamer
/// already applies when forwarding. A containment at a non-zero offset (e.g.
/// `AH`) would need a shift, which this pass does not synthesize, so it is
/// excluded here and left to the GVN memory pass.
fn register_store_low_aligned_contains(ctx: &Context, store: ValueId, var: ValueId) -> bool {
    let (Some(store), Some(var)) = (register_varnode(ctx, store), register_varnode(ctx, var))
    else {
        return false;
    };
    let (store, var) = (Varnode::from_id(ctx, store), Varnode::from_id(ctx, var));
    store.space().id == var.space().id
        && store.size() > var.size()
        && store.address() == var.address()
}

/// Whether the call terminating `block` (if any) clobbers register `var`.
///
/// A direct call clobbers the registers in its target's recorded clobber set,
/// overlap-aware so a callee writing RAX clobbers a caller's EAX read. A
/// `CallInd` has an unknown target and conservatively clobbers every register; a
/// target with no recorded clobber set is likewise treated conservatively —
/// *unless* it is [`externally_resolved`](Function::is_externally_resolved), in
/// which case its clobber set is exact (a prototype-derived volatile set, often
/// empty) and an absent/empty set clobbers nothing rather than everything.
/// Non-register vars (stack slots) are never clobbered by a call.
fn call_clobbers_var(ctx: &Context, block: BlockId, var: ValueId, aliases: &AliasResult) -> bool {
    if register_varnode(ctx, var).is_none() {
        return false;
    };
    let term = BasicBlock::from_id(ctx, block)
        .iter()
        .last()
        .map(|i| i.mnemonic().clone());
    match term {
        Some(Mnemonic::CallInd(_)) => true,
        Some(Mnemonic::Call(call)) => {
            let callee = Function::from_id(ctx, call.target);
            let resolved = callee.is_externally_resolved();
            let clobbered: Option<Vec<VarnodeId>> =
                callee.clobbered_regs().map(<[VarnodeId]>::to_vec);
            match clobbered {
                Some(clobbered) => clobbered
                    .iter()
                    .any(|&c| aliases.may_alias(ctx, ValueId::Varnode(c), var)),
                // A resolved callee with no recorded set clobbers nothing; an
                // unresolved one is unknown, so conservatively clobbers all.
                None => !resolved,
            }
        }
        _ => false,
    }
}

/// For each register store-pointer in the function, the set of promoted register
/// vars it may clobber (overlapping byte ranges in the same space).
///
/// Overlap is decided by [`AliasResult::may_alias`]. Note `simple` merges varnodes
/// into equivalence classes by a transitive union-find sweep, so disjoint
/// sub-registers that share a parent (e.g. `AL` and `AH`, both joined via `AX`)
/// are reported as may-alias even though their bytes don't overlap. That only ever
/// *over*-clobbers — it forces an extra reload / preserves an extra store — and can
/// never yield wrong SSA, so the imprecision is intentional and safe.
fn register_clobber_index(
    ctx: &Context,
    function_id: FunctionId,
    vars: &HashSet<ValueId>,
    aliases: &AliasResult,
) -> HashMap<ValueId, Vec<ValueId>> {
    let promoted_register_vars = vars
        .iter()
        .copied()
        .filter(|&var| register_varnode(ctx, var).is_some())
        .collect::<Vec<_>>();
    if promoted_register_vars.is_empty() {
        return HashMap::default();
    }

    let mut store_ptrs = HashSet::default();
    for block in Function::from_id(ctx, function_id).blocks() {
        for insn in block.iter() {
            if let Mnemonic::Store(Store { ptr, .. }) = insn.mnemonic()
                && register_varnode(ctx, *ptr).is_some()
            {
                store_ptrs.insert(*ptr);
            }
        }
    }

    store_ptrs
        .into_iter()
        .filter_map(|store_ptr| {
            let clobbered = promoted_register_vars
                .iter()
                .copied()
                .filter(|&var| var != store_ptr && aliases.may_alias(ctx, store_ptr, var))
                .collect::<Vec<_>>();
            (!clobbered.is_empty()).then_some((store_ptr, clobbered))
        })
        .collect()
}

/// The set of promotable locations, plus the access size for each promoted
/// stack slot (varnode sizes come from the varnode itself).
struct Promotable {
    vars: HashSet<ValueId>,
    sizes: HashMap<ValueId, usize>,
    /// Promoted register vars that have *no store to themselves* but are fully,
    /// low-alignedly covered by a wider register store (e.g. `AL` read in a loop
    /// while the body writes `EAX`). Their reaching value is sliced out of the
    /// wider store's source. Threaded into liveness, phi placement and renaming
    /// so the covering store acts as a definition of the narrow var.
    sliced: HashSet<ValueId>,
}

#[derive(Clone, Copy)]
enum MemoryAccessKind {
    Load,
    Store { src: ValueId },
}

#[derive(Clone, Copy)]
struct MemoryAccess {
    ptr: ValueId,
    size: usize,
    kind: MemoryAccessKind,
}

impl MemoryAccess {
    fn from_mnemonic(mnemonic: &Mnemonic) -> Option<Self> {
        match mnemonic {
            Mnemonic::Store(Store { ptr, src, size, .. }) => Some(Self {
                ptr: *ptr,
                size: *size,
                kind: MemoryAccessKind::Store { src: *src },
            }),
            Mnemonic::Load(Load { ptr, size, .. }) => Some(Self {
                ptr: *ptr,
                size: *size,
                kind: MemoryAccessKind::Load,
            }),
            _ => None,
        }
    }

    fn is_store(&self) -> bool {
        matches!(self.kind, MemoryAccessKind::Store { .. })
    }
}

/// A byte range of a stack access, in *frame-offset* coordinates (signed bytes
/// from the entry stack pointer). Offsets unify the `@SP ± N` and legacy
/// `@stack_base ± N` representations; overlap is invariant under the constant
/// shift between an absolute address and its offset.
#[derive(Clone, Copy, PartialEq, Eq)]
struct StackAccessRange {
    start: i64,
    end: i64,
}

impl StackAccessRange {
    fn new(start: i64, size: usize) -> Self {
        Self {
            start,
            end: start.saturating_add(size as i64),
        }
    }

    fn overlaps_slot(&self, slot_start: i64, slot_size: usize) -> bool {
        let slot_end = slot_start + slot_size as i64;
        slot_start < self.end && self.start < slot_end
    }
}

/// The promotion access size for stack slot `var`, or `None` if it fails the
/// single-size or no-overlap guard: a slot accessed at conflicting sizes, or
/// whose byte range is touched by any *other* stack access, cannot be promoted.
fn promotable_stack_slot_size(
    offset: i64,
    var: ValueId,
    stack_size: &HashMap<ValueId, usize>,
    stack_size_conflict: &HashSet<ValueId>,
    stack_intervals: &[StackAccessRange],
) -> Option<usize> {
    if stack_size_conflict.contains(&var) {
        return None;
    }
    let size = stack_size[&var];
    let addr = offset;
    let own = StackAccessRange::new(addr, size);
    let overlapped = stack_intervals
        .iter()
        .any(|access| *access != own && access.overlaps_slot(addr, size));
    (!overlapped).then_some(size)
}

type BlockParamsByVar = HashMap<ValueId, BlockParamId>;
type BlockParamAssignments = HashMap<BlockId, BlockParamsByVar>;

struct InsertedBlockParams {
    by_block: BlockParamAssignments,
    changed: bool,
    /// Vars declined for promotion because a join they would parameterize is fed
    /// by an implicit (argument-less) edge — see [`Mem2Reg::insert_block_params`].
    /// The caller must drop these from the rename set so their loads/stores are
    /// left in place.
    excluded: HashSet<ValueId>,
}

impl Mem2Reg<'_, '_> {
    fn block_param_name_for_var(&self, var: ValueId) -> Option<String> {
        match var {
            ValueId::Varnode(varnode_id) => Varnode::from_id(self.ctx, varnode_id)
                .name()
                .map(|n| n.to_owned()),
            _ => self.slot_offset(var).map(|off| format!("stack_{off:x}")),
        }
    }

    /// An existing param on `block_id` previously created to promote `var`,
    /// identified by its recorded [`origin`](qcode::value::BlockParam::origin)
    /// rather than its display name. `origin` is a stable cross-run identity, so
    /// this finds the param even for varnodes that have no name (the name-based
    /// match used to miss them, re-pushing a duplicate on every re-run).
    ///
    /// For an `@SP ± N` stack slot the per-run representative pointer is a fresh
    /// `ValueId` each round (the canonicalizer re-materializes it), so an exact
    /// `origin == var` miss falls back to matching by frame offset — the stable
    /// slot identity across runs.
    fn existing_param_for_var(
        &self,
        block_id: BlockId,
        size: usize,
        var: ValueId,
    ) -> Option<BlockParamId> {
        let var_offset = self.slot_offset(var);
        BasicBlock::from_id(self.ctx, block_id)
            .params()
            .find(|param| {
                param.size() == size
                    && (param.origin() == Some(var)
                        || (var_offset.is_some()
                            && param.origin().and_then(|o| self.slot_offset(o)) == var_offset))
            })
            .map(|param| param.id)
    }

    fn get_or_insert_param_for_var(
        &mut self,
        block_id: BlockId,
        size: usize,
        var: ValueId,
        name: Option<&str>,
    ) -> (BlockParamId, bool) {
        if let Some(param_id) = self.existing_param_for_var(block_id, size, var) {
            return (param_id, false);
        }

        let param_id = BasicBlock::from_id_mut(self.ctx, block_id)
            .push_param(size)
            .id;
        self.ctx.values.block_params[param_id].origin = Some(var);
        // Carry a global varnode type override (e.g. the `FS_OFFSET` segment base
        // typed `PtrTo<TEB>` by `windows_teb_seed`) onto the promoted param, so the
        // ambient register's richer type survives mem2reg instead of decaying to
        // the default `Int(size)`. Width matches by construction (the override is
        // installed with the varnode's own width).
        if let ValueId::Varnode(_) = var
            && let Some(ty) = self.ctx.stored_type_of(var)
        {
            self.ctx.values.block_params[param_id].type_id = ty;
        }
        if let Some(name) = name {
            self.ctx.values.block_params[param_id].name = Some(Cow::Owned(name.to_owned()));
        }
        (param_id, true)
    }

    fn collect_promotable_vars(
        &self,
        live_in_cache: &mut HashMap<ValueId, HashSet<BlockId>>,
    ) -> Promotable {
        let mut stored = HashSet::default();
        let mut loaded = HashSet::default();
        let mut store_counts: HashMap<ValueId, usize> = HashMap::default();
        let mut register_stores: HashSet<ValueId> = HashSet::default();

        // Stack-slot bookkeeping, keyed by the slot literal `ValueId`.
        let mut stack_stored: HashSet<ValueId> = HashSet::default();
        let mut stack_loaded: HashSet<ValueId> = HashSet::default();
        let mut stack_size: HashMap<ValueId, usize> = HashMap::default();
        let mut stack_size_conflict: HashSet<ValueId> = HashSet::default();
        let mut stack_intervals: Vec<StackAccessRange> = Vec::new();
        // A stack-typed pointer we could not resolve to a fixed slot disables all
        // stack promotion: it may alias any slot. The same applies when this
        // function hands a pointer into its own frame to a callee that may read it
        // unboundedly (`frame_escapes_to_unbounded`, a fact seeded by the driver
        // from the previous checkpoint+replay round): that callee may have written
        // any slot, so none may be promoted across it.
        let mut dynamic_stack =
            Function::from_id(self.ctx, self.function_id).frame_escapes_to_unbounded();
        // Locations disqualified because an access mis-sizes the stored value:
        //   - a store whose source is narrower than the access (e.g. the
        //     `MOV ESI, imm32` lift, an 8-byte RSI store of a 4-byte literal); or
        //   - a varnode access whose width differs from the varnode's own width
        //     (e.g. a 4-byte load of an 8-byte-stored register).
        // Promoting either would forward a wrong-width value.
        let mut mixed_width: HashSet<ValueId> = HashSet::default();

        let mut sliced: HashSet<ValueId> = HashSet::default();

        for block in Function::from_id(self.ctx, self.function_id).blocks() {
            for insn in block.iter() {
                let Some(access) = MemoryAccess::from_mnemonic(insn.mnemonic()) else {
                    continue;
                };
                // A store whose source width differs from the access width would
                // forward a mis-sized value — *unless* the source is a constant
                // literal, which the renamer resizes (zero-extends/truncates) to
                // each consumer's width on both the load and the phi-edge paths.
                // This is the `MOV EAX, imm32` zero-extend-into-RAX lift: an 8-byte
                // `store(RAX, imm:4)`. The store/forward semantics match (the
                // emulator zero-fills the wider store too), so a literal source
                // stays promotable; a non-literal width mismatch still disqualifies.
                if let MemoryAccessKind::Store { src } = access.kind
                    && ValueRef::new(src, self.ctx).size() != access.size
                    && !matches!(src, ValueId::Literal(_))
                {
                    mixed_width.insert(access.ptr);
                }

                if let ValueId::Varnode(vn_id) = access.ptr {
                    if access.size != Varnode::from_id(self.ctx, vn_id).size() {
                        mixed_width.insert(access.ptr);
                    }
                    if access.is_store() {
                        stored.insert(access.ptr);
                        *store_counts.entry(access.ptr).or_insert(0) += 1;
                        if matches!(
                            Varnode::from_id(self.ctx, vn_id).space().ty,
                            SpaceType::Register
                        ) {
                            register_stores.insert(access.ptr);
                        }
                    } else {
                        loaded.insert(access.ptr);
                    }
                } else if let Some(off) = self.slot_offset(access.ptr) {
                    stack_intervals.push(StackAccessRange::new(off, access.size));
                    match stack_size.get(&access.ptr) {
                        Some(&prev) if prev != access.size => {
                            stack_size_conflict.insert(access.ptr);
                        }
                        _ => {
                            stack_size.insert(access.ptr, access.size);
                        }
                    }
                    if access.is_store() {
                        stack_stored.insert(access.ptr);
                    } else {
                        stack_loaded.insert(access.ptr);
                    }
                } else if self.is_dynamic_sp_deref(access.ptr) {
                    dynamic_stack = true;
                }
            }
        }

        // Non-register varnodes: promote only when both stored and loaded (standard mem2reg).
        let mut vars: HashSet<ValueId> = stored.intersection(&loaded).copied().collect();

        // Register-space varnodes need special treatment at function boundaries:
        //   - Load-only: live-in inputs (e.g. EDI/ESI args) → add root block param.
        //   - Both stored and loaded: already included above; re-inserting is harmless.
        //   - Store-only with 2+ stores: detect dead overwrites (e.g. the intermediate
        //     RAX write before the final return-value write). A single store is always
        //     the live-out value; including it would keep `vars` non-empty across passes
        //     and prevent the promote-stack fixpoint from converging.
        for &var in stored.union(&loaded) {
            if let ValueId::Varnode(vn_id) = var
                && matches!(
                    Varnode::from_id(self.ctx, vn_id).space().ty,
                    SpaceType::Register
                )
            {
                let is_store_only = stored.contains(&var) && !loaded.contains(&var);
                let is_load_only = loaded.contains(&var) && !stored.contains(&var);
                let count = store_counts.get(&var).copied().unwrap_or(0);
                if is_load_only {
                    // Promote a load-only register only if it is a genuine
                    // function input — live-in to the root under the clobber-aware
                    // liveness. A register read reachable only through a clobbering
                    // call is the call's output, not an input; leaving it as a
                    // register load lets emulation read the clobbered value rather
                    // than a spurious entry parameter.
                    //
                    // Be conservative around overlapping register writes. A full
                    // register seed such as `store(RDX, @RDX)` followed by a
                    // sub-register read `load(EDX)` makes `EDX` look load-only when
                    // vars are keyed by exact varnode id. Promoting that apparent
                    // live-in creates a transient, usually-unused root param; for a
                    // `pure_reg` callee, that mutates the by-value call interface
                    // outside the lockstep helpers. Leave such loads in memory SSA
                    // so the overlap-aware GVN memory pass can forward/slice the
                    // dominating store instead.
                    let containing: Vec<ValueId> = register_stores
                        .iter()
                        .copied()
                        .filter(|&store| {
                            store != var && wider_register_store_contains(self.ctx, store, var)
                        })
                        .collect();
                    if containing.is_empty() {
                        if self.root_id.is_some_and(|r| {
                            self.live_in_blocks_cached(var, &sliced, live_in_cache)
                                .contains(&r)
                        }) {
                            vars.insert(var);
                        }
                    } else if containing
                        .iter()
                        .all(|&store| register_store_low_aligned_contains(self.ctx, store, var))
                    {
                        // Every covering store shares the var's start address, so the
                        // narrow value is the low bytes of the stored value — a slice
                        // the renamer can synthesize. Promote it as a *sliced* var:
                        // the covering store acts as its definition (threaded through
                        // liveness, phi placement and renaming). A var live-in to the
                        // root (no covering store dominates the read) is still dropped
                        // by the root-live-in filter below and stays a plain load.
                        vars.insert(var);
                        sliced.insert(var);
                    }
                    // A covering store at a non-zero offset (e.g. `AH`) is left in
                    // memory SSA for the overlap-aware GVN memory pass, as before.
                } else if !is_store_only || count >= 2 {
                    vars.insert(var);
                }
            }
        }

        // Stack slots: a local frame slot does not escape, so promote any slot that
        // is both stored and loaded at a single consistent size and whose byte range
        // is touched by no *other* stack access. Disabled entirely when a dynamic
        // stack pointer is present, since it may alias any slot.
        let mut sizes: HashMap<ValueId, usize> = HashMap::default();
        if !dynamic_stack {
            for &var in stack_stored.intersection(&stack_loaded) {
                let Some(offset) = self.slot_offset(var) else {
                    continue;
                };
                if let Some(size) = promotable_stack_slot_size(
                    offset,
                    var,
                    &stack_size,
                    &stack_size_conflict,
                    &stack_intervals,
                ) {
                    vars.insert(var);
                    sizes.insert(var, size);
                }
            }

            // Incoming stack arguments — a caller-frame slot (offset >= 0) that is
            // *loaded but never stored* — are deliberately NOT promoted to root
            // params here. Minting a stack-input param forces a separate
            // interprocedural backfill (the former `argpromote_stack`) to reconnect
            // the caller side, and threading the stack pointer through that channel
            // was a recurring source of frame-epilogue correctness bugs. Instead we
            // leave these slots as plain `load(@stack_base + offset)` memory reads
            // and let the post-lowering memory channel (`calls::argpromote`)
            // functionalize them as ordinary by-value pointer arguments, the same
            // way it handles any other caller-frame dereference. Local frame slots
            // (offset < 0) are still promoted by the stored+loaded path above.
        }

        // Drop any location fed by a narrower-than-access store: forwarding its value
        // into a wider load would mis-size the result.
        vars.retain(|var| !mixed_width.contains(var));
        sizes.retain(|var, _| vars.contains(var));

        // mem2reg must never produce a root block param. A root param is a positional
        // call argument, i.e. part of the by-value call interface that `argpromote`
        // owns exclusively — mem2reg minting one (for a var live-in to the function)
        // corrupts the `param[i] ↔ Call.args[i]` lockstep argpromote relies on and
        // crashes it. So drop *every* var that is live-in to the root from the promotion
        // set: the only thing that would turn such a var into a root param is the
        // root-param path in `insert_block_params`, and with no live-in var promoted it
        // never fires (and is asserted away below). These inputs stay as plain loads —
        // correct (emulation reads the real incoming value); the legacy register-ABI
        // summary still recognizes register inputs by raw-IR liveness
        // (`compute_input_regs`), independent of any mem2reg param. A var written before
        // it is read (RMW / local scratch) is not live-in and still promotes normally.
        let root_id = self.root_id();
        let root_live_in: Vec<ValueId> = vars
            .iter()
            .copied()
            .filter(|&var| {
                self.live_in_blocks_cached(var, &sliced, live_in_cache)
                    .contains(&root_id)
            })
            .collect();
        for var in root_live_in {
            vars.remove(&var);
            sizes.remove(&var);
            sliced.remove(&var);
        }

        Promotable {
            vars,
            sizes,
            sliced,
        }
    }

    fn insert_block_params(
        &mut self,
        vars: &HashSet<ValueId>,
        sizes: &HashMap<ValueId, usize>,
        sliced: &HashSet<ValueId>,
        frontier: &HashMap<BlockId, HashSet<BlockId>>,
        live_in_cache: &mut HashMap<ValueId, HashSet<BlockId>>,
    ) -> InsertedBlockParams {
        let mut var_params: BlockParamAssignments = HashMap::default();
        let mut changed = false;
        let mut excluded: HashSet<ValueId> = HashSet::default();

        // `vars` is a `HashSet`, so iterating it directly promotes variables in an
        // order seeded by their value ids' hashes. That order decides block-param
        // order and the value ids minted for those params, so it must be canonical:
        // otherwise equivalent runs emit the same IR with params (e.g. `arg_stack_8`
        // vs `arg_stack_16`) swapped. Sort into a stable, hash-independent order.
        let mut ordered_vars: Vec<ValueId> = vars.iter().copied().collect();
        ordered_vars.sort_unstable_by_key(|&v| value_id_order_key(v));

        for var in ordered_vars {
            // Block-param width: varnodes carry their own size; stack slots use the
            // (consistent) access size recorded during collection.
            let size = match var {
                ValueId::Varnode(varnode_id) => Varnode::from_id(self.ctx, varnode_id).size(),
                _ => match sizes.get(&var) {
                    Some(&size) => size,
                    None => continue,
                },
            };
            let var_name = self.block_param_name_for_var(var);

            let live_in = self.live_in_blocks_cached(var, sliced, live_in_cache);
            let phi_positions = {
                let function = Function::from_id(self.ctx, self.function_id);
                find_phi_insert_positions(self.ctx, var, &function, sliced, frontier, &live_in)
            };

            // A block param is only meaningful if every incoming edge can supply
            // its argument. Branch/CBranch edges are wired by `merge_branch_args`,
            // but a `Call`/`CallInd` fall-through or a `BranchInd` jump-table edge
            // carries no argument list. If any join we would parameterize is fed by
            // such an implicit edge, that param would be left unbound on it. Rather
            // than emit malformed IR — or, during renaming, forward a wrong value
            // (register var) or panic (stack slot) — decline to promote this var at
            // all, leaving its memory accesses in place. Conservative but correct;
            // this is rare (it needs a jump-table/call successor that is also a
            // multi-predecessor join for the var).
            if phi_positions
                .iter()
                .any(|&b| self.has_implicit_edge_predecessor(b))
            {
                excluded.insert(var);
                continue;
            }

            for block_id in phi_positions {
                let (param_id, inserted) =
                    self.get_or_insert_param_for_var(block_id, size, var, var_name.as_deref());
                changed |= inserted;
                var_params
                    .entry(block_id)
                    .or_default()
                    .insert(var, param_id);
            }

            // mem2reg never produces a root block param: the call interface (root
            // params ↔ Call.args) is owned exclusively by argpromote. Every var live-in
            // to the function root was dropped from the promotion set in
            // `collect_promotable_vars`, so this point is unreachable for a root-live-in
            // var — a register/stack *input* stays a plain load. (Non-root join params —
            // phis — are still created above.)
            debug_assert!(
                !live_in.contains(&self.root_id()),
                "mem2reg must not promote a root-live-in var ({var:?} in {}): the call \
                 interface is argpromote's",
                Function::from_id(self.ctx, self.function_id).name(),
            );
        }

        InsertedBlockParams {
            by_block: var_params,
            changed,
            excluded,
        }
    }

    /// True if `block` has a predecessor reaching it through an edge that cannot
    /// carry block-param arguments. Only `Branch`/`CBranch` terminators wire args
    /// (via [`merge_branch_args`](Self::merge_branch_args)); a `Call`/`CallInd`
    /// fall-through or a `BranchInd` jump-table edge is a bare CFG edge. A
    /// predecessor with no terminator is treated as implicit (conservative).
    fn has_implicit_edge_predecessor(&self, block: BlockId) -> bool {
        BasicBlock::from_id(self.ctx, block)
            .predecessors()
            .any(|(_, pred)| {
                let term = BasicBlock::from_id(self.ctx, pred)
                    .iter()
                    .last()
                    .map(|i| i.mnemonic().clone());
                !matches!(term, Some(Mnemonic::Branch(_)) | Some(Mnemonic::CBranch(_)))
            })
    }

    /// Memoized [`live_in_blocks`]. The returned set is cloned from the cache so
    /// the caller may freely take further `&mut self` borrows; the clone is cheap
    /// relative to recomputing the liveness fixpoint.
    fn live_in_blocks_cached(
        &self,
        var: ValueId,
        sliced: &HashSet<ValueId>,
        cache: &mut HashMap<ValueId, HashSet<BlockId>>,
    ) -> HashSet<BlockId> {
        cache
            .entry(var)
            .or_insert_with(|| {
                live_in_blocks(self.ctx, self.function_id, var, sliced, self.aliases)
            })
            .clone()
    }
}

/// Blocks into which `var` is live-in.
///
/// A call that clobbers `var` counts as a definition of it: a read of `var`
/// reachable only through that call is the call's *output*, not a value flowing
/// in from the function entry. This keeps a post-call register read (e.g. a
/// caller reading the callee's `RAX`/`EAX` result) from being treated as a
/// function input and promoted to a spurious root parameter.
fn live_in_blocks(
    ctx: &Context,
    function_id: FunctionId,
    var: ValueId,
    sliced: &HashSet<ValueId>,
    aliases: &AliasResult,
) -> HashSet<BlockId> {
    let mut upward_exposed = HashSet::default();
    let mut defined = HashSet::default();
    let var_is_sliced = sliced.contains(&var);

    for block in Function::from_id(ctx, function_id).blocks() {
        let block_id = block.id;
        let mut has_def = false;

        for insn in block.iter() {
            match insn.mnemonic() {
                // A store to the var itself, or — for a sliced var — a wider store
                // that low-alignedly covers it, fully defines the var's bytes.
                Mnemonic::Store(Store { ptr, .. })
                    if *ptr == var
                        || (var_is_sliced
                            && register_store_low_aligned_contains(ctx, *ptr, var)) =>
                {
                    has_def = true;
                }
                Mnemonic::Load(Load { ptr, .. }) if *ptr == var && !has_def => {
                    upward_exposed.insert(block_id);
                }
                _ => {}
            }
        }

        if call_clobbers_var(ctx, block_id, var, aliases) {
            has_def = true;
        }

        if has_def {
            defined.insert(block_id);
        }
    }

    let mut live_in = upward_exposed;
    let mut changed = true;
    while changed {
        changed = false;
        for block in Function::from_id(ctx, function_id).blocks() {
            if defined.contains(&block.id) || live_in.contains(&block.id) {
                continue;
            }
            if block
                .successors()
                .any(|(_, successor)| live_in.contains(&successor))
            {
                live_in.insert(block.id);
                changed = true;
            }
        }
    }

    live_in
}

struct BranchEdge<'a> {
    target: BlockId,
    source_block: BlockId,
    branch_insn: InstructionId,
    existing_args: &'a [ValueId],
}

impl Mem2Reg<'_, '_> {
    /// Computes the full argument list for a branch into `target`, by index.
    ///
    /// `mem2reg` runs repeatedly (interleaved with constant-folding), and each run
    /// only knows the variables *it* promoted (`var_params`). Block params, however,
    /// accumulate across runs. So this run fills only the argument slots for the
    /// params it manages and preserves `existing` arguments for every other slot —
    /// otherwise a later run would clobber the (correct) arguments an earlier run
    /// wired for params it no longer tracks, leaving params without matching args.
    ///
    /// Returns the merged arguments when every slot is resolved; otherwise returns
    /// `existing` unchanged rather than emit a partial (mis-aligned) argument list.
    fn merge_branch_args(
        &mut self,
        edge: BranchEdge<'_>,
        state: &mut RenameState<'_>,
    ) -> Vec<ValueId> {
        let param_count = BasicBlock::from_id(self.ctx, edge.target).params().count();

        // Seed every slot with the argument already on the branch (from a prior run).
        let mut slots: Vec<Option<ValueId>> = (0..param_count)
            .map(|i| edge.existing_args.get(i).copied())
            .collect();

        let params = state
            .var_params
            .get(&edge.target)
            .map(|params| {
                params
                    .iter()
                    .map(|(&var, &param_id)| (var, param_id))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        for (var, param_id) in params {
            let index = self.ctx.values.block_params[param_id].index;
            let (val, store_insn) = match decide_variable_value(var, &state.frames) {
                Some(FrameEntry::Defined(reaching)) => (reaching.value, reaching.store_insn),
                Some(FrameEntry::Clobbered) | None if self.is_register_var(var) => {
                    let ValueId::Varnode(vn_id) = var else {
                        unreachable!("is_register_var only matches varnodes");
                    };
                    let existing = slots.get(index).and_then(|slot| *slot);
                    let value = existing
                        .filter(|&value| self.is_load_from_var(value, var, edge.source_block))
                        .unwrap_or_else(|| {
                            self.load_register_before_branch(
                                edge.source_block,
                                edge.branch_insn,
                                vn_id,
                            )
                        });
                    (value, None)
                }
                Some(FrameEntry::Clobbered) | None => panic!(
                    "variable {:?} undefined at branch to parameterized block",
                    var
                ),
            };
            if let Some(id) = store_insn {
                state.consumed_stores.insert(id);
            }
            // The block param has the var's width, but a reaching definition may be
            // narrower or wider (a literal stored through a wider access — the
            // `MOV EAX, imm32` zero-extend-into-RAX idiom — or a truncating store).
            // The emulator binds branch args to params without resizing, so the
            // edge value must already match the param width. Resize before the
            // branch, mirroring the load-forwarding path. A no-op when widths match.
            let param_size = BlockParam::from_id(self.ctx, param_id).size();
            let val = self.resize_forwarded_load_value(
                edge.source_block,
                edge.branch_insn,
                val,
                param_size,
            );
            if let Some(slot) = slots.get_mut(index) {
                *slot = Some(val);
            }
        }

        // Only emit a new list when every slot is resolved; a partial list would
        // misalign args with params.
        slots
            .into_iter()
            .collect::<Option<Vec<ValueId>>>()
            .unwrap_or_else(|| edge.existing_args.to_vec())
    }

    fn load_register_before_branch(
        &mut self,
        branch_block: BlockId,
        branch_insn: InstructionId,
        vn_id: VarnodeId,
    ) -> ValueId {
        let (space, size) = {
            let vn = Varnode::from_id(self.ctx, vn_id);
            (vn.space().id, vn.size())
        };
        let mut builder = Builder::from_block(BasicBlock::from_id_mut(self.ctx, branch_block));
        builder.set_insert_point_before(branch_insn);
        builder
            .push_load::<false>(ValueId::Varnode(vn_id), size, space)
            .id()
    }

    /// Whether `value` is a `Load` of `var` that already sits in `block` — the
    /// shape produced by a prior run's [`Self::load_register_before_branch`]. The
    /// block check ensures we only reuse a reload that actually dominates the
    /// branch we are wiring; a load from `var` living elsewhere must not be
    /// silently adopted as the edge argument.
    fn is_load_from_var(&self, value: ValueId, var: ValueId, block: BlockId) -> bool {
        let ValueId::Instruction(insn_id) = value else {
            return false;
        };
        let insn = Instruction::from_id(self.ctx, insn_id);
        if insn.parent().map(|b| b.id) != Some(block) {
            return false;
        }
        matches!(
            insn.mnemonic(),
            Mnemonic::Load(Load { ptr, .. }) if *ptr == var
        )
    }

    fn remove_promoted_stores(
        &mut self,
        vars: &HashSet<ValueId>,
        consumed_stores: &HashSet<InstructionId>,
        dead_stores: &HashSet<InstructionId>,
        preserved_stores: &HashSet<InstructionId>,
    ) -> bool {
        let mut changed = false;
        let block_ids: Vec<BlockId> = Function::from_id(self.ctx, self.function_id)
            .blocks()
            .map(|b| b.id)
            .collect();

        // Vars that still have a Load somewhere in the function — i.e. a load the
        // renaming DFS did not forward away (a successor block it never reached, or
        // a partially promoted var). Removing a store to such a var would strand its
        // load reading an undefined location (the SLEIGH `v0`/`v1` unique-space
        // leak), so those stores fall back to the consumed/dead guard below.
        let mut vars_with_surviving_loads: HashSet<ValueId> = HashSet::default();
        // Register loads left *unpromoted* — e.g. a narrow sub-register read (`CL`)
        // declined by `has_overlapping_register_store` because a wider overlapping
        // store (`ECX`) exists. The decline defers such a load to the overlap-aware
        // GVN memory pass to slice out of "the dominating store", which assumes that
        // store still exists. So a register store that *contains* one of these loads
        // must not be removed here (even when it is a dead overwrite): dropping it
        // would strand the narrow load as a free register read — a spurious live-in —
        // silently discarding the value the wider store carried.
        let mut unpromoted_register_loads: Vec<ValueId> = Vec::new();
        for &block_id in &block_ids {
            for &insn_id in BasicBlock::from_id(self.ctx, block_id).instruction_ids() {
                if let Mnemonic::Load(Load { ptr, .. }) =
                    Instruction::from_id(self.ctx, insn_id).mnemonic()
                {
                    if vars.contains(ptr) {
                        vars_with_surviving_loads.insert(*ptr);
                    } else if register_varnode(self.ctx, *ptr).is_some() {
                        unpromoted_register_loads.push(*ptr);
                    }
                }
            }
        }

        for block_id in block_ids {
            let insn_ids: Vec<InstructionId> = BasicBlock::from_id(self.ctx, block_id)
                .instruction_ids()
                .to_vec();

            for insn_id in insn_ids {
                let mnemonic = Instruction::from_id(self.ctx, insn_id).mnemonic();
                let Mnemonic::Store(Store { ptr, .. }) = mnemonic else {
                    continue;
                };
                if !vars.contains(ptr) {
                    continue;
                }
                // A wider register store an unpromoted narrow load overlaps is the
                // slice source that load was deferred to — keep it (see above).
                if register_varnode(self.ctx, *ptr).is_some()
                    && unpromoted_register_loads
                        .iter()
                        .any(|&load| wider_register_store_contains(self.ctx, *ptr, load))
                {
                    continue;
                }
                // For register-space varnodes, preserve live-out stores (callee-save
                // restores, return values); only consumed or overwritten stores are
                // safe to drop. Temporary-space varnodes are never live-out, but a
                // store whose load was not forwarded (the var still has a surviving
                // load) must be kept under the same guard — dropping it would strand
                // that load on an undefined temp read. A temp var with no surviving
                // load is fully promoted, so its remaining stores (e.g. a dead store
                // the conservative frame analysis did not flag) are safe to remove.
                // Stack-slot literals keep the unconditional removal.
                let guarded = if let ValueId::Varnode(vn_id) = ptr {
                    match Varnode::from_id(self.ctx, *vn_id).space().ty {
                        SpaceType::Register => true,
                        SpaceType::Temporary => vars_with_surviving_loads.contains(ptr),
                        _ => false,
                    }
                } else {
                    false
                };
                if guarded {
                    let removable = (consumed_stores.contains(&insn_id)
                        || dead_stores.contains(&insn_id))
                        && !preserved_stores.contains(&insn_id);
                    if !removable {
                        continue;
                    }
                }
                self.ctx.remove_instruction(insn_id);
                changed = true;
            }
        }
        changed
    }

    fn is_register_var(&self, var: ValueId) -> bool {
        register_varnode(self.ctx, var).is_some()
    }

    /// A partial-register store (e.g. a write to `AL`) invalidates the promoted
    /// SSA value of every overlapping full register (e.g. `EAX`): the in-register
    /// bytes no longer match the promoted value, so later reads must reload.
    ///
    /// TODO: this is deliberately coarse — it drops the whole overlapping var to
    /// `Clobbered` and forces a full reload. A more precise pass could model the
    /// partial update (splice the stored bytes into the promoted value) and keep
    /// the overlapping register promoted; mem2reg cannot represent sub-register
    /// splicing today.
    fn clobber_overlapping_register_vars(
        &mut self,
        stored_ptr: ValueId,
        store_insn: InstructionId,
        state: &mut RenameState<'_>,
    ) {
        let Some(clobbered_vars) = state.register_clobbers.get(&stored_ptr) else {
            return;
        };
        let clobbered = clobbered_vars
            .iter()
            .copied()
            // A sliced var this store low-alignedly covers is *defined*, not
            // clobbered — `define_sliced_vars` installs its precise value right
            // after. Excluding it here also keeps this store off the preserved
            // list on its account, so a genuinely dead wider store stays
            // removable.
            .filter(|&var| {
                !(state.sliced.contains(&var)
                    && register_store_low_aligned_contains(self.ctx, stored_ptr, var))
            })
            .map(|var| {
                let reaching_store = match decide_variable_value(var, &state.frames) {
                    Some(FrameEntry::Defined(ReachingValue {
                        store_insn: Some(id),
                        ..
                    })) => Some(id),
                    _ => None,
                };
                (var, reaching_store)
            })
            .collect::<Vec<_>>();
        if clobbered.is_empty() {
            return;
        }

        let frame = state.frames.last_mut().unwrap();
        state.preserved_stores.insert(store_insn);
        for (var, reaching_store) in clobbered {
            if let Some(id) = reaching_store {
                state.preserved_stores.insert(id);
            }
            frame.insert(var, FrameEntry::Clobbered);
        }
    }

    /// Install the sliced-var definitions a store contributes: for every sliced
    /// register var the store low-alignedly covers, its reaching value becomes
    /// the stored source (sliced to width on each later forward/edge by the
    /// resize the renamer already applies). Recorded with `store_insn: None` so
    /// the covering store is never marked consumed/dead on the slice's behalf —
    /// the narrow var only reads the SSA value, not the store's memory effect.
    fn define_sliced_vars(
        &self,
        block: BlockId,
        stored_ptr: ValueId,
        src: ValueId,
        state: &mut RenameState<'_>,
    ) {
        if state.sliced.is_empty() {
            return;
        }
        let covered: Vec<ValueId> = state
            .sliced
            .iter()
            .copied()
            .filter(|&var| register_store_low_aligned_contains(self.ctx, stored_ptr, var))
            .collect();
        let frame = state.frames.last_mut().unwrap();
        for var in covered {
            frame.insert(
                var,
                FrameEntry::Defined(ReachingValue {
                    _defining_block: block,
                    value: src,
                    store_insn: None,
                }),
            );
        }
    }
}

fn block_contains_store_to_var(
    ctx: &Context,
    block: &BlockRef,
    var: ValueId,
    var_is_sliced: bool,
) -> bool {
    block.iter().any(|insn| {
        if let Mnemonic::Store(Store { ptr, .. }) = insn.mnemonic() {
            *ptr == var || (var_is_sliced && register_store_low_aligned_contains(ctx, *ptr, var))
        } else {
            false
        }
    })
}

fn find_phi_insert_positions(
    ctx: &Context,
    var: ValueId,
    function: &FunctionRef,
    sliced: &HashSet<ValueId>,
    frontier: &HashMap<BlockId, HashSet<BlockId>>,
    live_in: &HashSet<BlockId>,
) -> HashSet<BlockId> {
    let var_is_sliced = sliced.contains(&var);
    let block_containing_store = function
        .blocks()
        .filter(|b| block_contains_store_to_var(ctx, b, var, var_is_sliced))
        .map(|b| b.id)
        .collect::<HashSet<_>>();

    let mut worklist: Vec<BlockId> = block_containing_store.iter().copied().collect();
    let mut result = HashSet::default();

    while let Some(block) = worklist.pop() {
        // `frontier` is keyed only by blocks reachable from the root. A store in
        // unreachable code places no phi, so skip blocks with no frontier entry.
        let Some(block_frontier) = frontier.get(&block) else {
            continue;
        };
        for dominated in block_frontier.iter().copied() {
            if !live_in.contains(&dominated) {
                continue;
            }
            if result.insert(dominated) && !block_containing_store.contains(&dominated) {
                worklist.push(dominated);
            }
        }
    }

    result
}

#[derive(Clone, Copy)]
struct ReachingValue {
    _defining_block: BlockId,
    value: ValueId,
    // `None` for block params, `Some(id)` for actual store instructions.
    store_insn: Option<InstructionId>,
}

#[derive(Clone, Copy)]
enum FrameEntry {
    Defined(ReachingValue),
    // Clobber markers shadow any reaching definition in an enclosing frame.
    Clobbered,
}

type Frame = HashMap<ValueId, FrameEntry>;

struct RenameState<'a> {
    var_params: &'a BlockParamAssignments,
    vars: &'a HashSet<ValueId>,
    /// Sliced register vars (see [`Promotable::sliced`]). A covering wider store
    /// defines these instead of clobbering them.
    sliced: &'a HashSet<ValueId>,
    register_clobbers: HashMap<ValueId, Vec<ValueId>>,
    visited: HashSet<BlockId>,
    frames: Vec<Frame>,
    consumed_stores: HashSet<InstructionId>,
    dead_stores: HashSet<InstructionId>,
    preserved_stores: HashSet<InstructionId>,
    changed: bool,
}

impl<'a> RenameState<'a> {
    fn new(
        var_params: &'a BlockParamAssignments,
        vars: &'a HashSet<ValueId>,
        sliced: &'a HashSet<ValueId>,
        register_clobbers: HashMap<ValueId, Vec<ValueId>>,
        changed: bool,
    ) -> Self {
        Self {
            var_params,
            vars,
            sliced,
            register_clobbers,
            visited: HashSet::default(),
            frames: vec![Frame::default()],
            consumed_stores: HashSet::default(),
            dead_stores: HashSet::default(),
            preserved_stores: HashSet::default(),
            changed,
        }
    }
}

fn decide_variable_value(var: ValueId, frames: &[Frame]) -> Option<FrameEntry> {
    for frame in frames.iter().rev() {
        if let Some(entry) = frame.get(&var) {
            // The nearest frame wins, whether it holds a value or a clobber
            // marker — do not fall through to an enclosing definition.
            return Some(*entry);
        }
    }

    None
}

impl Mem2Reg<'_, '_> {
    fn decide_values_start_from(&mut self, block: BlockId, state: &mut RenameState<'_>) {
        if state.visited.contains(&block) {
            return;
        }
        state.visited.insert(block);

        // Seed the current frame with block params so loads within this block
        // and its Branch-reachable descendants see them as the current SSA value.
        let params = state
            .var_params
            .get(&block)
            .map(|params| {
                params
                    .iter()
                    .map(|(&var, &param_id)| (var, param_id))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        {
            let frame = state.frames.last_mut().unwrap();
            for (var, param_id) in params {
                frame.insert(
                    var,
                    FrameEntry::Defined(ReachingValue {
                        _defining_block: block,
                        value: ValueId::BlockParam(param_id),
                        store_insn: None,
                    }),
                );
            }
        }

        let insn_ids: Vec<InstructionId> = BasicBlock::from_id(self.ctx, block)
            .instruction_ids()
            .to_vec();

        for insn_id in insn_ids {
            // Clone the mnemonic so we can release the immutable borrow on ctx
            // before taking mutable borrows in each arm.
            let mnemonic = Instruction::from_id(self.ctx, insn_id).mnemonic().clone();

            match mnemonic {
                Mnemonic::Store(Store { ptr, src, .. }) => {
                    if state.vars.contains(&ptr) {
                        // If the current reaching definition for this var was never consumed,
                        // it is being overwritten without being read — mark it as dead.
                        // Only look in the top frame: an overwrite on one CBranch/BranchInd
                        // arm must not kill a store made in an ancestor frame that a sibling
                        // path still relies on as a live-out value. Within one frame, blocks
                        // are chained by unconditional edges, so the overwrite post-dominates
                        // the old store (DFS order = path order).
                        if let Some(FrameEntry::Defined(ReachingValue {
                            store_insn: Some(old_id),
                            ..
                        })) = state.frames.last().unwrap().get(&ptr)
                            && !state.consumed_stores.contains(old_id)
                        {
                            state.dead_stores.insert(*old_id);
                        }
                        state.frames.last_mut().unwrap().insert(
                            ptr,
                            FrameEntry::Defined(ReachingValue {
                                _defining_block: block,
                                value: src,
                                store_insn: Some(insn_id),
                            }),
                        );
                    }
                    self.clobber_overlapping_register_vars(ptr, insn_id, state);
                    self.define_sliced_vars(block, ptr, src, state);
                }

                Mnemonic::Load(Load { ptr, size, .. }) if state.vars.contains(&ptr) => {
                    let reaching = decide_variable_value(ptr, &state.frames);
                    let (mut load_value, store_insn) = match reaching {
                        Some(FrameEntry::Defined(reaching)) => {
                            (reaching.value, reaching.store_insn)
                        }
                        // No reaching definition. For a register var this means the
                        // value comes from a clobbering call (or is uninitialized) —
                        // [`live_in_blocks`] treats the call as a def, so no entry
                        // param was created. Leave the load in place to read the live
                        // register rather than fabricate a value. A non-register var
                        // (stack slot) must always have a reaching def; missing one is
                        // a bug, so keep the assertion.
                        Some(FrameEntry::Clobbered) | None if self.is_register_var(ptr) => {
                            continue;
                        }
                        Some(FrameEntry::Clobbered) | None => panic!(
                            "Unable to decide value for variable {:?} in block {:?}",
                            ptr, block
                        ),
                    };
                    if let Some(store_id) = store_insn {
                        state.consumed_stores.insert(store_id);
                    }
                    load_value = self.resize_forwarded_load_value(block, insn_id, load_value, size);
                    self.ctx
                        .replace_all_uses_with(ValueId::Instruction(insn_id), load_value);
                    self.ctx.remove_instruction(insn_id);
                    state.changed = true;
                }

                Mnemonic::CBranch(CBranch {
                    condition,
                    success_block,
                    failure_block,
                    success_args: existing_success,
                    failure_args: existing_failure,
                }) => {
                    // Compute args before recursing; even if a target is already visited
                    // (back-edge), we still need to wire the correct values.
                    let success_args = self.merge_branch_args(
                        BranchEdge {
                            target: success_block,
                            source_block: block,
                            branch_insn: insn_id,
                            existing_args: &existing_success,
                        },
                        state,
                    );
                    let failure_args = self.merge_branch_args(
                        BranchEdge {
                            target: failure_block,
                            source_block: block,
                            branch_insn: insn_id,
                            existing_args: &existing_failure,
                        },
                        state,
                    );

                    // Update through `replace_instruction_mnemonic` so the new args
                    // are registered in the reverse use map — a directly mutated
                    // `args` field would leave the passed values looking unused, so
                    // a later DCE/fold pass would delete them and dangle the arg.
                    if existing_success != success_args || existing_failure != failure_args {
                        self.ctx.replace_instruction_mnemonic(
                            insn_id,
                            Mnemonic::CBranch(CBranch {
                                condition,
                                success_block,
                                success_args,
                                failure_block,
                                failure_args,
                            }),
                        );
                        state.changed = true;
                    }

                    state.frames.push(Frame::default());
                    self.decide_values_start_from(success_block, state);
                    state.frames.pop();

                    state.frames.push(Frame::default());
                    self.decide_values_start_from(failure_block, state);
                    state.frames.pop();
                }

                Mnemonic::Branch(Branch {
                    target,
                    args: existing,
                }) => {
                    let args = self.merge_branch_args(
                        BranchEdge {
                            target,
                            source_block: block,
                            branch_insn: insn_id,
                            existing_args: &existing,
                        },
                        state,
                    );
                    // Update through `replace_instruction_mnemonic` so the passed
                    // values are recorded as uses (see the CBranch note above).
                    if existing != args {
                        self.ctx.replace_instruction_mnemonic(
                            insn_id,
                            Mnemonic::Branch(Branch { target, args }),
                        );
                        state.changed = true;
                    }
                    self.decide_values_start_from(target, state);
                }

                call_mnemonic @ (Mnemonic::Call(_) | Mnemonic::CallInd(_)) => {
                    // A call clobbers its callee's registers. Shadow each promoted
                    // register var it clobbers with a clobber marker so a read in the
                    // continuation sees the call's output (left as a register load),
                    // not the value the caller held before the call.
                    let clobbered = self.call_clobbered_register_vars(&call_mnemonic, state.vars);
                    {
                        let frame = state.frames.last_mut().unwrap();
                        for v in clobbered {
                            // A pending store to a clobbered register, never read
                            // before the call, is overwritten by the call's own
                            // write — it is dead, exactly as a store-overwrites-store
                            // would be (see the `Store` arm). Only the top frame is
                            // consulted, for the same path-domination reason.
                            if let Some(FrameEntry::Defined(ReachingValue {
                                store_insn: Some(old_id),
                                ..
                            })) = frame.get(&v)
                            {
                                let old_id = *old_id;
                                if !state.consumed_stores.contains(&old_id) {
                                    state.dead_stores.insert(old_id);
                                }
                            }
                            frame.insert(v, FrameEntry::Clobbered);
                        }
                    }
                    self.visit_successors(block, state);
                }

                Mnemonic::BranchInd(_) => {
                    self.visit_successors(block, state);
                }

                _ => {}
            }
        }
    }

    fn visit_successors(&mut self, block: BlockId, state: &mut RenameState<'_>) {
        let successors: Vec<BlockId> = BasicBlock::from_id(self.ctx, block)
            .successors()
            .map(|(_, id)| id)
            .collect();
        // Successors of a call fall-through / BranchInd jump table are mutually
        // exclusive paths. Push a fresh frame per edge (like the CBranch arm) so a
        // store made while visiting one successor's subtree is not visible when a
        // sibling successor is visited.
        for successor in successors {
            state.frames.push(Frame::default());
            self.decide_values_start_from(successor, state);
            state.frames.pop();
        }
    }

    /// The promoted register vars in `vars` clobbered by call terminator `call`.
    ///
    /// Computed once per call block: the callee's clobber set is fetched a single
    /// time rather than re-derived per var (as a per-var [`call_clobbers_var`]
    /// would). A `CallInd`, or a callee with no recorded clobber set, is treated
    /// conservatively as clobbering every promoted register var.
    fn call_clobbered_register_vars(
        &self,
        call: &Mnemonic,
        vars: &HashSet<ValueId>,
    ) -> Vec<ValueId> {
        let register_vars = || {
            vars.iter()
                .copied()
                .filter(|&v| register_varnode(self.ctx, v).is_some())
        };
        match call {
            Mnemonic::CallInd(_) => register_vars().collect(),
            Mnemonic::Call(call) => {
                let callee = Function::from_id(self.ctx, call.target);
                let resolved = callee.is_externally_resolved();
                let clobbered = callee.clobbered_regs().map(<[VarnodeId]>::to_vec);
                match clobbered {
                    // A resolved callee with no recorded set clobbers nothing; an
                    // unresolved one is unknown, so conservatively clobbers all.
                    None => {
                        if resolved {
                            Vec::new()
                        } else {
                            register_vars().collect()
                        }
                    }
                    Some(clobbered) => register_vars()
                        .filter(|&v| {
                            clobbered
                                .iter()
                                .any(|&c| self.aliases.may_alias(self.ctx, ValueId::Varnode(c), v))
                        })
                        .collect(),
                }
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {

    use jstd::graph::analysis::compute_dominators;
    use qcode::value::{BasicBlock, Function, insn::Mnemonic};
    use qcode_macro::qcode;

    use super::*;

    #[test]
    fn phi_insert_pos() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;

            fn test:
                <bb1>
                    store(&A, i32 1);
                    if i8 1 goto <bb2> else goto <bb4>;

                <bb2>
                    %a0 = load(i32, &A);
                    store(&A, i32 0);
                    goto <bb3>;

                <bb3>
                    %a1 = load(i32, &A);
                    store(&A, i32 1);
                    if i8 1 goto <bb2> else goto <bb8>;

                <bb4>
                    if i8 1 goto <bb5> else goto <bb6>;

                <bb5>
                    %a2 = load(i32, &A);
                    store(&A, i32 2);
                    goto <bb7>;

                <bb6>
                    %a3 = load(i32, &A);
                    store(&A, i32 3);
                    goto <bb7>;

                <bb7>
                    %a4 = load(i32, &A);
                    goto <bb8>;

                <bb8>
                    %a5 = load(i32, &A);
                    return [0];
        "
        );

        let function = Function::from_id(&ctx, test);
        let dom = compute_dominators(&ctx, function.root().unwrap().id);
        let frontier = dom.dominator_frontier();
        let aliases = AliasResult::simple(&ctx);
        let sliced = HashSet::default();
        let live_in = live_in_blocks(&ctx, test, A.into(), &sliced, &aliases);

        let result =
            find_phi_insert_positions(&ctx, A.into(), &function, &sliced, frontier, &live_in);

        let named_result = result
            .iter()
            .map(|b| BasicBlock::from_id(&ctx, *b).name().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            named_result.len(),
            3,
            "Expected 3 blocks to require arguments nodes"
        );
        assert!(named_result.contains(&"bb2"), "bb2 should be in the result");
        assert!(named_result.contains(&"bb7"), "bb7 should be in the result");
        assert!(named_result.contains(&"bb8"), "bb8 should be in the result");
    }

    #[test]
    fn forwarded_load_value_is_resized_to_load_width() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;

            fn test:
                <bb>
                    store(&A, i32 0xffffffff);
                    %loaded = load(i64, &A);
                    %masked = %loaded & 0xf;
                    return [%masked];
        "
        );

        let aliases = AliasResult::simple(&ctx);
        mem2reg(&mut ctx, test, &aliases);

        let Mnemonic::Binop(masked_mnemonic) = Instruction::from_id(&ctx, masked).mnemonic() else {
            panic!("masked instruction should remain a binop");
        };
        assert_eq!(ValueRef::new(masked_mnemonic.lhs, &ctx).size(), 8);
        assert_eq!(ValueRef::new(masked_mnemonic.rhs, &ctx).size(), 8);
    }

    #[test]
    fn mem2reg_diamond() {
        // Diamond CFG: two paths each storing different values, joined at exit.
        // exit needs a block param for A.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;

            fn test:
                <entry>
                    store(&A, i32 1);
                    if i8 1 goto <left> else goto <right>;

                <left>
                    store(&A, i32 2);
                    goto <exit>;

                <right>
                    store(&A, i32 3);
                    goto <exit>;

                <exit>
                    %val = load(i32, &A);
                    return [0];
        "
        );

        let aliases = AliasResult::simple(&ctx);
        mem2reg(&mut ctx, test, &aliases);

        // exit block should have exactly one block param for A
        let exit_block = BasicBlock::from_id(&ctx, exit);
        assert_eq!(exit_block.num_params(), 1, "exit should have 1 block param");

        // No loads or stores to A should remain
        let function = Function::from_id(&ctx, test);
        for block in function.blocks() {
            for insn in block.iter() {
                assert!(
                    !matches!(insn.mnemonic(), Mnemonic::Load(_) | Mnemonic::Store(_)),
                    "no loads or stores should remain after mem2reg"
                );
            }
        }

        // left and right branches to exit should carry args
        let left_block = BasicBlock::from_id(&ctx, left);
        let left_branch = left_block.iter().last().unwrap();
        let Mnemonic::Branch(b) = left_branch.mnemonic() else {
            panic!("expected branch");
        };
        assert_eq!(b.args.len(), 1, "left→exit branch should pass 1 arg");

        let right_block = BasicBlock::from_id(&ctx, right);
        let right_branch = right_block.iter().last().unwrap();
        let Mnemonic::Branch(b) = right_branch.mnemonic() else {
            panic!("expected branch");
        };
        assert_eq!(b.args.len(), 1, "right→exit branch should pass 1 arg");
    }

    #[test]
    fn phi_insert_pos_does_not_revisit_discovered_frontier_blocks() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;

            fn test:
                <entry>
                    store(&A, i32 1);
                    goto <loop_header>;

                <loop_header>
                    %val = load(i32, &A);
                    if i8 1 goto <loop_header> else goto <exit>;

                <exit>
                    return [0];
        "
        );

        let function = Function::from_id(&ctx, test);
        let live_in = HashSet::from_iter([loop_header]);
        let frontier = HashMap::from_iter([
            (entry, HashSet::from_iter([loop_header])),
            (loop_header, HashSet::from_iter([loop_header])),
        ]);

        let sliced = HashSet::default();
        let result =
            find_phi_insert_positions(&ctx, A.into(), &function, &sliced, &frontier, &live_in);

        assert_eq!(result, HashSet::from_iter([loop_header]));
    }

    #[test]
    fn mem2reg_linear() {
        // No join points: no block params needed.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;

            fn test:
                <entry>
                    store(&A, i32 42);
                    goto <exit>;

                <exit>
                    %val = load(i32, &A);
                    return [0];
        "
        );

        let aliases = AliasResult::simple(&ctx);
        mem2reg(&mut ctx, test, &aliases);

        // No block params anywhere
        let function = Function::from_id(&ctx, test);
        for block in function.blocks() {
            assert_eq!(
                block.num_params(),
                0,
                "no block params expected in linear CFG"
            );
        }

        // No loads or stores remain
        for block in Function::from_id(&ctx, test).blocks() {
            for insn in block.iter() {
                assert!(
                    !matches!(insn.mnemonic(), Mnemonic::Load(_) | Mnemonic::Store(_)),
                    "no loads or stores should remain"
                );
            }
        }
    }

    #[test]
    fn remove_mem() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;

            fn test:
                <bb1>
                    store(&A, i32 1);
                    if i8 1 goto <bb2> else goto <bb4>;

                <bb2>
                    %a0 = load(i32, &A);
                    %t0 = %a0 + i32 1;
                    store(&A, %t0);
                    goto <bb3>;

                <bb3>
                    %a1 = load(i32, &A);
                    %t1 = %a1 + %a1;
                    store(&A, %t1);
                    if i8 1 goto <bb2> else goto <bb8>;

                <bb4>
                    if i8 1 goto <bb5> else goto <bb6>;

                <bb5>
                    %a2 = load(i32, &A);
                    %t2 = %a2 + i32 2;
                    store(&A, %t2);
                    goto <bb7>;

                <bb6>
                    %a3 = load(i32, &A);
                    %t3 = %a3 + i32 3;
                    store(&A, %t3);
                    goto <bb7>;

                <bb7>
                    %a4 = load(i32, &A);
                    goto <bb8>;

                <bb8>
                    %a5 = load(i32, &A);
                    return [0];
        "
        );

        let function = Function::from_id(&ctx, test);
        let dom = compute_dominators(&ctx, function.root().unwrap().id);
        let frontier = dom.dominator_frontier();
        let aliases = AliasResult::simple(&ctx);
        let sliced = HashSet::default();
        let live_in = live_in_blocks(&ctx, test, A.into(), &sliced, &aliases);

        let result =
            find_phi_insert_positions(&ctx, A.into(), &function, &sliced, frontier, &live_in);

        let named_result = result
            .iter()
            .map(|b| BasicBlock::from_id(&ctx, *b).name().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            named_result.len(),
            3,
            "Expected 3 blocks to require phi nodes"
        );
        assert!(named_result.contains(&"bb2"), "bb2 should be in the result");
        assert!(named_result.contains(&"bb7"), "bb7 should be in the result");
        assert!(named_result.contains(&"bb8"), "bb8 should be in the result");
    }

    #[test]
    fn overlapping_full_register_seed_does_not_create_subregister_entry_param() {
        use qcode::{builder::Builder, testing::TestContext};

        let mut tc = TestContext::new();
        let fun_id = Function::make(&mut tc.ctx, "callee".into()).unwrap().id;
        let block_id = tc.ctx.get_or_make_block(0x1000);
        Function::from_id_mut(&mut tc.ctx, fun_id)
            .set_root(block_id)
            .unwrap();
        Function::from_id_mut(&mut tc.ctx, fun_id).set_pure_reg(true);

        let full_param = BasicBlock::from_id_mut(&mut tc.ctx, block_id)
            .push_param(8)
            .id;
        tc.ctx.values.block_params[full_param].name = Some("r0".into());

        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            b.push_store(
                ValueId::BlockParam(full_param),
                ValueId::Varnode(tc.r0),
                tc.reg_space,
            );
            let sub = b
                .push_load::<false>(ValueId::Varnode(tc.r0_lo32), 4, tc.reg_space)
                .id();
            b.push_store(sub, ValueId::Varnode(tc.r1), tc.reg_space);
            unsafe { b.dont_finalize() };
        }

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, fun_id, &aliases);

        let params: Vec<_> = BasicBlock::from_id(&tc.ctx, block_id).params().collect();
        assert_eq!(
            params.len(),
            1,
            "mem2reg must not turn the sub-register read into a new pure_reg \
             entry param; it is covered by the wider seed store:\n{}",
            Function::from_id(&tc.ctx, fun_id)
        );
        assert_eq!(params[0].id, full_param);
    }

    /// Build `store(0x1234, @SP-8); reload(@SP-8)` in one block off an `@SP`
    /// param (origin = `sp_reg`). The two `@SP-8` addresses are distinct `Sub`
    /// values until canonicalized. Returns `(fun_id, block, sp_param, sp_reg)`.
    fn sp_slot_function(tc: &mut qcode::testing::TestContext) -> (FunctionId, ValueId, VarnodeId) {
        use qcode::builder::Builder;
        let sp_reg = tc.r0;
        let ram = tc.ctx.default_space;
        let fun_id = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let block = tc.ctx.get_or_make_block(0x1000);
        Function::from_id_mut(&mut tc.ctx, fun_id)
            .set_root(block)
            .unwrap();
        let pid = BasicBlock::from_id_mut(&mut tc.ctx, block).push_param(8).id;
        tc.ctx.values.block_params[pid].origin = Some(ValueId::Varnode(sp_reg));
        tc.ctx.values.block_params[pid].name = Some("RSP".into());
        let sp = ValueId::BlockParam(pid);

        let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
        let v = b.context_mut().get_const(0x1234, 8).id();
        let c8 = b.context_mut().get_const(8, 8).id();
        let addr_store = b.push_sub(sp, c8).id();
        b.push_store(v, addr_store, ram);
        let addr_load = b.push_sub(sp, c8).id();
        b.push_load::<false>(addr_load, 8, ram);
        unsafe { b.dont_finalize() };
        (fun_id, sp, sp_reg)
    }

    fn has_load(ctx: &Context, fun_id: FunctionId) -> bool {
        Function::from_id(ctx, fun_id)
            .blocks()
            .any(|b| b.iter().any(|i| matches!(i.mnemonic(), Mnemonic::Load(_))))
    }

    /// A canonical `@SP - N` local slot is promoted just like a `@stack_base`
    /// literal: the reload forwards from the store and the load disappears.
    #[test]
    fn promotes_canonical_sp_relative_slot() {
        use crate::stack::canonicalize::canonicalize_sp_slots;
        let mut tc = qcode::testing::TestContext::new();
        let (fun_id, sp, sp_reg) = sp_slot_function(&mut tc);

        canonicalize_sp_slots(&mut tc.ctx, fun_id, sp_reg);
        let aliases = AliasResult::simple(&tc.ctx);
        let changed = mem2reg_framed(&mut tc.ctx, fun_id, &aliases, Some(sp));

        assert!(changed, "the @SP-8 slot should be promoted");
        assert!(
            !has_load(&tc.ctx, fun_id),
            "the reload should be forwarded from the store:\n{}",
            Function::from_id(&tc.ctx, fun_id)
        );
    }

    /// Without the `@SP` parameter the `@SP - N` address is unrecognised (it is
    /// not a `@stack_base` literal), so the slot is left in memory.
    #[test]
    fn sp_relative_slot_not_promoted_without_sp_param() {
        let mut tc = qcode::testing::TestContext::new();
        let (fun_id, _sp, _sp_reg) = sp_slot_function(&mut tc);

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg_framed(&mut tc.ctx, fun_id, &aliases, None);

        assert!(
            has_load(&tc.ctx, fun_id),
            "an unrecognised @SP-relative slot must stay in memory"
        );
    }

    /// A dynamically indexed `@SP + reg` access may alias any slot, so it poisons
    /// stack promotion for the whole function: an otherwise-promotable `@SP - 8`
    /// local is left in memory.
    #[test]
    fn dynamic_sp_indexed_access_disables_promotion() {
        use crate::stack::canonicalize::canonicalize_sp_slots;
        use qcode::builder::Builder;

        let mut tc = qcode::testing::TestContext::new();
        let sp_reg = tc.r0;
        let ram = tc.ctx.default_space;
        let fun_id = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let block = tc.ctx.get_or_make_block(0x1000);
        Function::from_id_mut(&mut tc.ctx, fun_id)
            .set_root(block)
            .unwrap();
        let pid = BasicBlock::from_id_mut(&mut tc.ctx, block).push_param(8).id;
        tc.ctx.values.block_params[pid].origin = Some(ValueId::Varnode(sp_reg));
        let sp = ValueId::BlockParam(pid);

        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            let v = b.context_mut().get_const(0x1234, 8).id();
            let c8 = b.context_mut().get_const(8, 8).id();
            // A promotable local: store then reload `@SP - 8`.
            let addr_store = b.push_sub(sp, c8).id();
            b.push_store(v, addr_store, ram);
            let addr_load = b.push_sub(sp, c8).id();
            let reloaded = b.push_load::<false>(addr_load, 8, ram).id();
            // A dynamic `@SP + reloaded` access — index is not a constant.
            let dyn_ptr = b.push_add(sp, reloaded).id();
            b.push_load::<false>(dyn_ptr, 1, ram);
            unsafe { b.dont_finalize() };
        }

        let loads_before = Function::from_id(&tc.ctx, fun_id)
            .blocks()
            .flat_map(|b| b.iter().collect::<Vec<_>>())
            .filter(|i| matches!(i.mnemonic(), Mnemonic::Load(_)))
            .count();

        canonicalize_sp_slots(&mut tc.ctx, fun_id, sp_reg);
        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg_framed(&mut tc.ctx, fun_id, &aliases, Some(sp));

        let loads_after = Function::from_id(&tc.ctx, fun_id)
            .blocks()
            .flat_map(|b| b.iter().collect::<Vec<_>>())
            .filter(|i| matches!(i.mnemonic(), Mnemonic::Load(_)))
            .count();

        assert_eq!(
            loads_after,
            loads_before,
            "the dynamic @SP+reg access must disable promotion of the @SP-8 local:\n{}",
            Function::from_id(&tc.ctx, fun_id)
        );
    }

    #[test]
    fn partial_register_store_clobbers_promoted_full_register_value() {
        use qcode::{builder::Builder, testing::TestContext};

        let mut tc = TestContext::new();
        let full = ValueId::Varnode(tc.r0_lo32);
        let low_byte = ValueId::Varnode(tc.r0_byte0);
        let pre_clobber_sink = ValueId::Varnode(tc.r1);
        let post_clobber_sink = ValueId::Varnode(tc.r2);

        let fun_id = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
        let block_id = tc.ctx.get_or_make_block(0x1000);
        Function::from_id_mut(&mut tc.ctx, fun_id)
            .set_root(block_id)
            .unwrap();

        let zero_store;
        let full_load;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            let zero = b.context_mut().get_const(0, 4).id();
            let one = b.context_mut().get_const(1, 1).id();
            zero_store = b.push_store(zero, full, tc.reg_space).id;
            let pre_clobber_load = b.push_load::<false>(full, 4, tc.reg_space).id();
            b.push_store(pre_clobber_load, pre_clobber_sink, tc.reg_space);
            b.push_store(one, low_byte, tc.reg_space);
            full_load = b.push_load::<false>(full, 4, tc.reg_space).id();
            b.push_store(full_load, post_clobber_sink, tc.reg_space);
            unsafe { b.dont_finalize() };
        }

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, fun_id, &aliases);

        let ValueId::Instruction(full_load_id) = full_load else {
            panic!("push_load should produce an instruction value");
        };
        let block = BasicBlock::from_id(&tc.ctx, block_id);
        assert!(
            block.instruction_ids().contains(&zero_store),
            "the full-register zeroing store must survive because the later \
             partial-register store does not overwrite the full value:\n{block}"
        );
        assert!(
            block.instruction_ids().contains(&full_load_id),
            "the full-register load after a low-byte write must not be promoted \
             to the stale full-register SSA value:\n{block}"
        );
    }

    /// A low-aligned sub-register read (`AL`) covered by a wider register store
    /// (`EAX`) is *sliced* out of the stored value during renaming: the narrow
    /// load is forwarded (and removed), not left dangling. With a constant store
    /// the slice folds to the low byte of the constant. This is the read-side
    /// dual of the `clobber_overlapping_register_vars` sub-register-splice TODO,
    /// and supersedes the earlier conservative behaviour where the byte load was
    /// kept and deferred to the GVN memory pass.
    #[test]
    fn subregister_load_is_sliced_from_overlapping_full_store() {
        use qcode::{builder::Builder, testing::TestContext};

        let mut tc = TestContext::new();
        let full = ValueId::Varnode(tc.r0_lo32);
        let low_byte = ValueId::Varnode(tc.r0_byte0);
        let full_sink = ValueId::Varnode(tc.r1);
        let byte_sink = ValueId::Varnode(tc.r2);

        let fun_id = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
        let block_id = tc.ctx.get_or_make_block(0x1000);
        Function::from_id_mut(&mut tc.ctx, fun_id)
            .set_root(block_id)
            .unwrap();

        let byte_store;
        let byte_load;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            let value = b.context_mut().get_const(0x12345678, 4).id();
            b.push_store(value, full, tc.reg_space);
            let full_load = b.push_load::<false>(full, 4, tc.reg_space).id();
            b.push_store(full_load, full_sink, tc.reg_space);
            byte_load = b.push_load::<false>(low_byte, 1, tc.reg_space).id();
            byte_store = b.push_store(byte_load, byte_sink, tc.reg_space).id;
            unsafe { b.dont_finalize() };
        }

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, fun_id, &aliases);

        let block = BasicBlock::from_id(&tc.ctx, block_id);
        assert!(
            !block
                .iter()
                .any(|i| matches!(i.mnemonic(), Mnemonic::Load(_))),
            "both the full and the sliced byte load should be forwarded away:\n{block}"
        );
        // The byte sink receives the low byte of the stored constant: 0x78.
        let Mnemonic::Store(Store { src, .. }) =
            Instruction::from_id(&tc.ctx, byte_store).mnemonic()
        else {
            panic!("byte sink store vanished");
        };
        let ValueId::Literal(lit) = src else {
            panic!("expected the sliced byte to fold to a constant, got {src:?}");
        };
        assert_eq!(
            tc.ctx.values.literals[*lit].value, 0x78,
            "the slice is the low byte of 0x12345678"
        );
    }

    /// The motivating loop (the `410f50` string-copy shape): a loop body writes a
    /// full register (`EAX`) and reads its low byte (`AL`) at the top, so `AL` is
    /// loop-carried. The low-aligned slice promotes `AL` to a byte-wide phi at the
    /// loop header; the register load disappears, the back-edge carries the slice
    /// of the body's store, and the entry edge the slice of the seed.
    #[test]
    fn subregister_read_in_loop_is_sliced_to_a_phi() {
        let mut tc = qcode::testing::TestContext::new();
        let r0_lo32 = tc.r0_lo32;
        let r0_byte0 = tc.r0_byte0;
        let r1 = tc.r1;
        let r2 = tc.r2;
        let r3 = tc.r3;

        qcode!(
            tc.ctx,
            "
                fn func:
                    <entry>
                        store({r0_lo32}, i32 0x12345678);
                        goto <body>;
                    <body>
                        %al = load(i8, {r0_byte0});
                        store({r3}, %al);
                        %next = load(i32, {r1});
                        store({r0_lo32}, %next);
                        %c = load(i8, {r2});
                        if %c goto <body> else goto <exit>;
                    <exit>
                        return [0x1000];
            "
        );

        let aliases = AliasResult::simple(&tc.ctx);
        let changed = mem2reg(&mut tc.ctx, func, &aliases);
        assert!(changed, "the sliced AL should be promoted");

        let body_block = BasicBlock::from_id(&tc.ctx, body);
        // The loop header gains a 1-byte phi carrying AL around the back-edge.
        assert!(
            body_block
                .params()
                .any(|p| BlockParam::from_id(&tc.ctx, p.id).size() == 1),
            "the loop header should gain a 1-byte phi for the sliced AL:\n{body_block}"
        );
        // The narrow AL load is forwarded away (sliced from the EAX store).
        assert!(
            !body_block.iter().any(|i| matches!(
                i.mnemonic(),
                Mnemonic::Load(Load { ptr, .. }) if *ptr == ValueId::Varnode(r0_byte0)
            )),
            "the AL load should be sliced from the wider EAX store, not left in place:\n{body_block}"
        );
    }

    /// A sub-register read at a *non-zero* offset (`AH`, `r0_byte1`) is not
    /// low-aligned in the covering `EAX` store, so slicing it would need a shift
    /// this pass does not synthesize. It stays a register load (deferred to the
    /// GVN memory pass), and no phi is minted for it.
    #[test]
    fn non_low_aligned_subregister_read_in_loop_is_not_sliced() {
        let mut tc = qcode::testing::TestContext::new();
        let r0_lo32 = tc.r0_lo32;
        let r0_byte1 = tc.r0_byte1;
        let r1 = tc.r1;
        let r2 = tc.r2;
        let r3 = tc.r3;

        qcode!(
            tc.ctx,
            "
                fn func:
                    <entry>
                        store({r0_lo32}, i32 0x12345678);
                        goto <body>;
                    <body>
                        %ah = load(i8, {r0_byte1});
                        store({r3}, %ah);
                        %next = load(i32, {r1});
                        store({r0_lo32}, %next);
                        %c = load(i8, {r2});
                        if %c goto <body> else goto <exit>;
                    <exit>
                        return [0x1000];
            "
        );

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, func, &aliases);

        let body_block = BasicBlock::from_id(&tc.ctx, body);
        assert!(
            body_block.iter().any(|i| matches!(
                i.mnemonic(),
                Mnemonic::Load(Load { ptr, .. }) if *ptr == ValueId::Varnode(r0_byte1)
            )),
            "the non-low-aligned AH load must stay a register load:\n{body_block}"
        );
        assert_eq!(
            body_block.num_params(),
            0,
            "no phi should be minted for the un-sliceable AH read:\n{body_block}"
        );
    }

    /// A register the caller writes before a call and reads after it must not be
    /// promoted to the pre-call value when the callee clobbers it: the post-call
    /// read is the call's output and must stay a register load.
    #[test]
    fn call_clobbered_register_read_not_forwarded_across_call() {
        use qcode::{builder::Builder, testing::TestContext};

        let mut tc = TestContext::new();
        let (r0, r1, reg) = (tc.r0, tc.r1, tc.reg_space);

        // Callee that writes r0 (and only writes it), so r0 is a clobber.
        let callee = Function::make(&mut tc.ctx, "callee".into()).unwrap().id;
        let cbody = tc.ctx.get_or_make_block(0x2000);
        Function::from_id_mut(&mut tc.ctx, callee)
            .set_root(cbody)
            .unwrap();
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x2000);
            let v = b.context_mut().get_const(0x99u64, 8).id();
            b.push_store(v, ValueId::Varnode(r0), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
            unsafe { b.dont_finalize() };
        }
        crate::set_all_call_clobbered_regs(&mut tc.ctx);
        assert!(
            Function::from_id(&tc.ctx, callee)
                .clobbered_regs()
                .unwrap()
                .contains(&r0),
            "callee must record r0 as call-clobbered"
        );

        // Caller: write r0 before the call, read it after.
        let caller = Function::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let cont = tc.ctx.get_or_make_block(0x1100);
        Function::from_id_mut(&mut tc.ctx, caller)
            .set_root(entry)
            .unwrap();
        Function::from_id_mut(&mut tc.ctx, caller).add_block(cont);
        let post_load;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            let pre = b.context_mut().get_const(0x1u64, 8).id();
            b.push_store(pre, ValueId::Varnode(r0), reg);
            b.push_call(callee);
            b.switch_to_block(cont);
            post_load = b.push_load::<false>(ValueId::Varnode(r0), 8, reg).id();
            // Use the loaded value so it is not trivially dead.
            b.push_store(post_load, ValueId::Varnode(r1), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
            unsafe { b.dont_finalize() };
        }
        tc.ctx.add_cfg_edge(entry, cont);

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, caller, &aliases);

        let ValueId::Instruction(post_load_id) = post_load else {
            unreachable!()
        };
        let cont_block = BasicBlock::from_id(&tc.ctx, cont);
        assert!(
            cont_block.instruction_ids().contains(&post_load_id),
            "the post-call read of r0 must survive as a register load, not be \
             forwarded to the pre-call value:\n{cont_block}"
        );
    }

    /// Regression: a wide register write (`r0`, the `ECX` analog) followed by a
    /// read of its low byte (`r0_byte0`, the `CL` analog) must not leave the byte
    /// read orphaned. mem2reg declines to promote the narrow load because a wider
    /// overlapping store exists (`has_overlapping_register_store`), deferring it to
    /// a later overlap-aware GVN slice of "the dominating store". But if mem2reg
    /// *also* removes that wider store — here the first `r0` write is dead, being
    /// overwritten by the second with no intervening `r0` (full-width) read; the
    /// byte read is a different varnode and does not count as a use — then the
    /// deferral target is gone and the narrow load dangles as a free register read
    /// (treated as a function live-in), silently discarding the computed value.
    ///
    /// The invariant: mem2reg must not delete the wide store while leaving an
    /// unpromoted overlapping narrow load behind. A fix may either keep the
    /// dominating store (so GVN can slice it) or slice the byte out of the promoted
    /// value during renaming (the read-side dual of the
    /// `clobber_overlapping_register_vars` sub-register-splice TODO).
    ///
    /// Fixed by preserving a register store an unpromoted overlapping narrow load
    /// depends on (see `remove_promoted_stores`); originally surfaced by the
    /// argpromote `fn_40b970` investigation.
    #[test]
    fn narrow_subregister_read_is_not_orphaned_by_wide_store_removal() {
        use qcode::{builder::Builder, testing::TestContext};

        let mut tc = TestContext::new();
        let (r0, r0_byte0, r2, r3, reg) = (tc.r0, tc.r0_byte0, tc.r2, tc.r3, tc.reg_space);

        let f = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        Function::from_id_mut(&mut tc.ctx, f)
            .set_root(entry)
            .unwrap();

        let (c1_store, byte_load);
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            // Compute a wide value into r0 (the `ECX` build).
            let v = b.push_load::<false>(ValueId::Varnode(r2), 8, reg).id();
            let one = b.context_mut().get_const(1u64, 8).id();
            let c1 = b.push_add(v, one).id();
            c1_store = b.push_store(c1, ValueId::Varnode(r0), reg).id;
            // Read the low byte (the `mov [mem], cl` source) and make it observable.
            let byte_val = b
                .push_load::<false>(ValueId::Varnode(r0_byte0), 1, reg)
                .id();
            let ValueId::Instruction(byte_load_id) = byte_val else {
                unreachable!("a load is an instruction value")
            };
            byte_load = byte_load_id;
            b.push_store(byte_val, ValueId::Varnode(r3), reg);
            // Overwrite r0, making the c1 store a dead overwrite that mem2reg drops.
            let c2 = b.context_mut().get_const(0u64, 8).id();
            b.push_store(c2, ValueId::Varnode(r0), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
            unsafe { b.dont_finalize() };
        }

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, f, &aliases);

        let entry_block = BasicBlock::from_id(&tc.ctx, entry);
        let ids = entry_block.instruction_ids();
        let wide_store_removed = !ids.contains(&c1_store);
        let byte_load_survives = ids.contains(&byte_load);
        assert!(
            !(wide_store_removed && byte_load_survives),
            "mem2reg orphaned the narrow r0_byte0 read: it removed the wider r0 store \
             (the deferred GVN slice's source) while leaving the byte load dangling as a \
             live-in. Keep the dominating store or slice the byte from the promoted \
             value.\n{entry_block}"
        );
    }

    /// A store to a register the callee clobbers, never read before the call, is
    /// dead — the call overwrites it — so mem2reg removes it (the call-clobber
    /// analogue of a store-overwrites-store).
    #[test]
    fn dead_pre_call_store_to_clobbered_register_is_removed() {
        use qcode::{builder::Builder, testing::TestContext};

        let mut tc = TestContext::new();
        let (r0, r1, reg) = (tc.r0, tc.r1, tc.reg_space);

        // Callee that writes (clobbers) r0.
        let callee = Function::make(&mut tc.ctx, "callee".into()).unwrap().id;
        let cbody = tc.ctx.get_or_make_block(0x2000);
        Function::from_id_mut(&mut tc.ctx, callee)
            .set_root(cbody)
            .unwrap();
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x2000);
            let v = b.context_mut().get_const(0x99u64, 8).id();
            b.push_store(v, ValueId::Varnode(r0), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
            unsafe { b.dont_finalize() };
        }
        crate::set_all_call_clobbered_regs(&mut tc.ctx);

        // Caller: write r0 (dead — never read before the call), call, then read r0
        // after (so r0 is a promoted var) into r1.
        let caller = Function::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let cont = tc.ctx.get_or_make_block(0x1100);
        Function::from_id_mut(&mut tc.ctx, caller)
            .set_root(entry)
            .unwrap();
        Function::from_id_mut(&mut tc.ctx, caller).add_block(cont);
        let pre_store;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            let pre = b.context_mut().get_const(0x1u64, 8).id();
            pre_store = b.push_store(pre, ValueId::Varnode(r0), reg).id;
            b.push_call(callee);
            b.switch_to_block(cont);
            let post = b.push_load::<false>(ValueId::Varnode(r0), 8, reg).id();
            b.push_store(post, ValueId::Varnode(r1), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
            unsafe { b.dont_finalize() };
        }
        tc.ctx.add_cfg_edge(entry, cont);

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, caller, &aliases);

        let entry_block = BasicBlock::from_id(&tc.ctx, entry);
        assert!(
            !entry_block.instruction_ids().contains(&pre_store),
            "the dead pre-call store to clobbered r0 must be removed:\n{entry_block}"
        );
    }

    /// A write-only register written before a clobbering call and again after it
    /// (so it is promoted as store-only with 2+ stores) has its dead pre-call
    /// store removed; the call clobbers it with no intervening read.
    #[test]
    fn dead_pre_call_store_to_write_only_clobbered_register_is_removed() {
        use qcode::{builder::Builder, testing::TestContext};

        let mut tc = TestContext::new();
        let (r0, reg) = (tc.r0, tc.reg_space);

        let callee = Function::make(&mut tc.ctx, "callee".into()).unwrap().id;
        let cbody = tc.ctx.get_or_make_block(0x2000);
        Function::from_id_mut(&mut tc.ctx, callee)
            .set_root(cbody)
            .unwrap();
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x2000);
            let v = b.context_mut().get_const(0x99u64, 8).id();
            b.push_store(v, ValueId::Varnode(r0), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
            unsafe { b.dont_finalize() };
        }
        crate::set_all_call_clobbered_regs(&mut tc.ctx);

        let caller = Function::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let cont = tc.ctx.get_or_make_block(0x1100);
        Function::from_id_mut(&mut tc.ctx, caller)
            .set_root(entry)
            .unwrap();
        Function::from_id_mut(&mut tc.ctx, caller).add_block(cont);
        let pre_store;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            let v1 = b.context_mut().get_const(0x1u64, 8).id();
            pre_store = b.push_store(v1, ValueId::Varnode(r0), reg).id; // dead
            b.push_call(callee);
            b.switch_to_block(cont);
            // A second store after the call — never read — making r0 store-only
            // with 2 stores (promoted), as the obfuscated junk does.
            let v2 = b.context_mut().get_const(0x2u64, 8).id();
            b.push_store(v2, ValueId::Varnode(r0), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
            unsafe { b.dont_finalize() };
        }
        tc.ctx.add_cfg_edge(entry, cont);

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, caller, &aliases);

        let entry_block = BasicBlock::from_id(&tc.ctx, entry);
        assert!(
            !entry_block.instruction_ids().contains(&pre_store),
            "the dead pre-call store to write-only clobbered r0 must be removed:\n{entry_block}"
        );
    }

    #[test]
    fn clobbered_register_branch_arg_is_loaded_before_branch() {
        use qcode::{builder::Builder, testing::TestContext};

        let mut tc = TestContext::new();
        let (r0, r1, reg) = (tc.r0, tc.r1, tc.reg_space);

        let callee = Function::make(&mut tc.ctx, "callee".into()).unwrap().id;
        let callee_body = tc.ctx.get_or_make_block(0x2000);
        Function::from_id_mut(&mut tc.ctx, callee)
            .set_root(callee_body)
            .unwrap();
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x2000);
            let v = b.context_mut().get_const(0x99u64, 8).id();
            b.push_store(v, ValueId::Varnode(r0), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
            unsafe { b.dont_finalize() };
        }
        crate::set_all_call_clobbered_regs(&mut tc.ctx);

        let caller = Function::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let left = tc.ctx.get_or_make_block(0x1100);
        let left_cont = tc.ctx.get_or_make_block(0x1200);
        let right = tc.ctx.get_or_make_block(0x1300);
        let join = tc.ctx.get_or_make_block(0x1400);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, caller);
            f.set_root(entry).unwrap();
            f.add_block(left);
            f.add_block(left_cont);
            f.add_block(right);
            f.add_block(join);
        }

        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let one = b.context_mut().get_const(1u64, 8).id();
            let cond = b.context_mut().get_const(1u64, 1).id();
            b.push_store(one, ValueId::Varnode(r0), reg);
            b.push_cbranch(cond, left, right);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, left));
            b.push_call(callee);
        }
        tc.ctx.add_cfg_edge(left, left_cont);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, left_cont));
            b.push_branch(join);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, right));
            let two = b.context_mut().get_const(2u64, 8).id();
            b.push_store(two, ValueId::Varnode(r0), reg);
            b.push_branch(join);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, join));
            let loaded = b.push_load::<false>(ValueId::Varnode(r0), 8, reg).id();
            b.push_store(loaded, ValueId::Varnode(r1), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        }

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, caller, &aliases);

        let left_cont_block = BasicBlock::from_id(&tc.ctx, left_cont);
        let load_before_branch = left_cont_block.iter().any(|insn| {
            matches!(insn.mnemonic(), Mnemonic::Load(load) if load.ptr == ValueId::Varnode(r0))
        });
        assert!(
            load_before_branch,
            "clobbered r0 should be reloaded on the edge into the parameterized join:\n{left_cont_block}"
        );

        let branch = left_cont_block.iter().last().expect("left_cont terminates");
        let Mnemonic::Branch(branch) = branch.mnemonic() else {
            panic!("left_cont should still end in a branch");
        };
        assert_eq!(branch.args.len(), 1, "join branch should pass r0");
        assert!(
            matches!(branch.args[0], ValueId::Instruction(_)),
            "join branch should pass the inserted register load, got {:?}",
            branch.args[0]
        );
    }

    #[test]
    fn load_only_clobbered_register_rerun_reuses_entry_param() {
        use qcode::testing::TestContext;

        let mut tc = TestContext::new();
        let (r0, r1) = (tc.r0, tc.r1);

        qcode!(
            tc.ctx,
            "
            fn callee:
                <callee_entry>
                    store({r0}, i64 0x99);
                    return [i64 0];

            fn caller:
                <entry>
                    if i8 1 goto <clobbered_path> else goto <live_in_path>;

                <clobbered_path>
                    call <callee>;

                <clobbered_cont>
                    goto <join>;

                <live_in_path>
                    goto <join>;

                <join>
                    %loaded = load(i64, {r0});
                    store({r1}, %loaded);
                    return [i64 0];
            "
        );
        tc.ctx.add_cfg_edge(clobbered_path, clobbered_cont);
        crate::set_all_call_clobbered_regs(&mut tc.ctx);

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, caller, &aliases);
        // mem2reg must NOT mint a root param for a live-in register input (here r0,
        // live-in via the non-clobbered path): the call interface is argpromote's.
        assert_eq!(
            BasicBlock::from_id(&tc.ctx, entry).num_params(),
            0,
            "mem2reg must not mint a root param for a live-in register input"
        );
    }

    #[test]
    fn clobbered_register_branch_arg_rerun_reuses_edge_load() {
        use qcode::testing::TestContext;

        let mut tc = TestContext::new();
        let (r0, r1) = (tc.r0, tc.r1);

        qcode!(
            tc.ctx,
            "
            fn callee:
                <callee_entry>
                    store({r0}, i64 0x99);
                    return [i64 0];

            fn caller:
                <entry>
                    store({r0}, i64 1);
                    if i8 1 goto <clobbered_path> else goto <live_in_path>;

                <clobbered_path>
                    call <callee>;

                <clobbered_cont>
                    goto <join>;

                <live_in_path>
                    store({r0}, i64 2);
                    goto <join>;

                <join>
                    %loaded = load(i64, {r0});
                    store({r1}, %loaded);

                    goto <exit>;

                <exit>
                    store({r0}, i64 7);
                    return [i64 0];
            "
        );
        tc.ctx.add_cfg_edge(clobbered_path, clobbered_cont);
        crate::set_all_call_clobbered_regs(&mut tc.ctx);

        let edge_reload_count = |ctx: &qcode::context::Context<'_>| {
            BasicBlock::from_id(ctx, clobbered_cont)
                .iter()
                .filter(|insn| {
                    matches!(
                        insn.mnemonic(),
                        Mnemonic::Load(Load { ptr, .. }) if *ptr == ValueId::Varnode(r0)
                    )
                })
                .count()
        };

        let aliases = AliasResult::simple(&tc.ctx);
        assert!(mem2reg(&mut tc.ctx, caller, &aliases));
        assert_eq!(
            edge_reload_count(&tc.ctx),
            1,
            "the clobbered edge should get one r0 reload before the join branch"
        );

        let aliases = AliasResult::simple(&tc.ctx);
        assert!(
            !mem2reg(&mut tc.ctx, caller, &aliases),
            "the second run should converge instead of inserting another edge reload"
        );
        assert_eq!(
            edge_reload_count(&tc.ctx),
            1,
            "re-running mem2reg must reuse the existing clobbered-edge r0 reload"
        );
    }

    /// Same shape as `load_only_clobbered_register_rerun_reuses_entry_param`, but
    /// the promoted register has no `name`. Param reuse keys on the param's
    /// `origin` (the source varnode) rather than its display name, so an unnamed
    /// register still converges on re-run instead of accumulating a duplicate
    /// entry param each pass.
    #[test]
    fn load_only_unnamed_register_rerun_reuses_entry_param() {
        use qcode::testing::TestContext;

        let mut tc = TestContext::new();
        // A register-space varnode with no name, at an offset clear of r0..r3 and
        // the r0 sub-registers.
        let unnamed = Varnode::make(&mut tc.ctx, 40, 8, tc.reg_space).id;
        assert!(
            Varnode::from_id(&tc.ctx, unnamed).name().is_none(),
            "the test varnode must be unnamed to exercise origin-based reuse"
        );
        let r1 = tc.r1;

        qcode!(
            tc.ctx,
            "
            fn callee:
                <callee_entry>
                    store({unnamed}, i64 0x99);
                    return [i64 0];

            fn caller:
                <entry>
                    if i8 1 goto <clobbered_path> else goto <live_in_path>;

                <clobbered_path>
                    call <callee>;

                <clobbered_cont>
                    goto <join>;

                <live_in_path>
                    goto <join>;

                <join>
                    %loaded = load(i64, {unnamed});
                    store({r1}, %loaded);
                    return [i64 0];
            "
        );
        tc.ctx.add_cfg_edge(clobbered_path, clobbered_cont);
        crate::set_all_call_clobbered_regs(&mut tc.ctx);

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, caller, &aliases);
        // mem2reg must NOT mint a root param for a live-in register input: the call
        // interface is argpromote's. The unnamed register stays a plain load.
        assert_eq!(
            BasicBlock::from_id(&tc.ctx, entry).num_params(),
            0,
            "mem2reg must not mint a root param for a live-in register input"
        );
    }

    /// A store overwritten on one branch arm must not be marked dead when a
    /// sibling arm never overwrites it: that sibling relies on the store as its
    /// live-out register value. (Bug 1: path-insensitive dead-store marking.)
    #[test]
    fn store_on_one_arm_does_not_kill_live_out_store_on_sibling() {
        use qcode::{builder::Builder, testing::TestContext};

        let mut tc = TestContext::new();
        let (r0, reg) = (tc.r0, tc.reg_space);

        let f = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let left = tc.ctx.get_or_make_block(0x1100);
        let right = tc.ctx.get_or_make_block(0x1200);
        let join = tc.ctx.get_or_make_block(0x1300);
        {
            let mut fr = Function::from_id_mut(&mut tc.ctx, f);
            fr.set_root(entry).unwrap();
            fr.add_block(left);
            fr.add_block(right);
            fr.add_block(join);
        }

        let entry_store;
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let one = b.context_mut().get_const(1u64, 8).id();
            let cond = b.context_mut().get_const(1u64, 1).id();
            entry_store = b.push_store(one, ValueId::Varnode(r0), reg).id;
            b.push_cbranch(cond, left, right);
        }
        {
            // Left arm overwrites r0 before reading it.
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, left));
            let two = b.context_mut().get_const(2u64, 8).id();
            b.push_store(two, ValueId::Varnode(r0), reg);
            b.push_branch(join);
        }
        {
            // Right arm never touches r0: it leaves the function with the entry value.
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, right));
            b.push_branch(join);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, join));
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        }

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, f, &aliases);

        let entry_block = BasicBlock::from_id(&tc.ctx, entry);
        assert!(
            entry_block.instruction_ids().contains(&entry_store),
            "entry store r0=1 must survive: the right arm never overwrites it, so \
             an overwrite on the left arm must not mark it dead:\n{entry_block}"
        );
    }

    /// Mutually-exclusive BranchInd successors must not share reaching values: a
    /// store made while visiting one successor must be invisible to a sibling
    /// successor. (Bug 2: visit_successors shared one frame across siblings.)
    ///
    /// Both successors load r0 (each dominated only by the entry def) and then
    /// redefine it. Whichever sibling `successors()` happens to visit second is
    /// the one that observes the leak, so asserting *both* loads resolve to the
    /// entry value catches the bug regardless of the (nondeterministic) iteration
    /// order.
    #[test]
    fn branchind_sibling_successors_do_not_share_reaching_value() {
        use qcode::{builder::Builder, testing::TestContext};

        let mut tc = TestContext::new();
        let (r0, r1, r2, reg) = (tc.r0, tc.r1, tc.r2, tc.reg_space);

        let f = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let s1 = tc.ctx.get_or_make_block(0x1100);
        let s2 = tc.ctx.get_or_make_block(0x1200);
        let exit = tc.ctx.get_or_make_block(0x1300);
        {
            let mut fr = Function::from_id_mut(&mut tc.ctx, f);
            fr.set_root(entry).unwrap();
            fr.add_block(s1);
            fr.add_block(s2);
            fr.add_block(exit);
        }

        let zero;
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            zero = b.context_mut().get_const(0u64, 8).id();
            let target = b.context_mut().get_const(0x1100u64, 8).id();
            b.push_store(zero, ValueId::Varnode(r0), reg);
            b.push_branchind(target);
        }
        tc.ctx.add_cfg_edge(entry, s1);
        tc.ctx.add_cfg_edge(entry, s2);

        // Each sibling loads r0 (must see the entry def), sinks it, then
        // redefines r0 to a distinct value that must not leak to its sibling.
        let sink1;
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, s1));
            let loaded = b.push_load::<false>(ValueId::Varnode(r0), 8, reg).id();
            sink1 = b.push_store(loaded, ValueId::Varnode(r1), reg).id;
            let one = b.context_mut().get_const(1u64, 8).id();
            b.push_store(one, ValueId::Varnode(r0), reg);
            b.push_branch(exit);
        }
        let sink2;
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, s2));
            let loaded = b.push_load::<false>(ValueId::Varnode(r0), 8, reg).id();
            sink2 = b.push_store(loaded, ValueId::Varnode(r2), reg).id;
            let two = b.context_mut().get_const(2u64, 8).id();
            b.push_store(two, ValueId::Varnode(r0), reg);
            b.push_branch(exit);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, exit));
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        }

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, f, &aliases);

        for (sink, name) in [(sink1, "s1"), (sink2, "s2")] {
            let Mnemonic::Store(store) = Instruction::from_id(&tc.ctx, sink).mnemonic().clone()
            else {
                panic!("{name} sink should remain a store");
            };
            assert_eq!(
                store.src, zero,
                "{name}'s load of r0 must resolve to the entry def (0), not the \
                 sibling successor's redefinition"
            );
        }
    }

    /// A varnode accessed at a width that differs from its own width must not be
    /// promoted: forwarding the stored value into a differently-sized load would
    /// mis-size the result. (Hardening: varnode access-size consistency.)
    #[test]
    fn varnode_access_size_mismatch_blocks_promotion() {
        use qcode::{builder::Builder, testing::TestContext};

        let mut tc = TestContext::new();
        let (r0, r1, reg) = (tc.r0, tc.r1, tc.reg_space); // r0 is 8 bytes wide

        let f = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        Function::from_id_mut(&mut tc.ctx, f)
            .set_root(entry)
            .unwrap();

        let mismatched_load;
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let val = b.context_mut().get_const(0x1122_3344_5566_7788u64, 8).id();
            b.push_store(val, ValueId::Varnode(r0), reg); // 8-byte store: matches r0
            // 4-byte load through the 8-byte r0 — a width mismatch that must
            // disqualify r0 from promotion.
            mismatched_load = b.push_load::<false>(ValueId::Varnode(r0), 4, reg).id();
            b.push_store(mismatched_load, ValueId::Varnode(r1), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        }

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, f, &aliases);

        let ValueId::Instruction(load_id) = mismatched_load else {
            unreachable!()
        };
        let entry_block = BasicBlock::from_id(&tc.ctx, entry);
        assert!(
            entry_block.instruction_ids().contains(&load_id),
            "a 4-byte load of the 8-byte register r0 must not be promoted; the \
             size mismatch disqualifies r0:\n{entry_block}"
        );
    }

    /// A var live into a join that is also a direct `BranchInd` successor must
    /// not be promoted: the implicit (argument-less) jump-table edge into the
    /// join cannot supply the block-param argument, so the var stays in memory
    /// rather than yielding a param with an unbound edge. (Gap 3.)
    #[test]
    fn var_live_into_implicit_edge_join_is_not_promoted() {
        use qcode::{builder::Builder, testing::TestContext};

        let mut tc = TestContext::new();
        let (r0, r1, reg) = (tc.r0, tc.r1, tc.reg_space);

        let f = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let other = tc.ctx.get_or_make_block(0x1100);
        let join = tc.ctx.get_or_make_block(0x1200);
        {
            let mut fr = Function::from_id_mut(&mut tc.ctx, f);
            fr.set_root(entry).unwrap();
            fr.add_block(other);
            fr.add_block(join);
        }

        // entry stores r0, then an indirect jump that can land on `join` directly
        // (an argument-less edge) or on `other`.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let zero = b.context_mut().get_const(0u64, 8).id();
            let target = b.context_mut().get_const(0x1200u64, 8).id();
            b.push_store(zero, ValueId::Varnode(r0), reg);
            b.push_branchind(target);
        }
        tc.ctx.add_cfg_edge(entry, join); // implicit (BranchInd) edge into the join
        tc.ctx.add_cfg_edge(entry, other);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, other));
            let one = b.context_mut().get_const(1u64, 8).id();
            b.push_store(one, ValueId::Varnode(r0), reg);
            b.push_branch(join); // arg-carrying edge into the join
        }
        let join_load;
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, join));
            join_load = b.push_load::<false>(ValueId::Varnode(r0), 8, reg).id();
            b.push_store(join_load, ValueId::Varnode(r1), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        }

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, f, &aliases);

        let ValueId::Instruction(load_id) = join_load else {
            unreachable!()
        };
        let join_block = BasicBlock::from_id(&tc.ctx, join);
        assert!(
            join_block.instruction_ids().contains(&load_id),
            "r0's load must remain: the join is reached by an argument-less \
             BranchInd edge, so r0 cannot be safely promoted:\n{join_block}"
        );
        assert_eq!(
            join_block.num_params(),
            0,
            "the implicit-edge join must not receive a block param:\n{join_block}"
        );
    }

    #[test]
    fn promoted_temp_store_kept_when_its_load_is_not_forwarded() {
        // Regression: a temporary-space varnode is both stored and loaded (so it
        // is a promotion candidate), but its load lives in a block the renaming
        // DFS never reaches, so the load is left in place. The store must NOT be
        // removed — dropping it while the load survives stranded an undefined
        // unique-space read (the SLEIGH `v0`/`v1` leak in indirect-call lifts).
        use qcode::{builder::Builder, testing::TestContext};

        let mut tc = TestContext::new();
        let f = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        // `orphan` holds the load but is not wired as a CFG successor of `entry`,
        // standing in for a successor the renaming DFS does not visit.
        let orphan = tc.ctx.get_or_make_block(0x1100);
        {
            let mut fr = Function::from_id_mut(&mut tc.ctx, f);
            fr.set_root(entry).unwrap();
            fr.add_block(orphan);
        }

        let temp = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let temp = b.make_temp(4);
            let val = b.context_mut().get_const(7u64, 4).id();
            let space = Varnode::from_id(b.context(), temp).space().id;
            b.push_store(val, ValueId::Varnode(temp), space);
            let ret = b.context_mut().get_const(0u64, 4).id();
            b.push_return(ret);
            temp
        };
        let load_id = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, orphan));
            let space = Varnode::from_id(b.context(), temp).space().id;
            let load = b.push_load::<false>(ValueId::Varnode(temp), 4, space).id();
            let ret = b.context_mut().get_const(0u64, 4).id();
            b.push_return(ret);
            load
        };

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, f, &aliases);

        let ValueId::Instruction(load_id) = load_id else {
            unreachable!()
        };
        // The load survived (was not forwarded)...
        assert!(
            BasicBlock::from_id(&tc.ctx, orphan)
                .instruction_ids()
                .contains(&load_id),
            "precondition: the unreached load is left in place"
        );
        // ...so its store must survive too — no dangling temp read.
        let entry_has_store = BasicBlock::from_id(&tc.ctx, entry).iter().any(|insn| {
            matches!(
                insn.mnemonic(),
                Mnemonic::Store(Store { ptr, .. }) if *ptr == ValueId::Varnode(temp)
            )
        });
        assert!(
            entry_has_store,
            "the temp store must be kept while its load survives (no v0/v1 leak)"
        );
    }
}

// ----- pass ------------------------------------------------------------------

use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct Mem2RegPass;

impl FunctionPass for Mem2RegPass {
    const NAME: &'static str = "mem2reg";
    fn description(&self) -> &'static str {
        "Promote memory loads/stores to SSA block params"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        env: &PipelineEnv,
    ) -> Result<bool, String> {
        // Per-function pass: scope the alias oracle to this function so the stage
        // is O(program) total, not O(functions × program).
        let aliases = AliasResult::simple_for_function(ctx, fun_id);
        // Resolve `@SP` so canonical `@SP ± N` slots are recognised; `None` when
        // the function has no incoming stack-pointer param (legacy literal path).
        let sp_reg = ctx.registers[&env.cfg.stack_pointer];
        let sp_param = incoming_sp_param(ctx, fun_id, sp_reg);
        Ok(mem2reg_framed(ctx, fun_id, &aliases, sp_param))
    }
}

crate::register_function_pass!(Mem2RegPass);
