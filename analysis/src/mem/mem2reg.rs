use jstd::graph::analysis::compute_dominators;
use qcode::space::SpaceType;
#[cfg(test)]
use qcode::value::block::BlockRef;
use qcode::value::{
    BasicBlock, BlockId, BlockParam, BlockParamId, Function, FunctionId, Instruction, Value,
    ValueId, ValueRef, Varnode, VarnodeId,
    insn::{Branch, CBranch, InstructionId, InstructionRef, Load, Mnemonic, Range, Store, Zext},
};
use qcode::{builder::Builder, context::Context, value::FunctionRef};
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
        // Precompute the liveness inputs in one sweep so the collection and
        // block-param phases share a single per-var computation (memoized) instead
        // of rescanning the whole function for each variable.
        let mut live_in_cache = LiveInBlocks::new(&*self.ctx, self.function_id);

        let Promotable {
            vars,
            sizes,
            no_root_params,
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
        } = self.insert_block_params(
            &vars,
            &sizes,
            &no_root_params,
            &frontier,
            &mut live_in_cache,
        );

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

        let mut state = RenameState::new(&var_params, &vars, register_clobbers, changed);
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

/// Whether the call terminating `block` (if any) clobbers register `var`.
///
/// A direct call clobbers the registers in its target's recorded clobber set,
/// overlap-aware so a callee writing RAX clobbers a caller's EAX read. A
/// `CallInd` has an unknown target and conservatively clobbers every register; a
/// target with no recorded clobber set is likewise treated conservatively.
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
            let clobbered: Option<Vec<VarnodeId>> = Function::from_id(ctx, call.target)
                .clobbered_regs()
                .map(<[VarnodeId]>::to_vec);
            match clobbered {
                Some(clobbered) => clobbered
                    .iter()
                    .any(|&c| aliases.may_alias(ctx, ValueId::Varnode(c), var)),
                None => true,
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
    /// Promoted vars that should not receive a function-entry param. These are
    /// sub-register reads covered by a wider overlapping seed store (e.g. an EDX
    /// read after argpromote's RDX seed). The initial load stays as a register
    /// read, while later exact stores/loads can still be promoted in the body.
    no_root_params: HashSet<ValueId>,
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

    fn collect_promotable_vars(&self, live_in_cache: &mut LiveInBlocks) -> Promotable {
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
                    let has_overlapping_register_store = register_stores.iter().any(|&store| {
                        store != var && wider_register_store_contains(self.ctx, store, var)
                    });
                    if !has_overlapping_register_store
                        && self.root_id.is_some_and(|r| {
                            self.live_in_blocks_cached(var, live_in_cache).contains(&r)
                        })
                    {
                        vars.insert(var);
                    }
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

        let no_root_params = vars
            .iter()
            .copied()
            .filter(|&var| {
                register_varnode(self.ctx, var).is_some()
                    && register_stores
                        .iter()
                        .any(|&store| wider_register_store_contains(self.ctx, store, var))
            })
            .collect();

        Promotable {
            vars,
            sizes,
            no_root_params,
        }
    }

    fn insert_block_params(
        &mut self,
        vars: &HashSet<ValueId>,
        sizes: &HashMap<ValueId, usize>,
        no_root_params: &HashSet<ValueId>,
        frontier: &HashMap<BlockId, HashSet<BlockId>>,
        live_in_cache: &mut LiveInBlocks,
    ) -> InsertedBlockParams {
        let mut var_params: BlockParamAssignments = HashMap::default();
        let mut changed = false;
        let mut excluded: HashSet<ValueId> = HashSet::default();

        // One pass over the function yields the store-block set for every var,
        // so `find_phi_insert_positions` is a lookup rather than a full block
        // rescan per variable.
        let stores_by_var = {
            let function = Function::from_id(self.ctx, self.function_id);
            blocks_storing_to_vars(&function, vars)
        };
        let no_store_blocks = HashSet::default();

        for &var in vars {
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

            let live_in = self.live_in_blocks_cached(var, live_in_cache);
            let block_containing_store = stores_by_var.get(&var).unwrap_or(&no_store_blocks);
            let phi_positions =
                find_phi_insert_positions(block_containing_store, frontier, &live_in);

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

            // Add a root block param for variables that are live-in to the function
            // (i.e. have some upward-exposed use not covered by any dominator store).
            let root_id = self.root_id();
            let root_has_param = var_params
                .get(&root_id)
                .is_some_and(|m| m.contains_key(&var));
            if !root_has_param && live_in.contains(&root_id) && !no_root_params.contains(&var) {
                let (param_id, inserted) =
                    self.get_or_insert_param_for_var(root_id, size, var, var_name.as_deref());
                changed |= inserted;
                var_params.entry(root_id).or_default().insert(var, param_id);
            }
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

    /// Memoized live-in blocks for `var` (see [`LiveInBlocks`]). The returned set
    /// is cloned so the caller may freely take further `&mut self` borrows; the
    /// clone is cheap relative to the liveness propagation.
    fn live_in_blocks_cached(&self, var: ValueId, cache: &mut LiveInBlocks) -> HashSet<BlockId> {
        cache.get(self.ctx, var, self.aliases)
    }
}

/// How the call terminating a block clobbers register vars, classified once per
/// block so liveness need not re-read the terminator (and re-fetch the callee's
/// clobber set) for every variable.
enum CallClobber {
    /// A `CallInd`, or a `Call` whose target has no recorded clobber set:
    /// conservatively clobbers every register var.
    All,
    /// A `Call` with a known clobbered-register set (overlap decided per var).
    Regs(Vec<VarnodeId>),
}

/// Precomputed inputs for per-variable live-in analysis, shared across every
/// variable in one mem2reg run.
///
/// A block is live-in for `var` if it has an upward-exposed use of `var` (a load
/// not preceded by a store to it in that block), or a successor is live-in and
/// the block does not define `var`. A call that clobbers `var` counts as a
/// definition: a read reachable only through that call is the call's *output*,
/// not a value flowing in from the entry (this keeps a post-call register read of
/// e.g. `RAX` from being promoted to a spurious root parameter).
///
/// The store / upward-exposed block sets and the call classification are gathered
/// in a single sweep over the function, so each variable's liveness is only the
/// backward propagation from its seeds — turning the old O(vars × instructions)
/// per-var rescan into one O(instructions) sweep plus cheap per-var work. Results
/// are memoized because `collect_promotable_vars` and `insert_block_params` query
/// the same variables.
struct LiveInBlocks {
    /// Blocks containing a store to the pointer (a definition site).
    store_blocks: HashMap<ValueId, HashSet<BlockId>>,
    /// Blocks with an upward-exposed load of the pointer — the liveness seeds.
    upward_exposed: HashMap<ValueId, HashSet<BlockId>>,
    /// Call-terminated blocks and what each clobbers.
    call_blocks: Vec<(BlockId, CallClobber)>,
    /// The function's own blocks. The backward liveness walk stays inside this
    /// set: a tail-call edge into another function is a real CFG predecessor
    /// edge, but following it would let a foreign block flip this function's
    /// live-in decisions.
    blocks: HashSet<BlockId>,
    /// Memoized live-in sets, keyed by variable.
    memo: HashMap<ValueId, HashSet<BlockId>>,
}

impl LiveInBlocks {
    fn new(ctx: &Context, function_id: FunctionId) -> Self {
        let mut store_blocks: HashMap<ValueId, HashSet<BlockId>> = HashMap::default();
        let mut upward_exposed: HashMap<ValueId, HashSet<BlockId>> = HashMap::default();
        let mut call_blocks = Vec::new();
        let mut blocks: HashSet<BlockId> = HashSet::default();
        let mut stored_here: HashSet<ValueId> = HashSet::default();

        for block in Function::from_id(ctx, function_id).blocks() {
            let block_id = block.id;
            blocks.insert(block_id);
            stored_here.clear();
            for insn in block.iter() {
                match insn.mnemonic() {
                    Mnemonic::Store(Store { ptr, .. }) => {
                        store_blocks.entry(*ptr).or_default().insert(block_id);
                        stored_here.insert(*ptr);
                    }
                    Mnemonic::Load(Load { ptr, .. }) if !stored_here.contains(ptr) => {
                        upward_exposed.entry(*ptr).or_default().insert(block_id);
                    }
                    _ => {}
                }
            }

            match block.iter().last().map(|i| i.mnemonic().clone()) {
                Some(Mnemonic::CallInd(_)) => call_blocks.push((block_id, CallClobber::All)),
                Some(Mnemonic::Call(call)) => {
                    let clobber = match Function::from_id(ctx, call.target).clobbered_regs() {
                        Some(regs) => CallClobber::Regs(regs.to_vec()),
                        None => CallClobber::All,
                    };
                    call_blocks.push((block_id, clobber));
                }
                _ => {}
            }
        }

        Self {
            store_blocks,
            upward_exposed,
            call_blocks,
            blocks,
            memo: HashMap::default(),
        }
    }

    /// Memoized live-in block set for `var`.
    fn get(&mut self, ctx: &Context, var: ValueId, aliases: &AliasResult) -> HashSet<BlockId> {
        if let Some(cached) = self.memo.get(&var) {
            return cached.clone();
        }
        let live_in = self.compute(ctx, var, aliases);
        self.memo.insert(var, live_in.clone());
        live_in
    }

    fn compute(&self, ctx: &Context, var: ValueId, aliases: &AliasResult) -> HashSet<BlockId> {
        // A store defines `var`; a call that clobbers it does too (only register
        // vars can be call-clobbered).
        let mut defined = self.store_blocks.get(&var).cloned().unwrap_or_default();
        if register_varnode(ctx, var).is_some() {
            for (block_id, clobber) in &self.call_blocks {
                let clobbers = match clobber {
                    CallClobber::All => true,
                    CallClobber::Regs(regs) => regs
                        .iter()
                        .any(|&c| aliases.may_alias(ctx, ValueId::Varnode(c), var)),
                };
                if clobbers {
                    defined.insert(*block_id);
                }
            }
        }

        // Propagate backward from the upward-exposed seeds to predecessors,
        // stopping where `var` is defined. Each block enters the worklist at most
        // once, so this is O(edges in the live region) rather than an O(B²)
        // re-sweep to a fixpoint.
        let mut live_in = self.upward_exposed.get(&var).cloned().unwrap_or_default();
        let mut worklist: Vec<BlockId> = live_in.iter().copied().collect();
        while let Some(block_id) = worklist.pop() {
            for (_, pred) in BasicBlock::from_id(ctx, block_id).predecessors() {
                if self.blocks.contains(&pred) && !defined.contains(&pred) && live_in.insert(pred) {
                    worklist.push(pred);
                }
            }
        }
        live_in
    }
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
        // A tail-call edge into another function's entry carries that function's
        // own params, meaningless to wire from this promotion; leave it untouched.
        if !self.in_function(edge.target) {
            return edge.existing_args.to_vec();
        }
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
        for &block_id in &block_ids {
            for &insn_id in BasicBlock::from_id(self.ctx, block_id).instruction_ids() {
                if let Mnemonic::Load(Load { ptr, .. }) =
                    Instruction::from_id(self.ctx, insn_id).mnemonic()
                    && vars.contains(ptr)
                {
                    vars_with_surviving_loads.insert(*ptr);
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

        let frame = state.frames.last_mut().unwrap();
        state.preserved_stores.insert(store_insn);
        for (var, reaching_store) in clobbered {
            if let Some(id) = reaching_store {
                state.preserved_stores.insert(id);
            }
            frame.insert(var, FrameEntry::Clobbered);
        }
    }
}

#[cfg(test)]
fn block_contains_store_to_var(block: &BlockRef, var: ValueId) -> bool {
    block.iter().any(|insn| {
        if let Mnemonic::Store(Store { ptr, .. }) = insn.mnemonic() {
            *ptr == var
        } else {
            false
        }
    })
}

/// Blocks that contain a `Store` to `var`. Single-var form; production uses the
/// batched [`blocks_storing_to_vars`] instead.
#[cfg(test)]
fn blocks_storing_to_var(function: &FunctionRef, var: ValueId) -> HashSet<BlockId> {
    function
        .iter()
        .filter(|b| block_contains_store_to_var(b, var))
        .map(|b| b.id)
        .collect()
}

/// For each promotable var in `vars`, the set of blocks that store to it,
/// computed in a single pass over the function. This replaces rescanning every
/// block once per variable inside `find_phi_insert_positions` (which was
/// quadratic — vars × instructions — and the dominant mem2reg cost on large
/// functions).
fn blocks_storing_to_vars(
    function: &FunctionRef,
    vars: &HashSet<ValueId>,
) -> HashMap<ValueId, HashSet<BlockId>> {
    let mut map: HashMap<ValueId, HashSet<BlockId>> = HashMap::default();
    for block in function.iter() {
        let bid = block.id;
        for insn in block.iter() {
            if let Mnemonic::Store(Store { ptr, .. }) = insn.mnemonic()
                && vars.contains(ptr)
            {
                map.entry(*ptr).or_default().insert(bid);
            }
        }
    }
    map
}

fn find_phi_insert_positions(
    block_containing_store: &HashSet<BlockId>,
    frontier: &HashMap<BlockId, HashSet<BlockId>>,
    live_in: &HashSet<BlockId>,
) -> HashSet<BlockId> {
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
        register_clobbers: HashMap<ValueId, Vec<ValueId>>,
        changed: bool,
    ) -> Self {
        Self {
            var_params,
            vars,
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
    /// True if `block` belongs to the function being promoted.
    ///
    /// A tail-call `Branch`/`CBranch` names another function's *entry* as its
    /// terminator target — a legitimate inter-procedural jump (the CFG edge is
    /// stripped by `split`'s `remove_cross_function_edges`, but the terminator
    /// still records the target). The rename walk follows terminator targets, so
    /// it must stop at that boundary itself — exactly like `claimed_from` in
    /// split.rs — rather than descend into a foreign function's blocks, whose
    /// loads reference the same globally-interned stack slots yet have no
    /// reaching definition in *this* function's promotion.
    fn in_function(&self, block: BlockId) -> bool {
        BasicBlock::from_id(self.ctx, block).parent().map(|f| f.id) == Some(self.function_id)
    }

    fn decide_values_start_from(&mut self, block: BlockId, state: &mut RenameState<'_>) {
        // Never cross a tail-call boundary into another function's blocks.
        if !self.in_function(block) {
            return;
        }
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
    /// time rather than re-derived per var. A `CallInd`, or a callee with no
    /// recorded clobber set, is treated conservatively as clobbering every
    /// promoted register var.
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
                let clobbered = Function::from_id(self.ctx, call.target)
                    .clobbered_regs()
                    .map(<[VarnodeId]>::to_vec);
                match clobbered {
                    None => register_vars().collect(),
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
        let live_in = LiveInBlocks::new(&ctx, test).get(&ctx, A.into(), &aliases);

        let stores = blocks_storing_to_var(&function, A.into());
        let result = find_phi_insert_positions(&stores, frontier, &live_in);

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

        let stores = blocks_storing_to_var(&function, A.into());
        let result = find_phi_insert_positions(&stores, &frontier, &live_in);

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
        let live_in = LiveInBlocks::new(&ctx, test).get(&ctx, A.into(), &aliases);

        let stores = blocks_storing_to_var(&function, A.into());
        let result = find_phi_insert_positions(&stores, frontier, &live_in);

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
    fn sp_slot_function(
        tc: &mut qcode::testing::TestContext,
    ) -> (FunctionId, ValueId, VarnodeId) {
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
            loads_after, loads_before,
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

    #[test]
    #[ignore = "regression on the argpromote branch: an overlapping full-register \
                store is wrongly eliminated while a subregister load still reads \
                state derived from it. Fix on this branch before merge."]
    fn overlapping_store_survives_when_subregister_load_remains() {
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

        let full_store;
        let byte_load;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            let value = b.context_mut().get_const(0x12345678, 4).id();
            full_store = b.push_store(value, full, tc.reg_space).id;
            let full_load = b.push_load::<false>(full, 4, tc.reg_space).id();
            b.push_store(full_load, full_sink, tc.reg_space);
            byte_load = b.push_load::<false>(low_byte, 1, tc.reg_space).id();
            b.push_store(byte_load, byte_sink, tc.reg_space);
            unsafe { b.dont_finalize() };
        }

        let aliases = AliasResult::simple(&tc.ctx);
        mem2reg(&mut tc.ctx, fun_id, &aliases);

        let ValueId::Instruction(byte_load_id) = byte_load else {
            panic!("push_load should produce an instruction value");
        };
        let block = BasicBlock::from_id(&tc.ctx, block_id);
        assert!(
            block.instruction_ids().contains(&full_store),
            "the full-register store must remain because the later byte load \
             reads register state derived from it:\n{block}"
        );
        assert!(
            block.instruction_ids().contains(&byte_load_id),
            "the sub-register load should remain after the overlapping full \
             register write clobbers its promoted entry value:\n{block}"
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
        assert!(mem2reg(&mut tc.ctx, caller, &aliases));
        let entry_params_after_first = BasicBlock::from_id(&tc.ctx, entry).num_params();
        assert_eq!(
            entry_params_after_first, 1,
            "the live-in r0 path should create exactly one entry param"
        );

        let aliases = AliasResult::simple(&tc.ctx);
        assert!(
            !mem2reg(&mut tc.ctx, caller, &aliases),
            "the second run should converge instead of reporting a duplicate-param change"
        );
        assert_eq!(
            BasicBlock::from_id(&tc.ctx, entry).num_params(),
            entry_params_after_first,
            "re-running mem2reg must reuse the existing r0 entry param"
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
        assert!(mem2reg(&mut tc.ctx, caller, &aliases));
        let entry_params_after_first = BasicBlock::from_id(&tc.ctx, entry).num_params();
        assert_eq!(
            entry_params_after_first, 1,
            "the live-in unnamed register should create exactly one entry param"
        );

        let aliases = AliasResult::simple(&tc.ctx);
        assert!(
            !mem2reg(&mut tc.ctx, caller, &aliases),
            "the second run should converge instead of duplicating the unnamed entry param"
        );
        assert_eq!(
            BasicBlock::from_id(&tc.ctx, entry).num_params(),
            entry_params_after_first,
            "re-running mem2reg must reuse the existing unnamed-register entry param"
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

    /// Regression: the rename walk follows terminator targets, and a tail-call
    /// `Branch` names another function's entry as its target. Stack-slot literals
    /// are interned per `(offset, StackAddress)`, so the *same* `ValueId` appears
    /// in both functions. If the walk descends across that boundary it decides a
    /// reaching value for this function's promoted slot inside a block it never
    /// analyzed — and a stack slot with no reaching definition panics by design.
    /// The walk must stop at the function boundary, like split's `claimed_from`.
    #[test]
    fn rename_walk_stops_at_tail_call_into_another_function() {
        use qcode::{
            builder::Builder,
            space::{Space, SpaceType},
            testing::TestContext,
        };

        let mut tc = TestContext::new();
        let ctx = &mut tc.ctx;

        // A StackAddress-typed slot literal `s`, shared (by interning) between the
        // promoted function and the foreign block it must not descend into.
        let stack = ctx.add_space(Space {
            name: Some(Box::from("stack")),
            word_size: 1,
            addr_size: 8,
            ty: SpaceType::Ram,
        });
        let sa = ctx.types.get_or_make_stack_address(8, Some(stack));
        let slot = qcode::types::stack_base(8).wrapping_sub(8);
        let s = ValueId::Literal(ctx.values.get_or_make_typed_literal(slot, sa, 8));

        // Function A promotes `s` (stored+loaded on one CBranch arm); the other
        // arm tail-jumps into function B, which loads `s`. Neither `a_entry` nor
        // the tail arm defines `s`, so on the tail path `s` has no reaching value.
        let a = Function::make(ctx, "promoted".into()).unwrap().id;
        let a_entry = ctx.get_or_make_block(0x1000);
        let a_store = ctx.get_or_make_block(0x1100);
        let a_tail = ctx.get_or_make_block(0x1200);
        {
            let mut fr = Function::from_id_mut(ctx, a);
            fr.set_root(a_entry).unwrap();
            fr.add_block(a_store);
            fr.add_block(a_tail);
        }

        let b = Function::make(ctx, "callee".into()).unwrap().id;
        let b_entry = ctx.get_or_make_block(0x2000);
        Function::from_id_mut(ctx, b).set_root(b_entry).unwrap();

        // a_entry: if 1 goto a_store else goto a_tail
        {
            let mut bld = Builder::from_block(BasicBlock::from_id_mut(ctx, a_entry));
            let cond = bld.context_mut().get_const(1u64, 1).id();
            bld.push_cbranch(cond, a_store, a_tail);
        }
        // a_store: store(s, 0x42); load(s); return  -> makes `s` promotable in A
        {
            let mut bld = Builder::from_block(BasicBlock::from_id_mut(ctx, a_store));
            let val = bld.context_mut().get_const(0x42u64, 8).id();
            bld.push_store(val, s, stack);
            bld.push_load::<false>(s, 8, stack);
            let ret = bld.context_mut().get_const(0u64, 8).id();
            bld.push_return(ret);
        }
        // a_tail: goto b_entry  (a tail jump into function B, no store of `s`)
        {
            let mut bld = Builder::from_block(BasicBlock::from_id_mut(ctx, a_tail));
            bld.push_branch(b_entry);
        }
        // b_entry: load(s); return  (foreign block reading the shared slot)
        let b_load = {
            let mut bld = Builder::from_block(BasicBlock::from_id_mut(ctx, b_entry));
            let load = bld.push_load::<false>(s, 8, stack).id();
            let ret = bld.context_mut().get_const(0u64, 8).id();
            bld.push_return(ret);
            load
        };

        // Without the boundary guard this panics decoding `s` in `b_entry`.
        let aliases = AliasResult::simple(&*ctx);
        mem2reg(ctx, a, &aliases);

        // The foreign load is untouched — the walk never entered function B.
        let ValueId::Instruction(b_load) = b_load else {
            unreachable!()
        };
        assert!(
            BasicBlock::from_id(ctx, b_entry)
                .instruction_ids()
                .contains(&b_load),
            "the rename walk must not cross into another function's block",
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
        // Per-function pass: scope the alias oracle to this function so the
        // stage is O(program) total, not O(functions × program).
        let aliases = AliasResult::simple_for_function(ctx, fun_id);
        // Resolve `@SP` so canonical `@SP ± N` slots are recognised; `None` when
        // the function has no incoming stack-pointer param (legacy literal path).
        let sp_reg = ctx.registers[&env.cfg.stack_pointer];
        let sp_param = incoming_sp_param(ctx, fun_id, sp_reg);
        Ok(mem2reg_framed(ctx, fun_id, &aliases, sp_param))
    }
}

crate::register_function_pass!(Mem2RegPass);
