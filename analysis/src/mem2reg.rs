use jstd::graph::analysis::compute_dominators;
use qcode::space::SpaceType;
use qcode::value::{
    BasicBlock, BlockId, BlockParamId, Function, FunctionId, Instruction, Value, ValueId, ValueRef,
    Varnode, VarnodeId,
    insn::{Branch, CBranch, InstructionId, Load, Mnemonic, Store},
};
use qcode::{
    builder::Builder,
    context::Context,
    value::{FunctionRef, block::BlockRef},
};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use crate::AliasResult;

/// Returns `true` if any variables were promoted.
pub fn mem2reg(ctx: &mut Context, function_id: FunctionId, aliases: &AliasResult) -> bool {
    Mem2Reg::new(ctx, function_id, aliases).run()
}

struct Mem2Reg<'ctx, 'str> {
    ctx: &'ctx mut Context<'str>,
    function_id: FunctionId,
    root_id: Option<BlockId>,
    aliases: &'ctx AliasResult,
}

impl<'ctx, 'str> Mem2Reg<'ctx, 'str> {
    fn new(
        ctx: &'ctx mut Context<'str>,
        function_id: FunctionId,
        aliases: &'ctx AliasResult,
    ) -> Self {
        let root_id = Function::from_id(&*ctx, function_id).root().map(|b| b.id);
        Self {
            ctx,
            function_id,
            root_id,
            aliases,
        }
    }

    fn run(&mut self) -> bool {
        let Promotable { vars, sizes } = self.collect_promotable_vars();
        if vars.is_empty() {
            return false;
        }

        let root_id = self.root_id();
        let dom = compute_dominators(&*self.ctx, root_id);
        let frontier = dom.dominator_frontier().clone();
        let InsertedBlockParams {
            by_block: var_params,
            changed,
        } = self.insert_block_params(&vars, &sizes, &frontier);
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
}

/// The `StackAddress` offset encoded by a stack-typed literal pointer, if `v`
/// is one. Stack slots are interned per `(offset, StackAddress)`, so equal
/// offsets share a single `ValueId` — making the literal a stable slot key.
fn stack_slot_addr(ctx: &Context, v: ValueId) -> Option<u64> {
    let ValueId::Literal(id) = v else {
        return None;
    };
    let lit = &ctx.values.literals[id];
    ctx.types.is_stack_address(lit.type_id).then_some(lit.value)
}

/// True if `v`'s result type is a `StackAddress` pointer, whether or not it is
/// a concrete literal. A stack-typed value that is *not* a slot literal is a
/// dynamic stack pointer (e.g. an indexed stack array), which we cannot
/// resolve to a fixed byte range and which therefore defeats slot promotion.
fn is_stack_typed(ctx: &Context, v: ValueId) -> bool {
    let type_id = match v {
        ValueId::Literal(id) => ctx.values.literals[id].type_id,
        ValueId::Instruction(id) => Instruction::from_id(ctx, id).type_id(),
        _ => return false,
    };
    ctx.types.is_stack_address(type_id)
}

/// The `(offset, ptr_width)` a stack-slot literal encodes, relative to the
/// per-function stack base. `offset` is negative for locals below the entry SP
/// and `>= ptr_width` for the caller's frame (the return-address slot sits at
/// offset `0`, incoming parameters above it). Mirrors `compute_stack_delta`'s
/// `value - stack_base` decode, deriving the pointer width from the literal's
/// own `StackAddress` type so it needs no separate stack-pointer argument.
pub(crate) fn stack_slot_offset(ctx: &Context, v: ValueId) -> Option<(i64, usize)> {
    let ValueId::Literal(id) = v else {
        return None;
    };
    let lit = &ctx.values.literals[id];
    if !ctx.types.is_stack_address(lit.type_id) {
        return None;
    }
    let ptr_width = ctx.types.size_of(lit.type_id);
    let offset = lit.value.wrapping_sub(qcode::types::stack_base(ptr_width)) as i64;
    Some((offset, ptr_width))
}

/// Whether `function_id` performs any load or store through a *computed* pointer
/// — one that does not resolve to a fixed location (a register varnode, a stack
/// slot, or a constant/global address literal). Such an access reads or writes a
/// pointer the function cannot bound to a fixed byte range, so if that pointer is
/// a parameter the caller handed in (e.g. a frame pointer), the callee may touch
/// arbitrary bytes behind it. This is the broad "unresolved dynamic memory
/// access" signal used to flag a function as an unbounded reader.
pub(crate) fn has_dynamic_pointer_deref(ctx: &Context, function_id: FunctionId) -> bool {
    for block in Function::from_id(ctx, function_id).blocks() {
        for insn in block.iter() {
            let Some(access) = MemoryAccess::from_mnemonic(insn.mnemonic()) else {
                continue;
            };
            // A pointer is fixed only when it is a named location (varnode) or a
            // constant address (literal, including a stack-slot literal). An
            // instruction/block-param pointer is a computed — hence unbounded —
            // address.
            if !matches!(access.ptr, ValueId::Varnode(_) | ValueId::Literal(_)) {
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
        return HashMap::new();
    }

    let mut store_ptrs = HashSet::new();
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

#[derive(Clone, Copy, PartialEq, Eq)]
struct StackAccessRange {
    start: u64,
    end: u64,
}

impl StackAccessRange {
    fn new(start: u64, size: usize) -> Self {
        Self {
            start,
            end: start.saturating_add(size as u64),
        }
    }

    fn overlaps_slot(&self, slot_start: u64, slot_size: usize) -> bool {
        let slot_start = slot_start as u128;
        let slot_end = slot_start + slot_size as u128;
        slot_start < self.end as u128 && (self.start as u128) < slot_end
    }
}

type BlockParamsByVar = HashMap<ValueId, BlockParamId>;
type BlockParamAssignments = HashMap<BlockId, BlockParamsByVar>;

struct InsertedBlockParams {
    by_block: BlockParamAssignments,
    changed: bool,
}

impl Mem2Reg<'_, '_> {
    fn block_param_name_for_var(&self, var: ValueId) -> Option<String> {
        match var {
            ValueId::Varnode(varnode_id) => Varnode::from_id(self.ctx, varnode_id)
                .name()
                .map(|n| n.to_owned()),
            _ => stack_slot_addr(self.ctx, var).map(|addr| format!("stack_{addr:x}")),
        }
    }

    /// An existing param on `block_id` previously created to promote `var`,
    /// identified by its recorded [`origin`](qcode::value::BlockParam::origin)
    /// rather than its display name. `origin` is a stable cross-run identity, so
    /// this finds the param even for varnodes that have no name (the name-based
    /// match used to miss them, re-pushing a duplicate on every re-run).
    fn existing_param_for_var(
        &self,
        block_id: BlockId,
        size: usize,
        var: ValueId,
    ) -> Option<BlockParamId> {
        BasicBlock::from_id(self.ctx, block_id)
            .params()
            .find(|param| param.size() == size && param.origin() == Some(var))
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
        if let Some(name) = name {
            self.ctx.values.block_params[param_id].name = Some(Cow::Owned(name.to_owned()));
        }
        (param_id, true)
    }

    fn collect_promotable_vars(&self) -> Promotable {
        let mut stored = HashSet::new();
        let mut loaded = HashSet::new();
        let mut store_counts: HashMap<ValueId, usize> = HashMap::new();

        // Stack-slot bookkeeping, keyed by the slot literal `ValueId`.
        let mut stack_stored: HashSet<ValueId> = HashSet::new();
        let mut stack_loaded: HashSet<ValueId> = HashSet::new();
        let mut stack_size: HashMap<ValueId, usize> = HashMap::new();
        let mut stack_size_conflict: HashSet<ValueId> = HashSet::new();
        let mut stack_intervals: Vec<StackAccessRange> = Vec::new();
        // A stack-typed pointer we could not resolve to a fixed slot disables all
        // stack promotion: it may alias any slot. The same applies when this
        // function hands a pointer into its own frame to a callee that may read it
        // unboundedly (`frame_escapes_to_unbounded`, a fact seeded by the driver
        // from the previous checkpoint+replay round): that callee may have written
        // any slot, so none may be promoted across it.
        let mut dynamic_stack =
            Function::from_id(self.ctx, self.function_id).frame_escapes_to_unbounded();
        // Locations with a store whose source is narrower than the access (e.g. the
        // `MOV ESI, imm32` lift, an 8-byte RSI store of a 4-byte literal). Promoting
        // them would forward the narrow value into a wider load.
        let mut mixed_width: HashSet<ValueId> = HashSet::new();

        for block in Function::from_id(self.ctx, self.function_id).blocks() {
            for insn in block.iter() {
                let Some(access) = MemoryAccess::from_mnemonic(insn.mnemonic()) else {
                    continue;
                };
                if let MemoryAccessKind::Store { src } = access.kind
                    && ValueRef::new(src, self.ctx).size() != access.size
                {
                    mixed_width.insert(access.ptr);
                }

                if access.ptr.is_varnode() {
                    if access.is_store() {
                        stored.insert(access.ptr);
                        *store_counts.entry(access.ptr).or_insert(0) += 1;
                    } else {
                        loaded.insert(access.ptr);
                    }
                } else if let Some(addr) = stack_slot_addr(self.ctx, access.ptr) {
                    stack_intervals.push(StackAccessRange::new(addr, access.size));
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
                } else if is_stack_typed(self.ctx, access.ptr) {
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
                    if self
                        .root_id
                        .is_some_and(|r| self.live_in_blocks(var).contains(&r))
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
        let mut sizes: HashMap<ValueId, usize> = HashMap::new();
        if !dynamic_stack {
            for &var in stack_stored.intersection(&stack_loaded) {
                if stack_size_conflict.contains(&var) {
                    continue;
                }
                let size = stack_size[&var];
                let addr =
                    stack_slot_addr(self.ctx, var).expect("stack slot var is a stack literal");
                let own = StackAccessRange::new(addr, size);
                let overlapped = stack_intervals
                    .iter()
                    .any(|access| *access != own && access.overlaps_slot(addr, size));
                if overlapped {
                    continue;
                }
                vars.insert(var);
                sizes.insert(var, size);
            }

            // Incoming stack parameters: a caller-frame slot (offset >= ptr_width,
            // above the return-address slot at offset 0) that is *loaded but never
            // stored* is read before any definition — a function input passed on
            // the stack (cdecl, or an x86-64 stack-overflow argument). Promote it
            // to a root block param, mirroring the load-only register-input path,
            // so `compute_inputs` can recover it. The same single-size and
            // no-overlap guards apply.
            for &var in stack_loaded.difference(&stack_stored) {
                if stack_size_conflict.contains(&var) {
                    continue;
                }
                let Some((offset, ptr_width)) = stack_slot_offset(self.ctx, var) else {
                    continue;
                };
                if offset < ptr_width as i64 {
                    continue;
                }
                let size = stack_size[&var];
                let addr =
                    stack_slot_addr(self.ctx, var).expect("stack slot var is a stack literal");
                let own = StackAccessRange::new(addr, size);
                let overlapped = stack_intervals
                    .iter()
                    .any(|access| *access != own && access.overlaps_slot(addr, size));
                if overlapped {
                    continue;
                }
                vars.insert(var);
                sizes.insert(var, size);
            }
        }

        // Drop any location fed by a narrower-than-access store: forwarding its value
        // into a wider load would mis-size the result.
        vars.retain(|var| !mixed_width.contains(var));
        sizes.retain(|var, _| vars.contains(var));

        Promotable { vars, sizes }
    }

    fn insert_block_params(
        &mut self,
        vars: &HashSet<ValueId>,
        sizes: &HashMap<ValueId, usize>,
        frontier: &HashMap<BlockId, HashSet<BlockId>>,
    ) -> InsertedBlockParams {
        let mut var_params: BlockParamAssignments = HashMap::new();
        let mut changed = false;

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

            let live_in = self.live_in_blocks(var);
            let phi_positions = {
                let function = Function::from_id(self.ctx, self.function_id);
                find_phi_insert_positions(var, &function, frontier, &live_in)
            };

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
            if !root_has_param && live_in.contains(&root_id) {
                let (param_id, inserted) =
                    self.get_or_insert_param_for_var(root_id, size, var, var_name.as_deref());
                changed |= inserted;
                var_params.entry(root_id).or_default().insert(var, param_id);
            }
        }

        InsertedBlockParams {
            by_block: var_params,
            changed,
        }
    }

    fn live_in_blocks(&self, var: ValueId) -> HashSet<BlockId> {
        live_in_blocks(self.ctx, self.function_id, var, self.aliases)
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
    aliases: &AliasResult,
) -> HashSet<BlockId> {
    let mut upward_exposed = HashSet::new();
    let mut defined = HashSet::new();

    for block in Function::from_id(ctx, function_id).blocks() {
        let block_id = block.id;
        let mut has_def = false;

        for insn in block.iter() {
            match insn.mnemonic() {
                Mnemonic::Store(Store { ptr, .. }) if *ptr == var => {
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
                // restores, return values). A store is live-out when it is neither
                // consumed by any local load/branch-arg nor overwritten before being
                // read. Both consumed and dead (overwritten) stores are safe to remove.
                if let ValueId::Varnode(vn_id) = ptr
                    && matches!(
                        Varnode::from_id(self.ctx, *vn_id).space().ty,
                        SpaceType::Register
                    )
                {
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

fn block_contains_store_to_var(block: &BlockRef, var: ValueId) -> bool {
    block.iter().any(|insn| {
        if let Mnemonic::Store(Store { ptr, .. }) = insn.mnemonic() {
            *ptr == var
        } else {
            false
        }
    })
}

fn find_phi_insert_positions(
    var: ValueId,
    function: &FunctionRef,
    frontier: &HashMap<BlockId, HashSet<BlockId>>,
    live_in: &HashSet<BlockId>,
) -> HashSet<BlockId> {
    let block_containing_store = function
        .blocks()
        .filter(|b| block_contains_store_to_var(b, var))
        .map(|b| b.id)
        .collect::<Vec<_>>();

    let mut worklist = block_containing_store.clone();
    let mut result = HashSet::new();

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
            visited: HashSet::new(),
            frames: vec![Frame::new()],
            consumed_stores: HashSet::new(),
            dead_stores: HashSet::new(),
            preserved_stores: HashSet::new(),
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
                        if let Some(FrameEntry::Defined(ReachingValue {
                            store_insn: Some(old_id),
                            ..
                        })) = decide_variable_value(ptr, &state.frames)
                            && !state.consumed_stores.contains(&old_id)
                        {
                            state.dead_stores.insert(old_id);
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

                Mnemonic::Load(Load { ptr, .. }) if state.vars.contains(&ptr) => {
                    let reaching = decide_variable_value(ptr, &state.frames);
                    let (load_value, store_insn) = match reaching {
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

                    state.frames.push(Frame::new());
                    self.decide_values_start_from(success_block, state);
                    state.frames.pop();

                    state.frames.push(Frame::new());
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

                Mnemonic::Call(_) | Mnemonic::CallInd(_) => {
                    // A call clobbers its callee's registers. Shadow each promoted
                    // register var it clobbers with a clobber marker so a read in the
                    // continuation sees the call's output (left as a register load),
                    // not the value the caller held before the call.
                    let clobbered: Vec<ValueId> = state
                        .vars
                        .iter()
                        .copied()
                        .filter(|&v| call_clobbers_var(self.ctx, block, v, self.aliases))
                        .collect();
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
        for successor in successors {
            self.decide_values_start_from(successor, state);
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
        let live_in = live_in_blocks(&ctx, test, A.into(), &aliases);

        let result = find_phi_insert_positions(A.into(), &function, frontier, &live_in);

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
        let live_in = HashSet::from([loop_header]);
        let frontier = HashMap::from([
            (entry, HashSet::from([loop_header])),
            (loop_header, HashSet::from([loop_header])),
        ]);

        let result = find_phi_insert_positions(A.into(), &function, &frontier, &live_in);

        assert_eq!(result, HashSet::from([loop_header]));
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
        let live_in = live_in_blocks(&ctx, test, A.into(), &aliases);

        let result = find_phi_insert_positions(A.into(), &function, frontier, &live_in);

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
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        let aliases = AliasResult::simple(ctx);
        Ok(mem2reg(ctx, fun_id, &aliases))
    }
}

crate::register_function_pass!(Mem2RegPass);
