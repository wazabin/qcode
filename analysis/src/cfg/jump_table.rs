//! Resolves jump tables — the indirect `switch` dispatch a compiler emits as
//! `goto [table_base + index*scale]`.
//!
//! For each block terminated by a [`BranchInd`], the pass recognizes the
//! address computation feeding the indirect jump, bounds the table index with
//! the value-range analysis ([`crate::value_range`]), reads each table entry
//! straight out of the binary's initialized memory
//! ([`Context::read_uint`](qcode::context::Context::read_uint)), and connects
//! the block to every resolved target with a real CFG edge.
//!
//! Two table encodings are handled:
//!
//! - **Absolute** — each slot holds the target address directly; the loaded
//!   value *is* the branch pointer: `goto [Load(base + index*ptr_width)]`.
//! - **Relative** — each slot holds a (usually signed 32-bit) offset that is
//!   added back to a constant base before the branch, as emitted for
//!   position-independent code: `goto base + sext(Load(base + index*4))`.
//!
//! Because a resolved target is only valid while the table bytes stay put, every
//! entry read records a [`Proposition::ImmutableMemory`] assumption on the
//! context: if a later pass ever proves that memory writable, the assumption is
//! violated and the resolution is discarded by the replay driver.
//!
//! TODO: when the index cannot be bounded, scan the table for a run of
//! addresses "close to" one another to recover the size heuristically.

use qcode::{
    address_index::AddressIndex,
    assumption::Proposition,
    builder::Builder,
    context::Context,
    value::{
        BasicBlock, BlockMutRef, FunctionBody, FunctionId, Value, ValueId, ValueRef,
        block::{BlockId, EdgeId},
        insn::{Binary, Binop, BranchInd, IntBinop, Load, Mnemonic},
        util::base_ref::{WithCtx, WithCtxMut},
    },
};

use crate::{Pass, PipelineEnv, value_range::value_range};

/// Largest table the pass will materialize. Guards against a mis-bounded index
/// turning into millions of bogus edges.
const MAX_TABLE_ENTRIES: u64 = 4096;

pub struct HandleJumpTables;

impl Default for HandleJumpTables {
    fn default() -> Self {
        Self
    }
}

/// One resolved jump-table edge to apply after the read-only scan.
struct Edit {
    /// The block ending in the indirect branch.
    from: BlockId,
    /// Resolved target address.
    target: u64,
    /// The value "switched on"
    index: ValueId,
    /// The index value for this target
    value: u64,
}

struct MakeBranch {
    /// The block ending in the indirect branch.
    from: BlockId,
    /// Resolved target address.
    target: u64,
}

struct MakeCBranch {
    /// The block ending in the indirect branch.
    from: BlockId,

    index: ValueId,
    offset: u64,

    true_target: u64,
    false_target: u64,
}

impl Pass for HandleJumpTables {
    const NAME: &'static str = "handle_jump_tables";

    fn description(&self) -> &'static str {
        "Resolves jump tables, connecting indirect branches to their targets"
    }

    /// Jump-table resolution is interprocedural — it queues cross-function code
    /// discoveries ([`Context::discover_code`]), installs global
    /// [`Proposition::ImmutableMemory`] assumptions, and re-homes resolved target
    /// blocks into their owning functions — so it is a whole-program module pass,
    /// not a per-function one. Drive the per-function body over every non-external,
    /// non-ignored function, matching a `scope = "function"` stage's eligibility.
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        let fun_ids: Vec<FunctionId> = ctx
            .functions()
            .filter(|f| !f.is_external())
            .filter(|f| !ctx.is_function_ignored(f.address()))
            .map(|f| f.id)
            .collect();
        let mut addresses = AddressIndex::analyze(ctx);
        let mut changed = false;
        for fun_id in fun_ids {
            changed |= Self::resolve_function_indexed(ctx, &mut addresses, fun_id)?;
        }
        Ok(changed)
    }
}

impl HandleJumpTables {
    /// Resolve every jump table in the single function `fun_id`, connecting each
    /// indirect branch to its recovered CFG successors. Returns whether the IR
    /// changed. This is the body that ran once per function while the pass was a
    /// `FunctionPass`.
    #[cfg(test)]
    fn resolve_function(ctx: &mut Context, fun_id: FunctionId) -> Result<bool, String> {
        let mut addresses = AddressIndex::analyze(ctx);
        Self::resolve_function_indexed(ctx, &mut addresses, fun_id)
    }

    /// Indexed implementation shared by the whole-module pass and focused
    /// single-function tests. The caller owns one construction index for the
    /// complete topology-mutation operation.
    fn resolve_function_indexed(
        ctx: &mut Context,
        addresses: &mut AddressIndex,
        fun_id: FunctionId,
    ) -> Result<bool, String> {
        let function = FunctionBody::from_id(ctx, fun_id);

        // Nothing resolves in a function with no indirect branch. The vast majority
        // of functions have none, so bail before allocating the block-id vector and
        // walking every block through `resolve_block`.
        if !function.blocks().any(|b| {
            matches!(
                b.instructions().last().map(|i| i.mnemonic()),
                Some(Mnemonic::BranchInd(_))
            )
        }) {
            return Ok(false);
        }

        // Entry address of the function owning these branches, paired with every
        // discovered target so the re-lift loop knows which function to grow.
        let fn_entry = function.address();

        // Read-only scan: collect every resolvable edge, then mutate.
        let mut edits: Vec<Edit> = Vec::new();
        let mut single_branches: Vec<MakeBranch> = Vec::new();
        let mut branches: Vec<MakeCBranch> = Vec::new();

        let block_ids = function.blocks().map(|b| b.id).collect::<Vec<_>>();

        for id in block_ids {
            let block = BasicBlock::from_id_mut(ctx, id);

            if let Some(mut block_edits) = resolve_block(block) {
                match block_edits.len() {
                    1 => {
                        let e = block_edits.pop().unwrap();
                        single_branches.push(MakeBranch {
                            from: e.from,
                            target: e.target,
                        });
                    }
                    2 => {
                        let e1 = block_edits.pop().unwrap();
                        let e2 = block_edits.pop().unwrap();

                        // The false target is the case the index is compared
                        // *equal* to; `offset` carries that index value. A zero
                        // case is the common shape but not required.
                        let ((true_target, false_target), offset) = if e1.value == 0 {
                            ((e2.target, e1.target), e1.value)
                        } else {
                            ((e1.target, e2.target), e2.value)
                        };

                        branches.push(MakeCBranch {
                            from: e1.from,
                            index: e1.index,
                            offset,
                            true_target,
                            false_target,
                        });
                    }
                    _ => edits.append(&mut block_edits),
                }
            }
        }

        if edits.is_empty() && single_branches.is_empty() && branches.is_empty() {
            return Ok(false);
        }

        // The single- and two-target paths below rewrite the terminator itself,
        // so they always change the IR (and cannot re-trigger: the block no
        // longer ends in `BranchInd`).
        let mut changed = !single_branches.is_empty() || !branches.is_empty();

        // A table with more than two cases stays an indirect `switch`: we keep
        // the `BranchInd` but connect a real edge to every case body. Because the
        // terminator survives, the block is re-resolved on every fixpoint
        // iteration — so rewrite (and report a change) only when the resolved
        // target multiset differs from the edges already on the block. Tearing
        // down and rebuilding identical edges would keep every `repeat_until`
        // stage containing this pass spinning to its iteration cap.
        let mut by_from: rustc_hash::FxHashMap<BlockId, Vec<Edit>> =
            rustc_hash::FxHashMap::default();
        for e in edits {
            by_from.entry(e.from).or_default().push(e);
        }
        for (from, edits) in by_from {
            let mut existing: Vec<Option<u64>> = BasicBlock::from_id(ctx, from)
                .successors()
                .map(|(_, succ)| BasicBlock::from_id(ctx, succ).address())
                .collect();
            let mut desired: Vec<Option<u64>> = edits.iter().map(|e| Some(e.target)).collect();
            existing.sort_unstable();
            desired.sort_unstable();
            if existing == desired {
                continue;
            }
            changed = true;

            // Clear the block's existing edges first so a re-resolution does not
            // double them.
            clear_successors(ctx, from);
            let from_addr = BasicBlock::from_id(ctx, from).address();
            for Edit { target, .. } in edits {
                // A cross-function switch target carries no intra-function edge
                // (strict IR locality); a mid-function landing is split off inside
                // `resolve_local_target`. The `BranchInd` keeps its indirect form and
                // the resolution is recorded for disassembly via `discover`.
                if let LocalTarget::Local(tb) = resolve_local_target(ctx, addresses, target, fun_id)
                {
                    ctx.add_cfg_edge(from, tb);
                }
                discover(ctx, fn_entry, from_addr, target);
            }
        }

        // A single resolved target: the indirect branch is really an
        // unconditional jump. Replace `BranchInd` with a direct `Branch`.
        for MakeBranch { from, target } in single_branches {
            let from_addr = BasicBlock::from_id(ctx, from).address();
            let resolved = resolve_local_target(ctx, addresses, target, fun_id);
            clear_successors(ctx, from);
            let mut block = BasicBlock::from_id_mut(ctx, from);
            block.pop_insn();
            {
                let mut builder = Builder::from_block(block);
                match resolved {
                    LocalTarget::Local(target_block) => {
                        builder.push_branch(target_block);
                    }
                    LocalTarget::Foreign(g) => {
                        builder.push_tail_call(g);
                    }
                }
            }
            discover(ctx, fn_entry, from_addr, target);
        }

        for MakeCBranch {
            from,
            index,
            offset,
            true_target,
            false_target,
        } in branches
        {
            let from_addr = BasicBlock::from_id(ctx, from).address();
            // Each arm is materialized as an intra-function block: the resolved
            // target, or a trampoline tail-calling a foreign function (strict IR
            // locality — a `CBranch` arm is never a foreign block reference).
            let true_target_resolved = resolve_local_target(ctx, addresses, true_target, fun_id);
            let false_target_resolved = resolve_local_target(ctx, addresses, false_target, fun_id);
            let true_block = arm_block(ctx, fun_id, true_target_resolved);
            let false_block = arm_block(ctx, fun_id, false_target_resolved);
            discover(ctx, fn_entry, from_addr, true_target);
            discover(ctx, fn_entry, from_addr, false_target);

            clear_successors(ctx, from);
            let mut block = BasicBlock::from_id_mut(ctx, from);
            block.pop_insn();

            let mut builder = Builder::from_block(block);
            let size = ValueRef::from_id(builder.context(), index).size();
            let false_value = builder.context_mut().get_const(offset, size).id();
            // `index != false_value` selects the true target, else the false one.
            let cond = builder.push_ne(index, false_value).id();
            builder.push_cbranch(cond, true_block, false_block);
        }

        if changed {
            qcode::stat!("jump_table_edges", 1);
        }
        Ok(changed)
    }
}

/// The strict-local resolution of a jump-table target address (context-split
/// ruling 2): either an intra-function block to wire an edge/branch to, or a
/// foreign function the transfer tail-calls into. A target landing mid-another-
/// function is split into its own function first ([`Context::split_function_at`]);
/// a target that is (or becomes) another function's entry is foreign. No
/// cross-function CFG edge is ever created.
enum LocalTarget {
    Local(BlockId),
    Foreign(FunctionId),
}

fn resolve_local_target(
    ctx: &mut Context,
    addresses: &mut AddressIndex,
    addr: u64,
    fun_id: FunctionId,
) -> LocalTarget {
    let tb = ctx.get_or_make_block_indexed(addresses, addr, fun_id);
    match BasicBlock::from_id(ctx, tb).parent().map(|f| f.id) {
        Some(owner) if owner != fun_id => {
            let is_entry = addresses.function_at(addr) == Some(owner)
                && FunctionBody::from_id(ctx, owner).root().map(|root| root.id) == Some(tb);
            let g = if is_entry {
                owner
            } else {
                ctx.split_function_at_indexed(addresses, tb)
            };
            LocalTarget::Foreign(g)
        }
        _ => {
            FunctionBody::from_id_mut(ctx, fun_id).add_block(tb);
            LocalTarget::Local(tb)
        }
    }
}

/// Materialize a jump-table arm as an intra-function block: the resolved target
/// block itself, or a fresh trampoline block that tail-calls the foreign function
/// (a `CBranch` arm must be intra-function; the tail call carries control across).
fn arm_block(ctx: &mut Context, fun_id: FunctionId, target: LocalTarget) -> BlockId {
    match target {
        LocalTarget::Local(b) => b,
        LocalTarget::Foreign(g) => {
            let tramp = BasicBlock::make(ctx, fun_id).id;
            Builder::from_block(BasicBlock::from_id_mut(ctx, tramp)).push_tail_call(g);
            tramp
        }
    }
}

/// Remove every outgoing CFG edge of `from`. The jump-table edges the lifter
/// adds to the clean IR persist on the block alongside its `BranchInd`; clearing
/// them before the apply step rebuilds the terminator keeps a re-resolution from
/// doubling edges.
fn clear_successors(ctx: &mut Context, from: BlockId) {
    let edges: Vec<EdgeId> = BasicBlock::from_id(ctx, from)
        .successors()
        .map(|(edge, _)| edge)
        .collect();
    for edge in edges {
        ctx.remove_cfg_edge(from.func, edge);
    }
}

/// Record a resolved `target` for later disassembly, keyed by the owning
/// function's entry address. A no-op for synthetic functions that lack an
/// address (nothing to re-lift from a binary image).
fn discover(ctx: &mut Context, fn_entry: Option<u64>, source_block: Option<u64>, target: u64) {
    if let (Some(entry), Some(source)) = (fn_entry, source_block) {
        ctx.discover_code(entry, source, target);
    }
}

/// If `block_id` ends in an indirect branch whose table the pass can resolve,
/// push one [`Edit`] per case target onto `edits`.
fn resolve_block(mut block: BlockMutRef) -> Option<Vec<Edit>> {
    // A block still terminated by `BranchInd` is re-resolved every round, even
    // once the lifter has connected its targets in the clean IR: those edges let
    // function-splitting follow the switch, but the terminator itself is only
    // rewritten in this (disposable or final) optimized clone. The apply step
    // skips a block whose edges already match the resolution (reporting no
    // change), and otherwise clears them before rebuilding, so re-resolving an
    // already-connected block is a no-op rather than edge-doubling.
    let insn = block.instructions().last()?;

    let Mnemonic::BranchInd(BranchInd { ptr }) = insn.mnemonic() else {
        return None;
    };
    let ptr = ptr.qualify(insn.id.func);

    log::trace!(
        target: "jump_table",
        "considering block {:x} with indirect branch",
        block.address().unwrap_or_default(),
    );

    // Indirect jump through a single fixed pointer slot: `goto [load(const)]`.
    // Not a table (there is no index), but the slot is immutable data, so it
    // resolves to one concrete target.
    if let Some(edits) = resolve_constant_load(&mut block, ptr) {
        return Some(edits);
    }

    let ctx = block.ctx();
    let table = recognize_table(ctx, ptr)?;

    log::debug!(
        target: "jump_table",
        "recognized jump table at {:x} with base {:x} and scale {}",
        table.base, table.base, table.scale
    );

    let index_size = value_size(ctx, table.index);

    let range = value_range(ctx, table.index, block.id);

    log::debug!(
        target: "jump_table",
        "resolving jump table in block {:x}: base {:x}, size {}, offset {}",
        block.address().unwrap_or_default(),
        table.base,
        range.count(),
        range.min,
    );

    if !range.is_bounded(index_size)
        || range.fills_containing_width()
        || range.count() > MAX_TABLE_ENTRIES
    {
        log::debug!(target: "jump_table", "skipping unbounded or huge table: range {range:?}");
        return None;
    }
    log::trace!(
        target: "jump_table",
        "bounded index {} with range {range:?}",
        ValueRef::from_id(ctx, table.index),
    );

    // Materialize one edge per index value. Bail on the whole table if any slot
    // is unmapped or points outside executable memory — a partial resolution
    // would leave a misleading CFG.
    let mut resolved: Vec<Edit> = Vec::new();
    for index in range.min..=range.max {
        let entry_addr = table.base.wrapping_add(index.wrapping_mul(table.scale));

        // A table in writable memory is not trustworthy data (see
        // `resolve_constant_load`); bail on the whole table rather than resolve
        // against bytes the runtime may rewrite.
        if block.ctx().is_known_writable_addr(entry_addr) {
            log::debug!(target: "jump_table", "skipping table: entry {entry_addr:x} is writable");
            return None;
        }

        if !block.ctx_mut().assume_true(Proposition::ImmutableMemory {
            addr: entry_addr,
            size: table.slot_width as u8,
        }) {
            log::debug!(target: "jump_table", "skipping table: entry {entry_addr:x} not immutable");
            return None;
        }

        let ctx = block.ctx();

        let Some(raw) = ctx.read_uint(entry_addr, table.slot_width) else {
            log::debug!(target: "jump_table", "skipping table: slot {entry_addr:x} unmapped");
            return None;
        };

        let target = match table.relative_base {
            Some(base) => base.wrapping_add(sign_extend(raw, table.slot_width)),
            None => raw,
        };

        if !ctx.is_executable_addr(target) {
            log::debug!(target: "jump_table", "skipping table: target {target:x} not executable");
            return None;
        }

        log::debug!(
            target: "jump_table",
            "  slot[{index}] at {entry_addr:x} -> target {target:x}",
        );

        resolved.push(Edit {
            from: block.id,
            target,
            index: table.index,
            value: index,
        });
    }

    Some(resolved)
}

/// A `goto [load(const_addr)]`: the branch pointer is loaded from one fixed
/// data address (e.g. an indirect tail-jump through a relocated function
/// pointer). Reading that immutable slot yields the single concrete target.
///
/// Returns a one-element edit list on success so the caller's single-target
/// path rewrites the `BranchInd` into a direct `Branch`.
fn resolve_constant_load(block: &mut BlockMutRef, ptr: ValueId) -> Option<Vec<Edit>> {
    let ctx = block.ctx();
    let (load, load_func) = as_load(ctx, ptr)?;
    let addr = numeric_const(ctx, load.ptr.qualify(load_func))?;

    // A load from writable memory is not a reliable constant: the canonical case
    // is `jmp *[GOT]` in a PLT stub, whose slot the dynamic linker rewrites at
    // load time. The file image holds the pre-relocation value (the lazy resolver
    // stub), so resolving to it would fabricate a bogus direct branch and, worse,
    // make the stub look like a pure, side-effect-free function. Leave the
    // `BranchInd` in place so the stub stays an opaque external transfer.
    if ctx.is_known_writable_addr(addr) {
        log::debug!(target: "jump_table", "skipping constant load: slot {addr:x} is writable");
        return None;
    }

    if !block.ctx_mut().assume_true(Proposition::ImmutableMemory {
        addr,
        size: load.size as u8,
    }) {
        log::debug!(target: "jump_table", "skipping constant load: slot {addr:x} not immutable");
        return None;
    }

    let ctx = block.ctx();
    let Some(target) = ctx.read_uint(addr, load.size) else {
        log::debug!(target: "jump_table", "skipping constant load: slot {addr:x} unmapped");
        return None;
    };

    if !ctx.is_executable_addr(target) {
        log::debug!(target: "jump_table", "skipping constant load: target {target:x} not executable");
        return None;
    }

    log::debug!(target: "jump_table", "resolved indirect jump via {addr:x} to {target:x}");

    Some(vec![Edit {
        from: block.id,
        target,
        index: ptr,
        value: 0,
    }])
}

/// The shape of a recognized jump table.
struct Table {
    /// Address of the table's first slot.
    base: u64,
    /// Byte stride between slots (the entry width).
    scale: u64,
    /// Width of one slot in bytes (`ptr_width` for absolute, often 4 for relative).
    slot_width: usize,
    /// For a relative table, the constant the loaded offset is added back to.
    /// `None` for an absolute table (the slot holds the target directly).
    relative_base: Option<u64>,
    /// The switch index SSA value to bound.
    index: ValueId,
}

fn recognize_absolute_table(ctx: &Context, load: Load, load_func: FunctionId) -> Option<Table> {
    let (base, scale, index) = decompose_address(ctx, load.ptr.qualify(load_func))?;

    Some(Table {
        base,
        scale,
        slot_width: load.size,
        relative_base: None,
        index,
    })
}

/// Recognize the address computation feeding a `BranchInd` as a jump table.
///
/// Absolute: `ptr = Load(base + index*scale)`.
/// Relative: `ptr = Add(rel_base, [s|z]ext(Load(base + index*scale)))`.
fn recognize_table(ctx: &Context, ptr: ValueId) -> Option<Table> {
    // Absolute: the branch pointer is the loaded value itself.
    if let Some((load, load_func)) = as_load(ctx, ptr) {
        return recognize_absolute_table(ctx, load, load_func);
    }

    // Relative: `rel_base + ext(load)`. One operand is the constant base, the
    // other resolves (through an optional sext/zext) to a table load.
    let ptr_func = ptr.owning_function()?;
    if let Mnemonic::Binop(Binary {
        op: Binop::Int(IntBinop::Add),
        lhs,
        rhs,
    }) = def_mnemonic(ctx, ptr)?
    {
        let (lhs, rhs) = (lhs.qualify(ptr_func), rhs.qualify(ptr_func));
        let (rel_base, offset) = match (numeric_const(ctx, lhs), numeric_const(ctx, rhs)) {
            (Some(c), _) => (c, rhs),
            (_, Some(c)) => (c, lhs),
            _ => return None,
        };
        let (load, load_func) = as_load(ctx, strip_ext(ctx, offset))?;
        let (base, scale, index) = decompose_address(ctx, load.ptr.qualify(load_func))?;
        return Some(Table {
            base,
            scale,
            slot_width: load.size,
            relative_base: Some(rel_base),
            index,
        });
    }

    None
}

/// Decompose a table-element address `base + index*scale` into its parts.
/// Accepts `base + index` (scale 1), `base + index*c`, and `base + index<<c`.
fn decompose_address(ctx: &Context, addr: ValueId) -> Option<(u64, u64, ValueId)> {
    let func = addr.owning_function()?;
    let Mnemonic::Binop(Binary {
        op: Binop::Int(IntBinop::Add),
        lhs,
        rhs,
    }) = *def_mnemonic(ctx, addr)?
    else {
        log::trace!(target: "jump_table", "not a table: no add");
        return None;
    };
    let (lhs, rhs) = (lhs.qualify(func), rhs.qualify(func));

    // The base is the constant operand; the other is the (scaled) index.
    let (base, idx_expr) = match (numeric_const(ctx, lhs), numeric_const(ctx, rhs)) {
        (Some(c), _) => (c, rhs),
        (_, Some(c)) => (c, lhs),
        _ => {
            log::trace!(target: "jump_table", "not a table: no constant base");
            return None;
        }
    };

    let (scale, index) = decompose_scale(ctx, idx_expr);

    Some((base, scale, index))
}

/// Pull a constant scale out of `index*c` or `index<<c`; otherwise scale 1.
fn decompose_scale(ctx: &Context, v: ValueId) -> (u64, ValueId) {
    if let Some(func) = v.owning_function()
        && let Some(Mnemonic::Binop(Binary {
            op: Binop::Int(op),
            lhs,
            rhs,
        })) = def_mnemonic(ctx, v)
    {
        let (lhs, rhs) = (lhs.qualify(func), rhs.qualify(func));
        match op {
            IntBinop::Mul => match (numeric_const(ctx, lhs), numeric_const(ctx, rhs)) {
                (Some(c), _) => return (c, rhs),
                (_, Some(c)) => return (c, lhs),
                _ => {}
            },
            IntBinop::ShiftLeft => {
                if let Some(sh) = numeric_const(ctx, rhs)
                    && sh < 64
                {
                    return (1u64 << sh, lhs);
                }
            }
            _ => {}
        }
    }
    (1, v)
}

/// The `Load` defining `v` (paired with its owning function, which qualifies
/// the load's bare-local operands), if `v` is the result of a RAM load.
fn as_load(ctx: &Context, v: ValueId) -> Option<(Load, FunctionId)> {
    let ValueId::Instruction(id) = v else {
        return None;
    };
    match ctx.get_insn(id).mnemonic() {
        Mnemonic::Load(load) => Some((load.clone(), id.func)),
        _ => None,
    }
}

/// Peel a single sign/zero-extension off `v`, returning its source.
fn strip_ext(ctx: &Context, v: ValueId) -> ValueId {
    let func = match v {
        ValueId::Instruction(id) => id.func,
        _ => return v,
    };
    match def_mnemonic(ctx, v) {
        Some(Mnemonic::Sext(s)) => s.src.qualify(func),
        Some(Mnemonic::Zext(z)) => z.src.qualify(func),
        _ => v,
    }
}

/// The defining mnemonic of `v` when it is an instruction result.
fn def_mnemonic<'c>(ctx: &'c Context<'_>, v: ValueId) -> Option<&'c Mnemonic> {
    match v {
        ValueId::Instruction(id) => Some(ctx.get_insn(id).mnemonic()),
        _ => None,
    }
}

/// Concrete value of `v` when it is a literal.
fn numeric_const(ctx: &Context, v: ValueId) -> Option<u64> {
    match ValueRef::new(v, ctx) {
        ValueRef::Literal(c) => Some(c.value()),
        _ => None,
    }
}

fn value_size(ctx: &Context, v: ValueId) -> usize {
    ValueRef::new(v, ctx).size()
}

/// Sign-extend the low `size` bytes of `raw` to a full `u64`.
fn sign_extend(raw: u64, size: usize) -> u64 {
    if size == 0 || size >= 8 {
        return raw;
    }
    let shift = 64 - size * 8;
    (((raw << shift) as i64) >> shift) as u64
}

crate::register_module_pass!(HandleJumpTables);

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;

    /// Number of CFG successors of `block`.
    fn successor_count(ctx: &Context, block: BlockId) -> usize {
        BasicBlock::from_id(ctx, block).successors().count()
    }

    /// Seed `ctx` with an executable code region `[start, start+len)`.
    fn add_code(ctx: &mut Context, start: u64, len: usize) {
        ctx.shared
            .memory_image
            .add_segment(start, vec![0u8; len], true, false);
    }

    /// Seed `ctx` with a read-only data region holding `bytes`.
    fn add_rodata(ctx: &mut Context, start: u64, bytes: Vec<u8>) {
        ctx.shared
            .memory_image
            .add_segment(start, bytes, false, false);
    }

    /// An absolute table: each 8-byte slot holds the target address directly.
    #[test]
    fn resolves_absolute_table() {
        let mut ctx = Context::new();
        let targets = [0x1100u64, 0x1200, 0x1300];

        qcode!(
            ctx,
            "
            varnode i64 A;
            fn fun:
            <entry>
                %idx = load(A:8, &A);
                %c = %idx < 0x3;
                if %c goto <disp> else goto <oob>;
            <disp>
                %off = %idx * 0x8;
                %addr = i64 0x2000 + %off;
                %t = load(ram:8, %addr);
                goto [%t];
            <oob>
                goto <0x9000>;
            "
        );

        // Executable code the targets live in, plus the rodata table itself.
        add_code(&mut ctx, 0x1000, 0x1000);
        let mut table = Vec::new();
        for t in targets {
            table.extend_from_slice(&t.to_le_bytes());
        }
        add_rodata(&mut ctx, 0x2000, table);

        let changed = HandleJumpTables::resolve_function(&mut ctx, fun).unwrap();
        assert!(changed);

        // The dispatch block gained one successor per case target.
        assert_eq!(successor_count(&ctx, disp), 3);
        let addresses = AddressIndex::analyze(&ctx);
        for t in targets {
            assert!(addresses.get(t).is_some(), "no block created for {t:#x}");
        }
        // Each slot read recorded an immutable-memory assumption.
        for i in 0..3u64 {
            let prop = Proposition::ImmutableMemory {
                addr: 0x2000 + i * 8,
                size: 8,
            };
            assert_eq!(ctx.truth(prop).map(|t| t.value), Some(true));
        }

        // A >2-case table keeps its `BranchInd`, so the pass re-resolves it on
        // every fixpoint iteration. Re-running against the already-connected
        // block must be a no-op reporting no change — otherwise any
        // `repeat_until` stage containing this pass never converges.
        let changed = HandleJumpTables::resolve_function(&mut ctx, fun).unwrap();
        assert!(
            !changed,
            "re-resolving an already-connected table reported a change"
        );
        assert_eq!(successor_count(&ctx, disp), 3);
    }

    /// A plain indirect jump through a fixed pointer slot: `goto [load(const)]`.
    /// The slot is immutable data holding the single target, so the indirect
    /// branch collapses to a direct jump with exactly one successor.
    #[test]
    fn resolves_constant_pointer_load() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;
            fn fun:
            <entry>
                %t = load(ram:8, 0x2000);
                goto [%t];
            "
        );

        add_code(&mut ctx, 0x1000, 0x1000);
        add_rodata(&mut ctx, 0x2000, 0x1100u64.to_le_bytes().to_vec());

        let changed = HandleJumpTables::resolve_function(&mut ctx, fun).unwrap();
        assert!(changed);

        // The indirect branch is now a direct jump to the one resolved target.
        assert_eq!(successor_count(&ctx, entry), 1);
        assert!(
            AddressIndex::analyze(&ctx).get(0x1100).is_some(),
            "no block for target"
        );

        // The slot read recorded an immutable-memory assumption.
        let prop = Proposition::ImmutableMemory {
            addr: 0x2000,
            size: 8,
        };
        assert_eq!(ctx.truth(prop).map(|t| t.value), Some(true));
    }

    /// A relative table: each 4-byte slot holds a signed offset added back to a
    /// constant base (the position-independent `switch` shape).
    #[test]
    fn resolves_relative_table() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;
            fn fun:
            <entry>
                %idx = load(A:8, &A);
                %c = %idx < 0x3;
                if %c goto <disp> else goto <oob>;
            <disp>
                %off = %idx * 0x4;
                %addr = i64 0x5000 + %off;
                %rel = load(ram:4, %addr);
                %sx = sext(i64, %rel);
                %t = i64 0x3000 + %sx;
                goto [%t];
            <oob>
                goto <0x9000>;
            "
        );

        // Targets are 0x3000 + {0x100, 0x200, 0x300}.
        add_code(&mut ctx, 0x3000, 0x1000);
        let mut table = Vec::new();
        for off in [0x100i32, 0x200, 0x300] {
            table.extend_from_slice(&off.to_le_bytes());
        }
        add_rodata(&mut ctx, 0x5000, table);

        let changed = HandleJumpTables::resolve_function(&mut ctx, fun).unwrap();
        assert!(changed);

        assert_eq!(successor_count(&ctx, disp), 3);
        let addresses = AddressIndex::analyze(&ctx);
        for t in [0x3100u64, 0x3200, 0x3300] {
            assert!(addresses.get(t).is_some(), "no block created for {t:#x}");
        }
    }

    /// An unbounded index (no dominating guard) leaves the indirect branch alone.
    #[test]
    fn unbounded_index_is_left_alone() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;
            fn fun:
            <disp>
                %idx = load(A:8, &A);
                %off = %idx * 0x8;
                %addr = i64 0x2000 + %off;
                %t = load(ram:8, %addr);
                goto [%t];
            "
        );
        add_code(&mut ctx, 0x1000, 0x1000);
        add_rodata(&mut ctx, 0x2000, vec![0u8; 0x100]);

        let changed = HandleJumpTables::resolve_function(&mut ctx, fun).unwrap();
        assert!(!changed);
        assert_eq!(successor_count(&ctx, disp), 0);
    }

    /// The mnemonic of the last instruction in `block`.
    fn terminator<'a>(ctx: &'a Context, block: BlockId) -> &'a Mnemonic {
        BasicBlock::from_id(ctx, block)
            .instructions()
            .last()
            .expect("block has a terminator")
            .mnemonic()
    }

    /// A single resolved target collapses the indirect branch to a direct,
    /// unconditional `Branch`.
    #[test]
    fn collapses_single_target_to_branch() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;
            fn fun:
            <entry>
                %idx = load(A:8, &A);
                %c = %idx < 0x1;
                if %c goto <disp> else goto <oob>;
            <disp>
                %off = %idx * 0x8;
                %addr = i64 0x2000 + %off;
                %t = load(ram:8, %addr);
                goto [%t];
            <oob>
                goto <0x9000>;
            "
        );

        add_code(&mut ctx, 0x1000, 0x1000);
        add_rodata(&mut ctx, 0x2000, 0x1100u64.to_le_bytes().to_vec());

        let changed = HandleJumpTables::resolve_function(&mut ctx, fun).unwrap();
        assert!(changed);

        // BranchInd became a direct Branch to the single target.
        assert_eq!(successor_count(&ctx, disp), 1);
        assert!(
            matches!(terminator(&ctx, disp), Mnemonic::Branch(_)),
            "expected an unconditional Branch, got {:?}",
            terminator(&ctx, disp),
        );
        assert!(AddressIndex::analyze(&ctx).get(0x1100).is_some());
    }

    /// Two targets with a zero case lower to `if index != 0` (the common shape).
    #[test]
    fn two_targets_with_zero_case() {
        let mut ctx = Context::new();
        let targets = [0x1100u64, 0x1200];

        qcode!(
            ctx,
            "
            varnode i64 A;
            fn fun:
            <entry>
                %idx = load(A:8, &A);
                %c = %idx < 0x2;
                if %c goto <disp> else goto <oob>;
            <disp>
                %off = %idx * 0x8;
                %addr = i64 0x2000 + %off;
                %t = load(ram:8, %addr);
                goto [%t];
            <oob>
                goto <0x9000>;
            "
        );

        add_code(&mut ctx, 0x1000, 0x1000);
        let mut table = Vec::new();
        for t in targets {
            table.extend_from_slice(&t.to_le_bytes());
        }
        add_rodata(&mut ctx, 0x2000, table);

        let changed = HandleJumpTables::resolve_function(&mut ctx, fun).unwrap();
        assert!(changed);

        assert_eq!(successor_count(&ctx, disp), 2);
        assert!(matches!(terminator(&ctx, disp), Mnemonic::CBranch(_)));
        // The condition compares the index against the false-case value, 0.
        assert!(
            cbranch_compares_against(&ctx, disp, 0),
            "expected `index != 0` guard",
        );
    }

    /// A bitwise-`&` of two booleans (`(a != 0) & (b != 0)`) bounds the index to
    /// `{0,1}` on its own — no dominating guard — so the two-slot table resolves
    /// to a `CBranch`. Mirrors `two_targets_with_zero_case` for the `&` index
    /// form a compiler emits when both sides are already 0/1.
    #[test]
    fn bitwise_and_index_resolves_two_targets() {
        let mut ctx = Context::new();
        let targets = [0x1100u64, 0x1200];

        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            fn fun:
            <disp>
                %a = load(A:8, &A);
                %b = load(B:8, &B);
                %na = %a != 0x0;
                %nb = %b != 0x0;
                %and = %na & %nb;
                %idx = zext(i64, %and);
                %off = %idx * 0x8;
                %addr = i64 0x2000 + %off;
                %t = load(ram:8, %addr);
                goto [%t];
            "
        );

        add_code(&mut ctx, 0x1000, 0x1000);
        let mut table = Vec::new();
        for t in targets {
            table.extend_from_slice(&t.to_le_bytes());
        }
        add_rodata(&mut ctx, 0x2000, table);

        let changed = HandleJumpTables::resolve_function(&mut ctx, fun).unwrap();
        assert!(changed);

        assert_eq!(successor_count(&ctx, disp), 2);
        assert!(matches!(terminator(&ctx, disp), Mnemonic::CBranch(_)));
        // Index 0 is the false case, so the guard is `index != 0`.
        assert!(
            cbranch_compares_against(&ctx, disp, 0),
            "expected `index != 0` guard",
        );
    }

    /// A block that already carries the lifter's jump-table edges (the clean-IR
    /// state after `discover_code` connects the targets) must still be rewritten
    /// from `BranchInd` to a `CBranch`, and must not end up with doubled edges.
    #[test]
    fn rewrites_already_connected_branch() {
        let mut ctx = Context::new();
        let targets = [0x1100u64, 0x1200];

        qcode!(
            ctx,
            "
            varnode i64 A;
            fn fun:
            <entry>
                %idx = load(A:8, &A);
                %c = %idx < 0x2;
                if %c goto <disp> else goto <oob>;
            <disp>
                %off = %idx * 0x8;
                %addr = i64 0x2000 + %off;
                %t = load(ram:8, %addr);
                goto [%t];
            <oob>
                goto <0x9000>;
            "
        );

        add_code(&mut ctx, 0x1000, 0x1000);
        let mut table = Vec::new();
        for t in targets {
            table.extend_from_slice(&t.to_le_bytes());
        }
        add_rodata(&mut ctx, 0x2000, table);

        // Pre-connect the dispatch block to its targets, mimicking the edges the
        // lifter materializes in the clean IR before this pass re-runs. The targets
        // live in the dispatch's own function (strict IR locality).
        let mut addresses = AddressIndex::analyze(&ctx);
        for t in targets {
            let tb = ctx.get_or_make_block_indexed(&mut addresses, t, fun);
            ctx.add_cfg_edge(disp, tb);
        }
        assert_eq!(successor_count(&ctx, disp), 2);

        let changed = HandleJumpTables::resolve_function(&mut ctx, fun).unwrap();
        assert!(changed);

        // Rewritten to a CBranch with exactly two successors (no doubling).
        assert_eq!(successor_count(&ctx, disp), 2);
        assert!(matches!(terminator(&ctx, disp), Mnemonic::CBranch(_)));
    }

    /// Two targets whose indices straddle a nonzero base lower to a comparison
    /// against the false-case index value, not a hardcoded zero.
    #[test]
    fn two_targets_without_zero_case() {
        let mut ctx = Context::new();

        // `%j = idx + 5`, with idx bounded to {0,1}, gives index range {5,6}.
        qcode!(
            ctx,
            "
            varnode i64 A;
            fn fun:
            <entry>
                %idx = load(A:8, &A);
                %c = %idx < 0x2;
                if %c goto <disp> else goto <oob>;
            <disp>
                %j = %idx + 0x5;
                %off = %j * 0x8;
                %addr = i64 0x2000 + %off;
                %t = load(ram:8, %addr);
                goto [%t];
            <oob>
                goto <0x9000>;
            "
        );

        add_code(&mut ctx, 0x1000, 0x1000);
        // Slots 0..=6; only slots 5 and 6 are read.
        let mut table = vec![0u8; 7 * 8];
        table[5 * 8..6 * 8].copy_from_slice(&0x1100u64.to_le_bytes());
        table[6 * 8..7 * 8].copy_from_slice(&0x1200u64.to_le_bytes());
        add_rodata(&mut ctx, 0x2000, table);

        let changed = HandleJumpTables::resolve_function(&mut ctx, fun).unwrap();
        assert!(changed);

        assert_eq!(successor_count(&ctx, disp), 2);
        assert!(matches!(terminator(&ctx, disp), Mnemonic::CBranch(_)));
        // The false case is index 5, so the guard is `index != 5`.
        assert!(
            cbranch_compares_against(&ctx, disp, 5),
            "expected `index != 5` guard",
        );
    }

    /// True if `block` contains a `NotEqual` comparison with a literal operand
    /// equal to `value` (the cbranch guard the pass synthesizes).
    fn cbranch_compares_against(ctx: &Context, block: BlockId, value: u64) -> bool {
        BasicBlock::from_id(ctx, block).instructions().any(|insn| {
            if let Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::NotEqual),
                lhs,
                rhs,
            }) = insn.mnemonic()
            {
                [*lhs, *rhs]
                    .into_iter()
                    .any(|v| numeric_const(ctx, v.qualify(insn.id.func)) == Some(value))
            } else {
                false
            }
        })
    }
}
