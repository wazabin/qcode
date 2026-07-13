//! Loop-invariant code motion.
//!
//! Hoists computations that produce the same value on every iteration out of a
//! loop and into its preheader, so they run once instead of once per iteration.
//!
//! A loop is recognized exactly as in [`loop_unroll`](crate::loop_unroll): a
//! back-edge `latch -> header` where `header` dominates `latch`, with the natural
//! loop being every block that reaches the latch without going through the header.
//! Hoisting targets the loop's unique **preheader** — the single predecessor of
//! the header outside the loop. Loops without a unique preheader are left alone
//! (matching the unroller's `Require existing preheader` policy).
//!
//! ## What is invariant
//!
//! An instruction is loop-invariant when every operand is invariant, where an
//! operand is invariant if it is a literal/varnode/function reference, a value
//! defined outside the loop, or another instruction already proven invariant.
//! Block parameters of loop blocks are loop-carried (they receive a fresh value
//! across the back-edge) and are therefore treated as variant. The invariant set
//! is grown to a fixpoint.
//!
//! ## What is safe to hoist
//!
//! - **Pure value ops** (arithmetic, casts, bit ops, intrinsics, aggregate
//!   plumbing): always safe — their result is a pure function of their operands.
//! - **Loads**: safe only when no store, call, or `pcodeop` in the loop may write
//!   the loaded location. Stores are checked with the alias oracle
//!   ([`AliasResult::may_alias`]); any call / indirect call / `pcodeop` in the
//!   loop is treated as an opaque memory clobber and blocks every load.
//!
//! Stores, calls, indirect branches, `pcodeop`s, terminators, and `map`s are
//! never hoisted.
//!
//! Hoisting a load makes it run even when the loop iterates zero times. The
//! loaded address is loop-invariant and was dereferenced on the original entry
//! path, so this speculation is benign for the reverse-engineering setting.

use rustc_hash::FxHashSet as HashSet;
use std::collections::VecDeque;

use jstd::graph::analysis::compute_dominators;
use qcode::value::{
    FunctionId, ValueId,
    block::BlockId,
    insn::{InstructionId, Mnemonic},
    util::base_ref::HostRef,
};

#[cfg(test)]
use qcode::context::Context;

use crate::{AliasResult, ContextView, FunctionBody, FunctionPass, Outcome};

#[derive(Default)]
pub struct Licm;

impl FunctionPass for Licm {
    const NAME: &'static str = "licm";

    fn description(&self) -> &'static str {
        "Hoist loop-invariant computations into the loop preheader"
    }

    fn run<'str>(
        &self,
        f: &mut FunctionBody<'str>,
        m: ContextView<'_, 'str>,
        _next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String> {
        let fid = f.id();
        let aliases = build_aliases(m, m.read_host(f), fid);
        Ok(Outcome::changed(hoist_loop_invariants_with_aliases(
            f,
            m,
            fid,
            aliases.as_ref(),
        )))
    }
}

crate::register_function_pass!(Licm);

/// Whether `m` is a side-effect-free value-computing op (mirrors the predicate
/// used by `loop_to_map`/`partial_inline`). Loads are deliberately *excluded*
/// here and handled separately under an alias guard.
fn is_pure_expr_op(m: &Mnemonic) -> bool {
    !matches!(
        m,
        Mnemonic::Load(_)
            | Mnemonic::Store(_)
            | Mnemonic::Call(_)
            | Mnemonic::CallInd(_)
            | Mnemonic::BranchInd(_)
            | Mnemonic::PCodeOp(_)
            | Mnemonic::Map(_)
            | Mnemonic::Scan(_)
    ) && !m.is_terminator()
}

/// The back-edges of `fun_id` as `(latch, header)` pairs, where `header`
/// dominates `latch`.
fn back_edges(host: HostRef, fun_id: FunctionId) -> Vec<(BlockId, BlockId)> {
    let function = host.function_ref(fun_id);
    let Some(root) = function.root().map(|b| b.id) else {
        return Vec::new();
    };
    let block_ids = function.iter().map(|b| b.id).collect::<Vec<_>>();
    let dominators = compute_dominators(&function, root);

    let mut edges = Vec::new();
    for &latch in &block_ids {
        let successors = host
            .block_ref(latch)
            .successors()
            .map(|(_, header)| header)
            .collect::<Vec<_>>();
        for header in successors {
            if dominators.dominates(header, latch) {
                edges.push((latch, header));
            }
        }
    }
    edges
}

/// Every block in the natural loop of the back-edge `latch -> header`: the
/// header plus every block that reaches the latch without passing through the
/// header.
fn natural_loop(host: HostRef, latch: BlockId, header: BlockId) -> HashSet<BlockId> {
    let mut nodes = HashSet::from_iter([header, latch]);
    let mut worklist = VecDeque::from([latch]);
    while let Some(block) = worklist.pop_front() {
        for (_, pred) in host.block_ref(block).predecessors() {
            if nodes.insert(pred) && pred != header {
                worklist.push_back(pred);
            }
        }
    }
    nodes
}

/// The loop's unique preheader: the single predecessor of `header` that is not
/// itself in the loop. `None` if there is zero or more than one such block.
fn loop_preheader(
    host: HostRef,
    header: BlockId,
    loop_nodes: &HashSet<BlockId>,
) -> Option<BlockId> {
    let header_ref = host.block_ref(header);
    let mut preheaders = header_ref
        .predecessors()
        .filter_map(|(_, pred)| (!loop_nodes.contains(&pred)).then_some(pred));
    let preheader = preheaders.next()?;
    preheaders.next().is_none().then_some(preheader)
}

/// Whether the value `v` is loop-invariant given the set of instructions already
/// known invariant.
fn value_is_invariant(
    host: HostRef,
    v: ValueId,
    loop_nodes: &HashSet<BlockId>,
    invariant: &HashSet<InstructionId>,
) -> bool {
    match v {
        ValueId::Literal(_)
        | ValueId::Bytes(_)
        | ValueId::Varnode(_)
        | ValueId::Function(_)
        | ValueId::BasicBlock(_) => true,
        ValueId::BlockParam(p) => {
            // A block param defined outside the loop is invariant; one belonging
            // to a loop block is loop-carried (fed across the back-edge), hence
            // variant.
            match host.block_param(p).parent_id() {
                Some(block) => !loop_nodes.contains(&BlockId::new(p.func, block)),
                None => true,
            }
        }
        ValueId::Instruction(i) => match host.insn_ref(i).parent().map(|b| b.id) {
            Some(block) if loop_nodes.contains(&block) => invariant.contains(&i),
            _ => true,
        },
        _ => false,
    }
}

/// The store pointers and whether the loop contains an opaque memory clobber
/// (call / indirect call / `pcodeop`), used to vet load hoisting.
struct LoopMemory {
    store_ptrs: Vec<ValueId>,
    has_clobber: bool,
}

fn loop_memory(host: HostRef, loop_nodes: &HashSet<BlockId>) -> LoopMemory {
    let mut store_ptrs = Vec::new();
    let mut has_clobber = false;
    for &block in loop_nodes {
        let insns = host.block_ref(block).instruction_ids().to_vec();
        for id in insns {
            match host.insn_ref(id).mnemonic() {
                Mnemonic::Store(s) => store_ptrs.push(s.ptr.qualify(id.func)),
                Mnemonic::Call(_) | Mnemonic::CallInd(_) | Mnemonic::PCodeOp(_) => {
                    has_clobber = true;
                }
                _ => {}
            }
        }
    }
    LoopMemory {
        store_ptrs,
        has_clobber,
    }
}

/// Whether a loop load through `ptr` keeps its value across every iteration: no
/// opaque clobber, and no loop store that may-alias `ptr`.
fn load_is_safe(
    host: HostRef,
    aliases: Option<&AliasResult>,
    ptr: ValueId,
    mem: &LoopMemory,
) -> bool {
    if mem.has_clobber {
        return false;
    }
    let Some(aliases) = aliases else {
        // Without an oracle, any loop store conservatively kills the load.
        return mem.store_ptrs.is_empty();
    };
    !mem.store_ptrs
        .iter()
        .any(|&store| aliases.may_alias(host, ptr, store))
}

/// Grow the set of loop-invariant instructions to a fixpoint.
fn invariant_instructions(
    host: HostRef,
    loop_nodes: &HashSet<BlockId>,
    aliases: Option<&AliasResult>,
    mem: &LoopMemory,
) -> HashSet<InstructionId> {
    let mut invariant: HashSet<InstructionId> = HashSet::default();
    loop {
        let mut changed = false;
        for &block in loop_nodes {
            let insns = host.block_ref(block).instruction_ids().to_vec();
            for id in insns {
                if invariant.contains(&id) {
                    continue;
                }
                let m = host.insn_ref(id).mnemonic();
                let is_load = matches!(m, Mnemonic::Load(_));
                if !is_load && !is_pure_expr_op(m) {
                    continue;
                }
                let operands_invariant = m.args().into_iter().all(|op| {
                    value_is_invariant(host, op.qualify(id.func), loop_nodes, &invariant)
                });
                if !operands_invariant {
                    continue;
                }
                if let Mnemonic::Load(load) = m
                    && !load_is_safe(host, aliases, load.ptr.qualify(id.func), mem)
                {
                    continue;
                }
                invariant.insert(id);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    invariant
}

/// Order `invariant` so every instruction follows the invariant operands it
/// depends on (post-order over the dependency DAG), giving a placement order
/// that keeps definitions before uses in the preheader.
fn emission_order(
    host: HostRef,
    loop_nodes: &HashSet<BlockId>,
    invariant: &HashSet<InstructionId>,
) -> Vec<InstructionId> {
    let mut order = Vec::new();
    let mut done: HashSet<InstructionId> = HashSet::default();
    // Iterate blocks/instructions for a deterministic starting order.
    let mut roots: Vec<InstructionId> = Vec::new();
    for &block in loop_nodes {
        for id in host.block_ref(block).instruction_ids() {
            if invariant.contains(&id) {
                roots.push(id);
            }
        }
    }
    for root in roots {
        if done.contains(&root) {
            continue;
        }
        // Iterative post-order DFS.
        let mut stack = vec![(root, false)];
        while let Some((id, expanded)) = stack.pop() {
            if expanded {
                if done.insert(id) {
                    order.push(id);
                }
                continue;
            }
            if done.contains(&id) {
                continue;
            }
            stack.push((id, true));
            for op in host.insn_ref(id).operands() {
                if let ValueId::Instruction(o) = op
                    && invariant.contains(&o)
                    && !done.contains(&o)
                {
                    stack.push((o, false));
                }
            }
        }
    }
    order
}

/// Hoist invariant instructions of one loop into `preheader`, in `order`. Each
/// instruction is rebuilt in the preheader (before its terminator), its uses are
/// redirected to the rebuilt copy, and the original is deleted.
fn hoist_into_preheader<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    preheader: BlockId,
    order: &[InstructionId],
) -> bool {
    let mut hoisted = false;
    for &old in order {
        let old_ref = cx.read_host(body).insn_ref(old);
        let mnemonic = old_ref.mnemonic().clone();
        let type_id = old_ref.type_id();
        let new = body.push_mnemonic_with_type(mnemonic, type_id);

        let term = *cx
            .read_host(body)
            .block_ref(preheader)
            .instruction_ids()
            .last()
            .expect("preheader must have a terminator");
        body.insert_insn_before(preheader, term, new);

        // Redirect every remaining use (in the loop and beyond) to the hoisted
        // copy. Processing in dependency order means a later invariant operand
        // already points at its hoisted copy when we clone the consumer.
        body.replace_all_uses_with(ValueId::Instruction(old), ValueId::Instruction(new));
        hoisted = true;
    }

    // Delete the now-dead originals from their loop blocks.
    for &old in order {
        body.remove_instruction(old);
    }
    hoisted
}

/// The `&mut Context` version of [`hoist_into_preheader`], kept for the
/// `&mut Context` test entry point below.
/// TODO(5b-ii): remove once tests migrate off the `&mut Context` helper.
#[cfg(test)]
fn hoist_into_preheader_generic<'str>(
    host: &mut Context<'str>,
    preheader: BlockId,
    order: &[InstructionId],
) -> bool {
    let mut hoisted = false;
    for &old in order {
        let old_ref = host.insn_ref(old);
        let mnemonic = old_ref.mnemonic().clone();
        let type_id = old_ref.type_id();
        let new = host.push_mnemonic_with_type(preheader.func, mnemonic, type_id);

        let term = *host
            .block_ref(preheader)
            .instruction_ids()
            .last()
            .expect("preheader must have a terminator");
        host.insert_insn_before(preheader, term, new);

        // Redirect every remaining use (in the loop and beyond) to the hoisted
        // copy. Processing in dependency order means a later invariant operand
        // already points at its hoisted copy when we clone the consumer.
        host.replace_all_uses_with(ValueId::Instruction(old), ValueId::Instruction(new));
        hoisted = true;
    }

    // Delete the now-dead originals from their loop blocks.
    for &old in order {
        host.remove_instruction(old);
    }
    hoisted
}

/// Build the per-function alias oracle the same way [`crate::gvn::Gvn`] does, so
/// load hoisting sees per-slot stack locations and frame freshness. Returns
/// `None` when no stack pointer is registered (arch-agnostic test envs).
fn build_aliases<'a, 'str: 'a>(
    m: ContextView<'_, 'str>,
    host: HostRef<'a, 'str>,
    fun_id: FunctionId,
) -> Option<AliasResult> {
    let shared = m.shr();
    let env = m.env();
    let sp_reg = shared.registers.get(&env.cfg.stack_pointer).copied()?;
    Some(
        env.alias_base(shared)
            .for_function(host, fun_id)
            .with_frame_freshness(host, fun_id, Some(sp_reg)),
    )
}

/// Hoisting core, parameterized on an already-built alias oracle (or `None` for
/// the conservative path where any loop store blocks load hoisting). Reads and
/// mutates the function through the concrete pass surface.
fn hoist_loop_invariants_with_aliases<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    fun_id: FunctionId,
    aliases: Option<&AliasResult>,
) -> bool {
    let edges = back_edges(cx.read_host(body), fun_id);
    if edges.is_empty() {
        return false;
    }

    // Plan all hoists first (read-only), then apply. Moving instructions never
    // changes the CFG, so dominators / loop membership stay valid throughout.
    let mut plans: Vec<(BlockId, Vec<InstructionId>)> = Vec::new();
    for (latch, header) in edges {
        let loop_nodes = natural_loop(cx.read_host(body), latch, header);
        let Some(preheader) = loop_preheader(cx.read_host(body), header, &loop_nodes) else {
            continue;
        };
        let mem = loop_memory(cx.read_host(body), &loop_nodes);
        let invariant = invariant_instructions(cx.read_host(body), &loop_nodes, aliases, &mem);
        if invariant.is_empty() {
            continue;
        }
        let order = emission_order(cx.read_host(body), &loop_nodes, &invariant);
        plans.push((preheader, order));
    }

    let mut changed = false;
    for (preheader, order) in plans {
        changed |= hoist_into_preheader(body, cx, preheader, &order);
    }
    changed
}

/// The `&mut Context` version of [`hoist_loop_invariants_with_aliases`]; kept
/// for tests that drive the hoist over a bare `&mut Context` with their own
/// oracle.
/// TODO(5b-ii): remove once tests migrate off the `&mut Context` helper.
#[cfg(test)]
fn hoist_loop_invariants_with_aliases_generic<'str>(
    host: &mut Context<'str>,
    fun_id: FunctionId,
    aliases: Option<&AliasResult>,
) -> bool {
    let edges = back_edges(host.read_host(), fun_id);
    if edges.is_empty() {
        return false;
    }

    // Plan all hoists first (read-only), then apply. Moving instructions never
    // changes the CFG, so dominators / loop membership stay valid throughout.
    let mut plans: Vec<(BlockId, Vec<InstructionId>)> = Vec::new();
    for (latch, header) in edges {
        let loop_nodes = natural_loop(host.read_host(), latch, header);
        let Some(preheader) = loop_preheader(host.read_host(), header, &loop_nodes) else {
            continue;
        };
        let mem = loop_memory(host.read_host(), &loop_nodes);
        let invariant = invariant_instructions(host.read_host(), &loop_nodes, aliases, &mem);
        if invariant.is_empty() {
            continue;
        }
        let order = emission_order(host.read_host(), &loop_nodes, &invariant);
        plans.push((preheader, order));
    }

    let mut changed = false;
    for (preheader, order) in plans {
        changed |= hoist_into_preheader_generic(host, preheader, &order);
    }
    changed
}

#[cfg(test)]
mod tests;
