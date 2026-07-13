//! Redundant block-argument elimination.
//!
//! A block parameter `p` is *redundant* when, across every predecessor edge, the
//! argument bound to it is either a single common value `v0` or `p` itself. Then
//! `p` is congruent to `v0` on every path that reaches the block, so `p` can be
//! replaced by `v0` everywhere and dropped, along with the matching positional
//! argument at each predecessor terminator.
//!
//! The canonical case is a loop-invariant value threaded around a back-edge: the
//! header param gets `v0` from the off-loop edge and itself (`p = p`) from the
//! latch edge. Since `v0` is the only non-self value and it dominates every
//! predecessor (it is used on each), it dominates the header too, so the
//! substitution is sound. This is exactly redundant-φ elimination.
//!
//! Removing one redundant param can expose another — e.g. two params that only
//! forward each other around a loop (`p ← {v0, q}`, `q ← {p}`): `q` collapses to
//! `p` first, which turns `p`'s latch argument into `p = p`, leaving `v0` as its
//! only non-self source. The sweep therefore iterates to a fixpoint.
//!
//! Unlike [`remove_unused_no_pred_block_params`](super::remove_unused_no_pred_block_params),
//! which drops zero-user params on blocks with *no* incoming edge, this works on
//! join/loop blocks by reconciling their incoming arguments. The function entry
//! (`root`) is intentionally left alone: its params are the function's interface
//! and must be removed through `remove_entry_param` to keep the ABI aligned.

use rustc_hash::FxHashSet as HashSet;

use jstd::graph::analysis::{DominatorTree, compute_dominators};

use qcode::{
    context::Context,
    value::{
        BlockId, BlockParamId, FunctionId, LocalValueId, ValueId,
        insn::{Branch, CBranch, Mnemonic},
        util::{base_ref::HostRef, host_mut::PassBacking},
    },
};

use crate::{ContextView, FunctionBody};

// TODO(5b-ii): Public functions below are thin wrappers marked for future migration

use crate::gvn::affine::precompute_forms_for_blocks;
use crate::gvn::congruence::{Congruence, SymId};

/// Collect the distinct values bound to position `index` of `block` across all
/// predecessor edges, excluding the param's own value `self_val` (self-edges
/// carry no new information).
///
/// Returns `Some(v0)` when exactly one such value remains — the replacement the
/// param collapses to — and `None` when the param is a genuine merge of two or
/// more distinct values (not redundant) or has no non-self incoming value.
fn unique_incoming(
    host: HostRef,
    block: BlockId,
    index: usize,
    self_val: ValueId,
) -> Option<ValueId> {
    // Dedup predecessor blocks: a `CBranch` whose two edges both target `block`
    // shows up twice but its terminator is read once below (handling both arms).
    let preds: HashSet<BlockId> = host
        .block_ref(block)
        .predecessors()
        .map(|(_, b)| b)
        .collect();

    let mut found: Option<ValueId> = None;
    for pred in preds {
        let Some(&term_id) = host.block_ref(pred).instruction_ids().last() else {
            continue;
        };
        let mut consider = |arg: ValueId| -> bool {
            if arg == self_val {
                return true;
            }
            match found {
                Some(v) if v == arg => true,
                Some(_) => false, // a second distinct value: not redundant
                None => {
                    found = Some(arg);
                    true
                }
            }
        };

        let q = |t| BlockId::new(pred.func, t);
        let ok = match host.insn_ref(term_id).mnemonic() {
            Mnemonic::Branch(b) if q(b.target) == block => b
                .args
                .get(index)
                .map(|a| a.qualify(pred.func))
                .is_some_and(&mut consider),
            Mnemonic::CBranch(c) => {
                let mut ok = true;
                if q(c.success_block) == block {
                    ok &= c
                        .success_args
                        .get(index)
                        .map(|a| a.qualify(pred.func))
                        .is_some_and(&mut consider);
                }
                if q(c.failure_block) == block {
                    ok &= c
                        .failure_args
                        .get(index)
                        .map(|a| a.qualify(pred.func))
                        .is_some_and(&mut consider);
                }
                ok
            }
            // An indirect terminator carries no per-target argument list, so a
            // block reached that way has no incoming value to reconcile.
            _ => return None,
        };
        if !ok {
            return None;
        }
    }

    found
}

/// Replace every block parameter across `block_ids` that is redundant — bound to
/// a single common value (or itself) on all incoming edges — with that value,
/// then drop the param and its predecessor arguments. Iterates to a fixpoint so
/// chains and cycles collapse. Leaves `root`'s params untouched.
///
/// Returns whether anything was removed.
/// TODO(5b-ii): Takes Context; migrate to FunctionBody/ContextView when public API stabilizes.
pub fn remove_dead_block_args(
    ctx: &mut Context,
    block_ids: &[BlockId],
    root: Option<BlockId>,
) -> bool {
    remove_dead_block_args_generic(ctx, block_ids, root)
}

/// The `&mut Context` version of [`remove_dead_block_args`]'s core.
/// TODO(5b-ii): For backwards compatibility; prefer concrete version for new code.
pub fn remove_dead_block_args_generic<'str>(
    host: &mut Context<'str>,
    block_ids: &[BlockId],
    root: Option<BlockId>,
) -> bool {
    let mut changed = false;
    loop {
        // Cheap syntactic pass first (no value numbering); only when it is
        // exhausted do we build the dominator tree + congruence engine to catch
        // params whose incoming arguments are *congruent* but not identical.
        let found = find_redundant_param(host.read_host(), block_ids, root)
            .or_else(|| find_congruent_param(host.read_host(), block_ids, root));
        let Some((block, index, param, repl)) = found else {
            break;
        };

        // `p ≡ repl`: rewrite every use, then strip the param and the now-removed
        // column of arguments from each predecessor.
        host.replace_all_uses_with(ValueId::BlockParam(param), repl);
        remove_params_from_block_generic(host, block, &HashSet::from_iter([index]));
        changed = true;
    }
    changed
}

/// Host-generic core of [`remove_dead_block_args`]; see that function.
/// This is the concrete version for FunctionBody/ContextView (stage 5b).
pub fn remove_dead_block_args_host<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    block_ids: &[BlockId],
    root: Option<BlockId>,
) -> bool {
    let mut changed = false;
    loop {
        // Cheap syntactic pass first (no value numbering); only when it is
        // exhausted do we build the dominator tree + congruence engine to catch
        // params whose incoming arguments are *congruent* but not identical.
        let found = find_redundant_param(cx.read_host(body), block_ids, root)
            .or_else(|| find_congruent_param(cx.read_host(body), block_ids, root));
        let Some((block, index, param, repl)) = found else {
            break;
        };

        // `p ≡ repl`: rewrite every use, then strip the param and the now-removed
        // column of arguments from each predecessor.
        body.replace_all_uses_with(ValueId::BlockParam(param), repl);
        remove_params_from_block_host(body, cx, block, &HashSet::from_iter([index]));
        changed = true;
    }
    changed
}

/// Remove block parameters that are *dead by liveness*: a param whose value is
/// never observed — it feeds no real instruction, no branch condition, no call or
/// return — and only flows, around branch edges, into other equally-dead params.
///
/// This is the complement of [`remove_dead_block_args`]. That pass collapses a
/// param whose incoming values all *agree* (a redundant φ); this one drops a param
/// whose value is *unused* regardless of what it carries. The motivating case is a
/// loop register that is recomputed every iteration from the induction variable
/// and threaded around the back-edge but never read (e.g. `<40b3f0>`'s `@EAX`,
/// flags, `@EIP`): each iteration binds a *distinct* value, so it is not redundant,
/// yet it is a back-edge argument, so a use-counting instruction DCE sees it as
/// live. Only a transitive liveness fixpoint over the param graph removes it.
///
/// A param is live iff:
///   * it is an operand of some non-branch instruction, a `CBranch` condition, an
///     indirect branch/call pointer, a call argument, or the return slot
///     (a *direct* use — the value is observed), **or**
///   * it is passed, on a `Branch`/`CBranch` edge, into a param that is itself live
///     (it is observed indirectly, through the live param it feeds).
///
/// Root (entry) and `protected` params are seeded live: they are the function
/// interface / pinned, never removed, and they keep their sources alive. The pass
/// iterates to a fixpoint over the forwarding edges, then drops every param that
/// stays dead, stripping the matching predecessor argument columns.
///
/// Returns whether anything was removed.
/// TODO(5b-ii): Takes Context; migrate to FunctionBody/ContextView when public API stabilizes.
pub fn remove_dead_block_params(
    ctx: &mut Context,
    block_ids: &[BlockId],
    root: Option<BlockId>,
) -> bool {
    remove_dead_block_params_generic(ctx, block_ids, root)
}

/// The `&mut Context` version of [`remove_dead_block_params`]'s core.
/// TODO(5b-ii): For backwards compatibility; prefer concrete version for new code.
pub fn remove_dead_block_params_generic<'str>(
    host: &mut Context<'str>,
    block_ids: &[BlockId],
    root: Option<BlockId>,
) -> bool {
    // Seed: directly-used params. Edges: forwarding (src param -> target param) on
    // every branch-argument slot.
    let mut live: HashSet<BlockParamId> = HashSet::default();
    let mut edges: Vec<(BlockParamId, BlockParamId)> = Vec::new();

    let mark = |v: ValueId, live: &mut HashSet<BlockParamId>| {
        if let ValueId::BlockParam(p) = v {
            live.insert(p);
        }
    };

    for &block in block_ids {
        let insns: Vec<_> = host.block_ref(block).iter().map(|i| i.id).collect();
        for id in insns {
            match host.insn_ref(id).mnemonic() {
                Mnemonic::Branch(b) => forward_edges(
                    host.read_host(),
                    block.func,
                    &b.args,
                    BlockId::new(block.func, b.target),
                    &mut edges,
                ),
                Mnemonic::CBranch(c) => {
                    // The condition is a real read; only the per-target argument
                    // lists are forwarding edges.
                    if let ValueId::BlockParam(p) = c.condition.qualify(block.func) {
                        live.insert(p);
                    }
                    forward_edges(
                        host.read_host(),
                        block.func,
                        &c.success_args,
                        BlockId::new(block.func, c.success_block),
                        &mut edges,
                    );
                    forward_edges(
                        host.read_host(),
                        block.func,
                        &c.failure_args,
                        BlockId::new(block.func, c.failure_block),
                        &mut edges,
                    );
                }
                // Every other instruction (incl. indirect branch/call pointers,
                // call args, the return slot) observes all of its operands.
                other => {
                    for v in other.args() {
                        mark(v.qualify(block.func), &mut live);
                    }
                }
            }
        }
    }

    // Seed root + protected params live, then propagate liveness backwards along
    // the forwarding edges to a fixpoint: a param feeding a live param is live.
    if let Some(root) = root {
        for &local in host.read_host().block(root).param_ids() {
            live.insert(BlockParamId::new(root.func, local));
        }
    }
    for &(src, _) in &edges {
        if host.read_host().block_param(src).protected {
            live.insert(src);
        }
    }
    loop {
        let mut grew = false;
        for &(src, tgt) in &edges {
            if live.contains(&tgt) && live.insert(src) {
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }

    // Collect dead params per block (protected params are always seeded live, so
    // they never appear here; root params likewise).
    let mut dead_by_block: rustc_hash::FxHashMap<BlockId, HashSet<usize>> = Default::default();
    for &block in block_ids {
        for (index, &local) in host.read_host().block(block).param_ids().iter().enumerate() {
            let p = BlockParamId::new(block.func, local);
            if !live.contains(&p) {
                dead_by_block.entry(block).or_default().insert(index);
            }
        }
    }

    if dead_by_block.is_empty() {
        return false;
    }
    for (block, indices) in dead_by_block {
        remove_params_from_block_generic(host, block, &indices);
    }
    true
}

/// Host-generic core of [`remove_dead_block_params`]; see that function.
/// This is the concrete version for FunctionBody/ContextView (stage 5b).
pub fn remove_dead_block_params_host<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    block_ids: &[BlockId],
    root: Option<BlockId>,
) -> bool {
    // Seed: directly-used params. Edges: forwarding (src param -> target param) on
    // every branch-argument slot.
    let mut live: HashSet<BlockParamId> = HashSet::default();
    let mut edges: Vec<(BlockParamId, BlockParamId)> = Vec::new();

    let mark = |v: ValueId, live: &mut HashSet<BlockParamId>| {
        if let ValueId::BlockParam(p) = v {
            live.insert(p);
        }
    };

    for &block in block_ids {
        let insns: Vec<_> = cx
            .read_host(body)
            .block_ref(block)
            .iter()
            .map(|i| i.id)
            .collect();
        for id in insns {
            match cx.read_host(body).insn_ref(id).mnemonic() {
                Mnemonic::Branch(b) => forward_edges(
                    cx.read_host(body),
                    block.func,
                    &b.args,
                    BlockId::new(block.func, b.target),
                    &mut edges,
                ),
                Mnemonic::CBranch(c) => {
                    // The condition is a real read; only the per-target argument
                    // lists are forwarding edges.
                    if let ValueId::BlockParam(p) = c.condition.qualify(block.func) {
                        live.insert(p);
                    }
                    forward_edges(
                        cx.read_host(body),
                        block.func,
                        &c.success_args,
                        BlockId::new(block.func, c.success_block),
                        &mut edges,
                    );
                    forward_edges(
                        cx.read_host(body),
                        block.func,
                        &c.failure_args,
                        BlockId::new(block.func, c.failure_block),
                        &mut edges,
                    );
                }
                // Every other instruction (incl. indirect branch/call pointers,
                // call args, the return slot) observes all of its operands.
                other => {
                    for v in other.args() {
                        mark(v.qualify(block.func), &mut live);
                    }
                }
            }
        }
    }

    // Seed root + protected params live, then propagate liveness backwards along
    // the forwarding edges to a fixpoint: a param feeding a live param is live.
    if let Some(root) = root {
        for &local in cx.read_host(body).block(root).param_ids() {
            live.insert(BlockParamId::new(root.func, local));
        }
    }
    for &(src, _) in &edges {
        if cx.read_host(body).block_param(src).protected {
            live.insert(src);
        }
    }
    loop {
        let mut grew = false;
        for &(src, tgt) in &edges {
            if live.contains(&tgt) && live.insert(src) {
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }

    // Collect dead params per block (protected params are always seeded live, so
    // they never appear here; root params likewise).
    let mut dead_by_block: rustc_hash::FxHashMap<BlockId, HashSet<usize>> = Default::default();
    for &block in block_ids {
        for (index, &local) in cx
            .read_host(body)
            .block(block)
            .param_ids()
            .iter()
            .enumerate()
        {
            let p = BlockParamId::new(block.func, local);
            if !live.contains(&p) {
                dead_by_block.entry(block).or_default().insert(index);
            }
        }
    }

    if dead_by_block.is_empty() {
        return false;
    }
    for (block, indices) in dead_by_block {
        remove_params_from_block_host(body, cx, block, &indices);
    }
    true
}

/// Record a forwarding edge `src_param -> target_param` for each branch argument
/// at `target`'s matching param slot that is itself a block parameter.
fn forward_edges(
    host: HostRef,
    func: FunctionId,
    args: &[LocalValueId],
    target: BlockId,
    edges: &mut Vec<(BlockParamId, BlockParamId)>,
) {
    let params = &host.block(target).params;
    for (i, &a) in args.iter().enumerate() {
        if let (ValueId::BlockParam(src), Some(&tgt)) = (a.qualify(func), params.get(i)) {
            edges.push((src, BlockParamId::new(target.func, tgt)));
        }
    }
}

/// Scan for the first param that is *congruent*-redundant: every non-self
/// incoming argument computes the same value (by structural value number, not
/// just the same `ValueId`), and at least one of those arguments comes from a
/// predecessor that dominates the block — so it dominates every use of the param
/// and can replace it.
///
/// This is the loop-aware generalization of [`find_redundant_param`]: it unifies
/// reassociated/recomputed pure expressions (e.g. an off-loop `%a + %b` and a
/// back-edge `%b + %a`) that the syntactic pass leaves as a genuine merge. Loads
/// and calls stay identity leaves in the congruence, so no value that depends on
/// mutable memory is ever assumed stable across iterations.
fn find_congruent_param(
    host: HostRef,
    block_ids: &[BlockId],
    root: Option<BlockId>,
) -> Option<(BlockId, usize, BlockParamId, ValueId)> {
    let root = root?;
    let dom = compute_dominators(&host.function_ref(root.func), root);
    let mut cong = Congruence::new(precompute_forms_for_blocks(host, block_ids));

    for &block in block_ids {
        if block == root {
            continue;
        }
        if host.block_ref(block).predecessors().next().is_none() {
            continue;
        }
        let params = host.block(block).params.clone();
        for (index, &local) in params.iter().enumerate() {
            let param = BlockParamId::new(block.func, local);
            if host.block_param(param).protected {
                continue;
            }
            if let Some(repl) = congruent_incoming(
                host,
                &mut cong,
                &dom,
                block,
                index,
                ValueId::BlockParam(param),
            ) {
                return Some((block, index, param, repl));
            }
        }
    }
    None
}

/// The replacement for a congruent-redundant param, or `None`.
///
/// Groups the incoming arguments at `index` by structural value number, ignoring
/// any that are congruent to the param itself (loop pass-throughs). If all
/// remaining arguments share one value number and at least one comes from a
/// predecessor that dominates `block`, returns that dominating argument — it
/// dominates `block` and hence every use of the param, so the substitution is
/// sound. Returns `None` for a genuine merge of distinct values or when no
/// dominating predecessor carries the value.
fn congruent_incoming(
    host: HostRef,
    cong: &mut Congruence,
    dom: &DominatorTree<BlockId>,
    block: BlockId,
    index: usize,
    self_val: ValueId,
) -> Option<ValueId> {
    let self_sym = cong.id(host, self_val);

    // Dedup predecessor blocks (a `CBranch` with both edges to `block` lists it
    // twice but its terminator is read once, covering both arms).
    let preds: HashSet<BlockId> = host
        .block_ref(block)
        .predecessors()
        .map(|(_, b)| b)
        .collect();

    let mut common: Option<SymId> = None;
    let mut dom_repl: Option<ValueId> = None;

    for pred in preds {
        let Some(&term_local) = host.block(pred).instructions.last() else {
            continue;
        };
        let term_id = qcode::value::insn::InstructionId::new(pred.func, term_local);
        for arg in incoming_args(host, term_id, block, index) {
            let s = cong.id(host, arg);
            if s == self_sym {
                continue; // self / congruent-to-self: a pass-through edge
            }
            match common {
                Some(c) if c == s => {}
                Some(_) => return None, // a second distinct value: a real merge
                None => common = Some(s),
            }
            if dom.dominates(pred, block) {
                dom_repl.get_or_insert(arg);
            }
        }
    }

    common?; // there must be at least one non-self incoming value
    dom_repl
}

/// The argument bound to position `index` of `block` by `term_id` (a
/// predecessor's terminator), considering both arms of a `CBranch`.
fn incoming_args(
    host: HostRef,
    term_id: qcode::value::insn::InstructionId,
    block: BlockId,
    index: usize,
) -> Vec<ValueId> {
    let mut out = Vec::new();
    let q = |t| BlockId::new(term_id.func, t);
    match host.insn_ref(term_id).mnemonic() {
        Mnemonic::Branch(b) if q(b.target) == block => {
            out.extend(b.args.get(index).map(|a| a.qualify(term_id.func)))
        }
        Mnemonic::CBranch(c) => {
            if q(c.success_block) == block {
                out.extend(c.success_args.get(index).map(|a| a.qualify(term_id.func)));
            }
            if q(c.failure_block) == block {
                out.extend(c.failure_args.get(index).map(|a| a.qualify(term_id.func)));
            }
        }
        _ => {}
    }
    out
}

/// Scan for the first redundant param: a non-root, non-protected param on a block
/// with predecessors whose incoming arguments reduce to a single value `repl`.
fn find_redundant_param(
    host: HostRef,
    block_ids: &[BlockId],
    root: Option<BlockId>,
) -> Option<(BlockId, usize, BlockParamId, ValueId)> {
    for &block in block_ids {
        if Some(block) == root {
            continue;
        }
        if host.block_ref(block).predecessors().next().is_none() {
            continue;
        }
        let params = host.block(block).params.clone();
        for (index, &local) in params.iter().enumerate() {
            let param = BlockParamId::new(block.func, local);
            if host.block_param(param).protected {
                continue;
            }
            if let Some(repl) = unique_incoming(host, block, index, ValueId::BlockParam(param)) {
                return Some((block, index, param, repl));
            }
        }
    }
    None
}

/// Drop the params at `dead_indices` from `block`, reindexing the survivors, and
/// strip the matching positional argument from every predecessor terminator.
pub(crate) fn remove_params_from_block(
    ctx: &mut Context,
    block: BlockId,
    dead_indices: &HashSet<usize>,
) {
    remove_params_from_block_generic(ctx, block, dead_indices);
}

/// Module (`&mut Context`) core of [`remove_params_from_block`]; the checked-out
/// pass path uses [`remove_params_from_block_c`].
pub(crate) fn remove_params_from_block_generic<'str>(
    host: &mut Context<'str>,
    block: BlockId,
    dead_indices: &HashSet<usize>,
) {
    let params = host.read_host().block(block).params.clone();
    let mut kept = Vec::with_capacity(params.len());
    let mut removed = Vec::new();
    for (i, &local) in params.iter().enumerate() {
        let p = BlockParamId::new(block.func, local);
        if dead_indices.contains(&i) {
            removed.push(p);
        } else {
            host.block_param_mut(p).index = kept.len();
            kept.push(local);
        }
    }
    host.block_mut(block).params = kept;

    // A predecessor reaching `block` through both edges of a `CBranch` appears
    // twice; dedup so we rewrite its terminator exactly once.
    let preds: HashSet<BlockId> = host
        .block_ref(block)
        .predecessors()
        .map(|(_, b)| b)
        .collect();

    for pred in preds {
        let Some(term_local) = host.read_host().block(pred).instructions.last().copied() else {
            continue;
        };
        let term_id = qcode::value::insn::InstructionId::new(pred.func, term_local);
        let new = match host.read_host().instruction(term_id).mnemonic().clone() {
            Mnemonic::Branch(b) if BlockId::new(pred.func, b.target) == block => {
                Mnemonic::Branch(Branch {
                    target: b.target,
                    args: filter_kept(&b.args, dead_indices),
                })
            }
            Mnemonic::CBranch(c) => Mnemonic::CBranch(CBranch {
                condition: c.condition,
                success_block: c.success_block,
                success_args: if BlockId::new(pred.func, c.success_block) == block {
                    filter_kept(&c.success_args, dead_indices)
                } else {
                    c.success_args
                },
                failure_block: c.failure_block,
                failure_args: if BlockId::new(pred.func, c.failure_block) == block {
                    filter_kept(&c.failure_args, dead_indices)
                } else {
                    c.failure_args
                },
            }),
            // Indirect terminators carry no per-target argument list, so a block
            // reached that way has no params to feed and never reaches here.
            _ => continue,
        };
        host.replace_instruction_mnemonic(term_id, new);
    }
    for param in removed {
        host.remove_block_param(param);
    }
}

/// Host-generic core of [`remove_params_from_block`]; see that function.
/// This is the concrete version for FunctionBody/ContextView (stage 5b).
pub(crate) fn remove_params_from_block_host<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    block: BlockId,
    dead_indices: &HashSet<usize>,
) {
    let params = cx.read_host(body).block(block).params.clone();
    let mut kept = Vec::with_capacity(params.len());
    let mut removed = Vec::new();
    for (i, &local) in params.iter().enumerate() {
        let p = BlockParamId::new(block.func, local);
        if dead_indices.contains(&i) {
            removed.push(p);
        } else {
            body.block_param_mut(p).index = kept.len();
            kept.push(local);
        }
    }
    body.block_mut(block).params = kept;

    // A predecessor reaching `block` through both edges of a `CBranch` appears
    // twice; dedup so we rewrite its terminator exactly once.
    let preds: HashSet<BlockId> = cx
        .read_host(body)
        .block_ref(block)
        .predecessors()
        .map(|(_, b)| b)
        .collect();

    for pred in preds {
        let Some(term_local) = cx.read_host(body).block(pred).instructions.last().copied() else {
            continue;
        };
        let term_id = qcode::value::insn::InstructionId::new(pred.func, term_local);
        let new = match cx.read_host(body).instruction(term_id).mnemonic().clone() {
            Mnemonic::Branch(b) if BlockId::new(pred.func, b.target) == block => {
                Mnemonic::Branch(Branch {
                    target: b.target,
                    args: filter_kept(&b.args, dead_indices),
                })
            }
            Mnemonic::CBranch(c) => Mnemonic::CBranch(CBranch {
                condition: c.condition,
                success_block: c.success_block,
                success_args: if BlockId::new(pred.func, c.success_block) == block {
                    filter_kept(&c.success_args, dead_indices)
                } else {
                    c.success_args
                },
                failure_block: c.failure_block,
                failure_args: if BlockId::new(pred.func, c.failure_block) == block {
                    filter_kept(&c.failure_args, dead_indices)
                } else {
                    c.failure_args
                },
            }),
            // Indirect terminators carry no per-target argument list, so a block
            // reached that way has no params to feed and never reaches here.
            _ => continue,
        };
        body.replace_instruction_mnemonic(term_id, new);
    }
    for param in removed {
        body.remove_block_param(param);
    }
}

/// Return `args` with the entries at `drop` positions removed.
fn filter_kept(args: &[LocalValueId], drop: &HashSet<usize>) -> Vec<LocalValueId> {
    args.iter()
        .enumerate()
        .filter(|(i, _)| !drop.contains(i))
        .map(|(_, &a)| a)
        .collect()
}

/// Concrete pass twin of [`remove_params_from_block`] over a checked-out function
/// (`&mut PassBacking`), for the pass-path caller strlen. Mirrors
/// [`remove_params_from_block_generic`]; the module path keeps the generic.
pub(crate) fn remove_params_from_block_c<'str>(
    host: &mut PassBacking<'_, 'str>,
    block: BlockId,
    dead_indices: &HashSet<usize>,
) {
    let params = host.read_host().block(block).params.clone();
    let mut kept = Vec::with_capacity(params.len());
    let mut removed = Vec::new();
    for (i, &local) in params.iter().enumerate() {
        let p = BlockParamId::new(block.func, local);
        if dead_indices.contains(&i) {
            removed.push(p);
        } else {
            host.block_param_mut(p).index = kept.len();
            kept.push(local);
        }
    }
    host.block_mut(block).params = kept;

    let preds: HashSet<BlockId> = host
        .block_ref(block)
        .predecessors()
        .map(|(_, b)| b)
        .collect();

    for pred in preds {
        let Some(term_local) = host.read_host().block(pred).instructions.last().copied() else {
            continue;
        };
        let term_id = qcode::value::insn::InstructionId::new(pred.func, term_local);
        let new = match host.read_host().instruction(term_id).mnemonic().clone() {
            Mnemonic::Branch(b) if BlockId::new(pred.func, b.target) == block => {
                Mnemonic::Branch(Branch {
                    target: b.target,
                    args: filter_kept(&b.args, dead_indices),
                })
            }
            Mnemonic::CBranch(c) => Mnemonic::CBranch(CBranch {
                condition: c.condition,
                success_block: c.success_block,
                success_args: if BlockId::new(pred.func, c.success_block) == block {
                    filter_kept(&c.success_args, dead_indices)
                } else {
                    c.success_args
                },
                failure_block: c.failure_block,
                failure_args: if BlockId::new(pred.func, c.failure_block) == block {
                    filter_kept(&c.failure_args, dead_indices)
                } else {
                    c.failure_args
                },
            }),
            _ => continue,
        };
        host.replace_instruction_mnemonic(term_id, new);
    }
    for param in removed {
        host.remove_block_param(param);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::{BasicBlock, FunctionRef};
    use qcode_macro::qcode;

    /// All block ids of `fun`, used to drive the standalone sweep in tests.
    fn block_ids(ctx: &Context, fun: qcode::value::FunctionId) -> Vec<BlockId> {
        FunctionRef::from_id(ctx, fun)
            .blocks()
            .map(|b| b.id)
            .collect()
    }

    fn root_of(ctx: &Context, fun: qcode::value::FunctionId) -> Option<BlockId> {
        FunctionRef::from_id(ctx, fun).root().map(|b| b.id)
    }

    fn num_params(ctx: &Context, block: BlockId) -> usize {
        BasicBlock::from_id(ctx, block).num_params()
    }

    fn param_names(ctx: &Context, block: BlockId) -> Vec<String> {
        BasicBlock::from_id(ctx, block)
            .params()
            .map(|p| p.name().unwrap_or("?").to_string())
            .collect()
    }

    /// A loop-invariant param threaded `%c` off-loop and forwarded to itself on
    /// the back-edge — even though it is *read* inside the loop — collapses to
    /// `%c` (mirrors `<40b3f0>`'s `@ESP`). The genuinely-varying induction param
    /// fed `%next` on the back-edge is a real merge and stays.
    #[test]
    fn loop_invariant_param_collapses_to_its_value() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                %c = load(ram:8, 0x2000);
                goto <header @inv=%c @ind=0x0>;
            <header @inv:i64 @ind:i64>
                %cond = @ind < 0x3;
                %next = @ind + @inv;
                if %cond goto <header @inv=@inv @ind=%next> else goto <exit>;
            <exit>
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(remove_dead_block_args(&mut ctx, &blocks, root));

        // @inv is redundant (only {%c, @inv}) and removed; @ind merges {0x0,%next}
        // and stays. Its uses of @inv are rewritten to %c.
        assert_eq!(param_names(&ctx, header), vec!["ind"]);
        let next = BasicBlock::from_id(&ctx, header)
            .iter()
            .find(|i| i.as_statement().to_string().contains('+'))
            .unwrap()
            .as_statement()
            .to_string();
        assert_eq!(
            next, "i64 %next = i64 @ind + i64 %c;",
            "use of @inv rewritten to %c"
        );
    }

    /// The user's motivating case: two params that only feed each other around a
    /// loop. `@y` collapses to `@x` first, turning `@x`'s back-edge arg into
    /// `@x=@x`, after which `@x` collapses to its off-loop value `0x0`. Both go.
    #[test]
    fn mutually_recursive_params_both_removed() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <header @x=0x0>;
            <header @x:i64>
                %i = load(ram:1, 0x1000);
                if %i goto <latch @y=@x> else goto <exit>;
            <latch @y:i64>
                goto <header @x=@y>;
            <exit>
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(remove_dead_block_args(&mut ctx, &blocks, root));

        assert_eq!(num_params(&ctx, header), 0, "@x should be gone");
        assert_eq!(num_params(&ctx, latch), 0, "@y should be gone");
        let header_term = BasicBlock::from_id(&ctx, header)
            .iter()
            .last()
            .unwrap()
            .as_statement()
            .to_string();
        assert_eq!(header_term, "if i8 %i goto <latch> else goto <exit>;");
    }

    /// A real merge of two distinct values is not redundant and is kept.
    #[test]
    fn genuine_merge_is_kept() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                %c = load(ram:1, 0x1000);
                if %c goto <join @m=0x1> else goto <other>;
            <other>
                goto <join @m=0x2>;
            <join @m:i64>
                store(ram:8, 0x4000 <- i64 @m);
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(!remove_dead_block_args(&mut ctx, &blocks, root));
        assert_eq!(num_params(&ctx, join), 1);
    }

    /// Only the redundant column is removed from a two-predecessor join; the
    /// genuinely-merged column and both predecessors' live args survive,
    /// re-indexed.
    #[test]
    fn one_redundant_column_among_two_preds() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                %c = load(ram:1, 0x1000);
                if %c goto <join @same=0x7 @merge=0x2> else goto <other>;
            <other>
                goto <join @same=0x7 @merge=0x4>;
            <join @same:i64 @merge:i64>
                store(ram:8, 0x4000 <- i64 @merge);
                store(ram:8, 0x4008 <- i64 @same);
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(remove_dead_block_args(&mut ctx, &blocks, root));

        // @same is 0x7 on both edges → collapses; @merge differs → kept.
        assert_eq!(param_names(&ctx, join), vec!["merge"]);
        let other_term = BasicBlock::from_id(&ctx, other)
            .iter()
            .last()
            .unwrap()
            .as_statement()
            .to_string();
        assert_eq!(other_term, "goto <join @merge=i64 0x4>;");
    }

    /// Congruence generalization: a loop-invariant param fed a *recomputed* (but
    /// congruent) expression on the back-edge — `%a + %b` off-loop and `%b + %a`
    /// on the latch — is not a syntactic match, but value numbering proves the
    /// two equal, so the param collapses to the off-loop (dominating) value.
    #[test]
    fn congruent_recomputed_loop_invariant_collapses() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            fn test:
            <entry>
                %a = load(A:8, &A);
                %b = load(B:8, &B);
                %off = %a + %b;
                goto <header @inv=%off>;
            <header @inv:i64>
                %i = load(ram:1, 0x1000);
                %re = %b + %a;
                store(ram:8, 0x4000 <- i64 @inv);
                if %i goto <header @inv=%re> else goto <exit>;
            <exit>
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(remove_dead_block_args(&mut ctx, &blocks, root));

        assert_eq!(num_params(&ctx, header), 0, "@inv collapses to %off");
        let store = BasicBlock::from_id(&ctx, header)
            .iter()
            .map(|i| i.as_statement().to_string())
            .find(|s| s.contains("0x4000"))
            .expect("store survives");
        assert!(
            store.contains("%off"),
            "use of @inv rewritten to %off: {store}"
        );
    }

    /// Soundness boundary: the back-edge re-*loads* the same address rather than
    /// reusing the off-loop load. Two loads are not congruent (an intervening
    /// store could differ), so the param is a genuine merge and must be KEPT.
    #[test]
    fn reloaded_value_on_backedge_is_not_collapsed() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            fn test:
            <entry>
                %c1 = load(A:8, &A);
                goto <header @inv=%c1>;
            <header @inv:i64>
                %i = load(ram:1, 0x1000);
                %c2 = load(A:8, &A);
                store(ram:8, 0x4000 <- i64 @inv);
                if %i goto <header @inv=%c2> else goto <exit>;
            <exit>
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(!remove_dead_block_args(&mut ctx, &blocks, root));
        assert_eq!(
            num_params(&ctx, header),
            1,
            "reloaded back-edge value is a real merge"
        );
    }

    /// Memory edge case: the back-edge value is *arithmetic over a reloaded*
    /// address (`%c2 + 1`) while the off-loop value is `%c1 + 1`. The two adds are
    /// affine-identical but built on distinct loads, so they are not congruent and
    /// the param is a genuine merge — KEPT. (A reload could see a store from the
    /// loop body, so collapsing would be unsound.)
    #[test]
    fn arithmetic_over_reloaded_value_is_kept() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            fn test:
            <entry>
                %c1 = load(A:8, &A);
                %off = %c1 + 0x1;
                goto <header @inv=%off>;
            <header @inv:i64>
                %i = load(ram:1, 0x1000);
                %c2 = load(A:8, &A);
                %re = %c2 + 0x1;
                store(ram:8, 0x4000 <- i64 @inv);
                if %i goto <header @inv=%re> else goto <exit>;
            <exit>
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(!remove_dead_block_args(&mut ctx, &blocks, root));
        assert_eq!(
            num_params(&ctx, header),
            1,
            "arithmetic over a reload is a real merge"
        );
    }

    /// Dual: an intervening store does NOT block collapse when the carried value
    /// is SSA-stable. The back-edge recomputes `%c + 1` from the *same* dominating
    /// load `%c`; a store to a different address sits in the loop, but `%c` is one
    /// SSA value, so `@inv` is loop-invariant and collapses to the off-loop value.
    #[test]
    fn intervening_store_does_not_block_ssa_stable_collapse() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            fn test:
            <entry>
                %c = load(A:8, &A);
                %off = %c + 0x1;
                goto <header @inv=%off>;
            <header @inv:i64>
                %i = load(ram:1, 0x1000);
                store(ram:8, 0x4000 <- i64 @inv);
                %re = %c + 0x1;
                if %i goto <header @inv=%re> else goto <exit>;
            <exit>
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(remove_dead_block_args(&mut ctx, &blocks, root));
        assert_eq!(
            num_params(&ctx, header),
            0,
            "SSA-stable @inv collapses despite the store"
        );
    }

    /// Root (entry) params are the function interface and must not be touched
    /// here even when a back-edge makes them look redundant.
    #[test]
    fn root_params_left_alone() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <root @x:i64>
                %c = load(ram:1, 0x1000);
                if %c goto <root @x=@x> else goto <exit>;
            <exit>
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(!remove_dead_block_args(&mut ctx, &blocks, root));
        assert_eq!(num_params(&ctx, root.unwrap()), 1);
    }

    /// A multi-block cycle. Block `b` has a single predecessor, so *both* its
    /// params are trivially redundant and forward into `a` (jump-threading). In
    /// `a`, the `@inv` column only ever carries `%seed` (off-loop) or itself and
    /// collapses to `%seed`, while `@merg` merges two distinct values and stays.
    #[test]
    fn chain_collapses_redundant_column_keeps_merged_one() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                %seed = load(ram:8, 0x2000);
                goto <a @inva=%seed @merga=0x0>;
            <a @inva:i64 @merga:i64>
                goto <b @invb=@inva @mergb=@merga>;
            <b @invb:i64 @mergb:i64>
                store(ram:8, 0x5000 <- i64 @invb);
                %n = @mergb + 0x1;
                %c = load(ram:1, 0x1000);
                if %c goto <a @inva=@invb @merga=%n> else goto <exit>;
            <exit>
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(remove_dead_block_args(&mut ctx, &blocks, root));

        // b's params both forward into a (single predecessor). In a, @inva
        // collapses to %seed and @merga merges {0x0, %n} → kept.
        assert_eq!(param_names(&ctx, a), vec!["merga"]);
        assert!(
            param_names(&ctx, b).is_empty(),
            "b's params all forward into a"
        );
        // The store now reads %seed directly (via @invb → @inva → %seed).
        let store = BasicBlock::from_id(&ctx, b)
            .iter()
            .map(|i| i.as_statement().to_string())
            .find(|s| s.contains("0x5000"))
            .expect("store on 0x5000 survives in b");
        assert!(store.contains("%seed"), "store rewritten to %seed: {store}");
    }

    // ---- liveness-based dead-param removal (`remove_dead_block_params`) ----

    /// The motivating case (`<40b3f0>`): a loop carries an accumulator `@acc` that
    /// reads itself (live), an induction `@i` used in the exit test (live), and a
    /// junk register `@junk` recomputed every iteration *from the index* — never
    /// read — and threaded around the back-edge. Each iteration binds `@junk` a
    /// distinct value, so it is not redundant; it is a back-edge arg, so naive DCE
    /// sees it live. Liveness removes it and keeps the two real carriers.
    #[test]
    fn dead_recomputed_register_removed_keeps_carriers() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <hdr @acc=0x0 @i=0x0 @junk=0x0>;
            <hdr @acc:i64 @i:i64 @junk:i64>
                %na = @acc + @i;
                store(ram:8, 0x4000 <- i64 %na);
                %ni = @i + 0x1;
                %nj = @i & 0x1;
                %c = %ni < 0x270;
                if %c goto <hdr @acc=%na @i=%ni @junk=%nj> else goto <exit>;
            <exit>
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(remove_dead_block_params(&mut ctx, &blocks, root));
        assert_eq!(
            param_names(&ctx, hdr),
            vec!["acc", "i"],
            "@junk is dead; @acc (self-read) and @i (exit test) stay"
        );
        // The back-edge no longer carries a @junk column.
        let term = BasicBlock::from_id(&ctx, hdr)
            .iter()
            .last()
            .unwrap()
            .as_statement()
            .to_string();
        assert!(!term.contains("@junk"), "back-edge dropped @junk: {term}");
    }

    /// A cycle of two params that only forward into each other — neither read by
    /// any instruction, condition, or return — is a dead SCC and both go.
    #[test]
    fn dead_param_cycle_both_removed() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <hdr @a=0x0 @b=0x0>;
            <hdr @a:i64 @b:i64>
                %i = load(ram:1, 0x1000);
                if %i goto <latch @a2=@b @b2=@a> else goto <exit>;
            <latch @a2:i64 @b2:i64>
                goto <hdr @a=@a2 @b=@b2>;
            <exit>
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(remove_dead_block_params(&mut ctx, &blocks, root));
        assert_eq!(num_params(&ctx, hdr), 0, "@a, @b dead SCC");
        assert_eq!(num_params(&ctx, latch), 0, "@a2, @b2 dead SCC");
    }

    /// Soundness: a param that is *not* directly used but forwards into a param
    /// that IS read must be KEPT — its value is observed indirectly. `@x` is never
    /// touched in `hdr`, yet it feeds `@y`, which is stored, so `@x` is live.
    #[test]
    fn param_forwarding_into_live_param_is_kept() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                %s = load(ram:8, 0x2000);
                goto <hdr @x=%s>;
            <hdr @x:i64>
                %i = load(ram:1, 0x1000);
                if %i goto <usr @y=@x> else goto <exit>;
            <usr @y:i64>
                store(ram:8, 0x4000 <- i64 @y);
                goto <exit>;
            <exit>
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(!remove_dead_block_params(&mut ctx, &blocks, root));
        assert_eq!(num_params(&ctx, hdr), 1, "@x feeds the stored @y → live");
        assert_eq!(num_params(&ctx, usr), 1, "@y is read → live");
    }

    /// A param read only by the branch *condition* is live and kept.
    #[test]
    fn param_used_in_condition_is_kept() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <hdr @cnt=0x0>;
            <hdr @cnt:i64>
                %n = @cnt + 0x1;
                %c = @cnt < 0xa;
                if %c goto <hdr @cnt=%n> else goto <exit>;
            <exit>
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(!remove_dead_block_params(&mut ctx, &blocks, root));
        assert_eq!(num_params(&ctx, hdr), 1, "@cnt drives the condition → live");
    }

    /// A param read only by the `return` slot is live and kept.
    #[test]
    fn param_used_by_return_is_kept() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                %p = load(ram:8, 0x2000);
                goto <ret @r=%p>;
            <ret @r:i64>
                return at @r;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(!remove_dead_block_params(&mut ctx, &blocks, root));
        assert_eq!(num_params(&ctx, ret), 1, "@r is the return pointer → live");
    }

    /// Mixed column: one dead and one live param on the same loop header. Only the
    /// dead column is dropped; the live induction column and its back-edge arg
    /// survive, re-indexed.
    #[test]
    fn dead_column_dropped_live_column_survives() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                goto <hdr @i=0x0 @dead=0x0>;
            <hdr @i:i64 @dead:i64>
                %ni = @i + 0x1;
                %nd = @i * 0x3;
                %c = @i < 0x5;
                if %c goto <hdr @i=%ni @dead=%nd> else goto <exit>;
            <exit>
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(remove_dead_block_params(&mut ctx, &blocks, root));
        assert_eq!(param_names(&ctx, hdr), vec!["i"], "@dead removed, @i kept");
        let term = BasicBlock::from_id(&ctx, hdr)
            .iter()
            .last()
            .unwrap()
            .as_statement()
            .to_string();
        assert_eq!(term, "if bool %c goto <hdr @i=i64 %ni> else goto <exit>;");
    }

    /// Root (entry) params are the function interface and are never removed even
    /// when no instruction reads them.
    #[test]
    fn root_params_not_removed_by_liveness() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <root @x:i64 @y:i64>
                store(ram:8, 0x4000 <- i64 @x);
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        // @y is unused but it is a root param → kept.
        assert!(!remove_dead_block_params(&mut ctx, &blocks, root));
        assert_eq!(num_params(&ctx, root.unwrap()), 2);
    }

    /// A dead param feeds, across two blocks, into another dead param — a forward
    /// chain with no read anywhere. The whole chain is removed.
    #[test]
    fn dead_forward_chain_removed() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn test:
            <entry>
                %i = load(ram:1, 0x1000);
                if %i goto <a @za=0x1> else goto <exit>;
            <a @za:i64>
                goto <b @zb=@za>;
            <b @zb:i64>
                store(ram:8, 0x4000 <- 0x9);
                return at 0x0;
            <exit>
                return at 0x0;
            "
        );

        let blocks = block_ids(&ctx, test);
        let root = root_of(&ctx, test);
        assert!(remove_dead_block_params(&mut ctx, &blocks, root));
        assert_eq!(num_params(&ctx, a), 0, "@za never read");
        assert_eq!(num_params(&ctx, b), 0, "@zb never read");
    }
}
