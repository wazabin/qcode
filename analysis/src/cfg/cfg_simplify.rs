use qcode::context::Context;
use qcode::value::{
    BlockId, BlockParamId, FunctionId, ValueId,
    insn::{Branch, Mnemonic},
};

use crate::{ContextView, FunctionBody, FunctionPass, Outcome};

// TODO(5b-ii): Public functions below are thin wrappers marked for future migration

#[derive(Default)]
pub struct SimplifyCfg;

impl FunctionPass for SimplifyCfg {
    const NAME: &'static str = "simplify_cfg";

    fn description(&self) -> &'static str {
        "Merge straight-line blocks, drop empty forwarding blocks, fold degenerate branches"
    }

    fn run<'str>(
        &self,
        body: &mut FunctionBody<'_, 'str>,
        cx: ContextView<'_, 'str>,
    ) -> Result<Outcome<'str>, String> {
        let fid = body.id();
        Ok(Outcome::changed(simplify_cfg_concrete(body, cx, fid)))
    }
}

crate::register_function_pass!(SimplifyCfg);

/// Simplifies the CFG of `function_id` to a fixpoint. Each round applies, in
/// priority order, the first transform that fits any block:
///
/// 1. **Fold degenerate conditional branches** ([`try_fold_cbranch`]): a
///    `CBranch` whose two arms target the same block with the same arguments is
///    rewritten to an unconditional `Branch` (the condition becomes irrelevant).
/// 2. **Bypass empty forwarding blocks** ([`try_bypass_empty_block`]): a block
///    whose sole instruction is an unconditional `Branch` is spliced out by
///    redirecting every predecessor straight to its target.
/// 3. **Merge straight-line chains** ([`try_merge_block`]): if A has exactly one
///    successor B, B has exactly one predecessor A, and A ends with an
///    unconditional `Branch { target: B }`, A absorbs B.
///
/// Each round first drops every block unreachable from the entry
/// ([`prune_unreachable`]) — an unconditional cleanup with no proof obligation,
/// e.g. the residual loop left behind once `dce` bypasses a dead loop.
///
/// The pass repeats until a full scan applies no transform.
pub fn simplify_cfg_concrete<'a, 'str>(
    body: &'a mut FunctionBody<'_, 'str>,
    cx: ContextView<'a, 'str>,
    function_id: FunctionId,
) -> bool {
    let mut changed = false;

    loop {
        let blocks = body.function_ref(cx, function_id).block_ids();
        let mut progress = prune_unreachable_concrete(body, cx, function_id);

        for block_id in blocks {
            if try_fold_cbranch_concrete(body, cx, block_id)
                || try_bypass_empty_block_concrete(body, cx, function_id, block_id)
                || try_merge_block_concrete(body, cx, function_id, block_id)
            {
                progress = true;
                break;
            }
        }

        if progress {
            changed = true;
        } else {
            break;
        }
    }

    changed
}

/// Generic wrapper for backwards compatibility; see concrete [`simplify_cfg_concrete`].
/// TODO(5b-ii): Remove after callers migrate to FunctionBody/ContextView.
#[allow(dead_code)]
pub fn simplify_cfg<'str>(host: &mut Context<'str>, function_id: FunctionId) -> bool {
    let mut changed = false;

    loop {
        let blocks = host.function_ref(function_id).block_ids();
        let mut progress = prune_unreachable_generic(host, function_id);

        for block_id in blocks {
            if try_fold_cbranch_generic(host, block_id)
                || try_bypass_empty_block_generic(host, function_id, block_id)
                || try_merge_block_generic(host, function_id, block_id)
            {
                progress = true;
                break;
            }
        }

        if progress {
            changed = true;
        } else {
            break;
        }
    }

    changed
}

/// Generic wrapper for [`prune_unreachable`]; see that function.
#[allow(dead_code)]
fn prune_unreachable_generic<'str>(host: &mut Context<'str>, function_id: FunctionId) -> bool {
    let Some(root) = host.function_ref(function_id).root().map(|b| b.id) else {
        return false;
    };

    // Mark reachable blocks by CFG walk from the entry.
    let mut reachable = rustc_hash::FxHashSet::default();
    let mut stack = vec![root];
    while let Some(b) = stack.pop() {
        if !reachable.insert(b) {
            continue;
        }
        let succs: Vec<BlockId> = host.block_ref(b).successors().map(|(_, s)| s).collect();
        stack.extend(succs);
    }

    let dead: Vec<BlockId> = host
        .function_ref(function_id)
        .block_ids()
        .into_iter()
        .filter(|b| !reachable.contains(b))
        .collect();
    if dead.is_empty() {
        return false;
    }

    for block in dead {
        // `delete_block` unwinds CFG edges, removes the block's instructions
        // (clearing their use records) and detaches its params — the full cleanup.
        host.delete_block(block, function_id);
    }
    true
}

/// Generic wrapper for [`merge_candidate`]; see that function.
#[allow(dead_code)]
fn merge_candidate_generic<'str>(
    host: &mut Context<'str>,
    a_id: BlockId,
) -> Option<(qcode::value::block::EdgeId, BlockId)> {
    // Collect at most 2 successors to check the "exactly one" condition.
    // Collecting eagerly releases the immutable borrow before any mutation.
    let a_succs: Vec<_> = host.block_ref(a_id).successors().take(2).collect();

    let &(edge_ab, b_id) = a_succs.first()?;
    if a_succs.len() != 1 || b_id == a_id {
        return None; // more than one successor, or a self-loop
    }
    if host.block_ref(b_id).predecessors().count() != 1 {
        return None;
    }

    // A's terminal must be an unconditional Branch to B.
    let a_terminal = host.block_ref(a_id).instruction_ids().last().copied();
    let is_branch_to_b = a_terminal
        .map(|id| {
            matches!(
                host.insn_ref(id).mnemonic(),
                Mnemonic::Branch(b) if b.target == b_id
            )
        })
        .unwrap_or(false);
    if !is_branch_to_b {
        return None;
    }

    // Merging rewrites B's params to the branch's args, so the branch must
    // supply one arg per param. A mismatch means a malformed edge — e.g. a
    // CRT stub's tail `jmp` into another routine that mem2reg gave a param,
    // lifted as an intra-function `goto` carrying no args. Leave such edges
    // unmerged rather than absorbing an unsatisfiable param.
    let b_params = host.block_ref(b_id).num_params();
    let branch_args = a_terminal
        .and_then(|id| match host.insn_ref(id).mnemonic() {
            Mnemonic::Branch(b) => Some(b.args.len()),
            _ => None,
        })
        .unwrap_or(0);
    if b_params != branch_args {
        return None;
    }

    Some((edge_ab, b_id))
}

/// Generic wrapper for [`try_merge_block`]; see that function.
#[allow(dead_code)]
fn try_merge_block_generic<'str>(
    host: &mut Context<'str>,
    function_id: FunctionId,
    a_id: BlockId,
) -> bool {
    let Some((edge_ab, b_id)) = merge_candidate_generic(host, a_id) else {
        return false;
    };
    // Intra-function only (see the note above): B must be both *stored in* and
    // *owned by* this function. A block stored in this function's arena but
    // reattributed — owned/rostered by another function (common in real binaries:
    // shared CRT stubs, thunks) — must not be absorbed: `absorb_block` →
    // `unroster_block` mutates the *owner*'s roster, which a checked-out pass may
    // not do. A cross-function successor (thunk/tail-call) is likewise left as-is.
    if b_id.func != function_id
        || host.function(function_id).block(b_id).parent != Some(function_id)
    {
        return false;
    }
    host.absorb_block(a_id, b_id, edge_ab, function_id);
    true
}

/// Generic wrapper for [`try_fold_cbranch`]; see that function.
#[allow(dead_code)]
fn try_fold_cbranch_generic<'str>(host: &mut Context<'str>, block_id: BlockId) -> bool {
    let Some(term_id) = host.block_ref(block_id).instruction_ids().last().copied() else {
        return false;
    };
    let (target, args) = {
        let Mnemonic::CBranch(cb) = host.insn_ref(term_id).mnemonic() else {
            return false;
        };
        if cb.success_block != cb.failure_block || cb.success_args != cb.failure_args {
            return false;
        }
        (cb.success_block, cb.success_args.clone())
    };
    host.replace_instruction_mnemonic(term_id, Mnemonic::Branch(Branch { target, args }));

    // Collapse the two parallel `block -> target` edges into one: keep the
    // first, drop the second.
    let dup_edge = host
        .block_ref(block_id)
        .successors()
        .filter(|&(_, to)| to == target)
        .map(|(e, _)| e)
        .nth(1);
    if let Some(dup_edge) = dup_edge {
        host.remove_cfg_edge(block_id.func, dup_edge);
    }

    true
}

/// Generic wrapper for [`try_bypass_empty_block`]; see that function.
#[allow(dead_code)]
fn try_bypass_empty_block_generic<'str>(
    host: &mut Context<'str>,
    function_id: FunctionId,
    b_id: BlockId,
) -> bool {
    // The entry block dominates everything; deleting it would orphan the body.
    if host.function(function_id).root_id() == Some(b_id) {
        return false;
    }

    // B must hold exactly one instruction, an unconditional branch.
    let (term_id, target, b_args) = {
        let b = host.function(function_id).block(b_id);
        if b.instructions.len() != 1 {
            return false;
        }
        let term_id = b.instructions[0];
        match host.function(function_id).insn(term_id).mnemonic() {
            Mnemonic::Branch(br) => (term_id, br.target, br.args.clone()),
            _ => return false,
        }
    };
    if target == b_id {
        return false; // bypassing `goto self` is meaningless and unsound
    }
    // Intra-function only: bypassing a block that forwards into another function
    // (a thunk) is a cross-function edit — left to the module pass.
    if target.func != function_id {
        return false;
    }

    let params: Vec<BlockParamId> = host.function(function_id).block(b_id).params.clone();

    // B's params must flow nowhere but B's own terminator. In valid SSA a block
    // param is only visible inside dominated blocks via forwarded args, so this
    // normally holds; bail if it doesn't rather than risk a dangling use.
    for &p in &params {
        if host
            .function(function_id)
            .users_of(ValueId::BlockParam(p))
            .iter()
            .any(|&u| u != term_id)
        {
            return false;
        }
    }

    // Distinct predecessors of B.
    let preds: Vec<BlockId> = {
        let mut seen = rustc_hash::FxHashSet::default();
        host.block_ref(b_id)
            .predecessors()
            .map(|(_, p)| p)
            .filter(|&p| seen.insert(p))
            .collect()
    };

    // A forwarding block with no predecessors is unreachable dead code. Splicing
    // is about removing a block *on a path*; deleting unreachable blocks is a
    // separate concern, so leave it to a dedicated pass.
    if preds.is_empty() {
        return false;
    }

    // Intra-function only: a predecessor in another function (a tail-call into B)
    // cannot be rewritten by a checked-out pass. Leave such blocks untouched.
    if preds.iter().any(|p| p.func != function_id) {
        return false;
    }

    // Pre-validate every predecessor before mutating anything: each must reach B
    // through a rewritable terminator that names B with a matching arg count on
    // each arm that targets B.
    for &p in &preds {
        let Some(p_term) = host
            .function(function_id)
            .block(p)
            .instructions
            .last()
            .copied()
        else {
            return false;
        };
        match host.function(function_id).insn(p_term).mnemonic() {
            Mnemonic::Branch(br) => {
                if br.target != b_id || br.args.len() != params.len() {
                    return false;
                }
            }
            Mnemonic::CBranch(cb) => {
                let mut names_b = false;
                if cb.success_block == b_id {
                    if cb.success_args.len() != params.len() {
                        return false;
                    }
                    names_b = true;
                }
                if cb.failure_block == b_id {
                    if cb.failure_args.len() != params.len() {
                        return false;
                    }
                    names_b = true;
                }
                if !names_b {
                    return false;
                }
            }
            _ => return false,
        }
    }

    // Rewrite each predecessor to branch straight to `target`, substituting B's
    // params with the arguments that predecessor supplied.
    for &p in &preds {
        let p_term = host
            .function(function_id)
            .block(p)
            .instructions
            .last()
            .copied()
            .unwrap();
        let new_mnemonic = match host.function(function_id).insn(p_term).mnemonic().clone() {
            Mnemonic::Branch(br) => Mnemonic::Branch(Branch {
                target,
                args: substitute(&b_args, &params, &br.args),
            }),
            Mnemonic::CBranch(mut cb) => {
                if cb.success_block == b_id {
                    cb.success_args = substitute(&b_args, &params, &cb.success_args);
                    cb.success_block = target;
                }
                if cb.failure_block == b_id {
                    cb.failure_args = substitute(&b_args, &params, &cb.failure_args);
                    cb.failure_block = target;
                }
                Mnemonic::CBranch(cb)
            }
            _ => unreachable!("predecessor terminator validated above"),
        };

        // Rehome the `p -> b` edges to `p -> target`, preserving multiplicity
        // (a CBranch with both arms on B contributes two edges).
        let redirect: Vec<_> = host
            .block_ref(p)
            .successors()
            .filter(|&(_, to)| to == b_id)
            .map(|(e, _)| e)
            .collect();
        host.replace_instruction_mnemonic(p_term, new_mnemonic);
        for &edge in &redirect {
            host.remove_cfg_edge(p.func, edge);
        }
        for _ in &redirect {
            host.add_cfg_edge(p, target);
        }
    }

    // B has no predecessors left; delete it (drops its branch and `b -> target`).
    host.delete_block(b_id, function_id);
    true
}

/// Deletes every block of `function_id` not reachable from the entry, along
/// with its instructions (which detaches CFG edges and clears use records) and
/// params. Removing the terminator of an unreachable block only drops edges into
/// *other* unreachable blocks, so a single pass suffices; returns `true` if any
/// block was removed.
///
/// Sound with no side conditions: a block with no path from the entry executes
/// on no run, so nothing it computes or branches to is observable. This is what
/// lets a dead loop, once `dce` reroutes its preheader past it, disappear.
fn prune_unreachable_concrete<'a, 'str>(
    body: &'a mut FunctionBody<'_, 'str>,
    cx: ContextView<'a, 'str>,
    function_id: FunctionId,
) -> bool {
    let Some(root) = body.function_ref(cx, function_id).root().map(|b| b.id) else {
        return false;
    };

    // Mark reachable blocks by CFG walk from the entry.
    let mut reachable = rustc_hash::FxHashSet::default();
    let mut stack = vec![root];
    while let Some(b) = stack.pop() {
        if !reachable.insert(b) {
            continue;
        }
        let succs: Vec<BlockId> = body.block_ref(cx, b).successors().map(|(_, s)| s).collect();
        stack.extend(succs);
    }

    let dead: Vec<BlockId> = body
        .function_ref(cx, function_id)
        .block_ids()
        .into_iter()
        .filter(|b| !reachable.contains(b))
        .collect();
    if dead.is_empty() {
        return false;
    }

    for block in dead {
        // `delete_block` unwinds CFG edges, removes the block's instructions
        // (clearing their use records) and detaches its params — the full cleanup.
        body.delete_block(cx, block);
    }
    true
}

/// If `a_id` is the tail of a straight-line chain — exactly one successor `B`,
/// `B` has exactly one predecessor (`a_id`), `a_id` ends with an unconditional
/// `Branch { target: B }`, and the branch supplies one arg per B-param — returns
/// the `(edge, B)` to absorb. Otherwise `None`.
///
/// This inspects only the CFG structure and does *not* check whether `B` lives in
/// the same function as `a_id`; [`try_merge_block`] applies that gate (a
/// checked-out pass merges only within its own function).
fn merge_candidate_concrete<'a, 'str>(
    body: &'a FunctionBody<'_, 'str>,
    cx: ContextView<'a, 'str>,
    a_id: BlockId,
) -> Option<(qcode::value::block::EdgeId, BlockId)> {
    // Collect at most 2 successors to check the "exactly one" condition.
    // Collecting eagerly releases the immutable borrow before any mutation.
    let a_succs: Vec<_> = body.block_ref(cx, a_id).successors().take(2).collect();

    let &(edge_ab, b_id) = a_succs.first()?;
    if a_succs.len() != 1 || b_id == a_id {
        return None; // more than one successor, or a self-loop
    }
    if body.block_ref(cx, b_id).predecessors().count() != 1 {
        return None;
    }

    // A's terminal must be an unconditional Branch to B.
    let a_terminal = body.block_ref(cx, a_id).instruction_ids().last().copied();
    let is_branch_to_b = a_terminal
        .map(|id| {
            matches!(
                body.insn_ref(cx, id).mnemonic(),
                Mnemonic::Branch(b) if b.target == b_id
            )
        })
        .unwrap_or(false);
    if !is_branch_to_b {
        return None;
    }

    // Merging rewrites B's params to the branch's args, so the branch must
    // supply one arg per param. A mismatch means a malformed edge — e.g. a
    // CRT stub's tail `jmp` into another routine that mem2reg gave a param,
    // lifted as an intra-function `goto` carrying no args. Leave such edges
    // unmerged rather than absorbing an unsatisfiable param.
    let b_params = body.block_ref(cx, b_id).num_params();
    let branch_args = a_terminal
        .and_then(|id| match body.insn_ref(cx, id).mnemonic() {
            Mnemonic::Branch(b) => Some(b.args.len()),
            _ => None,
        })
        .unwrap_or(0);
    if b_params != branch_args {
        return None;
    }

    Some((edge_ab, b_id))
}

/// Merges `a_id` with its unique successor when the straight-line conditions hold
/// **and the successor is in the same function**. The cross-function (thunk /
/// tail-call) case is intentionally *not* handled: a checked-out function pass may
/// not mutate another function, and absorbing another function's block would drag
/// its instructions (which stay in that function's arena) into this one. Returns
/// `true` if a merge happened.
fn try_merge_block_concrete<'a, 'str>(
    body: &'a mut FunctionBody<'_, 'str>,
    cx: ContextView<'a, 'str>,
    function_id: FunctionId,
    a_id: BlockId,
) -> bool {
    let Some((edge_ab, b_id)) = merge_candidate_concrete(body, cx, a_id) else {
        return false;
    };
    // Intra-function only (see the note above): B must be both *stored in* and
    // *owned by* this function. A block stored in this function's arena but
    // reattributed — owned/rostered by another function (common in real binaries:
    // shared CRT stubs, thunks) — must not be absorbed: `absorb_block` →
    // `unroster_block` mutates the *owner*'s roster, which a checked-out pass may
    // not do. A cross-function successor (thunk/tail-call) is likewise left as-is.
    if b_id.func != function_id || body.read_host(cx).block(b_id).parent != Some(function_id) {
        return false;
    }
    body.absorb_block(cx, a_id, b_id, edge_ab);
    true
}

/// Rewrites a conditional branch whose two arms are indistinguishable — same
/// target block *and* same per-arm arguments — into an unconditional `Branch`,
/// discarding the now-irrelevant condition. Returns `true` if it fired.
///
/// The `CBranch` contributed two parallel CFG edges to the shared target; one
/// is dropped so the edge multiplicity matches the new single-successor branch.
fn try_fold_cbranch_concrete<'a, 'str>(
    body: &'a mut FunctionBody<'_, 'str>,
    cx: ContextView<'a, 'str>,
    block_id: BlockId,
) -> bool {
    let Some(term_id) = body
        .block_ref(cx, block_id)
        .instruction_ids()
        .last()
        .copied()
    else {
        return false;
    };
    let (target, args) = {
        let Mnemonic::CBranch(cb) = body.insn_ref(cx, term_id).mnemonic() else {
            return false;
        };
        if cb.success_block != cb.failure_block || cb.success_args != cb.failure_args {
            return false;
        }
        (cb.success_block, cb.success_args.clone())
    };
    body.replace_instruction_mnemonic(cx, term_id, Mnemonic::Branch(Branch { target, args }));

    // Collapse the two parallel `block -> target` edges into one: keep the
    // first, drop the second.
    let dup_edge = body
        .block_ref(cx, block_id)
        .successors()
        .filter(|&(_, to)| to == target)
        .map(|(e, _)| e)
        .nth(1);
    if let Some(dup_edge) = dup_edge {
        body.remove_cfg_edge(cx, dup_edge);
    }

    true
}

/// Splices out an *empty forwarding block* `b_id` — one whose only instruction
/// is an unconditional `Branch { target, .. }` — by redirecting each predecessor
/// straight to `target`, then deleting `b_id`. Returns `true` if it fired.
///
/// Block arguments are threaded per predecessor: if B declares params and its
/// branch forwards them (`<b @x> goto <t @y=@x>`), each predecessor's incoming
/// argument is substituted for the corresponding param in the rewritten branch,
/// so `p: goto <b @x=v>` becomes `p: goto <t @y=v>`.
///
/// Conservative guards (any failure leaves the block untouched):
/// * B is not the function entry (the root must remain).
/// * `target != b_id` (a self-loop cannot be bypassed).
/// * B's params are referenced only by B's own branch — otherwise the
///   substitution would leave dangling references.
/// * Every predecessor reaches B through a *static* `Branch`/`CBranch` that
///   names B with a matching argument count. Call continuations, indirect
///   branches, and jump-table edges carry no rewritable target and are skipped.
fn try_bypass_empty_block_concrete<'a, 'str>(
    body: &'a mut FunctionBody<'_, 'str>,
    cx: ContextView<'a, 'str>,
    function_id: FunctionId,
    b_id: BlockId,
) -> bool {
    // The entry block dominates everything; deleting it would orphan the body.
    if body
        .read_host(cx)
        .function_ref(function_id)
        .root()
        .map(|r| r.id)
        == Some(b_id)
    {
        return false;
    }

    // B must hold exactly one instruction, an unconditional branch.
    let (term_id, target, b_args) = {
        let b = body.read_host(cx).block(b_id);
        if b.instructions.len() != 1 {
            return false;
        }
        let term_id = b.instructions[0];
        match body.insn(cx, term_id).mnemonic() {
            Mnemonic::Branch(br) => (term_id, br.target, br.args.clone()),
            _ => return false,
        }
    };
    if target == b_id {
        return false; // bypassing `goto self` is meaningless and unsound
    }
    // Intra-function only: bypassing a block that forwards into another function
    // (a thunk) is a cross-function edit — left to the module pass.
    if target.func != function_id {
        return false;
    }

    let params: Vec<BlockParamId> = body.read_host(cx).block(b_id).params.clone();

    // B's params must flow nowhere but B's own terminator. In valid SSA a block
    // param is only visible inside dominated blocks via forwarded args, so this
    // normally holds; bail if it doesn't rather than risk a dangling use.
    for &p in &params {
        if body
            .users_of(ValueId::BlockParam(p))
            .iter()
            .any(|&u| u != term_id)
        {
            return false;
        }
    }

    // Distinct predecessors of B.
    let preds: Vec<BlockId> = {
        let mut seen = rustc_hash::FxHashSet::default();
        body.block_ref(cx, b_id)
            .predecessors()
            .map(|(_, p)| p)
            .filter(|&p| seen.insert(p))
            .collect()
    };

    // A forwarding block with no predecessors is unreachable dead code. Splicing
    // is about removing a block *on a path*; deleting unreachable blocks is a
    // separate concern, so leave it to a dedicated pass.
    if preds.is_empty() {
        return false;
    }

    // Intra-function only: a predecessor in another function (a tail-call into B)
    // cannot be rewritten by a checked-out pass. Leave such blocks untouched.
    if preds.iter().any(|p| p.func != function_id) {
        return false;
    }

    // Pre-validate every predecessor before mutating anything: each must reach B
    // through a rewritable terminator that names B with a matching arg count on
    // each arm that targets B.
    for &p in &preds {
        let Some(p_term) = body.read_host(cx).block(p).instructions.last().copied() else {
            return false;
        };
        match body.insn(cx, p_term).mnemonic() {
            Mnemonic::Branch(br) => {
                if br.target != b_id || br.args.len() != params.len() {
                    return false;
                }
            }
            Mnemonic::CBranch(cb) => {
                let mut names_b = false;
                if cb.success_block == b_id {
                    if cb.success_args.len() != params.len() {
                        return false;
                    }
                    names_b = true;
                }
                if cb.failure_block == b_id {
                    if cb.failure_args.len() != params.len() {
                        return false;
                    }
                    names_b = true;
                }
                if !names_b {
                    return false;
                }
            }
            _ => return false,
        }
    }

    // Rewrite each predecessor to branch straight to `target`, substituting B's
    // params with the arguments that predecessor supplied.
    for &p in &preds {
        let p_term = body
            .read_host(cx)
            .block(p)
            .instructions
            .last()
            .copied()
            .unwrap();
        let new_mnemonic = match body.insn(cx, p_term).mnemonic().clone() {
            Mnemonic::Branch(br) => Mnemonic::Branch(Branch {
                target,
                args: substitute(&b_args, &params, &br.args),
            }),
            Mnemonic::CBranch(mut cb) => {
                if cb.success_block == b_id {
                    cb.success_args = substitute(&b_args, &params, &cb.success_args);
                    cb.success_block = target;
                }
                if cb.failure_block == b_id {
                    cb.failure_args = substitute(&b_args, &params, &cb.failure_args);
                    cb.failure_block = target;
                }
                Mnemonic::CBranch(cb)
            }
            _ => unreachable!("predecessor terminator validated above"),
        };

        // Rehome the `p -> b` edges to `p -> target`, preserving multiplicity
        // (a CBranch with both arms on B contributes two edges).
        let redirect: Vec<_> = body
            .block_ref(cx, p)
            .successors()
            .filter(|&(_, to)| to == b_id)
            .map(|(e, _)| e)
            .collect();
        body.replace_instruction_mnemonic(cx, p_term, new_mnemonic);
        for &edge in &redirect {
            body.remove_cfg_edge(cx, edge);
        }
        for _ in &redirect {
            body.add_cfg_edge(cx, p, target);
        }
    }

    // B has no predecessors left; delete it (drops its branch and `b -> target`).
    body.delete_block(cx, b_id);
    true
}

/// Maps each value of `template` (an empty block's forwarded branch args)
/// through one predecessor's incoming arguments: a value that is one of the
/// block's `params` is replaced by the predecessor's argument at the same index;
/// any other value (e.g. one defined in a dominating block) is kept as-is.
fn substitute(template: &[ValueId], params: &[BlockParamId], incoming: &[ValueId]) -> Vec<ValueId> {
    template
        .iter()
        .map(|&v| match v {
            ValueId::BlockParam(p) => params
                .iter()
                .position(|&q| q == p)
                .map(|idx| incoming[idx])
                .unwrap_or(v),
            _ => v,
        })
        .collect()
}

#[cfg(test)]
mod tests {

    use qcode::{
        context::Context,
        value::{
            BasicBlock, Function, ValueId,
            insn::{Binary, Mnemonic},
        },
    };
    use qcode_macro::qcode;

    use super::{
        prune_unreachable_generic, simplify_cfg, try_bypass_empty_block_generic,
        try_fold_cbranch_generic, try_merge_block_generic,
    };

    fn make_ctx() -> Context<'static> {
        Context::new()
    }

    #[test]
    fn prunes_unreachable_blocks() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <t>;
            <t>
                goto <0x1001>;
            <dead @x:i64>
                %s = @x + 1;
                goto <t>;
            "
        );

        // `<dead>` has no path from the entry `<a>` — a full run removes it.
        let changed = simplify_cfg(&mut ctx, f);
        assert!(changed, "pruning an unreachable block reports progress");
        assert!(
            BasicBlock::from_id(&ctx, dead).parent().is_none(),
            "unreachable block should be pruned"
        );
        assert!(
            BasicBlock::from_id(&ctx, a).parent().is_some(),
            "the reachable entry survives"
        );
    }

    #[test]
    fn prune_unreachable_is_a_noop_when_all_reachable() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a @cond:i8>
                if @cond goto <t> else goto <u>;
            <t>
                goto <0x1001>;
            <u>
                goto <0x1002>;
            "
        );

        assert!(
            !prune_unreachable_generic(&mut ctx, f),
            "no unreachable blocks means no change"
        );
    }

    #[test]
    fn merges_two_block_chain() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <b>;
            <b>
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        let blocks: Vec<_> = Function::from_id(&ctx, f).blocks().map(|b| b.id).collect();
        assert_eq!(blocks.len(), 1, "two-block chain should merge into one");
        assert_eq!(blocks[0], a);
    }

    #[test]
    fn merges_three_block_chain() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <b>;
            <b>
                goto <c>;
            <c>
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        let blocks: Vec<_> = Function::from_id(&ctx, f).blocks().map(|b| b.id).collect();
        assert_eq!(blocks.len(), 1, "three-block chain should collapse to one");
    }

    #[test]
    fn no_merge_when_a_has_two_successors() {
        // A->B and A->C (diamond entry): A has two successors, no merge possible.
        // B and C carry real instructions so they are not empty forwarding blocks
        // (which would be spliced out) — this isolates the merge guard.
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a @cond:i8>
                if @cond goto <b> else goto <c>;
            <b>
                %x = i64 1 + i64 1;
                goto <0x1001>;
            <c>
                %y = i64 2 + i64 2;
                goto <0x1002>;
            "
        );

        simplify_cfg(&mut ctx, f);

        let blocks: Vec<_> = Function::from_id(&ctx, f).blocks().map(|b| b.id).collect();
        assert_eq!(blocks.len(), 3, "diamond entry should not be merged");
    }

    #[test]
    fn no_merge_when_b_has_two_predecessors() {
        // A->B and D->B: B has two predecessors, no merge. B carries a real
        // instruction so it is not an empty forwarding block (which would be
        // spliced out) — this isolates the merge guard.
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <b>;
            <d>
                goto <b>;
            <b>
                %x = i64 1 + i64 1;
                goto <0x1001>;
            "
        );

        // Isolate the merge guard: `<d>` is unreachable, so a full `simplify_cfg`
        // run would prune it, leaving B single-predecessor and mergeable.
        assert!(
            !try_merge_block_generic(&mut ctx, f, a),
            "B has two predecessors, should not merge"
        );
    }

    #[test]
    fn parent_cleared_on_merged_block() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <b>;
            <b>
                goto <0x1001>;
            "
        );

        {
            let b = BasicBlock::from_id(&ctx, b);
            assert!(b.parent().is_some(), "b should have parent before merge");
        }

        simplify_cfg(&mut ctx, f);

        {
            let a = BasicBlock::from_id(&ctx, a);
            let b = BasicBlock::from_id(&ctx, b);
            assert!(
                b.parent().is_none(),
                "b's parent should be cleared after merge"
            );
            assert!(a.parent().is_some(), "a should still have a parent");
        }
    }

    #[test]
    fn merged_block_arguments_are_rewritten_to_branch_args() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a @input:i64>
                goto <b @x=@input>;

            <b @x:i64>
                %sum = @x + 1;
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        let blocks: Vec<_> = Function::from_id(&ctx, f).blocks().map(|b| b.id).collect();
        assert_eq!(blocks, [a], "branch-with-args chain should merge");
        assert!(BasicBlock::from_id(&ctx, b).parent().is_none());

        let Mnemonic::Binop(Binary { lhs, .. }) = ctx.instruction(sum).mnemonic() else {
            panic!("expected merged sum to be a binop");
        };
        assert_eq!(*lhs, ValueId::BlockParam(input));
    }

    #[test]
    fn merged_instructions_have_correct_parent() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                %x = i64 1 + i64 1;
                goto <b>;
            <b>
                %y = i64 2 + i64 2;
                goto <0x1001>;
            "
        );

        let y_insn = y;
        simplify_cfg(&mut ctx, f);

        let insn = ctx.get_insn(y_insn);
        let parent_id = insn.parent().map(|b| b.id());
        assert_eq!(
            parent_id,
            Some(ValueId::BasicBlock(a)),
            "instruction from b should be reparented to a after merge"
        );
    }

    /// Regression: `absorb_block` must *drain* the absorbed block's instruction
    /// list, not merely copy it. Leaving the ids in both blocks puts every merged
    /// instruction in two blocks at once; a later `remove_instruction` (which
    /// unlinks via the instruction's `parent`) then clears one copy while the
    /// other lingers, corrupting block membership and spinning `remove_dead_insns`
    /// forever.
    #[test]
    fn absorbed_block_instruction_list_is_drained() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                %x = i64 1 + i64 1;
                goto <b>;
            <b>
                %y = i64 2 + i64 2;
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        assert!(
            ctx.block(b).instructions.is_empty(),
            "absorbed block `b` must not retain its instructions after merge"
        );
    }

    /// Regression (invariant): after CFG simplification no instruction id may
    /// appear in more than one block's instruction list. This is the property the
    /// `absorb_block` drain fix restores; its violation was the root cause of an
    /// infinite loop in `remove_dead_insns`.
    #[test]
    fn no_instruction_belongs_to_two_blocks_after_simplify() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                %x = i64 1 + i64 1;
                goto <b>;
            <b>
                %y = i64 2 + i64 2;
                goto <c>;
            <c>
                %z = i64 3 + i64 3;
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        let mut seen = rustc_hash::FxHashSet::default();
        for block_id in ctx.block_ids() {
            for &insn in &ctx.block(block_id).instructions {
                assert!(
                    seen.insert(insn),
                    "instruction {insn:?} appears in more than one block after simplify_cfg"
                );
            }
        }
    }

    /// Regression: `remove_dead_insns` must terminate (and actually remove the
    /// dead instruction) after a merge. With the duplicate-membership bug a dead
    /// instruction left in a stale block list was re-discovered every round, so
    /// this loop never finished.
    #[test]
    fn remove_dead_insns_terminates_after_merge() {
        use crate::remove_dead_insns;
        use qcode::value::Function;

        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <b>;
            <b>
                %y = i64 2 + i64 2;
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        let blocks: Vec<_> = Function::from_id(&ctx, f)
            .iter()
            .map(|blk| blk.id)
            .collect();
        for bid in blocks {
            remove_dead_insns(&mut ctx, bid);
        }

        assert!(
            Function::from_id(&ctx, f)
                .iter()
                .all(|blk| !blk.instruction_ids().contains(&y)),
            "dead merged instruction must be removed, not looped on"
        );
    }

    // ----- empty-block bypass -----------------------------------------------

    /// A no-param empty forwarding block with two predecessors (so the
    /// straight-line merge can't fire) is spliced out, and both predecessors are
    /// redirected to its target.
    #[test]
    fn bypasses_empty_block_with_two_predecessors() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <b>;
            <d>
                goto <b>;
            <b>
                goto <t>;
            <t>
                %s = i64 1 + i64 1;
                goto <0x1001>;
            "
        );

        // Bypass in isolation: `<d>` is unreachable, so a full `simplify_cfg`
        // run would prune it (and then splice the forwarding `<a>`/`<d>`).
        try_bypass_empty_block_generic(&mut ctx, f, b);

        // b is gone; a and d both branch straight to t.
        assert!(
            BasicBlock::from_id(&ctx, b).parent().is_none(),
            "empty forwarding block b should be spliced out"
        );
        for pred in [a, d] {
            let term = BasicBlock::from_id(&ctx, pred)
                .iter()
                .last()
                .expect("pred has a terminator");
            let Mnemonic::Branch(br) = term.mnemonic() else {
                panic!("predecessor should end in an unconditional branch");
            };
            assert_eq!(br.target, t, "predecessor should target t directly");
        }
    }

    /// When the empty block carries a parameter and forwards it, each
    /// predecessor's incoming argument is substituted into the rewritten branch.
    #[test]
    fn bypass_threads_block_arguments_per_predecessor() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a @av:i64>
                goto <b @x=@av>;
            <d @dv:i64>
                goto <b @x=@dv>;
            <b @x:i64>
                goto <t @y=@x>;
            <t @y:i64>
                %s = @y + 1;
                goto <0x1001>;
            "
        );

        // Bypass in isolation (see `bypasses_empty_block_with_two_predecessors`):
        // a full run would prune the unreachable `<d>` predecessor.
        try_bypass_empty_block_generic(&mut ctx, f, b);

        assert!(
            BasicBlock::from_id(&ctx, b).parent().is_none(),
            "forwarding block b should be spliced out"
        );

        // a forwards its own @av straight to t; d forwards @dv.
        let arg_to_t = |pred| {
            let term = BasicBlock::from_id(&ctx, pred)
                .iter()
                .last()
                .expect("pred has a terminator");
            let Mnemonic::Branch(br) = term.mnemonic() else {
                panic!("expected branch");
            };
            assert_eq!(br.target, t);
            br.args[0]
        };
        assert_eq!(arg_to_t(a), ValueId::BlockParam(av));
        assert_eq!(arg_to_t(d), ValueId::BlockParam(dv));
    }

    /// A predecessor that reaches the empty block through one arm of a
    /// conditional branch has just that arm retargeted.
    #[test]
    fn bypass_redirects_cbranch_arm() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a @cond:i8>
                if @cond goto <b> else goto <other>;
            <other>
                %o = i64 9 + i64 9;
                goto <0x1002>;
            <b>
                goto <t>;
            <t>
                %s = i64 1 + i64 1;
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        assert!(
            BasicBlock::from_id(&ctx, b).parent().is_none(),
            "empty block b should be spliced out"
        );
        let term = BasicBlock::from_id(&ctx, a)
            .iter()
            .last()
            .expect("a has a terminator");
        let Mnemonic::CBranch(cb) = term.mnemonic() else {
            panic!("a should still be a conditional branch");
        };
        assert_eq!(cb.success_block, t, "the b arm should now target t");
        assert_eq!(cb.failure_block, other, "the other arm is untouched");
    }

    /// A self-looping forwarding block (`goto self`) is never bypassed.
    #[test]
    fn does_not_bypass_self_loop() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <b>;
            <b>
                goto <b>;
            "
        );

        simplify_cfg(&mut ctx, f);

        assert!(
            BasicBlock::from_id(&ctx, b).parent().is_some(),
            "self-looping block must survive"
        );
    }

    /// The function entry is never deleted, even when it is an empty forwarding
    /// block with an incoming back-edge.
    #[test]
    fn does_not_bypass_root_block() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a @cond:i8>
                goto <b @c=@cond>;
            <b @c:i8>
                if @c goto <a @cond=@c> else goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        assert!(
            BasicBlock::from_id(&ctx, a).parent().is_some(),
            "the root block must never be spliced out"
        );
        assert_eq!(
            Function::from_id(&ctx, f).root().map(|r| r.id),
            Some(a),
            "a should remain the function root"
        );
    }

    /// An unreachable forwarding block (no predecessors, not the root) is left
    /// alone — splicing is only for blocks on a path.
    #[test]
    fn prunes_unreachable_forwarding_block() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                %s = i64 1 + i64 1;
                goto <0x1001>;
            <orphan>
                goto <t>;
            <t>
                %u = i64 2 + i64 2;
                goto <0x1002>;
            "
        );

        simplify_cfg(&mut ctx, f);

        // `<orphan>` and `<t>` (reachable only via orphan) have no path from the
        // entry `<a>`; simplify_cfg now prunes such dead blocks itself.
        assert!(
            BasicBlock::from_id(&ctx, orphan).parent().is_none(),
            "unreachable forwarding block should be pruned"
        );
        assert!(
            BasicBlock::from_id(&ctx, t).parent().is_none(),
            "block reachable only from an unreachable block should be pruned too"
        );
    }

    // ----- conditional-branch folding ---------------------------------------

    /// A conditional branch whose arms share a target and arguments collapses to
    /// an unconditional branch with a single successor edge.
    #[test]
    fn folds_cbranch_with_identical_arms() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a @cond:i8>
                if @cond goto <t> else goto <t>;
            <d>
                goto <t>;
            <t>
                %s = i64 1 + i64 1;
                goto <0x1001>;
            "
        );

        // Exercise the fold in isolation: a full `simplify_cfg` run would prune
        // the unreachable `<d>`, then merge the folded `goto <t>` into t, hiding
        // the very Branch this test inspects.
        try_fold_cbranch_generic(&mut ctx, a);

        let term = BasicBlock::from_id(&ctx, a)
            .iter()
            .last()
            .expect("a has a terminator");
        let Mnemonic::Branch(br) = term.mnemonic() else {
            panic!("identical-armed cbranch should fold to an unconditional branch");
        };
        assert_eq!(br.target, t);
        assert_eq!(
            BasicBlock::from_id(&ctx, a).successors().count(),
            1,
            "the duplicate parallel edge should be dropped"
        );
    }

    /// A conditional branch to a single target but with *different* per-arm
    /// arguments is NOT folded (the condition still selects the argument).
    #[test]
    fn does_not_fold_cbranch_with_differing_args() {
        let mut ctx = make_ctx();
        qcode!(
            ctx,
            "
            fn f:
            <a @cond:i8 @p:i64 @q:i64>
                if @cond goto <t @y=@p> else goto <t @y=@q>;
            <t @y:i64>
                %s = @y + 1;
                goto <0x1001>;
            "
        );

        simplify_cfg(&mut ctx, f);

        let term = BasicBlock::from_id(&ctx, a)
            .iter()
            .last()
            .expect("a has a terminator");
        assert!(
            matches!(term.mnemonic(), Mnemonic::CBranch(_)),
            "a cbranch with differing arm arguments must not be folded"
        );
    }
}
