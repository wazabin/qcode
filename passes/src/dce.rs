//! Dead-instruction elimination: removing pure instructions nothing reads.

use rustc_hash::FxHashSet as HashSet;

use qcode::{
    context::Context,
    value::{BlockId, FunctionBody, InstructionId, QCodeView, ValueId},
};

use crate::{PassCtx, with_body_mut};

/// This host's users of `v` as body-local ids, without allocating.
///
/// Body-local because that is how they are stored: the qualified form exists
/// only to be built, and this is read once per instruction.
fn users_of_slice<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    v: ValueId,
) -> &'a [qcode::value::insn::LocalInsnId] {
    match v.owning_function() {
        Some(f) => host.function_ref(f).local_users_of(v),
        None => &[],
    }
}

/// Whether this host records any user of `v`. A shared value with no owning
/// function has none. Mirrors [`Context::has_users`].
///
/// Deliberately not `users_of(v).is_empty()`: this is asked once per
/// instruction per round, and building a vector only to discard it was the
/// largest single cost of lifting a block.
pub fn has_users<'a, 'str: 'a>(host: impl QCodeView<'a, 'str>, v: ValueId) -> bool {
    match v.owning_function() {
        Some(f) => host.function_ref(f).has_users(v),
        None => false,
    }
}

/// This host's users of `v`. Allocates; prefer [`has_users`] to ask whether
/// there are any, and `users_of_slice` to look at them in a hot loop.
pub fn host_users<'a, 'str: 'a>(host: impl QCodeView<'a, 'str>, v: ValueId) -> Vec<InstructionId> {
    match v.owning_function() {
        Some(f) => host.function_ref(f).users_of(v),
        None => Vec::new(),
    }
}

/// Returns instructions in `block_id` that are pure and have no *live* users:
/// no users at all, or none that this same answer does not already condemn.
///
/// Walked in reverse so that one pass reaches the fixed point. Within a block
/// the IR is SSA, so a definition precedes its uses; going backwards means
/// every user of an instruction has already been judged by the time the
/// instruction itself is, and a whole dead chain falls in one sweep. Repeating
/// a forward scan until nothing changes reaches the same answer, but rescans
/// the entire block once per link in the longest chain — which on a lifted
/// guest basic block was the dominant cost of translation.
pub fn dead_insns<'a, 'str: 'a>(
    host: impl QCodeView<'a, 'str>,
    block_id: BlockId,
) -> HashSet<InstructionId> {
    let mut dead: HashSet<InstructionId> = HashSet::default();
    let insn_ids: Vec<InstructionId> = host.block_ref(block_id).instruction_ids().to_vec();
    let func = block_id.func;
    for id in insn_ids.into_iter().rev() {
        if host.instruction(id).mnemonic().has_side_effects() {
            continue;
        }
        // A user in another block is never in `dead`, so it keeps this
        // instruction alive — as it must.
        let live_user = users_of_slice(host, ValueId::Instruction(id))
            .iter()
            .any(|&user| !dead.contains(&InstructionId::new(func, user)));
        if !live_user {
            dead.insert(id);
        }
    }
    dead
}

/// Removes dead pure instructions from `block_id` iteratively until fixed point,
/// updating the users reverse map after each round.
pub fn remove_dead_insns(ctx: &mut Context, block_id: BlockId) -> bool {
    with_body_mut(ctx, block_id.func, |body, cx| {
        remove_dead_insns_body(body, cx, block_id)
    })
}

/// Body-local core of [`remove_dead_insns`].
pub fn remove_dead_insns_body<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: PassCtx<'a, 'str>,
    block_id: BlockId,
) -> bool {
    let mut changed = false;
    loop {
        let dead = dead_insns(cx.body_view(body), block_id);
        if dead.is_empty() {
            break;
        }

        changed = true;
        // One pass over the block's instruction list however many go: removing
        // them one at a time is quadratic in the size of the block, and lifted
        // guest basic blocks are large.
        let dead: HashSet<_> = dead
            .into_iter()
            .map(|id| id.localize(block_id.func))
            .collect();
        body.remove_block_instructions(block_id, &dead);
    }

    let params_changed = remove_unused_no_pred_block_params(body, cx, block_id);
    changed || params_changed
}

/// Removes block params that have no users when the block has no incoming
/// control-flow edges. This covers function-entry params introduced for
/// load-before-store registers that later become dead, without touching join
/// blocks whose predecessor terminators carry positional arguments.
pub fn remove_unused_no_pred_block_params<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: PassCtx<'a, 'str>,
    block_id: BlockId,
) -> bool {
    if cx
        .body_view(body)
        .block_ref(block_id)
        .predecessors()
        .next()
        .is_some()
    {
        return false;
    }

    // A `pure_reg` function's entry params are its canonical interface, aligned
    // index-for-index with every caller's `Call.args`. Removing one is an
    // interprocedural change that must drop the param and the matching argument at
    // every caller in lockstep — that is the
    // job of the `dead_signature` module pass (via `remove_entry_param`), not of
    // this per-function sweep. A function pass must not reach across functions, so
    // leave pure_reg entry params for `dead_signature`; the *local* fallback below
    // would silently drop the param and break the interface alignment.
    let is_reg_materialized_entry = cx
        .body_view(body)
        .block_ref(block_id)
        .function()
        .is_some_and(|f| f.is_reg_materialized() && f.root().map(|b| b.id) == Some(block_id));
    if is_reg_materialized_entry {
        return false;
    }

    let params: Vec<_> = cx.body_view(body).block(block_id).param_ids().to_vec();
    let mut kept = Vec::with_capacity(params.len());
    let mut changed = false;
    for local in params {
        let param = qcode::value::BlockParamId::new(block_id.func, local);
        if !has_users(cx.body_view(body), ValueId::BlockParam(param)) {
            body.remove_block_param(param);
            changed = true;
        } else {
            body.block_param_mut(param).index = kept.len();
            kept.push(local);
        }
    }

    if changed {
        body.block_mut(block_id).params = kept;
    }
    changed
}
