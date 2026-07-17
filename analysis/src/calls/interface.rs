//! Lockstep mutation of a `pure_reg` function's call interface.
//!
//! A `pure_reg` function maintains the invariant `param[i] ↔ arg[i]` — its root
//! block params and every direct caller's `Call.args` are aligned
//! index-for-index (see the emulator's `bind_entry_params_from_args` and
//! `dead_signature`, both of which drive off the params). These two helpers are
//! the only places that grow or shrink that interface, so the alignment holds
//! by construction:
//!
//! * [`append_entry_param`] — the *add* side, used by the argpromote passes to
//!   thread a by-value input (a register or a stack slot) into the callee and
//!   every caller in one shot.
//! * [`remove_entry_param`] — the *remove* side, used by DCE's dead-param sweep
//!   and `dead_signature` to drop an input no body reads.
//!
//! `signature.inputs` (`input_regs`) is the conventional ABI register list. It is
//! never populated for a `pure_reg` function (argpromote leaves it `None` and
//! the params carry the interface), so [`append_entry_param`] does not touch it;
//! [`remove_entry_param`] trims it only defensively, when it happens to be set.

use qcode::value::QCodeMut;
use std::borrow::Cow;

use qcode::{
    context::Context,
    value::{
        BasicBlock, BlockId, BlockParamId, FunctionBody, FunctionId, ValueId,
        insn::{Call, InstructionId, Mnemonic},
    },
};

/// Append a new by-value input to `fid` and thread it through every direct
/// caller, in lockstep: a new root block param (`size` bytes, named `name`,
/// recording `origin` so mem2reg can reuse it) and a new positional `Call.args`
/// value at each direct call site. The argument value is built by
/// `build_caller_value`, given the call instruction's id and block so it can
/// insert loads before the call. Returns the new param's `ValueId`, or `None`
/// if `fid` has no root block.
///
/// The add-mirror of [`remove_entry_param`]. The new param/arg go at the end,
/// preserving every existing index. `signature.inputs` is intentionally left
/// untouched (see the module docs).
pub fn append_entry_param(
    ctx: &mut Context,
    fid: FunctionId,
    size: usize,
    name: Option<String>,
    origin: Option<ValueId>,
    build_caller_value: impl FnMut(&mut Context, InstructionId, BlockId) -> ValueId,
) -> Option<ValueId> {
    let call_sites = super::fresh_direct_call_sites(ctx, fid);
    append_entry_param_at_sites(
        ctx,
        fid,
        size,
        name,
        origin,
        &call_sites,
        build_caller_value,
    )
}

/// [`append_entry_param`] with caller sites supplied by an analysis snapshot
/// known to be preserved by the enclosing pass.
pub(crate) fn append_entry_param_at_sites(
    ctx: &mut Context,
    fid: FunctionId,
    size: usize,
    name: Option<String>,
    origin: Option<ValueId>,
    call_sites: &[InstructionId],
    mut build_caller_value: impl FnMut(&mut Context, InstructionId, BlockId) -> ValueId,
) -> Option<ValueId> {
    let root = FunctionBody::from_id(ctx, fid).root().map(|b| b.id)?;

    // Changing the parameter list invalidates any inferred per-param attributes,
    // whose vector is indexed by the old positions. Drop them; the `param_attrs`
    // pass re-infers over the rewritten signature.
    FunctionBody::from_id_mut(ctx, fid).clear_param_attrs();

    // New root block param, recording its source for mem2reg reuse and naming.
    let pid = BasicBlock::from_id_mut(ctx, root).push_param(size).id;
    if let Some(name) = name {
        ctx.block_param_mut(pid).name = Some(Cow::Owned(name));
    }
    if let Some(origin) = origin {
        ctx.block_param_mut(pid)
            .set_origin_id(origin.localize(pid.func));
    }

    // Append the matching positional argument at every direct call site.
    append_caller_arg_at_sites(ctx, call_sites, |ctx, call_id, block| {
        Some(build_caller_value(ctx, call_id, block))
    });

    Some(ValueId::BlockParam(pid))
}

/// Append one positional argument — built per call site by `build` — to every
/// direct caller of `fid`, in place. The low-level lockstep primitive used by
/// [`append_entry_param`] (after it adds the param). The value lands at the end of
/// `Call.args`, so to keep `param[i] ↔ arg[i]` aligned the caller must invoke this
/// in param order. `build` returning `None` skips that call site (e.g. one whose
/// argument is already present, for idempotent re-runs). Returns `true` if it
/// appended an argument at any call site.
pub fn append_caller_arg(
    ctx: &mut Context,
    fid: FunctionId,
    build: impl FnMut(&mut Context, InstructionId, BlockId) -> Option<ValueId>,
) -> bool {
    let call_sites = super::fresh_direct_call_sites(ctx, fid);
    append_caller_arg_at_sites(ctx, &call_sites, build)
}

pub(crate) fn append_caller_arg_at_sites(
    ctx: &mut Context,
    call_sites: &[InstructionId],
    mut build: impl FnMut(&mut Context, InstructionId, BlockId) -> Option<ValueId>,
) -> bool {
    let mut changed = false;
    for &call_id in call_sites {
        let Some(block) = ctx.get_insn(call_id).parent().map(|b| b.id) else {
            continue;
        };
        let Some(value) = build(ctx, call_id, block) else {
            continue;
        };
        let Mnemonic::Call(call) = ctx.get_insn(call_id).mnemonic().clone() else {
            continue;
        };
        let mut args = call.args;
        args.push(value.localize(call_id.func));
        ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: call.target,
                args,
                clobbers: call.clobbers,
                tag: call.tag,
            }),
        );
        changed = true;
    }
    changed
}

/// Remove the entry param at position `index` from `fid` and keep its interface
/// aligned: drop the root block param, the `input_regs[index]` entry, and the
/// `Call.args[index]` argument at every direct caller, all in lockstep.
///
/// This is the single ABI-consistent entry-param removal both DCE's dead-param
/// sweep and `dead_signature` route through, so a `pure_reg` function's
/// `param[i] ↔ input_regs[i] ↔ arg[i]` alignment holds by construction after
/// any removal. The caller must ensure the param has no remaining users.
pub fn remove_entry_param(ctx: &mut Context, fid: FunctionId, index: usize) {
    // Snapshot caller IDs before the first structural mutation, then drop the
    // graph. The relationship itself is unchanged by this lockstep rewrite.
    let call_sites = super::fresh_direct_call_sites(ctx, fid);
    remove_entry_params_at_sites(ctx, fid, &[index], &call_sites);
}

/// Remove several entry params using caller sites supplied by an analysis
/// snapshot known to remain valid for the rewrite.
///
/// All params and matching call arguments are removed in one batch, so each
/// caller instruction is cloned and replaced at most once. `indices` may be in
/// any order and may contain duplicates or out-of-range positions.
pub(crate) fn remove_entry_params_at_sites(
    ctx: &mut Context,
    fid: FunctionId,
    indices: &[usize],
    call_sites: &[InstructionId],
) {
    let Some(root) = FunctionBody::from_id(ctx, fid).root().map(|b| b.id) else {
        return;
    };

    let old_params = ctx.block(root).params.clone();
    let mut removed = indices
        .iter()
        .copied()
        .filter(|&index| index < old_params.len())
        .collect::<Vec<_>>();
    removed.sort_unstable();
    removed.dedup();
    if removed.is_empty() {
        return;
    }

    // Drop the selected root-param payloads, then install and reindex the
    // survivors once rather than after every individual removal.
    for &index in &removed {
        ctx.remove_block_param(BlockParamId::new(root.func, old_params[index]));
    }
    let params = old_params
        .into_iter()
        .enumerate()
        .filter_map(|(index, param)| (!removed.contains(&index)).then_some(param))
        .collect::<Vec<_>>();

    // Any inferred per-param attributes are indexed by the old positions; drop
    // them rather than reindex. The `param_attrs` pass re-infers afterward.
    FunctionBody::from_id_mut(ctx, fid).clear_param_attrs();
    for (i, &p) in params.iter().enumerate() {
        ctx.block_param_mut(BlockParamId::new(root.func, p)).index = i;
    }
    ctx.block_mut(root).params = params;

    // Drop the matching conventional input-register entries. This only acts
    // when `input_regs` is set (conventional functions); for
    // `pure_reg` it is `None` and this is a no-op (see the module docs).
    if let Some(inputs) = FunctionBody::from_id(ctx, fid).input_regs() {
        let inputs = inputs
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, input)| (!removed.contains(&index)).then_some(input))
            .collect();
        FunctionBody::from_id_mut(ctx, fid).set_input_regs(inputs);
    }

    // Drop all matching positional arguments at every direct caller, replacing
    // each call instruction only once.
    for &call_id in call_sites {
        if !ctx.contains_instruction(call_id) {
            continue;
        }
        let Mnemonic::Call(call) = ctx.get_insn(call_id).mnemonic().clone() else {
            continue;
        };
        let args = call
            .args
            .into_iter()
            .enumerate()
            .filter_map(|(index, arg)| (!removed.contains(&index)).then_some(arg))
            .collect();
        ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: call.target,
                args,
                clobbers: call.clobbers,
                tag: call.tag,
            }),
        );
    }
}
