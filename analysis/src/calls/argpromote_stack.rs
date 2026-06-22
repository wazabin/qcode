//! `argpromote_stack`: functionalize the **stack input** channel of `pure_reg`
//! functions, the source of truth for stack arguments (legacy `bind_call_args`
//! stays bypassed for `pure_reg`; see [`super::binding`]).
//!
//! Stack inputs are **positional** like registers, not `(addr, value)` pairs
//! like the RAM channel: a stack offset is statically known and *frame-relative*
//! (`@stack_base+4` is a different physical byte in each frame), so no single
//! address can travel as a runtime value, and none needs to — the payload passed
//! is the value, positionally, and the address is analysis-time metadata for
//! caller translation only.
//!
//! For each `pure_reg` callee, every statically-resolved stack **read** at a
//! `@stack_base` offset `O` is functionalized:
//!
//! * **Callee** — a by-value param `P` plus `store(@stack_base + O, P)` at entry.
//!   The slot becomes stored-then-loaded and no longer live-in, so mem2reg's
//!   normal stack promotion forwards `P` into the body loads and adds no root
//!   param of its own (the same seed-and-replay shape as the register channel).
//! * **Caller** — `load(args[rsp_idx] + O)` before each direct call, threaded as
//!   a positional argument. `args[rsp_idx]` is the already-passed stack-pointer
//!   argument: the callee's entry stack pointer (the slot the call's
//!   `store(inst_next)` writes), i.e. callee offset 0, brightened to
//!   `@stack_base_caller + N`, so each call site self-encodes its own stack depth;
//!   the next mem2reg/gvn forwards the load to the caller's push-store.
//!
//! The return-address slot (`O = 0`) is not special: the caller value loads the
//! slot the call's `store(inst_next)` writes (`@stack_base ± 0` from the SP arg
//! sampled before the call), so the continuation address rides the same
//! mechanism.
//!
//! Gate: `O ≥ 0` is threaded (return address + caller stack args); `O < 0` is a
//! callee local we cannot supply, so it is warned about and left in place. An
//! unresolved address bails this round — the outer checkpoint+replay driver
//! retries once mem2reg has resolved more (e.g. RBP-relative addressing). The
//! pass is idempotent (it skips offsets already backed by a root param), so the
//! driver fixpoints.
//!
//! mem2reg's own incoming-stack-param promotion is gated off for `pure_reg`
//! (`mem/mem2reg.rs`), ceding the channel here: a bail then degrades to an
//! unpromoted-but-correct stack load, never an orphan param.

use qcode::{
    builder::Builder,
    context::Context,
    value::{BasicBlock, BlockParamId, Function, FunctionId, RegisterId, Value, insn::Mnemonic},
};

use crate::{Pass, PipelineEnv, append_caller_arg, mem::mem2reg::stack_slot_offset};

/// Backfill the caller stack arguments of every eligible `pure_reg` function (see
/// [`try_promote_stack`]). Returns `true` if anything changed.
pub fn argpromote_stack(ctx: &mut Context, stack_ptr: RegisterId) -> bool {
    let mut changed = false;
    for fid in ctx.function_ids() {
        if try_promote_stack(ctx, fid, stack_ptr) {
            changed = true;
        }
    }
    changed
}

/// Supply the missing caller arguments for the stack inputs mem2reg has already
/// promoted into `fid`'s entry block.
///
/// mem2reg promotes each load-only caller-frame stack slot into a root block param
/// and records the slot literal as the param's `origin`. For a `pure_reg` callee
/// that param has no matching `Call.args` entry yet — the lockstep gap this closes.
/// We read the offset back off each such param's `origin` and append the positional
/// argument `load(rsp_arg + (offset - ptr_width))` at every direct call site, where
/// `rsp_arg` is the already-threaded stack-pointer argument (brightened to
/// `@stack_base_caller + N`). The following gvn forwards that load to the caller's
/// push. We do **not** re-detect or re-promote slots, mint params, or seed the
/// callee body — mem2reg owns all of that; this pass is purely the interprocedural
/// caller half. Idempotent: a call already carrying the argument is skipped.
fn try_promote_stack(ctx: &mut Context, fid: FunctionId, stack_ptr: RegisterId) -> bool {
    {
        let f = Function::from_id(ctx, fid);
        // Only `pure_reg` functions carry the param↔arg lockstep this maintains
        // (it already implies non-external, address-not-taken, direct callers only).
        if !f.is_pure_reg() || f.is_external() || f.root().is_none() {
            return false;
        }
    }
    let root = Function::from_id(ctx, fid)
        .root()
        .expect("checked above")
        .id;
    let default_space = ctx.default_space;

    // The caller-arg position carrying the stack pointer: the root param named
    // after the stack-pointer register (kept and named by argpromote_registers;
    // brighten relabels its *uses* to `@stack_base` but keeps the param).
    let Some(sp_name) = ctx.get_register(stack_ptr).name().map(str::to_owned) else {
        return false;
    };
    let params: Vec<BlockParamId> = BasicBlock::from_id(ctx, root)
        .params()
        .map(|p| p.id)
        .collect();
    let Some(rsp_idx) = params
        .iter()
        .position(|&pid| ctx.values.block_params[pid].name.as_deref() == Some(sp_name.as_str()))
    else {
        return false;
    };

    // The stack inputs are exactly the entry params mem2reg promoted from a
    // caller-frame stack slot — identified by an `origin` decoding to a stack
    // offset `>= 0` (the return-address slot at offset 0 and the caller's stack
    // arguments above it; a local below the entry SP is `< 0` and is the callee's
    // own, not a caller input). Recorded in param order as
    // (param_index, delta-from-rsp-arg, ptr_width, size).
    let mut stack_inputs: Vec<(usize, i64, usize, usize)> = Vec::new();
    for (i, &pid) in params.iter().enumerate() {
        let bp = &ctx.values.block_params[pid];
        let Some((offset, ptr_width)) = bp.origin.and_then(|v| stack_slot_offset(ctx, v)) else {
            continue;
        };
        if offset < 0 {
            continue;
        }
        let size = ctx.types.size_of(bp.type_id);
        stack_inputs.push((i, offset, ptr_width, size));
    }
    if stack_inputs.is_empty() {
        return false;
    }

    // Every direct caller must already carry the stack-pointer argument (threaded
    // by argpromote_registers) so we can offset from it; otherwise bail.
    let mut has_caller = false;
    for insn in ctx.instructions() {
        if let Mnemonic::Call(c) = insn.mnemonic()
            && c.target == fid
        {
            has_caller = true;
            if c.args.len() <= rsp_idx {
                return false;
            }
        }
    }
    if !has_caller {
        return false;
    }

    // Backfill each stack input's caller argument, in param order. `append_caller_arg`
    // only fills a call site whose next positional slot is exactly this param
    // (`args.len() == idx`), keeping `param[i] ↔ arg[i]` aligned and skipping a call
    // that already carries the argument. The SP argument is the callee's entry stack
    // pointer (the slot the call's `store(inst_next)` writes the return address into),
    // i.e. callee offset 0, so a callee slot at `offset` sits exactly `offset` bytes
    // above it.
    let mut changed = false;
    for &(idx, offset, ptr_width, size) in &stack_inputs {
        let delta = offset;
        changed |= append_caller_arg(ctx, fid, |ctx, call_id, block| {
            let Mnemonic::Call(c) = ctx.get_insn(call_id).mnemonic() else {
                return None;
            };
            if c.args.len() != idx {
                return None;
            }
            let rsp_arg = c.args[rsp_idx];
            let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, block));
            b.set_insert_point_before(call_id);
            // `offset >= 0` is enforced above, so `delta` is never negative.
            let addr = if delta == 0 {
                rsp_arg
            } else {
                let c = b.context_mut().get_const(delta as u64, ptr_width).id();
                b.push_add(rsp_arg, c).id()
            };
            Some(b.push_load::<false>(addr, size, default_space).id())
        });

        // Rename the promoted param to `arg_stack_<offset>` so it is visible in the
        // IR that argpromote_stack has threaded its caller argument — distinct from
        // mem2reg's raw `stack_<addr>`. Cosmetic only: mem2reg and this pass both
        // key on `origin`, never the name.
        let pid = params[idx];
        let new_name = format!("arg_stack_{offset}");
        let bp = &mut ctx.values.block_params[pid];
        if bp.name.as_deref() != Some(new_name.as_str()) {
            bp.name = Some(new_name.into());
            changed = true;
        }
    }

    changed
}

#[derive(Default)]
pub struct ArgPromoteStack;

impl Pass for ArgPromoteStack {
    const NAME: &'static str = "argpromote_stack";
    fn description(&self) -> &'static str {
        "Functionalize stack inputs of pure-reg functions"
    }
    fn run(&self, ctx: &mut Context, env: &PipelineEnv) -> Result<bool, String> {
        Ok(argpromote_stack(ctx, env.cfg.stack_pointer))
    }
}

crate::register_module_pass!(ArgPromoteStack);
