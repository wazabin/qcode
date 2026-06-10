//! Architecture-independent call summaries: callee input/clobber inference and
//! per-call-site argument + alias binding.
//!
//! Everything here is derived from the IR itself — there are no calling-
//! convention tables. Two stages cooperate:
//!
//! 1. [`set_function_summaries`] fills each function's [`FunctionSignature`]
//!    with the registers it reads before writing (`inputs`, a backward
//!    register-liveness) and the registers it writes (`clobbered`, reusing
//!    [`compute_clobbered_regs`]). Run it over every function first so callee
//!    summaries exist before callers consume them.
//! 2. [`bind_call_args`] runs *after* mem2reg: for each direct `Call`, it looks
//!    up the callee's inferred `inputs` and resolves the reaching value of each
//!    in the caller, storing them in `Call.args`. It also records the locations
//!    the call may alias/clobber in `Call.clobbers`.

use std::collections::{HashMap, HashSet};

use qcode::{
    builder::Builder,
    context::Context,
    space::SpaceType,
    value::{
        BasicBlock, BlockId, Function, FunctionId, Value, ValueId, Varnode, VarnodeId,
        insn::{InstructionId, Mnemonic},
        literal::{Literal, SymbolicRef},
    },
};

use crate::compute_clobbered_regs;

/// True when `vn` lives in a register address space.
fn is_register(ctx: &Context, vn: VarnodeId) -> bool {
    matches!(Varnode::from_id(ctx, vn).space().ty, SpaceType::Register)
}

/// Upward-exposed register reads (`used`) and register writes (`defined`) of a
/// single block, computed by a forward scan.
fn block_reg_flow(ctx: &Context, block: BlockId) -> (HashSet<VarnodeId>, HashSet<VarnodeId>) {
    let mut used = HashSet::new();
    let mut defined = HashSet::new();
    // Registers already written earlier in this block; a later read of one of
    // these is satisfied locally and is not upward-exposed.
    let mut written: HashSet<VarnodeId> = HashSet::new();

    for insn in BasicBlock::from_id(ctx, block).iter() {
        match insn.mnemonic() {
            Mnemonic::Load(load) => {
                if let ValueId::Varnode(vn) = load.ptr
                    && is_register(ctx, vn)
                    && !written.contains(&vn)
                {
                    used.insert(vn);
                }
            }
            Mnemonic::Store(store) => {
                if let ValueId::Varnode(vn) = store.ptr
                    && is_register(ctx, vn)
                {
                    written.insert(vn);
                    defined.insert(vn);
                }
            }
            _ => {}
        }
    }

    (used, defined)
}

/// Registers read before being written along some path from the entry block —
/// the function's inferred inputs. Returned sorted by [`VarnodeId`] for a stable
/// order shared by callers when binding arguments.
pub fn compute_input_regs(ctx: &Context, function_id: FunctionId) -> Vec<VarnodeId> {
    let Some(root) = Function::from_id(ctx, function_id).root().map(|b| b.id) else {
        return Vec::new();
    };

    let blocks: Vec<BlockId> = Function::from_id(ctx, function_id)
        .iter()
        .map(|b| b.id)
        .collect();

    let flow: HashMap<BlockId, (HashSet<VarnodeId>, HashSet<VarnodeId>)> = blocks
        .iter()
        .map(|&b| (b, block_reg_flow(ctx, b)))
        .collect();

    let succs: HashMap<BlockId, Vec<BlockId>> = blocks
        .iter()
        .map(|&b| {
            (
                b,
                BasicBlock::from_id(ctx, b)
                    .successors()
                    .map(|(_, s)| s)
                    .collect(),
            )
        })
        .collect();

    let mut live_in: HashMap<BlockId, HashSet<VarnodeId>> =
        blocks.iter().map(|&b| (b, HashSet::new())).collect();

    let mut changed = true;
    while changed {
        changed = false;
        for &block in &blocks {
            let mut live_out: HashSet<VarnodeId> = HashSet::new();
            for &succ in &succs[&block] {
                if let Some(set) = live_in.get(&succ) {
                    live_out.extend(set.iter().copied());
                }
            }
            let (used, defined) = &flow[&block];
            let mut new_live_in = used.clone();
            for vn in live_out {
                if !defined.contains(&vn) {
                    new_live_in.insert(vn);
                }
            }
            if new_live_in != live_in[&block] {
                live_in.insert(block, new_live_in);
                changed = true;
            }
        }
    }

    let mut inputs: HashSet<VarnodeId> = live_in[&root].iter().copied().collect();

    // mem2reg promotes load-before-store registers (the classic argument
    // registers) into root block params named after the register, removing the
    // loads the liveness above keys on. Recover those inputs from the params.
    for param in BasicBlock::from_id(ctx, root).params() {
        if let Some(name) = param.name()
            && let Some(vn) = ctx.get_named(name).and_then(|v| v.as_varnode())
            && is_register(ctx, vn)
        {
            inputs.insert(vn);
        }
    }

    let mut inputs: Vec<VarnodeId> = inputs.into_iter().collect();
    inputs.sort_by_key(|&vn| usize::from(vn));
    inputs
}

/// Registers read on entry and restored unchanged before exit — the
/// save/restore prologue/epilogue pattern (`push rbp` … `pop rbp`, lifted as
/// `*[register]:8 RBP = @RBP;`).
///
/// mem2reg surfaces such a register as a root-block param whose incoming value
/// is used *only* to be stored straight back to its own register. A saved
/// register is preserved across any call, so it is neither a real input nor a
/// clobber: marking it clobbered would wrongly tell callers their preserved copy
/// is destroyed. Returned sorted by [`VarnodeId`].
pub fn compute_saved_regs(ctx: &Context, function_id: FunctionId) -> Vec<VarnodeId> {
    let Some(root) = Function::from_id(ctx, function_id).root().map(|b| b.id) else {
        return Vec::new();
    };

    let mut saved: Vec<VarnodeId> = Vec::new();
    for param in BasicBlock::from_id(ctx, root).params() {
        let Some(name) = param.name() else { continue };
        let Some(vn) = ctx.get_named(name).and_then(|v| v.as_varnode()) else {
            continue;
        };
        if !is_register(ctx, vn) {
            continue;
        }
        let param_id = param.id();
        let users = ctx.users(param_id);
        // Restored unchanged: at least one use, and every use stores this exact
        // incoming value straight back to its own register.
        let restored_only = !users.is_empty()
            && users.iter().all(|&uid| {
                matches!(
                    ctx.get_insn(uid).mnemonic(),
                    Mnemonic::Store(store)
                        if store.ptr == ValueId::Varnode(vn) && store.src == param_id
                )
            });
        if restored_only {
            saved.push(vn);
        }
    }
    saved.sort_by_key(|&vn| usize::from(vn));
    saved
}

/// The net change a function applies to the stack pointer between entry and
/// return, derived from the final `RSP = @stack_base + N` write in each
/// return-terminated block (the offset `N` from [`STACK_BASE`]).
///
/// Returns `None` when no such write is found or the return blocks disagree, in
/// which case the stack pointer must be treated as an ordinary clobber. Must run
/// *before* the lower-stack pass rewrites `@stack_base` back into real RSP
/// arithmetic, while the `StackAddress`-typed literal store still exists.
pub fn compute_stack_delta(
    ctx: &Context,
    function_id: FunctionId,
    stack_ptr: VarnodeId,
) -> Option<i64> {
    let sp = ValueId::Varnode(stack_ptr);
    let mut delta: Option<i64> = None;

    for block in Function::from_id(ctx, function_id).iter() {
        // Only consider blocks that actually return.
        let returns = matches!(
            block.iter().last().map(|i| i.mnemonic().clone()),
            Some(Mnemonic::Return(_))
        );
        if !returns {
            continue;
        }

        // The last `RSP = <stack-address literal>` store reaching the return.
        let mut block_delta: Option<i64> = None;
        for insn in block.iter() {
            let Mnemonic::Store(store) = insn.mnemonic() else {
                continue;
            };
            if store.ptr != sp {
                continue;
            }
            let ValueId::Literal(lit) = store.src else {
                continue;
            };
            let lit = &ctx.values.literals[lit];
            if ctx.types.is_stack_address(lit.type_id) {
                let base = qcode::types::stack_base(ctx.types.size_of(lit.type_id));
                block_delta = Some(lit.value.wrapping_sub(base) as i64);
            }
        }

        match (delta, block_delta) {
            // A return block with no resolvable final RSP write: give up.
            (_, None) => return None,
            (None, Some(d)) => delta = Some(d),
            // Return blocks must agree on the net delta.
            (Some(a), Some(b)) if a != b => return None,
            _ => {}
        }
    }

    delta
}

/// Computes and stores `inputs` (register live-in), `clobbered` (registers
/// written), `saved` (registers preserved via save/restore), and `stack_delta`
/// (net stack-pointer change) on `function_id`'s signature. Saved registers are
/// excluded from both `inputs` and `clobbered`. When the stack delta is known,
/// the stack pointer is excluded from both as well: the delta models its effect
/// precisely, so it is neither a data argument nor a clobber.
pub fn set_function_summaries(ctx: &mut Context, function_id: FunctionId, stack_ptr: VarnodeId) {
    let stack_delta = compute_stack_delta(ctx, function_id, stack_ptr);

    let saved_set: HashSet<VarnodeId> = compute_saved_regs(ctx, function_id).into_iter().collect();
    let inputs: Vec<VarnodeId> = compute_input_regs(ctx, function_id)
        .into_iter()
        .filter(|vn| !saved_set.contains(vn))
        .filter(|&vn| stack_delta.is_none() || vn != stack_ptr)
        .collect();
    let clobbered: Vec<VarnodeId> = compute_clobbered_regs(ctx, function_id)
        .into_iter()
        .filter(|vn| !saved_set.contains(vn))
        .filter(|&vn| stack_delta.is_none() || vn != stack_ptr)
        .collect();
    let mut saved: Vec<VarnodeId> = saved_set.into_iter().collect();
    saved.sort_by_key(|&vn| usize::from(vn));

    let mut f = Function::from_id_mut(ctx, function_id);
    f.set_input_regs(inputs);
    f.set_clobbered_regs(clobbered);
    f.set_saved_regs(saved);
    if let Some(delta) = stack_delta {
        f.set_stack_delta(delta);
    }
}

/// Runs [`set_function_summaries`] over every non-external function.
pub fn set_all_function_summaries(ctx: &mut Context, stack_ptr: VarnodeId) {
    let ids: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| !f.is_external())
        .map(|f| f.id)
        .collect();
    for id in ids {
        set_function_summaries(ctx, id, stack_ptr);
    }
}

/// Byte-overlap of two register varnodes in the same space (e.g. RAX/EAX/AX/AL).
fn reg_overlap(ctx: &Context, a: VarnodeId, b: VarnodeId) -> bool {
    let (va, vb) = (Varnode::from_id(ctx, a), Varnode::from_id(ctx, b));
    va.space().id == vb.space().id
        && va.address() < vb.address() + vb.size() as i64
        && vb.address() < va.address() + va.size() as i64
}

/// The registers a *caller* must treat as clobbered across a call to
/// `function_id`: the registers it writes, minus those it reads before writing.
///
/// A read-before-written register is either a callee-saved register the function
/// pushes on entry and pops on exit — restored to its incoming value, so the
/// caller's copy survives — or an incoming argument register the caller itself
/// defines before the call. In both cases forwarding the caller's value across
/// the call stays correct, so such registers must not be reported as clobbered.
/// This is a verified property of the IR; it assumes no calling convention.
///
/// Unlike [`compute_clobbered_regs`] (raw writes) this is suitable for the
/// value-producing passes, which would otherwise mistake a callee's `push rbp` /
/// `pop rbp` for a real clobber and refuse to promote the caller's frame pointer.
pub fn compute_call_clobbered_regs(ctx: &Context, function_id: FunctionId) -> Vec<VarnodeId> {
    let inputs = compute_input_regs(ctx, function_id);
    compute_clobbered_regs(ctx, function_id)
        .into_iter()
        .filter(|&c| !inputs.iter().any(|&i| reg_overlap(ctx, c, i)))
        .collect()
}

/// Stores [`compute_call_clobbered_regs`] on every non-external function. Seeds
/// the clobber set the value-producing passes (mem2reg/GVN) consult before the
/// precise summaries exist; [`set_all_function_summaries`] later refines it.
pub fn set_all_call_clobbered_regs(ctx: &mut Context) {
    let ids: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| !f.is_external())
        .map(|f| f.id)
        .collect();
    for id in ids {
        let regs = compute_call_clobbered_regs(ctx, id);
        Function::from_id_mut(ctx, id).set_clobbered_regs(regs);
    }
}

// ---------------------------------------------------------------------------
// Stage 2: bind arguments + clobbers at call sites
// ---------------------------------------------------------------------------

/// Binds arguments and clobber sets on every direct `Call` in `function_id`.
///
/// For each input register the callee reads (see [`compute_input_regs`]), we
/// insert a `Load` of that register immediately before the call and use the
/// load's result as the argument. The argument is therefore always an SSA value;
/// a subsequent gvn/const-fold pass (see [`resolve_arg_loads`]) forwards each
/// load to the value actually reaching the call (a literal, a prior SSA def, or
/// the low bytes of a wider register write), handling sub-register overlap for
/// free. `Call.clobbers` records the callee's clobbered registers; pointer
/// arguments escape via the alias analysis's treatment of `Call.args`.
/// The continuation block reached when a returning call falls through — the CFG
/// successor `assume_call_returns` links onto the call block. A noreturn callee
/// has no such edge (and no continuation), so this returns `None`. Reading the
/// edge rather than the recorded assumption is robust to the call instruction
/// being renumbered by the PRE_BIND passes.
fn continuation_of(ctx: &Context, call_block: BlockId) -> Option<BlockId> {
    BasicBlock::from_id(ctx, call_block)
        .successors()
        .next()
        .map(|(_, b)| b)
}

/// Re-establish the stack pointer across a returning call by emitting
/// `RSP = RSP + delta` at the start of the call's continuation block. With the
/// callee's net stack delta modelled explicitly (and the stack pointer excluded
/// from its clobber set), the caller's RSP stays a tracked SSA chain instead of
/// being re-read from scratch after every call.
fn relink_stack_pointer(
    ctx: &mut Context,
    continuation: BlockId,
    stack_ptr: VarnodeId,
    delta: i64,
) {
    let (reg_space, ptr_width) = {
        let vn = Varnode::from_id(ctx, stack_ptr);
        (vn.space().id, vn.size())
    };
    let delta_const = ctx.get_const(delta.unsigned_abs(), ptr_width).id();

    let mut builder = Builder::from_block(BasicBlock::from_id_mut(ctx, continuation));
    builder.set_insert_point_to_start();
    unsafe { builder.dont_finalize() };

    let rsp = builder
        .push_load::<false>(ValueId::Varnode(stack_ptr), ptr_width, reg_space)
        .id();
    let adjusted = if delta >= 0 {
        builder.push_add(rsp, delta_const).id()
    } else {
        builder.push_sub(rsp, delta_const).id()
    };
    builder.push_store(adjusted, ValueId::Varnode(stack_ptr), reg_space);
}

/// Link the return address the caller pushes for this call to its continuation
/// block. The lifter pushes the raw fall-through address (`inst_next`); this
/// retags that pushed literal as `&<continuation>`, so the call/return linkage is
/// explicit in the data flow: the caller stores the continuation address into the
/// slot the callee returns through (its entry stack pointer). The literal keeps
/// its concrete value, so emulation is unaffected.
///
/// Returns the pointer (the SSA value addressing the push slot) of the
/// return-address store, or `None` if no such store was found. That slot address
/// *is* the callee's entry stack pointer, so the caller uses it to decrement the
/// stack-pointer register across the call (see [`decrement_stack_pointer`]).
fn link_return_address(
    ctx: &mut Context,
    call_block: BlockId,
    continuation: BlockId,
) -> Option<ValueId> {
    let cont_addr = BasicBlock::from_id(ctx, continuation).address()?;

    let insns: Vec<InstructionId> = BasicBlock::from_id(ctx, call_block)
        .instruction_ids()
        .to_vec();
    let mut slot: Option<ValueId> = None;
    for iid in insns {
        // A store of the continuation's address — the pushed return address.
        let (old_src, ptr, lid) = match ctx.get_insn(iid).mnemonic() {
            Mnemonic::Store(s) => match s.src {
                ValueId::Literal(l) => (s.src, s.ptr, l),
                _ => continue,
            },
            _ => continue,
        };
        let lit = &ctx.values.literals[lid];
        if lit.value != cont_addr || lit.symbolic.is_some() {
            continue;
        }
        let type_id = lit.type_id;
        let new_src = ValueId::Literal(ctx.values.push_literal(Literal {
            value: cont_addr,
            type_id,
            symbolic: Some(SymbolicRef::Block(continuation)),
        }));
        let mut mnemonic = ctx.get_insn(iid).mnemonic().clone();
        mnemonic.replace_value(old_src, new_src);
        ctx.replace_instruction_mnemonic(iid, mnemonic);
        slot.get_or_insert(ptr);
    }
    slot
}

/// Model the call's push of the return address as a decrement of the stack-pointer
/// *register*: emit `*[register]:N RSP = <slot>` immediately before the call, where
/// `slot` is the address the return address was pushed to (from
/// [`link_return_address`]). That slot *is* the callee's entry stack pointer, so
/// seeding the callee's `@RSP` entry param from the register file now lands on the
/// right slot.
///
/// Without this, the caller never writes the RSP register (its frame is addressed
/// through the `@stack_base`/`@RSP` chain), so a callee would load its return
/// address from the caller's *un-decremented* RSP and return to garbage.
fn decrement_stack_pointer(
    ctx: &mut Context,
    call_block: BlockId,
    call_id: InstructionId,
    stack_ptr: VarnodeId,
    slot: ValueId,
) {
    let reg_space = {
        let vn = Varnode::from_id(ctx, stack_ptr);
        vn.space().id
    };
    let mut builder = Builder::from_block(BasicBlock::from_id_mut(ctx, call_block));
    builder.set_insert_point_before(call_id);
    builder.push_store(slot, ValueId::Varnode(stack_ptr), reg_space);
}

pub fn bind_call_args(ctx: &mut Context, function_id: FunctionId, stack_ptr: VarnodeId) {
    let block_ids: Vec<BlockId> = Function::from_id(ctx, function_id)
        .iter()
        .map(|b| b.id)
        .collect();

    for block in block_ids {
        let Some(&call_id) = ctx.values.basic_blocks[block].instructions.last() else {
            continue;
        };
        let Mnemonic::Call(call) = ctx.get_insn(call_id).mnemonic() else {
            continue;
        };
        let target = call.target;

        // For a returning call: re-establish RSP across it when the callee's net
        // stack delta is known, link the pushed return address to the
        // continuation block, and decrement the stack-pointer register to the
        // push slot so the callee's seeded entry RSP matches it.
        if let Some(continuation) = continuation_of(ctx, block) {
            if let Some(delta) = Function::from_id(ctx, target).stack_delta() {
                relink_stack_pointer(ctx, continuation, stack_ptr, delta);
            }
            if let Some(slot) = link_return_address(ctx, block, continuation) {
                decrement_stack_pointer(ctx, block, call_id, stack_ptr, slot);
            }
        }

        let inputs: Vec<VarnodeId> = Function::from_id(ctx, target)
            .input_regs()
            .map(<[VarnodeId]>::to_vec)
            .unwrap_or_default();
        let clobbered: Vec<VarnodeId> = Function::from_id(ctx, target)
            .clobbered_regs()
            .map(<[VarnodeId]>::to_vec)
            .unwrap_or_default();

        let mut args = Vec::with_capacity(inputs.len());
        {
            let mut builder = Builder::from_block(BasicBlock::from_id_mut(ctx, block));
            builder.set_insert_point_before(call_id);
            for vn in inputs {
                let (space, size) = {
                    let v = Varnode::from_id(builder.context(), vn);
                    (v.space().id, v.size())
                };
                args.push(
                    builder
                        .push_load::<false>(ValueId::Varnode(vn), size, space)
                        .id(),
                );
            }
        }

        // Locations the callee may write/alias. Pointer arguments escape via the
        // alias analysis's handling of `Call.args`, so only the clobbered
        // registers are recorded here.
        let clobbers: Vec<ValueId> = clobbered.into_iter().map(ValueId::Varnode).collect();

        bind(ctx, call_id, args, clobbers);
    }
}

/// Forward the argument loads inserted by [`bind_call_args`] to the values
/// actually reaching each call, then drop the now-dead loads. Reuses gvn's
/// store→load forwarding (including wide-store/narrow-load sub-register
/// extraction) and constant folding.
pub fn resolve_arg_loads(ctx: &mut Context, function_id: FunctionId) {
    use crate::{AliasResult, constant_fold_function, gvn_function, remove_dead_insns};

    constant_fold_function(ctx, function_id);
    let aliases = AliasResult::simple(ctx);
    gvn_function(ctx, function_id, Some(&aliases));
    constant_fold_function(ctx, function_id);

    let block_ids: Vec<BlockId> = Function::from_id(ctx, function_id)
        .iter()
        .map(|b| b.id)
        .collect();
    for block in block_ids {
        remove_dead_insns(ctx, block);
    }
}

/// Write the resolved `args`/`clobbers` back onto the call mnemonic, preserving
/// use-def bookkeeping for the (read) arguments via `replace_instruction_mnemonic`.
fn bind(ctx: &mut Context, call_id: InstructionId, args: Vec<ValueId>, clobbers: Vec<ValueId>) {
    let Mnemonic::Call(call) = ctx.get_insn(call_id).mnemonic() else {
        return;
    };
    let new = Mnemonic::Call(qcode::value::insn::Call {
        target: call.target,
        args,
        clobbers,
    });
    ctx.replace_instruction_mnemonic(call_id, new);
}

/// Runs [`bind_call_args`] then [`resolve_arg_loads`] over every non-external
/// function: insert the argument loads at every call site, then forward them to
/// the values reaching each call.
pub fn bind_all_call_args(ctx: &mut Context, stack_ptr: VarnodeId) {
    let ids: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| !f.is_external())
        .map(|f| f.id)
        .collect();
    for &id in &ids {
        bind_call_args(ctx, id, stack_ptr);
    }
    for &id in &ids {
        resolve_arg_loads(ctx, id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AliasResult, alias::NodeId, alias_analysis};
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{Function, FunctionId, Value},
    };

    /// Build a function rooted at `addr` in `tc`, populated by `f`.
    fn build_fn(
        tc: &mut TestContext,
        name: &'static str,
        addr: u64,
        f: impl FnOnce(&mut Builder<'static, '_>),
    ) -> FunctionId {
        let fun_id = Function::make(&mut tc.ctx, name.into()).unwrap().id;
        let block_id = tc.ctx.get_or_make_block(addr);
        Function::from_id_mut(&mut tc.ctx, fun_id)
            .set_root(block_id)
            .unwrap();
        let mut builder = Builder::from_context(&mut tc.ctx, addr);
        f(&mut builder);
        unsafe { builder.dont_finalize() };
        drop(builder);
        fun_id
    }

    /// The first `Call` mnemonic found in `fun_id`.
    fn first_call(ctx: &Context, fun_id: FunctionId) -> Mnemonic {
        for block in Function::from_id(ctx, fun_id).iter() {
            for insn in block.iter() {
                if matches!(insn.mnemonic(), Mnemonic::Call(_)) {
                    return insn.mnemonic().clone();
                }
            }
        }
        panic!("no call found");
    }

    /// The rendered statement of the first `Call` in `fun_id`.
    fn first_call_text(ctx: &Context, fun_id: FunctionId) -> String {
        for block in Function::from_id(ctx, fun_id).iter() {
            for insn in block.iter() {
                if matches!(insn.mnemonic(), Mnemonic::Call(_)) {
                    return insn.as_statement().to_string();
                }
            }
        }
        panic!("no call found");
    }

    /// Insert argument loads at `caller`'s call sites and forward them.
    fn bind_and_resolve(ctx: &mut Context, caller: FunctionId, stack_ptr: VarnodeId) {
        bind_call_args(ctx, caller, stack_ptr);
        resolve_arg_loads(ctx, caller);
    }

    /// Build a callee rooted at `addr` that reads each of `reads` at `read_size`
    /// bytes (making them inputs) and a caller rooted at `0x1000` that runs
    /// `setup` then calls it. Binds and resolves the caller's arguments and
    /// returns its first `Call`.
    fn build_and_bind(
        tc: &mut TestContext,
        reads: &[VarnodeId],
        read_size: usize,
        addr: u64,
        setup: impl FnOnce(&mut Builder<'static, '_>),
    ) -> (FunctionId, Mnemonic) {
        let reg = tc.reg_space;
        let reads = reads.to_vec();
        let callee = build_fn(tc, "callee", addr, |b| {
            for &vn in &reads {
                b.push_load::<false>(ValueId::Varnode(vn), read_size, reg);
            }
        });
        let caller = build_fn(tc, "caller", 0x1000, |b| {
            setup(b);
            b.push_call(callee);
        });
        let sp = tc.r3;
        set_function_summaries(&mut tc.ctx, callee, sp);
        bind_and_resolve(&mut tc.ctx, caller, sp);
        let call = first_call(&tc.ctx, caller);
        (caller, call)
    }

    #[test]
    fn load_before_store_is_input() {
        let mut tc = TestContext::new();
        let r0 = tc.r0;
        let reg = tc.reg_space;
        let fun = build_fn(&mut tc, "callee", 0x1000, |b| {
            b.push_load::<false>(ValueId::Varnode(r0), 8, reg);
        });
        assert_eq!(compute_input_regs(&tc.ctx, fun), vec![r0]);
    }

    #[test]
    fn store_before_load_not_input() {
        let mut tc = TestContext::new();
        let r0 = tc.r0;
        let reg = tc.reg_space;
        let fun = build_fn(&mut tc, "callee", 0x1000, |b| {
            let v = b.context_mut().get_const(7u64, 8).id();
            b.push_store(v, ValueId::Varnode(r0), reg);
            b.push_load::<false>(ValueId::Varnode(r0), 8, reg);
        });
        assert!(
            compute_input_regs(&tc.ctx, fun).is_empty(),
            "r0 is written before being read, so it is not an input"
        );
    }

    #[test]
    fn promoted_register_param_is_input() {
        // A load-before-store register is promoted by mem2reg into a root block
        // param (named after the register), removing the load. compute_input_regs
        // must still recover it as an input from the param.
        let mut tc = TestContext::new();
        let r0 = tc.r0;
        let r1 = tc.r1;
        let reg = tc.reg_space;
        let fun = build_fn(&mut tc, "callee", 0x1000, |b| {
            let v = b.push_load::<false>(ValueId::Varnode(r0), 8, reg).id();
            b.push_store(v, ValueId::Varnode(r1), reg);
        });

        let aliases = crate::AliasResult::simple(&tc.ctx);
        crate::mem2reg(&mut tc.ctx, fun, &aliases);
        // The load is gone; r0 now flows in via a root param.
        assert!(
            compute_input_regs(&tc.ctx, fun).contains(&r0),
            "a register promoted to a root param must still be reported as input"
        );
    }

    #[test]
    fn saved_and_restored_register_is_not_input_or_clobber() {
        // The push rbp / pop rbp pattern: a register is read on entry and stored
        // straight back unchanged at exit. mem2reg surfaces it as a root param
        // whose only use is the restore store. It must be reported as `saved`, and
        // excluded from both `inputs` and `clobbered` — marking it clobbered would
        // wrongly tell callers their preserved copy is destroyed across the call.
        let mut tc = TestContext::new();
        let (r0, reg) = (tc.r0, tc.reg_space);
        let fun = build_fn(&mut tc, "callee", 0x1000, |b| {
            let v = b.push_load::<false>(ValueId::Varnode(r0), 8, reg).id();
            b.push_store(v, ValueId::Varnode(r0), reg); // restore r0 unchanged
        });

        let aliases = crate::AliasResult::simple(&tc.ctx);
        crate::mem2reg(&mut tc.ctx, fun, &aliases);
        assert!(
            compute_saved_regs(&tc.ctx, fun).contains(&r0),
            "a saved-and-restored register must be detected as saved"
        );

        set_function_summaries(&mut tc.ctx, fun, tc.r3);
        let f = Function::from_id(&tc.ctx, fun);
        assert!(f.saved_regs().unwrap().contains(&r0));
        assert!(
            !f.input_regs().unwrap().contains(&r0),
            "a saved register must not be reported as an input"
        );
        assert!(
            !f.clobbered_regs().unwrap().contains(&r0),
            "a saved register must not be reported as clobbered"
        );
    }

    #[test]
    fn register_written_with_new_value_is_clobber_not_saved() {
        // A register the function overwrites with a fresh value is a genuine
        // clobber, never a saved register.
        let mut tc = TestContext::new();
        let (r0, reg) = (tc.r0, tc.reg_space);
        let fun = build_fn(&mut tc, "callee", 0x1000, |b| {
            let v = b.context_mut().get_const(7u64, 8).id();
            b.push_store(v, ValueId::Varnode(r0), reg);
        });

        set_function_summaries(&mut tc.ctx, fun, tc.r3);
        let f = Function::from_id(&tc.ctx, fun);
        assert!(f.clobbered_regs().unwrap().contains(&r0));
        assert!(
            !f.saved_regs().unwrap().contains(&r0),
            "an overwritten register is a clobber, not a saved register"
        );
    }

    #[test]
    fn arg_resolves_to_reaching_store_value() {
        // Caller stores a constant into r0 before calling a callee that reads r0;
        // after binding+resolution the argument is that constant, not the register.
        let mut tc = TestContext::new();
        let (r0, reg) = (tc.r0, tc.reg_space);
        let mut stored = ValueId::Varnode(r0);
        let (_caller, call) = build_and_bind(&mut tc, &[r0], 8, 0x2000, |b| {
            stored = b.context_mut().get_const(123u64, 8).id();
            b.push_store(stored, ValueId::Varnode(r0), reg);
        });
        let Mnemonic::Call(call) = call else {
            unreachable!()
        };
        assert_eq!(call.args, vec![stored], "arg is the reaching store value");
    }

    #[test]
    fn args_are_ssa_values_never_varnodes() {
        // Even when nothing in the caller defines the register, the argument is an
        // SSA value (the inserted load instruction), never a bare varnode.
        let mut tc = TestContext::new();
        let (r0, _reg) = (tc.r0, tc.reg_space);
        let (_caller, call) = build_and_bind(&mut tc, &[r0], 8, 0x2000, |_b| {});
        let Mnemonic::Call(call) = call else {
            unreachable!()
        };
        assert_eq!(call.args.len(), 1);
        assert!(
            matches!(call.args[0], ValueId::Instruction(_)),
            "argument must be an SSA value, got {:?}",
            call.args[0]
        );
    }

    #[test]
    fn subregister_arg_resolves_to_literal() {
        // The hello/add case: the caller writes the full 8-byte r0 while the callee
        // reads the 4-byte sub-register r0_lo32. Resolution must forward the low
        // bytes of the stored literal, not leave a register reference.
        let mut tc = TestContext::new();
        let (r0, r0_lo32, reg) = (tc.r0, tc.r0_lo32, tc.reg_space);
        let (_caller, call) = build_and_bind(&mut tc, &[r0_lo32], 4, 0x2000, |b| {
            let v = b.context_mut().get_const(0x3u64, 8).id();
            b.push_store(v, ValueId::Varnode(r0), reg);
        });
        let Mnemonic::Call(call) = call else {
            unreachable!()
        };
        assert_eq!(call.args.len(), 1);
        let expected = tc.ctx.get_const(0x3u64, 4).id();
        assert_eq!(
            call.args[0], expected,
            "sub-register arg must resolve to the low bytes of the stored literal"
        );
    }

    #[test]
    fn args_follow_callee_input_order() {
        // Callee reads r0 and r1; inputs are sorted (r0, r1). The caller's args
        // must line up positionally with that order, each the reaching value.
        let mut tc = TestContext::new();
        let (r0, r1, reg) = (tc.r0, tc.r1, tc.reg_space);
        let (mut v0, mut v1) = (ValueId::Varnode(r0), ValueId::Varnode(r1));
        let (_caller, call) = build_and_bind(&mut tc, &[r0, r1], 8, 0x2000, |b| {
            v0 = b.context_mut().get_const(0xa0u64, 8).id();
            v1 = b.context_mut().get_const(0xb1u64, 8).id();
            b.push_store(v0, ValueId::Varnode(r0), reg);
            b.push_store(v1, ValueId::Varnode(r1), reg);
        });
        let Mnemonic::Call(call) = call else {
            unreachable!()
        };
        assert_eq!(call.args, vec![v0, v1], "args ordered by callee inputs");
    }

    #[test]
    fn latest_store_before_call_wins() {
        // Two stores to the same arg register: the reaching value is the last.
        let mut tc = TestContext::new();
        let (r0, reg) = (tc.r0, tc.reg_space);
        let mut v2 = ValueId::Varnode(r0);
        let (_caller, call) = build_and_bind(&mut tc, &[r0], 8, 0x2000, |b| {
            let v1 = b.context_mut().get_const(0x11u64, 8).id();
            b.push_store(v1, ValueId::Varnode(r0), reg);
            v2 = b.context_mut().get_const(0x22u64, 8).id();
            b.push_store(v2, ValueId::Varnode(r0), reg);
        });
        let Mnemonic::Call(call) = call else {
            unreachable!()
        };
        assert_eq!(call.args, vec![v2], "latest store to the arg register wins");
    }

    #[test]
    fn no_inputs_gives_empty_args() {
        // A callee that only writes a register has no inputs; calls take no args.
        let mut tc = TestContext::new();
        let (r0, r1, reg) = (tc.r0, tc.r1, tc.reg_space);
        let callee = build_fn(&mut tc, "callee", 0x2000, |b| {
            let v = b.context_mut().get_const(7u64, 8).id();
            b.push_store(v, ValueId::Varnode(r1), reg);
        });
        let caller = build_fn(&mut tc, "caller", 0x1000, |b| {
            let v = b.context_mut().get_const(1u64, 8).id();
            b.push_store(v, ValueId::Varnode(r0), reg);
            b.push_call(callee);
        });
        set_function_summaries(&mut tc.ctx, callee, tc.r3);
        bind_and_resolve(&mut tc.ctx, caller, tc.r3);
        let Mnemonic::Call(call) = first_call(&tc.ctx, caller) else {
            unreachable!()
        };
        assert!(call.args.is_empty(), "callee with no inputs takes no args");
    }

    #[test]
    fn reaching_value_resolves_across_blocks() {
        // The store to the arg register is in the entry block; the call is in a
        // dominated successor. Forwarding must reach the entry store.
        let mut tc = TestContext::new();
        let (r0, reg) = (tc.r0, tc.reg_space);

        let callee = build_fn(&mut tc, "callee", 0x2000, |b| {
            b.push_load::<false>(ValueId::Varnode(r0), 8, reg);
        });

        let caller = Function::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let bb2 = tc.ctx.get_or_make_block(0x1100);
        Function::from_id_mut(&mut tc.ctx, caller)
            .set_root(entry)
            .unwrap();
        Function::from_id_mut(&mut tc.ctx, caller).add_block(bb2);

        let stored;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            let v = b.context_mut().get_const(0x42u64, 8).id();
            stored = v;
            b.push_store(v, ValueId::Varnode(r0), reg);
            b.push_branch(bb2);
            b.switch_to_block(bb2);
            b.push_call(callee);
            unsafe { b.dont_finalize() };
        }

        set_function_summaries(&mut tc.ctx, callee, tc.r3);
        bind_and_resolve(&mut tc.ctx, caller, tc.r3);

        let Mnemonic::Call(call) = first_call(&tc.ctx, caller) else {
            unreachable!()
        };
        assert_eq!(
            call.args,
            vec![stored],
            "arg resolves to the dominating store in the entry block"
        );
    }

    #[test]
    fn display_shows_args_but_not_clobbers() {
        let mut tc = TestContext::new();
        let (r0, reg) = (tc.r0, tc.reg_space);
        let (caller, _) = build_and_bind(&mut tc, &[r0], 8, 0x2000, |b| {
            let v = b.context_mut().get_const(5u64, 8).id();
            b.push_store(v, ValueId::Varnode(r0), reg);
        });
        let text = first_call_text(&tc.ctx, caller);
        assert!(
            text.starts_with("call fn callee(") && text.ends_with(");"),
            "call renders its argument list: {text:?}"
        );
        assert!(
            !text.contains("clobber"),
            "the clobber set must not be rendered: {text:?}"
        );
    }

    #[test]
    fn bound_args_escape_in_alias_analysis() {
        let mut tc = TestContext::new();
        let (r0, reg) = (tc.r0, tc.reg_space);
        let mut stored = ValueId::Varnode(r0);
        let (_caller, _call) = build_and_bind(&mut tc, &[r0], 8, 0x2000, |b| {
            stored = b.context_mut().get_const(0x4000u64, 8).id();
            b.push_store(stored, ValueId::Varnode(r0), reg);
        });

        let result: AliasResult = alias_analysis(&tc.ctx);
        assert_eq!(
            result.alias_class(stored),
            Some(NodeId::Unknown),
            "a value passed as a call argument must escape to Unknown"
        );
    }

    #[test]
    fn clobbered_register_escapes_in_alias_analysis() {
        let mut tc = TestContext::new();
        let r0 = tc.r0;
        let r1 = tc.r1;
        let reg = tc.reg_space;

        // Callee writes r1 (clobbered) and reads nothing.
        let callee = build_fn(&mut tc, "callee", 0x2000, |b| {
            let v = b.context_mut().get_const(9u64, 8).id();
            b.push_store(v, ValueId::Varnode(r1), reg);
        });
        let caller = build_fn(&mut tc, "caller", 0x1000, |b| {
            let v = b.context_mut().get_const(1u64, 8).id();
            b.push_store(v, ValueId::Varnode(r0), reg);
            b.push_call(callee);
        });

        set_function_summaries(&mut tc.ctx, callee, tc.r3);
        bind_and_resolve(&mut tc.ctx, caller, tc.r3);

        let result = alias_analysis(&tc.ctx);
        assert_eq!(
            result.alias_class(ValueId::Varnode(r1)),
            Some(NodeId::Unknown),
            "a register clobbered by the callee must be Unknown after the call"
        );
    }

    // -----------------------------------------------------------------------
    // Stack delta + caller relink (Phase B)
    // -----------------------------------------------------------------------

    /// A `StackAddress`-typed literal for `STACK_BASE + offset`, mirroring the
    /// stack-slot literals brighten+mem2reg leave behind.
    fn stack_addr_lit(ctx: &mut Context, offset: i64) -> ValueId {
        let sa = ctx.types.get_or_make_stack_address(8, None);
        ctx.get_typed_const(qcode::types::STACK_BASE.wrapping_add(offset as u64), sa)
            .id()
    }

    /// Build a callee rooted at `addr` whose final `RSP = @stack_base+offset`
    /// write encodes a net stack delta of `offset`.
    fn build_callee_with_delta(
        tc: &mut TestContext,
        sp: VarnodeId,
        addr: u64,
        offset: i64,
    ) -> FunctionId {
        let reg = tc.reg_space;
        build_fn(tc, "callee", addr, |b| {
            let v = stack_addr_lit(b.context_mut(), offset);
            b.push_store(v, ValueId::Varnode(sp), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        })
    }

    #[test]
    fn stack_delta_extracted_from_final_rsp_store() {
        let mut tc = TestContext::new();
        let sp = tc.r3;
        let fun = build_callee_with_delta(&mut tc, sp, 0x1000, 8);
        assert_eq!(compute_stack_delta(&tc.ctx, fun, sp), Some(8));
    }

    #[test]
    fn sp_excluded_from_clobbers_when_delta_known() {
        let mut tc = TestContext::new();
        let sp = tc.r3;
        let fun = build_callee_with_delta(&mut tc, sp, 0x1000, 8);
        set_function_summaries(&mut tc.ctx, fun, sp);
        let f = Function::from_id(&tc.ctx, fun);
        assert_eq!(f.stack_delta(), Some(8));
        assert!(
            !f.clobbered_regs().unwrap().contains(&sp),
            "the stack pointer must be excluded from clobbers when its delta is known"
        );
        assert!(
            !f.input_regs().unwrap().contains(&sp),
            "the stack pointer is structural, not a data input"
        );
    }

    #[test]
    fn unknown_delta_keeps_rsp_clobbered() {
        // A plain-integer (non-stack-address) write to the stack pointer leaves
        // the delta unknown, so the stack pointer stays an ordinary clobber.
        let mut tc = TestContext::new();
        let (sp, reg) = (tc.r3, tc.reg_space);
        let fun = build_fn(&mut tc, "callee", 0x1000, |b| {
            let v = b.context_mut().get_const(0x10u64, 8).id();
            b.push_store(v, ValueId::Varnode(sp), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        });
        assert_eq!(compute_stack_delta(&tc.ctx, fun, sp), None);
        set_function_summaries(&mut tc.ctx, fun, sp);
        let f = Function::from_id(&tc.ctx, fun);
        assert!(f.stack_delta().is_none());
        assert!(f.clobbered_regs().unwrap().contains(&sp));
    }

    #[test]
    fn caller_emits_rsp_adjustment_in_continuation() {
        let mut tc = TestContext::new();
        let sp = tc.r3;
        let callee = build_callee_with_delta(&mut tc, sp, 0x2000, 8);
        set_function_summaries(&mut tc.ctx, callee, sp);

        // Caller whose entry block ends in the call, with a continuation block
        // linked by a CFG edge (as assume_call_returns would add).
        let caller = Function::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let cont = tc.ctx.get_or_make_block(0x1100);
        Function::from_id_mut(&mut tc.ctx, caller)
            .set_root(entry)
            .unwrap();
        Function::from_id_mut(&mut tc.ctx, caller).add_block(cont);
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            b.push_call(callee);
            b.switch_to_block(cont);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
            unsafe { b.dont_finalize() };
        }
        tc.ctx.add_cfg_edge(entry, cont);

        bind_call_args(&mut tc.ctx, caller, sp);

        // The continuation now begins with `RSP = RSP + 8`: load, add, store.
        let cont_block = BasicBlock::from_id(&tc.ctx, cont);
        let mut insns = cont_block.iter();
        let load = insns.next().expect("continuation has a leading load");
        assert!(
            matches!(load.mnemonic(), Mnemonic::Load(l) if l.ptr == ValueId::Varnode(sp)),
            "first instruction loads the stack pointer, got {load}"
        );
        let add = insns.next().expect("continuation has the add");
        let add_id = add.id();
        assert!(
            matches!(add.mnemonic(), Mnemonic::Binop(_)),
            "second instruction is the +delta, got {add}"
        );
        let store = insns.next().expect("continuation stores RSP back");
        assert!(
            matches!(store.mnemonic(), Mnemonic::Store(s) if s.ptr == ValueId::Varnode(sp) && s.src == add_id),
            "third instruction stores the adjusted value back to the stack pointer, got {store}"
        );
    }

    #[test]
    fn caller_links_return_address_to_continuation() {
        let mut tc = TestContext::new();
        let (sp, reg) = (tc.r3, tc.reg_space);
        let callee = build_callee_with_delta(&mut tc, sp, 0x2000, 8);
        set_function_summaries(&mut tc.ctx, callee, sp);

        let caller = Function::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let cont = tc.ctx.get_or_make_block(0x1100);
        Function::from_id_mut(&mut tc.ctx, caller)
            .set_root(entry)
            .unwrap();
        Function::from_id_mut(&mut tc.ctx, caller).add_block(cont);
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            // The lifter pushes the raw continuation address (0x1100) onto the stack.
            let retaddr = b.context_mut().get_const(0x1100u64, 8).id();
            b.push_store(retaddr, ValueId::Varnode(sp), reg);
            b.push_call(callee);
            b.switch_to_block(cont);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
            unsafe { b.dont_finalize() };
        }
        tc.ctx.add_cfg_edge(entry, cont);

        bind_call_args(&mut tc.ctx, caller, sp);

        // The pushed return address is retagged as a reference to the continuation.
        let linked = BasicBlock::from_id(&tc.ctx, entry).iter().any(|i| {
            matches!(i.mnemonic(), Mnemonic::Store(s)
                if matches!(s.src, ValueId::Literal(l)
                    if tc.ctx.values.literals[l].symbolic == Some(SymbolicRef::Block(cont))))
        });
        assert!(
            linked,
            "the pushed return address should be retagged as &<continuation>"
        );
    }

    /// The caller writes the return-address push slot into the stack-pointer
    /// register immediately before the call, so the callee's seeded entry stack
    /// pointer lands on that slot (and finds its return address there).
    #[test]
    fn caller_decrements_stack_pointer_to_push_slot() {
        let mut tc = TestContext::new();
        let sp = tc.r3;
        let ram = tc.ctx.default_space;
        let callee = build_callee_with_delta(&mut tc, sp, 0x2000, 8);
        set_function_summaries(&mut tc.ctx, callee, sp);

        let caller = Function::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let cont = tc.ctx.get_or_make_block(0x1100);
        Function::from_id_mut(&mut tc.ctx, caller)
            .set_root(entry)
            .unwrap();
        Function::from_id_mut(&mut tc.ctx, caller).add_block(cont);

        let slot;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            // Push the continuation address to the stack slot at @stack_base-8.
            slot = stack_addr_lit(b.context_mut(), -8);
            let retaddr = b.context_mut().get_const(0x1100u64, 8).id();
            b.push_store(retaddr, slot, ram);
            b.push_call(callee);
            b.switch_to_block(cont);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
            unsafe { b.dont_finalize() };
        }
        tc.ctx.add_cfg_edge(entry, cont);

        bind_call_args(&mut tc.ctx, caller, sp);

        let entry_block = BasicBlock::from_id(&tc.ctx, entry);
        let decremented = entry_block.iter().any(|i| {
            matches!(i.mnemonic(), Mnemonic::Store(s)
                if s.ptr == ValueId::Varnode(sp) && s.src == slot)
        });
        assert!(
            decremented,
            "caller must write the push slot into the SP register before the call:\n{entry_block}"
        );
    }

    /// Even when the callee's net stack delta is unknown, the caller still emits
    /// the pre-call stack-pointer decrement (so the callee can find its return
    /// address), but does *not* re-establish RSP in the continuation — the stack
    /// pointer stays an ordinary clobber the caller cannot reason about.
    #[test]
    fn pre_call_decrement_emitted_but_no_relink_when_delta_unknown() {
        let mut tc = TestContext::new();
        let (sp, reg) = (tc.r3, tc.reg_space);
        let ram = tc.ctx.default_space;
        // A plain-integer (non-stack-address) final RSP write leaves the delta
        // unknown, so the stack pointer stays a clobber.
        let callee = build_fn(&mut tc, "callee", 0x2000, |b| {
            let v = b.context_mut().get_const(0x10u64, 8).id();
            b.push_store(v, ValueId::Varnode(sp), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        });
        set_function_summaries(&mut tc.ctx, callee, sp);
        assert!(Function::from_id(&tc.ctx, callee).stack_delta().is_none());

        let caller = Function::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let cont = tc.ctx.get_or_make_block(0x1100);
        Function::from_id_mut(&mut tc.ctx, caller)
            .set_root(entry)
            .unwrap();
        Function::from_id_mut(&mut tc.ctx, caller).add_block(cont);

        let slot;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            slot = stack_addr_lit(b.context_mut(), -8);
            let retaddr = b.context_mut().get_const(0x1100u64, 8).id();
            b.push_store(retaddr, slot, ram);
            b.push_call(callee);
            b.switch_to_block(cont);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
            unsafe { b.dont_finalize() };
        }
        tc.ctx.add_cfg_edge(entry, cont);

        bind_call_args(&mut tc.ctx, caller, sp);

        // The decrement is still emitted.
        let entry_block = BasicBlock::from_id(&tc.ctx, entry);
        let decremented = entry_block.iter().any(|i| {
            matches!(i.mnemonic(), Mnemonic::Store(s)
                if s.ptr == ValueId::Varnode(sp) && s.src == slot)
        });
        assert!(
            decremented,
            "the pre-call decrement must be emitted even when the delta is unknown:\n{entry_block}"
        );

        // But the continuation has no RSP relink (no store back to the SP register).
        let cont_block = BasicBlock::from_id(&tc.ctx, cont);
        let relinked = cont_block
            .iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Store(s) if s.ptr == ValueId::Varnode(sp)));
        assert!(
            !relinked,
            "no RSP relink should be emitted when the callee's stack delta is unknown:\n{cont_block}"
        );
    }
}
