//! Stage 2 of call analysis: binding arguments and clobber sets at each direct
//! call site, after mem2reg. See [`super::summaries`] for the callee summaries
//! these consume.

use qcode::{
    builder::Builder,
    context::Context,
    value::{
        BasicBlock, BlockId, Function, FunctionId, Value, ValueId, Varnode, VarnodeId,
        insn::{InstructionId, Mnemonic},
        literal::{Literal, SymbolicRef},
    },
};

use super::summaries::{is_stack_input, stack_address_literal_value, value_is_frame_pointer};
use crate::{Pass, PipelineEnv};

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
    // SAFETY: `continuation` is an existing block that already ends in a
    // terminator; we only prepend instructions, so the builder's
    // must-terminate-before-drop check does not apply.
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
        // push slot so the callee's seeded entry RSP matches it. The push slot is
        // the callee's entry stack pointer in the *caller's* coordinates — the
        // base its stack arguments are measured from.
        let mut frame_base: Option<u64> = None;
        if let Some(continuation) = continuation_of(ctx, block) {
            if let Some(delta) = Function::from_id(ctx, target).stack_delta() {
                relink_stack_pointer(ctx, continuation, stack_ptr, delta);
            }
            if let Some(slot) = link_return_address(ctx, block, continuation) {
                frame_base = stack_address_literal_value(ctx, slot);
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

        // Resolve each input to how it loads in the caller: a register input
        // reloads the register; a stack input (a stack-space varnode at callee
        // offset N) reads the caller's frame at `frame_base + N`.
        let ptr_width = Varnode::from_id(ctx, stack_ptr).size();
        let default_space = ctx.default_space;
        let stack_space = ctx.try_get_space("stack");
        enum ArgLoad {
            Reg(VarnodeId),
            Stack { addr: u64, size: usize },
        }
        let plans: Vec<ArgLoad> = inputs
            .iter()
            .map(|&vn| {
                if is_stack_input(ctx, vn) {
                    let v = Varnode::from_id(ctx, vn);
                    let (offset, size) = (v.address(), v.size());
                    match frame_base {
                        // Translate the callee offset into the caller's frame.
                        Some(base) => ArgLoad::Stack {
                            addr: base.wrapping_add(offset as u64),
                            size,
                        },
                        // No resolvable push slot: fall back to a bare load that
                        // keeps the argument list aligned with the callee inputs.
                        None => ArgLoad::Reg(vn),
                    }
                } else {
                    ArgLoad::Reg(vn)
                }
            })
            .collect();

        let mut args = Vec::with_capacity(plans.len());
        {
            let mut builder = Builder::from_block(BasicBlock::from_id_mut(ctx, block));
            builder.set_insert_point_before(call_id);
            for plan in plans {
                let value = match plan {
                    ArgLoad::Reg(vn) => {
                        let (space, size) = {
                            let v = Varnode::from_id(builder.context(), vn);
                            (v.space().id, v.size())
                        };
                        builder
                            .push_load::<false>(ValueId::Varnode(vn), size, space)
                            .id()
                    }
                    ArgLoad::Stack { addr, size } => {
                        let ptr = {
                            let sa = builder
                                .context_mut()
                                .types
                                .get_or_make_stack_address(ptr_width, stack_space);
                            builder.context_mut().get_typed_const(addr, sa).id()
                        };
                        builder.push_load::<false>(ptr, size, default_space).id()
                    }
                };
                args.push(value);
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
    // Only this function's pointers are forwarded here, so build the alias set
    // from its own instructions instead of re-scanning the whole program once
    // per function (resolve_arg_loads runs per function across the program).
    let aliases = AliasResult::simple_for_function(ctx, function_id);
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

/// Flag `function_id` when it hands a pointer into its own frame to a callee that
/// may read it unboundedly (see [`FunctionSignature::frame_escapes_to_unbounded`]).
///
/// Runs *after* [`resolve_arg_loads`], so a frame pointer passed in a register or
/// a stack slot has been forwarded into `Call.args` as a `StackAddress` value:
///   - a direct call carrying a `StackAddress` argument to an external or
///     already-unbounded callee, or
///   - an indirect call (whose args we do not bind) in a function that pushes a
///     `StackAddress` into a stack slot (a cdecl stack argument).
///
/// The fact is monotonic — once set it is never cleared — so it converges under
/// the checkpoint+replay driver, which seeds it back onto the signature each
/// round for `mem2reg` to consume.
///
/// [`FunctionSignature::frame_escapes_to_unbounded`]:
/// qcode::value::function::FunctionSignature::frame_escapes_to_unbounded
fn mark_frame_escapes(ctx: &mut Context, function_id: FunctionId) {
    if Function::from_id(ctx, function_id).frame_escapes_to_unbounded() {
        return;
    }

    let block_ids: Vec<BlockId> = Function::from_id(ctx, function_id)
        .iter()
        .map(|b| b.id)
        .collect();

    let mut has_callind = false;
    let mut pushes_frame_ptr_to_slot = false;
    let mut escapes = false;

    for block in &block_ids {
        for insn in BasicBlock::from_id(ctx, *block).iter() {
            match insn.mnemonic() {
                Mnemonic::CallInd(_) => has_callind = true,
                Mnemonic::Call(call) => {
                    let target = Function::from_id(ctx, call.target);
                    let unbounded = target.is_external() || target.reads_unbounded_stack();
                    if unbounded && call.args.iter().any(|&a| value_is_frame_pointer(ctx, a)) {
                        escapes = true;
                    }
                }
                // A frame pointer written into a stack slot is an outgoing stack
                // argument (the cdecl / indirect-call case, where args are not
                // bound). Pair it with a `CallInd` below.
                Mnemonic::Store(store)
                    if value_is_frame_pointer(ctx, store.src)
                        && stack_address_literal_value(ctx, store.ptr).is_some() =>
                {
                    pushes_frame_ptr_to_slot = true;
                }
                _ => {}
            }
        }
    }

    if escapes || (has_callind && pushes_frame_ptr_to_slot) {
        Function::from_id_mut(ctx, function_id).set_frame_escapes_to_unbounded(true);
    }
}

/// Runs [`bind_call_args`] then [`resolve_arg_loads`] over every non-external
/// function: insert the argument loads at every call site, then forward them to
/// the values reaching each call. Finally records which functions let a frame
/// pointer escape into an unbounded-reading callee (see [`mark_frame_escapes`]).
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
    for &id in &ids {
        mark_frame_escapes(ctx, id);
    }
}

// ----- pass ------------------------------------------------------------------

#[derive(Default)]
pub struct BindArgs;

impl Pass for BindArgs {
    const NAME: &'static str = "bind_args";
    fn description(&self) -> &'static str {
        "Bind argument and per-call alias sets at every call site"
    }
    fn run(&self, ctx: &mut Context, env: &PipelineEnv) -> Result<bool, String> {
        bind_all_call_args(ctx, env.sp_varnode);
        Ok(false)
    }
}

crate::register_module_pass!(BindArgs);
