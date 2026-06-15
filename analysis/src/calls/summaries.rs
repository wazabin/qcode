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
    context::Context,
    space::SpaceType,
    value::{
        BasicBlock, BlockId, Function, FunctionId, Instruction, ValueId, Varnode, VarnodeId,
        insn::Mnemonic,
    },
};

use crate::{
    compute_clobbered_regs,
    mem::mem2reg::{has_dynamic_pointer_deref, stack_slot_offset},
};

/// The maximum caller-frame span (bytes above the return-address slot) over which
/// stack parameters are enumerated. A function reading beyond this is treated as
/// reading its incoming arguments unboundedly (e.g. a variadic walk) rather than
/// as having that many fixed parameters.
const MAX_STACK_PARAM_BYTES: i64 = 64;

/// True when `vn` is a stack-space varnode — how a stack-passed parameter is
/// recorded in a function's `inputs`, distinguishing it from a register input.
/// The `"stack"` space is created by `brighten_stack`.
pub(crate) fn is_stack_input(ctx: &Context, vn: VarnodeId) -> bool {
    ctx.try_get_space("stack") == Some(Varnode::from_id(ctx, vn).space().id)
}

/// Window around a stack base within which a bare integer is taken to be a frame
/// pointer. Comfortably larger than any plausible stack frame, yet far smaller
/// than the distance from a stack base to code/global/immediate constants.
const FRAME_POINTER_WINDOW: u64 = 0x0100_0000;

/// True when `v` addresses some function's stack frame.
///
/// A frame pointer keeps its `StackAddress` type while the symbolic stack base is
/// live, but constant folding can collapse `@stack_base ± k` to a bare integer
/// (losing the type) before this runs. So we accept either: an explicit
/// `StackAddress` value, or a constant within [`FRAME_POINTER_WINDOW`] of a
/// pointer-width stack base. Over-approximate by design — a false positive only
/// keeps a caller's frame in memory, which is always sound.
pub(crate) fn value_is_frame_pointer(ctx: &Context, v: ValueId) -> bool {
    match v {
        ValueId::Literal(id) => {
            let lit = &ctx.values.literals[id];
            if ctx.types.is_stack_address(lit.type_id) {
                return true;
            }
            [qcode::types::stack_base(4), qcode::types::stack_base(8)]
                .into_iter()
                .any(|base| lit.value.abs_diff(base) < FRAME_POINTER_WINDOW)
        }
        ValueId::Instruction(id) => ctx
            .types
            .is_stack_address(Instruction::from_id(ctx, id).type_id()),
        _ => false,
    }
}

/// Incoming stack parameters of `function_id` as `(offset, size)` pairs relative
/// to the entry stack pointer — offset `>= ptr_width`, above the return-address
/// slot at offset `0`. Recovered from the root-block params `mem2reg` created for
/// load-before-store caller-frame slots (keyed by their `origin` stack-slot
/// literal). Gaps are filled contiguously up to the highest accessed offset.
///
/// Returns `(slots, unbounded)`: when the accessed span exceeds
/// [`MAX_STACK_PARAM_BYTES`], `unbounded` is `true` and no slots are enumerated
/// (the function reads its arguments unboundedly).
fn compute_stack_inputs(ctx: &Context, function_id: FunctionId) -> (Vec<(i64, usize)>, bool) {
    let Some(root) = Function::from_id(ctx, function_id).root().map(|b| b.id) else {
        return (Vec::new(), false);
    };

    let mut real: Vec<(i64, usize)> = Vec::new();
    let mut ptr_width = 0usize;
    for param in BasicBlock::from_id(ctx, root).params() {
        let Some(origin) = param.origin() else {
            continue;
        };
        let Some((offset, pw)) = stack_slot_offset(ctx, origin) else {
            continue;
        };
        if offset < pw as i64 {
            continue;
        }
        ptr_width = pw;
        real.push((offset, param.size()));
    }
    if real.is_empty() {
        return (Vec::new(), false);
    }
    real.sort_by_key(|&(off, _)| off);

    let max_end = real
        .iter()
        .map(|&(off, size)| off + size as i64)
        .max()
        .expect("real is non-empty");
    if max_end - ptr_width as i64 > MAX_STACK_PARAM_BYTES {
        return (Vec::new(), true);
    }

    // Walk a contiguous grid from the first parameter slot to the highest
    // accessed offset, taking each real slot's size where one starts and assuming
    // a pointer-width slot otherwise, so the parameter list has no holes.
    let by_offset: HashMap<i64, usize> = real.into_iter().collect();
    let mut slots = Vec::new();
    let mut off = ptr_width as i64;
    while off < max_end {
        let size = by_offset.get(&off).copied().unwrap_or(ptr_width).max(1);
        slots.push((off, size));
        off += size as i64;
    }
    (slots, false)
}

/// True when `function_id` makes a call that could read a pointer argument
/// unboundedly: an indirect call (unknown target), or a direct call to an
/// external function or one already flagged [`Function::reads_unbounded_stack`].
/// Such a function may forward a caller-supplied pointer into that read, so it is
/// itself treated as an unbounded reader.
/// The absolute (per-function) stack address a frame-pointer literal `v` encodes,
/// or `None` if `v` is not one. Accepts an explicit `StackAddress` literal or a
/// bare constant near a stack base (constant folding can strip the type). Used to
/// read the caller-frame base a callee's stack arguments are measured from.
pub(crate) fn stack_address_literal_value(ctx: &Context, v: ValueId) -> Option<u64> {
    let ValueId::Literal(id) = v else {
        return None;
    };
    value_is_frame_pointer(ctx, v).then(|| ctx.values.literals[id].value)
}

fn function_makes_unbounded_call(ctx: &Context, function_id: FunctionId) -> bool {
    for block in Function::from_id(ctx, function_id).blocks() {
        for insn in block.iter() {
            match insn.mnemonic() {
                Mnemonic::CallInd(_) => return true,
                Mnemonic::Call(call) => {
                    let target = Function::from_id(ctx, call.target);
                    if target.is_external() || target.reads_unbounded_stack() {
                        return true;
                    }
                }
                _ => {}
            }
        }
    }
    false
}

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
    // Register inputs (existing path). The saved/stack-pointer exclusions apply
    // only to registers; stack-passed parameters are appended below untouched.
    let mut inputs: Vec<VarnodeId> = compute_input_regs(ctx, function_id)
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

    // Stack-passed parameters (cdecl / x86-64 overflow args).
    let (stack_slots, stack_unbounded) = compute_stack_inputs(ctx, function_id);

    // This function reads a passed pointer (or its own frame) unboundedly when it
    // has a dynamic stack access, enumerates more stack args than the cap, or
    // forwards into an unbounded/indirect/external call. Union with the seeded
    // value so the fact only grows across checkpoint+replay rounds.
    let reads_unbounded = Function::from_id(ctx, function_id).reads_unbounded_stack()
        || stack_unbounded
        || has_dynamic_pointer_deref(ctx, function_id)
        || function_makes_unbounded_call(ctx, function_id);

    // Append the stack parameters as stack-space varnodes (offset/size carriers,
    // distinguished from register inputs by their space). `bind_call_args`
    // translates each to the caller's frame.
    if let Some(stack_space) = ctx.try_get_space("stack") {
        for (offset, size) in stack_slots {
            let vn = Varnode::make(ctx, offset, size, stack_space).id;
            inputs.push(vn);
        }
    }

    let mut f = Function::from_id_mut(ctx, function_id);
    f.set_input_regs(inputs);
    f.set_clobbered_regs(clobbered);
    f.set_saved_regs(saved);
    f.set_reads_unbounded_stack(reads_unbounded);
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

#[cfg(test)]
mod tests {
    use super::super::binding::{bind_all_call_args, bind_call_args, resolve_arg_loads};
    use super::*;
    use crate::{AliasResult, alias::NodeId, alias_analysis};
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{Function, FunctionId, Value, literal::SymbolicRef},
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
            text.contains("@r0=0x5"),
            "call renders named arguments: {text:?}"
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

    // -----------------------------------------------------------------------
    // Stack-passed parameters + frame-escape safety
    // -----------------------------------------------------------------------

    /// The stack inputs of `fun`, as `(offset, size)` pairs.
    fn stack_inputs_of(ctx: &Context, fun: FunctionId) -> Vec<(i64, usize)> {
        Function::from_id(ctx, fun)
            .input_regs()
            .unwrap_or_default()
            .iter()
            .filter(|&&vn| is_stack_input(ctx, vn))
            .map(|&vn| {
                let v = Varnode::from_id(ctx, vn);
                (v.address(), v.size())
            })
            .collect()
    }

    #[test]
    fn caller_frame_slot_read_before_write_is_a_stack_input() {
        // A callee that loads [entry_sp + 8] (above the return-address slot) before
        // writing it reads a stack-passed parameter. After mem2reg promotes the
        // load-only slot, the summary records it as a stack-space varnode input.
        let mut tc = TestContext::new();
        let sp = tc.r3;
        let ram = tc.ctx.default_space;
        crate::stack::brighten::get_or_make_stack_space(&mut tc.ctx, 8);

        let callee = build_fn(&mut tc, "callee", 0x2000, |b| {
            let slot = stack_addr_lit(b.context_mut(), 8);
            b.push_load::<false>(slot, 8, ram);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        });
        let aliases = AliasResult::simple(&tc.ctx);
        crate::mem2reg(&mut tc.ctx, callee, &aliases);
        set_function_summaries(&mut tc.ctx, callee, sp);

        assert_eq!(
            stack_inputs_of(&tc.ctx, callee),
            vec![(8, 8)],
            "the load-before-store caller-frame slot must be a stack input"
        );
    }

    #[test]
    fn local_slot_below_entry_sp_is_not_a_stack_input() {
        // A slot at a negative offset is a local, not an incoming parameter.
        let mut tc = TestContext::new();
        let sp = tc.r3;
        let ram = tc.ctx.default_space;
        crate::stack::brighten::get_or_make_stack_space(&mut tc.ctx, 8);

        let callee = build_fn(&mut tc, "callee", 0x2000, |b| {
            let slot = stack_addr_lit(b.context_mut(), -8);
            b.push_load::<false>(slot, 8, ram);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        });
        let aliases = AliasResult::simple(&tc.ctx);
        crate::mem2reg(&mut tc.ctx, callee, &aliases);
        set_function_summaries(&mut tc.ctx, callee, sp);

        assert!(
            stack_inputs_of(&tc.ctx, callee).is_empty(),
            "a slot below the entry stack pointer is a local, not a parameter"
        );
    }

    #[test]
    fn stack_param_gaps_are_filled_contiguously() {
        // Reads at +8 and +24 (a hole at +16) yield a contiguous 3-slot list.
        let mut tc = TestContext::new();
        let sp = tc.r3;
        let ram = tc.ctx.default_space;
        crate::stack::brighten::get_or_make_stack_space(&mut tc.ctx, 8);

        let callee = build_fn(&mut tc, "callee", 0x2000, |b| {
            let s1 = stack_addr_lit(b.context_mut(), 8);
            b.push_load::<false>(s1, 8, ram);
            let s2 = stack_addr_lit(b.context_mut(), 24);
            b.push_load::<false>(s2, 8, ram);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        });
        let aliases = AliasResult::simple(&tc.ctx);
        crate::mem2reg(&mut tc.ctx, callee, &aliases);
        set_function_summaries(&mut tc.ctx, callee, sp);

        assert_eq!(
            stack_inputs_of(&tc.ctx, callee),
            vec![(8, 8), (16, 8), (24, 8)],
            "the gap at +16 must be filled so the parameter list is contiguous"
        );
    }

    #[test]
    fn far_stack_read_is_unbounded_not_enumerated() {
        // A read far above the frame exceeds the cap: treated as an unbounded
        // argument read, not hundreds of fixed parameters.
        let mut tc = TestContext::new();
        let sp = tc.r3;
        let ram = tc.ctx.default_space;
        crate::stack::brighten::get_or_make_stack_space(&mut tc.ctx, 8);

        let callee = build_fn(&mut tc, "callee", 0x2000, |b| {
            let slot = stack_addr_lit(b.context_mut(), MAX_STACK_PARAM_BYTES + 64);
            b.push_load::<false>(slot, 8, ram);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        });
        let aliases = AliasResult::simple(&tc.ctx);
        crate::mem2reg(&mut tc.ctx, callee, &aliases);
        set_function_summaries(&mut tc.ctx, callee, sp);

        assert!(
            stack_inputs_of(&tc.ctx, callee).is_empty(),
            "a read beyond the cap must not enumerate stack parameters"
        );
        assert!(
            Function::from_id(&tc.ctx, callee).reads_unbounded_stack(),
            "a read beyond the cap must flag the function as an unbounded reader"
        );
    }

    #[test]
    fn stack_argument_binds_to_caller_push() {
        // End-to-end: a callee reads its first stack parameter at [entry_sp + 8];
        // the caller stores the argument at the matching frame slot and calls.
        // Binding must resolve the argument to that stored value.
        let mut tc = TestContext::new();
        let sp = tc.r3;
        let ram = tc.ctx.default_space;
        crate::stack::brighten::get_or_make_stack_space(&mut tc.ctx, 8);

        let callee = build_fn(&mut tc, "callee", 0x2000, |b| {
            let slot = stack_addr_lit(b.context_mut(), 8);
            b.push_load::<false>(slot, 8, ram);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        });
        let aliases = AliasResult::simple(&tc.ctx);
        crate::mem2reg(&mut tc.ctx, callee, &aliases);
        set_function_summaries(&mut tc.ctx, callee, sp);

        let caller = Function::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let cont = tc.ctx.get_or_make_block(0x1100);
        Function::from_id_mut(&mut tc.ctx, caller)
            .set_root(entry)
            .unwrap();
        Function::from_id_mut(&mut tc.ctx, caller).add_block(cont);

        let arg_val;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            // Return address pushed at slot -8; this is the callee's entry SP, so
            // the parameter at callee offset +8 lives at caller slot 0.
            let arg_slot = stack_addr_lit(b.context_mut(), 0);
            arg_val = b.context_mut().get_const(0xABCDu64, 8).id();
            b.push_store(arg_val, arg_slot, ram);
            let push_slot = stack_addr_lit(b.context_mut(), -8);
            let retaddr = b.context_mut().get_const(0x1100u64, 8).id();
            b.push_store(retaddr, push_slot, ram);
            b.push_call(callee);
            b.switch_to_block(cont);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
            unsafe { b.dont_finalize() };
        }
        tc.ctx.add_cfg_edge(entry, cont);

        bind_and_resolve(&mut tc.ctx, caller, sp);

        let Mnemonic::Call(call) = first_call(&tc.ctx, caller) else {
            unreachable!()
        };
        assert_eq!(
            call.args,
            vec![arg_val],
            "the stack argument must resolve to the caller's pushed value"
        );
    }

    #[test]
    fn frame_escape_flag_disables_stack_promotion() {
        // A normally-promotable local (stored then loaded) must stay in memory once
        // the function is flagged as letting a frame pointer escape unboundedly.
        let mut tc = TestContext::new();
        let ram = tc.ctx.default_space;
        crate::stack::brighten::get_or_make_stack_space(&mut tc.ctx, 8);

        let stack_load_count = |ctx: &Context, fun: FunctionId| {
            Function::from_id(ctx, fun)
                .blocks()
                .flat_map(|b| b.iter().collect::<Vec<_>>())
                .filter(|i| {
                    matches!(i.mnemonic(), Mnemonic::Load(l) if stack_address_literal_value(ctx, l.ptr).is_some())
                })
                .count()
        };

        let build_local_fn = |tc: &mut TestContext, name: &'static str, addr: u64| {
            build_fn(tc, name, addr, |b| {
                let slot = stack_addr_lit(b.context_mut(), -8);
                let v = b.context_mut().get_const(7u64, 8).id();
                b.push_store(v, slot, ram);
                let loaded = b.push_load::<false>(slot, 8, ram).id();
                let sink_slot = stack_addr_lit(b.context_mut(), -16);
                b.push_store(loaded, sink_slot, ram);
                let sink = b.context_mut().get_const(0u64, 8).id();
                b.push_return(sink);
            })
        };

        // Baseline: the local is promoted away (no stack loads remain).
        let promoted = build_local_fn(&mut tc, "promoted", 0x1000);
        let aliases = AliasResult::simple(&tc.ctx);
        crate::mem2reg(&mut tc.ctx, promoted, &aliases);
        assert_eq!(
            stack_load_count(&tc.ctx, promoted),
            0,
            "without the flag, the local stack slot is promoted to SSA"
        );

        // Flagged: promotion is disabled and the load survives.
        let escaping = build_local_fn(&mut tc, "escaping", 0x2000);
        Function::from_id_mut(&mut tc.ctx, escaping).set_frame_escapes_to_unbounded(true);
        let aliases = AliasResult::simple(&tc.ctx);
        crate::mem2reg(&mut tc.ctx, escaping, &aliases);
        assert!(
            stack_load_count(&tc.ctx, escaping) > 0,
            "a frame-escaping function must keep its stack frame in memory"
        );
    }

    #[test]
    fn passing_frame_pointer_to_external_flags_caller() {
        // Passing &local to an external callee (conservatively unbounded) flags the
        // caller as frame-escaping, so its frame will not be promoted next round.
        let mut tc = TestContext::new();
        let sp = tc.r3;
        let reg = tc.reg_space;
        let r0 = tc.r0;
        crate::stack::brighten::get_or_make_stack_space(&mut tc.ctx, 8);

        // External callee taking one pointer argument in r0.
        let ext = Function::make(&mut tc.ctx, "gets".into()).unwrap().id;
        Function::from_id_mut(&mut tc.ctx, ext).set_external(true);
        Function::from_id_mut(&mut tc.ctx, ext).set_input_regs(vec![r0]);

        let caller = build_fn(&mut tc, "caller", 0x1000, |b| {
            // r0 = &local; call gets(r0)
            let local = stack_addr_lit(b.context_mut(), -8);
            b.push_store(local, ValueId::Varnode(r0), reg);
            b.push_call(ext);
        });

        bind_all_call_args(&mut tc.ctx, sp);

        assert!(
            Function::from_id(&tc.ctx, caller).frame_escapes_to_unbounded(),
            "passing a frame pointer to an external must flag the caller as escaping"
        );
    }
}

// ----- passes ----------------------------------------------------------------

use crate::{Pass, PipelineEnv};

#[derive(Default)]
pub struct SeedClobbers;

impl Pass for SeedClobbers {
    const NAME: &'static str = "seed_clobbers";
    fn description(&self) -> &'static str {
        "Seed each function's call-clobbered-register set from the lifted IR"
    }
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        set_all_call_clobbered_regs(ctx);
        Ok(false)
    }
}

crate::register_module_pass!(SeedClobbers);

#[derive(Default)]
pub struct Summaries;

impl Pass for Summaries {
    const NAME: &'static str = "summaries";
    fn description(&self) -> &'static str {
        "Infer each function's input/clobber/saved summary and stack delta"
    }
    fn run(&self, ctx: &mut Context, env: &PipelineEnv) -> Result<bool, String> {
        set_all_function_summaries(ctx, env.sp_varnode);
        Ok(false)
    }
}

crate::register_module_pass!(Summaries);
