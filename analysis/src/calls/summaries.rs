//! Architecture-independent call summaries: callee input/clobber inference.
//!
//! Everything here is derived from the IR itself — there are no calling-
//! convention tables. [`set_function_summaries`] fills each function's
//! [`FunctionSignature`] with the registers it reads before writing (`inputs`, a
//! backward register-liveness) and the registers it writes (`clobbered`, reusing
//! [`compute_clobbered_regs`]). Run it over every function first so callee
//! summaries exist before callers consume them.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    context::Context,
    space::SpaceType,
    value::{
        BasicBlock, BlockId, BlockParam, Function, FunctionId, ValueId, Varnode,
        VarnodeId,
        insn::{Binop, IntBinop, Mnemonic},
    },
};
// Only the test-gated frame-pointer helpers below resolve an instruction's type.
#[cfg(test)]
use qcode::value::Instruction;

use crate::{
    compute_clobbered_regs,
    mem::mem2reg::{has_dynamic_stack_pointer_deref, stack_slot_offset},
};

/// Window around a stack base within which a bare integer is taken to be a frame
/// pointer. Comfortably larger than any plausible stack frame, yet far smaller
/// than the distance from a stack base to code/global/immediate constants.
#[cfg(test)]
const FRAME_POINTER_WINDOW: u64 = 0x0100_0000;

/// True when `v` addresses some function's stack frame.
///
/// A frame pointer keeps its `StackAddress` type while the symbolic stack base is
/// live, but constant folding can collapse `@stack_base ± k` to a bare integer
/// (losing the type) before this runs. So we accept either: an explicit
/// `StackAddress` value, or a constant within [`FRAME_POINTER_WINDOW`] of a
/// pointer-width stack base. Over-approximate by design — a false positive only
/// keeps a caller's frame in memory, which is always sound.
#[cfg(test)]
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

/// True when `function_id` makes a call that could read a pointer argument
/// unboundedly: an indirect call (unknown target), or a direct call to an
/// external function or one already flagged [`Function::reads_unbounded_stack`].
/// Such a function may forward a caller-supplied pointer into that read, so it is
/// itself treated as an unbounded reader.
/// The absolute (per-function) stack address a frame-pointer literal `v` encodes,
/// or `None` if `v` is not one. Accepts an explicit `StackAddress` literal or a
/// bare constant near a stack base (constant folding can strip the type). Used to
/// read the caller-frame base a callee's stack arguments are measured from.
#[cfg(test)]
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

/// True when `function_id` *writes* through a stack-passed pointer parameter at
/// an unbounded offset — e.g. walking the pointer across a loop. Such a store
/// mutates memory the *caller* owns at an offset this function does not bound, so
/// a caller that passes a pointer into its own frame must keep that frame in
/// memory. A bounded write (`*p` or `*(p + const)`) or a mere *read* through such
/// a pointer is not enough — only an unbounded write counts (see the unit tests).
fn function_writes_through_stack_arg(ctx: &Context, function_id: FunctionId) -> bool {
    let ram = ctx.default_space;
    for block in Function::from_id(ctx, function_id).blocks() {
        for insn in block.iter() {
            if let Mnemonic::Store(s) = insn.mnemonic()
                && s.space == ram
            {
                let mut visited = HashSet::default();
                let (derives, dynamic) = trace_stack_arg_pointer(ctx, s.ptr, &mut visited);
                if derives && dynamic {
                    return true;
                }
            }
        }
    }
    false
}

/// Trace `v` back toward a stack-passed pointer parameter. Returns
/// `(derives, dynamic)`: whether `v` derives from such a parameter, and whether
/// the path to it carries a non-constant (loop- or index-driven) offset.
fn trace_stack_arg_pointer(
    ctx: &Context,
    v: ValueId,
    visited: &mut HashSet<ValueId>,
) -> (bool, bool) {
    if !visited.insert(v) {
        return (false, false);
    }
    match v {
        ValueId::BlockParam(pid) => {
            let param = BlockParam::from_id(ctx, pid);
            // A stack-input parameter: promoted from a caller-frame slot at or
            // above the return-address slot. This is the static pointer base.
            if let Some(origin) = param.origin()
                && let Some((offset, ptr_width)) = stack_slot_offset(ctx, origin)
                && offset >= ptr_width as i64
            {
                return (true, false);
            }
            // A merge/loop-carried parameter: follow its incoming values. Reaching
            // a stack arg through such a param means the pointer varies (the
            // induction pointer of a loop), which is the unbounded case.
            let Some(block) = param.parent().map(|b| b.id) else {
                return (false, false);
            };
            let index = param.index();
            let derives = incoming_values(ctx, block, index)
                .into_iter()
                .any(|incoming| trace_stack_arg_pointer(ctx, incoming, visited).0);
            (derives, derives)
        }
        ValueId::Instruction(iid) => match ctx.get_insn(iid).mnemonic().clone() {
            Mnemonic::Binop(b) if matches!(b.op, Binop::Int(IntBinop::Add | IntBinop::Sub)) => {
                let (ld, ldyn) = trace_stack_arg_pointer(ctx, b.lhs, visited);
                let (rd, rdyn) = trace_stack_arg_pointer(ctx, b.rhs, visited);
                let derives = ld || rd;
                // The operand that does not derive from the pointer is the offset;
                // a non-constant offset makes the access unbounded.
                let mut dynamic = ldyn || rdyn;
                if ld && !matches!(b.rhs, ValueId::Literal(_)) {
                    dynamic = true;
                }
                if rd && !matches!(b.lhs, ValueId::Literal(_)) {
                    dynamic = true;
                }
                (derives, derives && dynamic)
            }
            _ => (false, false),
        },
        _ => (false, false),
    }
}

/// The values flowing into parameter `index` of `block` from its predecessors'
/// branch arguments.
fn incoming_values(ctx: &Context, block: BlockId, index: usize) -> Vec<ValueId> {
    let preds: Vec<BlockId> = BasicBlock::from_id(ctx, block)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    let mut out = Vec::new();
    for pred in preds {
        let Some(term) = BasicBlock::from_id(ctx, pred).iter().last() else {
            continue;
        };
        match term.mnemonic() {
            Mnemonic::Branch(br) if br.target == block => {
                if let Some(&a) = br.args.get(index) {
                    out.push(a);
                }
            }
            Mnemonic::CBranch(cb) => {
                if cb.success_block == block
                    && let Some(&a) = cb.success_args.get(index)
                {
                    out.push(a);
                }
                if cb.failure_block == block
                    && let Some(&a) = cb.failure_args.get(index)
                {
                    out.push(a);
                }
            }
            _ => {}
        }
    }
    out
}

/// True when `vn` lives in a register address space.
fn is_register(ctx: &Context, vn: VarnodeId) -> bool {
    matches!(Varnode::from_id(ctx, vn).space().ty, SpaceType::Register)
}

/// Upward-exposed register reads (`used`) and register writes (`defined`) of a
/// single block, computed by a forward scan.
fn block_reg_flow(ctx: &Context, block: BlockId) -> (HashSet<VarnodeId>, HashSet<VarnodeId>) {
    let mut used = HashSet::default();
    let mut defined = HashSet::default();
    // Registers already written earlier in this block; a later read of one of
    // these is satisfied locally and is not upward-exposed.
    let mut written: HashSet<VarnodeId> = HashSet::default();

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
        blocks.iter().map(|&b| (b, HashSet::default())).collect();

    // Predecessor map: a block's live-in feeds its predecessors' live-out, so a
    // change only needs the predecessors recomputed.
    let mut preds: HashMap<BlockId, Vec<BlockId>> =
        blocks.iter().map(|&b| (b, Vec::new())).collect();
    for &block in &blocks {
        for &succ in &succs[&block] {
            if let Some(entry) = preds.get_mut(&succ) {
                entry.push(block);
            }
        }
    }

    // Worklist fixpoint: recompute a block only when a successor's live-in
    // changed, rather than re-sweeping every block to stability — same monotone
    // equations and same fixpoint, but no O(B²) sweep on large functions.
    let mut worklist: Vec<BlockId> = blocks.clone();
    let mut queued: HashSet<BlockId> = blocks.iter().copied().collect();
    while let Some(block) = worklist.pop() {
        queued.remove(&block);
        let mut live_out: HashSet<VarnodeId> = HashSet::default();
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
            for &pred in &preds[&block] {
                if queued.insert(pred) {
                    worklist.push(pred);
                }
            }
        }
    }

    // `root` should be one of the function's own blocks, but a boundary-splitting
    // reattribution can leave a stub function's root pointing outside its block
    // set. Fall back to an empty live-in rather than panicking; the param sweep
    // below still recovers the mem2reg-promoted argument registers.
    let mut inputs: HashSet<VarnodeId> =
        live_in.get(&root).into_iter().flatten().copied().collect();

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
    // A functionalized (`pure_reg`) function's interface — its by-value input
    // params and returned write-set — is owned by `argpromote_registers`. The
    // legacy register-ABI summary (recomputing input/clobbered/saved/stack-delta
    // from a conventional prologue/epilogue) does not model a functionalized body
    // and would desync `input_regs` from the arguments `argpromote_registers`
    // already bound at every call site. Leave its signature untouched.
    if Function::from_id(ctx, function_id).is_pure_reg() {
        return;
    }

    let stack_delta = compute_stack_delta(ctx, function_id, stack_ptr);

    let saved_set: HashSet<VarnodeId> = compute_saved_regs(ctx, function_id).into_iter().collect();
    // Register inputs (existing path). The saved/stack-pointer exclusions apply
    // only to registers; stack-passed parameters are appended below untouched.
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

    // This function reads a passed pointer (or its own frame) unboundedly when it
    // has a dynamic stack access or forwards into an unbounded/indirect/external
    // call. Union with the seeded value so the fact only grows across
    // checkpoint+replay rounds.
    let reads_unbounded = Function::from_id(ctx, function_id).reads_unbounded_stack()
        || has_dynamic_stack_pointer_deref(ctx, function_id)
        || function_makes_unbounded_call(ctx, function_id)
        || function_writes_through_stack_arg(ctx, function_id);

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
    use super::*;
    use crate::AliasResult;
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
    fn compute_input_regs_tolerates_root_outside_block_set() {
        // A boundary-splitting reattribution can leave a stub function whose
        // `root` points at a block reassigned away — no longer in its block set.
        // compute_input_regs must degrade to best-effort, not panic (regression
        // for the whole-binary "no entry found for key" crash on /usr/bin/less).
        let mut tc = TestContext::new();
        let r0 = tc.r0;
        let reg = tc.reg_space;
        let fun = build_fn(&mut tc, "stub", 0x1000, |b| {
            b.push_load::<false>(ValueId::Varnode(r0), 8, reg);
        });
        let root = tc.ctx.values.functions[fun].root.unwrap();
        // Orphan the root, as the splitter used to leave it.
        tc.ctx.values.functions[fun].blocks.remove(&root);
        // Best-effort, and specifically no panic.
        let _ = compute_input_regs(&tc.ctx, fun);
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


    // -----------------------------------------------------------------------
    // Stack-passed parameters + frame-escape safety
    // -----------------------------------------------------------------------

    #[test]
    fn dynamic_stack_frame_pointer_read_is_unbounded_stack_reader() {
        let mut tc = TestContext::new();
        let sp = tc.r3;
        let ram = tc.ctx.default_space;
        let reg = tc.reg_space;
        let r0 = tc.r0;
        crate::stack::brighten::get_or_make_stack_space(&mut tc.ctx, 8);

        let callee = build_fn(&mut tc, "callee", 0x2000, |b| {
            let local = stack_addr_lit(b.context_mut(), -8);
            let idx = b.push_load::<false>(ValueId::Varnode(r0), 8, reg).id();
            let ptr = b.push_add(local, idx).id();
            let loaded = b.push_load::<false>(ptr, 1, ram).id();
            let ret = b.push_zext(loaded, 8).id();
            b.push_return(ret);
        });
        let aliases = AliasResult::simple(&tc.ctx);
        crate::mem2reg(&mut tc.ctx, callee, &aliases);
        set_function_summaries(&mut tc.ctx, callee, sp);

        assert!(
            Function::from_id(&tc.ctx, callee).reads_unbounded_stack(),
            "a computed pointer into this function's own frame remains unbounded"
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
