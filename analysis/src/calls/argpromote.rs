//! `argpromote`: interprocedural by-reference → by-value parameter promotion.
//!
//! Some functions take a pointer and mutate `*ptr` in place — an in/out
//! parameter. This pass rewrites such a function so it takes the *value* and
//! returns the new value, and hoists the load/store to every caller:
//!
//! ```text
//!   callee:  f(ptr)            -->  v = f(load(ptr));  store(ptr, v)   (caller)
//! ```
//!
//! It fires only when it can prove the rewrite is faithful (see [`analyze_param`]
//! and [`try_promote`]):
//!
//! * the parameter is a stack-passed pointer used *only* as the base of
//!   loads/stores within a statically bounded, contiguous region of
//!   1/2/4/8 bytes (the by-value width), plus the optional "return the pointer
//!   in a register" passthrough;
//! * the by-value width fits the pointer-sized parameter slot (it may be
//!   *narrower* — e.g. a 4-byte `int*` on a 64-bit target — in which case the
//!   caller zero-extends into the slot);
//! * the function both reads and writes the region (a true in/out buffer);
//! * it is the *only* promotable in/out parameter of its function — the promoted
//!   value claims the function's single return channel ([`Return::value`]), so a
//!   second one would have nowhere to go (lifting this needs an aggregate/tuple
//!   return; see the module-level TODO);
//! * no *other* parameter is a pointer the callee dereferences — otherwise it
//!   could alias the promoted pointee, and hoisting the promoted read before the
//!   call and its write after would reorder the two accesses across the call;
//! * the function's address is never taken, so every caller is a direct
//!   [`Call`] this pass can rewrite. NB: this assumes a *closed world* over
//!   discovered code — a caller in code we never disassembled would still see
//!   the old by-reference ABI. That gap is intentional and unguarded.
//!
//! A function that returns a real value through its normal output register is
//! *not* excluded: that register return is left untouched and coexists with the
//! promoted channel.
//!
//! Anything else makes the function ineligible and it is left untouched.
//!
//! ## Representation
//!
//! The callee keeps its dynamic byte indexing by spilling the incoming value
//! into a fresh, private [`Temporary`](SpaceType::Temporary) address space (a
//! distinct `SpaceId`, so it cannot alias anything), running the existing body
//! against that buffer, and reloading it into [`Return::value`]. The call gains
//! a result value the natural way — an instruction *is* the value it defines, so
//! the call terminator is given a non-zero result width (`%r = call f(...)`),
//! referenced in the continuation by the store-back. The old "return the pointer
//! in EAX" store becomes dead; any caller that forwarded that register is
//! repointed to the pointer it already holds.

use qcode::{
    builder::Builder,
    context::Context,
    space::SpaceType,
    value::{
        BasicBlock, BlockId, Function, FunctionId, Instruction, Value, ValueId, Varnode, VarnodeId,
        insn::{Binop, Call, InstructionId, IntBinop, Mnemonic},
    },
};

use crate::{Pass, PipelineEnv, value_range};

/// Promotes every eligible by-reference in/out parameter in the module. Returns
/// `true` if anything changed.
pub fn argpromote(ctx: &mut Context) -> bool {
    let mut changed = false;
    for fid in ctx.function_ids() {
        if try_promote(ctx, fid) {
            changed = true;
        }
    }
    changed
}

/// A proven-promotable parameter and the IR sites the rewrite must touch.
struct Plan {
    /// The pointer parameter (a root block param), as a value.
    param: ValueId,
    /// The parameter's display name, used to find its call-argument index.
    param_name: String,
    /// By-value width in bytes (the bounded region size); one of 1/2/4/8.
    region_size: usize,
    /// Width in bytes of the pointer-sized parameter slot the value rides in.
    /// `>= region_size`; when strictly greater the caller zero-extends into it.
    slot_width: usize,
    /// `param + offset` address computations to rebase onto the local buffer.
    adds: Vec<InstructionId>,
    /// Load/store instructions whose space must switch to the local buffer.
    accesses: Vec<InstructionId>,
    /// The optional `store(register R <- param)` passthrough (the "return the
    /// pointer in a register" idiom): its instruction and the register R that
    /// callers may read as the returned pointer. `None` for functions that
    /// return void or return a real value through their normal output register —
    /// both are fine, because the promoted value rides an orthogonal channel
    /// (`Return::value` / the call's result), which nothing outside this pass
    /// reads. Only the returns-the-pointer idiom needs the cleanup in [`apply`].
    passthrough: Option<(InstructionId, VarnodeId)>,
}

fn try_promote(ctx: &mut Context, fid: FunctionId) -> bool {
    let f = Function::from_id(ctx, fid);
    if f.is_external() {
        return false;
    }
    let Some(root) = f.root().map(|b| b.id) else {
        return false;
    };

    // Closed-world / direct-only gate: if the function's address is taken it may
    // be reached by an indirect call this pass cannot find and rewrite, leaving a
    // caller on the old by-reference ABI. (Callers in undiscovered code are an
    // accepted, unguardable gap — see the module docs.)
    if is_address_taken(ctx, fid) {
        return false;
    }

    // Candidate parameters: every named root parameter. Stack-passed inputs have
    // a nameless varnode, so we key off the param itself (its name maps it to a
    // call-argument index via `input_arg_name`) rather than a varnode lookup.
    let candidates: Vec<(ValueId, usize, String)> = BasicBlock::from_id(ctx, root)
        .params()
        .filter_map(|p| Some((p.id(), p.size(), p.name()?.to_string())))
        .collect();

    for (param, width, name) in &candidates {
        // Anti-aliasing cut: refuse if any *other* parameter is a pointer the
        // callee dereferences. It could alias the promoted pointee, and the
        // rewrite moves the promoted access to the other side of the call.
        if candidates
            .iter()
            .any(|(other, _, _)| other != param && param_is_memory_base(ctx, *other))
        {
            continue;
        }
        if let Some(plan) = analyze_param(ctx, *param, *width, name.clone())
            && apply(ctx, fid, plan)
        {
            return true;
        }
    }
    false
}

/// `true` if `fid`'s address is used as a value anywhere (stored, passed, or the
/// target of an indirect call). Direct calls reference the target through
/// [`Call::target`], which is *not* an operand, so they do not count.
fn is_address_taken(ctx: &Context, fid: FunctionId) -> bool {
    let target = ValueId::Function(fid);
    ctx.instructions()
        .any(|insn| insn.mnemonic().args().contains(&target))
}

/// `true` if `param` is dereferenced as a pointer: used directly as a load/store
/// address, or as `param + offset` feeding one. Used to reject promotion when a
/// *second* parameter could alias the promoted buffer.
fn param_is_memory_base(ctx: &Context, param: ValueId) -> bool {
    ctx.users(param).iter().any(|&uid| {
        match ctx.get_insn(uid).mnemonic() {
            Mnemonic::Load(l) => l.ptr == param,
            Mnemonic::Store(s) => s.ptr == param,
            Mnemonic::Binop(b)
                if matches!(b.op, Binop::Int(IntBinop::Add))
                    && (b.lhs == param || b.rhs == param) =>
            {
                let add_val = ValueId::Instruction(uid);
                ctx.users(add_val).iter().any(|&u2| {
                    matches!(
                        ctx.get_insn(u2).mnemonic(),
                        Mnemonic::Load(_) | Mnemonic::Store(_)
                    )
                })
            }
            _ => false,
        }
    })
}

fn is_register(ctx: &Context, vn: VarnodeId) -> bool {
    matches!(Varnode::from_id(ctx, vn).space().ty, SpaceType::Register)
}

/// Classify every use of `param` and, if it is a bounded in/out buffer pointer
/// that never escapes, return the [`Plan`] describing how to promote it.
fn analyze_param(
    ctx: &Context,
    param: ValueId,
    ptr_width: usize,
    param_name: String,
) -> Option<Plan> {
    let mut adds = Vec::new();
    // (load/store insn, offset value (None == 0), access size, is_store)
    let mut accesses: Vec<(InstructionId, Option<ValueId>, usize, bool)> = Vec::new();
    let mut passthrough: Option<(InstructionId, VarnodeId)> = None;
    let mut has_in = false;
    let mut has_out = false;

    for uid in ctx.users(param).to_vec() {
        match ctx.get_insn(uid).mnemonic().clone() {
            // param ± offset address computation feeding loads/stores.
            Mnemonic::Binop(b)
                if matches!(b.op, Binop::Int(IntBinop::Add))
                    && (b.lhs == param || b.rhs == param) =>
            {
                let offset = if b.lhs == param { b.rhs } else { b.lhs };
                let add_val = ValueId::Instruction(uid);
                for u2 in ctx.users(add_val).to_vec() {
                    match ctx.get_insn(u2).mnemonic().clone() {
                        Mnemonic::Load(l) if l.ptr == add_val => {
                            accesses.push((u2, Some(offset), l.size, false));
                            has_in = true;
                        }
                        Mnemonic::Store(s) if s.ptr == add_val => {
                            accesses.push((u2, Some(offset), s.size, true));
                            has_out = true;
                        }
                        // The address escaped to a non-load/store use.
                        _ => return None,
                    }
                }
                adds.push(uid);
            }
            // Direct access at offset 0.
            Mnemonic::Load(l) if l.ptr == param => {
                accesses.push((uid, None, l.size, false));
                has_in = true;
            }
            Mnemonic::Store(s) if s.ptr == param => {
                accesses.push((uid, None, s.size, true));
                has_out = true;
            }
            // The tolerated passthrough: `store(register R <- param)`.
            Mnemonic::Store(s) if s.src == param => {
                let ValueId::Varnode(reg) = s.ptr else {
                    return None;
                };
                if !is_register(ctx, reg) || passthrough.is_some() {
                    return None;
                }
                passthrough = Some((uid, reg));
            }
            // Any other use escapes the pointer.
            _ => return None,
        }
    }

    // Only true in/out buffers. The passthrough is optional: void- and
    // value-returning functions have none, and that is fine.
    if !(has_in && has_out) {
        return None;
    }

    // Bound the accessed region. An unbounded offset yields a huge `region_end`
    // that fails the width check below, so no separate Top test is needed.
    let mut region_end: u64 = 0;
    for (insn, offset, size, _) in &accesses {
        let block = ctx.get_insn(*insn).parent().map(|b| b.id)?;
        let hi = match offset {
            None => 0,
            Some(off) => value_range(ctx, *off, block).max,
        };
        region_end = region_end.max(hi.saturating_add(*size as u64));
    }

    let region_size = region_end as usize;
    if !matches!(region_size, 1 | 2 | 4 | 8) {
        return None;
    }
    // The by-value width must *fit* the pointer-sized parameter slot. It may be
    // narrower (e.g. a 4-byte `int*` on a 64-bit target): the caller zero-extends
    // the loaded value into the slot and the callee reads back the low bytes. A
    // region wider than the slot has nowhere to ride, so it is rejected.
    if region_size > ptr_width {
        return None;
    }

    Some(Plan {
        param,
        param_name,
        region_size,
        slot_width: ptr_width,
        adds,
        accesses: accesses.iter().map(|a| a.0).collect(),
        passthrough,
    })
}

/// Apply a promotion: rewrite the callee body and every direct caller. Returns
/// `false` (leaving the function untouched) if a precondition fails late.
fn apply(ctx: &mut Context, fid: FunctionId, plan: Plan) -> bool {
    // Caller plumbing needs the input index of the promoted param: the input
    // whose synthesized argument name matches the param's name.
    let inputs_len = Function::from_id(ctx, fid)
        .input_regs()
        .map_or(0, |i| i.len());
    let arg_idx = (0..inputs_len).find(|&i| {
        Function::from_id(ctx, fid).input_arg_name(i).as_deref() == Some(plan.param_name.as_str())
    });
    let Some(arg_idx) = arg_idx else {
        return false;
    };
    let call_sites: Vec<InstructionId> = ctx
        .instructions()
        .filter_map(|insn| match insn.mnemonic() {
            Mnemonic::Call(c) if c.target == fid => Some(insn.id),
            _ => None,
        })
        .collect();
    if call_sites.is_empty() {
        return false;
    }

    let region = plan.region_size;
    let slot_width = plan.slot_width;
    let ram = ctx.default_space;
    let buf_space = ctx.make_temp_space();
    // A space-typed base pointer: distinct per buffer space, so the alias
    // analysis's "one space per pointer value" invariant holds across functions.
    // Addresses use the native pointer (slot) width; the *value* load/stores
    // below use `region`.
    let base_ty = ctx.types.get_or_make_space_address(slot_width, buf_space);
    let base0 = ctx.get_typed_const(0, base_ty).id();
    let int_region = ctx.types.get_or_make_int(region);

    // ---- callee rewrite -----------------------------------------------------

    // Spill the incoming value into the private buffer at function entry.
    let root = Function::from_id(ctx, fid).root().map(|b| b.id).unwrap();
    {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, root));
        b.set_insert_point_to_start();
        b.push_store(plan.param, base0, buf_space);
    }

    // Rebase `param + offset` onto the buffer base (0).
    for &add in &plan.adds {
        let mut m = ctx.get_insn(add).mnemonic().clone();
        m.replace_value(plan.param, base0);
        ctx.replace_instruction_mnemonic(add, m);
    }

    // Point the loads/stores at the buffer space (and at base 0 for direct ones).
    for &acc in &plan.accesses {
        let mut m = ctx.get_insn(acc).mnemonic().clone();
        match &mut m {
            Mnemonic::Load(l) => {
                l.space = buf_space;
                if l.ptr == plan.param {
                    l.ptr = base0;
                }
            }
            Mnemonic::Store(s) => {
                s.space = buf_space;
                if s.ptr == plan.param {
                    s.ptr = base0;
                }
            }
            _ => {}
        }
        ctx.replace_instruction_mnemonic(acc, m);
    }

    // Drop the "return the pointer" passthrough, if any; the value return
    // replaces it. Void- and value-returning functions have none.
    if let Some((passthrough_insn, _)) = plan.passthrough {
        ctx.remove_instruction(passthrough_insn);
    }

    // Reload the buffer and hand it back through Return.value at each return.
    let ret_sites: Vec<(BlockId, InstructionId)> = Function::from_id(ctx, fid)
        .iter()
        .filter_map(|b| {
            let last = b.iter().last()?;
            matches!(last.mnemonic(), Mnemonic::Return(_)).then(|| (b.id, last.id))
        })
        .collect();
    for (blk, ret_id) in ret_sites {
        let ret_val = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, blk));
            b.set_insert_point_before(ret_id);
            b.push_load::<false>(base0, region, buf_space).id()
        };
        let mut m = ctx.get_insn(ret_id).mnemonic().clone();
        if let Mnemonic::Return(ref mut r) = m {
            r.value = Some(ret_val);
        }
        ctx.replace_instruction_mnemonic(ret_id, m);
    }

    // ---- caller rewrite -----------------------------------------------------
    for call_id in call_sites {
        let Some(call_block) = ctx.get_insn(call_id).parent().map(|b| b.id) else {
            continue;
        };
        let (target, args, clobbers) = match ctx.get_insn(call_id).mnemonic().clone() {
            Mnemonic::Call(c) => (c.target, c.args, c.clobbers),
            _ => continue,
        };
        if arg_idx >= args.len() {
            continue;
        }
        let p = args[arg_idx];

        // Load the region value just before the call and pass it by value,
        // zero-extending into the (wider) pointer-sized parameter slot when the
        // value is narrower. The callee reads back the low `region` bytes.
        let in_val = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, call_block));
            b.set_insert_point_before(call_id);
            let loaded = b.push_load::<false>(p, region, ram).id();
            if region < slot_width {
                b.push_zext(loaded, slot_width).id()
            } else {
                loaded
            }
        };
        let mut new_args = args.clone();
        new_args[arg_idx] = in_val;
        ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target,
                args: new_args,
                clobbers,
            }),
        );
        // The call now produces the returned value.
        Instruction::from_id_mut(ctx, call_id).set_type(int_region);

        let Some(cont) = BasicBlock::from_id(ctx, call_block)
            .successors()
            .next()
            .map(|(_, b)| b)
        else {
            continue;
        };
        // Store the returned value back into the caller's buffer.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, cont));
            b.set_insert_point_to_start();
            b.push_store(ValueId::Instruction(call_id), p, ram);
        }
        // For the returns-the-pointer idiom only: any forward of the old
        // "returned pointer" register now reads the value channel; repoint it to
        // the pointer the caller already holds.
        if let Some((_, return_reg)) = plan.passthrough {
            fixup_returned_pointer(ctx, cont, return_reg, p);
        }
    }

    true
}

/// Replace, in `block`, reads of `reg` (the callee's old "returned pointer"
/// register) with `p`, up to the point `reg` is redefined. The now-dead register
/// loads are left for DCE.
fn fixup_returned_pointer(ctx: &mut Context, block: BlockId, reg: VarnodeId, p: ValueId) {
    let regv = ValueId::Varnode(reg);
    let insns: Vec<InstructionId> = BasicBlock::from_id(ctx, block)
        .iter()
        .map(|i| i.id)
        .collect();
    for id in insns {
        match ctx.get_insn(id).mnemonic().clone() {
            // reg redefined here: later reads see the new value, stop.
            Mnemonic::Store(s) if s.ptr == regv => break,
            Mnemonic::Load(l) if l.ptr == regv => {
                ctx.replace_all_uses_with(ValueId::Instruction(id), p);
            }
            _ => {}
        }
    }
}

#[derive(Default)]
pub struct ArgPromote;

impl Pass for ArgPromote {
    const NAME: &'static str = "argpromote";
    fn description(&self) -> &'static str {
        "Promote by-reference in/out pointer parameters to by-value"
    }
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        Ok(argpromote(ctx))
    }
}

crate::register_module_pass!(ArgPromote);

#[cfg(test)]
mod tests {
    use qcode::value::Function;
    use qcode_macro::qcode;

    use super::*;

    /// Add a "stack" space (addr_size 4, as `brighten_stack` creates it) and a
    /// nameless stack-passed input varnode at `offset`. Its synthesized argument
    /// name is `stack_<stack_base(4) + offset>` — e.g. offset 4 → `stack_10000004`
    /// — which is what a callee's promoted param name must equal.
    fn stack_input(tc: &mut qcode::testing::TestContext, offset: i64, size: usize) -> VarnodeId {
        use qcode::space::Space;
        let space = tc
            .ctx
            .try_get_space("stack")
            .unwrap_or_else(|| tc.ctx.add_space(Space::new(Some("stack"), 1, 4)));
        Varnode::make(&mut tc.ctx, offset, size, space).id
    }

    /// Find the (single) call instruction in `block` and give it `args`.
    fn set_call(tc: &mut qcode::testing::TestContext, block: BlockId, target: FunctionId, args: Vec<ValueId>) -> InstructionId {
        let call_id = BasicBlock::from_id(&tc.ctx, block)
            .iter()
            .find(|i| matches!(i.mnemonic(), Mnemonic::Call(_)))
            .unwrap()
            .id;
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call { target, args, clobbers: vec![] }),
        );
        call_id
    }

    #[test]
    fn promotes_single_inout_stack_param() {
        let mut tc = qcode::testing::TestContext::new();
        // Bind the stack slot at offset 4 as f's single input.
        let input = stack_input(&mut tc, 4, 8);

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    %v = load(i32, @stack_10000004);
                    store(@stack_10000004, %v);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;

        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx), "single in/out param should be promoted");

        // The callee now hands the buffer back through Return::value.
        let returns_value = Function::from_id(&tc.ctx, f).iter().any(|b| {
            b.iter()
                .last()
                .is_some_and(|i| matches!(i.mnemonic(), Mnemonic::Return(r) if r.value.is_some()))
        });
        assert!(returns_value, "callee must return the buffer value");

        // The caller loads the region just before the call (the by-value arg).
        let has_load = BasicBlock::from_id(&tc.ctx, g_call)
            .iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Load(_)));
        assert!(has_load, "caller must load the region value before the call");
        let _ = call_id;
    }

    #[test]
    fn rejects_second_dereferenced_pointer_param() {
        let mut tc = qcode::testing::TestContext::new();
        let in0 = stack_input(&mut tc, 4, 8); // stack_10000004
        let in1 = stack_input(&mut tc, 16, 8); // stack_10000010

        qcode!(
            tc.ctx,
            "
            fn f2:
                <f2_entry @stack_10000004:i64 @stack_10000010:i64>
                    %v = load(i32, @stack_10000004);
                    store(@stack_10000004, %v);
                    %w = load(i32, @stack_10000010);
                    store(@stack_10000010, %w);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f2>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, f2).set_input_regs(vec![in0, in1]);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let b = tc.ctx.get_const(0x5000, 8).id();
        set_call(&mut tc, g_call, f2, vec![a, b]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        // Both params are dereferenced pointers that could alias at the call
        // site, so neither may be promoted (the blunt v1 anti-aliasing cut).
        assert!(
            !argpromote(&mut tc.ctx),
            "a callee with a second dereferenced pointer param must be left untouched"
        );
    }

    #[test]
    fn noop_when_no_promotable_param() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn foo:
                <entry>
                    return [0x1000];
            "
        );
        assert!(Function::from_id(&ctx, foo).root().is_some());
        // No pointer parameters, no callers: nothing to promote, and no panic.
        assert!(!argpromote(&mut ctx));
    }
}
