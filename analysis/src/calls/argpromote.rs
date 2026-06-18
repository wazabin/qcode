//! `argpromote`: functionalize a function's memory side effects via shadow
//! memory and a returned write-set.
//!
//! Some functions mutate memory through pointer parameters (and globals). This
//! pass rewrites such a function so it has no side effects: it operates on a
//! private *shadow* copy of memory and *returns the writes it made* as data —
//! a tuple of `(address, value)` pairs — which the caller replays:
//!
//! ```text
//!   void f(int *p, int *q) { *q = 100; *p += 1; }
//!     callee:  f(p_ptr, q_ptr, p)  ->  return ((q_ptr, 100), (p_ptr, p + 1))
//!     caller:  r = f(&x, &x, x);  store(&x, r.0); store(&x, r.1)
//! ```
//!
//! ## Why aliasing is handled
//!
//! Every dereferenced pointer parameter shares **one shadow space**, keyed by the
//! real address values. When two pointers are equal they collide in the shadow,
//! so a `load p` after a `store q` sees the written value — the value
//! computation stays correct with no equality guards and no anti-alias gate. The
//! writes are replayed by the caller in any order (aliased addresses carry the
//! same final value), which preserves the caller-visible effect.
//!
//! ## Mechanism (see [`analyze_param`], [`try_promote`], [`apply`])
//!
//! * each dereferenced pointer param keeps its *address* and gains a by-value
//!   snapshot of the bounded region it *reads* (1/2/4/8 bytes; write-only params
//!   need no snapshot). The caller loads that snapshot and passes it in.
//! * the callee seeds the shadow from the snapshot at entry, its loads/stores are
//!   redirected into the shadow (the real address kept as the index), and at the
//!   single return it reloads each written address and hands back the write-set
//!   through [`Return::value`] as an [`Aggregate`](qcode::types::TypeRepr::Aggregate).
//! * a real register-based return value is left untouched on its own channel.
//!
//! Eligibility (anything else leaves the function untouched):
//!
//! * at least one promoted pointer is written (otherwise nothing to move);
//! * the function makes no call (composition of effectful callees is deferred);
//! * it has a single return block (so written addresses dominate the return);
//! * every dereferenced pointer is promotable — an address leaked to memory bails
//!   the whole function, since promoting some-but-not-all would be unsound;
//! * the function's address is never taken, so every caller is a direct
//!   [`Call`] this pass can rewrite. NB: this assumes a *closed world* over
//!   discovered code — a caller we never disassembled would still see the old
//!   by-reference ABI. That gap is intentional and unguarded.
//!
//! Out of scope for now: loops / dynamically-sized write sets, addresses loaded
//! from memory (`**pp`), lifting register/stack effects, and effectful-callee
//! composition. See `ARGPROMOTE_DESIGN.md`.

use std::borrow::Cow;

use qcode::{
    builder::Builder,
    context::Context,
    space::SpaceType,
    value::{
        BasicBlock, Function, FunctionId, Instruction, Value, ValueId, Varnode, VarnodeId,
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

/// A dereferenced pointer parameter slated for promotion, and the IR sites the
/// rewrite must touch.
struct Promoted {
    /// The pointer parameter (a root block param) — its *address* value, kept in
    /// the rewritten function as the shadow-space index and write-set address.
    param: ValueId,
    /// Display name, used to name the added by-value snapshot param and to find
    /// the parameter's call-argument index.
    name: String,
    /// Call-argument index of the pointer parameter.
    arg_idx: usize,
    /// By-value width in bytes (the bounded region the body touches); 1/2/4/8.
    region: usize,
    /// Load/store instructions whose space must switch to the shadow space.
    accesses: Vec<InstructionId>,
    /// Distinct `(address, size)` of each store target written through this
    /// parameter — one write-set entry per address.
    write_targets: Vec<(ValueId, usize)>,
}

/// How a named parameter is used, deciding whether it is promoted.
enum ParamUse {
    /// Not used as a load/store base — left untouched (e.g. a pointer used only
    /// in arithmetic and returned, or a plain integer).
    NonPointer,
    /// A dereferenced pointer with a bounded region; promote it.
    Deref {
        region: usize,
        accesses: Vec<InstructionId>,
        write_targets: Vec<(ValueId, usize)>,
    },
    /// Dereferenced *and* the address leaks somewhere this pass cannot follow
    /// (stored to memory, mixed deref/non-deref arithmetic). Promoting some but
    /// not all dereferenced pointers would be unsound, so this bails the whole
    /// function.
    Escape,
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

    // Composition is out of scope: a callee with its own memory effects would
    // have to bubble its write-set up through ours. Bail if this function calls
    // anything.
    if function_makes_call(ctx, fid) {
        return false;
    }

    // A single return block keeps every written address dominating the return,
    // so the write-set can load each one's final value there.
    let returns: Vec<InstructionId> = Function::from_id(ctx, fid)
        .iter()
        .filter_map(|b| {
            let last = b.iter().last()?;
            matches!(last.mnemonic(), Mnemonic::Return(_)).then_some(last.id)
        })
        .collect();
    let &[ret_id] = returns.as_slice() else {
        return false;
    };

    // Classify every named root parameter. Dereferenced pointers are promoted and
    // share one shadow space (so aliasing among them stays correct); a leaked
    // address bails the whole function; everything else is left as-is.
    let candidates: Vec<(ValueId, String)> = BasicBlock::from_id(ctx, root)
        .params()
        .filter_map(|p| Some((p.id(), p.name()?.to_string())))
        .collect();

    let mut promoted: Vec<Promoted> = Vec::new();
    for (param, name) in candidates {
        match analyze_param(ctx, param) {
            ParamUse::NonPointer => {}
            ParamUse::Escape => return false,
            ParamUse::Deref {
                region,
                accesses,
                write_targets,
            } => {
                let Some(arg_idx) = arg_index_of(ctx, fid, &name) else {
                    return false;
                };
                promoted.push(Promoted {
                    param,
                    name,
                    arg_idx,
                    region,
                    accesses,
                    write_targets,
                });
            }
        }
    }

    // Nothing to move to the caller unless some promoted pointer is written.
    if promoted.iter().all(|p| p.write_targets.is_empty()) {
        return false;
    }

    apply(ctx, fid, promoted, ret_id)
}

/// `true` if `function_id` makes any call (direct or indirect).
fn function_makes_call(ctx: &Context, function_id: FunctionId) -> bool {
    Function::from_id(ctx, function_id).blocks().any(|b| {
        b.iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Call(_) | Mnemonic::CallInd(_)))
    })
}

/// The call-argument index whose synthesized name matches `name`.
fn arg_index_of(ctx: &Context, fid: FunctionId, name: &str) -> Option<usize> {
    let len = Function::from_id(ctx, fid).input_regs().map_or(0, |i| i.len());
    (0..len).find(|&i| Function::from_id(ctx, fid).input_arg_name(i).as_deref() == Some(name))
}

/// `true` if `fid`'s address is used as a value anywhere (stored, passed, or the
/// target of an indirect call). Direct calls reference the target through
/// [`Call::target`], which is *not* an operand, so they do not count.
fn is_address_taken(ctx: &Context, fid: FunctionId) -> bool {
    let target = ValueId::Function(fid);
    ctx.instructions()
        .any(|insn| insn.mnemonic().args().contains(&target))
}

fn is_register(ctx: &Context, vn: VarnodeId) -> bool {
    matches!(Varnode::from_id(ctx, vn).space().ty, SpaceType::Register)
}

/// Classify how `param` is used (see [`ParamUse`]). Collects the loads/stores to
/// redirect into the shadow space, the distinct write targets, and the by-value
/// snapshot width (bounded by the *reads* — writes may land at any offset).
fn analyze_param(ctx: &Context, param: ValueId) -> ParamUse {
    let mut accesses: Vec<InstructionId> = Vec::new();
    // (load insn, offset value (None == 0), size) — used only to bound the read
    // region the by-value snapshot must cover.
    let mut reads: Vec<(InstructionId, Option<ValueId>, usize)> = Vec::new();
    let mut write_targets: Vec<(ValueId, usize)> = Vec::new();
    let mut is_deref = false;

    for uid in ctx.users(param).to_vec() {
        match ctx.get_insn(uid).mnemonic().clone() {
            // param ± offset address computation.
            Mnemonic::Binop(b)
                if matches!(b.op, Binop::Int(IntBinop::Add))
                    && (b.lhs == param || b.rhs == param) =>
            {
                let offset = if b.lhs == param { b.rhs } else { b.lhs };
                let add_val = ValueId::Instruction(uid);
                let mut any = false;
                let mut all = true;
                for u2 in ctx.users(add_val).to_vec() {
                    match ctx.get_insn(u2).mnemonic().clone() {
                        Mnemonic::Load(l) if l.ptr == add_val => {
                            any = true;
                            accesses.push(u2);
                            reads.push((u2, Some(offset), l.size));
                        }
                        Mnemonic::Store(s) if s.ptr == add_val => {
                            any = true;
                            accesses.push(u2);
                            write_targets.push((add_val, s.size));
                        }
                        _ => all = false,
                    }
                }
                if any {
                    is_deref = true;
                    // The dereferenced address also flowed somewhere we cannot
                    // follow — unsound to shadow.
                    if !all {
                        return ParamUse::Escape;
                    }
                }
                // `!any`: pure address arithmetic (e.g. `return p + k`) — fine.
            }
            // Direct access at offset 0.
            Mnemonic::Load(l) if l.ptr == param => {
                is_deref = true;
                accesses.push(uid);
                reads.push((uid, None, l.size));
            }
            Mnemonic::Store(s) if s.ptr == param => {
                is_deref = true;
                accesses.push(uid);
                write_targets.push((param, s.size));
            }
            // The address stored as a *value* into memory leaks it (a store into
            // a register is the harmless "return the pointer" idiom).
            Mnemonic::Store(s)
                if s.src == param
                    && !matches!(s.ptr, ValueId::Varnode(vn) if is_register(ctx, vn)) =>
            {
                return ParamUse::Escape;
            }
            // Any other use treats the address as data (returned, compared,
            // branched, returned in a register). Harmless: the address is a value
            // the caller owns.
            _ => {}
        }
    }

    if !is_deref {
        return ParamUse::NonPointer;
    }

    // The snapshot must cover every offset *read*. An unbounded (e.g. loop-driven)
    // offset yields a huge region that fails the width check. A write-only param
    // needs no snapshot (region 0): its store precedes the write-set reload.
    let mut region_end: u64 = 0;
    for (insn, offset, size) in &reads {
        let Some(block) = ctx.get_insn(*insn).parent().map(|b| b.id) else {
            return ParamUse::Escape;
        };
        let hi = match offset {
            None => 0,
            Some(off) => value_range(ctx, *off, block).max,
        };
        region_end = region_end.max(hi.saturating_add(*size as u64));
    }
    let region = region_end as usize;
    if region != 0 && !matches!(region, 1 | 2 | 4 | 8) {
        return ParamUse::Escape;
    }

    // Dedup write targets by address value.
    let mut deduped: Vec<(ValueId, usize)> = Vec::new();
    for wt in write_targets {
        if !deduped.iter().any(|(a, _)| *a == wt.0) {
            deduped.push(wt);
        }
    }

    ParamUse::Deref {
        region,
        accesses,
        write_targets: deduped,
    }
}

/// Rewrite `fid` and every direct caller into the shadow-memory / write-set form.
/// `ret_id` is the function's single `Return`. Returns `false` if a precondition
/// fails late (e.g. no callers).
fn apply(ctx: &mut Context, fid: FunctionId, mut promoted: Vec<Promoted>, ret_id: InstructionId) -> bool {
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

    let ram = ctx.default_space;
    // One private shadow space shared by every promoted pointer, keyed by the
    // real address values: equal addresses collide here, so aliasing among the
    // pointers stays correct without any anti-alias gate.
    let shadow = ctx.make_temp_space();
    let root = Function::from_id(ctx, fid).root().map(|b| b.id).unwrap();

    // Deterministic order shared by the callee (param creation) and the callers
    // (snapshot argument order).
    promoted.sort_by_key(|p| p.arg_idx);

    // ---- callee rewrite -----------------------------------------------------

    // Add a by-value snapshot parameter per *read* region and seed the shadow
    // with it at entry. Write-only params (region 0) need no snapshot.
    let mut snapshots: Vec<(ValueId, ValueId, usize)> = Vec::new(); // (ptr_param, snapshot, region)
    for p in &promoted {
        if p.region == 0 {
            continue;
        }
        let val_pid = BasicBlock::from_id_mut(ctx, root).push_param(p.region).id;
        ctx.values.block_params[val_pid].name = Some(Cow::Owned(format!("{}_val", p.name)));
        snapshots.push((p.param, ValueId::BlockParam(val_pid), p.region));
    }
    if !snapshots.is_empty() {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, root));
        b.set_insert_point_to_start();
        for (ptr, snap, _region) in &snapshots {
            b.push_store(*snap, *ptr, shadow);
        }
    }

    // Redirect every promoted load/store into the shadow space, keeping the real
    // address as the index.
    for p in &promoted {
        for &acc in &p.accesses {
            let mut m = ctx.get_insn(acc).mnemonic().clone();
            match &mut m {
                Mnemonic::Load(l) => l.space = shadow,
                Mnemonic::Store(s) => s.space = shadow,
                _ => {}
            }
            ctx.replace_instruction_mnemonic(acc, m);
        }
    }

    // The distinct write targets across all promoted params, in a stable order.
    let mut write_targets: Vec<(ValueId, usize)> = Vec::new();
    for p in &promoted {
        for wt in &p.write_targets {
            if !write_targets.iter().any(|(a, _)| *a == wt.0) {
                write_targets.push(*wt);
            }
        }
    }

    // Build the write-set `((addr, final_value), …)` from the shadow at the
    // single return, and hand it back through `Return::value`. A real
    // register-based return is left untouched on its own channel.
    let ret_block = ctx.get_insn(ret_id).parent().map(|b| b.id).unwrap();
    let writeset_val = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, ret_block));
        b.set_insert_point_before(ret_id);
        let mut pairs: Vec<ValueId> = Vec::new();
        for (addr, size) in &write_targets {
            let v = b.push_load::<false>(*addr, *size, shadow).id();
            let pair = b.push_tuple(vec![*addr, v]).id;
            pairs.push(ValueId::Instruction(pair));
        }
        ValueId::Instruction(b.push_tuple(pairs).id)
    };
    {
        let mut m = ctx.get_insn(ret_id).mnemonic().clone();
        if let Mnemonic::Return(ref mut r) = m {
            r.value = Some(writeset_val);
        }
        ctx.replace_instruction_mnemonic(ret_id, m);
    }
    let writeset_ty = ctx.type_of(writeset_val);

    // ---- caller rewrite -----------------------------------------------------
    let n_writes = write_targets.len();
    for call_id in call_sites {
        let Some(call_block) = ctx.get_insn(call_id).parent().map(|b| b.id) else {
            continue;
        };
        let (target, args, clobbers) = match ctx.get_insn(call_id).mnemonic().clone() {
            Mnemonic::Call(c) => (c.target, c.args, c.clobbers),
            _ => continue,
        };

        // Pass each read-region snapshot by value, loaded from the pointer arg.
        let mut new_args = args.clone();
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, call_block));
            b.set_insert_point_before(call_id);
            for p in &promoted {
                if p.region == 0 || p.arg_idx >= args.len() {
                    continue;
                }
                let snap = b.push_load::<false>(args[p.arg_idx], p.region, ram).id();
                new_args.push(snap);
            }
        }
        ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target,
                args: new_args,
                clobbers,
            }),
        );
        // The call now produces the write-set aggregate.
        Instruction::from_id_mut(ctx, call_id).set_type(writeset_ty);

        // Replay each returned `(addr, value)` write into real memory.
        let Some(cont) = BasicBlock::from_id(ctx, call_block)
            .successors()
            .next()
            .map(|(_, b)| b)
        else {
            continue;
        };
        let result = ValueId::Instruction(call_id);
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, cont));
        b.set_insert_point_to_start();
        for i in 0..n_writes {
            let pair = ValueId::Instruction(b.push_extract(result, i).id);
            let addr = ValueId::Instruction(b.push_extract(pair, 0).id);
            let val = ValueId::Instruction(b.push_extract(pair, 1).id);
            b.push_store(val, addr, ram);
        }
    }

    true
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
    use qcode::value::{BlockId, Function};
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

    // ---- TDD acceptance suite for the shadow-memory / write-set redesign ----
    //
    // These encode the worked examples from the design (see ARGPROMOTE_DESIGN.md).
    // They are `#[ignore]`d until the shadow-memory rewrite lands: the current
    // single-buffer pass does not yet produce the `(real_return, write-set)`
    // aggregate return, nor replay it at the caller. Each test builds the
    // pre-transformation callee + a caller, runs `argpromote`, and asserts the
    // caller-observable contract: one replayed `store` per write-set entry. The
    // exact aggregate nesting is an implementation detail and deliberately not
    // asserted here.

    /// Number of `store` instructions the caller replays after the call — i.e.
    /// in the call block's continuation. The new design emits one per write-set
    /// `(address, value)` pair returned by the callee.
    fn replayed_stores(tc: &qcode::testing::TestContext, call_id: InstructionId) -> usize {
        let Some(call_block) = Instruction::from_id(&tc.ctx, call_id).parent().map(|b| b.id) else {
            return 0;
        };
        let Some(cont) = BasicBlock::from_id(&tc.ctx, call_block)
            .successors()
            .next()
            .map(|(_, b)| b)
        else {
            return 0;
        };
        BasicBlock::from_id(&tc.ctx, cont)
            .iter()
            .filter(|i| matches!(i.mnemonic(), Mnemonic::Store(_)))
            .count()
    }

    /// `void f(int *p, int *q) { *q = 100; *p += 1; }`
    /// → `((q_ptr, 100), (p_ptr, p + 1))`. The aliasing example: with both
    /// pointers sharing `shadow_ram`, `f(&x, &x)` stays correct, and the caller
    /// replays two writes. (This is the case v1 *rejects*.)
    #[test]
    fn example_two_inout_pointers() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8); // stack_10000004
        let in_q = stack_input(&mut tc, 16, 8); // stack_10000010
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64 @stack_10000010:i64>
                    store(@stack_10000010, i32 100);
                    %v = load(i32, @stack_10000004);
                    %s = %v + i32 1;
                    store(@stack_10000004, %s);
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
        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![in_p, in_q]);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let b = tc.ctx.get_const(0x5000, 8).id();
        let call_id = set_call(&mut tc, g_call, f, vec![a, b]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx), "both in/out pointers should be promoted");
        assert_eq!(replayed_stores(&tc, call_id), 2, "two writes => two replays");
    }

    /// `void foo(int *p) { *p += 10; }`
    /// → `(intptr, int) foo(intptr p_ptr, int p) { return (p_ptr, p + 10); }`.
    #[test]
    fn example_single_inout_void() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8);
        qcode!(
            tc.ctx,
            "
            fn foo:
                <foo_entry @stack_10000004:i64>
                    %v = load(i32, @stack_10000004);
                    %s = %v + i32 10;
                    store(@stack_10000004, %s);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, foo).set_input_regs(vec![in_p]);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, foo, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx));
        assert_eq!(replayed_stores(&tc, call_id), 1);
    }

    /// `int* foo(int* p) { *p += 10; return p; }`
    /// → `(intptr, (intptr, int)) foo(intptr p_ptr, int p) { return (p_ptr, (p_ptr, p + 10)); }`.
    /// A real return value coexists with the write-set.
    #[test]
    fn example_inout_returns_pointer() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8);
        let r0 = tc.r0; // the real-return register
        qcode!(
            tc.ctx,
            "
            fn foo:
                <foo_entry @stack_10000004:i64>
                    %v = load(i32, @stack_10000004);
                    %s = %v + i32 10;
                    store(@stack_10000004, %s);
                    store({r0}, @stack_10000004);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, foo).set_input_regs(vec![in_p]);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, foo, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx));
        // One write replayed; the returned pointer rides the real-return channel.
        assert_eq!(replayed_stores(&tc, call_id), 1);
    }

    /// `int foo(int* p) { p += 10; *p = 5; }`
    /// → `(intptr, int) foo(intptr p_ptr, int p) { return (p_ptr + 10, 5); }`.
    /// A write through an *advanced* pointer: the returned address is `p + 10`.
    #[test]
    fn example_write_through_advanced_pointer() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8);
        qcode!(
            tc.ctx,
            "
            fn foo:
                <foo_entry @stack_10000004:i64>
                    %adv = @stack_10000004 + i64 40;
                    store(%adv, i32 5);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, foo).set_input_regs(vec![in_p]);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, foo, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx));
        assert_eq!(replayed_stores(&tc, call_id), 1);
    }

    /// `int foo(int *p) { return *p + 10; }`
    /// → `int foo(int p) { return p + 10; }`. A read-only pointer collapses to a
    /// by-value scalar: no write-set, so the caller replays nothing.
    #[test]
    fn example_read_only_pointer_has_no_writeset() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8);
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn foo:
                <foo_entry @stack_10000004:i64>
                    %v = load(i32, @stack_10000004);
                    %s = %v + i32 10;
                    store({r0}, %s);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, foo).set_input_regs(vec![in_p]);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, foo, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        argpromote(&mut tc.ctx);
        assert_eq!(replayed_stores(&tc, call_id), 0, "read-only pointer writes nothing back");
    }

    /// `int* foo(int *p) { return p + 10; }`
    /// → `intptr foo(intptr p) { return p + 10; }`. Pure pointer arithmetic, no
    /// dereference: nothing to mirror, no write-set.
    #[test]
    fn example_pointer_arithmetic_only_has_no_writeset() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8);
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn foo:
                <foo_entry @stack_10000004:i64>
                    %adv = @stack_10000004 + i64 40;
                    store({r0}, %adv);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, foo).set_input_regs(vec![in_p]);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, foo, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        argpromote(&mut tc.ctx);
        assert_eq!(replayed_stores(&tc, call_id), 0, "no dereference => no write-set");
    }

    /// `(int*, int, int) f1(int* p_ptr, int p, int q) { return (p_ptr, p + 1, 100); }`
    /// The flat multi-value illustration: one write through `p` plus a real
    /// return, producing a multi-element functional return. Modeled here as
    /// `*p += 1` with an `int` return of `100`; the caller replays the single
    /// write while the `100` rides the real-return channel.
    #[test]
    fn example_multi_value_return() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8);
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f1:
                <f1_entry @stack_10000004:i64>
                    %v = load(i32, @stack_10000004);
                    %s = %v + i32 1;
                    store(@stack_10000004, %s);
                    store({r0}, i32 100);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f1>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, f1).set_input_regs(vec![in_p]);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, f1, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx));
        assert_eq!(replayed_stores(&tc, call_id), 1);
    }
}
