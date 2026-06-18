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
    space::{SpaceId, SpaceType},
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

// ===========================================================================
// Register channel (runs early, before mem2reg — see ARGPROMOTE_REGISTERS.md)
// ===========================================================================
//
// Unlike the RAM channel above, register effects are functionalized by a purely
// syntactic scan of the lifted body: a register *loaded* is an input, a register
// *stored* is an output. No shadow space is needed — every register access
// converts, so the body keeps operating on real register space and the *next*
// mem2reg run SSA-promotes it into a pure value function. Inputs become by-value
// params seeded into register space at entry; outputs are returned as a flat
// positional aggregate (slot i ↔ output register i) the caller replays.

/// A function's register interface, recovered by [`scan_register_effects`].
struct RegisterEffects {
    /// Registers the body loads — each becomes a by-value input parameter
    /// (over-approximated: a written-first register yields a dead param that
    /// mem2reg/DCE prune). Sorted by `(address, size)` for a deterministic
    /// param/argument order shared with the caller rewrite.
    inputs: Vec<VarnodeId>,
    /// Registers the body stores, canonicalized to the coarsest register per
    /// overlap group so the caller's replay is order-independent. Sorted.
    outputs: Vec<VarnodeId>,
}

/// The byte interval `(space, start, end)` a register varnode occupies.
fn reg_interval(ctx: &Context, vn: VarnodeId) -> (SpaceId, i64, i64) {
    let v = Varnode::from_id(ctx, vn);
    let start = v.address();
    (v.space().id, start, start + v.size() as i64)
}

/// `true` if two register varnodes occupy overlapping bytes of the same space.
fn regs_overlap(ctx: &Context, a: VarnodeId, b: VarnodeId) -> bool {
    let (sa, a0, a1) = reg_interval(ctx, a);
    let (sb, b0, b1) = reg_interval(ctx, b);
    sa == sb && a0 < b1 && b0 < a1
}

/// Collapse a set of register varnodes into the coarsest register per overlap
/// group. Register files nest (AL ⊂ AX ⊂ EAX ⊂ RAX), so each overlap group has a
/// unique member whose interval contains the rest; for *outputs*, loading that
/// register at the return reads the merged final state of every sub-write, and
/// for *inputs* one seed of it covers every overlapping read. Returns `None` if
/// some group has no single covering register (partial overlap with no cover) —
/// that function is left on the conservative path.
fn canonicalize_to_coarsest(ctx: &Context, regs: &[VarnodeId]) -> Option<Vec<VarnodeId>> {
    // Connected components under `regs_overlap` (tiny N, so O(N²) is fine).
    let mut group_of: Vec<usize> = (0..regs.len()).collect();
    for i in 0..regs.len() {
        for j in (i + 1)..regs.len() {
            if regs_overlap(ctx, regs[i], regs[j]) {
                let (gi, gj) = (group_of[i], group_of[j]);
                if gi != gj {
                    for g in &mut group_of {
                        if *g == gj {
                            *g = gi;
                        }
                    }
                }
            }
        }
    }

    let mut coarse: Vec<VarnodeId> = Vec::new();
    for g in 0..regs.len() {
        let members: Vec<VarnodeId> = (0..regs.len())
            .filter(|&i| group_of[i] == g)
            .map(|i| regs[i])
            .collect();
        if members.is_empty() {
            continue; // not a group representative
        }
        // The cover must contain every member's interval.
        let cover = members.iter().copied().find(|&m| {
            let (_, m0, m1) = reg_interval(ctx, m);
            members.iter().all(|&o| {
                let (_, o0, o1) = reg_interval(ctx, o);
                m0 <= o0 && m1 >= o1
            })
        })?;
        if !coarse.contains(&cover) {
            coarse.push(cover);
        }
    }
    Some(coarse)
}

/// Scan `fid` for register reads (inputs) and writes (outputs). Returns `None`
/// when there is no register write (nothing to functionalize) or an output
/// overlap group has no single covering register.
fn scan_register_effects(ctx: &Context, fid: FunctionId) -> Option<RegisterEffects> {
    let mut loaded: Vec<VarnodeId> = Vec::new();
    let mut stored: Vec<VarnodeId> = Vec::new();
    for block in Function::from_id(ctx, fid).blocks() {
        for insn in block.iter() {
            match insn.mnemonic() {
                Mnemonic::Load(l) => {
                    if let ValueId::Varnode(vn) = l.ptr
                        && is_register(ctx, vn)
                        && !loaded.contains(&vn)
                    {
                        loaded.push(vn);
                    }
                }
                Mnemonic::Store(s) => {
                    if let ValueId::Varnode(vn) = s.ptr
                        && is_register(ctx, vn)
                        && !stored.contains(&vn)
                    {
                        stored.push(vn);
                    }
                }
                _ => {}
            }
        }
    }
    if stored.is_empty() {
        return None;
    }
    let mut outputs = canonicalize_to_coarsest(ctx, &stored)?;

    // The rewritten body reads not only the originally-loaded registers but also
    // every output (the return write-set loads each one). An output written on
    // only some paths is therefore read-before-write at a return on a no-write
    // path; seeding it from an input param makes that read the caller's incoming
    // value (replayed back as a no-op), keeping callee params and caller args in
    // sync. Always-written outputs just yield a dead seed that DCE prunes.
    let mut read_set = loaded;
    for &o in &outputs {
        if !read_set.contains(&o) {
            read_set.push(o);
        }
    }
    let mut inputs = canonicalize_to_coarsest(ctx, &read_set)?;

    let key = |ctx: &Context, vn: &VarnodeId| {
        let v = Varnode::from_id(ctx, *vn);
        (v.address(), v.size())
    };
    inputs.sort_by_key(|vn| key(ctx, vn));
    outputs.sort_by_key(|vn| key(ctx, vn));
    Some(RegisterEffects { inputs, outputs })
}

/// Rewrite `fid`'s **callee** side for its register effects: add a by-value param
/// per input register (seeded into real register space at entry), and return the
/// outputs' final values as a flat positional aggregate at every return. The
/// caller rewrite (passing inputs, replaying outputs) is separate — see Phase 2.
fn rewrite_callee_registers(ctx: &mut Context, fid: FunctionId, eff: &RegisterEffects) {
    let root = Function::from_id(ctx, fid).root().map(|b| b.id).unwrap();

    // --- inputs: a by-value param per input register, seeded at entry ---------
    // Precompute (register, size, space, name) before taking any mutable borrow.
    let input_meta: Vec<(VarnodeId, usize, SpaceId, Option<String>)> = eff
        .inputs
        .iter()
        .map(|&r| {
            let v = Varnode::from_id(ctx, r);
            (r, v.size(), v.space().id, v.name().map(str::to_owned))
        })
        .collect();

    let mut seeds: Vec<(VarnodeId, SpaceId, ValueId)> = Vec::new();
    for (r, size, space, name) in &input_meta {
        let pid = BasicBlock::from_id_mut(ctx, root).push_param(*size).id;
        // Name the param after its register so the calling convention binds it
        // from the register file (mem2reg's promoted-register-param naming, which
        // the emulator's `seed_entry_params` keys on).
        ctx.values.block_params[pid].name = name.clone().map(Cow::Owned);
        seeds.push((*r, *space, ValueId::BlockParam(pid)));
    }
    if !seeds.is_empty() {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, root));
        b.set_insert_point_to_start();
        for (r, space, param) in &seeds {
            b.push_store(*param, ValueId::Varnode(*r), *space);
        }
    }

    // --- outputs: a flat positional write-set at every return -----------------
    let output_meta: Vec<(VarnodeId, usize, SpaceId)> = eff
        .outputs
        .iter()
        .map(|&r| {
            let v = Varnode::from_id(ctx, r);
            (r, v.size(), v.space().id)
        })
        .collect();

    let returns: Vec<InstructionId> = Function::from_id(ctx, fid)
        .iter()
        .filter_map(|b| {
            let last = b.iter().last()?;
            matches!(last.mnemonic(), Mnemonic::Return(_)).then_some(last.id)
        })
        .collect();

    for ret_id in returns {
        let Some(ret_block) = ctx.get_insn(ret_id).parent().map(|b| b.id) else {
            continue;
        };
        let tuple_val = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, ret_block));
            b.set_insert_point_before(ret_id);
            let fields: Vec<ValueId> = output_meta
                .iter()
                .map(|(r, size, space)| b.push_load::<false>(ValueId::Varnode(*r), *size, *space).id())
                .collect();
            ValueId::Instruction(b.push_tuple(fields).id)
        };
        let mut m = ctx.get_insn(ret_id).mnemonic().clone();
        if let Mnemonic::Return(ref mut r) = m {
            // The register run is the first to touch the slot (it runs before the
            // RAM run), so it sets rather than appends. Composition with a later
            // run is handled when that run lands (append-only — see the design).
            r.value = Some(tuple_val);
        }
        ctx.replace_instruction_mnemonic(ret_id, m);
    }
}

/// Functionalize every eligible function's register effects (see
/// [`try_promote_registers`]). Returns `true` if anything changed.
pub fn argpromote_registers(ctx: &mut Context) -> bool {
    let mut changed = false;
    for fid in ctx.function_ids() {
        if try_promote_registers(ctx, fid) {
            changed = true;
        }
    }
    changed
}

fn try_promote_registers(ctx: &mut Context, fid: FunctionId) -> bool {
    let f = Function::from_id(ctx, fid);
    if f.is_external() || f.root().is_none() {
        return false;
    }
    // Closed-world: an address-taken function may be reached by an indirect call
    // this pass cannot find and rewrite, leaving a caller on the old register ABI.
    if is_address_taken(ctx, fid) {
        return false;
    }
    let Some(eff) = scan_register_effects(ctx, fid) else {
        return false;
    };

    // Only direct callers can be rewritten to pass inputs / replay outputs; with
    // none, rewriting the callee would leave it expecting params nobody provides.
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

    rewrite_callee_registers(ctx, fid, &eff);

    // The flat positional write-set type: one integer field per output register.
    let field_tys: Vec<_> = eff
        .outputs
        .iter()
        .map(|&r| ctx.types.get_or_make_int(Varnode::from_id(ctx, r).size()))
        .collect();
    let writeset_ty = ctx.types.get_or_make_aggregate(field_tys);

    let meta = |ctx: &Context, regs: &[VarnodeId]| -> Vec<(VarnodeId, usize, SpaceId)> {
        regs.iter()
            .map(|&r| {
                let v = Varnode::from_id(ctx, r);
                (r, v.size(), v.space().id)
            })
            .collect()
    };
    let input_meta = meta(ctx, &eff.inputs);
    let output_meta = meta(ctx, &eff.outputs);

    for call_id in call_sites {
        rewrite_caller_registers(ctx, call_id, &input_meta, &output_meta, writeset_ty);
    }
    true
}

/// Rewrite one direct call site for the register channel: load each input
/// register before the call and append it as a positional argument, retype the
/// call to the write-set aggregate, then replay each output register from the
/// returned aggregate into register space at the continuation. The next mem2reg
/// run forwards the replay stores into precise SSA.
fn rewrite_caller_registers(
    ctx: &mut Context,
    call_id: InstructionId,
    input_meta: &[(VarnodeId, usize, SpaceId)],
    output_meta: &[(VarnodeId, usize, SpaceId)],
    writeset_ty: qcode::types::TypeId,
) {
    let Some(call_block) = ctx.get_insn(call_id).parent().map(|b| b.id) else {
        return;
    };
    let (target, mut args, clobbers) = match ctx.get_insn(call_id).mnemonic().clone() {
        Mnemonic::Call(c) => (c.target, c.args, c.clobbers),
        _ => return,
    };

    // Inputs: load each input register's current value just before the call.
    {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, call_block));
        b.set_insert_point_before(call_id);
        for (r, size, space) in input_meta {
            let v = b.push_load::<false>(ValueId::Varnode(*r), *size, *space).id();
            args.push(v);
        }
    }
    ctx.replace_instruction_mnemonic(call_id, Mnemonic::Call(Call { target, args, clobbers }));
    Instruction::from_id_mut(ctx, call_id).set_type(writeset_ty);

    // Outputs: replay the returned final values into register space.
    let Some(cont) = BasicBlock::from_id(ctx, call_block)
        .successors()
        .next()
        .map(|(_, b)| b)
    else {
        return;
    };
    let result = ValueId::Instruction(call_id);
    let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, cont));
    b.set_insert_point_to_start();
    for (i, (r, _size, space)) in output_meta.iter().enumerate() {
        let v = ValueId::Instruction(b.push_extract(result, i).id);
        b.push_store(v, ValueId::Varnode(*r), *space);
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

#[derive(Default)]
pub struct ArgPromoteRegisters;

impl Pass for ArgPromoteRegisters {
    const NAME: &'static str = "argpromote_registers";
    fn description(&self) -> &'static str {
        "Functionalize register side effects into a returned write-set (runs early)"
    }
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        Ok(argpromote_registers(ctx))
    }
}

crate::register_module_pass!(ArgPromoteRegisters);

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

    /// Phase 0 gate for the register channel (see `ARGPROMOTE_REGISTERS.md`): the
    /// early register run will send an **aggregate-typed call result** through
    /// mem2reg/gvn/dce for the first time (today's RAM run is post-mem2reg, so
    /// aggregates never reach those passes). An opaque call result is the riskiest
    /// shape — gvn cannot fold it, so the `extract` must thread through intact.
    /// This builds exactly that and asserts the value passes neither panic nor
    /// mangle it.
    #[test]
    fn aggregate_call_result_survives_value_passes() {
        let mut tc = qcode::testing::TestContext::new();
        let (r0, r1, r3) = (tc.r0, tc.r1, tc.r3);
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry>
                    %v = load(i64, {r1});
                    %s = %v + i64 5;
                    store({r0}, %s);
                    %fin = load(i64, {r0});
                    %agg = (%fin);
                    return [%agg];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = (f, g, r0, r1);

        // A one-field aggregate (the positional register write-set shape).
        let i64_ty = tc.ctx.types.get_or_make_int(8);
        let agg_ty = tc.ctx.types.get_or_make_aggregate(vec![i64_ty]);

        // Make the call produce that aggregate, then replay field 0 into r3.
        let call_id = set_call(&mut tc, g_call, f, vec![]);
        Instruction::from_id_mut(&mut tc.ctx, call_id).set_type(agg_ty);
        tc.ctx.add_cfg_edge(g_call, g_cont);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, g_cont));
            b.set_insert_point_to_start();
            let out = ValueId::Instruction(b.push_extract(ValueId::Instruction(call_id), 0).id);
            b.push_store(out, ValueId::Varnode(r3), tc.reg_space);
        }

        // The Phase 0 question: do the value passes survive an aggregate-typed,
        // opaque call result? (A panic here fails the test.)
        let aliases = crate::AliasResult::simple(&tc.ctx);
        let _ = crate::mem2reg(&mut tc.ctx, g, &aliases);
        let _ = crate::gvn_function(&mut tc.ctx, g, Some(&aliases));

        // gvn cannot see into the opaque call result, so the extract must remain.
        let has_extract = Function::from_id(&tc.ctx, g)
            .iter()
            .any(|b| b.iter().any(|i| matches!(i.mnemonic(), Mnemonic::Extract(_))));
        assert!(has_extract, "extract of an aggregate call result must survive mem2reg+gvn");
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

    // ---- Phase 1: register channel — callee-side scan + rewrite -------------

    /// The width of the flat register write-set on `fid`'s (first) return: the
    /// field count of the `Tuple` in `Return::value`, or `None` if unset.
    fn register_writeset_len(tc: &qcode::testing::TestContext, fid: FunctionId) -> Option<usize> {
        let ret = Function::from_id(&tc.ctx, fid).iter().find_map(|b| {
            let last = b.iter().last()?;
            matches!(last.mnemonic(), Mnemonic::Return(_)).then_some(last.id)
        })?;
        let val = match tc.ctx.get_insn(ret).mnemonic() {
            Mnemonic::Return(r) => r.value?,
            _ => return None,
        };
        let ValueId::Instruction(tuple_id) = val else {
            return None;
        };
        match tc.ctx.get_insn(tuple_id).mnemonic() {
            Mnemonic::Tuple(t) => Some(t.fields.len()),
            _ => None,
        }
    }

    /// `void f() { r0 = 42; }` — a pure clobber: no inputs, one output, and a
    /// one-field write-set. The return register has no special status.
    #[test]
    fn register_pure_clobber() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    store({r0}, i64 42);
                    return [i64 0];
            "
        );
        let _ = entry;
        let eff = scan_register_effects(&tc.ctx, f).expect("a written register is promotable");
        assert_eq!(eff.outputs, vec![r0]);
        // The output is over-approximated as an input too (seeded so a no-write
        // path reads the caller's incoming value); the seed is dead here and DCE
        // would prune it.
        assert_eq!(eff.inputs, vec![r0]);

        rewrite_callee_registers(&mut tc.ctx, f, &eff);
        assert_eq!(register_writeset_len(&tc, f), Some(1), "one output → one field");
    }

    /// `r0 += 1` — read-modify-write: r0 is both input and output. The input
    /// becomes a by-value param seeded into register space at entry.
    #[test]
    fn register_read_modify_write() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    %v = load(i64, {r0});
                    %s = %v + i64 1;
                    store({r0}, %s);
                    return [i64 0];
            "
        );
        let eff = scan_register_effects(&tc.ctx, f).expect("written register");
        assert_eq!(eff.inputs, vec![r0]);
        assert_eq!(eff.outputs, vec![r0]);

        rewrite_callee_registers(&mut tc.ctx, f, &eff);
        // One by-value input param added, and its seed store is the new entry head.
        assert_eq!(BasicBlock::from_id(&tc.ctx, entry).params().count(), 1);
        let first = BasicBlock::from_id(&tc.ctx, entry).iter().next().unwrap();
        assert!(
            matches!(first.mnemonic(), Mnemonic::Store(s) if s.ptr == ValueId::Varnode(r0)),
            "entry must begin by seeding r0 from its input param"
        );
        assert_eq!(register_writeset_len(&tc, f), Some(1));
    }

    /// Writes to `r0_lo32` and `r0` overlap; the write-set canonicalizes to the
    /// coarsest covering register (`r0`), so replay stays order-independent.
    #[test]
    fn register_overlap_canonicalizes_to_coarsest() {
        let mut tc = qcode::testing::TestContext::new();
        let (r0, r0_lo32) = (tc.r0, tc.r0_lo32);
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    store({r0_lo32}, i32 2);
                    store({r0}, i64 3);
                    return [i64 0];
            "
        );
        let eff = scan_register_effects(&tc.ctx, f).expect("written register");
        assert_eq!(eff.outputs, vec![r0], "overlap group collapses to the 8-byte r0");
        let _ = r0_lo32;
        rewrite_callee_registers(&mut tc.ctx, f, &eff);
        assert_eq!(register_writeset_len(&tc, f), Some(1));
    }

    /// `r0` (the return register) and `r1` (a clobber) are written. Both ride the
    /// single write-set — the return value is just another output register.
    #[test]
    fn register_return_value_folds_into_writeset() {
        let mut tc = qcode::testing::TestContext::new();
        let (r0, r1) = (tc.r0, tc.r1);
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    store({r0}, i64 7);
                    store({r1}, i64 8);
                    return [i64 0];
            "
        );
        let eff = scan_register_effects(&tc.ctx, f).expect("written register");
        assert_eq!(eff.outputs, vec![r0, r1], "both outputs, sorted by address");
        rewrite_callee_registers(&mut tc.ctx, f, &eff);
        assert_eq!(register_writeset_len(&tc, f), Some(2), "return reg folds in");
    }

    /// A function that only *reads* registers has nothing to functionalize.
    #[test]
    fn register_read_only_is_skipped() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    %v = load(i64, {r0});
                    return [i64 0];
            "
        );
        assert!(scan_register_effects(&tc.ctx, f).is_none(), "no write ⇒ nothing to promote");
    }

    // ---- Phase 2: register channel — caller replay + precision --------------

    /// `void f() { r0 = 42; }` called by `g`, which reads r0 afterwards. The
    /// caller replays the returned r0, and the following mem2reg forwards that
    /// replay into the post-call read — precise dataflow, not an opaque reload.
    #[test]
    fn register_caller_replays_output_precisely() {
        let mut tc = qcode::testing::TestContext::new();
        let (r0, r1) = (tc.r0, tc.r1);
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry>
                    store({r0}, i64 42);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    %x = load(i64, {r0});
                    store({r1}, %x);
                    return [i64 0];
            "
        );
        let _ = (f, g, r0);
        set_call(&mut tc, g_call, f, vec![]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote_registers(&mut tc.ctx), "f's register clobber should be promoted");

        // The continuation replays the returned register from the call result.
        let has_extract = BasicBlock::from_id(&tc.ctx, g_cont)
            .iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Extract(_)));
        assert!(has_extract, "continuation must extract the returned register value");

        // mem2reg realizes precision: r1 receives the extracted value, not a reload.
        let aliases = crate::AliasResult::simple(&tc.ctx);
        crate::mem2reg(&mut tc.ctx, g, &aliases);

        let stored_to_r1 = BasicBlock::from_id(&tc.ctx, g_cont).iter().find_map(|i| {
            match i.mnemonic() {
                Mnemonic::Store(s) if s.ptr == ValueId::Varnode(r1) => Some(s.src),
                _ => None,
            }
        });
        let src = stored_to_r1.expect("store to r1 present");
        let is_extract = matches!(
            src,
            ValueId::Instruction(id) if matches!(tc.ctx.get_insn(id).mnemonic(), Mnemonic::Extract(_))
        );
        assert!(is_extract, "r1 must receive the precise extracted return value, got {src:?}");
    }

    /// An RMW callee's input register is passed as a positional call argument.
    #[test]
    fn register_caller_passes_input_arg() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry>
                    %v = load(i64, {r0});
                    %s = %v + i64 1;
                    store({r0}, %s);
                    return [i64 0];

            fn g:
                <g_entry>
                    store({r0}, i64 10);
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = (f, g, r0);
        let call_id = set_call(&mut tc, g_call, f, vec![]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote_registers(&mut tc.ctx));

        // f gained exactly one by-value input param (r0).
        let root = Function::from_id(&tc.ctx, f).root().unwrap().id;
        assert_eq!(BasicBlock::from_id(&tc.ctx, root).params().count(), 1);

        // The call passes that input as one positional argument.
        let args_len = match tc.ctx.get_insn(call_id).mnemonic() {
            Mnemonic::Call(c) => c.args.len(),
            _ => unreachable!(),
        };
        assert_eq!(args_len, 1, "the input register is passed as one call argument");
    }

    /// Every return block gets its own write-set (the design's per-return rule).
    #[test]
    fn register_multiple_returns_each_get_writeset() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    store({r0}, i64 1);
                    return [i64 0];
                <other>
                    store({r0}, i64 2);
                    return [i64 0];
            "
        );
        let _ = (r0, entry, other);
        let eff = scan_register_effects(&tc.ctx, f).expect("written register");
        rewrite_callee_registers(&mut tc.ctx, f, &eff);
        let with_writeset = Function::from_id(&tc.ctx, f)
            .iter()
            .filter(|b| {
                b.iter().last().is_some_and(
                    |i| matches!(i.mnemonic(), Mnemonic::Return(r) if r.value.is_some()),
                )
            })
            .count();
        assert_eq!(with_writeset, 2, "every return carries the register write-set");
    }

    /// Phase 3: emulator round-trip. The rewritten module must produce the same
    /// caller-visible register state as the original. `g` calls an RMW `f`
    /// (`r0 += 1`); we pin `f`'s return at `g`'s continuation address so the
    /// nested return works, then emulate before and after the rewrite and compare
    /// `r0`. This exercises the whole functional path: input-param seeding, the
    /// returned aggregate (`Return.value`), and the caller's `extract`/replay.
    #[test]
    fn register_roundtrip_preserves_caller_visible_state() {
        use qcode_emulator::StandaloneEmulator;

        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry>
                    %v = load(i64, {r0});
                    %s = %v + i64 1;
                    store({r0}, %s);
                    return [i64 4096];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = (f, r0);
        // `f` returns to `g`'s continuation: pin it at the return address (4096).
        BasicBlock::from_id_mut(&mut tc.ctx, g_cont).set_address(4096).unwrap();
        set_call(&mut tc, g_call, f, vec![]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        let g_root = Function::from_id(&tc.ctx, g).root().unwrap().id;
        let run = |ctx: &Context<'_>| -> u64 {
            let mut emu = StandaloneEmulator::new(g_root);
            emu.set_varnode(ctx, r0, 10).unwrap();
            emu.run_function(ctx, g).unwrap();
            emu.read_varnode(ctx, r0).unwrap()
        };

        let original = run(&tc.ctx);
        assert!(argpromote_registers(&mut tc.ctx));
        let transformed = run(&tc.ctx);

        assert_eq!(original, 11, "original: r0 = 10 + 1");
        assert_eq!(transformed, original, "rewrite preserves caller-visible r0");
    }

    /// Two registers that overlap but where neither contains the other (no single
    /// covering register) leave the function on the conservative path.
    #[test]
    fn register_partial_overlap_no_cover_bails() {
        let mut tc = qcode::testing::TestContext::new();
        let r0_lo32 = tc.r0_lo32; // bytes [0, 4)
        // A 4-byte register at offset 2 → bytes [2, 6): overlaps r0_lo32 at [2,4)
        // but neither interval contains the other.
        let mid = Varnode::make(&mut tc.ctx, 2, 4, tc.reg_space).id;
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    store({r0_lo32}, i32 1);
                    store({mid}, i32 2);
                    return [i64 0];
            "
        );
        let _ = (r0_lo32, mid, entry);
        assert!(
            scan_register_effects(&tc.ctx, f).is_none(),
            "partial overlap with no covering register must bail"
        );
    }
}
