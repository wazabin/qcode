//! Bottom-up inference of per-parameter pointer attributes (`readonly` /
//! `nocapture`) for functionalized (`pure_reg`) functions.
//!
//! See `plans/alias-overhaul/07-param-attrs.md` for the full rule set. The pass
//! runs an **optimistic module fixpoint**
//! (mirroring [`mark_pure`](super::argpromote::mark_pure_functions)): every
//! pointer parameter starts fully permissive ([`ParamAttrs::OPTIMISTIC`]) and a
//! per-function body walk revokes bits until the whole module is stable. The
//! optimism lives only in that initial assumption; the leaf rules are
//! default-conservative, so any construct the walk cannot express revokes both
//! bits.
//!
//! Because a functionalized callee's positional arguments align with its root
//! block params by construction (see [`interface`](super::interface)), the
//! fixpoint edge — "argument `j` flows into callee param `j`" — is a plain index
//! lookup. External callees contribute their C-prototype attributes (read-only
//! from `const` pointees) through the same [`FunctionBody::param_attr`] channel.
//!
//! **`nocapture` has no consumer yet.** Only `readonly` is read today (by
//! `mem_forward`'s call-kill). `nocapture` is inferred here because the
//! frame-freshness reasoning of step 8
//! (`plans/alias-overhaul/08-nocapture-freshness.md`) needs it, and the bit is
//! exercised by this module's unit tests — do not "clean it up" as dead.

use rustc_hash::FxHashMap as HashMap;

use qcode::{
    context::Context,
    value::{FunctionBody, FunctionId, ParamAttrs, ValueId, insn::Mnemonic},
};

use crate::{Pass, PipelineEnv};

/// A parameter index set as a bitmask. Functions with more than 64 params are
/// left conservative (their attrs are never inferred), which is sound.
type Mask = u64;
const MAX_PARAMS: usize = 64;

/// Run the optimistic fixpoint and store the inferred attributes on every
/// eligible (`pure_reg`, non-external) function. Returns `true` if any stored
/// attribute vector changed.
pub fn infer_param_attrs(ctx: &mut Context) -> bool {
    !infer_param_attrs_changed_functions(ctx).is_empty()
}

fn infer_param_attrs_changed_functions(ctx: &mut Context) -> rustc_hash::FxHashSet<FunctionId> {
    // Eligible functions and their param counts. Only functionalized functions
    // have `param[i] ↔ arg[i]` alignment, so only they are inferred here.
    let eligible: Vec<(FunctionId, usize)> = ctx
        .function_ids()
        .into_iter()
        .filter_map(|fid| {
            let f = FunctionBody::from_id(ctx, fid);
            if f.is_external() || !f.is_pure_reg() {
                return None;
            }
            let n = f.root()?.params().count();
            (n > 0 && n <= MAX_PARAMS).then_some((fid, n))
        })
        .collect();

    // Optimistic seed: every param readonly + nocapture.
    let mut attrs: HashMap<FunctionId, Vec<ParamAttrs>> = eligible
        .iter()
        .map(|&(fid, n)| (fid, vec![ParamAttrs::OPTIMISTIC; n]))
        .collect();

    // Revoke to a fixpoint. Each function is recomputed from the optimistic seed
    // using the current (monotonically shrinking) callee attributes, so the whole
    // system decreases monotonically and converges.
    loop {
        let mut round_changed = false;
        for &(fid, n) in &eligible {
            let fresh = compute_function_attrs(ctx, fid, n, &attrs);
            if attrs.get(&fid).map(|v| v != &fresh).unwrap_or(true) {
                attrs.insert(fid, fresh);
                round_changed = true;
            }
        }
        if !round_changed {
            break;
        }
    }

    // Commit.
    let mut changed = rustc_hash::FxHashSet::default();
    for (fid, vec) in attrs {
        let prev = FunctionBody::from_id(ctx, fid)
            .param_attrs()
            .map(|a| a.to_vec());
        if prev.as_deref() != Some(vec.as_slice()) {
            changed.insert(fid);
        }
        FunctionBody::from_id_mut(ctx, fid).set_param_attrs(vec);
    }
    changed
}

/// The attributes for `fid`'s parameters, starting optimistic and revoking bits
/// as the body walk finds writes, captures, or escapes. `attrs` holds the
/// in-progress attributes of the other in-fixpoint (`pure_reg`) callees.
fn compute_function_attrs(
    ctx: &Context,
    fid: FunctionId,
    n_params: usize,
    attrs: &HashMap<FunctionId, Vec<ParamAttrs>>,
) -> Vec<ParamAttrs> {
    let mut result = vec![ParamAttrs::OPTIMISTIC; n_params];

    // 1. Forward taint: which SSA values are affine-derived from each param.
    let taint = compute_taint(ctx, fid, n_params);

    let m = |v: ValueId| taint.get(&v).copied().unwrap_or(0);

    // 2. Sinks: walk every instruction and revoke bits for the params whose
    //    derived values reach a write / capture / escape.
    for block in FunctionBody::from_id(ctx, fid).iter() {
        for insn in block.iter() {
            let func = insn.id.func;
            match insn.mnemonic() {
                // A store through a p-derived address writes through p (revoke
                // readonly); a store *of* a p-derived value writes the pointer
                // itself into memory (revoke nocapture).
                Mnemonic::Store(s) => {
                    revoke_readonly(&mut result, m(s.ptr.qualify(func)));
                    revoke_nocapture(&mut result, m(s.src.qualify(func)));
                }
                // A direct call: consult the callee's per-param attributes. An
                // argument flowing into a non-readonly / non-nocapture param
                // revokes the corresponding bit; a callee with no usable
                // attributes revokes both. A p-derived clobbered location means
                // the callee writes through p (revoke readonly).
                Mnemonic::Call(c) => {
                    for (j, &arg) in c.args.iter().enumerate() {
                        let mask = m(arg.qualify(func));
                        if mask == 0 {
                            continue;
                        }
                        match c
                            .target
                            .real()
                            .and_then(|target| callee_param_attr(ctx, attrs, target, j))
                        {
                            Some(a) => {
                                if !a.readonly {
                                    revoke_readonly(&mut result, mask);
                                }
                                if !a.nocapture {
                                    revoke_nocapture(&mut result, mask);
                                }
                            }
                            None => revoke_both(&mut result, mask),
                        }
                    }
                    for &cl in &c.clobbers {
                        revoke_readonly(&mut result, m(cl.qualify(func)));
                    }
                }
                // Indirect call / opaque p-code / computed branch: the value
                // escapes through an unanalyzable edge — revoke both bits.
                Mnemonic::CallInd(c) => {
                    for &arg in &c.args {
                        revoke_both(&mut result, m(arg.qualify(func)));
                    }
                }
                Mnemonic::BranchInd(b) => revoke_both(&mut result, m(b.ptr.qualify(func))),
                Mnemonic::PCodeOp(_) | Mnemonic::Map(_) | Mnemonic::Scan(_) => {
                    for arg in insn.operands() {
                        revoke_both(&mut result, m(arg));
                    }
                }
                _ => {}
            }
        }
    }

    result
}

/// The per-param taint of every SSA value in `fid`: value → bitmask of the
/// params it is affine-derived from. Propagation flows forward through every
/// value-producing instruction **except [`Load`](Mnemonic::Load)** — the
/// one-level (`*const`) semantics: a value *loaded from* a param is not itself
/// derived from the param. Over-approximating derivation (propagating through
/// any non-load op) only causes extra revocations, which is sound.
fn compute_taint(ctx: &Context, fid: FunctionId, n_params: usize) -> HashMap<ValueId, Mask> {
    let mut taint: HashMap<ValueId, Mask> = HashMap::default();

    if let Some(root) = FunctionBody::from_id(ctx, fid).root() {
        for (i, param) in root.params().enumerate().take(n_params) {
            taint.insert(ValueId::BlockParam(param.id), 1 << i);
        }
    }

    // Forward dataflow fixpoint. Loops / block params can carry taint back, so
    // iterate until no value's mask grows.
    let mut changed = true;
    while changed {
        changed = false;
        for block in FunctionBody::from_id(ctx, fid).iter() {
            for insn in block.iter() {
                if insn.size() == 0 || matches!(insn.mnemonic(), Mnemonic::Load(_)) {
                    continue;
                }
                let mut om: Mask = 0;
                for arg in insn.operands() {
                    om |= taint.get(&arg).copied().unwrap_or(0);
                }
                if om == 0 {
                    continue;
                }
                let r = ValueId::Instruction(insn.id);
                let entry = taint.entry(r).or_insert(0);
                if *entry & om != om {
                    *entry |= om;
                    changed = true;
                }
            }
        }
        // Block params also carry taint from their incoming branch arguments.
        if propagate_block_params(ctx, fid, &mut taint) {
            changed = true;
        }
    }

    taint
}

/// Propagate taint into merge/loop block params from the branch arguments that
/// feed them. Returns whether any param mask grew. (Root params are seeded
/// directly and never widened here.)
fn propagate_block_params(
    ctx: &Context,
    fid: FunctionId,
    taint: &mut HashMap<ValueId, Mask>,
) -> bool {
    use qcode::value::BasicBlock;
    let mut changed = false;
    let root = FunctionBody::from_id(ctx, fid).root().map(|b| b.id);
    for block in FunctionBody::from_id(ctx, fid).iter() {
        if Some(block.id) == root {
            continue;
        }
        let params: Vec<(usize, ValueId)> = block
            .params()
            .enumerate()
            .map(|(i, p)| (i, ValueId::BlockParam(p.id)))
            .collect();
        if params.is_empty() {
            continue;
        }
        let preds: Vec<_> = BasicBlock::from_id(ctx, block.id)
            .predecessors()
            .map(|(_, p)| p)
            .collect();
        for (index, pval) in params {
            let mut om: Mask = 0;
            for &pred in &preds {
                let Some(term) = BasicBlock::from_id(ctx, pred).iter().last() else {
                    continue;
                };
                for arg in incoming_args(term.mnemonic(), pred.func, block.id, index) {
                    om |= taint.get(&arg).copied().unwrap_or(0);
                }
            }
            if om == 0 {
                continue;
            }
            let entry = taint.entry(pval).or_insert(0);
            if *entry & om != om {
                *entry |= om;
                changed = true;
            }
        }
    }
    changed
}

/// The value(s) passed to param `index` of `target` by a terminator `m` owned by
/// function `func` (its arena, used to qualify the terminator's local targets).
fn incoming_args(
    m: &Mnemonic,
    func: qcode::value::FunctionId,
    target: qcode::value::BlockId,
    index: usize,
) -> Vec<ValueId> {
    use qcode::value::BlockId;
    let mut out = Vec::new();
    match m {
        Mnemonic::Branch(br) if BlockId::new(func, br.target) == target => {
            if let Some(&a) = br.args.get(index) {
                out.push(a.qualify(func));
            }
        }
        Mnemonic::CBranch(cb) => {
            if BlockId::new(func, cb.success_block) == target
                && let Some(&a) = cb.success_args.get(index)
            {
                out.push(a.qualify(func));
            }
            if BlockId::new(func, cb.failure_block) == target
                && let Some(&a) = cb.failure_args.get(index)
            {
                out.push(a.qualify(func));
            }
        }
        _ => {}
    }
    out
}

/// The attributes of `target`'s parameter `j`, preferring the in-fixpoint value
/// for a `pure_reg` callee and otherwise the stored attributes (externals, or a
/// callee with a C prototype). `None` when nothing is known — treat
/// conservatively.
fn callee_param_attr(
    ctx: &Context,
    attrs: &HashMap<FunctionId, Vec<ParamAttrs>>,
    target: FunctionId,
    j: usize,
) -> Option<ParamAttrs> {
    if let Some(v) = attrs.get(&target) {
        return v.get(j).copied();
    }
    FunctionBody::from_id(ctx, target).param_attr(j)
}

fn revoke_readonly(result: &mut [ParamAttrs], mask: Mask) {
    for_each_bit(mask, result.len(), |i| result[i].readonly = false);
}

fn revoke_nocapture(result: &mut [ParamAttrs], mask: Mask) {
    for_each_bit(mask, result.len(), |i| result[i].nocapture = false);
}

fn revoke_both(result: &mut [ParamAttrs], mask: Mask) {
    for_each_bit(mask, result.len(), |i| {
        result[i].readonly = false;
        result[i].nocapture = false;
    });
}

fn for_each_bit(mask: Mask, len: usize, mut f: impl FnMut(usize)) {
    let mut m = mask;
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        m &= m - 1;
        if i < len {
            f(i);
        }
    }
}

#[derive(Default)]
pub struct ParamAttrsPass;

impl Pass for ParamAttrsPass {
    const NAME: &'static str = "param_attrs";
    fn description(&self) -> &'static str {
        "Infer readonly/nocapture pointer attributes for functionalized functions"
    }
    fn run(
        &self,
        ctx: &mut Context,
        _env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        Ok(crate::ModulePassOutcome::functions(
            infer_param_attrs_changed_functions(ctx),
        ))
    }
}

crate::register_module_pass!(ParamAttrsPass);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{
            BasicBlock, FunctionBody, Value,
            insn::{Call, CallInd},
        },
    };

    /// Build a `pure_reg` function rooted at `addr` with `nparams` 8-byte root
    /// params; `body` receives the builder and the param `ValueId`s. Marked
    /// `pure_reg` so the inference pass treats it as eligible.
    fn build_pure_fn(
        tc: &mut TestContext,
        name: &'static str,
        addr: u64,
        nparams: usize,
        body: impl FnOnce(&mut Builder<'static, '_>, &[ValueId]),
    ) -> FunctionId {
        let fid = FunctionBody::make(&mut tc.ctx, name.into()).unwrap().id;
        let block = { tc.ctx.get_or_make_block(addr, fid) };
        FunctionBody::from_id_mut(&mut tc.ctx, fid)
            .set_root(block)
            .unwrap();
        let mut params = Vec::new();
        for _ in 0..nparams {
            let pid = BasicBlock::from_id_mut(&mut tc.ctx, block).push_param(8).id;
            params.push(ValueId::BlockParam(pid));
        }
        let mut b = (&mut tc.ctx).builder_at(addr);
        body(&mut b, &params);
        unsafe { b.dont_finalize() };
        drop(b);
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_pure_reg(true);
        fid
    }

    /// Overwrite the (empty) arg list of the direct call instruction `cid`.
    fn set_call_args(
        tc: &mut TestContext,
        cid: qcode::value::InstructionId,
        target: FunctionId,
        args: Vec<ValueId>,
    ) {
        tc.ctx.replace_instruction_mnemonic(
            cid,
            Mnemonic::Call(Call {
                target: qcode::value::insn::Callee::Real(target),
                args: args.into_iter().map(|arg| arg.localize(cid.func)).collect(),
                clobbers: vec![],
            }),
        );
    }

    #[test]
    fn store_through_param_revokes_readonly_only() {
        let mut tc = TestContext::new();
        let ram = tc.ctx.shared.default_space;
        let f = build_pure_fn(&mut tc, "f", 0x1000, 1, |b, p| {
            let v = b.shr().get_const(7, 8);
            b.push_store(v, p[0], ram);
        });
        infer_param_attrs(&mut tc.ctx);
        let a = FunctionBody::from_id(&tc.ctx, f).param_attr(0).unwrap();
        assert!(!a.readonly, "a store through the param writes through it");
        assert!(a.nocapture, "the param value itself is never stored");
    }

    #[test]
    fn store_of_param_revokes_nocapture_only() {
        // f(dst, p): *dst = p — the pointer p is written to memory (captured),
        // but nothing is written *through* p.
        let mut tc = TestContext::new();
        let ram = tc.ctx.shared.default_space;
        let f = build_pure_fn(&mut tc, "f", 0x1000, 2, |b, p| {
            b.push_store(p[1], p[0], ram);
        });
        infer_param_attrs(&mut tc.ctx);
        let dst = FunctionBody::from_id(&tc.ctx, f).param_attr(0).unwrap();
        let p = FunctionBody::from_id(&tc.ctx, f).param_attr(1).unwrap();
        assert!(!dst.readonly, "dst is a store address");
        assert!(dst.nocapture, "dst is not itself stored");
        assert!(p.readonly, "p is never a store address");
        assert!(!p.nocapture, "p is written into memory → captured");
    }

    #[test]
    fn read_only_param_keeps_both_bits() {
        // f(p): return *p — a pure read through p.
        let mut tc = TestContext::new();
        let ram = tc.ctx.shared.default_space;
        let f = build_pure_fn(&mut tc, "f", 0x1000, 1, |b, p| {
            let x = b.push_load::<false>(p[0], 8, ram).id();
            b.push_return(x);
        });
        infer_param_attrs(&mut tc.ctx);
        let a = FunctionBody::from_id(&tc.ctx, f).param_attr(0).unwrap();
        assert!(a.readonly && a.nocapture, "a pure read revokes nothing");
    }

    #[test]
    fn forwarding_to_readonly_callee_preserves_readonly() {
        let mut tc = TestContext::new();
        let ram = tc.ctx.shared.default_space;
        // callee(p): return *p — readonly + nocapture.
        let callee = build_pure_fn(&mut tc, "callee", 0x2000, 1, |b, p| {
            let x = b.push_load::<false>(p[0], 8, ram).id();
            b.push_return(x);
        });
        // caller(p): callee(p).
        let caller = build_pure_fn(&mut tc, "caller", 0x1000, 1, |b, _p| {
            b.push_call(callee);
        });
        // Wire the argument (the builder created the call with no args).
        let cid = call_insn(&tc, caller);
        let p0 = root_param(&tc, caller, 0);
        set_call_args(&mut tc, cid, callee, vec![p0]);

        infer_param_attrs(&mut tc.ctx);
        let a = FunctionBody::from_id(&tc.ctx, caller)
            .param_attr(0)
            .unwrap();
        assert!(
            a.readonly && a.nocapture,
            "forwarding into a readonly+nocapture param preserves both"
        );
    }

    #[test]
    fn forwarding_to_writing_callee_revokes_readonly() {
        let mut tc = TestContext::new();
        let ram = tc.ctx.shared.default_space;
        // callee(p): *p = 7 — not readonly.
        let callee = build_pure_fn(&mut tc, "callee", 0x2000, 1, |b, p| {
            let v = b.shr().get_const(7, 8);
            b.push_store(v, p[0], ram);
        });
        let caller = build_pure_fn(&mut tc, "caller", 0x1000, 1, |b, _p| {
            b.push_call(callee);
        });
        let cid = call_insn(&tc, caller);
        let p0 = root_param(&tc, caller, 0);
        set_call_args(&mut tc, cid, callee, vec![p0]);

        infer_param_attrs(&mut tc.ctx);
        let a = FunctionBody::from_id(&tc.ctx, caller)
            .param_attr(0)
            .unwrap();
        assert!(
            !a.readonly,
            "forwarding into a non-readonly callee param revokes readonly"
        );
    }

    #[test]
    fn indirect_call_revokes_both() {
        let mut tc = TestContext::new();
        let f = build_pure_fn(&mut tc, "f", 0x1000, 1, |b, _p| {
            let target = b.shr().get_const(0x9999, 8);
            b.push_call_ind(target);
        });
        // Give the indirect call an argument of p (reusing its existing ptr).
        let cid = call_insn(&tc, f);
        let p0 = root_param(&tc, f, 0);
        let ptr = match tc.ctx.get_insn(cid).mnemonic() {
            Mnemonic::CallInd(c) => c.ptr,
            _ => unreachable!(),
        };
        tc.ctx.replace_instruction_mnemonic(
            cid,
            Mnemonic::CallInd(CallInd {
                ptr,
                args: vec![p0.localize(cid.func)],
            }),
        );
        infer_param_attrs(&mut tc.ctx);
        let a = FunctionBody::from_id(&tc.ctx, f).param_attr(0).unwrap();
        assert!(
            !a.readonly && !a.nocapture,
            "an argument escaping through an indirect call revokes both bits"
        );
    }

    // --- small helpers over the built IR ---

    fn call_insn(tc: &TestContext, fid: FunctionId) -> qcode::value::InstructionId {
        FunctionBody::from_id(&tc.ctx, fid)
            .blocks()
            .flat_map(|b| b.iter().map(|i| i.id).collect::<Vec<_>>())
            .find(|&id| {
                matches!(
                    tc.ctx.get_insn(id).mnemonic(),
                    Mnemonic::Call(_) | Mnemonic::CallInd(_)
                )
            })
            .expect("function has a call")
    }

    fn root_param(tc: &TestContext, fid: FunctionId, index: usize) -> ValueId {
        let pid = FunctionBody::from_id(&tc.ctx, fid)
            .root()
            .unwrap()
            .params()
            .nth(index)
            .unwrap()
            .id;
        ValueId::BlockParam(pid)
    }
}
