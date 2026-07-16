//! Interprocedural memory-write summaries: which memory spaces each function may
//! store to.
//!
//! [`set_all_written_spaces`] computes, for every non-external function, the
//! exact set of non-register memory [`SpaceId`]s it may write — directly in its
//! own body, or transitively through a (direct, resolved) callee. The result is
//! recorded on the function's
//! [`written_spaces`](qcode::value::FunctionRef::written_spaces)
//! signature field.
//!
//! The consumer is store-to-load forwarding's call-prune
//! ([`crate::gvn::mem_forward`]): a call to a callee whose witnessed write-set
//! does **not** include a cell's space cannot clobber that cell, so the forwarded
//! value survives the call. This is what keeps a buffer pointer spilled into the
//! caller's frame forwardable across a call to a functionalized (`pure_reg`)
//! callee that writes only its own private scratch space — never the real `ram`
//! the buffer lives in.
//!
//! A function whose effect cannot be bounded — it makes an indirect call, or
//! calls an external function — is recorded `None` (unknown / may write any
//! space), the conservative answer the prune already assumes.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    context::Context,
    space::{Space, SpaceId, SpaceType},
    value::{FunctionBody, FunctionId, insn::Mnemonic},
};

/// A function's witnessed memory-write effect: either a bounded set of
/// non-register spaces, or unbounded (an indirect/external call leaks the effect
/// out of view).
#[derive(Clone)]
enum Effect {
    Bounded(HashSet<SpaceId>),
    Unbounded,
}

/// The *local* write effect of `function_id`: the non-register spaces its own
/// body stores to, the resolved functions it calls, and whether it makes a call
/// whose effect cannot be bounded (indirect, or to an external function).
fn local_effect(
    ctx: &Context,
    function_id: FunctionId,
) -> (HashSet<SpaceId>, Vec<FunctionId>, bool) {
    let mut spaces = HashSet::default();
    let mut callees = Vec::new();
    let mut unbounded = false;
    for block in FunctionBody::from_id(ctx, function_id).blocks() {
        for insn in block.iter() {
            match insn.mnemonic() {
                // Register writes are tracked separately (`clobbered_regs`); only
                // memory spaces matter to the forwarding prune.
                Mnemonic::Store(s) => {
                    // Function-local temporary writes never escape into the
                    // published interprocedural shared-space summary.
                    if let Some(space) = s.space.shared()
                        && !matches!(Space::from_id(ctx, space).ty, SpaceType::Register)
                    {
                        spaces.insert(space);
                    }
                }
                Mnemonic::Call(call) => {
                    let Some(target) = call.target.real() else {
                        unbounded = true;
                        continue;
                    };
                    if FunctionBody::from_id(ctx, target).is_external() {
                        unbounded = true;
                    } else {
                        callees.push(target);
                    }
                }
                // An unresolved indirect transfer leaves to unknown code — an
                // indirect call, or an indirect *tail-branch* (the shape a PLT
                // stub lifts to: `goto [GOT_slot]`). Either can write anything, so
                // the effect is unbounded. A `BranchInd` that stayed in-function
                // would have been resolved to a direct `Branch` by the jump-table
                // pass; one that survives is a genuine escape.
                Mnemonic::CallInd(_) | Mnemonic::BranchInd(_) => unbounded = true,
                _ => {}
            }
        }
    }
    (spaces, callees, unbounded)
}

/// Compute and record [`written_spaces`](qcode::value::FunctionRef::written_spaces)
/// for every non-external function as a least fixpoint over the call graph: a
/// function's write-set is its own stores unioned with every resolved callee's
/// write-set, becoming unbounded as soon as any (transitive) callee is unbounded.
///
/// Sound and order-independent: the per-function effect only ever grows, and an
/// unbounded effect (`None`) is the conservative value the prune already assumes,
/// so an under-approximation is impossible.
pub fn set_all_written_spaces(ctx: &mut Context) {
    let ids: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| !f.is_external())
        .map(|f| f.id)
        .collect();

    // Seed each function with its local effect, remembering callees for the
    // transitive union below.
    let mut effect: HashMap<FunctionId, Effect> = HashMap::default();
    let mut callees: HashMap<FunctionId, Vec<FunctionId>> = HashMap::default();
    for &id in &ids {
        let (spaces, cs, unbounded) = local_effect(ctx, id);
        effect.insert(
            id,
            if unbounded {
                Effect::Unbounded
            } else {
                Effect::Bounded(spaces)
            },
        );
        callees.insert(id, cs);
    }

    // Propagate to a fixpoint: a callee's spaces flow into its callers; a callee
    // going unbounded makes its callers unbounded. A call to a function not in
    // `effect` (should not happen for resolved direct calls) is treated as
    // unbounded, never silently dropped.
    let mut changed = true;
    while changed {
        changed = false;
        for &id in &ids {
            if matches!(effect[&id], Effect::Unbounded) {
                continue;
            }
            let mut go_unbounded = false;
            let mut additions: HashSet<SpaceId> = HashSet::default();
            for &c in &callees[&id] {
                match effect.get(&c) {
                    Some(Effect::Bounded(cs)) => additions.extend(cs.iter().copied()),
                    _ => {
                        go_unbounded = true;
                        break;
                    }
                }
            }
            match effect.get_mut(&id).expect("seeded above") {
                Effect::Unbounded => {}
                Effect::Bounded(_) if go_unbounded => {
                    effect.insert(id, Effect::Unbounded);
                    changed = true;
                }
                Effect::Bounded(set) => {
                    let before = set.len();
                    set.extend(additions);
                    if set.len() != before {
                        changed = true;
                    }
                }
            }
        }
    }

    for &id in &ids {
        let summary = match &effect[&id] {
            Effect::Unbounded => None,
            Effect::Bounded(set) => {
                let mut v: Vec<SpaceId> = set.iter().copied().collect();
                v.sort_by_key(|&s| usize::from(s));
                Some(v)
            }
        };
        FunctionBody::from_id_mut(ctx, id).set_written_spaces(summary);
    }
}

// ----- pass ------------------------------------------------------------------

use crate::{Pass, PipelineEnv};

#[derive(Default)]
pub struct SeedWrittenSpaces;

impl Pass for SeedWrittenSpaces {
    const NAME: &'static str = "seed_written_spaces";
    fn description(&self) -> &'static str {
        "Infer each function's transitive memory-write space set (for the GVN forwarding call-prune)"
    }
    fn run(
        &self,
        ctx: &mut Context,
        _env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        let affected: Vec<FunctionId> = ctx
            .functions()
            .filter(|f| !f.is_external())
            .map(|f| f.id)
            .collect();
        set_all_written_spaces(ctx);
        Ok(crate::ModulePassOutcome::functions(affected)
            .preserving_global::<crate::CallGraphAnalysis>()
            .preserving_global::<crate::AddressAnalysis>())
    }
}

crate::register_module_pass!(SeedWrittenSpaces);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AliasResult, constant_fold_function, gvn_function};
    use qcode::{
        context::Context,
        lower::lower_str,
        testing::TestContext,
        value::{BasicBlock, FunctionBody, TempSpace, ValueId},
    };

    /// The motivating shape, reduced: a buffer pointer is spilled into a frame
    /// slot, the block ends in a `call` to a callee that writes only its own
    /// scratch space, the pointer is reloaded after the call and the buffer is
    /// filled with constants, with an unrelated frame-slot store in between, then
    /// read back by a wide load. The wide load can only concretize if the spilled
    /// reload forwards across the call — which requires the spill cell to survive
    /// the call-prune. It does precisely because the callee's witnessed write-set
    /// excludes the buffer's space.
    const SPILL_CALL_RELOAD: &str = "\
fn caller:
<entry @sp:i32>
    i32 %bufptr = i32 @sp - i32 0x1a0;
    i32 %slot = i32 @sp - i32 0x78;
    store(ram:4, i32 %slot <- i32 %bufptr);
    i32 %argp = i32 @sp - i32 0xc04;
    call fn callee(@cp=i32 %argp) // -> <resume>;
<resume>
    i32 %buf = load(ram:4, i32 %slot);
    store(ram:4, i32 %buf <- i32 0x97079551);
    i32 %a1 = i32 %buf + i32 0x4;
    store(ram:4, i32 %a1 <- i32 0x9306913e);
    i32 %a2 = i32 %buf + i32 0x8;
    store(ram:4, i32 %a2 <- i32 0x9f309d3b);
    i32 %fslot = i32 @sp - i32 0xc00;
    store(ram:4, i32 %fslot <- i32 0xdeadbeef);
    i96 %v = load(ram:12, i32 %buf);
    return i96 %v;
fn callee:
<centry @cp:i32>
    store(scratch:4, i32 @cp <- i32 0x0);
    i32 %x = i32 @cp + i32 0x4;
    store(scratch:4, i32 %x <- i32 0x1);
    return at i32 @cp;
";

    /// The same shape but with `<header>` a *loop* header (back-edge): the buffer
    /// fill now runs through the not-yet-forwarded reload inside the loop body, so
    /// the loop-carried prune would drop the spill cell — the chicken-and-egg —
    /// unless it peels the reload (gated on `LoadedPointerDisjointFromSlot`).
    const LOOP_SPILL_RELOAD: &str = "\
fn caller:
<entry @sp:i32>
    i32 %bufptr = i32 @sp - i32 0x1a0;
    i32 %slot = i32 @sp - i32 0x78;
    store(ram:4, i32 %slot <- i32 %bufptr);
    i32 %argp = i32 @sp - i32 0xc04;
    call fn callee(@cp=i32 %argp) // -> <header>;
<header @i:i32>
    i32 %buf = load(ram:4, i32 %slot);
    store(ram:4, i32 %buf <- i32 0x97079551);
    i32 %a1 = i32 %buf + i32 0x4;
    store(ram:4, i32 %a1 <- i32 0x9306913e);
    i32 %a2 = i32 %buf + i32 0x8;
    store(ram:4, i32 %a2 <- i32 0x9f309d3b);
    i32 %fslot = i32 @sp - i32 0xc00;
    store(ram:4, i32 %fslot <- i32 0xdeadbeef);
    i96 %v = load(ram:12, i32 %buf);
    i32 %n = i32 @i + i32 0x1;
    i8 %more = i32 %n < i32 0x10;
    if i8 %more goto <header @i=i32 %n> else goto <exit>;
<exit>
    return i96 %v;
fn callee:
<centry @cp:i32>
    store(scratch:4, i32 @cp <- i32 0x0);
    return at i32 @cp;
";

    /// Run fold+gvn to a small fixpoint and return the rendered `caller`.
    fn optimize_caller(seed_summaries: bool) -> String {
        let mut ctx = Context::new();
        let syms = lower_str(&mut ctx, SPILL_CALL_RELOAD).expect("parse");
        let caller = syms.functions["caller"];
        if seed_summaries {
            set_all_written_spaces(&mut ctx);
        }
        for _ in 0..2 {
            constant_fold_function(&mut ctx, caller);
            let aliases = AliasResult::simple_for_function(&ctx, caller);
            gvn_function(&mut ctx, caller, Some(&aliases));
        }
        format!("{}", FunctionBody::from_id(&ctx, caller))
    }

    /// Lower the loop shape, seed write-spaces, optionally record
    /// `LoadedPointerDisjointFromSlot`, run fold+gvn, return the rendered `caller`.
    fn optimize_loop_caller(assume_disjoint: bool) -> String {
        use qcode::assumption::Proposition;
        let mut ctx = Context::new();
        let syms = lower_str(&mut ctx, LOOP_SPILL_RELOAD).expect("parse");
        let caller = syms.functions["caller"];
        set_all_written_spaces(&mut ctx);
        if assume_disjoint {
            ctx.assume_true(Proposition::LoadedPointerDisjointFromSlot(caller));
        }
        for _ in 0..3 {
            constant_fold_function(&mut ctx, caller);
            let aliases = AliasResult::simple_for_function(&ctx, caller);
            gvn_function(&mut ctx, caller, Some(&aliases));
        }
        format!("{}", FunctionBody::from_id(&ctx, caller))
    }

    /// A function that ends in an unresolved indirect tail-branch (`goto [p]` —
    /// the shape a PLT stub lifts to) escapes to unknown code, so its write-set is
    /// unbounded (`None`), not the empty set. Otherwise a caller would forward
    /// constant stores across the call, folding away input-dependent values.
    #[test]
    fn indirect_tailbranch_is_unbounded() {
        use qcode_macro::qcode;
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn stub:
            <entry>
                i64 %p = load(ram:8, i64 0x3fa0);
                goto [i64 %p];
            "
        );
        let _ = entry;
        set_all_written_spaces(&mut ctx);
        assert!(
            FunctionBody::from_id(&ctx, stub).written_spaces().is_none(),
            "an unresolved indirect tail-branch must have an unbounded write-set"
        );
    }

    /// The witnessed write-set of the functionalized callee is exactly its scratch
    /// space — never the default `ram` the buffer lives in.
    #[test]
    fn callee_written_spaces_excludes_ram() {
        let mut ctx = Context::new();
        let syms = lower_str(&mut ctx, SPILL_CALL_RELOAD).expect("parse");
        set_all_written_spaces(&mut ctx);
        let callee = FunctionBody::from_id(&ctx, syms.functions["callee"]);
        let spaces = callee
            .written_spaces()
            .expect("callee write-set is bounded");
        assert!(
            !spaces.contains(&ctx.shared.default_space),
            "callee writes only scratch, never the default ram space"
        );
    }

    #[test]
    fn body_local_scratch_is_not_published() {
        let mut tc = TestContext::new();
        let fid = FunctionBody::make(&mut tc.ctx, "local_scratch".into())
            .unwrap()
            .id;
        let block = BasicBlock::make(&mut tc.ctx, fid).id;
        FunctionBody::from_id_mut(&mut tc.ctx, fid)
            .set_root(block)
            .unwrap();
        let scratch = tc.ctx.bodies[fid].push_temp_space(TempSpace::new(Some("scratch"), 1, 8));
        let ptr = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, block).push_param(8).id);
        {
            let mut b = (&mut tc.ctx).builder(block);
            let value = b.shr().get_const(1, 1);
            b.push_store(
                value,
                ptr,
                qcode::space::LocalMemorySpaceId::Temp(scratch.local),
            );
            b.push_return(value);
        }

        set_all_written_spaces(&mut tc.ctx);
        assert_eq!(
            FunctionBody::from_id(&tc.ctx, fid).written_spaces(),
            Some(&[][..]),
            "body-local scratch must not escape into a shared-space summary"
        );
    }

    /// With the write-set summary the spilled reload forwards across the call and
    /// the wide load coalesces to a constant byte blob.
    #[test]
    fn wide_load_concretizes_with_write_summary() {
        let out = optimize_caller(true);
        assert!(
            out.contains("b\"\\x51\\x95\\x07\\x97"),
            "wide load should concretize to a byte blob, got:\n{out}"
        );
    }

    /// Without it the call-prune conservatively drops the buffer cells, so the
    /// wide load stays an opaque memory read — the pre-fix behaviour.
    #[test]
    fn wide_load_stays_opaque_without_write_summary() {
        let out = optimize_caller(false);
        assert!(
            out.contains("load(ram:12"),
            "without the write summary the wide load must remain opaque, got:\n{out}"
        );
    }

    /// In the loop shape, under `LoadedPointerDisjointFromSlot` the loop-carried
    /// prune peels the buffer-fill stores' reload, so the spill cell survives, the
    /// reload forwards, and the wide load concretizes — breaking the chicken-and-egg.
    #[test]
    fn loop_wide_load_concretizes_under_assumption() {
        let out = optimize_loop_caller(true);
        assert!(
            out.contains("b\"\\x51\\x95\\x07\\x97"),
            "under the assumption the loop-carried wide load should concretize, got:\n{out}"
        );
    }

    /// Without the assumption the reload cannot be soundly peeled (a self-referential
    /// slot could be overwritten through it), so the prune stays conservative and the
    /// wide load remains opaque.
    #[test]
    fn loop_wide_load_stays_opaque_without_assumption() {
        let out = optimize_loop_caller(false);
        assert!(
            out.contains("load(ram:12"),
            "without the assumption the loop-carried wide load must remain opaque, got:\n{out}"
        );
    }
}
