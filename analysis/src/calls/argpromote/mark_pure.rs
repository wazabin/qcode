use qcode::{
    context::Context,
    space::LocalMemorySpaceId,
    value::{FunctionBody, FunctionId, insn::Mnemonic},
};
use rustc_hash::FxHashSet;

use crate::{
    Pass, PipelineEnv,
    calls::{
        CallEdge,
        effect_engine::{EffectChannel, solve_summaries},
    },
};

/// Assert [`FunctionRef::is_pure`](qcode::value::FunctionRef::is_pure) on every
/// `pure_reg` function whose body has no
/// residual side effect — no memory access, no calls, and no raw register/global
/// (varnode) reads — so it is a deterministic pure function of its params.
///
/// This is the final argpromote step: the register/stack/memory channels each
/// functionalize one kind of effect, and a function with nothing left for any of
/// them is fully pure. Idempotent; returns `true` if any flag was newly set.
/// The [`verify`](crate::verify) pure-function rule re-checks this invariant.
pub fn mark_pure_functions(ctx: &mut Context) -> bool {
    let targets = ctx.function_ids();
    mark_pure_functions_targeted(ctx, &targets)
}

/// The purity [`EffectChannel`]: effects are the set of (transitively) reachable
/// direct callees; ⊤ is any body-local impurity, an external / bodyless callee,
/// or a call site with residual clobbers. A function is pure iff its summary is
/// solved *and* it cannot reach itself (recursion never flags — a recursive
/// "pure" call cannot be emulated or dead-call-deleted without a termination
/// argument, matching the pre-engine behaviour).
///
/// Externals are ⊤ leaves even when `Materialized` (a prototype bounds their
/// register interface; their memory behaviour — `printf` writing stdout — is
/// still an observable effect, so a call to one must never be deleted as a dead
/// pure call).
struct PureChannel;

impl EffectChannel for PureChannel {
    type Effects = FxHashSet<FunctionId>;

    fn scan(&self, ctx: &Context, fid: FunctionId) -> Option<Self::Effects> {
        body_locally_pure(ctx, fid).then(FxHashSet::default)
    }

    fn external_leaf(&self, _ctx: &Context, _fid: FunctionId) -> Option<Self::Effects> {
        None
    }

    fn transfer(
        &self,
        ctx: &Context,
        edge: &CallEdge,
        callee_eff: &Self::Effects,
    ) -> Option<Self::Effects> {
        // A `Call` site with residual clobbers is not register-transparent even
        // if its callee is pure — guard against a stale over-approximated
        // clobber set. A synthetic edge has no site to vet: conservative ⊤.
        let site = edge.site?;
        if let Mnemonic::Call(c) = ctx.get_insn(site).mnemonic()
            && !c.clobbers.is_empty()
        {
            return None;
        }
        let crate::calls::CallTarget::Function(callee) = edge.target else {
            return None;
        };
        let mut eff = callee_eff.clone();
        eff.insert(callee);
        Some(eff)
    }

    fn join(&self, into: &mut Self::Effects, from: &Self::Effects) -> bool {
        let before = into.len();
        into.extend(from.iter().copied());
        into.len() != before
    }
}

/// Solve the purity summary and collect the `targets` that are newly provably
/// pure (not already pure, reg-materialized, non-external, and unable to reach
/// themselves). Read-only: the caller stamps `is_pure` on the returned set.
fn newly_pure_functions(
    ctx: &Context,
    targets: impl IntoIterator<Item = FunctionId>,
) -> Vec<FunctionId> {
    let graph = crate::CallGraph::analyze(ctx);
    let summaries = solve_summaries(ctx, &graph, &PureChannel);
    targets
        .into_iter()
        .filter(|&fid| {
            let f = FunctionBody::from_id(ctx, fid);
            !f.is_pure()
                && f.is_reg_materialized()
                && !f.is_external()
                && matches!(summaries.get(fid), Ok(reachable) if !reachable.contains(&fid))
        })
        .collect()
}

fn mark_pure_functions_targeted(ctx: &mut Context, targets: &[FunctionId]) -> bool {
    let to_mark = newly_pure_functions(ctx, targets.iter().copied());
    for &fid in &to_mark {
        FunctionBody::from_id_mut(ctx, fid).set_is_pure(true);
    }
    !to_mark.is_empty()
}

/// Whether `fid`'s body computes its returned values as a deterministic function
/// of its params, with no value flowing in from outside the SSA graph.
///
/// Concretely it forbids: **loads** (a memory read is an untracked value source —
/// the return projection follows SSA only and cannot prove a loaded value is
/// independent of a symbolic input), **indirect transfers / p-code ops**, and any
/// **raw varnode operand** on a value-producing instruction (an un-promoted
/// register/global read). A **direct call to a function already proven `is_pure`**
/// is permitted: its result is itself a deterministic function of its (SSA)
/// arguments, and a pure callee clobbers nothing — so the call introduces no
/// untracked value or side effect. (The call's own `clobbers` must be empty,
/// guarding against a stale over-approximated clobber set.) **Only shadow-space
/// stores are permitted**: a store to argpromote's private temporary space is no
/// caller-visible effect, but a store to real memory is an observable side effect
/// — allowing it would let the dead-pure-call sweep delete the call (when its
/// return is unused) and drop that store.
#[cfg(test)]
pub(crate) fn body_is_pure(ctx: &Context, fid: FunctionId) -> bool {
    let callee_pure = |target: FunctionId| FunctionBody::from_id(ctx, target).is_pure();
    FunctionBody::from_id(ctx, fid).iter().all(|block| {
        block
            .iter()
            .all(|insn| mnemonic_is_pure(insn.mnemonic(), &callee_pure))
    })
}

/// Body-local purity only — callee purity is the effect engine's business
/// ([`PureChannel`]): every call-shaped mnemonic is judged solely on its
/// site-local conditions here, with the callee treated as pure.
fn body_locally_pure(ctx: &Context, fid: FunctionId) -> bool {
    FunctionBody::from_id(ctx, fid).iter().all(|block| {
        block
            .iter()
            .all(|insn| mnemonic_is_pure(insn.mnemonic(), &|_| true))
    })
}

fn mnemonic_is_pure(m: &Mnemonic, callee_pure: &dyn Fn(FunctionId) -> bool) -> bool {
    let is_temp = |space| match space {
        LocalMemorySpaceId::Shared(_) => false,
        LocalMemorySpaceId::Temp(_) => true,
    };
    match m {
        // A load from a temporary (shadow) space is private to the function —
        // argpromote seeds it from inputs — so it is a deterministic value of the
        // params, not an untracked source. A dynamic-index region loop leaves such
        // loads permanently (they cannot be forwarded away like constant-offset
        // ones), so exempting them is what lets a region-promoted function be pure.
        Mnemonic::Load(l) => is_temp(l.space),
        Mnemonic::CallInd(_) | Mnemonic::BranchInd(_) | Mnemonic::PCodeOp(_) => false,
        // A map is pure exactly when its per-element body is pure. The body is a
        // symbol, not an operand, so the generic varnode check below cannot see it.
        Mnemonic::Map(m) => m.body.real().is_some_and(callee_pure),
        // A scan is pure exactly when its per-element body is pure (same as map).
        Mnemonic::Scan(m) => m.body.real().is_some_and(callee_pure),
        Mnemonic::Apply(m) => m.target.real().is_some_and(callee_pure),
        // A direct call to a pure function is a deterministic value of its args
        // and clobbers nothing — provided the call site carries no residual
        // clobbers of its own.
        Mnemonic::Call(c) => c.clobbers.is_empty() && c.target.real().is_some_and(callee_pure),
        // A store is pure only when it writes the function's *private* shadow space
        // (argpromote's functionalized memory) — that produces no caller-visible
        // effect. A store to REAL memory is an observable side effect and keeps the
        // function impure: marking it pure would let the dead-pure-call sweep
        // (`dce::remove_dead_pure_call`) delete the call when its return is unused,
        // dropping that store. Symmetric with the `Load` arm above.
        Mnemonic::Store(s) => is_temp(s.space),
        // Every other op is a value computation or structured control flow; it is
        // pure as long as it reads no raw varnode (un-promoted register/global).
        _ => m
            .args()
            .iter()
            .all(|a| !matches!(a, qcode::value::LocalValueId::Varnode(_))),
    }
}

#[derive(Default)]
pub struct MarkPure;

impl Pass for MarkPure {
    const NAME: &'static str = "mark_pure";
    fn description(&self) -> &'static str {
        "Assert is_pure on fully functionalized (side-effect-free) functions"
    }
    fn run(
        &self,
        cone: &mut crate::ConeMut,
        _env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        // Whole-program purity solve reads `&Context`; the `is_pure` stamp writes
        // through the cone-checked interface setter.
        let to_mark = newly_pure_functions(cone.ctx(), cone.cone_functions());
        for &fid in &to_mark {
            cone.function_mut(fid).set_is_pure(true);
        }
        Ok(crate::ModulePassOutcome::functions(to_mark)
            .preserving_global::<crate::CallGraphAnalysis>()
            .preserving_global::<crate::AddressAnalysis>()
            .preserving_local::<crate::AliasAnalysis>())
    }
}

crate::register_module_pass!(MarkPure);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        testing::TestContext,
        value::{BasicBlock, FunctionBody, ValueId},
    };

    /// Build a one-block function with a single store through a pointer param,
    /// into either real ram or a private shadow (temporary) space.
    fn one_store_fn(
        tc: &mut TestContext,
        name: &'static str,
        addr: u64,
        to_shadow: bool,
    ) -> FunctionId {
        let fid = FunctionBody::make(&mut tc.ctx, name.into()).unwrap().id;
        let root = { tc.ctx.get_or_make_block(addr, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let p_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(4).id;
        let p = ValueId::BlockParam(p_pid);
        let space = if to_shadow {
            let space = tc.ctx.bodies[fid].push_temp_space(qcode::value::TempSpace::new(
                Some("shadow"),
                1,
                8,
            ));
            LocalMemorySpaceId::Temp(space.local)
        } else {
            LocalMemorySpaceId::Shared(tc.ctx.shared.default_space)
        };
        let mut b = tc.ctx.builder(root);
        let v = b.shr().get_const(0x1234, 4);
        b.push_store(v, p, space);
        fid
    }

    /// A store to real memory is an observable side effect, so the function stays
    /// impure (its call must not be dead-pure-call-eliminated). A store to the
    /// private shadow space is functionalized memory and keeps the function pure.
    #[test]
    fn real_ram_store_keeps_function_impure() {
        let mut tc = TestContext::new();
        let real = one_store_fn(&mut tc, "real", 0x1000, false);
        let shadow = one_store_fn(&mut tc, "shadow", 0x2000, true);
        assert!(
            !body_is_pure(&tc.ctx, real),
            "a real-ram store is an observable side effect → impure"
        );
        assert!(
            body_is_pure(&tc.ctx, shadow),
            "a shadow-space store is private → pure"
        );
    }

    /// Regression: a prototyped external is `Materialized` (reg-materialized)
    /// and bodyless, so the pre-engine per-body scan was vacuously pure — and a
    /// pure external with an unused result would be deleted by
    /// `remove_dead_pure_call`, dropping its real side effects (`printf`).
    /// Externals must never be flagged pure.
    #[test]
    fn materialized_external_is_never_pure() {
        let mut tc = TestContext::new();
        let ext = FunctionBody::make_external(&mut tc.ctx, 0x9000, Some("printf".into())).id;
        FunctionBody::from_id_mut(&mut tc.ctx, ext).set_register_effects(
            qcode::value::RegisterChannelState::Materialized(qcode::value::RegisterInterfaceMap {
                inputs: vec![tc.r1],
                outputs: vec![tc.r0],
                returns: 1,
                projections: Vec::new(),
            }),
        );
        mark_pure_functions(&mut tc.ctx);
        assert!(
            !FunctionBody::from_id(&tc.ctx, ext).is_pure(),
            "a materialized external keeps observable effects and must stay impure"
        );
    }

    /// A self-recursive function whose body is otherwise pure is *not* flagged:
    /// purity is only asserted for functions that cannot reach themselves
    /// (no termination argument, so neither emulation nor dead-call deletion is
    /// justified). Matches the pre-engine behaviour.
    #[test]
    fn recursion_never_flags_pure() {
        use wazabin_qcode_macro::qcode;
        let mut tc = TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry>
                    call fn f();
                <f_cont>
                    return at i64 0;
            "
        );
        let _ = (f_entry, f_cont);
        FunctionBody::from_id_mut(&mut tc.ctx, f).set_register_effects(
            qcode::value::RegisterChannelState::Materialized(
                qcode::value::RegisterInterfaceMap::default(),
            ),
        );
        mark_pure_functions(&mut tc.ctx);
        assert!(
            !FunctionBody::from_id(&tc.ctx, f).is_pure(),
            "self-recursion must never be flagged pure"
        );
    }
}
