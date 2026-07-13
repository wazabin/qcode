use qcode::{
    context::Context,
    value::{FunctionBody, FunctionId, insn::Mnemonic},
};

use crate::{Pass, PipelineEnv};

/// Assert [`FunctionBody::is_pure`] on every `pure_reg` function whose body has no
/// residual side effect — no memory access, no calls, and no raw register/global
/// (varnode) reads — so it is a deterministic pure function of its params.
///
/// This is the final argpromote step: the register/stack/memory channels each
/// functionalize one kind of effect, and a function with nothing left for any of
/// them is fully pure. Idempotent; returns `true` if any flag was newly set.
/// The [`verify`](crate::verify) pure-function rule re-checks this invariant.
pub fn mark_pure_functions(ctx: &mut Context) -> bool {
    // Loop to a fixpoint: a pure function may call pure functions
    // ([`mnemonic_is_pure`]), so a caller becomes provably pure only once its
    // callees are flagged. A single pass in an unlucky (caller-before-callee)
    // order would miss the caller; iterating until nothing new is flagged makes
    // the result independent of `function_ids()` order (acyclic call graphs settle
    // in depth-many rounds; recursion never flags, which is correct).
    let mut changed = false;
    loop {
        let mut round = false;
        for fid in ctx.function_ids() {
            let f = FunctionBody::from_id(ctx, fid);
            if f.is_pure() || !f.is_pure_reg() {
                continue;
            }
            if body_is_pure(ctx, fid) {
                FunctionBody::from_id_mut(ctx, fid).set_is_pure(true);
                round = true;
            }
        }
        if !round {
            break;
        }
        changed = true;
    }
    changed
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
pub(crate) fn body_is_pure(ctx: &Context, fid: FunctionId) -> bool {
    FunctionBody::from_id(ctx, fid).iter().all(|block| {
        block
            .iter()
            .all(|insn| mnemonic_is_pure(ctx, insn.mnemonic()))
    })
}

fn mnemonic_is_pure(ctx: &Context, m: &Mnemonic) -> bool {
    match m {
        // A load from a temporary (shadow) space is private to the function —
        // argpromote seeds it from inputs — so it is a deterministic value of the
        // params, not an untracked source. A dynamic-index region loop leaves such
        // loads permanently (they cannot be forwarded away like constant-offset
        // ones), so exempting them is what lets a region-promoted function be pure.
        Mnemonic::Load(l) => matches!(
            qcode::space::Space::from_id(ctx, l.space).ty,
            qcode::space::SpaceType::Temporary
        ),
        Mnemonic::CallInd(_) | Mnemonic::BranchInd(_) | Mnemonic::PCodeOp(_) => false,
        // A map is pure exactly when its per-element body is pure. The body is a
        // symbol, not an operand, so the generic varnode check below cannot see it.
        Mnemonic::Map(m) => m
            .body
            .real()
            .is_some_and(|body| FunctionBody::from_id(ctx, body).is_pure()),
        // A scan is pure exactly when its per-element body is pure (same as map).
        Mnemonic::Scan(m) => m
            .body
            .real()
            .is_some_and(|body| FunctionBody::from_id(ctx, body).is_pure()),
        Mnemonic::Apply(m) => m
            .target
            .real()
            .is_some_and(|target| FunctionBody::from_id(ctx, target).is_pure()),
        // A direct call to a pure function is a deterministic value of its args
        // and clobbers nothing — provided the call site carries no residual
        // clobbers of its own.
        Mnemonic::Call(c) => {
            c.clobbers.is_empty()
                && c.target
                    .real()
                    .is_some_and(|target| FunctionBody::from_id(ctx, target).is_pure())
        }
        // A store is pure only when it writes the function's *private* shadow space
        // (argpromote's functionalized memory) — that produces no caller-visible
        // effect. A store to REAL memory is an observable side effect and keeps the
        // function impure: marking it pure would let the dead-pure-call sweep
        // (`dce::remove_dead_pure_call`) delete the call when its return is unused,
        // dropping that store. Symmetric with the `Load` arm above.
        Mnemonic::Store(s) => matches!(
            qcode::space::Space::from_id(ctx, s.space).ty,
            qcode::space::SpaceType::Temporary
        ),
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
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        Ok(mark_pure_functions(ctx))
    }
}

crate::register_module_pass!(MarkPure);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        builder::Builder,
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
            tc.ctx.make_temp_space()
        } else {
            tc.ctx.shared.default_space
        };
        let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
        let v = b.context_mut().get_const(0x1234, 4).id();
        b.push_store(v, p, space);
        unsafe { b.dont_finalize() };
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
}
