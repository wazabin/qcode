use qcode::{
    context::Context,
    value::{Function, FunctionId, ValueId, insn::Mnemonic},
};

use crate::{Pass, PipelineEnv};

/// Assert [`Function::is_pure`] on every `pure_reg` function whose body has no
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
            let f = Function::from_id(ctx, fid);
            if f.is_pure() || !f.is_pure_reg() {
                continue;
            }
            if body_is_pure(ctx, fid) {
                Function::from_id_mut(ctx, fid).set_is_pure(true);
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
/// guarding against a stale over-approximated clobber set.) **Stores are
/// permitted**: they produce no value, so they never feed a returned field, and
/// the emulation harvesting this property keeps the call in place — any real side
/// effect the store represents is preserved.
pub(crate) fn body_is_pure(ctx: &Context, fid: FunctionId) -> bool {
    Function::from_id(ctx, fid).iter().all(|block| {
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
        Mnemonic::Map(m) => Function::from_id(ctx, m.body).is_pure(),
        // A direct call to a pure function is a deterministic value of its args
        // and clobbers nothing — provided the call site carries no residual
        // clobbers of its own.
        Mnemonic::Call(c) => c.clobbers.is_empty() && Function::from_id(ctx, c.target).is_pure(),
        // A store produces no value, so it never feeds a returned field.
        Mnemonic::Store(_) => true,
        // Every other op is a value computation or structured control flow; it is
        // pure as long as it reads no raw varnode (un-promoted register/global).
        _ => m.args().iter().all(|a| !matches!(a, ValueId::Varnode(_))),
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
