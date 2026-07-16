//! Verify the `is_pure` invariant.
//!
//! A function flagged [`FunctionRef::is_pure`](qcode::value::FunctionRef::is_pure)
//! is asserted to be a deterministic
//! pure function of its params — argpromote has functionalized every side-effect
//! channel. Pure-function emulation in constant propagation relies on this: it
//! emulates such a callee to harvest constant return values. This rule re-derives
//! the property independently of the code that asserts it, so a pass that marks a
//! function pure while leaving a residual effect is caught at its source.

use qcode::{
    context::Context,
    space::LocalMemorySpaceId,
    value::{
        FunctionBody, FunctionId, Instruction, LocalValueId,
        insn::{InstructionId, Mnemonic},
    },
};

/// A function marked `is_pure` whose body still contains a side effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PureFunctionViolation {
    pub function: FunctionId,
    pub insn: InstructionId,
    pub reason: &'static str,
}

impl PureFunctionViolation {
    pub fn diagnostic(&self, ctx: &Context<'_>) -> String {
        let name = FunctionBody::from_id(ctx, self.function).name().to_owned();
        let insn = Instruction::from_id(ctx, self.insn);
        let addr = insn
            .address()
            .map(|a| format!(" at {a:#x}"))
            .unwrap_or_default();
        format!(
            "function `{name}` is marked is_pure but has a {} {insn}{addr}",
            self.reason
        )
    }
}

/// Whether `space` is a builder/argpromote scratch space (a temporary). A
/// load/store there is private to the function — seeded from inputs or written
/// earlier in the body — so it is not a caller-visible effect.
fn is_temp_space(_ctx: &Context, space: LocalMemorySpaceId) -> bool {
    match space {
        LocalMemorySpaceId::Shared(_) => false,
        LocalMemorySpaceId::Temp(_) => true,
    }
}

/// The residual side effect a mnemonic carries, or `None` if it is pure.
fn impurity(ctx: &Context, m: &Mnemonic) -> Option<&'static str> {
    match m {
        // A *real-memory* load is an untracked value source; the rest are
        // non-functionalized effects/transfers. A load from a temporary (shadow)
        // space is private — seeded from inputs by argpromote — so it is pure
        // (a dynamic-index region loop leaves such loads permanently). Stores are
        // permitted (they produce no value and the call site is retained),
        // matching `argpromote::body_is_pure`.
        Mnemonic::Load(l) => (!is_temp_space(ctx, l.space)).then_some("memory load"),
        Mnemonic::Call(_) | Mnemonic::CallInd(_) => Some("call"),
        Mnemonic::BranchInd(_) => Some("indirect branch"),
        Mnemonic::PCodeOp(_) => Some("architecture p-code op"),
        Mnemonic::Store(_) => None,
        // A map is a deterministic value of its array argument iff its per-element
        // body is itself pure (the body symbol is not an operand, so the generic
        // varnode check below would miss an impure body).
        Mnemonic::Map(m) => (!m
            .body
            .real()
            .is_some_and(|body| FunctionBody::from_id(ctx, body).is_pure()))
        .then_some("map with impure or minted body"),
        // A scan, like a map, is a deterministic value of its array argument and
        // initial accumulator iff its per-element body is pure (the body symbol is
        // not an operand, so the generic varnode check below would miss it).
        Mnemonic::Scan(m) => (!m
            .body
            .real()
            .is_some_and(|body| FunctionBody::from_id(ctx, body).is_pure()))
        .then_some("scan with impure or minted body"),
        Mnemonic::Apply(m) => (!m
            .target
            .real()
            .is_some_and(|target| FunctionBody::from_id(ctx, target).is_pure()))
        .then_some("apply with impure or minted target"),
        // A pure function reads its inputs only through params: a raw varnode
        // operand is an un-functionalized register/global read.
        _ => m
            .args()
            .iter()
            .any(|a| matches!(a, LocalValueId::Varnode(_)))
            .then_some("raw varnode read"),
    }
}

/// Every `is_pure` function whose body still has a side effect, one violation per
/// offending instruction.
pub fn verify_pure_functions(ctx: &Context<'_>) -> Vec<PureFunctionViolation> {
    let mut violations = Vec::new();
    for fid in ctx.function_ids() {
        if !FunctionBody::from_id(ctx, fid).is_pure() {
            continue;
        }
        for block in FunctionBody::from_id(ctx, fid).iter() {
            for insn in block.iter() {
                if let Some(reason) = impurity(ctx, insn.mnemonic()) {
                    violations.push(PureFunctionViolation {
                        function: fid,
                        insn: insn.id,
                        reason,
                    });
                }
            }
        }
    }
    violations
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{Value, ValueId},
    };

    /// Build a single-block function, mark it `is_pure`, and run `body` to fill it.
    fn pure_flagged_fn(tc: &mut TestContext, body: impl FnOnce(&mut Builder)) -> FunctionId {
        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let entry = { tc.ctx.get_or_make_block(0x1000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        {
            let mut b = (&mut tc.ctx).builder(entry);
            body(&mut b);
        }
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);
        fid
    }

    #[test]
    fn accepts_side_effect_free_pure_function() {
        let mut tc = TestContext::new();
        pure_flagged_fn(&mut tc, |b| {
            let a = b.push_param(8).id();
            let c = b.shr().get_const(1, 8);
            let s = b.push_add(a, c).id();
            let tuple = b.push_tuple(vec![s]).id();
            let _ = tuple;
            let ptr = b.shr().get_const(0x2000, 8);
            b.push_return(ptr);
        });
        assert!(verify_pure_functions(&tc.ctx).is_empty());
    }

    #[test]
    fn flags_residual_load_in_pure_function() {
        let mut tc = TestContext::new();
        let reg = ValueId::Varnode(tc.r0);
        let space = tc.reg_space;
        pure_flagged_fn(&mut tc, |b| {
            let _ = b.push_load::<false>(reg, 8, space).id();
            let ptr = b.shr().get_const(0x2000, 8);
            b.push_return(ptr);
        });
        let violations = verify_pure_functions(&tc.ctx);
        assert_eq!(violations.len(), 1, "the load must be flagged");
        assert_eq!(violations[0].reason, "memory load");
    }

    /// A store is *not* a purity violation: it produces no value, so it cannot
    /// feed a returned field, and the emulation that relies on `is_pure` keeps
    /// the call in place.
    #[test]
    fn allows_store_in_pure_function() {
        let mut tc = TestContext::new();
        let reg = ValueId::Varnode(tc.r0);
        let space = tc.reg_space;
        pure_flagged_fn(&mut tc, |b| {
            let c = b.shr().get_const(7, 8);
            b.push_store(c, reg, space);
            let ptr = b.shr().get_const(0x2000, 8);
            b.push_return(ptr);
        });
        assert!(verify_pure_functions(&tc.ctx).is_empty());
    }
}
