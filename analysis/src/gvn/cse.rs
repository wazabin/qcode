//! Value-numbering sub-pass.
//!
//! Numbers pure instructions by a canonical [`NormalForm`](super::affine) and
//! forwards repeats to the first occurrence. Arithmetic (`c + Σ kᵢ·termᵢ`) and
//! bitwise-mask expressions are canonicalized so that reassociated shapes — e.g.
//! `(@ESP - 0xc) + 4` and `@ESP - 8` — value-number identically; everything else
//! falls back to its commutativity-normalized mnemonic. Pure values computed in a
//! dominator are always available to its descendants, so this state flows freely
//! down the dominator tree.
//!
//! On a miss for a covered (arithmetic/mask) form, the instruction is rebuilt
//! into its canonical minimal shape in place (reusing dominating sub-results),
//! unless it is already canonical. See [`super::affine`] for the normal form,
//! key/emit split, and idempotence argument.

use qcode::value::{
    ValueId,
    insn::{Binop, FloatBinop, IntBinop, Mnemonic},
};

use std::any::Any;

use super::affine::{NormalForm, Numbering, arith_form, key_for, materialize_c};
use super::walk::{Claim, Editor, InsnCtx, SubPassC};

#[cfg(test)]
use qcode::context::Context;

use crate::{ContextView, FunctionBody};

/// CSE numbers pure values down a whole dominator tree. Fully host-routed (the
/// value-numbering reads through a [`HostRef`](qcode::value::util::base_ref::HostRef)
/// and rebuilds canonical forms through the [`HostMut`] verbs), so it runs on the checked-out function-pass path.
pub(super) struct Cse;

/// The function-pass [`SubPassC`] impl (context-split stage 5b-ii): reads
/// route through `body.read_host(cx)`, shallow forwards through `Editor`'s `_c`
/// methods, and the in-place canonical rebuild runs through `materialize_c` over
/// `&mut PassBacking`.
impl<'str> SubPassC<'str> for Cse {
    fn init_state(&self) -> Box<dyn Any> {
        Box::new(Numbering::default())
    }

    fn clone_state(&self, state: &dyn Any) -> Box<dyn Any> {
        Box::new(
            state
                .downcast_ref::<Numbering>()
                .expect("cse state")
                .clone(),
        )
    }

    fn on_block_entry(
        &self,
        _body: &mut FunctionBody<'_, 'str>,
        _cx: ContextView<'_, 'str>,
        state: &mut dyn Any,
        _block_id: qcode::value::block::BlockId,
        _tree: &jstd::graph::analysis::DominatorTree<qcode::value::block::BlockId>,
        _aliases: Option<&crate::AliasResult>,
        _numbering: &Numbering,
        is_shared: bool,
    ) {
        let state = state.downcast_mut::<Numbering>().expect("cse state");
        if is_shared {
            state.clear_leaders();
        }
    }

    fn on_insn(
        &self,
        body: &mut FunctionBody<'_, 'str>,
        cx: ContextView<'_, 'str>,
        state: &mut dyn Any,
        ic: &InsnCtx,
        ed: &mut Editor,
    ) -> Claim {
        if ic.mnemonic.is_terminator() || ic.size == 0 {
            return Claim::Pass;
        }
        let state = state.downcast_mut::<Numbering>().expect("cse state");

        let form = arith_form(body.read_host(cx), ic.id, ic.mnemonic, ic.size, state);
        state.record_form(ic.id, form.clone());
        let key = key_for(&form, ic.id, ic.mnemonic);

        state.seed_operand_leaders(ic.mnemonic.args());

        if let Some(leader) = state.lookup(&key) {
            if leader != ic.id {
                ed.replace_c(body, cx, ic.insn_id, leader);
            }
            return Claim::Done;
        }

        match key {
            NormalForm::Opaque(_) => state.claim(key, ic.id),
            _ => {
                let root_ty = body.read_host(cx).type_of(ic.id);
                let v = materialize_c(
                    body,
                    cx,
                    ic.block_id,
                    ic.insn_id,
                    ic.mnemonic,
                    &key,
                    root_ty,
                    state,
                );
                if v != ic.id {
                    ed.replace_c(body, cx, ic.insn_id, v);
                }
            }
        }
        Claim::Done
    }
}

// ---------------------------------------------------------------------------
// Normalization (used for commutative binops)
// ---------------------------------------------------------------------------

/// Total order over `ValueId`s for commutative-operand canonicalization. The
/// variant rank breaks ties between equal indices of different variants (e.g.
/// `Literal(5)` vs `Instruction(5)`), which would otherwise leave `a + b` and
/// `b + a` un-normalized.
pub(super) fn value_id_key(v: ValueId) -> (u8, usize) {
    match v {
        ValueId::Literal(x) => (0, x.into()),
        ValueId::Instruction(x) => (1, usize::from(x.local)),
        ValueId::Varnode(x) => (2, x.into()),
        ValueId::BasicBlock(x) => (3, usize::from(x.local)),
        ValueId::Function(x) => (4, x.into()),
        ValueId::BlockParam(x) => (5, usize::from(x.local)),
        // A `Bytes` blob (a constant too wide for a `Literal`) can appear as a
        // commutative operand; give it a stable rank so canonicalization is total.
        ValueId::Bytes(x) => (6, x.into()),
        // `ValueId` is `#[non_exhaustive]`; keep any future variant last but stable
        // rather than panicking mid-analysis.
        _ => (u8::MAX, 0),
    }
}

pub(super) fn is_commutative(op: &Binop) -> bool {
    matches!(
        op,
        Binop::Int(
            IntBinop::Add
                | IntBinop::Mul
                | IntBinop::And
                | IntBinop::Or
                | IntBinop::Xor
                | IntBinop::Equal
                | IntBinop::NotEqual
        ) | Binop::Float(FloatBinop::Equal | FloatBinop::NotEqual)
    )
}

/// We want to canonicalize commutative binops so that e.g. a + b and b + a are
/// represented by the same value number.
pub(super) fn normalize(m: &mut Mnemonic) {
    if let Mnemonic::Binop(b) = m
        && is_commutative(&b.op)
    {
        let l = value_id_key(b.lhs);
        let r = value_id_key(b.rhs);
        if l > r {
            std::mem::swap(&mut b.lhs, &mut b.rhs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gvn::{gvn, gvn_function};
    use qcode::value::{BasicBlock, InstructionId, insn::Binary};
    use qcode_macro::qcode;

    #[test]
    fn test_normalize() {
        let mut m = Mnemonic::Binop(Binary {
            op: Binop::Int(IntBinop::Add),
            lhs: ValueId::Instruction(InstructionId::default()),
            rhs: ValueId::Literal(0.into()),
        });

        normalize(&mut m);
        assert_eq!(
            m,
            Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::Add),
                lhs: ValueId::Literal(0.into()),
                rhs: ValueId::Instruction(InstructionId::default()),
            }),
            "normalize should swap operands to canonicalize commutative binop"
        );
    }

    // 1. Same binop -> second is redundant, leader is first
    #[test]
    fn test_same_binop_redundant() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                varnode i64 A;
                varnode i64 B;

                <block>
                    %a = load(A:8, &A);
                    %b = load(B:8, &B);

                    %v1 = %a + %b;
                    %v2 = %a + %b;

                    goto <0x1001>;
            "
        );

        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        gvn(&mut block, None);

        assert!(block.instruction_ids().contains(&v1));
        assert!(!block.instruction_ids().contains(&v2));
    }

    // 2. Commutative normalization: a + b and b + a -> same value number
    #[test]
    fn test_commutative_normalization() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                varnode i64 A;
                varnode i64 B;

                <block>
                    %a = load(A:8, &A);
                    %b = load(B:8, &B);
                    %v1 = %a + %b;
                    %v2 = %b + %a;
                    goto <0x1001>;
            "
        );

        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        gvn(&mut block, None);

        assert!(block.instruction_ids().contains(&v1));
        assert!(!block.instruction_ids().contains(&v2));
    }

    // 3. Non-commutative not swapped: a - b != b - a
    #[test]
    fn test_non_commutative_not_swapped() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                varnode i64 A;
                varnode i64 B;

                <block>
                    %a = load(A:8, &A);
                    %b = load(B:8, &B);
                    %v1 = %a - %b;
                    %v2 = %b - %a;
                    goto <0x1001>;
            "
        );

        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        gvn(&mut block, None);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));
    }

    // 4. Different ops -> distinct
    #[test]
    fn test_different_ops_distinct() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                varnode i64 A;
                varnode i64 B;

                <block>
                    %a = load(A:8, &A);
                    %b = load(B:8, &B);
                    %v1 = %a + %b;
                    %v2 = %a * %b;
                    goto <0x1001>;
            "
        );

        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        gvn(&mut block, None);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));
    }

    // 5. Cross-block redundancy: a+b in entry propagates to dominated successor
    #[test]
    fn test_gvn_function_cross_block_redundancy() {
        // Layout: entry → succ (linear chain, succ dominated by entry)
        // entry: v1 = a + b
        // succ:  v2 = a + b  <- redundant, dominated by entry
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                varnode i64 A;
                varnode i64 B;

                fn f:
                    <entry>
                        %a = load(A:8, &A);
                        %b = load(B:8, &B);

                        %v1 = %a + %b;
                        goto <succ>;

                    <succ>
                        %v2 = %a + %b;
                        return at 0x1000;
                "
        );

        gvn_function(&mut ctx, f, None);

        assert!(
            BasicBlock::from_id(&ctx, entry)
                .instruction_ids()
                .contains(&v1)
        );
        assert!(
            !BasicBlock::from_id(&ctx, succ)
                .instruction_ids()
                .contains(&v2),
            "a+b in dominated successor should be eliminated"
        );
    }

    // 6. Diamond CFG: a+b computed in one branch is NOT available at merge
    #[test]
    fn test_gvn_function_no_propagation_across_merge() {
        // Layout: entry → left, entry → right, left → merge, right → merge
        // left:  v1 = a + b
        // right: (no a+b)
        // merge: v2 = a + b  <- NOT redundant; merge is dominated only by entry
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                varnode i64 A;
                varnode i64 B;

                fn g:
                    <entry>
                        %a = load(A:8, &A);
                        %b = load(B:8, &B);
                        if i8 1 goto <left> else goto <right>;

                    <left>
                        %v1 = %a + %b;
                        goto <merge>;

                    <right>
                        goto <merge>;

                    <merge>
                        %v2 = %a + %b;"
        );

        gvn_function(&mut ctx, g, None);

        assert!(
            BasicBlock::from_id(&ctx, left)
                .instruction_ids()
                .contains(&v1)
        );
        assert!(
            BasicBlock::from_id(&ctx, merge)
                .instruction_ids()
                .contains(&v2),
            "a+b at merge must NOT be eliminated: merge is not dominated by left"
        );
    }

    /// Commutative canonicalization must order operands even when their indices
    /// tie across different `ValueId` variants (e.g. `Literal(0)` vs
    /// `Instruction(0)`), otherwise `a + b` and `b + a` value-number differently.
    #[test]
    fn normalize_orders_equal_indices_across_value_id_variants() {
        let a = ValueId::Literal(0usize.into());
        let b = ValueId::Instruction(InstructionId::default());
        let make = |lhs, rhs| {
            Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::Add),
                lhs,
                rhs,
            })
        };
        let mut m1 = make(a, b);
        let mut m2 = make(b, a);
        normalize(&mut m1);
        normalize(&mut m2);
        assert_eq!(
            m1, m2,
            "a + b and b + a must normalize to the same mnemonic"
        );
    }

    // ----- normal-form value numbering ------------------------------------

    /// Helper: numeric value of a literal operand.
    fn lit_value(ctx: &Context, v: ValueId) -> Option<u64> {
        match v {
            ValueId::Literal(id) => Some(ctx.shared.values.literals[id].value),
            _ => None,
        }
    }

    /// `(@SP - 0xc) + 4` reassociates to `@SP - 8`, which a dominating
    /// instruction already computes, so the chain forwards and is deleted.
    #[test]
    fn test_reassociation_forwards_to_dominating_value() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 SP;
                varnode i32 OUT;

                fn f:
                    <entry>
                        %b = &SP - i32 0x8;
                        %a = &SP - i32 0xc;
                        %c = %a + i32 0x4;
                        store(OUT:4, &OUT <- %c);
                        return at 0x1000;
                "
        );

        assert!(gvn_function(&mut ctx, f, None));

        let insns = BasicBlock::from_id(&ctx, entry).instruction_ids().to_vec();
        assert!(
            !insns.contains(&c),
            "(@SP-0xc)+4 must forward to the existing @SP-8 and be removed"
        );
        assert!(insns.contains(&b), "@SP-8 is the surviving leader");
    }

    /// A second GVN run over an already-canonical function must report no change.
    #[test]
    fn test_reassociation_is_idempotent() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 SP;
                varnode i32 OUT;

                fn f:
                    <entry>
                        %b = &SP - i32 0x8;
                        %a = &SP - i32 0xc;
                        %c = %a + i32 0x4;
                        store(OUT:4, &OUT <- %c);
                        return at 0x1000;
                "
        );

        assert!(
            gvn_function(&mut ctx, f, None),
            "first run rewrites the chain"
        );
        assert!(
            !gvn_function(&mut ctx, f, None),
            "second run must reach a fixpoint and report no change"
        );
    }

    /// Same-op constant-mask chains coalesce: `(x & 0xff0) & 0x0ff` forwards to
    /// the dominating `x & 0xf0`.
    #[test]
    fn test_mask_chain_coalesces_and_forwards() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 X;
                varnode i32 OUT;

                fn f:
                    <entry>
                        %x = load(X:4, &X);
                        %p = %x & i32 0xf0;
                        %m = %x & i32 0xff0;
                        %n = %m & i32 0xff;
                        store(OUT:4, &OUT <- %n);
                        return at 0x1000;
                "
        );

        assert!(gvn_function(&mut ctx, f, None));

        let insns = BasicBlock::from_id(&ctx, entry).instruction_ids().to_vec();
        assert!(
            !insns.contains(&n),
            "(x & 0xff0) & 0xff == x & 0xf0 must forward to %p"
        );
        assert!(insns.contains(&p), "x & 0xf0 is the surviving leader");
    }

    /// When rebuilding, a negative offset is emitted as `sub 8`, never
    /// `add 0xfffffff8` — preserving the sign that stack lowering relies on.
    #[test]
    fn test_rebuild_emits_signed_subtraction() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 SP;
                varnode i32 OUT;

                fn f:
                    <entry>
                        %a = &SP + i32 0x4;
                        %b = %a - i32 0xc;
                        store(OUT:4, &OUT <- %b);
                        return at 0x1000;
                "
        );

        assert!(gvn_function(&mut ctx, f, None));

        // No instruction may carry the wrapped positive constant 0xfffffff8.
        for &id in BasicBlock::from_id(&ctx, entry).instruction_ids() {
            for arg in ctx.get_insn(id).mnemonic().args() {
                assert_ne!(
                    lit_value(&ctx, arg),
                    Some(0xffff_fff8),
                    "rebuild must use signed `sub 8`, not `add 0xfffffff8`"
                );
            }
        }

        // The rebuilt value must be `sub(_, 8)`.
        let mut found_sub_by_8 = false;
        for &id in BasicBlock::from_id(&ctx, entry).instruction_ids() {
            if let Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::Sub),
                rhs,
                ..
            }) = ctx.get_insn(id).mnemonic()
                && lit_value(&ctx, *rhs) == Some(8)
            {
                found_sub_by_8 = true;
            }
        }
        assert!(found_sub_by_8, "@SP+4-0xc must rebuild to @SP - 8");
    }

    /// Widening is a width boundary: `zext(a) + zext(b)` must NOT value-number
    /// with `zext(a + b)` (distributing across zext is unsound on carry).
    #[test]
    fn test_zext_is_an_opaque_leaf() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 B;
                varnode i64 OUT;

                fn f:
                    <entry>
                        %a = load(A:4, &A);
                        %b = load(B:4, &B);
                        %za = zext(i64, %a);
                        %zb = zext(i64, %b);
                        %wide = %za + %zb;
                        %sum = %a + %b;
                        %zs = zext(i64, %sum);
                        store(OUT:8, &OUT <- %wide);
                        store(OUT:8, &OUT <- %zs);
                        return at 0x1000;
                "
        );

        gvn_function(&mut ctx, f, None);

        let insns = BasicBlock::from_id(&ctx, entry).instruction_ids().to_vec();
        assert!(
            insns.contains(&wide) && insns.contains(&zs),
            "zext(a)+zext(b) and zext(a+b) are different values and must both survive"
        );
    }

    /// An arithmetic value computed in one orphan-region entry must not be
    /// forwarded into a block both entries reach: that block's inherited
    /// dominance claims are invalid (mirrors the load-forwarding guard).
    #[test]
    fn test_arithmetic_not_forwarded_into_shared_orphan_block() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 OUT;

                fn shared_orphans:
                    <entry>
                        return at 0x1000;

                    <e1>
                        %x1 = &A - i32 0x8;
                        store(OUT:4, &OUT <- %x1);
                        goto <shared>;

                    <e2>
                        goto <shared>;

                    <shared>
                        %x2 = &A - i32 0x8;
                        store(OUT:4, &OUT <- %x2);
                        return at 0x1001;
                "
        );

        gvn_function(&mut ctx, shared_orphans, None);

        assert!(
            BasicBlock::from_id(&ctx, shared)
                .instruction_ids()
                .contains(&x2),
            "@A-8 in a block reachable from two orphan entries must not be forwarded"
        );
    }
}
