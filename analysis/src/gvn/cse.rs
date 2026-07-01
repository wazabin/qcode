//! Common-subexpression-elimination sub-pass.
//!
//! Numbers pure instructions by their (commutativity-normalized) mnemonic and
//! forwards repeats to the first occurrence. Pure values computed in a
//! dominator are always available to its descendants, so this state flows
//! freely down the dominator tree.

use rustc_hash::FxHashMap as HashMap;

use qcode::{
    context::Context,
    value::{
        ValueId,
        insn::{Binop, BoolBinop, FloatBinop, IntBinop, Mnemonic},
    },
};

use super::walk::{Claim, Editor, InsnCtx, SubPass};

pub(super) struct Cse;

impl SubPass for Cse {
    type State = HashMap<Mnemonic, ValueId>;

    fn on_insn(
        &self,
        ctx: &mut Context,
        state: &mut Self::State,
        ic: &InsnCtx,
        ed: &mut Editor,
    ) -> Claim {
        if ic.mnemonic.is_terminator() || ic.size == 0 {
            return Claim::Pass;
        }
        let mut mnemonic = ic.mnemonic.clone();
        normalize(&mut mnemonic);
        match state.get(&mnemonic) {
            Some(&leader) => ed.replace(ctx, ic.insn_id, leader),
            None => {
                state.insert(mnemonic, ic.id);
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
fn value_id_key(v: ValueId) -> (u8, usize) {
    match v {
        ValueId::Literal(x) => (0, x.into()),
        ValueId::Instruction(x) => (1, x.into()),
        ValueId::Varnode(x) => (2, x.into()),
        ValueId::BasicBlock(x) => (3, x.into()),
        ValueId::Function(x) => (4, x.into()),
        ValueId::BlockParam(x) => (5, x.into()),
        _ => todo!("unsupported value id type in value_id_key: {:?}", v),
    }
}

fn is_commutative(op: &Binop) -> bool {
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
        ) | Binop::Bool(BoolBinop::And | BoolBinop::Or | BoolBinop::Xor)
            | Binop::Float(FloatBinop::Equal | FloatBinop::NotEqual)
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
    use qcode::value::{BasicBlock, insn::Binary};
    use qcode_macro::qcode;

    #[test]
    fn test_normalize() {
        let mut m = Mnemonic::Binop(Binary {
            op: Binop::Int(IntBinop::Add),
            lhs: ValueId::Instruction(0.into()),
            rhs: ValueId::Literal(0.into()),
        });

        normalize(&mut m);
        assert_eq!(
            m,
            Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::Add),
                lhs: ValueId::Literal(0.into()),
                rhs: ValueId::Instruction(0.into()),
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
                    %a = load(i64, &A);
                    %b = load(i64, &B);

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
                    %a = load(i64, &A);
                    %b = load(i64, &B);
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
                    %a = load(i64, &A);
                    %b = load(i64, &B);
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
                    %a = load(i64, &A);
                    %b = load(i64, &B);
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
                        %a = load(i64, &A);
                        %b = load(i64, &B);

                        %v1 = %a + %b;
                        goto <succ>;

                    <succ>
                        %v2 = %a + %b;
                        return [0x1000];
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
                        %a = load(i64, &A);
                        %b = load(i64, &B);
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
        let b = ValueId::Instruction(0usize.into());
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
}
