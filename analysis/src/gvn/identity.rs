//! Bitwise/arithmetic identity sub-pass: collapse multi-instruction DAGs that
//! compilers emit for `^`/`+` into the single operation they compute.
//!
//! Unlike [`super::fold::algebraic_identity`], which applies a law to a single
//! `Binop`'s own operands, these patterns span several instructions. They rely
//! on GVN having value-numbered the shared sub-expressions, so the same `(a, b)`
//! feeding two different operations are recognized as identical operands.
//!
//! All identities below are exact in `n`-bit modular arithmetic (every operation
//! is the same width), from the full-adder relation
//! `a + b == (a ^ b) + ((a & b) << 1)` and the inclusion/exclusion relation
//! `a + b == (a | b) + (a & b)`:
//!
//! ```text
//!   (a + b) - ((a & b) << 1)  →  a ^ b
//!   (a | b) - (a & b)         →  a ^ b
//!   (a ^ b) + ((a & b) << 1)  →  a + b
//!   (a | b) + (a & b)         →  a + b
//! ```

use qcode::{
    context::Context,
    value::{
        ValueId,
        insn::{Binary, Binop, IntBinop, Mnemonic},
    },
};

use super::fold::const_value;
use super::walk::{Claim, Editor, InsnCtx, SubPass};

/// Recognize the add/and/shift (and or/and) idioms and rewrite the root
/// instruction to the single `^`/`+` it computes. The now-unused sub-expressions
/// are left for DCE.
pub(super) struct Identities;

impl SubPass for Identities {
    type State = ();

    fn on_insn(&self, ctx: &mut Context, _state: &mut (), ic: &InsnCtx, ed: &mut Editor) -> Claim {
        if ic.mnemonic.is_terminator() || ic.size == 0 {
            return Claim::Pass;
        }
        // Pure intrinsics carry their own algebraic simplifier (e.g.
        // `rol(x, 0) → x`), which forwards uses to an existing value.
        if let Mnemonic::Intrinsic(intr) = ic.mnemonic
            && let Some(simplify) = intr.id.desc().simplify
        {
            let args = intr.args.clone();
            if let Some(repl) = simplify(ctx, &args) {
                ed.replace(ctx, ic.insn_id, repl);
                return Claim::Done;
            }
        }
        match simplify_identity(ctx, ic.mnemonic) {
            Some(new_mnemonic) => {
                ed.replace_with_new_insn(ctx, ic.block_id, ic.insn_id, new_mnemonic, ic.size);
                Claim::Done
            }
            None => Claim::Pass,
        }
    }
}

/// If `v` is defined by an `IntBinop::want` binop, return its `(lhs, rhs)`.
fn as_int_binop(ctx: &Context, v: ValueId, want: IntBinop) -> Option<(ValueId, ValueId)> {
    let ValueId::Instruction(id) = v else {
        return None;
    };
    match ctx.get_insn(id).mnemonic() {
        Mnemonic::Binop(Binary {
            lhs,
            rhs,
            op: Binop::Int(op),
        }) if *op == want => Some((*lhs, *rhs)),
        _ => None,
    }
}

/// Whether `(a, b)` and `(c, d)` are the same unordered operand pair. The feeding
/// operations (`+`/`^`/`|` and `&`) are all commutative, and GVN may have
/// normalized their operand order independently, so match either assignment.
fn same_operands(p: (ValueId, ValueId), q: (ValueId, ValueId)) -> bool {
    (p.0 == q.0 && p.1 == q.1) || (p.0 == q.1 && p.1 == q.0)
}

/// If `v` computes `x * 2` — as `x << 1`, `x * 2`, or `2 * x` — return `x`.
fn as_doubled(ctx: &Context, v: ValueId) -> Option<ValueId> {
    if let Some((x, amt)) = as_int_binop(ctx, v, IntBinop::ShiftLeft)
        && const_value(ctx, amt) == Some(1)
    {
        return Some(x);
    }
    if let Some((x, y)) = as_int_binop(ctx, v, IntBinop::Mul) {
        if const_value(ctx, y) == Some(2) {
            return Some(x);
        }
        if const_value(ctx, x) == Some(2) {
            return Some(y);
        }
    }
    None
}

/// If `v` is `(a & b) * 2`, return `(a, b)`.
fn as_doubled_and(ctx: &Context, v: ValueId) -> Option<(ValueId, ValueId)> {
    as_int_binop(ctx, as_doubled(ctx, v)?, IntBinop::And)
}

fn int_binop(lhs: ValueId, rhs: ValueId, op: IntBinop) -> Mnemonic {
    Mnemonic::Binop(Binary {
        lhs,
        rhs,
        op: Binop::Int(op),
    })
}

/// Collapse a bitwise/arithmetic identity rooted at `m`, returning the
/// equivalent single-operation mnemonic, or `None` if no pattern matches.
pub(super) fn simplify_identity(ctx: &Context, m: &Mnemonic) -> Option<Mnemonic> {
    let &Mnemonic::Binop(Binary {
        lhs,
        rhs,
        op: Binop::Int(op),
    }) = m
    else {
        return None;
    };

    match op {
        // (a + b) - ((a & b) << 1)  →  a ^ b
        // (a | b) - (a & b)         →  a ^ b
        // `-` is not commutative: the reducible term is always the rhs.
        IntBinop::Sub => {
            if let Some((a, b)) = as_int_binop(ctx, lhs, IntBinop::Add)
                && let Some(cd) = as_doubled_and(ctx, rhs)
                && same_operands((a, b), cd)
            {
                return Some(int_binop(a, b, IntBinop::Xor));
            }
            if let Some((a, b)) = as_int_binop(ctx, lhs, IntBinop::Or)
                && let Some(cd) = as_int_binop(ctx, rhs, IntBinop::And)
                && same_operands((a, b), cd)
            {
                return Some(int_binop(a, b, IntBinop::Xor));
            }
            None
        }
        // (a ^ b) + ((a & b) << 1)  →  a + b
        // (a | b) + (a & b)         →  a + b
        // `+` is commutative, so the base and extra term may appear on either side.
        IntBinop::Add => {
            for (base, extra) in [(lhs, rhs), (rhs, lhs)] {
                if let Some((a, b)) = as_int_binop(ctx, base, IntBinop::Xor)
                    && let Some(cd) = as_doubled_and(ctx, extra)
                    && same_operands((a, b), cd)
                {
                    return Some(int_binop(a, b, IntBinop::Add));
                }
                if let Some((a, b)) = as_int_binop(ctx, base, IntBinop::Or)
                    && let Some(cd) = as_int_binop(ctx, extra, IntBinop::And)
                    && same_operands((a, b), cd)
                {
                    return Some(int_binop(a, b, IntBinop::Add));
                }
            }
            None
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::AliasResult;
    use crate::gvn::gvn_function;
    use qcode::{context::Context, value::BasicBlock};
    use qcode_macro::qcode;

    /// `(a + b) - ((a & b) << 1)` is the add/and/shift form of `a ^ b` and must
    /// collapse to a single xor, leaving the add/and/shift math dead.
    #[test]
    fn add_minus_doubled_and_collapses_to_xor() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 B;

                fn f:
                    <entry>
                        %a = load(i32, &A);
                        %b = load(i32, &B);
                        %sum = %a + %b;
                        %and = %a & %b;
                        %dbl = %and << 0x1;
                        %root = %sum - %dbl;
                        return [%root];
                "
        );

        let aliases = AliasResult::simple(&ctx);
        gvn_function(&mut ctx, f, Some(&aliases));

        assert!(
            !BasicBlock::from_id(&ctx, entry)
                .instruction_ids()
                .contains(&root),
            "the `sub` root should be rewritten to an xor"
        );

        crate::remove_dead_insns(&mut ctx, entry);
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(text.contains('^'), "expected a single xor, got:\n{text}");
        assert!(
            !text.contains("<<") && !text.contains(" + ") && !text.contains(" - "),
            "the add/and/shift math should be dead after the rewrite, got:\n{text}"
        );
    }

    /// `(a | b) - (a & b)` is another form of `a ^ b`.
    #[test]
    fn or_minus_and_collapses_to_xor() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 B;

                fn f:
                    <entry>
                        %a = load(i32, &A);
                        %b = load(i32, &B);
                        %or = %a | %b;
                        %and = %a & %b;
                        %root = %or - %and;
                        return [%root];
                "
        );

        let aliases = AliasResult::simple(&ctx);
        gvn_function(&mut ctx, f, Some(&aliases));
        crate::remove_dead_insns(&mut ctx, entry);
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(text.contains('^'), "expected a single xor, got:\n{text}");
        assert!(
            !text.contains(" | ") && !text.contains(" - "),
            "the or/and math should be dead, got:\n{text}"
        );
    }

    /// `(a | b) + (a & b)` is a form of `a + b`.
    #[test]
    fn or_plus_and_collapses_to_add() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 B;

                fn f:
                    <entry>
                        %a = load(i32, &A);
                        %b = load(i32, &B);
                        %or = %a | %b;
                        %and = %a & %b;
                        %root = %or + %and;
                        return [%root];
                "
        );

        let aliases = AliasResult::simple(&ctx);
        gvn_function(&mut ctx, f, Some(&aliases));
        crate::remove_dead_insns(&mut ctx, entry);
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(text.contains(" + "), "expected a single add, got:\n{text}");
        assert!(
            !text.contains(" | ") && !text.contains(" & "),
            "the or/and math should be dead, got:\n{text}"
        );
    }

    /// An unrelated `sub` whose operands don't form the idiom must be left alone.
    #[test]
    fn unrelated_sub_is_untouched() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 B;

                fn f:
                    <entry>
                        %a = load(i32, &A);
                        %b = load(i32, &B);
                        %sum = %a + %b;
                        %and = %a & %b;
                        %root = %sum - %and;
                        return [%root];
                "
        );

        let aliases = AliasResult::simple(&ctx);
        gvn_function(&mut ctx, f, Some(&aliases));
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(
            text.contains(" - "),
            "a non-idiom subtraction must be preserved, got:\n{text}"
        );
    }
}
