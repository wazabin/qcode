//! Constant-folding sub-pass: arithmetic on literals and algebraic identities.

use qcode::{
    context::Context,
    value::{
        Value, ValueId, ValueRef,
        insn::{Binary, Binop, BoolBinop, IntBinop, Mnemonic, Unop},
        literal::LiteralRef,
    },
};

use super::walk::{Claim, Editor, InsnCtx, SubPass};

/// Fold constant arithmetic and algebraic identities into interned literals.
pub(super) struct Fold;

impl SubPass for Fold {
    type State = ();

    fn on_insn(&self, ctx: &mut Context, _state: &mut (), ic: &InsnCtx, ed: &mut Editor) -> Claim {
        if ic.mnemonic.is_terminator() || ic.size == 0 {
            return Claim::Pass;
        }
        match try_fold(ctx, ic.mnemonic, ic.size) {
            Some(folded) => {
                ed.replace(ctx, ic.insn_id, folded);
                Claim::Done
            }
            None => Claim::Pass,
        }
    }
}

// ---------------------------------------------------------------------------
// Constant folding
// ---------------------------------------------------------------------------

fn get_const<'a>(ctx: &'a Context<'a>, v: ValueId) -> Option<LiteralRef<'a, 'a>> {
    match ValueRef::new(v, ctx) {
        ValueRef::Literal(c) => Some(c),
        _ => None,
    }
}

fn is_symbolic_literal(ctx: &Context, v: ValueId) -> bool {
    let ValueId::Literal(id) = v else {
        return false;
    };
    ctx.values.literals[id].symbolic.is_some()
}

/// `get_const`, but refusing Block/Function/String symbolic literals: those are
/// not numeric constants and folding them would discard the symbolic annotation.
/// StackAddress-typed literals have symbolic=None and are foldable.
fn get_numeric_const<'a>(ctx: &'a Context<'a>, v: ValueId) -> Option<LiteralRef<'a, 'a>> {
    if is_symbolic_literal(ctx, v) {
        return None;
    }
    get_const(ctx, v)
}

// TODO: use the existing `evaluate_const` machinery instead of re-implementing it here.
pub(super) fn constant_folding(
    ctx: &mut Context,
    m: &Mnemonic,
    _output_size: usize,
) -> Option<ValueId> {
    match m {
        &Mnemonic::Binop(Binary { lhs, rhs, op }) => {
            // StackAddress-typed literals are foldable; their type is preserved
            // through the output type computed by binop_result below.
            let lhs = get_numeric_const(ctx, lhs)?;
            let rhs = get_numeric_const(ctx, rhs)?;
            // Shifts take a shift *count* whose operand width may be narrower than
            // the value being shifted (e.g. `shr eax, cl`: 4-byte value, 1-byte
            // count). For every other binop the operands must be the same width —
            // the lifter guarantees it, so a mismatch is a type error.
            let is_shift = matches!(
                op,
                Binop::Int(IntBinop::ShiftLeft | IntBinop::ShiftRight | IntBinop::SShiftRight)
            );
            assert!(
                is_shift || lhs.size() == rhs.size(),
                "type error in binop constant folding: lhs is {} bytes, rhs is {} bytes",
                lhs.size(),
                rhs.size()
            );

            let l = lhs.value();
            let r = rhs.value();

            let size = lhs.size();
            let lhs_type = lhs.type_id();
            let rhs_type = rhs.type_id();

            let mask = if size >= 8 {
                u64::MAX
            } else {
                (1u64 << (size * 8)) - 1
            };

            let value = match op {
                Binop::Int(IntBinop::Equal) => (l == r) as u64,
                Binop::Int(IntBinop::NotEqual) => (l != r) as u64,
                Binop::Int(IntBinop::Less) => (l < r) as u64,
                Binop::Int(IntBinop::LessEqual) => (l <= r) as u64,

                Binop::Int(IntBinop::Add) => l.wrapping_add(r) & mask,
                Binop::Int(IntBinop::Sub) => l.wrapping_sub(r) & mask,

                Binop::Int(IntBinop::Xor) => (l ^ r) & mask,
                Binop::Int(IntBinop::And) => (l & r) & mask,
                Binop::Int(IntBinop::Or) => (l | r) & mask,
                // Logical shifts by >= the operand width yield 0 (pcode
                // semantics); going through Rust's `<<`/`>>` with such a count
                // would panic (debug) or wrap the count (release).
                Binop::Int(IntBinop::ShiftLeft) => {
                    if r >= size as u64 * 8 {
                        0
                    } else {
                        (l << r) & mask
                    }
                }
                Binop::Int(IntBinop::ShiftRight) => {
                    if r >= size as u64 * 8 {
                        0
                    } else {
                        (l >> r) & mask
                    }
                }
                // Arithmetic shift: sign-extend the lhs from its byte width to
                // i64, shift, then re-mask to the lhs width.
                Binop::Int(IntBinop::SShiftRight) => {
                    let bits = size * 8;
                    let signed = if bits >= 64 {
                        l as i64
                    } else {
                        ((l << (64 - bits)) as i64) >> (64 - bits)
                    };
                    (signed >> r.min(63)) as u64 & mask
                }

                Binop::Int(IntBinop::Mul) => l.wrapping_mul(r) & mask,
                // Division by a constant zero is left unfolded: `wrapping_div`
                // still panics on a zero divisor.
                Binop::Int(IntBinop::Div) => l.checked_div(r)? & mask,
                Binop::Int(IntBinop::Rem) => l.checked_rem(r)? & mask,

                Binop::Bool(BoolBinop::And) => (l != 0 && r != 0) as u64,
                Binop::Bool(BoolBinop::Or) => (l != 0 || r != 0) as u64,
                Binop::Bool(BoolBinop::Xor) => (l != 0) as u64 ^ (r != 0) as u64,

                _ => {
                    return None;
                }
            };

            // Preserve the semantic type (e.g. StackAddress) through folding.
            let out_type = ctx.types.binop_result(lhs_type, op, rhs_type);
            Some(ctx.get_typed_const(value, out_type).id())
        }

        Mnemonic::Unop(unop) => {
            let src = get_numeric_const(ctx, unop.src)?;

            let v = src.value();
            let size = src.size();
            let mask = src.mask();

            let value = match unop.op {
                Unop::IntNot => !v,
                Unop::IntNegate => v.wrapping_neg(),

                _ => {
                    return None;
                }
            };

            Some(ctx.get_const(value & mask, size).id())
        }

        Mnemonic::Zext(zext) => {
            let src = get_numeric_const(ctx, zext.src)?;
            Some(ctx.get_const(src.value(), zext.size).id())
        }

        Mnemonic::Sext(sext) => {
            let src = get_numeric_const(ctx, sext.src)?;
            // Sign-extend from the *source* width to 64 bits, then mask down to
            // the destination width.
            let src_bits = src.size() * 8;
            let extended = if src_bits >= 64 {
                src.value()
            } else {
                (((src.value() << (64 - src_bits)) as i64) >> (64 - src_bits)) as u64
            };
            Some(
                ctx.get_const(extended & all_ones(sext.size), sext.size)
                    .id(),
            )
        }

        Mnemonic::Range(range) => {
            // Extract `range.size` bytes starting at byte `range.start` of a
            // constant (e.g. EDI = low 4 bytes of a wide RDI literal).
            let src = get_numeric_const(ctx, range.src)?;
            let shifted = src.value().overflowing_shr(range.start as u32 * 8).0;
            Some(
                ctx.get_const(shifted & all_ones(range.size), range.size)
                    .id(),
            )
        }

        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Algebraic identities
// ---------------------------------------------------------------------------

/// The all-ones bit pattern for an `output_size`-byte value.
fn all_ones(output_size: usize) -> u64 {
    if output_size >= 8 {
        u64::MAX
    } else {
        (1u64 << (output_size * 8)) - 1
    }
}

/// Concrete value of `v` when it is a non-symbolic literal, else `None`.
pub(super) fn const_value(ctx: &Context, v: ValueId) -> Option<u64> {
    get_numeric_const(ctx, v).map(|c| c.value())
}

/// [`constant_folding`] then [`algebraic_identity`]: the value `m` collapses to
/// when either rewrite applies.
pub(super) fn try_fold(ctx: &mut Context, m: &Mnemonic, output_size: usize) -> Option<ValueId> {
    constant_folding(ctx, m, output_size).or_else(|| algebraic_identity(ctx, m, output_size))
}

/// Algebraic simplifications that, unlike [`constant_folding`], do *not* require
/// both operands to be constant: idempotent, self-inverse, identity-element and
/// annihilator laws. Returns the value the instruction collapses to (an existing
/// operand or an interned constant) when a law applies.
pub(super) fn algebraic_identity(
    ctx: &mut Context,
    m: &Mnemonic,
    output_size: usize,
) -> Option<ValueId> {
    let &Mnemonic::Binop(Binary { lhs, rhs, op }) = m else {
        return None;
    };
    let Binop::Int(op) = op else {
        return None;
    };

    // x OP x laws (operand identity, independent of any constant value).
    if lhs == rhs {
        match op {
            // x & x = x ; x | x = x
            IntBinop::And | IntBinop::Or => return Some(lhs),
            // x ^ x = 0 ; x - x = 0
            IntBinop::Xor | IntBinop::Sub => return Some(ctx.get_const(0, output_size).id()),
            _ => {}
        }
    }

    let l = const_value(ctx, lhs);
    let r = const_value(ctx, rhs);

    match op {
        // x + 0 = x ; 0 + x = x ; x - 0 = x  (Sub is not commutative)
        IntBinop::Add => {
            if r == Some(0) {
                return Some(lhs);
            }
            if l == Some(0) {
                return Some(rhs);
            }
        }
        IntBinop::Sub if r == Some(0) => {
            return Some(lhs);
        }
        // x | 0 = x ; x ^ 0 = x  (both commutative)
        IntBinop::Or | IntBinop::Xor => {
            if r == Some(0) {
                return Some(lhs);
            }
            if l == Some(0) {
                return Some(rhs);
            }
        }
        // x * 0 = 0 ; x * 1 = x
        IntBinop::Mul => {
            if l == Some(0) || r == Some(0) {
                return Some(ctx.get_const(0, output_size).id());
            }
            if r == Some(1) {
                return Some(lhs);
            }
            if l == Some(1) {
                return Some(rhs);
            }
        }
        // x & 0 = 0 ; x & ~0 = x
        IntBinop::And => {
            if l == Some(0) || r == Some(0) {
                return Some(ctx.get_const(0, output_size).id());
            }
            if r == Some(all_ones(output_size)) {
                return Some(lhs);
            }
            if l == Some(all_ones(output_size)) {
                return Some(rhs);
            }
        }
        // x << 0 = x ; x >> 0 = x  (shift amount is the rhs)
        IntBinop::ShiftLeft | IntBinop::ShiftRight | IntBinop::SShiftRight if r == Some(0) => {
            return Some(lhs);
        }
        _ => {}
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AliasResult;
    use crate::gvn::gvn;
    use qcode::value::{
        BasicBlock,
        insn::{Sext, Zext},
    };
    use qcode_macro::qcode;

    #[test]
    #[should_panic(expected = "type error in binop constant folding")]
    fn constant_folding_reports_mixed_size_literals() {
        let mut ctx = Context::new();
        let lhs = ctx.get_const(0xf0, 4).id();
        let rhs = ctx.get_const(0xff, 1).id();

        let _ = constant_folding(
            &mut ctx,
            &Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::And),
                lhs,
                rhs,
            }),
            4,
        );
    }

    #[test]
    fn constant_folding_preserves_stack_address_type_through_folding() {
        let mut ctx = Context::new();
        let stack = ctx.add_space(qcode::space::Space {
            name: Some(Box::from("stack")),
            word_size: 1,
            addr_size: 8,
            ty: qcode::space::SpaceType::Ram,
        });
        // StackAddress-typed literals carry provenance in their TypeId, not in a
        // symbolic annotation. Constant folding must propagate that type so that
        // alias analysis can still distinguish SA results from plain integers.
        let sa_type = ctx.types.get_or_make_stack_address(8, Some(stack));
        let base_lid = ctx
            .values
            .get_or_make_typed_literal(0x1000_0000_0000_0000, sa_type, 8);
        let base = ValueId::Literal(base_lid);
        let offset = ctx.get_const(8, 8).id();

        let folded = constant_folding(
            &mut ctx,
            &Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::Sub),
                lhs: base,
                rhs: offset,
            }),
            8,
        );

        let folded_id = folded.expect("SA - Int should constant-fold to a SA-typed literal");
        let ValueId::Literal(lid) = folded_id else {
            panic!("folded result must be a literal");
        };
        assert_eq!(
            ctx.values.literals[lid].type_id, sa_type,
            "folded SA - Int must preserve the StackAddress TypeId for alias analysis"
        );
    }

    // -----------------------------------------------------------------------
    // Algebraic identities
    // -----------------------------------------------------------------------

    /// `x & x` is idempotent: the AND is replaced by `x` itself.
    #[test]
    fn test_algebraic_and_self_is_idempotent() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i64 A;
                varnode i64 B;
                <block>
                    %a = load(i64, &A);
                    %v = %a & %a;
                    store(&B, %v);
                    goto <0x1001>;
            "
        );

        let aliases = AliasResult::simple(&ctx);
        let mut block = BasicBlock::from_id_mut(&mut ctx, block);
        assert!(block.instruction_ids().contains(&v));

        gvn(&mut block, Some(&aliases));

        assert!(
            !block.instruction_ids().contains(&v),
            "x & x should be eliminated"
        );
        assert!(
            block.instruction_ids().contains(&a),
            "the AND should be replaced by x itself, which stays live"
        );
    }

    /// `x + 0` collapses to `x`.
    #[test]
    fn test_algebraic_add_zero_identity() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i64 A;
                varnode i64 B;
                <block>
                    %a = load(i64, &A);
                    %v = %a + 0x0;
                    store(&B, %v);
                    goto <0x1001>;
            "
        );

        let aliases = AliasResult::simple(&ctx);
        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        gvn(&mut block, Some(&aliases));

        assert!(
            !block.instruction_ids().contains(&v),
            "x + 0 should be eliminated"
        );
        assert!(block.instruction_ids().contains(&a));
    }

    /// `x ^ x` folds to the zero constant.
    #[test]
    fn test_algebraic_xor_self_is_zero() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i64 A;
                varnode i64 B;
                <block>
                    %a = load(i64, &A);
                    %v = %a ^ %a;
                    store(&B, %v);
                    goto <0x1001>;
            "
        );

        let aliases = AliasResult::simple(&ctx);
        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        gvn(&mut block, Some(&aliases));

        assert!(
            !block.instruction_ids().contains(&v),
            "x ^ x should be eliminated"
        );
        assert!(
            block.to_string().contains("B = 0x0"),
            "x ^ x should fold to the zero constant, got:\n{block}"
        );
    }

    // -----------------------------------------------------------------------
    // Constant-folding edge cases
    // -----------------------------------------------------------------------

    /// Sext must sign-extend from the *source* width; an earlier version shifted
    /// by the destination width, computing `sext(x) << (dst_bits - src_bits)`.
    #[test]
    fn constant_folding_sext_extends_from_source_width() {
        let mut ctx = Context::new();

        let src = ctx.get_const(0x80, 1).id();
        let folded = constant_folding(&mut ctx, &Mnemonic::Sext(Sext { src, size: 4 }), 4)
            .expect("sext of a constant must fold");
        let ValueId::Literal(lid) = folded else {
            panic!("folded result must be a literal");
        };
        assert_eq!(ctx.values.literals[lid].value, 0xFFFF_FF80);

        let src = ctx.get_const(0x7f, 1).id();
        let folded = constant_folding(&mut ctx, &Mnemonic::Sext(Sext { src, size: 8 }), 8)
            .expect("sext of a constant must fold");
        let ValueId::Literal(lid) = folded else {
            panic!("folded result must be a literal");
        };
        assert_eq!(ctx.values.literals[lid].value, 0x7f);
    }

    /// A constant zero divisor must not fold (and must not panic).
    #[test]
    fn constant_folding_leaves_division_by_zero_unfolded() {
        for op in [IntBinop::Div, IntBinop::Rem] {
            let mut ctx = Context::new();
            let lhs = ctx.get_const(42, 4).id();
            let rhs = ctx.get_const(0, 4).id();
            let folded = constant_folding(
                &mut ctx,
                &Mnemonic::Binop(Binary {
                    op: Binop::Int(op),
                    lhs,
                    rhs,
                }),
                4,
            );
            assert!(
                folded.is_none(),
                "{op:?} by constant 0 must be left unfolded"
            );
        }
    }

    /// Logical shifts by >= the operand width fold to 0 (and must not panic on
    /// counts >= 64, which overflow Rust's shift operators).
    #[test]
    fn constant_folding_oversized_shift_counts_fold_to_zero() {
        for op in [IntBinop::ShiftLeft, IntBinop::ShiftRight] {
            let mut ctx = Context::new();
            let lhs = ctx.get_const(1, 8).id();
            let rhs = ctx.get_const(64, 8).id();
            let folded = constant_folding(
                &mut ctx,
                &Mnemonic::Binop(Binary {
                    op: Binop::Int(op),
                    lhs,
                    rhs,
                }),
                8,
            )
            .expect("oversized shift must fold to 0");
            let ValueId::Literal(lid) = folded else {
                panic!("folded result must be a literal");
            };
            assert_eq!(ctx.values.literals[lid].value, 0, "{op:?} by 64 must be 0");
        }
    }

    /// The symbolic-literal guard applies to cast/extract arms too, not just
    /// binops: folding a symbolic literal would discard its annotation.
    #[test]
    fn constant_folding_does_not_fold_symbolic_literals_in_casts() {
        let mut ctx = Context::new();
        let type_id = ctx.types.get_or_make_int(4);
        let lid = ctx.values.push_literal(qcode::value::literal::Literal {
            value: 0x1000,
            type_id,
            symbolic: Some(qcode::value::literal::SymbolicRef::String("s".into())),
        });
        let src = ValueId::Literal(lid);

        assert!(constant_folding(&mut ctx, &Mnemonic::Zext(Zext { src, size: 8 }), 8).is_none());
        assert!(constant_folding(&mut ctx, &Mnemonic::Sext(Sext { src, size: 8 }), 8).is_none());
        assert!(
            constant_folding(
                &mut ctx,
                &Mnemonic::Range(qcode::value::insn::Range {
                    src,
                    start: 0,
                    size: 2,
                }),
                2,
            )
            .is_none()
        );
    }
}
