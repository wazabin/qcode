use qcode::{
    context::Context,
    value::{
        Value, ValueId, ValueRef,
        insn::{Binary, Binop, BoolBinop, FloatBinop, IntBinop, Mnemonic, Unop},
        literal::LiteralRef,
    },
};

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
fn const_value(ctx: &Context, v: ValueId) -> Option<u64> {
    get_numeric_const(ctx, v).map(|c| c.value())
}

/// [`constant_folding`] then [`algebraic_identity`]: the value `m` collapses to
/// when either rewrite applies. Shared by the standalone constant-folding pass
/// and the GVN block walk so the two cannot drift.
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

// ---------------------------------------------------------------------------
// Flag-idiom recognition
// ---------------------------------------------------------------------------

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

/// If `v` is defined by an `sborrow`, return its `(lhs, rhs)`.
fn as_sborrow(ctx: &Context, v: ValueId) -> Option<(ValueId, ValueId)> {
    let ValueId::Instruction(id) = v else {
        return None;
    };
    match ctx.get_insn(id).mnemonic() {
        Mnemonic::SBorrow(sb) => Some((sb.lhs, sb.rhs)),
        _ => None,
    }
}

/// Recognize the x86 signed-less-than flag idiom and collapse it to `a s< b`.
///
/// A signed `cmp a, b` lowers to `OF != SF` of `a - b`:
/// ```text
///   %of  = sborrow(a, b)
///   %sub = a - b
///   %sf  = %sub s< 0
///   %lt  = %of != %sf      // == (a s< b)
/// ```
/// Returns the rewritten `SLess(a, b)` mnemonic. The original `sborrow`/`sub`/
/// `slt` instructions are left for DCE to remove once their last use is gone.
pub(super) fn simplify_flag_idiom(ctx: &Context, m: &Mnemonic) -> Option<Mnemonic> {
    let &Mnemonic::Binop(Binary {
        lhs,
        rhs,
        op: Binop::Int(IntBinop::NotEqual),
    }) = m
    else {
        return None;
    };

    // The `!=` is commutative (and GVN may have normalized it), so try both
    // assignments of which side is the sborrow and which is the `s< 0`.
    let resolve = |sborrow_side: ValueId, slt_side: ValueId| -> Option<(ValueId, ValueId)> {
        let (a, b) = as_sborrow(ctx, sborrow_side)?;
        let (sub_v, zero) = as_int_binop(ctx, slt_side, IntBinop::SLess)?;
        if const_value(ctx, zero) != Some(0) {
            return None;
        }
        let (sa, sb) = as_int_binop(ctx, sub_v, IntBinop::Sub)?;
        (sa == a && sb == b).then_some((a, b))
    };

    let (a, b) = resolve(lhs, rhs).or_else(|| resolve(rhs, lhs))?;
    Some(Mnemonic::Binop(Binary {
        lhs: a,
        rhs: b,
        op: Binop::Int(IntBinop::SLess),
    }))
}
