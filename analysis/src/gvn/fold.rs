//! Constant-folding sub-pass: arithmetic on literals and algebraic identities.

use qcode::value::{
    FunctionId, QCodeView, Value, ValueId, ValueRef,
    insn::{Binary, Binop, IntBinop, Mnemonic, Unop},
    literal::LiteralRef,
    util::base_ref::AsShared,
};

use std::any::Any;

use super::walk::{Claim, Editor, InsnCtx, SubPass};

#[cfg(test)]
use qcode::context::Context;
#[cfg(test)]
use qcode::value::ModuleView;

use crate::{ContextView, FunctionBody};

/// Fold constant arithmetic and algebraic identities into interned literals.
///
/// Every body read goes through the selected function's static view and every
/// literal/type it mints goes through the interners' `&self` paths on shared
/// storage.
pub(super) struct Fold;

/// The function-pass [`SubPass`] impl (body-local):
/// `try_fold_insn` reads through `cx.body_view(body)` and the fold forwards
/// through `Editor::replace`.
impl<'str> SubPass<'str> for Fold {
    fn init_state(&self) -> Box<dyn Any> {
        Box::new(())
    }

    fn clone_state(&self, _state: &dyn Any) -> Box<dyn Any> {
        Box::new(())
    }

    fn on_insn(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        _state: &mut dyn Any,
        ic: &InsnCtx,
        ed: &mut Editor,
    ) -> Claim {
        if ic.mnemonic.is_terminator() || ic.size == 0 {
            return Claim::Pass;
        }
        match try_fold_insn(cx.body_view(body), ic) {
            Some(folded) => {
                ed.replace(body, cx, ic.insn_id, folded);
                Claim::Done
            }
            None => Claim::Pass,
        }
    }
}

// ---------------------------------------------------------------------------
// Constant folding
// ---------------------------------------------------------------------------

fn get_const<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    v: ValueId,
) -> Option<LiteralRef<'str, 'ctx>> {
    match ValueRef::from_view(host, v) {
        ValueRef::Literal(c) => Some(c),
        _ => None,
    }
}

fn is_symbolic_literal<'ctx, 'str: 'ctx>(host: impl QCodeView<'ctx, 'str>, v: ValueId) -> bool {
    let ValueId::Literal(id) = v else {
        return false;
    };
    host.shared().values.literals[id].symbolic.is_some()
}

/// `get_const`, but refusing Block/Function/String symbolic literals: those are
/// not numeric constants and folding them would discard the symbolic annotation.
/// StackAddress-typed literals have symbolic=None and are foldable.
fn get_numeric_const<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    v: ValueId,
) -> Option<LiteralRef<'str, 'ctx>> {
    if is_symbolic_literal(host, v) {
        return None;
    }
    get_const(host, v)
}

// TODO: use the existing `evaluate_const` machinery instead of re-implementing it here.
#[cfg(test)]
pub(super) fn constant_folding(
    ctx: &mut Context,
    m: &Mnemonic,
    output_size: usize,
) -> Option<ValueId> {
    // This helper models an instruction with a canonical integer result. The
    // real pass receives the instruction's already-created TypeId via InsnCtx.
    ctx.shared.types.get_or_make_int(output_size);
    // Test-only: the test mnemonics hold shared literal operands, which carry no
    // function, so any func qualifies them unchanged.
    constant_folding_with_location(
        ModuleView::new(&*ctx),
        FunctionId::from(0usize),
        m,
        output_size,
        None,
    )
}

fn constant_folding_with_location<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    func: FunctionId,
    m: &Mnemonic,
    output_size: usize,
    location: Option<&InsnCtx>,
) -> Option<ValueId> {
    match m {
        &Mnemonic::Binop(Binary { lhs, rhs, op }) => {
            let (lhs, rhs) = (lhs.qualify(func), rhs.qualify(func));
            // StackAddress-typed literals are foldable; their type is preserved
            // through the output type computed by binop_result below.
            let lhs_id = lhs;
            let rhs_id = rhs;
            let lhs = get_numeric_const(host, lhs)?;
            let rhs = get_numeric_const(host, rhs)?;
            // Binops require equal-width operands. If GVN sees mixed-width
            // literals, an earlier fold/rewrite failed to preserve a use-site
            // type and should be fixed there instead of normalizing it here.
            if lhs.size() != rhs.size() {
                log_mixed_width_binop(host, location, lhs_id, rhs_id, lhs.size(), rhs.size(), op);
            }
            assert!(
                lhs.size() == rhs.size(),
                "type error in binop constant folding at {}: lhs {} is {} bytes, rhs {} is {} bytes, op {}",
                fold_location(host, location),
                lhs.id(),
                lhs.size(),
                rhs.id(),
                rhs.size(),
                op,
            );

            let size = lhs.size();
            let lhs_type = lhs.type_id();
            let rhs_type = rhs.type_id();

            let mask = if size >= 8 {
                u64::MAX
            } else {
                (1u64 << (size * 8)) - 1
            };
            let l = lhs.value() & mask;
            let r = rhs.value() & mask;

            let value = match op {
                Binop::Int(IntBinop::Equal) => (l == r) as u64,
                Binop::Int(IntBinop::NotEqual) => (l != r) as u64,
                Binop::Int(IntBinop::Less) => (l < r) as u64,
                Binop::Int(IntBinop::LessEqual) => (l <= r) as u64,
                Binop::Int(IntBinop::SLess) => {
                    (signed_value(l, size) < signed_value(r, size)) as u64
                }
                Binop::Int(IntBinop::SLessEqual) => {
                    (signed_value(l, size) <= signed_value(r, size)) as u64
                }

                Binop::Int(IntBinop::Add) => l.wrapping_add(r) & mask,
                Binop::Int(IntBinop::Sub) => l.wrapping_sub(r) & mask,

                Binop::Int(IntBinop::Xor) => (l ^ r) & mask,
                Binop::Int(IntBinop::And) => (l & r) & mask,
                Binop::Int(IntBinop::Or) => (l | r) & mask,
                // Logical shifts by >= the operand width yield 0 (pcode
                // semantics); going through Rust's `<<`/`>>` with such a count
                // would panic (debug) or wrap the count (release).
                Binop::Int(IntBinop::ShiftLeft) => {
                    if r >= size as u64 * 8 || r >= u64::BITS as u64 {
                        0
                    } else {
                        (l << r) & mask
                    }
                }
                Binop::Int(IntBinop::ShiftRight) => {
                    if r >= size as u64 * 8 || r >= u64::BITS as u64 {
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

                _ => {
                    return None;
                }
            };

            // Preserve the semantic type (e.g. StackAddress) through folding.
            let out_type = host.shared().types.binop_result(lhs_type, op, rhs_type);
            let out_type = if host.shared().types.size_of(out_type) == output_size {
                out_type
            } else if let Some(location) = location {
                // Folding replaces this instruction's value. Preserve its
                // declared type exactly, including owned/named identities.
                location.type_id
            } else {
                host.shared().types.get_int(output_size)
            };
            Some(host.shared().get_typed_const(value, out_type))
        }

        Mnemonic::Unop(unop) => {
            let src = get_numeric_const(host, unop.src.qualify(func))?;

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

            Some(host.shared().get_const(value & mask, size))
        }

        Mnemonic::Zext(zext) => {
            let src = get_numeric_const(host, zext.src.qualify(func))?;
            Some(host.shared().get_const(src.value(), zext.size))
        }

        Mnemonic::Sext(sext) => {
            let src = get_numeric_const(host, sext.src.qualify(func))?;
            // Sign-extend from the *source* width to 64 bits, then mask down to
            // the destination width.
            let src_bits = src.size() * 8;
            let extended = if src_bits >= 64 {
                src.value()
            } else {
                (((src.value() << (64 - src_bits)) as i64) >> (64 - src_bits)) as u64
            };
            Some(
                host.shared()
                    .get_const(extended & all_ones(sext.size), sext.size),
            )
        }

        Mnemonic::Range(range) => {
            // Extract `range.size` bytes starting at byte `range.start` of a
            // constant. The source may be a numeric literal (e.g. EDI = low 4
            // bytes of a wide RDI literal) or an opaque byte blob.
            if let Some(bid) = range.src.qualify(func).as_bytes() {
                let data = &host.shared().values.bytes[bid].data;
                let start = range.start;
                let end = start.checked_add(range.size)?;
                let slice = data.get(start..end)?;
                if range.size <= 8 {
                    // Downconvert: a sub-blob that now fits in a u64 rejoins the
                    // numeric pipeline as an ordinary literal.
                    let mut buf = [0u8; 8];
                    buf[..slice.len()].copy_from_slice(slice);
                    return Some(host.shared().get_const(u64::from_le_bytes(buf), range.size));
                }
                // Still wider than a u64: a narrower byte blob.
                return Some(host.shared().get_bytes(slice.to_vec()));
            }
            let src = get_numeric_const(host, range.src.qualify(func))?;
            let shifted = src.value().overflowing_shr(range.start as u32 * 8).0;
            Some(
                host.shared()
                    .get_const(shifted & all_ones(range.size), range.size),
            )
        }

        // Pure intrinsics fold through their shared `eval` (the same evaluator
        // the emulator uses), when every operand is a non-symbolic constant.
        Mnemonic::Intrinsic(intr) => {
            let mut operands = Vec::with_capacity(intr.args.len());
            for &arg in &intr.args {
                let c = get_numeric_const(host, arg.qualify(func))?;
                operands.push((u128::from(c.value()), c.size()));
            }
            let value = intr.id.desc().eval(&operands, output_size)?;
            Some(host.shared().get_const(value as u64, output_size))
        }

        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Algebraic identities
// ---------------------------------------------------------------------------

/// The all-ones bit pattern for an `output_size`-byte value.
pub(super) fn all_ones(output_size: usize) -> u64 {
    if output_size >= 8 {
        u64::MAX
    } else {
        (1u64 << (output_size * 8)) - 1
    }
}

fn signed_value(value: u64, size: usize) -> i64 {
    let bits = size * 8;
    if bits >= 64 {
        value as i64
    } else {
        ((value << (64 - bits)) as i64) >> (64 - bits)
    }
}

/// Concrete value of `v` when it is a non-symbolic literal, else `None`.
///
/// Non-symbolic literals live in shared storage, so a bare `&Shared` suffices
/// (a `&Context` narrows via [`AsShared`]); a non-literal (including a
/// checked-out function's own instructions) is not a constant and yields `None`
/// without needing arena routing.
pub(super) fn const_value<'a, 'str: 'a>(src: impl AsShared<'a, 'str>, v: ValueId) -> Option<u64> {
    let shared = src.as_shared();
    let ValueId::Literal(id) = v else {
        return None;
    };
    if shared.values.literals[id].symbolic.is_some() {
        return None;
    }
    Some(LiteralRef::from_id(shared, id).value())
}

fn try_fold_insn<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    ic: &InsnCtx,
) -> Option<ValueId> {
    let func = ic.insn_id.func;
    let folded = constant_folding_with_location(host, func, ic.mnemonic, ic.size, Some(ic))
        .or_else(|| algebraic_identity(host, func, ic.mnemonic, ic.size))
        .or_else(|| cast_identity(host, func, ic.mnemonic))?;

    let root_ty = host.type_of(ic.id);
    let folded_ty = host.type_of(folded);
    let types = &host.shared().types;
    if types.is_bool(folded_ty) == types.is_bool(root_ty) {
        return Some(folded);
    }

    // Algebraic identities such as `x & 0 = 0` historically minted an
    // ordinary integer literal from `output_size`. When the root is a logical
    // bool operation, that silently changed `bool` into same-width `i8` and
    // made its users ill-typed. Guard only that bool/non-bool boundary: other
    // semantic type differences are intentionally transparent to GVN.
    // Constants crossing the boundary can be re-interned with the root type;
    // a non-constant requires an explicit cast and cannot replace the root.
    let ValueId::Literal(lid) = folded else {
        return None;
    };
    let literal = &host.shared().values.literals[lid];
    literal
        .symbolic
        .is_none()
        .then(|| host.shared().get_typed_const(literal.value, root_ty))
}

/// The size in bytes of `v`'s output.
fn value_size<'ctx, 'str: 'ctx>(host: impl QCodeView<'ctx, 'str>, v: ValueId) -> usize {
    ValueRef::from_view(host, v).size()
}

/// Width-preserving casts and extracts that collapse to an existing value.
/// Unlike [`constant_folding`] these need no constant operand — they hold for
/// any value whose width already matches:
///
/// * `zext(iN, v)` where `v` is already `N` bytes wide → `v`
/// * `v[0:N]` (i.e. `Range { start: 0, size: N }`) where `v` is `N` bytes → `v`
/// * `zext(_, v)[0:N]` where `v` is `N` bytes → `v` (e.g. `zext(i32, i1 v)[0:1]`)
pub(super) fn cast_identity<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    func: FunctionId,
    m: &Mnemonic,
) -> Option<ValueId> {
    match m {
        Mnemonic::Zext(zext) => {
            let src = zext.src.qualify(func);
            let src_is_bool = host
                .stored_type_of(src)
                .is_some_and(|ty| host.shared().types.is_bool(ty));
            (value_size(host, src) == zext.size && !src_is_bool).then_some(src)
        }
        Mnemonic::Range(range) if range.start == 0 => {
            let src = range.src.qualify(func);
            // `v[0:N]` keeps the low `N` bytes. If `v` is exactly `N` bytes the
            // extract is a no-op.
            if value_size(host, src) == range.size {
                return Some(src);
            }
            // `zext(_, inner)[0:N]` where `inner` is exactly `N` bytes: the low
            // `N` bytes of the zext are `inner` untouched, so the extract peels
            // the widening back off.
            if let ValueId::Instruction(id) = src
                && let Mnemonic::Zext(inner) = host.insn_ref(id).mnemonic()
                && value_size(host, inner.src.qualify(id.func)) == range.size
            {
                return Some(inner.src.qualify(id.func));
            }
            None
        }
        _ => None,
    }
}

fn log_mixed_width_binop<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    location: Option<&InsnCtx>,
    lhs: ValueId,
    rhs: ValueId,
    lhs_size: usize,
    rhs_size: usize,
    op: Binop,
) {
    qcode::pass_log!(
        warn,
        "mixed-width constant-fold binop at {}: op {op}, lhs {} bytes [{}], rhs {} bytes [{}]",
        fold_location(host, location),
        lhs_size,
        value_detail(host, lhs),
        rhs_size,
        value_detail(host, rhs),
    );
}

fn value_detail<'ctx, 'str: 'ctx>(host: impl QCodeView<'ctx, 'str>, value: ValueId) -> String {
    match value {
        ValueId::Literal(_) => {
            let ValueRef::Literal(lit) = ValueRef::from_view(host, value) else {
                unreachable!();
            };
            format!(
                "{value:?} literal value={:#x} size={} type={:?}",
                lit.value(),
                lit.size(),
                lit.type_id()
            )
        }
        ValueId::Instruction(id) => {
            let insn = host.insn_ref(id);
            let address = insn
                .address()
                .map(|addr| format!("{addr:#x}"))
                .unwrap_or_else(|| "unknown".to_string());
            let function = insn
                .function()
                .map(|f| f.name().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            format!(
                "{value:?} insn={id} addr={address} fn={function} size={} mnemonic={:?}",
                insn.size(),
                insn.mnemonic()
            )
        }
        _ => {
            let value_ref = ValueRef::from_view(host, value);
            format!("{value:?} value=`{value_ref}` size={}", value_ref.size())
        }
    }
}

fn fold_location<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    location: Option<&InsnCtx>,
) -> String {
    let Some(ic) = location else {
        return "unknown instruction".to_string();
    };
    let insn = host.insn_ref(ic.insn_id);
    let address = insn
        .address()
        .map(|addr| format!("{addr:#x}"))
        .unwrap_or_else(|| "unknown".to_string());
    format!(
        "address {address}, insn {}, value {}, block {}",
        ic.insn_id, ic.id, ic.block_id
    )
}

/// Algebraic simplifications that, unlike [`constant_folding`], do *not* require
/// both operands to be constant: idempotent, self-inverse, identity-element and
/// annihilator laws. Returns the value the instruction collapses to (an existing
/// operand or an interned constant) when a law applies.
/// Whether `v` is the literal `true`. A bool stores 0/1, so `true` plays the
/// all-ones-mask role that `all_ones(size)` plays for integer operands.
fn is_bool_true<'ctx, 'str: 'ctx>(host: impl QCodeView<'ctx, 'str>, v: ValueId) -> bool {
    const_value(host.shared(), v) == Some(1) && host.shared().types.is_bool(host.type_of(v))
}

pub(super) fn algebraic_identity<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    func: FunctionId,
    m: &Mnemonic,
    output_size: usize,
) -> Option<ValueId> {
    let &Mnemonic::Binop(Binary { lhs, rhs, op }) = m else {
        return None;
    };
    let (lhs, rhs) = (lhs.qualify(func), rhs.qualify(func));
    let Binop::Int(op) = op else {
        return None;
    };

    // x OP x laws (operand identity, independent of any constant value).
    if lhs == rhs {
        match op {
            // x & x = x ; x | x = x
            IntBinop::And | IntBinop::Or => return Some(lhs),
            // x ^ x = 0 ; x - x = 0
            IntBinop::Xor | IntBinop::Sub => {
                return Some(host.shared().get_const(0, output_size));
            }
            _ => {}
        }
    }

    let l = const_value(host.shared(), lhs);
    let r = const_value(host.shared(), rhs);

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
        // x | 0 = x ; x | true = true (bool annihilator)
        IntBinop::Or => {
            if r == Some(0) {
                return Some(lhs);
            }
            if l == Some(0) {
                return Some(rhs);
            }
            if is_bool_true(host, rhs) {
                return Some(rhs);
            }
            if is_bool_true(host, lhs) {
                return Some(lhs);
            }
        }
        // x ^ 0 = x
        IntBinop::Xor => {
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
                return Some(host.shared().get_const(0, output_size));
            }
            if r == Some(1) {
                return Some(lhs);
            }
            if l == Some(1) {
                return Some(rhs);
            }
        }
        // x & 0 = 0 ; x & ~0 = x ; x & true = x (bool identity)
        IntBinop::And => {
            if l == Some(0) || r == Some(0) {
                return Some(host.shared().get_const(0, output_size));
            }
            if r == Some(all_ones(output_size)) {
                return Some(lhs);
            }
            if l == Some(all_ones(output_size)) {
                return Some(rhs);
            }
            if is_bool_true(host, rhs) {
                return Some(lhs);
            }
            if is_bool_true(host, lhs) {
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
        BasicBlock, FunctionBody,
        insn::{Sext, Zext},
    };
    use qcode_macro::qcode;

    #[test]
    #[should_panic(expected = "type error in binop constant folding")]
    fn constant_folding_reports_mixed_size_literals() {
        let mut ctx = Context::new();
        let lhs = ctx.get_const(0xf0, 4).id().strip_func();
        let rhs = ctx.get_const(0xff, 1).id().strip_func();

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
    fn constant_folding_respects_instruction_result_width() {
        let mut ctx = Context::new();
        let lhs = ctx.get_const(0xffff_ffff, 8).id().strip_func();
        let rhs = ctx.get_const(0x1_0000_00ff, 8).id().strip_func();

        let folded = constant_folding(
            &mut ctx,
            &Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::And),
                lhs,
                rhs,
            }),
            4,
        )
        .expect("same-size literal binop should fold");

        let ValueId::Literal(lid) = folded else {
            panic!("folded result must be a literal");
        };
        assert_eq!(
            ctx.shared
                .types
                .size_of(ctx.shared.values.literals[lid].type_id),
            4
        );
        assert_eq!(ctx.shared.values.literals[lid].value, 0xff);
    }

    #[test]
    fn bool_annihilator_keeps_boolean_constant_type() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry @a:i8 @b:i8>
                    %cond = @a == @b;
                    %masked = %cond & false;
                    %combined = %masked | %cond;
                    return %combined;
            "
        );

        assert!(crate::verify::verify_bool_typing(&ctx).is_empty());
        gvn(&mut BasicBlock::from_id_mut(&mut ctx, entry), None);
        assert!(
            crate::verify::verify_bool_typing(&ctx).is_empty(),
            "folding a bool annihilator must mint bool false, not i8 zero"
        );
    }

    /// `true` is bool's all-ones mask, so `x & true = x` and `x | true = true`
    /// (and `x | false = x` via the integer zero law) must all fold without
    /// crossing the bool/integer boundary.
    #[test]
    fn bool_identity_and_annihilator_laws_fold() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry @a:i8 @b:i8>
                    %cond = @a == @b;
                    %and_true = %cond & true;
                    %or_false = %and_true | false;
                    %or_true = %or_false | true;
                    return %or_true;
            "
        );

        assert!(crate::verify::verify_bool_typing(&ctx).is_empty());

        let view = ModuleView::new(&ctx);
        assert_eq!(
            algebraic_identity(view, f, ctx.get_insn(and_true).mnemonic(), 1),
            Some(ValueId::Instruction(cond)),
            "x & true must fold to x"
        );
        assert_eq!(
            algebraic_identity(view, f, ctx.get_insn(or_false).mnemonic(), 1),
            Some(ValueId::Instruction(and_true)),
            "x | false must fold to x"
        );
        let folded = algebraic_identity(view, f, ctx.get_insn(or_true).mnemonic(), 1)
            .expect("x | true must fold to true");
        assert_eq!(const_value(&ctx.shared, folded), Some(1));
        assert!(
            ctx.shared.types.is_bool(view.type_of(folded)),
            "the annihilator result must stay bool-typed"
        );

        gvn(&mut BasicBlock::from_id_mut(&mut ctx, entry), None);
        assert!(crate::verify::verify_bool_typing(&ctx).is_empty());
    }

    /// Constant-folding `StructPointer + Int` (the shape `windows_teb_seed`
    /// creates: a `TEB*`-typed base plus a field offset). Must fold to the sum
    /// without panicking on the non-integer pointer operand, and the result must
    /// keep the pointer type so struct typing can still recognize it downstream.
    #[test]
    fn constant_folding_on_pointer_addition() {
        let mut ctx = Context::new();
        let teb = ctx.shared.types.get_or_make_struct("TEB", 0x1000, vec![]);
        let teb_ptr = ctx.shared.types.get_or_make_struct_pointer(4, teb);
        let base = ctx
            .shared
            .values
            .get_or_make_typed_literal(0x7ffd_f000, teb_ptr, 4);
        let base = ValueId::Literal(base);
        let offset = ctx.get_const(0x30, 4).id();

        let folded = constant_folding(
            &mut ctx,
            &Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::Add),
                lhs: base.strip_func(),
                rhs: offset.strip_func(),
            }),
            4,
        );

        let folded_id = folded.expect("TEB* + Int should constant-fold");
        let ValueId::Literal(lid) = folded_id else {
            panic!("folded result must be a literal");
        };
        assert_eq!(ctx.shared.values.literals[lid].value, 0x7ffd_f030);
        assert_eq!(
            ctx.shared.values.literals[lid].type_id, teb_ptr,
            "folded TEB* + Int must preserve the struct-pointer type"
        );
    }

    /// The offset feeding a struct-pointer add must still constant-fold. Models
    /// the real `fs:[c]` shape where `c` is a foldable expression (`0x31 + 0x32`)
    /// rather than a bare literal, and the base is a `TEB*`-typed varnode. The
    /// pointer-typed add (`teb + c`) must not suppress folding of its int offset.
    #[test]
    fn typed_pointer_add_does_not_suppress_folding_its_offset() {
        use crate::gvn::gvn_function;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 fs;
            fn f:
                <entry>
                    %c = i32 0x31 + i32 0x32;
                    %addr = &fs + %c;
                    %v = load(fs:4, %addr);
                    return at i32 0;
            "
        );
        // Type `fs` as a struct pointer, exactly as `windows_teb_seed` does.
        let teb = ctx.shared.types.get_or_make_struct("TEB", 0x1000, vec![]);
        let teb_ptr = ctx.shared.types.get_or_make_struct_pointer(4, teb);
        ctx.set_varnode_type(fs, teb_ptr);

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, f, Some(&aliases));

        // `0x31 + 0x32` must fold to 0x63 — so the address add reads `fs + 0x63`,
        // the canonical `base + const` shape struct typing needs.
        let addr_rhs = FunctionBody::from_id(&ctx, f)
            .blocks()
            .flat_map(|b| b.iter().collect::<Vec<_>>())
            .find_map(|i| match i.mnemonic() {
                Mnemonic::Binop(Binary {
                    rhs,
                    op: Binop::Int(IntBinop::Add),
                    ..
                }) => Some(rhs),
                _ => None,
            })
            .expect("an add survives");
        assert_eq!(
            const_value(&ctx, addr_rhs.qualify(f)),
            Some(0x63),
            "the offset feeding the TEB* add must fold to 0x63"
        );
    }

    /// The byte-assembly idiom `zext(b0) | (zext(b1) << 8) | …` over constant
    /// bytes must fold to the assembled constant. This is how the test binary
    /// builds the `fs:[c]` offset; if it stays unfolded, struct typing can never
    /// match a constant field offset.
    #[test]
    fn byte_assembly_of_constants_folds() {
        use crate::gvn::gvn_function;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry>
                    varnode i32 sink;
                    %b0 = zext(i32, i8 0x30);
                    %b1 = zext(i32, i8 0x0);
                    %s1 = %b1 << i32 0x8;
                    %o1 = %b0 | %s1;
                    %b2 = zext(i32, i8 0x0);
                    %s2 = %b2 << i32 0x10;
                    %o2 = %o1 | %s2;
                    store(sink:4, &sink <- %o2);
                    return at i32 0;
            "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, f, Some(&aliases));

        // The stored value must be the folded constant 0x30.
        let stored = FunctionBody::from_id(&ctx, f)
            .blocks()
            .flat_map(|b| b.iter().collect::<Vec<_>>())
            .find_map(|i| match i.mnemonic() {
                Mnemonic::Store(s) => Some(s.src),
                _ => None,
            })
            .expect("a store survives");
        assert_eq!(
            const_value(&ctx, stored.qualify(f)),
            Some(0x30),
            "the byte-assembly chain must fold to 0x30"
        );
    }

    /// Faithful reproduction of the test binary's `fs:[c]` shape: the offset
    /// byte is written by a narrow store overwriting a wider one, read back,
    /// zext-assembled, then added to a `TEB*`-typed base — all in one function.
    /// The forwarded byte must fold so the address add reads `fs + 0x30`.
    #[test]
    fn forwarded_byte_offset_to_typed_pointer_folds() {
        use crate::gvn::gvn_function;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 fs;
            varnode i32 slot;
            fn f:
                <entry>
                    store(slot:4, &slot <- i32 0x1f1e1d2c);
                    store(slot:1, &slot <- i8 0x30);
                    %lo = load(slot:1, &slot);
                    %b0 = zext(i32, %lo);
                    %b1 = zext(i32, i8 0x0);
                    %s1 = %b1 << i32 0x8;
                    %off = %b0 | %s1;
                    %addr = &fs + %off;
                    %v = load(fs:4, %addr);
                    return at i32 0;
            "
        );
        let teb = ctx.shared.types.get_or_make_struct("TEB", 0x1000, vec![]);
        let teb_ptr = ctx.shared.types.get_or_make_struct_pointer(4, teb);
        ctx.set_varnode_type(fs, teb_ptr);

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, f, Some(&aliases));

        let addr_rhs = FunctionBody::from_id(&ctx, f)
            .blocks()
            .flat_map(|b| b.iter().collect::<Vec<_>>())
            .find_map(|i| match i.mnemonic() {
                Mnemonic::Binop(Binary {
                    rhs,
                    op: Binop::Int(IntBinop::Add),
                    ..
                }) => Some(*rhs),
                _ => None,
            })
            .expect("an add survives");
        assert_eq!(
            const_value(&ctx, addr_rhs.qualify(f)),
            Some(0x30),
            "the forwarded byte offset feeding the TEB* add must fold to 0x30"
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
                    %a = load(A:8, &A);
                    %v = %a & %a;
                    store(B:8, &B <- %v);
                    goto <0x1001>;
            "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
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
                    %a = load(A:8, &A);
                    %v = %a + 0x0;
                    store(B:8, &B <- %v);
                    goto <0x1001>;
            "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
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
                    %a = load(A:8, &A);
                    %v = %a ^ %a;
                    store(B:8, &B <- %v);
                    goto <0x1001>;
            "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        gvn(&mut block, Some(&aliases));

        assert!(
            !block.instruction_ids().contains(&v),
            "x ^ x should be eliminated"
        );
        assert!(
            block.to_string().contains("B <- i64 0x0"),
            "x ^ x should fold to the zero constant, got:\n{block}"
        );
    }

    // -----------------------------------------------------------------------
    // Cast / extract identities
    // -----------------------------------------------------------------------

    /// `zext(i32, i32 v)` is a no-op: the zext is replaced by `v`.
    #[test]
    fn test_zext_to_same_width_is_noop() {
        use crate::gvn::gvn_function;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry>
                    varnode i32 A;
                    varnode i32 B;
                    %a = load(A:4, &A);
                    %z = zext(i32, %a);
                    store(B:4, &B <- %z);
                    return at i32 0;
            "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, f, Some(&aliases));

        let stored = FunctionBody::from_id(&ctx, f)
            .blocks()
            .flat_map(|b| b.iter().collect::<Vec<_>>())
            .find_map(|i| match i.mnemonic() {
                Mnemonic::Store(s) => Some(s.src),
                _ => None,
            })
            .expect("a store survives");
        assert_eq!(
            stored.qualify(f),
            ValueId::Instruction(a),
            "zext(i32, i32 v) must collapse to v"
        );
    }

    /// `v[0:4]` for a 4-byte `v` is a no-op: the range is replaced by `v`.
    #[test]
    fn test_full_width_range_is_noop() {
        use crate::gvn::gvn_function;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry>
                    varnode i32 A;
                    varnode i32 B;
                    %a = load(A:4, &A);
                    %r = %a[0:4];
                    store(B:4, &B <- %r);
                    return at i32 0;
            "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, f, Some(&aliases));

        let stored = FunctionBody::from_id(&ctx, f)
            .blocks()
            .flat_map(|b| b.iter().collect::<Vec<_>>())
            .find_map(|i| match i.mnemonic() {
                Mnemonic::Store(s) => Some(s.src),
                _ => None,
            })
            .expect("a store survives");
        assert_eq!(
            stored.qualify(f),
            ValueId::Instruction(a),
            "v[0:4] for a 4-byte v must collapse to v"
        );
    }

    /// `zext(i32, i1 v)[0:1]` is `v`: the extract peels off the widening.
    #[test]
    fn test_range_of_zext_back_to_source_is_noop() {
        use crate::gvn::gvn_function;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry>
                    varnode i8 A;
                    varnode i8 B;
                    %a = load(A:1, &A);
                    %z = zext(i32, %a);
                    %r = %z[0:1];
                    store(B:1, &B <- %r);
                    return at i32 0;
            "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, f, Some(&aliases));

        let stored = FunctionBody::from_id(&ctx, f)
            .blocks()
            .flat_map(|b| b.iter().collect::<Vec<_>>())
            .find_map(|i| match i.mnemonic() {
                Mnemonic::Store(s) => Some(s.src),
                _ => None,
            })
            .expect("a store survives");
        assert_eq!(
            stored.qualify(f),
            ValueId::Instruction(a),
            "zext(i32, i1 v)[0:1] must collapse to v"
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

        let src = ctx.get_const(0x80, 1).id().strip_func();
        let folded = constant_folding(&mut ctx, &Mnemonic::Sext(Sext { src, size: 4 }), 4)
            .expect("sext of a constant must fold");
        let ValueId::Literal(lid) = folded else {
            panic!("folded result must be a literal");
        };
        assert_eq!(ctx.shared.values.literals[lid].value, 0xFFFF_FF80);

        let src = ctx.get_const(0x7f, 1).id().strip_func();
        let folded = constant_folding(&mut ctx, &Mnemonic::Sext(Sext { src, size: 8 }), 8)
            .expect("sext of a constant must fold");
        let ValueId::Literal(lid) = folded else {
            panic!("folded result must be a literal");
        };
        assert_eq!(ctx.shared.values.literals[lid].value, 0x7f);
    }

    /// A constant zero divisor must not fold (and must not panic).
    #[test]
    fn constant_folding_leaves_division_by_zero_unfolded() {
        for op in [IntBinop::Div, IntBinop::Rem] {
            let mut ctx = Context::new();
            let lhs = ctx.get_const(42, 4).id().strip_func();
            let rhs = ctx.get_const(0, 4).id().strip_func();
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
            let lhs = ctx.get_const(1, 8).id().strip_func();
            let rhs = ctx.get_const(64, 8).id().strip_func();
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
            assert_eq!(
                ctx.shared.values.literals[lid].value, 0,
                "{op:?} by 64 must be 0"
            );
        }
    }

    /// The symbolic-literal guard applies to cast/extract arms too, not just
    /// binops: folding a symbolic literal would discard its annotation.
    #[test]
    fn constant_folding_does_not_fold_symbolic_literals_in_casts() {
        let mut ctx = Context::new();
        let type_id = ctx.shared.types.get_or_make_int(4);
        let lid = ctx
            .shared
            .values
            .push_literal(qcode::value::literal::Literal {
                value: 0x1000,
                type_id,
                symbolic: Some(qcode::value::literal::SymbolicRef::String("s".into())),
            });
        let src = ValueId::Literal(lid).strip_func();

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

    /// Extracting <= 8 bytes from a byte blob downconverts to a numeric literal,
    /// reassembled little-endian.
    #[test]
    fn range_of_bytes_downconverts_to_literal() {
        let mut ctx = Context::new();
        let src = ctx
            .get_bytes(vec![
                0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa,
            ])
            .id()
            .strip_func();

        let folded = constant_folding(
            &mut ctx,
            &Mnemonic::Range(qcode::value::insn::Range {
                src,
                start: 2,
                size: 4,
            }),
            4,
        )
        .expect("range of a byte blob folds");

        let ValueId::Literal(lid) = folded else {
            panic!("expected a numeric literal, got {folded:?}");
        };
        // bytes 2..6 = 33 44 55 66, little-endian => 0x66554433
        assert_eq!(
            qcode::value::LiteralRef::from_id(&ctx, lid).value(),
            0x6655_4433
        );
    }

    /// Extracting > 8 bytes from a byte blob yields a narrower byte blob.
    #[test]
    fn range_of_bytes_keeps_wide_slice_as_bytes() {
        let mut ctx = Context::new();
        let src = ctx.get_bytes((0..16u8).collect()).id().strip_func();

        let folded = constant_folding(
            &mut ctx,
            &Mnemonic::Range(qcode::value::insn::Range {
                src,
                start: 2,
                size: 12,
            }),
            12,
        )
        .expect("range of a byte blob folds");

        let ValueId::Bytes(bid) = folded else {
            panic!("expected a byte blob, got {folded:?}");
        };
        assert_eq!(
            ctx.shared.values.bytes[bid].data,
            (2..14u8).collect::<Vec<_>>()
        );
    }
}
