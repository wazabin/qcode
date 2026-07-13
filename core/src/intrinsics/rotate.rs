//! Bit-rotation intrinsics: `rol` (rotate left) and `ror` (rotate right).
//!
//! Both are pure 2-operand intrinsics `(x, k)` producing an `x`-wide result.
//! `rol` additionally recognizes the `(x << c1) | (x >> c2)` idiom; `ror`'s
//! constant form shares that shape and is normalised into a `rol`.

use crate::register_intrinsic;
use crate::types::{TypeId, TypeManager};
use crate::value::ValueId;
use crate::value::insn::intrinsic::{as_int_binop, const_u64, mask_for};
use crate::value::insn::{
    InstructionId, InstructionRef, IntBinop, Intrinsic, IntrinsicApp, IntrinsicId, Mnemonic,
    RootOp, Simplified,
};
use crate::value::util::base_ref::HostRef;

fn eval_rotate(args: &[(u128, usize)], out_size: usize, left: bool) -> Option<u128> {
    let (x, _) = *args.first()?;
    let (k, _) = *args.get(1)?;
    let bits = (out_size * 8) as u32;
    if bits == 0 || bits > 128 {
        return None;
    }
    let mask = mask_for(out_size);
    let x = x & mask;
    let k = (k % bits as u128) as u32;
    let res = if k == 0 {
        x
    } else if left {
        (x << k) | (x >> (bits - k))
    } else {
        (x >> k) | (x << (bits - k))
    };
    Some(res & mask)
}

fn eval_rol(args: &[(u128, usize)], out_size: usize) -> Option<u128> {
    eval_rotate(args, out_size, true)
}

fn eval_ror(args: &[(u128, usize)], out_size: usize) -> Option<u128> {
    eval_rotate(args, out_size, false)
}

/// The left-shift side of the rotate idiom: a real `x << c1`, or the
/// strength-reduced `x * 2^c1` (either operand order) that affine
/// canonicalization leaves behind. Returns `(x, c1, amount)` where `c1` is the
/// shift count and `amount` is the existing literal for it when the source was
/// a shift (so the caller can reuse it), or `None` for the multiply form (whose
/// amount must be materialized).
fn as_shl_idiom(host: HostRef, v: ValueId) -> Option<(ValueId, u64, Option<ValueId>)> {
    // Canonical shift form: `x << c1`.
    if let Some((x, c)) = as_int_binop(host, v, IntBinop::ShiftLeft) {
        let cv = const_u64(host, c)?;
        return Some((x, cv, Some(c)));
    }
    // Strength-reduced form: `x * 2^c1` (or `2^c1 * x`). Recover the shift count
    // as the log2 of the power-of-two multiplier.
    if let Some((a, b)) = as_int_binop(host, v, IntBinop::Mul) {
        let (x, m) = match (const_u64(host, b), const_u64(host, a)) {
            (Some(m), _) => (a, m),
            (None, Some(m)) => (b, m),
            _ => return None,
        };
        if m.is_power_of_two() {
            return Some((x, m.trailing_zeros() as u64, None));
        }
    }
    None
}

/// Recognize `(x << c1) | (x >> c2)` with `c1 + c2 == bits` as `rol(x, c1)`.
///
/// Tries both operand orderings of the `or`, and accepts the strength-reduced
/// `x * 2^c1` in place of `x << c1` (see [`as_shl_idiom`]). The constant ror
/// idiom is captured here too: `ror(x, c2)` has the same shape and is
/// represented as `rol(x, bits - c2)`.
fn recognize_rol(host: HostRef, root: crate::value::InstructionId) -> Option<Vec<ValueId>> {
    let root_size = InstructionRef::new(host, root).size();
    let bits = (root_size * 8) as u64;
    if bits == 0 {
        return None;
    }

    let (lhs, rhs) = as_int_binop(host, ValueId::Instruction(root), IntBinop::Or)?;

    // (shl_side, shr_side) — try both orderings of the commutative `or`.
    for (shl_side, shr_side) in [(lhs, rhs), (rhs, lhs)] {
        let Some((x1, c1v, c1_amount)) = as_shl_idiom(host, shl_side) else {
            continue;
        };
        let Some((x2, c2)) = as_int_binop(host, shr_side, IntBinop::ShiftRight) else {
            continue;
        };
        if x1 != x2 {
            continue;
        }
        let Some(c2v) = const_u64(host, c2) else {
            continue;
        };
        if c1v == 0 || c2v == 0 || c1v + c2v != bits {
            continue;
        }
        // rol(x, c1): reuse the left-shift amount literal when present, else
        // materialize one (sized like the right-shift amount) for the `x * 2^c1`
        // form.
        let amount = match c1_amount {
            Some(existing) => existing,
            None => {
                let amt_size = host.shr().types.size_of(host.type_of(c2));
                host.shr().get_const(c1v, amt_size)
            }
        };
        return Some(vec![x1, amount]);
    }
    None
}

/// If `v` is defined by a rotate intrinsic, return `(name, x, k)`.
fn as_rotate(host: HostRef, v: ValueId) -> Option<(&'static str, ValueId, ValueId)> {
    let ValueId::Instruction(id) = v else {
        return None;
    };
    let Mnemonic::Intrinsic(intr) = host.instruction(id).mnemonic() else {
        return None;
    };
    let name = intr.id.name();
    if name != "rol" && name != "ror" {
        return None;
    }
    let &[x, k] = intr.args.as_slice() else {
        return None;
    };
    // Operands are stored bare-local; qualify with the intrinsic's own function.
    Some((name, x.qualify(id.func), k.qualify(id.func)))
}

/// Simplify a rotate `op(x, k)` (where `op` is `rol`/`ror`, named by `id`) over
/// an `out_size`-byte result:
///
/// * **Inverse cancellation** — `ror(rol(a, c), c) → a` and `rol(ror(a, c), c)
///   → a`. The inner amount must provably match the outer one: the same value,
///   or two constants equal modulo the bit width (rotating by `c` then by `-c`
///   is the identity for any `c`).
/// * **Modulo-width** — for a constant amount `c`, normalise to `c mod bits`:
///   `c` that is a multiple of the width collapses to `x`; otherwise the rotate
///   is rebuilt on the reduced amount (e.g. `rol(a, 33) → rol(a, 1)` at 32 bits).
fn simplify_rotate(
    host: HostRef,
    id: IntrinsicId,
    out_size: usize,
    args: &[ValueId],
) -> Option<Simplified> {
    let &[x, k] = args else {
        return None;
    };
    let bits = (out_size * 8) as u64;
    if bits == 0 {
        return None;
    }

    // Inverse cancellation: op(inv_op(a, k2), k) → a when k and k2 agree.
    if let Some((inner_name, a, k2)) = as_rotate(host, x) {
        let outer_name = id.name();
        let is_inverse = (outer_name == "rol" && inner_name == "ror")
            || (outer_name == "ror" && inner_name == "rol");
        let same_amount = k == k2
            || match (const_u64(host, k), const_u64(host, k2)) {
                (Some(a), Some(b)) => a % bits == b % bits,
                _ => false,
            };
        if is_inverse && same_amount {
            return Some(Simplified::Value(a));
        }
    }

    // Modulo-width normalisation of a constant amount.
    if let Some(c) = const_u64(host, k) {
        let r = c % bits;
        if r == 0 {
            // A whole number of turns: the rotate is the identity.
            return Some(Simplified::Value(x));
        }
        if r != c {
            // Rebuild the same rotate on the reduced amount. The amount keeps
            // the operand's width.
            let k_size = host.shr().types.size_of(host.type_of(k));
            let reduced = host.shr().get_const(r, k_size);
            let rotate = IntrinsicApp {
                id,
                // Expression operands live in the same body; store bare-local.
                args: vec![x.strip_func(), reduced.strip_func()],
            };
            return Some(Simplified::Expression(Mnemonic::Intrinsic(rotate)));
        }
    }

    None
}

/// A rotate's result is the same sized integer as its first operand.
fn rotate_result_type(types: &TypeManager, args: &[TypeId]) -> TypeId {
    types.get_or_make_int(types.size_of(args[0]))
}

/// `rol` — rotate left. Recognizes the `(x << c1) | (x >> c2)` idiom.
struct Rol;

impl Intrinsic for Rol {
    fn name(&self) -> &'static str {
        "rol"
    }
    fn arity(&self) -> usize {
        2
    }
    fn result_type(&self, types: &TypeManager, args: &[TypeId]) -> TypeId {
        rotate_result_type(types, args)
    }
    fn eval(&self, args: &[(u128, usize)], out_size: usize) -> Option<u128> {
        eval_rol(args, out_size)
    }
    fn root_op(&self) -> Option<RootOp> {
        Some(RootOp::IntBinop(IntBinop::Or))
    }
    fn recognize(&self, host: HostRef, at: InstructionId) -> Option<Vec<ValueId>> {
        recognize_rol(host, at)
    }
    fn simplify(
        &self,
        host: HostRef,
        id: IntrinsicId,
        out_size: usize,
        args: &[ValueId],
    ) -> Option<Simplified> {
        simplify_rotate(host, id, out_size, args)
    }
}

/// `ror` — rotate right. Its constant idiom is normalised into a `rol` by the
/// recognizer, so it has no `root_op` of its own.
struct Ror;

impl Intrinsic for Ror {
    fn name(&self) -> &'static str {
        "ror"
    }
    fn arity(&self) -> usize {
        2
    }
    fn result_type(&self, types: &TypeManager, args: &[TypeId]) -> TypeId {
        rotate_result_type(types, args)
    }
    fn eval(&self, args: &[(u128, usize)], out_size: usize) -> Option<u128> {
        eval_ror(args, out_size)
    }
    fn simplify(
        &self,
        host: HostRef,
        id: IntrinsicId,
        out_size: usize,
        args: &[ValueId],
    ) -> Option<Simplified> {
        simplify_rotate(host, id, out_size, args)
    }
}

register_intrinsic!(Rol);
register_intrinsic!(Ror);

#[cfg(test)]
mod tests {
    use crate::value::insn::{IntBinop, IntrinsicId, RootOp, recognizers_for};

    #[test]
    fn rol_ror_registered_and_resolve() {
        let rol = IntrinsicId::from_name("rol").expect("rol registered");
        let ror = IntrinsicId::from_name("ror").expect("ror registered");
        assert_eq!(rol.name(), "rol");
        assert_eq!(ror.name(), "ror");
        assert_eq!(rol.desc().arity(), 2);
        assert!(IntrinsicId::from_name("nope").is_none());
    }

    #[test]
    fn eval_rol_matches_native() {
        let rol = IntrinsicId::from_name("rol").unwrap();
        // rol(0x12345678, 8) over 4 bytes == u32 rotate_left
        let got = rol.desc().eval(&[(0x1234_5678, 4), (8, 4)], 4).unwrap();
        assert_eq!(got as u32, 0x1234_5678u32.rotate_left(8));
    }

    #[test]
    fn eval_ror_matches_native() {
        let ror = IntrinsicId::from_name("ror").unwrap();
        let got = ror.desc().eval(&[(0x1234_5678, 4), (12, 4)], 4).unwrap();
        assert_eq!(got as u32, 0x1234_5678u32.rotate_right(12));
    }

    #[test]
    fn rol_zero_is_identity_eval() {
        let rol = IntrinsicId::from_name("rol").unwrap();
        let got = rol.desc().eval(&[(0xdead_beef, 4), (0, 4)], 4).unwrap();
        assert_eq!(got as u32, 0xdead_beef);
    }

    #[test]
    fn recognizers_indexed_by_root() {
        let ids = recognizers_for(RootOp::IntBinop(IntBinop::Or));
        assert!(ids.iter().any(|id| id.name() == "rol"));
    }
}
