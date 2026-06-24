//! Bit-rotation intrinsics: `rol` (rotate left) and `ror` (rotate right).
//!
//! Both are pure 2-operand intrinsics `(x, k)` producing an `x`-wide result.
//! `rol` additionally recognizes the `(x << c1) | (x >> c2)` idiom; `ror`'s
//! constant form shares that shape and is normalised into a `rol`.

use crate::context::Context;
use crate::register_intrinsic;
use crate::value::ValueId;
use crate::value::insn::intrinsic::{as_int_binop, const_u64, mask_for};
use crate::value::insn::{IntBinop, Intrinsic, IntrinsicId, Mnemonic, RootOp, Simplified};

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

/// Recognize `(x << c1) | (x >> c2)` with `c1 + c2 == bits` as `rol(x, c1)`.
///
/// Tries both operand orderings of the `or`. The constant ror idiom is
/// captured here too: `ror(x, c2)` has the same shape and is represented as
/// `rol(x, bits - c2)`.
fn recognize_rol(ctx: &Context, root: crate::value::InstructionId) -> Option<Vec<ValueId>> {
    let root_size = ctx.get_insn(root).size();
    let bits = (root_size * 8) as u64;
    if bits == 0 {
        return None;
    }

    let (lhs, rhs) = as_int_binop(ctx, ValueId::Instruction(root), IntBinop::Or)?;

    // (shl_side, shr_side) — try both orderings of the commutative `or`.
    for (shl_side, shr_side) in [(lhs, rhs), (rhs, lhs)] {
        let Some((x1, c1)) = as_int_binop(ctx, shl_side, IntBinop::ShiftLeft) else {
            continue;
        };
        let Some((x2, c2)) = as_int_binop(ctx, shr_side, IntBinop::ShiftRight) else {
            continue;
        };
        if x1 != x2 {
            continue;
        }
        let (Some(c1v), Some(c2v)) = (const_u64(ctx, c1), const_u64(ctx, c2)) else {
            continue;
        };
        if c1v == 0 || c2v == 0 || c1v + c2v != bits {
            continue;
        }
        // rol(x, c1): reuse the left-shift amount as the rotate amount.
        return Some(vec![x1, c1]);
    }
    None
}

/// If `v` is defined by a rotate intrinsic, return `(name, x, k)`.
fn as_rotate(ctx: &Context, v: ValueId) -> Option<(&'static str, ValueId, ValueId)> {
    let ValueId::Instruction(id) = v else {
        return None;
    };
    let Mnemonic::Intrinsic(intr) = ctx.get_insn(id).mnemonic() else {
        return None;
    };
    let name = intr.id.name();
    if name != "rol" && name != "ror" {
        return None;
    }
    let &[x, k] = intr.args.as_slice() else {
        return None;
    };
    Some((name, x, k))
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
    ctx: &mut Context,
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
    if let Some((inner_name, a, k2)) = as_rotate(ctx, x) {
        let outer_name = id.name();
        let is_inverse = (outer_name == "rol" && inner_name == "ror")
            || (outer_name == "ror" && inner_name == "rol");
        let same_amount = k == k2
            || match (const_u64(ctx, k), const_u64(ctx, k2)) {
                (Some(a), Some(b)) => a % bits == b % bits,
                _ => false,
            };
        if is_inverse && same_amount {
            return Some(Simplified::Value(a));
        }
    }

    // Modulo-width normalisation of a constant amount.
    if let Some(c) = const_u64(ctx, k) {
        let r = c % bits;
        if r == 0 {
            // A whole number of turns: the rotate is the identity.
            return Some(Simplified::Value(x));
        }
        if r != c {
            // Rebuild the same rotate on the reduced amount. The amount keeps
            // the operand's width.
            let k_size = ctx.type_of(k);
            let k_size = ctx.types.size_of(k_size);
            let reduced = ctx.get_const(r, k_size).id();
            let rotate = Intrinsic {
                id,
                args: vec![x, reduced],
            };
            return Some(Simplified::Expression(Mnemonic::Intrinsic(rotate)));
        }
    }

    None
}

register_intrinsic! {
    name: "rol",
    arity: 2,
    result_size: |sz| sz[0],
    eval: eval_rol,
    root_op: Some(RootOp::IntBinop(IntBinop::Or)),
    recognize: Some(recognize_rol),
    simplify: Some(simplify_rotate),
}

register_intrinsic! {
    name: "ror",
    arity: 2,
    result_size: |sz| sz[0],
    eval: eval_ror,
    root_op: None,
    recognize: None,
    simplify: Some(simplify_rotate),
}

#[cfg(test)]
mod tests {
    use crate::value::insn::{IntBinop, IntrinsicId, RootOp, recognizers_for};

    #[test]
    fn rol_ror_registered_and_resolve() {
        let rol = IntrinsicId::from_name("rol").expect("rol registered");
        let ror = IntrinsicId::from_name("ror").expect("ror registered");
        assert_eq!(rol.name(), "rol");
        assert_eq!(ror.name(), "ror");
        assert_eq!(rol.desc().arity, 2);
        assert!(IntrinsicId::from_name("nope").is_none());
    }

    #[test]
    fn eval_rol_matches_native() {
        let rol = IntrinsicId::from_name("rol").unwrap();
        // rol(0x12345678, 8) over 4 bytes == u32 rotate_left
        let got = (rol.desc().eval)(&[(0x1234_5678, 4), (8, 4)], 4).unwrap();
        assert_eq!(got as u32, 0x1234_5678u32.rotate_left(8));
    }

    #[test]
    fn eval_ror_matches_native() {
        let ror = IntrinsicId::from_name("ror").unwrap();
        let got = (ror.desc().eval)(&[(0x1234_5678, 4), (12, 4)], 4).unwrap();
        assert_eq!(got as u32, 0x1234_5678u32.rotate_right(12));
    }

    #[test]
    fn rol_zero_is_identity_eval() {
        let rol = IntrinsicId::from_name("rol").unwrap();
        let got = (rol.desc().eval)(&[(0xdead_beef, 4), (0, 4)], 4).unwrap();
        assert_eq!(got as u32, 0xdead_beef);
    }

    #[test]
    fn recognizers_indexed_by_root() {
        let ids = recognizers_for(RootOp::IntBinop(IntBinop::Or));
        assert!(ids.iter().any(|id| id.name() == "rol"));
    }
}
