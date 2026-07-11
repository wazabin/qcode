//! The `iota` range intrinsic: `i64 -> [i64]` — the driver array `[0, 1, …, n-1]`.
//!
//! `iota(n)` is the index/driver sequence a `scanl`/`map` ranges over when the
//! carried computation depends on the loop counter rather than on stored data
//! (the MT19937 seeding loop `mt[i] = f(mt[i-1], i)` scans over `iota(624)`). Its
//! length lives in its *operand value* `n`, not in a type, so:
//!
//! * its [`result_type`](Intrinsic::result_type) is the length-erased unbounded
//!   list `[i64;*]` — the type manager cannot see `n`'s value;
//! * when `n` is a constant it **folds to a `Bytes` literal** `[i64; n]` holding
//!   `0, 1, …, n-1` (each an 8-byte little-endian word), feeding the existing
//!   `emulate_map` constant-projection path exactly like any other constant
//!   array. Any start offset is baked into the scanned body, so `iota` is unary.

use crate::register_intrinsic;
use crate::types::{TypeId, TypeManager};
use crate::value::ValueId;
use crate::value::insn::{Intrinsic, IntrinsicId, Simplified};
use crate::value::util::base_ref::HostRef;

/// `iota` — the index driver array `[0, 1, …, n-1]` of `i64` elements.
struct Iota;

impl Intrinsic for Iota {
    fn name(&self) -> &'static str {
        "iota"
    }

    fn arity(&self) -> usize {
        1
    }

    fn result_type(&self, types: &TypeManager, _args: &[TypeId]) -> TypeId {
        // The element count is `n`'s *value*, invisible to the type layer, so the
        // pre-fold type is the length-erased `[i64;*]`. A constant `n` recovers a
        // fixed `[i64; n]` in `simplify` (below).
        let i64_ty = types.get_or_make_int(8);
        types.get_or_make_unbounded_list(i64_ty)
    }

    fn eval(&self, _args: &[(u128, usize)], _out_size: usize) -> Option<u128> {
        // Produces an array, not a scalar — whole-array materialization happens in
        // `simplify` (like `enumerate`/`map`), never through scalar `eval`.
        None
    }

    fn simplify(
        &self,
        host: HostRef,
        _id: IntrinsicId,
        _out_size: usize,
        args: &[ValueId],
    ) -> Option<Simplified> {
        let &[n_val] = args else {
            return None;
        };
        // Only a constant length folds; a symbolic `n` stays an `iota` (its length
        // is recovered symbolically by `len(iota n) = n`).
        let ValueId::Literal(lid) = n_val else {
            return None;
        };
        let ctx = host.shared();
        let n = ctx.shared.values.literals[lid].value as usize;

        let i64_ty = ctx.shared.types.get_or_make_int(8);
        let arr_ty = ctx.shared.types.get_or_make_array(i64_ty, n);

        // `n == 0` is a degenerate empty array; `n == 1` (8 bytes) still fits a
        // numeric literal. Everything wider becomes a `Bytes` blob typed `[i64;n]`.
        let mut data = Vec::with_capacity(n * 8);
        for i in 0..n as u64 {
            data.extend_from_slice(&i.to_le_bytes());
        }
        if data.len() <= 8 {
            let value = {
                let mut buf = [0u8; 8];
                buf[..data.len()].copy_from_slice(&data);
                u64::from_le_bytes(buf)
            };
            return Some(Simplified::Value(ctx.get_typed_const(value, arr_ty).id()));
        }
        let bid = ctx.get_typed_bytes(data, arr_ty).id();
        Some(Simplified::Value(bid))
    }
}

register_intrinsic!(Iota);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::value::insn::IntrinsicId;

    #[test]
    fn iota_registered_and_resolves() {
        let id = IntrinsicId::from_name("iota").expect("iota registered");
        assert_eq!(id.name(), "iota");
        assert_eq!(id.desc().arity(), 1);
    }

    #[test]
    fn result_type_is_unbounded_i64_list() {
        let mut types = TypeManager::default();
        let i64_ty = types.get_or_make_int(8);
        let id = IntrinsicId::from_name("iota").unwrap();
        let ty = id.desc().result_type(&types, &[i64_ty]);
        assert_eq!(types.list_of(ty), Some((i64_ty, None)));
    }

    /// A constant `n` folds `iota(n)` to a `Bytes` array `[i64; n]` = `0,1,…,n-1`.
    #[test]
    fn iota_of_const_folds_to_bytes_array() {
        let ctx = Context::new();
        let n = ctx.get_const(3, 8).id();
        let id = IntrinsicId::from_name("iota").unwrap();
        let Some(Simplified::Value(ValueId::Bytes(bid))) =
            id.desc().simplify((&ctx).into(), id, 24, &[n])
        else {
            panic!("iota(3) should fold to a Bytes array");
        };
        let bytes = &ctx.shared.values.bytes[bid];
        assert_eq!(bytes.data.len(), 24);
        assert_eq!(&bytes.data[0..8], &0u64.to_le_bytes());
        assert_eq!(&bytes.data[8..16], &1u64.to_le_bytes());
        assert_eq!(&bytes.data[16..24], &2u64.to_le_bytes());
        let i64_ty = ctx.shared.types.get_or_make_int(8);
        assert_eq!(ctx.shared.types.array_of(bytes.type_id), Some((i64_ty, 3)));
    }

    /// A symbolic `n` does not fold.
    #[test]
    fn iota_of_symbolic_does_not_fold() {
        let mut ctx = Context::new();
        let blk = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let p = crate::value::BasicBlock::from_id_mut(&mut ctx, blk)
            .push_param(8)
            .id;
        let id = IntrinsicId::from_name("iota").unwrap();
        assert!(
            id.desc()
                .simplify((&ctx).into(), id, 8, &[ValueId::BlockParam(p)])
                .is_none()
        );
    }
}
