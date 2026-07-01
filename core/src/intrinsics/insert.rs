//! The `insert` intrinsic: `[T; N], i64, T -> [T; N]` — a pure functional array
//! update, the dual of [`Extract`](crate::value::insn::Extract)/`at`.
//!
//! `insert(arr, i, v)` produces a new array equal to `arr` with lane `i` replaced
//! by `v`. It is the SSA form of a loop's per-iteration store once the buffer has
//! been promoted to a carried array value (see the `array_promote` pass): the
//! whole-array `store(ram, base <- arr)` is emitted once at loop exit, and each
//! `store base[i] <- v` inside the loop becomes `arr' = insert(arr, i, v)`.
//!
//! Being an [`Intrinsic`] it is **categorically pure** — no memory effect — so
//! DCE/GVN/alias treat it correctly with no special classification. Its dual
//! reader `at(arr, i)` forwards through it: `at(insert(a, i, v), i) = v`.

use crate::context::Context;
use crate::register_intrinsic;
use crate::types::{TypeId, TypeManager};
use crate::value::ValueId;
use crate::value::insn::{Intrinsic, IntrinsicId, Simplified};

/// `insert` — functional single-lane array update.
struct Insert;

impl Intrinsic for Insert {
    fn name(&self) -> &'static str {
        "insert"
    }

    fn arity(&self) -> usize {
        3
    }

    fn result_type(&self, _types: &mut TypeManager, args: &[TypeId]) -> TypeId {
        // Same shape as the array being updated.
        args[0]
    }

    fn eval(&self, _args: &[(u128, usize)], _out_size: usize) -> Option<u128> {
        // Produces an array; whole-array folding is done in `simplify` over a
        // constant `Bytes` base, never through scalar `eval`.
        None
    }

    fn simplify(
        &self,
        ctx: &mut Context,
        _id: IntrinsicId,
        _out_size: usize,
        args: &[ValueId],
    ) -> Option<Simplified> {
        let &[arr, index, val] = args else {
            return None;
        };
        // Fold `insert(Bytes, const i, const v)` into a patched `Bytes` blob so a
        // chain of pre-loop constant inserts collapses to one constant array.
        let (ValueId::Bytes(bid), ValueId::Literal(ilit), ValueId::Literal(vlit)) =
            (arr, index, val)
        else {
            return None;
        };
        let arr_ty = ctx.values.bytes[bid].type_id;
        let (elem, count) = ctx.types.array_of(arr_ty)?;
        let esz = ctx.types.size_of(elem);
        let i = ctx.values.literals[ilit].value as usize;
        if i >= count {
            return None;
        }
        let v = ctx.values.literals[vlit].value;
        let mut data = ctx.values.bytes[bid].data.clone();
        let off = i * esz;
        let v_bytes = v.to_le_bytes();
        data[off..off + esz].copy_from_slice(&v_bytes[..esz]);
        let nid = ctx.get_bytes(data).id();
        if let ValueId::Bytes(b) = nid {
            ctx.values.bytes[b].type_id = arr_ty;
        }
        Some(Simplified::Value(nid))
    }
}

register_intrinsic!(Insert);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::insn::IntrinsicId;

    #[test]
    fn insert_registered_with_arity_3() {
        let id = IntrinsicId::from_name("insert").expect("insert registered");
        assert_eq!(id.desc().arity(), 3);
    }

    #[test]
    fn result_type_is_the_array_type() {
        let mut types = TypeManager::default();
        let i32 = types.get_or_make_int(4);
        let arr = types.get_or_make_array(i32, 5);
        let id = IntrinsicId::from_name("insert").unwrap();
        assert_eq!(id.desc().result_type(&mut types, &[arr, i32, i32]), arr);
    }

    /// `insert(Bytes[i32;3], 1, 0xaa)` patches lane 1 of the constant blob.
    #[test]
    fn insert_into_const_bytes_folds() {
        let mut ctx = Context::new();
        let i32 = ctx.types.get_or_make_int(4);
        let arr_ty = ctx.types.get_or_make_array(i32, 3);
        let bid = ctx.get_bytes(vec![0; 12]).id();
        if let ValueId::Bytes(b) = bid {
            ctx.values.bytes[b].type_id = arr_ty;
        }
        let i = ctx.get_const(1, 8).id();
        let v = ctx.get_const(0xaa, 4).id();
        let id = IntrinsicId::from_name("insert").unwrap();
        let Some(Simplified::Value(ValueId::Bytes(nb))) =
            id.desc().simplify(&mut ctx, id, 12, &[bid, i, v])
        else {
            panic!("insert into const Bytes should fold");
        };
        assert_eq!(&ctx.values.bytes[nb].data[4..8], &[0xaa, 0, 0, 0]);
    }
}
