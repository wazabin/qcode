//! The `splat` intrinsic: `splat(x, n) -> [T; n]` — the constant array whose `n`
//! lanes are all `x`.
//!
//! `array_promote` needs a base array to seed its `insert` chain (`insert(base,
//! 0, seed)` then the per-lane inserts). A fully-filled region overwrites every
//! lane, so that base is a pure placeholder — conceptually a zero-filled array.
//! Materializing it as a `Bytes` literal is fine when small, but a 624-element
//! MT state becomes a ~2.5 KB blob that swamps any textual dump. So `splat` keeps
//! the constant array *symbolic*: [`simplify`](Intrinsic::simplify) only collapses
//! it to a `Bytes` literal when the lane count is small (`<= SPLAT_LITERAL_MAX`);
//! larger splats stay `$splat(x, n)`. `at(splat(x, _)) = x` (see `at.rs`) recovers
//! any lane without ever expanding the array.

use crate::register_intrinsic;
use crate::types::{TypeId, TypeManager};
use crate::value::ValueId;
use crate::value::insn::{Intrinsic, IntrinsicId, Simplified};
use crate::value::util::base_ref::HostRef;

/// Lane count at or below which a constant `splat` folds to a `Bytes` literal.
/// Above it the array stays a symbolic `$splat` so dumps remain readable.
pub const SPLAT_LITERAL_MAX: usize = 100;

/// `splat` — the constant array `[x; n]`.
struct Splat;

impl Intrinsic for Splat {
    fn name(&self) -> &'static str {
        "splat"
    }

    fn arity(&self) -> usize {
        2
    }

    fn result_type(&self, types: &TypeManager, args: &[TypeId]) -> TypeId {
        // The lane count is `n`'s *value*, invisible to the type layer, so the
        // pre-fold type is the length-erased `[T;*]`. A constant fold recovers a
        // fixed `[T; n]` (either the `Bytes` literal below or the explicit type the
        // creator attached).
        types.get_or_make_unbounded_list(args[0])
    }

    fn eval(&self, _args: &[(u128, usize)], _out_size: usize) -> Option<u128> {
        // Produces an array, not a scalar — materialization happens in `simplify`.
        None
    }

    fn simplify(
        &self,
        host: HostRef,
        _id: IntrinsicId,
        _out_size: usize,
        args: &[ValueId],
    ) -> Option<Simplified> {
        let &[val, count] = args else {
            return None;
        };
        // Only a constant lane value and count fold; anything symbolic stays a
        // `splat`. Large counts deliberately stay symbolic to keep dumps small.
        let (ValueId::Literal(vlid), ValueId::Literal(clid)) = (val, count) else {
            return None;
        };
        let ctx = host.shared();
        let n = ctx.values.literals[clid].value as usize;
        if n == 0 || n > SPLAT_LITERAL_MAX {
            return None;
        }
        let bits = ctx.values.literals[vlid].value;
        let elem_type_id = ctx.values.literals[vlid].type_id;
        let esz = ctx.types.size_of(elem_type_id);
        let elem_ty = ctx.types.get_or_make_int(esz);
        let arr_ty = ctx.types.get_or_make_array(elem_ty, n);
        let mut data = Vec::with_capacity(n * esz);
        for _ in 0..n {
            data.extend_from_slice(&bits.to_le_bytes()[..esz]);
        }
        let bid = ctx.get_typed_bytes(data, arr_ty).id();
        Some(Simplified::Value(bid))
    }
}

register_intrinsic!(Splat);

#[cfg(test)]
mod tests {
    use crate::value::insn::IntrinsicId;

    #[test]
    fn splat_registered() {
        let id = IntrinsicId::from_name("splat").expect("splat registered");
        assert_eq!(id.name(), "splat");
        assert_eq!(id.desc().arity(), 2);
    }
}
