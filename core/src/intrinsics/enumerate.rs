//! The `enumerate` array intrinsic: `[T; N] -> [(index: i64, elem: T); N]`.
//!
//! `enumerate(X)` pairs each element of an array with its index. Unlike the
//! rotate intrinsics it is **never recognized** from machine code — it is only
//! ever introduced deliberately (via [`Builder::push_intrinsic`]) — and its
//! result is an *array of tuples*, not a scalar. It is **projectable**
//! (`enumerate(X)[i] = (i, X[i])`, handled by the array-projection sub-pass) and
//! survives to the GUI as a returned array value.
//!
//! [`Builder::push_intrinsic`]: crate::builder::Builder::push_intrinsic

use crate::register_intrinsic;
use crate::types::{AggregateField, TypeId, TypeManager};
use crate::value::insn::Intrinsic;

/// `enumerate` — pair each array element with its `i64` index.
struct Enumerate;

impl Intrinsic for Enumerate {
    fn name(&self) -> &'static str {
        "enumerate"
    }

    fn arity(&self) -> usize {
        1
    }

    fn result_type(&self, types: &mut TypeManager, args: &[TypeId]) -> TypeId {
        // `[(index: i64, elem: T); N]` from the operand array `[T; N]`.
        let (elem, count) = types
            .array_of(args[0])
            .expect("enumerate operand must be an array");
        let i64_ty = types.get_or_make_int(8);
        let tuple = types.get_or_make_named_aggregate(vec![
            AggregateField::new("index", i64_ty),
            AggregateField::new("elem", elem),
        ]);
        types.get_or_make_array(tuple, count)
    }

    fn eval(&self, _args: &[(u128, usize)], _out_size: usize) -> Option<u128> {
        // `enumerate` produces an array; whole-array evaluation is deferred (the
        // same way `map` has no interpreter). It is only ever consumed by
        // projection, which never routes through `eval`. `None` is the "not
        // foldable / trap" signal: constant folding skips it and the emulator
        // raises a recoverable `UnsupportedIntrinsic` (so pure-call emulation of a
        // function returning `enumerate` declines to harvest rather than crashing).
        None
    }
}

register_intrinsic!(Enumerate);

#[cfg(test)]
mod tests {
    use crate::value::insn::IntrinsicId;

    #[test]
    fn enumerate_registered_and_resolves() {
        let id = IntrinsicId::from_name("enumerate").expect("enumerate registered");
        assert_eq!(id.name(), "enumerate");
        assert_eq!(id.desc().arity(), 1);
    }

    #[test]
    fn result_type_is_array_of_index_elem_tuples() {
        let mut types = crate::types::TypeManager::default();
        let i8 = types.get_or_make_int(1);
        let arr = types.get_or_make_array(i8, 4);

        let id = IntrinsicId::from_name("enumerate").unwrap();
        let result = id.desc().result_type(&mut types, &[arr]);

        // `[(index: i64, elem: i8); 4]`.
        let (tuple, count) = types.array_of(result).expect("result is an array");
        assert_eq!(count, 4);
        let fields = types
            .aggregate_fields(tuple)
            .expect("element is an aggregate");
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name, "index");
        assert_eq!(types.size_of(fields[0].type_id), 8);
        assert_eq!(fields[1].name, "elem");
        assert_eq!(fields[1].type_id, i8);
    }
}
