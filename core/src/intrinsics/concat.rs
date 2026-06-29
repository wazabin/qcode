//! The `concat` sequence intrinsic: `Seq<T>, Seq<T> -> Seq<T>`.
//!
//! `concat(a, b)` appends two homogeneous sequences. For fixed arrays it produces
//! a fixed array with summed length; if either side is a list, the result is a
//! list with the summed bound when both bounds are known, or an unbounded list
//! otherwise.

use crate::register_intrinsic;
use crate::types::{TypeId, TypeManager};
use crate::value::insn::Intrinsic;

struct Concat;

fn seq_or_list(types: &TypeManager, id: TypeId) -> Option<(TypeId, Option<usize>, bool)> {
    if let Some((elem, n)) = types.array_of(id) {
        return Some((elem, Some(n), false));
    }
    types.list_of(id).map(|(elem, bound)| (elem, bound, true))
}

impl Intrinsic for Concat {
    fn name(&self) -> &'static str {
        "concat"
    }

    fn arity(&self) -> usize {
        2
    }

    fn result_type(&self, types: &mut TypeManager, args: &[TypeId]) -> TypeId {
        let (a_elem, a_len, a_list) =
            seq_or_list(types, args[0]).expect("concat lhs must be a sequence");
        let (b_elem, b_len, b_list) =
            seq_or_list(types, args[1]).expect("concat rhs must be a sequence");
        assert_eq!(
            a_elem, b_elem,
            "concat operands must have the same element type"
        );

        match (a_len, b_len, a_list || b_list) {
            (Some(a), Some(b), false) => types.get_or_make_array(a_elem, a + b),
            (Some(a), Some(b), true) => types.get_or_make_list(a_elem, a + b),
            _ => types.get_or_make_unbounded_list(a_elem),
        }
    }

    fn eval(&self, _args: &[(u128, usize)], _out_size: usize) -> Option<u128> {
        None
    }
}

register_intrinsic!(Concat);

#[cfg(test)]
mod tests {
    use crate::types::TypeManager;
    use crate::value::insn::IntrinsicId;

    #[test]
    fn concat_registered_and_resolves() {
        let id = IntrinsicId::from_name("concat").expect("concat registered");
        assert_eq!(id.name(), "concat");
        assert_eq!(id.desc().arity(), 2);
    }

    #[test]
    fn concat_arrays_yields_larger_array() {
        let mut types = TypeManager::default();
        let i32 = types.get_or_make_int(4);
        let a = types.get_or_make_array(i32, 3);
        let b = types.get_or_make_array(i32, 5);

        let id = IntrinsicId::from_name("concat").unwrap();
        let result = id.desc().result_type(&mut types, &[a, b]);

        assert_eq!(types.array_of(result), Some((i32, 8)));
        assert_eq!(types.size_of(result), 32);
    }

    #[test]
    fn concat_with_list_yields_list() {
        let mut types = TypeManager::default();
        let i8 = types.get_or_make_int(1);
        let a = types.get_or_make_array(i8, 3);
        let b = types.get_or_make_list(i8, 5);

        let id = IntrinsicId::from_name("concat").unwrap();
        let result = id.desc().result_type(&mut types, &[a, b]);

        assert_eq!(types.list_of(result), Some((i8, Some(8))));
        assert_eq!(types.array_of(result), None);
    }

    #[test]
    fn concat_unbounded_list_yields_unbounded_list() {
        let mut types = TypeManager::default();
        let i8 = types.get_or_make_int(1);
        let a = types.get_or_make_unbounded_list(i8);
        let b = types.get_or_make_array(i8, 5);

        let id = IntrinsicId::from_name("concat").unwrap();
        let result = id.desc().result_type(&mut types, &[a, b]);

        assert_eq!(types.list_of(result), Some((i8, None)));
    }
}
