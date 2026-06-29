//! The `take_while` array intrinsic: `[T; N] -> [T; N]`.
//!
//! `take_while(X)` is the maximal **prefix** of array `X` whose elements are all
//! nonzero — i.e. it truncates `X` at the first zero element. This is the
//! C-string / NUL-terminator view (`take_while(≠ 0)`), the shape that lets a
//! NUL-bounded copy loop (`while (*src) *dst++ = *src++;`) be lifted to a
//! `map(body, take_while(src))` instead of a counted total map.
//!
//! Like [`enumerate`](super::enumerate), it is:
//! * **never recognized** from machine code — only introduced deliberately by a
//!   recognizer (via [`Builder::push_intrinsic`]);
//! * **array-producing**, so it has no scalar [`eval`](Intrinsic::eval); and
//! * **projectable in principle** (`take_while(X)[i] = X[i]` for `i` below the
//!   first zero), though its length is *data-dependent* rather than static.
//!
//! ## The predicate is fixed to "nonzero"
//!
//! The intrinsic framework carries only value operands, not a per-element body
//! function (that is what [`Map`](crate::value::insn::Map) is for). A fully
//! general `take_while(pred, X)` would therefore need a body-carrying mnemonic,
//! not an intrinsic. The NUL predicate is the only one a copy-until-terminator
//! loop needs, so v1 bakes it in.
//!
//! ## Finite array → list
//!
//! `take_while` is the point where a finite `[T; N]` array becomes a
//! variable-length **list**. Its result is therefore [`List<T>`] (see
//! [`TypeManager::get_or_make_list`]) with `bound = N`: a sequence of *at most*
//! `N` elements whose actual length is the position of the first zero. The
//! `bound` is the static storage upper bound; the length is data-dependent.
//!
//! [`List<T>`]: crate::types::TypeRepr::List
//! [`TypeManager::get_or_make_list`]: crate::types::TypeManager::get_or_make_list
//!
//! [`Builder::push_intrinsic`]: crate::builder::Builder::push_intrinsic

use crate::register_intrinsic;
use crate::types::{TypeId, TypeManager};
use crate::value::insn::Intrinsic;

/// `take_while` — the nonzero (NUL-terminated) prefix of an array.
struct TakeWhile;

impl Intrinsic for TakeWhile {
    fn name(&self) -> &'static str {
        "take_while"
    }

    fn arity(&self) -> usize {
        1
    }

    fn result_type(&self, types: &mut TypeManager, args: &[TypeId]) -> TypeId {
        // A *sequence* source of `T` becomes a `List<T>` with the source's bound: an
        // array `[T; N]` makes the finite-array → list transition (bound `N`), a list
        // keeps its bound (a prefix is no longer). A *pointer* source — a `char*`
        // string of unknown length with no snapshot — becomes an **unbounded**
        // `List<i8>`: a NUL-terminated byte string whose length is fully
        // data-dependent. Either way the result kind is a list (the length is
        // data-dependent); only the static bound differs.
        if let Some((elem, len, _is_list)) = types.seq_of(args[0]) {
            types.get_or_make_list(elem, len)
        } else {
            let i8 = types.get_or_make_int(1);
            types.get_or_make_unbounded_list(i8)
        }
    }

    fn eval(&self, _args: &[(u128, usize)], _out_size: usize) -> Option<u128> {
        // Array-producing, like `enumerate`/`map`: no whole-array interpreter.
        // `None` is the "not foldable / trap" signal — folding skips it and the
        // emulator raises a recoverable `UnsupportedIntrinsic`.
        None
    }
}

register_intrinsic!(TakeWhile);

#[cfg(test)]
mod tests {
    use crate::value::insn::IntrinsicId;

    #[test]
    fn take_while_registered_and_resolves() {
        let id = IntrinsicId::from_name("take_while").expect("take_while registered");
        assert_eq!(id.name(), "take_while");
        assert_eq!(id.desc().arity(), 1);
    }

    #[test]
    fn result_type_is_a_list_with_the_source_bound() {
        let mut types = crate::types::TypeManager::default();
        let i8 = types.get_or_make_int(1);
        let arr = types.get_or_make_array(i8, 7);

        let id = IntrinsicId::from_name("take_while").unwrap();
        let result = id.desc().result_type(&mut types, &[arr]);

        // `[i8; 7]` in → `List<i8>` bound 7: a list, NOT a fixed array.
        assert_eq!(types.array_of(result), None, "result is not a fixed array");
        let (elem, bound) = types.list_of(result).expect("result is a list");
        assert_eq!(elem, i8);
        assert_eq!(bound, Some(7));
        // The storage footprint upper bound matches the source array.
        assert_eq!(types.size_of(result), 7);
    }

    /// A *pointer* source (a non-sequence operand: a `char*` string of unknown
    /// length) yields an **unbounded** `List<i8>` with no static footprint.
    #[test]
    fn result_type_of_a_pointer_is_an_unbounded_list() {
        let mut types = crate::types::TypeManager::default();
        let i8 = types.get_or_make_int(1);
        let ptr = types.get_or_make_int(8); // a raw pointer, not a sequence

        let id = IntrinsicId::from_name("take_while").unwrap();
        let result = id.desc().result_type(&mut types, &[ptr]);

        assert_eq!(types.array_of(result), None, "result is not a fixed array");
        let (elem, bound) = types.list_of(result).expect("result is a list");
        assert_eq!((elem, bound), (i8, None), "an unbounded List<i8>");
        assert_eq!(types.size_of(result), 0, "no materialized footprint");
    }

    /// `take_while` composes onto a list (e.g. another `take_while`/`map` result):
    /// `take_while(List<T>) = List<T>`, same bound.
    #[test]
    fn take_while_accepts_a_list() {
        let mut types = crate::types::TypeManager::default();
        let i8 = types.get_or_make_int(1);
        let list = types.get_or_make_list(i8, 5);

        let id = IntrinsicId::from_name("take_while").unwrap();
        let result = id.desc().result_type(&mut types, &[list]);

        let (elem, bound) = types.list_of(result).expect("result is a list");
        assert_eq!((elem, bound), (i8, Some(5)));
    }

    #[test]
    fn take_while_is_not_foldable() {
        let id = IntrinsicId::from_name("take_while").unwrap();
        assert_eq!(id.desc().eval(&[(0x4142, 4)], 4), None);
    }
}
