//! The `singleton` intrinsic: `T -> [T; 1]` — the one-element array `[x]`.
//!
//! `singleton(x)` lifts a scalar into a length-one array so it can be `concat`ed
//! with a sequence — the functional form of a loop's pre-header prefix element
//! (`mt[0] = seed` becomes `concat(singleton(seed), scanl …)`). Structurally a
//! `[T; 1]` array has the same bit pattern as `x`, so a constant `x` folds
//! straight through [`eval`](Intrinsic::eval).

use crate::register_intrinsic;
use crate::types::{TypeId, TypeManager};
use crate::value::insn::Intrinsic;

/// `singleton` — wrap a scalar as a one-element array.
struct Singleton;

impl Intrinsic for Singleton {
    fn name(&self) -> &'static str {
        "singleton"
    }

    fn arity(&self) -> usize {
        1
    }

    fn result_type(&self, types: &TypeManager, args: &[TypeId]) -> TypeId {
        // `[T; 1]` where `T` is the operand's own type.
        types.get_or_make_array(args[0], 1)
    }

    fn eval(&self, args: &[(u128, usize)], _out_size: usize) -> Option<u128> {
        // A `[T; 1]` is bit-identical to its single element.
        args.first().map(|&(bits, _)| bits)
    }
}

register_intrinsic!(Singleton);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::insn::IntrinsicId;

    #[test]
    fn singleton_registered_and_resolves() {
        let id = IntrinsicId::from_name("singleton").expect("singleton registered");
        assert_eq!(id.name(), "singleton");
        assert_eq!(id.desc().arity(), 1);
    }

    #[test]
    fn result_type_is_one_element_array() {
        let types = TypeManager::default();
        let i32 = types.get_or_make_int(4);
        let id = IntrinsicId::from_name("singleton").unwrap();
        let ty = id.desc().result_type(&types, &[i32]);
        assert_eq!(types.array_of(ty), Some((i32, 1)));
    }

    #[test]
    fn eval_passes_bits_through() {
        let id = IntrinsicId::from_name("singleton").unwrap();
        assert_eq!(id.desc().eval(&[(0xdead, 4)], 4), Some(0xdead));
    }
}
