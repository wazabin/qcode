//! The `len` sequence intrinsic: `Seq<T> -> i64` — the number of elements.
//!
//! `len(X)` is the length of a sequence, and it is **sequence-polymorphic** like
//! `map`/`enumerate`/`take_while`:
//!
//! * on a fixed `[T; N]` array — or a `map`/`enumerate` over one, which preserves
//!   length — it is the static constant `N` and **folds to a literal** (via
//!   [`simplify`](Intrinsic::simplify), since the length lives in the operand
//!   *type*, not its bits);
//! * on a `List<T>` (a [`take_while`](super::take_while) result) it is the
//!   *data-dependent* runtime length — the position of the first element that
//!   failed the predicate — so it stays symbolic.
//!
//! This is the missing half of `strlen`: `strlen(s) = len(take_while(s))`.

use crate::register_intrinsic;
use crate::types::{TypeId, TypeManager};
use crate::value::ValueId;
use crate::value::insn::{Intrinsic, IntrinsicId, Mnemonic, Simplified};
use crate::value::util::base_ref::HostRef;

/// `len` — the element count of a sequence.
struct Len;

impl Intrinsic for Len {
    fn name(&self) -> &'static str {
        "len"
    }

    fn arity(&self) -> usize {
        1
    }

    fn result_type(&self, types: &TypeManager, _args: &[TypeId]) -> TypeId {
        // A count is a plain machine word, regardless of the element type.
        types.get_or_make_int(8)
    }

    fn eval(&self, _args: &[(u128, usize)], _out_size: usize) -> Option<u128> {
        // The length comes from the operand *type* (array count / list bound),
        // not its value bits, so scalar `eval` cannot compute it. The foldable
        // (array) case is handled in `simplify`.
        None
    }

    fn simplify(
        &self,
        host: HostRef,
        _id: IntrinsicId,
        out_size: usize,
        args: &[ValueId],
    ) -> Option<Simplified> {
        let &[seq] = args else {
            return None;
        };
        // A fixed array has a statically known length → fold to that constant.
        // A list's length is data-dependent (the NUL position for a string) and
        // stays symbolic.
        let ty = host.type_of(seq);
        if let Some((_, n)) = host.shared().types.array_of(ty) {
            let lit = host.shared().get_const(n as u64, out_size).id();
            return Some(Simplified::Value(lit));
        }
        // `len(iota n) = n`: the length of an as-yet-unfolded index driver is its
        // own operand, recovered symbolically even when `n` is not constant. (A
        // constant `iota` would already have folded to a fixed array above.)
        if let ValueId::Instruction(iid) = seq
            && let Mnemonic::Intrinsic(app) = host.instruction(iid).mnemonic()
            && app.id.name() == "iota"
        {
            return Some(Simplified::Value(app.args[0]));
        }
        None
    }
}

register_intrinsic!(Len);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::value::insn::IntrinsicId;
    use crate::value::{BasicBlock, ValueId};

    #[test]
    fn len_registered_and_resolves() {
        let id = IntrinsicId::from_name("len").expect("len registered");
        assert_eq!(id.name(), "len");
        assert_eq!(id.desc().arity(), 1);
    }

    #[test]
    fn result_type_is_a_machine_word() {
        let mut types = TypeManager::default();
        let i8 = types.get_or_make_int(1);
        let arr = types.get_or_make_array(i8, 4);
        let id = IntrinsicId::from_name("len").unwrap();
        assert_eq!(
            id.desc().result_type(&types, &[arr]),
            types.get_or_make_int(8)
        );
    }

    /// `len` of a fixed array folds to the constant element count; `len` of a
    /// list does not fold (its length is data-dependent).
    #[test]
    fn len_of_array_folds_list_does_not() {
        let mut ctx = Context::new();
        let i8 = ctx.types.get_or_make_int(1);
        let arr = ctx.types.get_or_make_array(i8, 6);
        let list = ctx.types.get_or_make_list(i8, 6);

        let blk = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let ap = BasicBlock::from_id_mut(&mut ctx, blk).push_param(6).id;
        ctx.values.block_param_mut(ap).type_id = arr;
        let lp = BasicBlock::from_id_mut(&mut ctx, blk).push_param(6).id;
        ctx.values.block_param_mut(lp).type_id = list;

        let id = IntrinsicId::from_name("len").unwrap();

        // Array → folds to the literal 6.
        match id
            .desc()
            .simplify((&ctx).into(), id, 8, &[ValueId::BlockParam(ap)])
        {
            Some(Simplified::Value(ValueId::Literal(lid))) => {
                assert_eq!(ctx.values.literals[lid].value, 6);
            }
            other => panic!("len of an array must fold to the literal 6, got {other:?}"),
        }

        // List → stays symbolic.
        assert!(
            id.desc()
                .simplify((&ctx).into(), id, 8, &[ValueId::BlockParam(lp)])
                .is_none(),
            "len of a list must not fold (data-dependent length)"
        );
    }
}
