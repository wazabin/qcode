//! First-class **poison** values (argpromote v2, `ARGPROMOTE_REGISTERS_V2.md`).
//!
//! A poison value is a typed placeholder for a datum whose concrete bits are
//! undefined — the clobber slots of a register/external return pack, and the
//! symbolic arguments of a pure-call emulation. Semantics:
//!
//! * **GVN never folds it.** Every poison is a distinct interned value (they are
//!   never deduped), so two poisons of the same type are *not* congruent and a
//!   poison is never congruent with a concrete value. See the congruence rule in
//!   the GVN CSE sub-pass.
//! * **DCE / `dead_signature` treat it as an ordinary pure value** — a
//!   `store(R, poison)` dies iff `R` is unread; a poison-only pack slot prunes
//!   through the normal unused-slot path. No special casing anywhere.
//! * **Reading poison in the emulator is a hard error.** Propagating it as an
//!   operand is fine; the trap fires only when its concrete value is demanded.
//!
//! Poison is engine-internal: it is minted mid-pipeline and is recomputable, so
//! it is deliberately not rendered into textual qcode.

use crate::{
    context::Shared,
    types::TypeId,
    value::{
        Value, ValueId,
        util::base_ref::{BaseRef, WithShared},
    },
};
use jstd::Identifier;

#[derive(Identifier)]
pub struct PoisonId(usize);

/// A typed poison value stored in a [`Context`](crate::context::Context). Carries
/// only its [`TypeId`] (hence its width); its bits are undefined.
#[derive(Debug, Clone, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Poison {
    /// The type (and thus byte width) of this poison value.
    pub type_id: TypeId,
}

pub type PoisonRef<'str, 'ctx> = BaseRef<&'ctx Shared<'str>, PoisonId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithShared<'s, 'ctx, 'str> for PoisonRef<'str, 'ctx> {
    fn shared(&'s self) -> &'ctx Shared<'str> {
        self.ctx
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx, Ctx> BaseRef<Ctx, PoisonId>
where
    Self: WithShared<'s, 'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx Poison {
        &self.shared().values.poisons[self.id]
    }

    /// Returns the [`TypeId`] of this poison value.
    pub fn type_id(&'s self) -> TypeId {
        self.inner().type_id
    }
}

impl std::fmt::Display for PoisonRef<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "poison")
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for PoisonRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        ValueId::Poison(self.id)
    }

    fn size(&self) -> usize {
        self.ctx
            .types
            .size_of(self.ctx.values.poisons[self.id].type_id)
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::TestContext;
    use crate::value::{Value, ValueId, ValueRef};

    /// A poison value interns, carries its type/width, and round-trips through a
    /// `ValueRef`.
    #[test]
    fn poison_interns_and_round_trips() {
        let tc = TestContext::new();
        let i32_ty = tc.ctx.shared.types.get_or_make_int(4);
        let p = tc.ctx.get_poison(i32_ty);
        assert!(p.is_poison());
        assert_eq!(tc.ctx.type_of(p), i32_ty);
        assert_eq!(tc.ctx.stored_type_of(p), Some(i32_ty));
        match ValueRef::new(p, &tc.ctx) {
            ValueRef::Poison(r) => {
                assert_eq!(r.type_id(), i32_ty);
                assert_eq!(r.size(), 4);
            }
            _ => panic!("expected a poison ref"),
        }
    }

    /// Two poisons of the *same* type are distinct values (never deduped), so
    /// GVN keeps them in separate congruence classes.
    #[test]
    fn same_type_poisons_are_distinct() {
        let tc = TestContext::new();
        let i64_ty = tc.ctx.shared.types.get_or_make_int(8);
        let a = tc.ctx.get_poison(i64_ty);
        let b = tc.ctx.get_poison(i64_ty);
        assert_ne!(a, b, "each poison must be its own interned value");
        assert!(matches!(a, ValueId::Poison(_)));
        assert!(matches!(b, ValueId::Poison(_)));
    }
}
