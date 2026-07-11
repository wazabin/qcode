//! Aggregate construction and projection.
//!
//! These are the functional-IR counterpart of a tuple: [`Tuple`] groups several
//! values into one aggregate-typed result, and [`Extract`] projects a single
//! field back out. They preserve the one-value-per-instruction invariant (an
//! `Extract` *is* the field value it names), so no multi-result instruction is
//! needed. `argpromote` uses them to return `(real_return, write-set)`.

use crate::{context::Context, value::ValueId};

use super::mnemonic::{Args, MnemonicKind};
use smallvec::{SmallVec, smallvec};

/// Builds an aggregate value from its ordered fields. The instruction's result
/// type is the [`Aggregate`](crate::types::TypeRepr::Aggregate) of the fields'
/// types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Tuple {
    pub fields: Vec<ValueId>,
}

impl MnemonicKind for Tuple {
    fn opcode(&self) -> &'static str {
        "pack"
    }

    fn args(&self) -> Args {
        SmallVec::from_vec(self.fields.clone())
    }
}

/// Projects field `index` out of an aggregate value. The instruction's result
/// type is that field's type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Extract {
    pub agg: ValueId,
    pub index: usize,
}

impl Extract {
    pub fn field_name<'a>(&self, ctx: &'a Context<'_>) -> Option<&'a str> {
        let agg_ty = ctx.stored_type_of(self.agg)?;
        ctx.shared.types.field_name(agg_ty, self.index)
    }
}

impl MnemonicKind for Extract {
    fn opcode(&self) -> &'static str {
        "extract"
    }

    fn args(&self) -> Args {
        smallvec![self.agg]
    }
}

/// Computes the address of a struct field: `gep(base, offset)` ≡
/// `base + offset`, but the result is *typed* `PtrTo<field.type>` and prints by
/// field **name** instead of the raw offset.
///
/// Unlike [`Extract`] — which projects a field *value* out of an in-register
/// aggregate — `Gep` does **no memory access**: it is pure pointer arithmetic.
/// The field value is obtained by a separate `load` of the `Gep` result. The
/// field name is recovered from the pointee of `base`'s
/// [`StructPointer`](crate::types::TypeRepr::StructPointer) type, keyed by the
/// byte `offset`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Gep {
    pub base: ValueId,
    pub offset: usize,
}

impl Gep {
    /// The name of the field this `Gep` addresses, recovered from the nominal
    /// struct that `base` points at. `None` if `base` is not a typed struct
    /// pointer or the offset matches no field.
    pub fn field_name<'a>(&self, ctx: &'a Context<'_>) -> Option<&'a str> {
        let base_ty = ctx.stored_type_of(self.base)?;
        let pointee = ctx.shared.types.pointee_of(base_ty)?;
        ctx.shared
            .types
            .field_by_offset(pointee, self.offset)
            .map(|(_, field)| field.name.as_str())
    }
}

impl MnemonicKind for Gep {
    fn opcode(&self) -> &'static str {
        "gep"
    }

    fn args(&self) -> Args {
        smallvec![self.base]
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use crate::{
        context::Context,
        value::{BasicBlock, ValueId, insn::Mnemonic},
    };

    #[test]
    fn tuple_and_extract_roundtrip() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                %a = i32 5 + i32 0;
                %b = i64 7 + i64 0;
                %t = pack(lhs=%a, rhs=%b);
                %x = extract(%t.rhs);
                return at i64 0;
            "
        );

        let insns: Vec<Mnemonic> = BasicBlock::from_id(&ctx, block)
            .iter()
            .map(|i| i.mnemonic().clone())
            .collect();

        // The tuple groups the two loaded values, in order.
        let tuple = insns
            .iter()
            .find_map(|m| match m {
                Mnemonic::Tuple(t) => Some(t.clone()),
                _ => None,
            })
            .expect("tuple instruction");
        assert_eq!(tuple.fields.len(), 2);
        assert_eq!(
            BasicBlock::from_id(&ctx, block)
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Tuple(_)))
                .unwrap()
                .as_statement()
                .to_string(),
            "i96 %t = pack(lhs=i32 %a, rhs=i64 %b);"
        );

        // The tuple's result type is an aggregate of (i32, i64).
        let tuple_id = BasicBlock::from_id(&ctx, block)
            .iter()
            .find(|i| matches!(i.mnemonic(), Mnemonic::Tuple(_)))
            .unwrap()
            .id;
        let agg_ty = ctx.type_of(ValueId::Instruction(tuple_id));
        let fields = ctx
            .shared
            .types
            .aggregate_fields(agg_ty)
            .expect("tuple result is an aggregate");
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name, "lhs");
        assert_eq!(fields[1].name, "rhs");
        assert_eq!(ctx.shared.types.size_of(fields[0].type_id), 4);
        assert_eq!(ctx.shared.types.size_of(fields[1].type_id), 8);

        // The extract projects field 1, so its result is 8 bytes wide.
        let extract_id = BasicBlock::from_id(&ctx, block)
            .iter()
            .find(|i| matches!(i.mnemonic(), Mnemonic::Extract(e) if e.index == 1))
            .expect("extract instruction with index 1")
            .id;
        assert_eq!(
            BasicBlock::from_id(&ctx, block)
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Extract(_)))
                .unwrap()
                .as_statement()
                .to_string(),
            "i64 %x = extract(%t.rhs);"
        );
        let extract_ty = ctx.type_of(ValueId::Instruction(extract_id));
        assert_eq!(ctx.shared.types.size_of(extract_ty), 8);
    }

    #[test]
    fn gep_via_qcode_resolves_field_name_and_pointer_type() {
        let mut ctx = Context::new();
        // `Inner { val: i32 @ 0x08 }` (0x08 via leading padding), `%p : Inner*`.
        qcode!(
            ctx,
            "
            type Inner { _: 8, val: 4 };
            varnode i64 base;
            <block>
                Inner* %p = load(base:8, base);
                %f = gep(%p.val);
                return at i64 0;
            "
        );

        let gep = BasicBlock::from_id(&ctx, block)
            .iter()
            .find(|i| matches!(i.mnemonic(), Mnemonic::Gep(_)))
            .expect("gep instruction");
        let gep_id = gep.id;
        // Prints by field name, not the raw 0x8 offset.
        assert!(
            gep.as_statement().to_string().contains("gep(%p.val)"),
            "got: {}",
            gep.as_statement()
        );

        // Result type is a pointer (width 8) to the i32 field.
        let gep_ty = ctx.type_of(ValueId::Instruction(gep_id));
        assert_eq!(ctx.shared.types.size_of(gep_ty), 8);
        let pointee = ctx
            .shared
            .types
            .pointee_of(gep_ty)
            .expect("gep result is a pointer");
        assert_eq!(ctx.shared.types.size_of(pointee), 4);
    }
}
