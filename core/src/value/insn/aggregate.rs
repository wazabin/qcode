//! Aggregate construction and projection.
//!
//! These are the functional-IR counterpart of a tuple: [`Tuple`] groups several
//! values into one aggregate-typed result, and [`Extract`] projects a single
//! field back out. They preserve the one-value-per-instruction invariant (an
//! `Extract` *is* the field value it names), so no multi-result instruction is
//! needed. `argpromote` uses them to return `(real_return, write-set)`.

use crate::{
    context::Context,
    types::TypeId,
    value::{ValueId, ValueRef},
};
use std::fmt::Formatter;

use super::mnemonic::MnemonicKind;

fn fmt_bare_value(f: &mut Formatter<'_>, ctx: &Context<'_>, value: ValueId) -> std::fmt::Result {
    match value {
        ValueId::Instruction(id) => {
            if let Some(name) = ctx.values.instructions[id].name.as_deref() {
                write!(f, "%{name}")
            } else {
                write!(f, "%tmp{:x}", usize::from(id))
            }
        }
        other => write!(f, "{}", ValueRef::new(other, ctx)),
    }
}

/// Builds an aggregate value from its ordered fields. The instruction's result
/// type is the [`Aggregate`](crate::types::TypeRepr::Aggregate) of the fields'
/// types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Tuple {
    pub fields: Vec<ValueId>,
}

impl Tuple {
    pub fn fmt_with_type(
        &self,
        f: &mut Formatter<'_>,
        ctx: &Context<'_>,
        type_id: TypeId,
    ) -> std::fmt::Result {
        write!(f, "pack(")?;
        for (i, &field) in self.fields.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            let name = ctx
                .types
                .field_name(type_id, i)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("field{}", i + 1));
            write!(f, "{}={}", name, ValueRef::new(field, ctx))?;
        }
        write!(f, ");")
    }
}

impl MnemonicKind for Tuple {
    fn opcode(&self) -> &'static str {
        "pack"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(f, "pack(")?;
        for (i, &field) in self.fields.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "field{}={}", i + 1, ValueRef::new(field, ctx))?;
        }
        write!(f, ");")
    }

    fn args(&self) -> Vec<ValueId> {
        self.fields.clone()
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
        ctx.types.field_name(agg_ty, self.index)
    }
}

impl MnemonicKind for Extract {
    fn opcode(&self) -> &'static str {
        "extract"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        let name = self
            .field_name(ctx)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("field{}", self.index + 1));
        f.write_str("extract(")?;
        fmt_bare_value(f, ctx, self.agg)?;
        write!(f, ".{name});")
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.agg]
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
                return [i64 0];
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
            .types
            .aggregate_fields(agg_ty)
            .expect("tuple result is an aggregate");
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name, "lhs");
        assert_eq!(fields[1].name, "rhs");
        assert_eq!(ctx.types.size_of(fields[0].type_id), 4);
        assert_eq!(ctx.types.size_of(fields[1].type_id), 8);

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
        assert_eq!(ctx.types.size_of(extract_ty), 8);
    }
}
