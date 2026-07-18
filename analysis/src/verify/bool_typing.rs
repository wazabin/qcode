//! Verify the `bool` type discipline.
//!
//! `bool` is byte-stored but distinct from `Int(1)`: it is minted only by
//! comparisons and the `true`/`false` literals, and the type system forbids
//! silently mixing it with integers. These rules pin that discipline so a pass
//! that mistypes a value (or a lifter that forgets a cast) is caught at the
//! offending pass rather than surfacing as a miscompile downstream.
//!
//! Rules (see `docs/plans/bool-type/00-overview.md`):
//! - `And`/`Or`/`Xor` must have *both* operands `bool` or *neither* — a mixed
//!   `bool`/`iN` bitwise op is an error.
//! - Any other arithmetic/shift binop with a `bool` operand is an error.
//! - `IntNot` on a `bool` is an error (negation is `x == false`).
//! - A branch condition must be `bool`.
//! - A `bool` constant must be `0` or `1`.

use qcode::{
    context::Context,
    value::{
        ValueId,
        insn::{Binop, IntBinop, Mnemonic, Unop},
    },
};

/// Whether `v` is a `bool`-typed value. Varnodes (which carry no stored type)
/// and non-data values read as non-bool.
fn is_bool(ctx: &Context, v: ValueId) -> bool {
    ctx.stored_type_of(v)
        .is_some_and(|t| ctx.shared.types.is_bool(t))
}

/// The literal value of `v` if it is a `bool`-typed literal, for domain checks.
fn bool_literal_value(ctx: &Context, v: ValueId) -> Option<u64> {
    match v {
        ValueId::Literal(lid) if is_bool(ctx, v) => Some(ctx.shared.values.literals[lid].value),
        _ => None,
    }
}

fn describe_value(ctx: &Context, value: ValueId) -> String {
    match value {
        ValueId::Instruction(id) => {
            format!("{value:?} ({:?})", ctx.get_insn(id).mnemonic())
        }
        _ => format!("{value:?}"),
    }
}

/// Run the `bool`-typing checks over every instruction.
pub fn verify_bool_typing(ctx: &Context) -> Vec<String> {
    verify_bool_typing_scoped(ctx, super::Scope::All)
}

pub(crate) fn verify_bool_typing_scoped(ctx: &Context, scope: super::Scope<'_>) -> Vec<String> {
    let mut out = Vec::new();

    let check_domain = |v: ValueId, out: &mut Vec<String>| {
        if let Some(val) = bool_literal_value(ctx, v)
            && val > 1
        {
            out.push(format!("bool constant {val} outside the {{0, 1}} domain"));
        }
    };

    for insn in scope.instructions(ctx) {
        let func = insn.id.func;
        match insn.mnemonic() {
            Mnemonic::Binop(b) => {
                check_domain(b.lhs.qualify(func), &mut out);
                check_domain(b.rhs.qualify(func), &mut out);
                let lhs_bool = is_bool(ctx, b.lhs.qualify(func));
                let rhs_bool = is_bool(ctx, b.rhs.qualify(func));
                match b.op {
                    // Bitwise and/or/xor: logical when both bool, integer when
                    // neither, but never mixed.
                    Binop::Int(IntBinop::And | IntBinop::Or | IntBinop::Xor)
                        if lhs_bool != rhs_bool =>
                    {
                        let lhs = b.lhs.qualify(func);
                        let rhs = b.rhs.qualify(func);
                        let lhs = describe_value(ctx, lhs);
                        let rhs = describe_value(ctx, rhs);
                        out.push(format!(
                            "instruction {:?} (parent {:?}) bitwise `{}` mixes bool and integer operands: lhs {lhs} is {}, rhs {rhs} is {}",
                            insn.id,
                            insn.parent().map(|block| block.id),
                            b.op,
                            if lhs_bool { "bool" } else { "integer" },
                            if rhs_bool { "bool" } else { "integer" },
                        ));
                    }
                    Binop::Int(IntBinop::And | IntBinop::Or | IntBinop::Xor) => {}
                    // Comparisons legitimately consume bool operands
                    // (`x == false` is the canonical negation) and any integers.
                    Binop::Int(op) if op.is_comparison() => {}
                    Binop::Float(op) if op.is_comparison() => {}
                    // Every other arithmetic/shift op rejects bool operands.
                    Binop::Int(_) | Binop::Float(_) if lhs_bool || rhs_bool => {
                        out.push(format!("arithmetic `{}` applied to a bool operand", b.op));
                    }
                    _ => {}
                }
            }
            Mnemonic::Unop(u)
                if matches!(u.op, Unop::IntNot) && is_bool(ctx, u.src.qualify(func)) =>
            {
                out.push("bitwise `~` applied to a bool operand (use `x == false`)".to_string());
            }
            Mnemonic::CBranch(cb) => {
                check_domain(cb.condition.qualify(func), &mut out);
                if !is_bool(ctx, cb.condition.qualify(func)) {
                    out.push(format!(
                        "branch condition {:?} is not bool-typed",
                        cb.condition
                    ));
                }
            }
            Mnemonic::Assert(a) => {
                check_domain(a.condition.qualify(func), &mut out);
            }
            _ => {}
        }
    }

    out
}

#[cfg(test)]
mod tests {

    use qcode::context::Context;

    use super::verify_bool_typing;

    #[test]
    fn rejects_arithmetic_and_mixed_bitwise_on_bool() {
        let mut ctx = Context::new();
        let a = ctx.get_bool_const(true).id();
        let b = ctx.get_const(3, 1).id();
        {
            let source = ctx.builder_at(0x1000).current_block();
            let target = ctx.get_or_make_block(0x1001, source.func);
            let mut builder = ctx.builder(source);
            builder.push_add(a, b); // Add(bool, i8): rejected
            builder.push_bit_and(a, b); // And(bool, i8): mixed, rejected
            builder.push_bit_negate(a); // IntNot(bool): rejected
            builder.finalize(target);
        }
        let diags = verify_bool_typing(&ctx);
        assert_eq!(diags.len(), 3, "expected 3 diagnostics, got {diags:?}");
    }

    #[test]
    fn accepts_bool_bitwise_and_comparison() {
        let mut ctx = Context::new();
        let a = ctx.get_bool_const(true).id();
        let b = ctx.get_bool_const(false).id();
        {
            let source = ctx.builder_at(0x1000).current_block();
            let target = ctx.get_or_make_block(0x1001, source.func);
            let mut builder = ctx.builder(source);
            builder.push_bit_and(a, b); // And(bool, bool): logical, ok
            builder.push_eq(a, b); // Equal(bool, bool): negation idiom, ok
            builder.finalize(target);
        }
        assert!(verify_bool_typing(&ctx).is_empty());
    }
}
