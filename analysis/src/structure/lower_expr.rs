//! Lowering from IR values ([`ValueId`]) into the C expression AST ([`Expr`]).
//!
//! Pure operations (arithmetic, comparisons, loads, width casts) are inlined
//! recursively so conditions render as real expressions (e.g. `a == 0x5`), which
//! later phases — notably SAILR switch recovery — need to inspect structurally.
//! Impure or unmodeled instructions are referenced by their SSA name instead, or
//! fall back to [`Expr::Unknown`], so lowering is total.

use qcode::{
    context::Context,
    value::{
        BlockParam, Instruction, InstructionId, LiteralRef, ValueId, Varnode,
        insn::{Binop, BoolBinop, FloatBinop, IntBinop, Mnemonic, Unop},
    },
};

use super::cast::{BinOp, Expr, UnOp};

/// Lowers an IR value into a C expression.
pub fn lower_expr(ctx: &Context, value: ValueId) -> Expr {
    match value {
        ValueId::Literal(id) => Expr::Const(LiteralRef::from_id(ctx, id).value()),
        ValueId::Varnode(id) => Expr::Var(name_or(Varnode::from_id(ctx, id).name(), "var", value)),
        ValueId::BlockParam(id) => {
            Expr::Var(name_or(BlockParam::from_id(ctx, id).name(), "p", value))
        }
        ValueId::Instruction(id) => lower_instruction(ctx, id),
        // Blocks and functions appearing in an expression context are opaque
        // names (address-of-label, function pointer).
        _ => Expr::Var(format!("{}", qcode::value::ValueRef::new(value, ctx))),
    }
}

fn lower_instruction(ctx: &Context, id: InstructionId) -> Expr {
    let insn = Instruction::from_id(ctx, id);
    match insn.mnemonic() {
        Mnemonic::Binop(b) => match map_binop(&b.op) {
            Some(op) => Expr::Binary(
                op,
                Box::new(lower_expr(ctx, b.lhs)),
                Box::new(lower_expr(ctx, b.rhs)),
            ),
            None => Expr::Unknown {
                op: insn.opcode().to_string(),
                operands: vec![lower_expr(ctx, b.lhs), lower_expr(ctx, b.rhs)],
            },
        },
        Mnemonic::Unop(u) => match map_unop(&u.op) {
            Some(op) => Expr::Unary(op, Box::new(lower_expr(ctx, u.src))),
            None => Expr::Unknown {
                op: insn.opcode().to_string(),
                operands: vec![lower_expr(ctx, u.src)],
            },
        },
        Mnemonic::Load(l) => Expr::Deref(Box::new(lower_expr(ctx, l.ptr))),
        Mnemonic::Zext(z) => Expr::Cast {
            signed: false,
            bits: z.size * 8,
            expr: Box::new(lower_expr(ctx, z.src)),
        },
        Mnemonic::Sext(s) => Expr::Cast {
            signed: true,
            bits: s.size * 8,
            expr: Box::new(lower_expr(ctx, s.src)),
        },
        // Not modeled structurally: refer to the SSA temporary by name if it has
        // one, else emit an opaque pseudo-call so lowering stays total.
        _ => match insn.name() {
            Some(name) => Expr::Var(name.to_string()),
            None => Expr::Unknown {
                op: insn.opcode().to_string(),
                operands: Vec::new(),
            },
        },
    }
}

/// The C unary operator for an IR [`Unop`], or `None` for ops with no C operator
/// (float abs/sqrt/ceil/floor/round), which the caller renders as a pseudo-call.
fn map_unop(op: &Unop) -> Option<UnOp> {
    match op {
        Unop::IntNegate | Unop::FloatNegate => Some(UnOp::Neg),
        Unop::IntNot => Some(UnOp::Not),
        Unop::BoolNot => Some(UnOp::LNot),
        _ => None,
    }
}

fn map_binop(op: &Binop) -> Option<BinOp> {
    match op {
        Binop::Int(i) => map_int_binop(i),
        Binop::Bool(b) => map_bool_binop(b),
        Binop::Float(fl) => map_float_binop(fl),
        _ => None,
    }
}

fn map_int_binop(op: &IntBinop) -> Option<BinOp> {
    Some(match op {
        IntBinop::Equal => BinOp::Eq,
        IntBinop::NotEqual => BinOp::Ne,
        IntBinop::Less | IntBinop::SLess => BinOp::Lt,
        IntBinop::LessEqual | IntBinop::SLessEqual => BinOp::Le,
        IntBinop::Add => BinOp::Add,
        IntBinop::Sub => BinOp::Sub,
        IntBinop::Xor => BinOp::BitXor,
        IntBinop::And => BinOp::BitAnd,
        IntBinop::Or => BinOp::BitOr,
        IntBinop::ShiftLeft => BinOp::Shl,
        IntBinop::ShiftRight | IntBinop::SShiftRight => BinOp::Shr,
        IntBinop::Mul => BinOp::Mul,
        IntBinop::Div | IntBinop::Sdiv => BinOp::Div,
        IntBinop::Rem | IntBinop::Srem => BinOp::Rem,
        _ => return None,
    })
}

fn map_bool_binop(op: &BoolBinop) -> Option<BinOp> {
    Some(match op {
        BoolBinop::And => BinOp::LAnd,
        BoolBinop::Or => BinOp::LOr,
        BoolBinop::Xor => BinOp::BitXor,
        _ => return None,
    })
}

fn map_float_binop(op: &FloatBinop) -> Option<BinOp> {
    Some(match op {
        FloatBinop::Equal => BinOp::Eq,
        FloatBinop::NotEqual => BinOp::Ne,
        FloatBinop::Less => BinOp::Lt,
        FloatBinop::LessEqual => BinOp::Le,
        FloatBinop::Add => BinOp::Add,
        FloatBinop::Sub => BinOp::Sub,
        FloatBinop::Mul => BinOp::Mul,
        FloatBinop::Div => BinOp::Div,
        _ => return None,
    })
}

/// A name for a leaf value, or a synthetic `prefix_<id>` fallback.
fn name_or(name: Option<&str>, prefix: &str, value: ValueId) -> String {
    match name {
        Some(name) => name.to_string(),
        None => format!("{prefix}_{}", value_index(value)),
    }
}

fn value_index(value: ValueId) -> usize {
    match value {
        ValueId::Varnode(id) => id.into(),
        ValueId::BlockParam(id) => id.into(),
        ValueId::Instruction(id) => id.into(),
        ValueId::Literal(id) => id.into(),
        ValueId::BasicBlock(id) => id.into(),
        ValueId::Function(id) => id.into(),
        _ => 0,
    }
}
