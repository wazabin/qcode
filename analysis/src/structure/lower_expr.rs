//! Lowering from IR values ([`ValueId`]) into the C expression AST ([`Expr`]).
//!
//! Pure operations (arithmetic, comparisons, loads, width casts) are inlined
//! recursively so conditions render as real expressions (e.g. `a == 0x5`), which
//! later phases — notably SAILR switch recovery — need to inspect structurally.
//! Impure or unmodeled instructions are referenced by their SSA name instead, or
//! fall back to [`ExprKind::Unknown`], so lowering is total.

use std::collections::HashSet;

use qcode::{
    context::Context,
    value::{
        BlockParam, Instruction, InstructionId, LiteralRef, ValueId, Varnode,
        insn::{Binop, BoolBinop, FloatBinop, IntBinop, Mnemonic, Unop},
    },
};

use super::cast::{BinOp, Expr, ExprKind, UnOp};

/// The set of instructions that are rendered as their own statements ("roots")
/// rather than inlined. A root reached as an operand is printed by name.
pub(crate) type Roots<'a> = Option<&'a HashSet<InstructionId>>;

/// Lowers an IR value into a C expression, inlining all pure operations.
pub fn lower_expr(ctx: &Context, value: ValueId) -> Expr {
    lower(ctx, value, None)
}

/// Like [`lower_expr`], but stops inlining at any instruction in `roots`,
/// printing it by name. Used by the statement emitter to avoid duplicating a
/// value that also appears as its own assignment.
pub(crate) fn lower_expr_rooted(
    ctx: &Context,
    value: ValueId,
    roots: &HashSet<InstructionId>,
) -> Expr {
    lower(ctx, value, Some(roots))
}

fn lower(ctx: &Context, value: ValueId, roots: Roots) -> Expr {
    let e = match value {
        ValueId::Literal(id) => Expr::konst(LiteralRef::from_id(ctx, id).value()),
        ValueId::Varnode(id) => Expr::var(name_or(Varnode::from_id(ctx, id).name(), "var", value)),
        ValueId::BlockParam(id) => {
            Expr::var(name_or(BlockParam::from_id(ctx, id).name(), "p", value))
        }
        ValueId::Instruction(id) => {
            if roots.is_some_and(|r| r.contains(&id)) {
                Expr::var(instruction_name(ctx, id))
            } else {
                lower_instruction(ctx, id, roots, false)
            }
        }
        // Blocks and functions appearing in an expression context are opaque
        // names (address-of-label, function pointer).
        _ => Expr::var(format!("{}", qcode::value::ValueRef::new(value, ctx))),
    };
    // Stamp the node with the value it renders. For an inlined instruction this
    // overrides any inner leaf's provenance (e.g. a load-of-varnode's `Var`
    // token now points at the load's SSA value — the useful click target).
    Expr {
        value: Some(value),
        ..e
    }
}

/// Lowers a memory reference: a named location (varnode) reads as the variable
/// itself, a computed address as an explicit `*ptr` dereference. Shared by load
/// (rvalue) and store (lvalue) lowering.
pub(crate) fn deref_location(ctx: &Context, ptr: ValueId, roots: Roots) -> Expr {
    match ptr {
        ValueId::Varnode(_) => lower(ctx, ptr, roots),
        _ => Expr::bare(ExprKind::Deref(Box::new(lower(ctx, ptr, roots)))),
    }
}

/// The display name of an instruction's SSA result.
pub(crate) fn instruction_name(ctx: &Context, id: InstructionId) -> String {
    match Instruction::from_id(ctx, id).name() {
        Some(name) => name.to_string(),
        None => format!("v{}", Into::<usize>::into(id)),
    }
}

/// Lowers the *defining* expression of a root instruction (its own operator
/// expanded, operands referenced by name when they are themselves roots). Used
/// to render `name = <expr>;` assignments.
pub(crate) fn lower_defining_expr(
    ctx: &Context,
    id: InstructionId,
    roots: &HashSet<InstructionId>,
) -> Expr {
    // `lower_instruction` bypasses `lower`, so stamp the result with the
    // instruction's own value (its node is otherwise provenance-free).
    let mut e = lower_instruction(ctx, id, Some(roots), true);
    e.value = Some(ValueId::Instruction(id));
    e
}

fn lower_instruction(ctx: &Context, id: InstructionId, roots: Roots, expand_unknown: bool) -> Expr {
    let insn = Instruction::from_id(ctx, id);
    match insn.mnemonic() {
        Mnemonic::Binop(b) => match map_binop(&b.op) {
            Some(op) => Expr::bare(ExprKind::Binary(
                op,
                Box::new(lower(ctx, b.lhs, roots)),
                Box::new(lower(ctx, b.rhs, roots)),
            )),
            None => Expr::bare(ExprKind::Unknown {
                op: insn.opcode().to_string(),
                operands: vec![lower(ctx, b.lhs, roots), lower(ctx, b.rhs, roots)],
            }),
        },
        Mnemonic::Unop(u) => match map_unop(&u.op) {
            Some(op) => Expr::bare(ExprKind::Unary(op, Box::new(lower(ctx, u.src, roots)))),
            None => Expr::bare(ExprKind::Unknown {
                op: insn.opcode().to_string(),
                operands: vec![lower(ctx, u.src, roots)],
            }),
        },
        // A load from a named location (varnode) reads that variable directly;
        // a load through a computed pointer is a real dereference.
        Mnemonic::Load(l) => deref_location(ctx, l.ptr, roots),
        Mnemonic::Zext(z) => Expr::bare(ExprKind::Cast {
            signed: false,
            bits: z.size * 8,
            expr: Box::new(lower(ctx, z.src, roots)),
        }),
        Mnemonic::Sext(s) => Expr::bare(ExprKind::Cast {
            signed: true,
            bits: s.size * 8,
            expr: Box::new(lower(ctx, s.src, roots)),
        }),
        // Not modeled structurally. As an operand (`expand_unknown` false) refer
        // to the SSA temporary by name; as a defining expression expand it to an
        // opaque `opcode(args)` pseudo-call so the assignment is meaningful.
        _ => {
            if !expand_unknown && let Some(name) = insn.name() {
                return Expr::var(name.to_string());
            }
            Expr::bare(ExprKind::Unknown {
                op: insn.opcode().to_string(),
                operands: insn
                    .mnemonic()
                    .args()
                    .into_iter()
                    .map(|v| lower(ctx, v, roots))
                    .collect(),
            })
        }
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
