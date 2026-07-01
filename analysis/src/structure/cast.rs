//! A small, self-owned C expression AST and its precedence-aware printer.
//!
//! Decompiler output is not valid C and must carry provenance back to the IR, so
//! there is no off-the-shelf C-AST crate that fits (`lang-c` is a parser with no
//! printer). We roll our own `Expr` here and lower IR values into it with
//! [`lower_expr`](super::lower_expr::lower_expr). All C-syntax knowledge —
//! operator spellings, precedence, parenthesization — lives in this module's
//! [`Display`] impl; nothing else in the structuring backend builds C text by
//! hand.
//!
//! Every node carries the [`ValueId`] it was lowered from (`Expr::value`) so the
//! emitter can tag each token with its provenance. This provenance is
//! deliberately excluded from equality: [`Stmt::Switch`](super::ast::Stmt)
//! recovery in [`switch`](super::switch) compares expressions structurally to
//! match a common scrutinee, and two occurrences of the same variable must
//! compare equal regardless of which IR value each was lowered from.

use std::fmt::{self, Display, Formatter};

use qcode::value::ValueId;

use super::tokens::{LineBuf, TokenKind};

/// A C expression together with the IR value it was lowered from.
#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    /// The IR value this expression renders, when known. Ignored by equality.
    pub value: Option<ValueId>,
}

/// The shape of a C expression.
///
/// [`ExprKind::Unknown`] is the total fallback so lowering never fails on an IR
/// op we have not taught it yet: it prints as `opcode(arg, ...)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprKind {
    /// An integer constant, printed in hex.
    Const(u64),
    /// A named value (varnode, block parameter, or SSA temporary).
    Var(String),
    /// A unary operation.
    Unary(UnOp, Box<Expr>),
    /// A binary operation.
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// A pointer dereference (`*e`), i.e. a load.
    Deref(Box<Expr>),
    /// An integer width/sign cast (`(type)e`).
    Cast {
        signed: bool,
        bits: usize,
        expr: Box<Expr>,
    },
    /// A pseudo-call fallback for ops we do not model structurally.
    Unknown { op: String, operands: Vec<Expr> },
}

// Equality compares structure only, never provenance — see the module docs.
// `ExprKind`'s derived `PartialEq` recurses into `Box<Expr>` through this impl,
// so the provenance-insensitivity holds at every depth.
impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}
impl Eq for Expr {}

/// A C unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    /// Arithmetic negation `-`.
    Neg,
    /// Bitwise complement `~`.
    Not,
    /// Logical not `!`.
    LNot,
}

/// A C binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Mul,
    Div,
    Rem,
    Add,
    Sub,
    Shl,
    Shr,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    BitAnd,
    BitXor,
    BitOr,
    LAnd,
    LOr,
}

impl UnOp {
    fn spelling(self) -> &'static str {
        match self {
            UnOp::Neg => "-",
            UnOp::Not => "~",
            UnOp::LNot => "!",
        }
    }
}

impl BinOp {
    fn spelling(self) -> &'static str {
        match self {
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::BitAnd => "&",
            BinOp::BitXor => "^",
            BinOp::BitOr => "|",
            BinOp::LAnd => "&&",
            BinOp::LOr => "||",
        }
    }

    /// C precedence level: smaller binds tighter. Mirrors the C operator table.
    fn precedence(self) -> u8 {
        match self {
            BinOp::Mul | BinOp::Div | BinOp::Rem => 3,
            BinOp::Add | BinOp::Sub => 4,
            BinOp::Shl | BinOp::Shr => 5,
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => 6,
            BinOp::Eq | BinOp::Ne => 7,
            BinOp::BitAnd => 8,
            BinOp::BitXor => 9,
            BinOp::BitOr => 10,
            BinOp::LAnd => 11,
            BinOp::LOr => 12,
        }
    }
}

/// Precedence of unary operators and casts (binds tighter than any binary op).
const UNARY_PREC: u8 = 2;
/// Precedence of atoms that never need parentheses.
const PRIMARY_PREC: u8 = 0;

impl Expr {
    /// An expression with the given `kind` and no provenance.
    pub fn bare(kind: ExprKind) -> Expr {
        Expr { kind, value: None }
    }

    /// A provenance-free variable atom.
    pub fn var(name: impl Into<String>) -> Expr {
        Expr::bare(ExprKind::Var(name.into()))
    }

    /// A provenance-free integer constant.
    pub fn konst(value: u64) -> Expr {
        Expr::bare(ExprKind::Const(value))
    }

    /// Convenience: logically negate this expression, folding double negation.
    #[must_use]
    pub fn logical_not(self) -> Expr {
        match self.kind {
            ExprKind::Unary(UnOp::LNot, inner) => *inner,
            kind => Expr::bare(ExprKind::Unary(
                UnOp::LNot,
                Box::new(Expr {
                    kind,
                    value: self.value,
                }),
            )),
        }
    }

    /// Emits this expression as classified tokens, parenthesizing exactly as
    /// [`Display`] does. Each node's primary token carries that node's
    /// provenance ([`Expr::value`]); colours are the front-end's concern.
    pub(crate) fn write_tokens(&self, out: &mut LineBuf) {
        match &self.kind {
            ExprKind::Const(v) => out.push_value(format!("0x{v:x}"), TokenKind::Number, self.value),
            ExprKind::Var(name) => out.push_value(name.clone(), TokenKind::Variable, self.value),
            ExprKind::Unary(op, e) => {
                out.push_value(op.spelling(), TokenKind::Operator, self.value);
                Self::child_tokens(out, e, UNARY_PREC, false);
            }
            ExprKind::Deref(e) => {
                out.push_value("*", TokenKind::Operator, self.value);
                Self::child_tokens(out, e, UNARY_PREC, false);
            }
            ExprKind::Cast { signed, bits, expr } => {
                let sign = if *signed { "int" } else { "uint" };
                out.punct("(");
                out.push_value(format!("{sign}{bits}_t"), TokenKind::Type, self.value);
                out.punct(")");
                Self::child_tokens(out, expr, UNARY_PREC, false);
            }
            ExprKind::Binary(op, lhs, rhs) => {
                let p = op.precedence();
                Self::child_tokens(out, lhs, p, false);
                out.space();
                out.push_value(op.spelling(), TokenKind::Operator, self.value);
                out.space();
                Self::child_tokens(out, rhs, p, true);
            }
            ExprKind::Unknown { op, operands } => {
                out.push_value(op.clone(), TokenKind::Label, self.value);
                out.punct("(");
                for (i, e) in operands.iter().enumerate() {
                    if i > 0 {
                        out.punct(",");
                        out.space();
                    }
                    e.write_tokens(out);
                }
                out.punct(")");
            }
        }
    }

    fn child_tokens(out: &mut LineBuf, child: &Expr, parent_prec: u8, on_right: bool) {
        let cp = child.precedence();
        let needs = cp > parent_prec || (cp == parent_prec && on_right && cp != PRIMARY_PREC);
        if needs {
            out.punct("(");
            child.write_tokens(out);
            out.punct(")");
        } else {
            child.write_tokens(out);
        }
    }

    /// The precedence of this expression's top-level operator.
    fn precedence(&self) -> u8 {
        match &self.kind {
            ExprKind::Const(_) | ExprKind::Var(_) | ExprKind::Unknown { .. } => PRIMARY_PREC,
            ExprKind::Unary(..) | ExprKind::Deref(_) | ExprKind::Cast { .. } => UNARY_PREC,
            ExprKind::Binary(op, ..) => op.precedence(),
        }
    }

    /// Formats `child`, wrapping it in parentheses only when C precedence /
    /// associativity would otherwise change the meaning. `parent_prec` is the
    /// enclosing operator's precedence; `on_right` marks the right operand of a
    /// left-associative binary operator, which needs parens at equal precedence.
    fn fmt_child(
        f: &mut Formatter<'_>,
        child: &Expr,
        parent_prec: u8,
        on_right: bool,
    ) -> fmt::Result {
        let cp = child.precedence();
        let needs = cp > parent_prec || (cp == parent_prec && on_right && cp != PRIMARY_PREC);
        if needs {
            write!(f, "({child})")
        } else {
            write!(f, "{child}")
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ExprKind::Const(v) => write!(f, "0x{v:x}"),
            ExprKind::Var(name) => write!(f, "{name}"),
            ExprKind::Unary(op, e) => {
                write!(f, "{}", op.spelling())?;
                Expr::fmt_child(f, e, UNARY_PREC, false)
            }
            ExprKind::Deref(e) => {
                write!(f, "*")?;
                Expr::fmt_child(f, e, UNARY_PREC, false)
            }
            ExprKind::Cast { signed, bits, expr } => {
                let sign = if *signed { "int" } else { "uint" };
                write!(f, "({sign}{bits}_t)")?;
                Expr::fmt_child(f, expr, UNARY_PREC, false)
            }
            ExprKind::Binary(op, lhs, rhs) => {
                let p = op.precedence();
                Expr::fmt_child(f, lhs, p, false)?;
                write!(f, " {} ", op.spelling())?;
                Expr::fmt_child(f, rhs, p, true)
            }
            ExprKind::Unknown { op, operands } => {
                write!(f, "{op}(")?;
                for (i, e) in operands.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{e}")?;
                }
                write!(f, ")")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::bare(ExprKind::Binary(op, Box::new(l), Box::new(r)))
    }

    #[test]
    fn precedence_omits_redundant_parens() {
        // a + b * c  ->  no parens around the product
        let e = bin(
            BinOp::Add,
            Expr::var("a"),
            bin(BinOp::Mul, Expr::var("b"), Expr::var("c")),
        );
        assert_eq!(e.to_string(), "a + b * c");
    }

    #[test]
    fn precedence_adds_needed_parens() {
        // (a + b) * c
        let e = bin(
            BinOp::Mul,
            bin(BinOp::Add, Expr::var("a"), Expr::var("b")),
            Expr::var("c"),
        );
        assert_eq!(e.to_string(), "(a + b) * c");
    }

    #[test]
    fn left_assoc_right_operand_is_parenthesized() {
        // a - (b - c) must keep parens; a - b - c would be wrong.
        let e = bin(
            BinOp::Sub,
            Expr::var("a"),
            bin(BinOp::Sub, Expr::var("b"), Expr::var("c")),
        );
        assert_eq!(e.to_string(), "a - (b - c)");
    }

    #[test]
    fn double_logical_not_folds() {
        let e = Expr::var("x").logical_not().logical_not();
        assert_eq!(e, Expr::var("x"));
    }
}
