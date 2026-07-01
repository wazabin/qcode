//! A small, self-owned C expression AST and its precedence-aware printer.
//!
//! Decompiler output is not valid C and must carry provenance back to the IR, so
//! there is no off-the-shelf C-AST crate that fits (`lang-c` is a parser with no
//! printer). We roll our own `Expr` here and lower IR values into it with
//! [`lower_expr`]. All C-syntax knowledge — operator spellings, precedence,
//! parenthesization — lives in this module's [`Display`] impl; nothing else in
//! the structuring backend builds C text by hand.

use std::fmt::{self, Display, Formatter};

/// A C expression.
///
/// [`Expr::Unknown`] is the total fallback so lowering never fails on an IR op
/// we have not taught it yet: it prints as `opcode(arg, ...)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
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
    /// Convenience: logically negate this expression, folding double negation.
    #[must_use]
    pub fn logical_not(self) -> Expr {
        match self {
            Expr::Unary(UnOp::LNot, inner) => *inner,
            other => Expr::Unary(UnOp::LNot, Box::new(other)),
        }
    }

    /// The precedence of this expression's top-level operator.
    fn precedence(&self) -> u8 {
        match self {
            Expr::Const(_) | Expr::Var(_) | Expr::Unknown { .. } => PRIMARY_PREC,
            Expr::Unary(..) | Expr::Deref(_) | Expr::Cast { .. } => UNARY_PREC,
            Expr::Binary(op, ..) => op.precedence(),
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
        match self {
            Expr::Const(v) => write!(f, "0x{v:x}"),
            Expr::Var(name) => write!(f, "{name}"),
            Expr::Unary(op, e) => {
                write!(f, "{}", op.spelling())?;
                Expr::fmt_child(f, e, UNARY_PREC, false)
            }
            Expr::Deref(e) => {
                write!(f, "*")?;
                Expr::fmt_child(f, e, UNARY_PREC, false)
            }
            Expr::Cast { signed, bits, expr } => {
                let sign = if *signed { "int" } else { "uint" };
                write!(f, "({sign}{bits}_t)")?;
                Expr::fmt_child(f, expr, UNARY_PREC, false)
            }
            Expr::Binary(op, lhs, rhs) => {
                let p = op.precedence();
                Expr::fmt_child(f, lhs, p, false)?;
                write!(f, " {} ", op.spelling())?;
                Expr::fmt_child(f, rhs, p, true)
            }
            Expr::Unknown { op, operands } => {
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
        Expr::Binary(op, Box::new(l), Box::new(r))
    }

    #[test]
    fn precedence_omits_redundant_parens() {
        // a + b * c  ->  no parens around the product
        let e = bin(
            BinOp::Add,
            Expr::Var("a".into()),
            bin(BinOp::Mul, Expr::Var("b".into()), Expr::Var("c".into())),
        );
        assert_eq!(e.to_string(), "a + b * c");
    }

    #[test]
    fn precedence_adds_needed_parens() {
        // (a + b) * c
        let e = bin(
            BinOp::Mul,
            bin(BinOp::Add, Expr::Var("a".into()), Expr::Var("b".into())),
            Expr::Var("c".into()),
        );
        assert_eq!(e.to_string(), "(a + b) * c");
    }

    #[test]
    fn left_assoc_right_operand_is_parenthesized() {
        // a - (b - c) must keep parens; a - b - c would be wrong.
        let e = bin(
            BinOp::Sub,
            Expr::Var("a".into()),
            bin(BinOp::Sub, Expr::Var("b".into()), Expr::Var("c".into())),
        );
        assert_eq!(e.to_string(), "a - (b - c)");
    }

    #[test]
    fn double_logical_not_folds() {
        let e = Expr::Var("x".into()).logical_not().logical_not();
        assert_eq!(e, Expr::Var("x".into()));
    }
}
