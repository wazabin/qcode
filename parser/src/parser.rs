use crate::ast::{Atom, CastOp, ExprNode, Statement, TypedAtom};
use pest::Parser;
use pest::iterators::Pair;
use pest::iterators::Pairs;
use pest_derive::Parser;
use std::fmt;

#[derive(Debug, Clone)]
pub struct ParseError {
    message: String,
}

impl ParseError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ParseError {}

#[derive(Parser)]
#[grammar = "qcode.pest"]
struct QCodeParser;

pub fn parse_program(program: &str) -> Result<Vec<Statement>, ParseError> {
    let mut parsed = QCodeParser::parse(Rule::program, program).map_err(to_parse_error)?;
    let root = parsed
        .next()
        .ok_or_else(|| ParseError::new("missing program"))?;

    let mut out = Vec::new();

    for pair in root.into_inner() {
        if pair.as_rule() != Rule::statement_list {
            continue;
        }
        for compound in pair.into_inner() {
            if compound.as_rule() != Rule::compound_stmt {
                continue;
            }
            parse_compound(compound, &mut out)?;
        }
    }

    Ok(out)
}

fn parse_compound(pair: Pair<'_, Rule>, out: &mut Vec<Statement>) -> Result<(), ParseError> {
    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::label => out.push(parse_label(part.into_inner())?),
            Rule::inner_stmt => parse_inner_stmt(part, out)?,
            _ => return Err(ParseError::new("unexpected compound statement")),
        }
    }
    Ok(())
}

fn parse_inner_stmt(pair: Pair<'_, Rule>, out: &mut Vec<Statement>) -> Result<(), ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::new("empty inner statement"))?;

    let stmt = match inner.as_rule() {
        Rule::local_decl => parse_local_decl(inner)?,
        Rule::assignment => parse_assignment(inner)?,
        Rule::terminator => {
            let specific = inner
                .into_inner()
                .next()
                .ok_or_else(|| ParseError::new("missing terminator kind"))?;
            let rule = specific.as_rule();
            parse_terminator(specific.into_inner(), rule)?
        }
        Rule::expr => Statement::Expr(parse_expr(inner)?),
        _ => return Err(ParseError::new("unexpected inner statement")),
    };

    out.push(stmt);
    Ok(())
}

fn parse_local_decl(pair: Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let mut inner = pair.into_inner();
    let size_bytes = parse_size_bytes(
        inner
            .next()
            .ok_or_else(|| ParseError::new("missing local type"))?
            .as_str(),
        "local declaration",
    )?;
    let name = inner
        .next()
        .ok_or_else(|| ParseError::new("missing local name"))?
        .as_str()
        .to_owned();
    let display_name = inner
        .next()
        .map(|pair| pair.as_str().to_owned())
        .unwrap_or_else(|| name.clone());

    Ok(Statement::LocalDecl {
        name,
        display_name,
        size_bytes,
    })
}

fn parse_label(mut inner: Pairs<'_, Rule>) -> Result<Statement, ParseError> {
    let name = inner
        .next()
        .ok_or_else(|| ParseError::new("missing label name"))?
        .as_str()
        .to_owned();
    Ok(Statement::LabelDecl { name })
}

/// Extracts the label name from a `label` pair (`< ident >`).
fn label_name(pair: Pair<'_, Rule>) -> Result<String, ParseError> {
    pair.into_inner()
        .next()
        .ok_or_else(|| ParseError::new("missing label name"))
        .map(|p| p.as_str().to_owned())
}

fn parse_terminator(mut inner: Pairs<'_, Rule>, rule: Rule) -> Result<Statement, ParseError> {
    match rule {
        Rule::branch_stmt => {
            let target = label_name(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing branch target"))?,
            )?;
            Ok(Statement::Branch { target })
        }

        Rule::branchind_stmt => {
            let ptr = parse_typed_atom(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing branchind pointer"))?,
            )?;
            Ok(Statement::BranchInd { ptr })
        }

        Rule::cbranch_stmt => {
            let condition = parse_typed_atom(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing cbranch condition"))?,
            )?;
            let target = label_name(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing cbranch target"))?,
            )?;
            let fallthrough = label_name(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing cbranch fallthrough"))?,
            )?;
            Ok(Statement::CBranch {
                condition,
                target,
                fallthrough,
            })
        }

        Rule::call_stmt => {
            let target = label_name(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing call target"))?,
            )?;
            Ok(Statement::Call { target })
        }

        Rule::callind_stmt => {
            let ptr = parse_typed_atom(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing callind pointer"))?,
            )?;
            Ok(Statement::CallInd { ptr })
        }

        Rule::return_stmt => {
            let ptr = parse_typed_atom(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing return pointer"))?,
            )?;
            Ok(Statement::Return { ptr })
        }

        _ => Err(ParseError::new("invalid terminator")),
    }
}

fn parse_assignment(pair: Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::new("missing assignment variant"))?;

    match inner.as_rule() {
        Rule::assignment_ssa => parse_assignment_ssa(inner),
        Rule::assignment_plain => parse_assignment_plain(inner),
        _ => Err(ParseError::new("unexpected assignment rule")),
    }
}

fn parse_assignment_ssa(pair: Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let mut inner = pair.into_inner();
    // Skip optional ty token
    let name_or_ty = inner
        .next()
        .ok_or_else(|| ParseError::new("missing ssa assignment name"))?;
    let ssa_pair = if name_or_ty.as_rule() == Rule::ty {
        inner
            .next()
            .ok_or_else(|| ParseError::new("missing ssa name after type"))?
    } else {
        name_or_ty
    };
    let name = ssa_pair
        .as_str()
        .strip_prefix('%')
        .ok_or_else(|| ParseError::new("ssa name missing % prefix"))?
        .to_owned();
    let expr_pair = inner
        .find(|p| p.as_rule() == Rule::expr)
        .ok_or_else(|| ParseError::new("missing ssa assignment expression"))?;
    Ok(Statement::Assign {
        name,
        expose: true,
        expr: parse_expr(expr_pair)?,
    })
}

fn parse_assignment_plain(pair: Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let mut inner = pair.into_inner();
    let name = inner
        .next()
        .ok_or_else(|| ParseError::new("missing assignment name"))?
        .as_str()
        .to_owned();
    let expr_pair = inner
        .find(|p| p.as_rule() == Rule::expr)
        .ok_or_else(|| ParseError::new("missing assignment expression"))?;
    Ok(Statement::Assign {
        name,
        expose: false,
        expr: parse_expr(expr_pair)?,
    })
}

fn parse_expr(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::new("missing expression"))?;

    match inner.as_rule() {
        Rule::atom_expr => {
            let typed_atom = inner
                .into_inner()
                .next()
                .ok_or_else(|| ParseError::new("missing atom"))?;
            Ok(ExprNode::Atom(parse_typed_atom(typed_atom)?))
        }
        Rule::unop => parse_unop(inner),
        Rule::func_unop => parse_func_unop(inner),
        Rule::func_call => parse_func_call(inner),
        Rule::binary => parse_binary(inner),
        Rule::memory => parse_memory(inner),
        Rule::cast => parse_cast(inner),
        _ => Err(ParseError::new("invalid expression")),
    }
}

fn parse_unop(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let mut inner = pair.into_inner();
    let op = inner
        .next()
        .ok_or_else(|| ParseError::new("missing unary operator"))?
        .as_str()
        .to_owned();
    let src = parse_typed_atom(
        inner
            .next()
            .ok_or_else(|| ParseError::new("missing unary source"))?,
    )?;

    Ok(ExprNode::Unop { op, src })
}

fn parse_func_unop(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let mut inner = pair.into_inner();
    let op = inner
        .next()
        .ok_or_else(|| ParseError::new("missing unary function"))?
        .as_str()
        .to_owned();
    let src = parse_typed_atom(
        inner
            .next()
            .ok_or_else(|| ParseError::new("missing unary function source"))?,
    )?;

    Ok(ExprNode::Unop { op, src })
}

fn parse_func_call(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let mut op = None;
    let mut args = Vec::new();

    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::func_ident => op = Some(part.as_str().to_owned()),
            Rule::typed_atom => args.push(parse_typed_atom(part)?),
            _ => {}
        }
    }

    Ok(ExprNode::FuncCall {
        op: op.ok_or_else(|| ParseError::new("missing function name"))?,
        args,
    })
}

fn parse_binary(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let mut inner = pair.into_inner();
    let lhs = parse_typed_atom(inner.next().ok_or_else(|| ParseError::new("missing lhs"))?)?;
    let op = inner
        .next()
        .ok_or_else(|| ParseError::new("missing operator"))?
        .as_str()
        .to_owned();
    let rhs = parse_typed_atom(inner.next().ok_or_else(|| ParseError::new("missing rhs"))?)?;

    Ok(ExprNode::Binary { lhs, op, rhs })
}

fn parse_memory(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::new("missing memory expression"))?;

    match inner.as_rule() {
        Rule::load => {
            let mut load_inner = inner.into_inner();
            let type_pair = load_inner
                .find(|p| p.as_rule() == Rule::ty)
                .ok_or_else(|| ParseError::new("missing load type"))?;
            let size_bytes = parse_size_bytes(type_pair.as_str(), "load")?;
            let ptr_pair = load_inner
                .find(|p| p.as_rule() == Rule::typed_atom)
                .ok_or_else(|| ParseError::new("missing load pointer"))?;
            let ptr = parse_typed_atom(ptr_pair)?;

            Ok(ExprNode::Load { size_bytes, ptr })
        }
        Rule::store => {
            let mut store_inner = inner
                .into_inner()
                .filter(|p| p.as_rule() == Rule::typed_atom);
            let ptr = parse_typed_atom(
                store_inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing store pointer"))?,
            )?;
            let src = parse_typed_atom(
                store_inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing store source"))?,
            )?;

            Ok(ExprNode::Store { ptr, src })
        }
        _ => Err(ParseError::new("invalid memory expression")),
    }
}

fn parse_cast(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let mut op = None;
    let mut ty = None;
    let mut src = None;

    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::cast_op => {
                op = Some(match part.as_str() {
                    "zext" => CastOp::Zext,
                    "sext" => CastOp::Sext,
                    "int2float" => CastOp::IntToFloat,
                    "float2float" => CastOp::FloatToFloat,
                    "trunc" => CastOp::Trunc,
                    _ => return Err(ParseError::new("invalid cast operation")),
                })
            }
            Rule::ty => ty = Some(parse_size_bytes(part.as_str(), "cast")?),
            Rule::typed_atom => src = Some(parse_typed_atom(part)?),
            _ => {}
        }
    }

    Ok(ExprNode::Cast {
        op: op.ok_or_else(|| ParseError::new("missing cast op"))?,
        size_bytes: ty.ok_or_else(|| ParseError::new("missing cast type"))?,
        src: src.ok_or_else(|| ParseError::new("missing cast source"))?,
    })
}

fn parse_typed_atom(pair: Pair<'_, Rule>) -> Result<TypedAtom, ParseError> {
    let mut size_bytes = None;
    let mut atom = None;

    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::ty => size_bytes = Some(parse_size_bytes(part.as_str(), "typed atom")?),
            Rule::atom => {
                atom = Some(parse_atom(part)?);
            }
            _ => {}
        }
    }

    Ok(TypedAtom {
        size_bytes,
        atom: atom.ok_or_else(|| ParseError::new("missing atom"))?,
    })
}

fn parse_atom(pair: Pair<'_, Rule>) -> Result<Atom, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::new("invalid atom"))?;

    match inner.as_rule() {
        Rule::capture => {
            let ident = inner
                .into_inner()
                .next()
                .ok_or_else(|| ParseError::new("invalid capture identifier"))?
                .as_str()
                .to_owned();
            Ok(Atom::External(ident))
        }
        Rule::ssa_name => Ok(Atom::Local(
            inner
                .as_str()
                .strip_prefix('%')
                .unwrap_or(inner.as_str())
                .to_owned(),
        )),
        Rule::ident => Ok(Atom::Local(inner.as_str().to_owned())),
        Rule::integer => parse_integer(inner.as_str()).map(Atom::Int),
        _ => Err(ParseError::new("invalid atom")),
    }
}

fn parse_integer(text: &str) -> Result<u64, ParseError> {
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        return u64::from_str_radix(hex, 16)
            .map_err(|_| ParseError::new("failed to parse integer literal"));
    }

    text.parse::<u64>()
        .map_err(|_| ParseError::new("failed to parse integer literal"))
}

fn parse_size_bytes(text: &str, context: &str) -> Result<usize, ParseError> {
    let bits = parse_size_bits(text).ok_or_else(|| {
        ParseError::new(format!(
            "invalid {context} size `{text}`; expected iNN or fNN"
        ))
    })?;

    if bits % 8 != 0 {
        return Err(ParseError::new(format!(
            "{context} size {text} is not byte-aligned"
        )));
    }

    Ok(bits / 8)
}

fn parse_size_bits(ident: &str) -> Option<usize> {
    if ident.len() < 2 {
        return None;
    }

    let mut chars = ident.chars();
    let prefix = chars.next()?;
    if prefix != 'i' && prefix != 'f' {
        return None;
    }

    let bits = chars.as_str();
    if bits.chars().all(|ch| ch.is_ascii_digit()) {
        bits.parse::<usize>().ok()
    } else {
        None
    }
}

fn to_parse_error(error: pest::error::Error<Rule>) -> ParseError {
    ParseError::new(format!("qcode parse error: {error}"))
}

#[cfg(test)]
mod tests {
    use super::parse_program;
    use crate::ast::{Atom, CastOp, ExprNode, Statement};

    #[test]
    fn parses_assignment_chain() {
        let statements = parse_program("tmp = {v1} + 3; tmp + 2").expect("parse should succeed");
        assert_eq!(statements.len(), 2);

        match &statements[0] {
            Statement::Assign { name, expr, .. } => {
                assert_eq!(name, "tmp");
                match expr {
                    ExprNode::Binary { lhs, op, rhs } => {
                        assert_eq!(op, "+");
                        match &lhs.atom {
                            Atom::External(name) => assert_eq!(name, "v1"),
                            _ => panic!("expected external lhs"),
                        }
                        match &rhs.atom {
                            Atom::Int(value) => assert_eq!(*value, 3),
                            _ => panic!("expected integer rhs"),
                        }
                    }
                    _ => panic!("expected binary expression"),
                }
            }
            _ => panic!("expected assignment statement"),
        }

        match &statements[1] {
            Statement::Expr(ExprNode::Binary { lhs, op, rhs }) => {
                assert_eq!(op, "+");
                match &lhs.atom {
                    Atom::Local(name) => assert_eq!(name, "tmp"),
                    _ => panic!("expected local lhs"),
                }
                match &rhs.atom {
                    Atom::Int(value) => assert_eq!(*value, 2),
                    _ => panic!("expected integer rhs"),
                }
            }
            _ => panic!("expected expression statement"),
        }
    }

    #[test]
    fn parses_local_declaration() {
        let statements = parse_program("local i64 ptr; ptr").expect("parse should succeed");
        assert_eq!(statements.len(), 2);

        match &statements[0] {
            Statement::LocalDecl {
                name,
                display_name,
                size_bytes,
            } => {
                assert_eq!(name, "ptr");
                assert_eq!(display_name, "ptr");
                assert_eq!(*size_bytes, 8);
            }
            _ => panic!("expected local declaration"),
        }

        match &statements[1] {
            Statement::Expr(ExprNode::Atom(atom)) => match &atom.atom {
                Atom::Local(name) => assert_eq!(name, "ptr"),
                _ => panic!("expected local atom"),
            },
            _ => panic!("expected local expression"),
        }
    }

    #[test]
    fn parses_local_declaration_with_display_name() {
        let statements = parse_program("local i64 ptr as PTR; ptr").expect("parse should succeed");
        assert_eq!(statements.len(), 2);

        match &statements[0] {
            Statement::LocalDecl {
                name,
                display_name,
                size_bytes,
            } => {
                assert_eq!(name, "ptr");
                assert_eq!(display_name, "PTR");
                assert_eq!(*size_bytes, 8);
            }
            _ => panic!("expected local declaration"),
        }
    }

    #[test]
    fn parses_cast_expression() {
        let statements = parse_program("zext(i32, {v1})").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Cast {
                op,
                size_bytes,
                src,
            }) => {
                assert!(matches!(op, CastOp::Zext));
                assert_eq!(*size_bytes, 4);
                match &src.atom {
                    Atom::External(name) => assert_eq!(name, "v1"),
                    _ => panic!("expected external cast source"),
                }
            }
            _ => panic!("expected cast expression"),
        }
    }

    #[test]
    fn parses_load_and_store_statements() {
        let statements =
            parse_program("load(i32, {ptr}); store({ptr}, {src})").expect("parse should succeed");
        assert_eq!(statements.len(), 2);

        match &statements[0] {
            Statement::Expr(ExprNode::Load { size_bytes, ptr }) => {
                assert_eq!(*size_bytes, 4);
                match &ptr.atom {
                    Atom::External(name) => assert_eq!(name, "ptr"),
                    _ => panic!("expected external load pointer"),
                }
            }
            _ => panic!("expected load statement"),
        }

        match &statements[1] {
            Statement::Expr(ExprNode::Store { ptr, src }) => {
                match &ptr.atom {
                    Atom::External(name) => assert_eq!(name, "ptr"),
                    _ => panic!("expected external store pointer"),
                }
                match &src.atom {
                    Atom::External(name) => assert_eq!(name, "src"),
                    _ => panic!("expected external store source"),
                }
            }
            _ => panic!("expected store statement"),
        }
    }

    #[test]
    fn parses_typed_binary_expression_statement() {
        let statements = parse_program("i32 {v0} + i32 0x2").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Binary { lhs, op, rhs }) => {
                assert_eq!(op, "+");
                assert_eq!(lhs.size_bytes, Some(4));
                assert_eq!(rhs.size_bytes, Some(4));
                match &lhs.atom {
                    Atom::External(name) => assert_eq!(name, "v0"),
                    _ => panic!("expected external lhs"),
                }
                match &rhs.atom {
                    Atom::Int(value) => assert_eq!(*value, 2),
                    _ => panic!("expected integer rhs"),
                }
            }
            _ => panic!("expected binary expression statement"),
        }
    }

    #[test]
    fn parses_unary_bool_not_expression_statement() {
        let statements = parse_program("!{v0}").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Unop { op, src }) => {
                assert_eq!(op, "!");
                assert_eq!(src.size_bytes, None);
                match &src.atom {
                    Atom::External(name) => assert_eq!(name, "v0"),
                    _ => panic!("expected external unary source"),
                }
            }
            _ => panic!("expected unary expression statement"),
        }
    }

    #[test]
    fn parses_bool_xor_expression_statement() {
        let statements = parse_program("{v0} ^^ {v1}").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Binary { op, .. }) => assert_eq!(op, "^^"),
            _ => panic!("expected binary expression statement"),
        }
    }

    #[test]
    fn parses_bool_and_expression_statement() {
        let statements = parse_program("{v0} && {v1}").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Binary { op, .. }) => assert_eq!(op, "&&"),
            _ => panic!("expected binary expression statement"),
        }
    }

    #[test]
    fn parses_bool_or_expression_statement() {
        let statements = parse_program("{v0} || {v1}").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Binary { op, .. }) => assert_eq!(op, "||"),
            _ => panic!("expected binary expression statement"),
        }
    }

    #[test]
    fn parses_float_negate_expression_statement() {
        let statements = parse_program("f-{v0}").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Unop { op, src }) => {
                assert_eq!(op, "f-");
                match &src.atom {
                    Atom::External(name) => assert_eq!(name, "v0"),
                    _ => panic!("expected external unary source"),
                }
            }
            _ => panic!("expected unary expression statement"),
        }
    }

    #[test]
    fn parses_float_abs_expression_statement() {
        let statements = parse_program("abs({v0})").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Unop { op, src }) => {
                assert_eq!(op, "abs");
                match &src.atom {
                    Atom::External(name) => assert_eq!(name, "v0"),
                    _ => panic!("expected external unary source"),
                }
            }
            _ => panic!("expected unary expression statement"),
        }
    }

    #[test]
    fn parses_float_binary_expression_statement() {
        let statements = parse_program("{v0} f+ {v1}").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Binary { op, .. }) => assert_eq!(op, "f+"),
            _ => panic!("expected binary expression statement"),
        }
    }

    #[test]
    fn parses_int_unary_negate_expression_statement() {
        let statements = parse_program("-{v0}").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Unop { op, src }) => {
                assert_eq!(op, "-");
                match &src.atom {
                    Atom::External(name) => assert_eq!(name, "v0"),
                    _ => panic!("expected external unary source"),
                }
            }
            _ => panic!("expected unary expression statement"),
        }
    }

    #[test]
    fn parses_int_signed_binary_expression_statement() {
        let statements = parse_program("{v0} s< {v1}").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Binary { op, .. }) => assert_eq!(op, "s<"),
            _ => panic!("expected binary expression statement"),
        }
    }

    #[test]
    fn parses_misc_single_arg_function_call() {
        let statements = parse_program("popcount({v0})").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::FuncCall { op, args }) => {
                assert_eq!(op, "popcount");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("expected function call expression statement"),
        }
    }

    #[test]
    fn parses_misc_two_arg_function_call() {
        let statements = parse_program("carry({v0}, {v1})").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::FuncCall { op, args }) => {
                assert_eq!(op, "carry");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("expected function call expression statement"),
        }
    }

    #[test]
    fn parses_ssa_assignment_chain() {
        let statements =
            parse_program("i64 %a = i64 1 + i64 2; i64 %b = i64 %a + i64 5")
                .expect("parse should succeed");
        assert_eq!(statements.len(), 2);

        match &statements[0] {
            Statement::Assign { name, expose, expr } => {
                assert_eq!(name, "a");
                assert!(expose);
                assert!(matches!(expr, ExprNode::Binary { .. }));
            }
            _ => panic!("expected ssa assignment"),
        }

        match &statements[1] {
            Statement::Assign { name, expose, expr } => {
                assert_eq!(name, "b");
                assert!(expose);
                match expr {
                    ExprNode::Binary { lhs, .. } => match &lhs.atom {
                        Atom::Local(name) => assert_eq!(name, "a"),
                        _ => panic!("expected local lhs referencing %a"),
                    },
                    _ => panic!("expected binary expression"),
                }
            }
            _ => panic!("expected ssa assignment"),
        }
    }

    #[test]
    fn parses_label_decl() {
        let statements = parse_program("<entry>").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::LabelDecl { name } => assert_eq!(name, "entry"),
            _ => panic!("expected label declaration"),
        }
    }

    #[test]
    fn parses_branch() {
        let statements = parse_program("goto <done>").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Branch { target } => assert_eq!(target, "done"),
            _ => panic!("expected branch statement"),
        }
    }

    #[test]
    fn parses_branchind() {
        let statements = parse_program("goto [{ptr}]").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::BranchInd { ptr } => match &ptr.atom {
                Atom::External(name) => assert_eq!(name, "ptr"),
                _ => panic!("expected external pointer"),
            },
            _ => panic!("expected branchind statement"),
        }
    }

    #[test]
    fn parses_cbranch() {
        let statements = parse_program("if {cond} goto <then_lbl> else goto <else_lbl>")
            .expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::CBranch {
                condition,
                target,
                fallthrough,
            } => {
                match &condition.atom {
                    Atom::External(name) => assert_eq!(name, "cond"),
                    _ => panic!("expected external condition"),
                }
                assert_eq!(target, "then_lbl");
                assert_eq!(fallthrough, "else_lbl");
            }
            _ => panic!("expected cbranch statement"),
        }
    }

    #[test]
    fn parses_call() {
        let statements = parse_program("call <target>").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Call { target } => assert_eq!(target, "target"),
            _ => panic!("expected call statement"),
        }
    }

    #[test]
    fn parses_callind() {
        let statements = parse_program("call [{ptr}]").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::CallInd { ptr } => match &ptr.atom {
                Atom::External(name) => assert_eq!(name, "ptr"),
                _ => panic!("expected external pointer"),
            },
            _ => panic!("expected callind statement"),
        }
    }

    #[test]
    fn parses_return() {
        let statements = parse_program("return [{ptr}]").expect("parse should succeed");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Return { ptr } => match &ptr.atom {
                Atom::External(name) => assert_eq!(name, "ptr"),
                _ => panic!("expected external pointer"),
            },
            _ => panic!("expected return statement"),
        }
    }

    #[test]
    fn parses_multi_block_program() {
        // Label prefixes the next statement without a `;` between them.
        let program = "tmp = {v1} + 3; goto <done>; <done> tmp + 2";
        let statements = parse_program(program).expect("parse should succeed");
        assert_eq!(statements.len(), 4);

        match &statements[0] {
            Statement::Assign { name, expose, .. } => {
                assert_eq!(name, "tmp");
                assert!(!expose);
            }
            _ => panic!("expected assignment"),
        }
        match &statements[1] {
            Statement::Branch { target } => assert_eq!(target, "done"),
            _ => panic!("expected branch"),
        }
        match &statements[2] {
            Statement::LabelDecl { name } => assert_eq!(name, "done"),
            _ => panic!("expected label declaration"),
        }
        match &statements[3] {
            Statement::Expr(ExprNode::Binary { op, .. }) => assert_eq!(op, "+"),
            _ => panic!("expected expression"),
        }
    }

    #[test]
    fn parses_standalone_label() {
        // A label with no following statement is valid (e.g. as a terminal label).
        let statements = parse_program("goto <end>; <end>").expect("parse should succeed");
        assert_eq!(statements.len(), 2);

        match &statements[1] {
            Statement::LabelDecl { name } => assert_eq!(name, "end"),
            _ => panic!("expected label declaration"),
        }
    }
}
