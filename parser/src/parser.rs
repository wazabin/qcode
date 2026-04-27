use crate::ast::{
    Atom, CastOp, ExprNode, FnDecl, Label, Program, SourcePosition, SourceSpan, Statement,
    TypedAtom,
};
use pest::Parser;
use pest::iterators::Pair;
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

pub fn parse_program(program: &str) -> Result<Program, ParseError> {
    let mut parsed = QCodeParser::parse(Rule::program, program).map_err(to_parse_error)?;
    let root = parsed
        .next()
        .ok_or_else(|| ParseError::new("missing program"))?;

    let mut fn_decls: Vec<FnDecl> = Vec::new();
    let mut statements: Vec<Statement> = Vec::new();
    let mut top_varnodes: Vec<Statement> = Vec::new();
    let mut is_fn_program = false;

    for pair in root.into_inner() {
        match pair.as_rule() {
            Rule::top_varnode_list => {
                for part in pair.into_inner() {
                    if part.as_rule() == Rule::local_decl {
                        top_varnodes.push(parse_local_decl(part)?);
                    }
                }
            }
            Rule::fn_decl => {
                is_fn_program = true;
                fn_decls.push(parse_fn_decl(pair)?);
            }
            Rule::statement_list => {
                for compound in pair.into_inner() {
                    if compound.as_rule() != Rule::compound_stmt {
                        continue;
                    }
                    parse_compound(compound, &mut statements)?;
                }
            }
            _ => {}
        }
    }

    if is_fn_program {
        Ok(Program::Functions {
            varnodes: top_varnodes,
            fns: fn_decls,
        })
    } else {
        Ok(Program::Statements(statements))
    }
}

fn parse_fn_decl(pair: Pair<'_, Rule>) -> Result<FnDecl, ParseError> {
    let span = source_span(pair.as_span());
    let mut inner = pair.into_inner();
    let name_pair = inner
        .next()
        .ok_or_else(|| ParseError::new("missing function name"))?;
    let name_span = source_span(name_pair.as_span());
    let name = name_pair.as_str().to_owned();

    let mut statements = Vec::new();
    for part in inner {
        if part.as_rule() == Rule::fn_body {
            for fn_stmt in part.into_inner() {
                if fn_stmt.as_rule() != Rule::fn_stmt {
                    continue;
                }
                for compound in fn_stmt.into_inner() {
                    if compound.as_rule() != Rule::compound_stmt {
                        continue;
                    }
                    parse_compound(compound, &mut statements)?;
                }
            }
        }
    }

    Ok(FnDecl {
        name,
        name_span,
        span,
        statements,
    })
}

fn parse_compound(pair: Pair<'_, Rule>, out: &mut Vec<Statement>) -> Result<(), ParseError> {
    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::label_decl => out.push(parse_label_decl(part)?),
            Rule::inner_stmt => parse_inner_stmt(part, out)?,
            _ => return Err(ParseError::new("unexpected compound statement")),
        }
    }
    Ok(())
}

fn parse_label_decl(pair: Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let span = source_span(pair.as_span());
    let mut inner = pair.into_inner();
    let name_pair = inner
        .next()
        .ok_or_else(|| ParseError::new("missing label name"))?;
    let label_span = source_span(name_pair.as_span());
    let label = match name_pair.as_rule() {
        Rule::ident => {
            let name = name_pair.as_str().to_owned();
            let params: Vec<String> = inner
                .filter(|p| p.as_rule() == Rule::block_param_name)
                .map(|p| {
                    p.as_str()
                        .strip_prefix('@')
                        .unwrap_or(p.as_str())
                        .to_owned()
                })
                .collect();
            Label::Named {
                name,
                params,
                span: label_span,
            }
        }
        Rule::integer => {
            let addr = parse_integer(name_pair.as_str())?;
            Label::Address {
                value: addr,
                span: label_span,
            }
        }
        _ => return Err(ParseError::new("invalid label declaration")),
    };
    Ok(Statement::LabelDecl { label, span })
}

fn parse_inner_stmt(pair: Pair<'_, Rule>, out: &mut Vec<Statement>) -> Result<(), ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::new("empty inner statement"))?;

    let stmt = match inner.as_rule() {
        Rule::local_decl => parse_local_decl(inner)?,
        Rule::assignment => parse_assignment(inner)?,
        Rule::terminator => parse_terminator(inner)?,
        Rule::expr => Statement::Expr(parse_expr(inner)?),
        _ => return Err(ParseError::new("unexpected inner statement")),
    };

    out.push(stmt);
    Ok(())
}

fn parse_local_decl(pair: Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let span = source_span(pair.as_span());
    let mut inner = pair.into_inner();
    let size_bytes = parse_size_bytes(
        inner
            .next()
            .ok_or_else(|| ParseError::new("missing local type"))?
            .as_str(),
        "local declaration",
    )?;
    let name_pair = inner
        .next()
        .ok_or_else(|| ParseError::new("missing local name"))?;
    let name_span = source_span(name_pair.as_span());
    let name = name_pair.as_str().to_owned();
    let display_name = inner
        .next()
        .map(|pair| pair.as_str().to_owned())
        .unwrap_or_else(|| name.clone());

    Ok(Statement::LocalDecl {
        name,
        name_span,
        display_name,
        size_bytes,
        span,
    })
}

/// Parses the content of a `<...>` label into a `Label`.
fn label_value(pair: Pair<'_, Rule>) -> Result<Label, ParseError> {
    let span = source_span(pair.as_span());
    match pair.as_rule() {
        Rule::ident => Ok(Label::Named {
            name: pair.as_str().to_owned(),
            params: vec![],
            span,
        }),
        Rule::integer => {
            let addr = parse_integer(pair.as_str())?;
            Ok(Label::Address { value: addr, span })
        }
        _ => Err(ParseError::new("invalid label content")),
    }
}

/// Parses a `branch_label` rule (`<(ident|int) branch_arg*>`) into a `Label` and args.
fn parse_branch_label(
    pair: Pair<'_, Rule>,
) -> Result<(Label, Vec<(String, TypedAtom)>), ParseError> {
    let mut inner = pair.into_inner();
    let target_pair = inner
        .next()
        .ok_or_else(|| ParseError::new("missing branch label target"))?;
    let label = label_value(target_pair)?;
    let mut args = Vec::new();
    for part in inner {
        if part.as_rule() == Rule::branch_arg {
            let mut arg_inner = part.into_inner();
            let name_pair = arg_inner
                .next()
                .ok_or_else(|| ParseError::new("missing branch arg name"))?;
            let name = name_pair
                .as_str()
                .strip_prefix('@')
                .unwrap_or(name_pair.as_str())
                .to_owned();
            let value_pair = arg_inner
                .next()
                .ok_or_else(|| ParseError::new("missing branch arg value"))?;
            args.push((name, parse_typed_atom(value_pair)?));
        }
    }
    Ok((label, args))
}

fn parse_terminator(pair: Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let span = source_span(pair.as_span());
    let specific = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::new("missing terminator kind"))?;

    match specific.as_rule() {
        Rule::branch_stmt => {
            let branch_label_pair = specific
                .into_inner()
                .next()
                .ok_or_else(|| ParseError::new("missing branch label"))?;
            let (target, args) = parse_branch_label(branch_label_pair)?;
            Ok(Statement::Branch { target, args, span })
        }

        Rule::branchind_stmt => {
            let mut inner = specific.into_inner();
            let ptr = parse_typed_atom(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing branchind pointer"))?,
            )?;
            Ok(Statement::BranchInd { ptr, span })
        }

        Rule::cbranch_stmt => {
            let mut inner = specific.into_inner();
            let condition = parse_typed_atom(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing cbranch condition"))?,
            )?;
            let target_pair = inner
                .next()
                .ok_or_else(|| ParseError::new("missing cbranch target"))?;
            let (target, target_args) = parse_branch_label(target_pair)?;
            let fallthrough_pair = inner
                .next()
                .ok_or_else(|| ParseError::new("missing cbranch fallthrough"))?;
            let (fallthrough, fallthrough_args) = parse_branch_label(fallthrough_pair)?;
            Ok(Statement::CBranch {
                condition,
                target,
                target_args,
                fallthrough,
                fallthrough_args,
                span,
            })
        }

        Rule::call_stmt => {
            let mut inner = specific.into_inner();
            let target = label_value(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing call target"))?
                    .into_inner()
                    .next()
                    .ok_or_else(|| ParseError::new("missing call target content"))?,
            )?;
            Ok(Statement::Call { target, span })
        }

        Rule::callind_stmt => {
            let mut inner = specific.into_inner();
            let ptr = parse_typed_atom(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing callind pointer"))?,
            )?;
            Ok(Statement::CallInd { ptr, span })
        }

        Rule::return_stmt => {
            let mut inner = specific.into_inner();
            let ptr = parse_typed_atom(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing return pointer"))?,
            )?;
            Ok(Statement::Return { ptr, span })
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
    let span = source_span(pair.as_span());
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
    let name_span = source_span(ssa_pair.as_span());
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
        name_span,
        expose: true,
        expr: parse_expr(expr_pair)?,
        span,
    })
}

fn parse_assignment_plain(pair: Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let span = source_span(pair.as_span());
    let mut inner = pair.into_inner();
    let name_pair = inner
        .next()
        .ok_or_else(|| ParseError::new("missing assignment name"))?;
    let name_span = source_span(name_pair.as_span());
    let name = name_pair.as_str().to_owned();
    let expr_pair = inner
        .find(|p| p.as_rule() == Rule::expr)
        .ok_or_else(|| ParseError::new("missing assignment expression"))?;
    Ok(Statement::Assign {
        name,
        name_span,
        expose: false,
        expr: parse_expr(expr_pair)?,
        span,
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
    let mut span = None;

    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::ty => size_bytes = Some(parse_size_bytes(part.as_str(), "typed atom")?),
            Rule::atom => {
                span = Some(source_span(part.as_span()));
                atom = Some(parse_atom(part)?);
            }
            _ => {}
        }
    }

    Ok(TypedAtom {
        size_bytes,
        atom: atom.ok_or_else(|| ParseError::new("missing atom"))?,
        span: span.ok_or_else(|| ParseError::new("missing atom span"))?,
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
        Rule::block_param_name => Ok(Atom::Local(
            inner
                .as_str()
                .strip_prefix('@')
                .unwrap_or(inner.as_str())
                .to_owned(),
        )),
        Rule::ssa_name => Ok(Atom::Local(
            inner
                .as_str()
                .strip_prefix('%')
                .unwrap_or(inner.as_str())
                .to_owned(),
        )),
        Rule::ident => Ok(Atom::Local(inner.as_str().to_owned())),
        Rule::addressof => {
            let name = inner
                .into_inner()
                .next()
                .ok_or_else(|| ParseError::new("invalid addressof: missing identifier"))?
                .as_str()
                .to_owned();
            Ok(Atom::AddressOf(name))
        }
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

fn source_span(span: pest::Span<'_>) -> SourceSpan {
    let start_pos = span.start_pos();
    let end_pos = span.end_pos();
    let (start_line, start_column) = start_pos.line_col();
    let (end_line, end_column) = end_pos.line_col();

    SourceSpan {
        start: SourcePosition {
            offset: span.start(),
            line: start_line,
            column: start_column,
        },
        end: SourcePosition {
            offset: span.end(),
            line: end_line,
            column: end_column,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::parse_program;
    use crate::ast::{Atom, CastOp, ExprNode, Label, Program, Statement};

    fn stmts(program: &str) -> Vec<Statement> {
        match parse_program(program).expect("parse should succeed") {
            Program::Statements(s) => s,
            Program::Functions { .. } => panic!("expected statements, got functions"),
        }
    }

    #[test]
    fn parses_assignment_chain() {
        let statements = stmts("tmp = {v1} + 3; tmp + 2");
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
        let statements = stmts("varnode i64 ptr; ptr");
        assert_eq!(statements.len(), 2);

        match &statements[0] {
            Statement::LocalDecl {
                name,
                display_name,
                size_bytes,
                name_span,
                span,
            } => {
                assert_eq!(name, "ptr");
                assert_eq!(display_name, "ptr");
                assert_eq!(*size_bytes, 8);
                assert_eq!(name_span.start.column, 13);
                assert_eq!(name_span.end.column, 16);
                assert_eq!(span.start.column, 1);
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
        let statements = stmts("varnode i64 ptr as PTR; ptr");
        assert_eq!(statements.len(), 2);

        match &statements[0] {
            Statement::LocalDecl {
                name,
                display_name,
                size_bytes,
                ..
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
        let statements = stmts("zext(i32, {v1})");
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
        let statements = stmts("load(i32, {ptr}); store({ptr}, {src})");
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
        let statements = stmts("i32 {v0} + i32 0x2");
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
        let statements = stmts("!{v0}");
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
        let statements = stmts("{v0} ^^ {v1}");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Binary { op, .. }) => assert_eq!(op, "^^"),
            _ => panic!("expected binary expression statement"),
        }
    }

    #[test]
    fn parses_bool_and_expression_statement() {
        let statements = stmts("{v0} && {v1}");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Binary { op, .. }) => assert_eq!(op, "&&"),
            _ => panic!("expected binary expression statement"),
        }
    }

    #[test]
    fn parses_bool_or_expression_statement() {
        let statements = stmts("{v0} || {v1}");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Binary { op, .. }) => assert_eq!(op, "||"),
            _ => panic!("expected binary expression statement"),
        }
    }

    #[test]
    fn parses_float_negate_expression_statement() {
        let statements = stmts("f-{v0}");
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
        let statements = stmts("abs({v0})");
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
        let statements = stmts("{v0} f+ {v1}");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Binary { op, .. }) => assert_eq!(op, "f+"),
            _ => panic!("expected binary expression statement"),
        }
    }

    #[test]
    fn parses_int_unary_negate_expression_statement() {
        let statements = stmts("-{v0}");
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
        let statements = stmts("{v0} s< {v1}");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Expr(ExprNode::Binary { op, .. }) => assert_eq!(op, "s<"),
            _ => panic!("expected binary expression statement"),
        }
    }

    #[test]
    fn parses_misc_single_arg_function_call() {
        let statements = stmts("popcount({v0})");
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
        let statements = stmts("carry({v0}, {v1})");
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
        let statements = stmts("i64 %a = i64 1 + i64 2; i64 %b = i64 %a + i64 5");
        assert_eq!(statements.len(), 2);

        match &statements[0] {
            Statement::Assign {
                name,
                expose,
                expr,
                name_span,
                span,
            } => {
                assert_eq!(name, "a");
                assert!(expose);
                assert!(matches!(expr, ExprNode::Binary { .. }));
                assert_eq!(name_span.start.column, 5);
                assert_eq!(name_span.end.column, 7);
                assert_eq!(span.start.column, 1);
            }
            _ => panic!("expected ssa assignment"),
        }

        match &statements[1] {
            Statement::Assign {
                name, expose, expr, ..
            } => {
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
    fn parses_label_decl_named() {
        let statements = stmts("<entry>");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::LabelDecl {
                label:
                    Label::Named {
                        name,
                        span: label_span,
                        ..
                    },
                span,
            } => {
                assert_eq!(name, "entry");
                assert_eq!(label_span.start.column, 2);
                assert_eq!(label_span.end.column, 7);
                assert_eq!(span.start.column, 1);
            }
            _ => panic!("expected named label declaration"),
        }
    }

    #[test]
    fn parses_label_decl_address() {
        let statements = stmts("<0x1000>");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::LabelDecl {
                label: Label::Address { value: addr, .. },
                ..
            } => assert_eq!(*addr, 0x1000),
            _ => panic!("expected address label declaration"),
        }
    }

    #[test]
    fn parses_branch_named() {
        let statements = stmts("goto <done>");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Branch {
                target: Label::Named { name, span, .. },
                ..
            } => {
                assert_eq!(name, "done");
                assert_eq!(span.start.column, 7);
            }
            _ => panic!("expected named branch statement"),
        }
    }

    #[test]
    fn parses_branch_address() {
        let statements = stmts("goto <0x1001>");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Branch {
                target: Label::Address { value: addr, .. },
                ..
            } => assert_eq!(*addr, 0x1001),
            _ => panic!("expected address branch statement"),
        }
    }

    #[test]
    fn parses_branchind() {
        let statements = stmts("goto [{ptr}]");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::BranchInd { ptr, .. } => match &ptr.atom {
                Atom::External(name) => assert_eq!(name, "ptr"),
                _ => panic!("expected external pointer"),
            },
            _ => panic!("expected branchind statement"),
        }
    }

    #[test]
    fn parses_cbranch() {
        let statements = stmts("if {cond} goto <then_lbl> else goto <else_lbl>");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::CBranch {
                condition,
                target,
                fallthrough,
                ..
            } => {
                match &condition.atom {
                    Atom::External(name) => assert_eq!(name, "cond"),
                    _ => panic!("expected external condition"),
                }
                assert!(matches!(target, Label::Named { name, .. } if name == "then_lbl"));
                assert!(matches!(fallthrough, Label::Named { name, .. } if name == "else_lbl"));
            }
            _ => panic!("expected cbranch statement"),
        }
    }

    #[test]
    fn parses_call() {
        let statements = stmts("call <target>");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Call {
                target: Label::Named { name, .. },
                ..
            } => assert_eq!(name, "target"),
            _ => panic!("expected call statement"),
        }
    }

    #[test]
    fn parses_callind() {
        let statements = stmts("call [{ptr}]");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::CallInd { ptr, .. } => match &ptr.atom {
                Atom::External(name) => assert_eq!(name, "ptr"),
                _ => panic!("expected external pointer"),
            },
            _ => panic!("expected callind statement"),
        }
    }

    #[test]
    fn parses_return() {
        let statements = stmts("return [{ptr}]");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Return { ptr, .. } => match &ptr.atom {
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
        let statements = stmts(program);
        assert_eq!(statements.len(), 4);

        match &statements[0] {
            Statement::Assign { name, expose, .. } => {
                assert_eq!(name, "tmp");
                assert!(!expose);
            }
            _ => panic!("expected assignment"),
        }
        match &statements[1] {
            Statement::Branch {
                target: Label::Named { name, .. },
                ..
            } => assert_eq!(name, "done"),
            _ => panic!("expected branch"),
        }
        match &statements[2] {
            Statement::LabelDecl {
                label: Label::Named { name, .. },
                ..
            } => assert_eq!(name, "done"),
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
        let statements = stmts("goto <end>; <end>");
        assert_eq!(statements.len(), 2);

        match &statements[1] {
            Statement::LabelDecl {
                label: Label::Named { name, .. },
                ..
            } => assert_eq!(name, "end"),
            _ => panic!("expected label declaration"),
        }
    }

    #[test]
    fn parses_fn_decl() {
        let program = parse_program("fn f: <entry> varnode i64 a; goto <done>; <done> a + 1")
            .expect("parse should succeed");
        match program {
            Program::Functions { fns, .. } => {
                assert_eq!(fns.len(), 1);
                let f = &fns[0];
                assert_eq!(f.name, "f");
                assert_eq!(f.name_span.start.column, 4);
                assert_eq!(f.statements.len(), 5);
                assert!(
                    matches!(&f.statements[0], Statement::LabelDecl { label: Label::Named { name, .. }, .. } if name == "entry")
                );
                assert!(
                    matches!(&f.statements[1], Statement::LocalDecl { name, .. } if name == "a")
                );
                assert!(
                    matches!(&f.statements[2], Statement::Branch { target: Label::Named { name, .. }, .. } if name == "done")
                );
                assert!(
                    matches!(&f.statements[3], Statement::LabelDecl { label: Label::Named { name, .. }, .. } if name == "done")
                );
                assert!(
                    matches!(&f.statements[4], Statement::Expr(ExprNode::Binary { op, .. }) if op == "+")
                );
            }
            _ => panic!("expected function program"),
        }
    }

    #[test]
    fn parses_fn_decl_with_address_labels() {
        let program = parse_program("fn f: <entry> varnode i64 a; goto <0x1001>")
            .expect("parse should succeed");
        match program {
            Program::Functions { fns, .. } => {
                assert_eq!(fns.len(), 1);
                let f = &fns[0];
                assert!(matches!(
                    &f.statements[2],
                    Statement::Branch {
                        target: Label::Address { value: 0x1001, .. },
                        ..
                    }
                ));
            }
            _ => panic!("expected function program"),
        }
    }

    #[test]
    fn parses_label_decl_with_params() {
        let statements = stmts("<entry @v1 @v2>");
        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::LabelDecl {
                label: Label::Named { name, params, .. },
                ..
            } => {
                assert_eq!(name, "entry");
                assert_eq!(params, &["v1", "v2"]);
            }
            _ => panic!("expected named label declaration with params"),
        }
    }

    #[test]
    fn parses_branch_with_args() {
        let statements = stmts("goto <done @v1=1 @v2=%x>");
        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::Branch {
                target: Label::Named { name, .. },
                args,
                ..
            } => {
                assert_eq!(name, "done");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0].0, "v1");
                assert!(matches!(args[0].1.atom, Atom::Int(1)));
                assert_eq!(args[1].0, "v2");
                assert!(matches!(&args[1].1.atom, Atom::Local(n) if n == "x"));
            }
            _ => panic!("expected branch with args"),
        }
    }

    #[test]
    fn parses_cbranch_with_args() {
        let statements = stmts("if %c goto <then_lbl @x=1> else goto <else_lbl @y=%v>");
        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::CBranch {
                target,
                target_args,
                fallthrough,
                fallthrough_args,
                ..
            } => {
                assert!(matches!(target, Label::Named { name, .. } if name == "then_lbl"));
                assert_eq!(target_args.len(), 1);
                assert_eq!(target_args[0].0, "x");
                assert!(matches!(target_args[0].1.atom, Atom::Int(1)));
                assert!(matches!(fallthrough, Label::Named { name, .. } if name == "else_lbl"));
                assert_eq!(fallthrough_args.len(), 1);
                assert_eq!(fallthrough_args[0].0, "y");
                assert!(matches!(&fallthrough_args[0].1.atom, Atom::Local(n) if n == "v"));
            }
            _ => panic!("expected cbranch with args"),
        }
    }

    #[test]
    fn parses_top_level_varnode_before_fn() {
        let program = parse_program("varnode i64 ptr; fn f: <entry> return [ptr]")
            .expect("parse should succeed");
        match program {
            Program::Functions { varnodes, fns } => {
                assert_eq!(varnodes.len(), 1);
                assert!(
                    matches!(&varnodes[0], Statement::LocalDecl { name, size_bytes, .. } if name == "ptr" && *size_bytes == 8)
                );
                assert_eq!(fns.len(), 1);
            }
            _ => panic!("expected function program"),
        }
    }
}
