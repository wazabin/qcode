use crate::ast::{
    Atom, BlockParamDecl, Callee, CastOp, ExprNode, ExtractField, FnDecl, FnKind, GepField, Label,
    Program, ProgramKind, SourcePosition, SourceSpan, Statement, StructDecl, StructFieldDecl,
    StructFieldType, TupleField, TypedAtom,
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
    let mut structs: Vec<StructDecl> = Vec::new();
    let mut is_fn_program = false;

    // A comment may appear at the program level (before statement_list or fn_decl)
    // or between elements inside those rules.
    let mut pending_comment: Option<String> = None;

    for pair in root.into_inner() {
        match pair.as_rule() {
            Rule::COMMENT => {
                pending_comment = Some(comment_text(pair.as_str()));
            }
            Rule::struct_decl => {
                pending_comment = None;
                structs.push(parse_struct_decl(pair)?);
            }
            Rule::top_varnode_list => {
                for part in pair.into_inner() {
                    if part.as_rule() == Rule::local_decl {
                        top_varnodes.push(parse_local_decl(part)?);
                    }
                }
            }
            Rule::fn_decl => {
                is_fn_program = true;
                pending_comment = None; // comments before fn_decl are not yet attached
                fn_decls.push(parse_fn_decl(pair)?);
            }
            Rule::statement_list => {
                // The pending_comment (if any) was outside statement_list; treat it as
                // preceding the first compound_stmt.
                let mut inner_comment = pending_comment.take();
                for item in pair.into_inner() {
                    match item.as_rule() {
                        Rule::COMMENT => {
                            inner_comment = Some(comment_text(item.as_str()));
                        }
                        Rule::compound_stmt => {
                            let mut compound_stmts = Vec::new();
                            parse_compound(item, &mut compound_stmts)?;
                            attach_comment(&mut inner_comment, &mut compound_stmts);
                            statements.extend(compound_stmts);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    let kind = if is_fn_program {
        ProgramKind::Functions {
            varnodes: top_varnodes,
            fns: fn_decls,
        }
    } else {
        ProgramKind::Statements(statements)
    };
    Ok(Program { structs, kind })
}

fn parse_struct_decl(pair: Pair<'_, Rule>) -> Result<StructDecl, ParseError> {
    let span = source_span(pair.as_span());
    let mut inner = pair.into_inner();
    let name = inner
        .next()
        .ok_or_else(|| ParseError::new("missing struct name"))?
        .as_str()
        .to_owned();
    let mut fields = Vec::new();
    for field in inner {
        if field.as_rule() != Rule::struct_field {
            continue;
        }
        let mut parts = field.into_inner();
        let field_name = parts
            .next()
            .ok_or_else(|| ParseError::new("missing struct field name"))?
            .as_str()
            .to_owned();
        let ty_pair = parts
            .next()
            .ok_or_else(|| ParseError::new("missing struct field type"))?;
        let ty = match ty_pair.as_rule() {
            Rule::struct_ptr_ty => {
                StructFieldType::StructPtr(ty_pair.as_str().trim_end_matches('*').to_owned())
            }
            Rule::integer => StructFieldType::Int(parse_integer(ty_pair.as_str())? as usize),
            _ => return Err(ParseError::new("invalid struct field type")),
        };
        fields.push(StructFieldDecl {
            name: field_name,
            ty,
        });
    }
    Ok(StructDecl { name, fields, span })
}

fn parse_fn_decl(pair: Pair<'_, Rule>) -> Result<FnDecl, ParseError> {
    let span = source_span(pair.as_span());
    let mut inner = pair.into_inner();
    let kind_pair = inner
        .next()
        .ok_or_else(|| ParseError::new("missing function kind"))?;
    let kind = match kind_pair.as_str() {
        "fn" => FnKind::Machine,
        "lambda" => FnKind::Lambda,
        _ => return Err(ParseError::new("invalid function kind")),
    };
    let name_pair = inner
        .next()
        .ok_or_else(|| ParseError::new("missing function name"))?;
    let name_span = source_span(name_pair.as_span());
    let name = name_pair.as_str().to_owned();

    let mut statements = Vec::new();
    for part in inner {
        if part.as_rule() == Rule::fn_body {
            let mut pending_comment: Option<String> = None;
            for fn_stmt in part.into_inner() {
                match fn_stmt.as_rule() {
                    Rule::COMMENT => {
                        pending_comment = Some(comment_text(fn_stmt.as_str()));
                    }
                    Rule::fn_stmt => {
                        for compound in fn_stmt.into_inner() {
                            if compound.as_rule() != Rule::compound_stmt {
                                continue;
                            }
                            let mut compound_stmts = Vec::new();
                            parse_compound(compound, &mut compound_stmts)?;
                            attach_comment(&mut pending_comment, &mut compound_stmts);
                            statements.extend(compound_stmts);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(FnDecl {
        kind,
        name,
        name_span,
        span,
        statements,
    })
}

fn parse_compound(pair: Pair<'_, Rule>, out: &mut Vec<Statement>) -> Result<(), ParseError> {
    let mut pending_comment: Option<String> = None;
    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::label_decl => out.push(parse_label_decl(part)?),
            Rule::inner_stmt => {
                let mut stmts = Vec::new();
                parse_inner_stmt(part, &mut stmts)?;
                attach_comment(&mut pending_comment, &mut stmts);
                out.extend(stmts);
            }
            Rule::COMMENT => {
                pending_comment = Some(comment_text(part.as_str()));
            }
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
            let params: Vec<BlockParamDecl> = inner
                .filter(|p| p.as_rule() == Rule::block_param_decl)
                .map(parse_block_param_decl)
                .collect::<Result<_, _>>()?;
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

fn parse_block_param_decl(pair: Pair<'_, Rule>) -> Result<BlockParamDecl, ParseError> {
    let mut name = None;
    let mut size_bytes = None;

    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::block_param_name => {
                name = Some(
                    part.as_str()
                        .strip_prefix('@')
                        .unwrap_or(part.as_str())
                        .to_owned(),
                );
            }
            Rule::ty => {
                size_bytes = Some(parse_size_bytes(part.as_str(), "block parameter")?);
            }
            _ => {}
        }
    }

    Ok(BlockParamDecl {
        name: name.ok_or_else(|| ParseError::new("missing block parameter name"))?,
        size_bytes,
    })
}

fn parse_inner_stmt(pair: Pair<'_, Rule>, out: &mut Vec<Statement>) -> Result<(), ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::new("empty inner statement"))?;

    let stmt = match inner.as_rule() {
        Rule::local_decl => parse_local_decl(inner)?,
        Rule::assignment_ssa => parse_assignment_ssa(inner)?,
        Rule::terminator => parse_terminator(inner)?,
        Rule::assert_stmt => parse_assert_stmt(inner)?,
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

/// Parses an `edge_hint` rule (`// -> <a>, <b>`) into its list of target labels.
fn parse_edge_hint(pair: Pair<'_, Rule>) -> Result<Vec<Label>, ParseError> {
    let mut targets = Vec::new();
    for label_pair in pair.into_inner() {
        if label_pair.as_rule() == Rule::label {
            let content = label_pair
                .into_inner()
                .next()
                .ok_or_else(|| ParseError::new("missing edge-hint label content"))?;
            targets.push(label_value(content)?);
        }
    }
    Ok(targets)
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
            let targets = match inner.next() {
                Some(hint) if hint.as_rule() == Rule::edge_hint => parse_edge_hint(hint)?,
                _ => Vec::new(),
            };
            Ok(Statement::BranchInd { ptr, targets, span })
        }

        Rule::switch_stmt => {
            let mut inner = specific.into_inner();
            let scrutinee = parse_typed_atom(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing switch scrutinee"))?,
            )?;
            let mut cases = Vec::new();
            let mut default = None;
            for arm in inner {
                let arm = arm
                    .into_inner()
                    .next()
                    .ok_or_else(|| ParseError::new("empty switch arm"))?;
                match arm.as_rule() {
                    Rule::switch_case => {
                        let mut parts = arm.into_inner();
                        let value = parse_integer(
                            parts
                                .next()
                                .ok_or_else(|| ParseError::new("missing switch case value"))?
                                .as_str(),
                        )?;
                        let (target, args) = parse_branch_label(
                            parts
                                .next()
                                .ok_or_else(|| ParseError::new("missing switch case target"))?,
                        )?;
                        cases.push((value, target, args));
                    }
                    Rule::switch_default => {
                        let target = arm
                            .into_inner()
                            .next()
                            .ok_or_else(|| ParseError::new("missing switch default target"))?;
                        default = Some(parse_branch_label(target)?);
                    }
                    other => {
                        return Err(ParseError::new(format!("unexpected switch arm {other:?}")));
                    }
                }
            }
            Ok(Statement::Switch {
                scrutinee,
                cases,
                default,
                span,
            })
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
            let mut specific_inner = specific.into_inner();
            let form = specific_inner
                .next()
                .ok_or_else(|| ParseError::new("missing call form"))?;
            let mut target = None;
            let mut args = Vec::new();
            match form.as_rule() {
                Rule::call_direct => {
                    for part in form.into_inner() {
                        match part.as_rule() {
                            Rule::ident | Rule::minted_callee => target = Some(parse_callee(part)?),
                            Rule::call_arg => {
                                let mut inner = part.into_inner();
                                let name = inner
                                    .next()
                                    .ok_or_else(|| ParseError::new("missing call arg name"))?
                                    .as_str()
                                    .to_owned();
                                let atom =
                                    parse_typed_atom(inner.next().ok_or_else(|| {
                                        ParseError::new("missing call arg value")
                                    })?)?;
                                args.push((name, atom));
                            }
                            _ => {}
                        }
                    }
                }
                Rule::call_legacy => {
                    let label = label_value(
                        form.into_inner()
                            .next()
                            .ok_or_else(|| ParseError::new("missing call target"))?
                            .into_inner()
                            .next()
                            .ok_or_else(|| ParseError::new("missing call target content"))?,
                    )?;
                    match label {
                        Label::Named { name, .. } => target = Some(Callee::Named(name)),
                        Label::Address { .. } => {
                            return Err(ParseError::new(
                                "call with address target is not supported",
                            ));
                        }
                    }
                }
                _ => {}
            }
            let targets = match specific_inner.next() {
                Some(hint) if hint.as_rule() == Rule::edge_hint => parse_edge_hint(hint)?,
                _ => Vec::new(),
            };
            Ok(Statement::Call {
                target: target.ok_or_else(|| ParseError::new("missing call target"))?,
                tail: false,
                args,
                targets,
                span,
            })
        }

        Rule::tailcall_stmt => {
            let mut inner = specific.into_inner();
            let target = parse_callee(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing tailcall target"))?,
            )?;
            let args = inner
                .filter(|part| part.as_rule() == Rule::typed_atom)
                .map(parse_typed_atom)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .enumerate()
                .map(|(i, atom)| (format!("arg{i}"), atom))
                .collect();
            Ok(Statement::Call {
                target,
                tail: true,
                args,
                targets: Vec::new(),
                span,
            })
        }

        Rule::callind_stmt => {
            let mut inner = specific.into_inner();
            let ptr = parse_typed_atom(
                inner
                    .next()
                    .ok_or_else(|| ParseError::new("missing callind pointer"))?,
            )?;
            let mut args = Vec::new();
            let mut targets = Vec::new();
            for part in inner {
                match part.as_rule() {
                    Rule::callind_args => {
                        for atom in part.into_inner() {
                            args.push(parse_typed_atom(atom)?);
                        }
                    }
                    Rule::edge_hint => targets = parse_edge_hint(part)?,
                    _ => {}
                }
            }
            Ok(Statement::CallInd {
                ptr,
                args,
                targets,
                span,
            })
        }

        Rule::badinsn_stmt => Ok(Statement::BadInsn { span }),

        Rule::return_stmt => {
            let mut inner = specific.into_inner();
            let ret = inner
                .next()
                .ok_or_else(|| ParseError::new("missing return body"))?;
            match ret.as_rule() {
                Rule::return_at_stmt => {
                    let ptr = parse_typed_atom(
                        ret.into_inner()
                            .find(|p| p.as_rule() == Rule::typed_atom)
                            .ok_or_else(|| ParseError::new("missing return pointer"))?,
                    )?;
                    Ok(Statement::Return {
                        ptr,
                        value: None,
                        span,
                    })
                }
                Rule::return_value_at_stmt => {
                    let mut atoms = ret.into_inner().filter(|p| p.as_rule() == Rule::typed_atom);
                    let value = parse_typed_atom(
                        atoms
                            .next()
                            .ok_or_else(|| ParseError::new("missing return value"))?,
                    )?;
                    let ptr = parse_typed_atom(
                        atoms
                            .next()
                            .ok_or_else(|| ParseError::new("missing return pointer"))?,
                    )?;
                    Ok(Statement::Return {
                        ptr,
                        value: Some(value),
                        span,
                    })
                }
                Rule::return_value_stmt => {
                    let value = parse_typed_atom(
                        ret.into_inner()
                            .find(|p| p.as_rule() == Rule::typed_atom)
                            .ok_or_else(|| ParseError::new("missing return value"))?,
                    )?;
                    Ok(Statement::ReturnValue { value, span })
                }
                _ => Err(ParseError::new("invalid return")),
            }
        }

        _ => Err(ParseError::new("invalid terminator")),
    }
}

fn parse_assert_stmt(pair: Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let span = source_span(pair.as_span());
    let mut inner = pair.into_inner();
    let condition = parse_typed_atom(
        inner
            .next()
            .ok_or_else(|| ParseError::new("missing assert condition"))?,
    )?;
    Ok(Statement::Assert { condition, span })
}

fn parse_assignment_ssa(pair: Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let span = source_span(pair.as_span());
    let mut inner = pair.into_inner();
    // Optional declared type (`decl_ty`): an `iN`/`fN` size, or a `Foo*` struct
    // pointer. Only the struct-pointer form is recorded (it retypes the result).
    let name_or_ty = inner
        .next()
        .ok_or_else(|| ParseError::new("missing ssa assignment name"))?;
    let (decl_struct_ptr, ssa_pair) = if name_or_ty.as_rule() == Rule::decl_ty {
        let decl = name_or_ty
            .into_inner()
            .next()
            .ok_or_else(|| ParseError::new("empty declared type"))?;
        let decl_struct_ptr = match decl.as_rule() {
            Rule::struct_ptr_ty => Some(decl.as_str().trim_end_matches('*').to_owned()),
            _ => None,
        };
        (
            decl_struct_ptr,
            inner
                .next()
                .ok_or_else(|| ParseError::new("missing ssa name after type"))?,
        )
    } else {
        (None, name_or_ty)
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
        expr: parse_expr(expr_pair)?,
        decl_struct_ptr,
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
        Rule::intrinsic_call => parse_intrinsic_call(inner),
        Rule::apply => parse_apply(inner),
        Rule::scan => parse_scan(inner),
        Rule::map => parse_map(inner),
        Rule::binary => parse_binary(inner),
        Rule::memory => parse_memory(inner),
        Rule::cast => parse_cast(inner),
        Rule::tuple => parse_tuple(inner),
        Rule::extract => parse_extract(inner),
        Rule::gep => parse_gep(inner),
        Rule::range => parse_range(inner),
        _ => Err(ParseError::new("invalid expression")),
    }
}

fn parse_apply(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let mut target = None;
    let mut args = Vec::new();

    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::ident | Rule::minted_callee => target = Some(parse_callee(part)?),
            Rule::typed_atom => args.push(parse_typed_atom(part)?),
            _ => {}
        }
    }

    Ok(ExprNode::Apply {
        target: target.ok_or_else(|| ParseError::new("missing apply target"))?,
        args,
    })
}

fn parse_callee(pair: Pair<'_, Rule>) -> Result<Callee, ParseError> {
    match pair.as_rule() {
        Rule::ident => Ok(Callee::Named(pair.as_str().to_owned())),
        Rule::minted_callee => {
            let slot = pair
                .as_str()
                .strip_prefix("<minted:")
                .and_then(|text| text.strip_suffix('>'))
                .ok_or_else(|| ParseError::new("invalid minted callee"))?
                .parse::<u32>()
                .map_err(|_| ParseError::new("minted callee slot exceeds u32"))?;
            Ok(Callee::Minted(slot))
        }
        _ => Err(ParseError::new("invalid callee")),
    }
}

fn parse_range(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let mut src = None;
    let mut start = None;
    let mut end = None;
    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::typed_atom => src = Some(parse_typed_atom(part)?),
            Rule::range_start => start = Some(parse_integer(part.as_str().trim())?),
            Rule::range_end => end = Some(parse_integer(part.as_str().trim())?),
            _ => {}
        }
    }
    let src = src.ok_or_else(|| ParseError::new("missing range source"))?;
    Ok(ExprNode::Range { src, start, end })
}

fn parse_gep(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::new("missing gep body"))?;
    let rule = inner.as_rule();
    let mut parts = inner.into_inner();
    let base = parse_typed_atom(
        parts
            .find(|p| p.as_rule() == Rule::typed_atom)
            .ok_or_else(|| ParseError::new("missing gep base"))?,
    )?;
    let field = match rule {
        Rule::named_gep => GepField::Name(
            parts
                .find(|p| p.as_rule() == Rule::ident)
                .ok_or_else(|| ParseError::new("missing gep field name"))?
                .as_str()
                .to_owned(),
        ),
        Rule::offset_gep => GepField::Offset(parse_integer(
            parts
                .find(|p| p.as_rule() == Rule::integer)
                .ok_or_else(|| ParseError::new("missing gep offset"))?
                .as_str(),
        )?),
        _ => return Err(ParseError::new("invalid gep")),
    };
    Ok(ExprNode::Gep { base, field })
}

fn parse_tuple(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::new("missing tuple body"))?;
    let fields = match inner.as_rule() {
        Rule::pack_tuple => inner
            .into_inner()
            .filter(|p| p.as_rule() == Rule::tuple_field)
            .map(|field| {
                let mut parts = field.into_inner();
                let name = parts
                    .find(|p| p.as_rule() == Rule::ident)
                    .ok_or_else(|| ParseError::new("missing tuple field name"))?
                    .as_str()
                    .to_owned();
                let value = parse_typed_atom(
                    parts
                        .find(|p| p.as_rule() == Rule::typed_atom)
                        .ok_or_else(|| ParseError::new("missing tuple field value"))?,
                )?;
                Ok(TupleField {
                    name: Some(name),
                    value,
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        Rule::positional_tuple => inner
            .into_inner()
            .filter(|p| p.as_rule() == Rule::typed_atom)
            .enumerate()
            .map(|(i, value)| {
                Ok(TupleField {
                    name: Some(format!("field{}", i + 1)),
                    value: parse_typed_atom(value)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err(ParseError::new("invalid tuple")),
    };
    if fields.is_empty() {
        return Err(ParseError::new("tuple must have at least one field"));
    }
    Ok(ExprNode::Tuple { fields })
}

fn parse_extract(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::new("missing extract body"))?;
    let rule = inner.as_rule();
    let mut parts = inner.into_inner();
    let agg = parse_typed_atom(
        parts
            .find(|p| p.as_rule() == Rule::typed_atom)
            .ok_or_else(|| ParseError::new("missing extract aggregate"))?,
    )?;
    let field = match rule {
        Rule::named_extract => ExtractField::Name(
            parts
                .find(|p| p.as_rule() == Rule::ident)
                .ok_or_else(|| ParseError::new("missing extract field name"))?
                .as_str()
                .to_owned(),
        ),
        Rule::indexed_extract => ExtractField::Index(parse_integer(
            parts
                .find(|p| p.as_rule() == Rule::integer)
                .ok_or_else(|| ParseError::new("missing extract index"))?
                .as_str(),
        )?),
        _ => return Err(ParseError::new("invalid extract")),
    };
    Ok(ExprNode::Extract { agg, field })
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

fn parse_intrinsic_call(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let mut name = None;
    let mut args = Vec::new();

    for part in pair.into_inner() {
        match part.as_rule() {
            // Strip the leading `$` sigil.
            Rule::intrinsic_name => name = Some(part.as_str()[1..].to_owned()),
            Rule::typed_atom => args.push(parse_typed_atom(part)?),
            _ => {}
        }
    }

    Ok(ExprNode::Intrinsic {
        name: name.ok_or_else(|| ParseError::new("missing intrinsic name"))?,
        args,
    })
}

fn parse_map(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let mut inner = pair.into_inner();
    // `map_app` holds the body callee and any parenthesized captures.
    let app = inner
        .next()
        .filter(|p| p.as_rule() == Rule::map_app)
        .ok_or_else(|| ParseError::new("missing map body"))?;
    let mut app_parts = app.into_inner();
    let body = app_parts
        .next()
        .ok_or_else(|| ParseError::new("missing map body function"))?;
    let body = parse_callee(body)?;
    let captures = app_parts
        .filter(|p| p.as_rule() == Rule::typed_atom)
        .map(parse_typed_atom)
        .collect::<Result<Vec<_>, _>>()?;
    let src = parse_typed_atom(
        inner
            .find(|p| p.as_rule() == Rule::typed_atom)
            .ok_or_else(|| ParseError::new("missing map source"))?,
    )?;
    Ok(ExprNode::Map {
        body,
        src,
        captures,
    })
}

fn parse_scan(pair: Pair<'_, Rule>) -> Result<ExprNode, ParseError> {
    let mut inner = pair.into_inner();
    // `scan_app` holds the `@body` symbol and any parenthesized captures.
    let app = inner
        .next()
        .filter(|p| p.as_rule() == Rule::scan_app)
        .ok_or_else(|| ParseError::new("missing scan body"))?;
    let mut app_parts = app.into_inner();
    let body = app_parts
        .next()
        .ok_or_else(|| ParseError::new("missing scan body function"))?;
    let body = match body.as_rule() {
        Rule::scan_body => Callee::Named(body.as_str().trim_start_matches('@').to_owned()),
        Rule::minted_callee => parse_callee(body)?,
        _ => return Err(ParseError::new("invalid scan body function")),
    };
    let captures = app_parts
        .filter(|p| p.as_rule() == Rule::typed_atom)
        .map(parse_typed_atom)
        .collect::<Result<Vec<_>, _>>()?;
    // After `scan_app` come two `typed_atom`s: the initial accumulator and the
    // scanned source array, in that order.
    let mut atoms = inner.filter(|p| p.as_rule() == Rule::typed_atom);
    let init = parse_typed_atom(
        atoms
            .next()
            .ok_or_else(|| ParseError::new("missing scan init"))?,
    )?;
    let src = parse_typed_atom(
        atoms
            .next()
            .ok_or_else(|| ParseError::new("missing scan source"))?,
    )?;
    Ok(ExprNode::Scan {
        body,
        init,
        src,
        captures,
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
            let mut space = None;
            let mut size_bytes = None;
            let mut ptr = None;
            for part in inner.into_inner() {
                match part.as_rule() {
                    Rule::mem_loc => {
                        let (name, bytes) = parse_mem_loc(part)?;
                        space = Some(name);
                        size_bytes = Some(bytes);
                    }
                    Rule::typed_atom => ptr = Some(parse_typed_atom(part)?),
                    _ => {}
                }
            }
            Ok(ExprNode::Load {
                space: space.ok_or_else(|| ParseError::new("missing load space"))?,
                size_bytes: size_bytes.ok_or_else(|| ParseError::new("missing load size"))?,
                ptr: ptr.ok_or_else(|| ParseError::new("missing load pointer"))?,
            })
        }
        Rule::store => {
            let mut space = None;
            let mut size_bytes = None;
            let mut atoms = Vec::new();
            for part in inner.into_inner() {
                match part.as_rule() {
                    Rule::mem_loc => {
                        let (name, bytes) = parse_mem_loc(part)?;
                        space = Some(name);
                        size_bytes = Some(bytes);
                    }
                    Rule::typed_atom => atoms.push(parse_typed_atom(part)?),
                    _ => {}
                }
            }
            let mut atoms = atoms.into_iter();
            let ptr = atoms
                .next()
                .ok_or_else(|| ParseError::new("missing store pointer"))?;
            let src = atoms
                .next()
                .ok_or_else(|| ParseError::new("missing store source"))?;
            Ok(ExprNode::Store {
                space: space.ok_or_else(|| ParseError::new("missing store space"))?,
                size_bytes: size_bytes.ok_or_else(|| ParseError::new("missing store size"))?,
                ptr,
                src,
            })
        }
        _ => Err(ParseError::new("invalid memory expression")),
    }
}

/// Parse a `space:bytes` memory location into `(space_name, byte_size)`.
fn parse_mem_loc(pair: Pair<'_, Rule>) -> Result<(String, usize), ParseError> {
    let mut name = None;
    let mut bytes = None;
    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::mem_space => name = Some(part.as_str().to_owned()),
            Rule::integer => bytes = Some(parse_integer(part.as_str())? as usize),
            _ => {}
        }
    }
    Ok((
        name.ok_or_else(|| ParseError::new("missing space name"))?,
        bytes.ok_or_else(|| ParseError::new("missing space size"))?,
    ))
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
        Rule::block_param_name => Ok(Atom::BlockParam(
            inner
                .as_str()
                .strip_prefix('@')
                .unwrap_or(inner.as_str())
                .to_owned(),
        )),
        Rule::ssa_name => Ok(Atom::Ssa(
            inner
                .as_str()
                .strip_prefix('%')
                .unwrap_or(inner.as_str())
                .to_owned(),
        )),
        Rule::ident => Ok(Atom::Varnode(inner.as_str().to_owned())),
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
        Rule::bool_lit => Ok(Atom::Bool(inner.as_str() == "true")),
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
    if text == "bool" {
        return Ok(1);
    }
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

fn comment_text(raw: &str) -> String {
    raw.strip_prefix('#').unwrap_or("").trim().to_owned()
}

fn attach_comment(pending: &mut Option<String>, stmts: &mut Vec<Statement>) {
    if let Some(comment) = pending.take()
        && !stmts.is_empty()
    {
        let first = stmts.remove(0);
        stmts.insert(
            0,
            Statement::Commented {
                comment,
                inner: Box::new(first),
            },
        );
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
    use crate::ast::{Atom, Callee, CastOp, ExprNode, Label, ProgramKind, Statement};

    fn stmts(program: &str) -> Vec<Statement> {
        match parse_program(program).expect("parse should succeed").kind {
            ProgramKind::Statements(s) => s,
            ProgramKind::Functions { .. } => panic!("expected statements, got functions"),
        }
    }

    #[test]
    fn parses_ssa_assignment_and_use() {
        let statements = stmts("%tmp = {v1} + 3; %tmp + 2");
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
                    Atom::Ssa(name) => assert_eq!(name, "tmp"),
                    _ => panic!("expected ssa lhs"),
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
    fn parses_range_with_and_without_default_bounds() {
        // Explicit `[start:end]`, defaulted start `[:end]`, defaulted end
        // `[start:]`, and fully defaulted `[:]`.
        let cases = [
            ("%r = %a[1:4]", Some(1), Some(4)),
            ("%r = %a[:4]", None, Some(4)),
            ("%r = %a[1:]", Some(1), None),
            ("%r = %a[:]", None, None),
        ];
        for (src, want_start, want_end) in cases {
            match &stmts(src)[0] {
                Statement::Assign {
                    expr:
                        ExprNode::Range {
                            start,
                            end,
                            src: atom,
                        },
                    ..
                } => {
                    assert_eq!(*start, want_start, "start for `{src}`");
                    assert_eq!(*end, want_end, "end for `{src}`");
                    match &atom.atom {
                        Atom::Ssa(name) => assert_eq!(name, "a"),
                        _ => panic!("expected ssa range source"),
                    }
                }
                other => panic!("expected range expression for `{src}`, got {other:?}"),
            }
        }
    }

    #[test]
    fn parses_map_without_captures() {
        let statements = stmts("%m = inc <$> %src");
        match &statements[0] {
            Statement::Assign {
                expr:
                    ExprNode::Map {
                        body,
                        src,
                        captures,
                    },
                ..
            } => {
                assert_eq!(body, &Callee::Named("inc".into()));
                assert!(captures.is_empty());
                match &src.atom {
                    Atom::Ssa(name) => assert_eq!(name, "src"),
                    _ => panic!("expected ssa src"),
                }
            }
            _ => panic!("expected map expression"),
        }
    }

    #[test]
    fn parses_map_with_captures() {
        let statements = stmts("%m = (addk %k0 %k1) <$> %src");
        match &statements[0] {
            Statement::Assign {
                expr:
                    ExprNode::Map {
                        body,
                        src,
                        captures,
                    },
                ..
            } => {
                assert_eq!(body, &Callee::Named("addk".into()));
                assert_eq!(captures.len(), 2);
                match &src.atom {
                    Atom::Ssa(name) => assert_eq!(name, "src"),
                    _ => panic!("expected ssa src"),
                }
            }
            _ => panic!("expected map expression"),
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
                Atom::Varnode(name) => assert_eq!(name, "ptr"),
                _ => panic!("expected varnode atom"),
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
        let statements = stmts("load(ram:4, {ptr}); store(ram:4, {ptr} <- {src})");
        assert_eq!(statements.len(), 2);

        match &statements[0] {
            Statement::Expr(ExprNode::Load {
                space,
                size_bytes,
                ptr,
            }) => {
                assert_eq!(space, "ram");
                assert_eq!(*size_bytes, 4);
                match &ptr.atom {
                    Atom::External(name) => assert_eq!(name, "ptr"),
                    _ => panic!("expected external load pointer"),
                }
            }
            _ => panic!("expected load statement"),
        }

        match &statements[1] {
            Statement::Expr(ExprNode::Store {
                space,
                size_bytes,
                ptr,
                src,
            }) => {
                assert_eq!(space, "ram");
                assert_eq!(*size_bytes, 4);
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
                expr,
                name_span,
                span,
                ..
            } => {
                assert_eq!(name, "a");
                assert!(matches!(expr, ExprNode::Binary { .. }));
                assert_eq!(name_span.start.column, 5);
                assert_eq!(name_span.end.column, 7);
                assert_eq!(span.start.column, 1);
            }
            _ => panic!("expected ssa assignment"),
        }

        match &statements[1] {
            Statement::Assign { name, expr, .. } => {
                assert_eq!(name, "b");
                match expr {
                    ExprNode::Binary { lhs, .. } => match &lhs.atom {
                        Atom::Ssa(name) => assert_eq!(name, "a"),
                        _ => panic!("expected ssa lhs referencing %a"),
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
            Statement::Call { target, args, .. } => {
                assert_eq!(target, &Callee::Named("target".into()));
                assert!(args.is_empty());
            }
            _ => panic!("expected call statement"),
        }
    }

    #[test]
    fn parses_call_with_args() {
        let statements = stmts("call fn callee(@arg0={x}, @arg1={y})");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Call { target, args, .. } => {
                assert_eq!(target, &Callee::Named("callee".into()));
                assert_eq!(args.len(), 2);
                assert_eq!(args[0].0, "@arg0");
                assert_eq!(args[1].0, "@arg1");
            }
            _ => panic!("expected call statement"),
        }
    }

    #[test]
    fn parses_call_with_edge_hint() {
        let statements = stmts("call fn callee(@p={x}) // -> <resume>");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::Call {
                target, targets, ..
            } => {
                assert_eq!(target, &Callee::Named("callee".into()));
                assert_eq!(targets.len(), 1);
                assert!(matches!(&targets[0], Label::Named { name, .. } if name == "resume"));
            }
            _ => panic!("expected call statement"),
        }
    }

    #[test]
    fn parses_branchind_with_edge_hint() {
        let statements = stmts("goto [{p}] // -> <a>, <b>");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::BranchInd { targets, .. } => {
                let names: Vec<&str> = targets
                    .iter()
                    .map(|t| match t {
                        Label::Named { name, .. } => name.as_str(),
                        _ => panic!("expected named target"),
                    })
                    .collect();
                assert_eq!(names, ["a", "b"]);
            }
            _ => panic!("expected branchind statement"),
        }
    }

    #[test]
    fn parses_callind_with_args() {
        let statements = stmts("call [{ptr}]({a}, {b})");
        assert_eq!(statements.len(), 1);

        match &statements[0] {
            Statement::CallInd { args, .. } => assert_eq!(args.len(), 2),
            _ => panic!("expected callind statement"),
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
        let statements = stmts("return at {ptr}");
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
        let program = "%tmp = {v1} + 3; goto <done>; <done> %tmp + 2";
        let statements = stmts(program);
        assert_eq!(statements.len(), 4);

        match &statements[0] {
            Statement::Assign { name, .. } => assert_eq!(name, "tmp"),
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
        match program.kind {
            ProgramKind::Functions { fns, .. } => {
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
        match program.kind {
            ProgramKind::Functions { fns, .. } => {
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
                assert_eq!(params[0].name, "v1");
                assert_eq!(params[0].size_bytes, None);
                assert_eq!(params[1].name, "v2");
                assert_eq!(params[1].size_bytes, None);
            }
            _ => panic!("expected named label declaration with params"),
        }
    }

    #[test]
    fn parses_label_decl_with_typed_params() {
        let statements = stmts("<entry @v1:i64 @v2:i32>");
        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::LabelDecl {
                label: Label::Named { name, params, .. },
                ..
            } => {
                assert_eq!(name, "entry");
                assert_eq!(params[0].name, "v1");
                assert_eq!(params[0].size_bytes, Some(8));
                assert_eq!(params[1].name, "v2");
                assert_eq!(params[1].size_bytes, Some(4));
            }
            _ => panic!("expected named label declaration with typed params"),
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
                assert!(matches!(&args[1].1.atom, Atom::Ssa(n) if n == "x"));
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
                condition,
                target,
                target_args,
                fallthrough,
                fallthrough_args,
                ..
            } => {
                assert!(matches!(&condition.atom, Atom::Ssa(n) if n == "c"));
                assert!(matches!(target, Label::Named { name, .. } if name == "then_lbl"));
                assert_eq!(target_args.len(), 1);
                assert_eq!(target_args[0].0, "x");
                assert!(matches!(target_args[0].1.atom, Atom::Int(1)));
                assert!(matches!(fallthrough, Label::Named { name, .. } if name == "else_lbl"));
                assert_eq!(fallthrough_args.len(), 1);
                assert_eq!(fallthrough_args[0].0, "y");
                assert!(matches!(&fallthrough_args[0].1.atom, Atom::Ssa(n) if n == "v"));
            }
            _ => panic!("expected cbranch with args"),
        }
    }

    #[test]
    fn parses_switch_with_cases_and_default() {
        let statements =
            stmts("switch %idx { 0x0 => <a_lbl>, 0x3 => <b_lbl @p=%v>, default => <d_lbl> }");
        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::Switch {
                scrutinee,
                cases,
                default,
                ..
            } => {
                assert!(matches!(&scrutinee.atom, Atom::Ssa(n) if n == "idx"));
                assert_eq!(cases.len(), 2);

                assert_eq!(cases[0].0, 0);
                assert!(matches!(&cases[0].1, Label::Named { name, .. } if name == "a_lbl"));
                assert!(cases[0].2.is_empty());

                // A case arm carries block arguments just as a `goto` does.
                assert_eq!(cases[1].0, 3);
                assert!(matches!(&cases[1].1, Label::Named { name, .. } if name == "b_lbl"));
                assert_eq!(cases[1].2.len(), 1);
                assert_eq!(cases[1].2[0].0, "p");
                assert!(matches!(&cases[1].2[0].1.atom, Atom::Ssa(n) if n == "v"));

                let (default_label, default_args) = default.as_ref().expect("default arm");
                assert!(matches!(default_label, Label::Named { name, .. } if name == "d_lbl"));
                assert!(default_args.is_empty());
            }
            other => panic!("expected switch, got {other:?}"),
        }
    }

    /// A jump table guarded by a bounds check is total over the values it lists,
    /// so the default arm is optional.
    #[test]
    fn parses_switch_without_default() {
        let statements = stmts("switch %idx { 0x0 => <a_lbl>, 0x1 => <b_lbl> }");
        match &statements[0] {
            Statement::Switch { cases, default, .. } => {
                assert_eq!(cases.len(), 2);
                assert!(default.is_none());
            }
            other => panic!("expected switch, got {other:?}"),
        }
    }

    #[test]
    fn parses_comment_attached_to_next_statement() {
        let statements = stmts("# add one\n%x = {v} + 1");
        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::Commented { comment, inner } => {
                assert_eq!(comment, "add one");
                assert!(matches!(inner.as_ref(), Statement::Assign { name, .. } if name == "x"));
            }
            _ => panic!("expected commented statement"),
        }
    }

    #[test]
    fn parses_comment_stripped_for_standalone_expression() {
        let statements = stmts("# first\n%a = 1; # second\n%b = 2");
        assert_eq!(statements.len(), 2);
        match &statements[0] {
            Statement::Commented { comment, .. } => assert_eq!(comment, "first"),
            _ => panic!("expected first statement to be commented"),
        }
        match &statements[1] {
            Statement::Commented { comment, .. } => assert_eq!(comment, "second"),
            _ => panic!("expected second statement to be commented"),
        }
    }

    #[test]
    fn parses_uncommented_statements_unaffected() {
        let statements = stmts("%x = 1 + 2");
        assert_eq!(statements.len(), 1);
        assert!(matches!(&statements[0], Statement::Assign { name, .. } if name == "x"));
    }

    #[test]
    fn parses_comment_in_fn_body() {
        let program = parse_program("fn f: <entry> # load value\n%x = {v} + 1; return at %x")
            .expect("parse should succeed");
        match program.kind {
            ProgramKind::Functions { fns, .. } => {
                let stmts = &fns[0].statements;
                // stmts[0] = LabelDecl <entry>, stmts[1] = Commented(%x = ...), stmts[2] = return
                assert!(
                    matches!(&stmts[1], Statement::Commented { comment, .. } if comment == "load value")
                );
            }
            _ => panic!("expected function program"),
        }
    }

    #[test]
    fn parses_top_level_varnode_before_fn() {
        let program = parse_program("varnode i64 ptr; fn f: <entry> return at ptr")
            .expect("parse should succeed");
        match program.kind {
            ProgramKind::Functions { varnodes, fns } => {
                assert_eq!(varnodes.len(), 1);
                assert!(
                    matches!(&varnodes[0], Statement::LocalDecl { name, size_bytes, .. } if name == "ptr" && *size_bytes == 8)
                );
                assert_eq!(fns.len(), 1);
            }
            _ => panic!("expected function program"),
        }
    }

    #[test]
    fn parses_lambda_apply_and_value_returns() {
        let program = parse_program(
            "lambda rec: <entry @s:i64> %next = @s + 1; %out = apply rec(%next); return %out",
        )
        .expect("parse should succeed");
        let ProgramKind::Functions { fns, .. } = program.kind else {
            panic!("expected function program")
        };
        assert_eq!(fns.len(), 1);
        assert_eq!(fns[0].kind, crate::ast::FnKind::Lambda);
        assert!(matches!(
            &fns[0].statements[2],
            Statement::Assign { expr: ExprNode::Apply { target, args }, .. }
                if target == &Callee::Named("rec".into()) && args.len() == 1
        ));
        assert!(matches!(
            &fns[0].statements[3],
            Statement::ReturnValue { value, .. } if matches!(&value.atom, Atom::Ssa(name) if name == "out")
        ));
    }

    #[test]
    fn parses_machine_return_at_forms() {
        let statements = stmts("return at %ptr; return %v at %ptr");
        assert_eq!(statements.len(), 2);
        assert!(matches!(
            &statements[0],
            Statement::Return { value: None, ptr, .. }
                if matches!(&ptr.atom, Atom::Ssa(name) if name == "ptr")
        ));
        assert!(matches!(
            &statements[1],
            Statement::Return { value: Some(value), ptr, .. }
                if matches!(&value.atom, Atom::Ssa(name) if name == "v")
                    && matches!(&ptr.atom, Atom::Ssa(name) if name == "ptr")
        ));
    }

    #[test]
    fn parses_struct_decl_offsets_and_padding() {
        use crate::ast::{ExprNode, GepField, StructFieldType};
        let program =
            parse_program("type Foo { a: 4, _: 5, b: 2, p: Bar* }; %x + 1").expect("parse");
        assert_eq!(program.structs.len(), 1);
        let foo = &program.structs[0];
        assert_eq!(foo.name, "Foo");
        // 4 named fields minus padding = a, b, p.
        let named: Vec<&str> = foo
            .fields
            .iter()
            .filter(|f| !f.is_padding())
            .map(|f| f.name.as_str())
            .collect();
        assert_eq!(named, ["a", "b", "p"]);
        // Padding `_` advances the running offset (a@0, then 4 + 5 = 9 → b@9).
        assert!(matches!(foo.fields[1].ty, StructFieldType::Int(5)));
        assert!(foo.fields[1].is_padding());
        // Pointer-to-struct field type is captured.
        assert!(matches!(&foo.fields[3].ty, StructFieldType::StructPtr(name) if name == "Bar"));

        // gep parses as its own expression node.
        let geps = parse_program("%y = gep(%p.field)").expect("parse").kind;
        let ProgramKind::Statements(stmts) = geps else {
            panic!("expected statements")
        };
        assert!(matches!(
            &stmts[0],
            Statement::Assign { expr: ExprNode::Gep { field: GepField::Name(n), .. }, .. } if n == "field"
        ));
    }

    #[test]
    fn parses_canonical_minted_callees_in_all_direct_forms() {
        let cases = [
            ("%x = apply <minted:1>()", 1),
            ("%x = <minted:2> <$> i64 0", 2),
            ("%x = scanl <minted:3> i64 0 i64 1", 3),
            ("call fn <minted:4>()", 4),
            ("tailcall fn <minted:5>()", 5),
        ];

        for (source, expected) in cases {
            let statement = stmts(source).pop().expect("one statement");
            let callee = match statement {
                Statement::Assign {
                    expr: ExprNode::Apply { target, .. },
                    ..
                } => target,
                Statement::Assign {
                    expr: ExprNode::Map { body, .. },
                    ..
                }
                | Statement::Assign {
                    expr: ExprNode::Scan { body, .. },
                    ..
                } => body,
                Statement::Call { target, .. } => target,
                other => panic!("unexpected parse for {source}: {other:?}"),
            };
            assert_eq!(callee, Callee::Minted(expected));
        }

        assert!(matches!(
            stmts("tailcall fn <minted:9>()").pop(),
            Some(Statement::Call { tail: true, .. })
        ));
    }
}
