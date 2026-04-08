// [AI Generated]
use qcode_parser::ast::{Atom, CastOp, ExprNode, Statement, TypedAtom};
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::Expr;

/// Terminator statements that end a block.
fn is_terminator(stmt: &Statement) -> bool {
    matches!(
        stmt,
        Statement::Branch { .. }
            | Statement::BranchInd { .. }
            | Statement::CBranch { .. }
            | Statement::Call { .. }
            | Statement::CallInd { .. }
            | Statement::Return { .. }
    )
}

fn all_local_decls(statements: &[Statement]) -> bool {
    statements
        .iter()
        .all(|statement| matches!(statement, Statement::LocalDecl { .. }))
}

pub(crate) fn compile_qcode_from_statements(
    builder: &Expr,
    statements: &[Statement],
    pcode_root: &proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    if statements.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "qcode program cannot be empty",
        ));
    }

    if all_local_decls(statements) {
        let mut emitted = Vec::with_capacity(statements.len());

        for statement in statements {
            let Statement::LocalDecl {
                name,
                display_name,
                size_bytes,
            } = statement
            else {
                unreachable!("checked above");
            };

            let ident = format_ident!("{}", name);
            let size = *size_bytes;
            emitted.push(quote! {
                let #ident = (#builder)
                    .make_named_temp(::std::borrow::Cow::Borrowed(#display_name), #size);
            });
        }

        return Ok(quote! {
            #(#emitted)*
        });
    }

    // Collect all SSA names that need to be exposed to the outer scope.
    let exposed: Vec<proc_macro2::Ident> = statements
        .iter()
        .filter_map(|s| {
            if let Statement::Assign {
                name, expose: true, ..
            } = s
            {
                Some(format_ident!("{}", name))
            } else {
                None
            }
        })
        .collect();

    let mut locals: HashMap<String, proc_macro2::Ident> = HashMap::new();
    let mut emitted = Vec::new();
    let mut final_expr: Option<proc_macro2::Ident> = None;

    for (index, statement) in statements.iter().enumerate() {
        match statement {
            Statement::LocalDecl {
                name,
                display_name,
                size_bytes,
            } => {
                let ident = format_ident!("__qcode_local_{}", name);
                let size = *size_bytes;
                emitted.push(quote! {
                    let #ident = __qcode_builder
                        .make_named_temp(Cow::Borrowed(#display_name), #size);
                });
                locals.insert(name.clone(), ident);
                final_expr = None;
            }

            Statement::Assign { name, expose, expr } => {
                let ident = format_ident!("__qcode_local_{}", name);
                let value_tokens = lower_expr(expr, &locals, pcode_root)?;
                if *expose {
                    let outer_ident = format_ident!("{}", name);
                    emitted.push(quote! {
                        let #ident = #value_tokens;
                        #outer_ident = #ident;
                    });
                } else {
                    emitted.push(quote! {
                        let #ident = #value_tokens;
                    });
                }
                locals.insert(name.clone(), ident);
                final_expr = None;
            }

            Statement::Expr(expr) => {
                let value_tokens = lower_expr(expr, &locals, pcode_root)?;
                let ident = format_ident!("__qcode_result_{}", index);
                emitted.push(quote! {
                    let #ident = #value_tokens;
                });
                final_expr = Some(ident);
            }

            Statement::LabelDecl { name } => {
                emitted.push(quote! {
                    {
                        let __qcode_label = __qcode_builder.get_or_make_local_label(Cow::Borrowed(#name));
                        __qcode_builder.switch_to_block(__qcode_label);
                    }
                });
                final_expr = None;
            }

            Statement::Branch { target } => {
                emitted.push(quote! {
                    {
                        let __qcode_target = __qcode_builder.get_or_make_local_label(Cow::Borrowed(#target));
                        __qcode_builder.push_branch(__qcode_target);
                    }
                });
                final_expr = None;
            }

            Statement::BranchInd { ptr } => {
                let ptr_tokens = lower_atom(ptr, None, &locals, pcode_root)?;
                emitted.push(quote! {
                    {
                        let __qcode_ptr = #ptr_tokens;
                        __qcode_builder.push_branchind(__qcode_ptr);
                    }
                });
                final_expr = None;
            }

            Statement::CBranch {
                condition,
                target,
                fallthrough,
            } => {
                let cond_tokens = lower_atom(condition, None, &locals, pcode_root)?;
                emitted.push(quote! {
                    {
                        let __qcode_cond = #cond_tokens;
                        let __qcode_target = __qcode_builder.get_or_make_local_label(Cow::Borrowed(#target));
                        let __qcode_fallthrough = __qcode_builder.get_or_make_local_label(Cow::Borrowed(#fallthrough));
                        __qcode_builder.push_cbranch(__qcode_cond, __qcode_target, __qcode_fallthrough, );
                        __qcode_builder.switch_to_block(__qcode_fallthrough);
                    }
                });
                final_expr = None;
            }

            Statement::Call { target } => {
                emitted.push(quote! {
                    {
                        let __qcode_target = __qcode_builder.get_or_make_local_function(Cow::Borrowed(#target));
                        __qcode_builder.push_call(__qcode_target);
                    }
                });
                final_expr = None;
            }

            Statement::CallInd { ptr } => {
                let ptr_tokens = lower_atom(ptr, None, &locals, pcode_root)?;
                emitted.push(quote! {
                    {
                        let __qcode_ptr = #ptr_tokens;
                        __qcode_builder.push_call_ind(__qcode_ptr);
                    }
                });
                final_expr = None;
            }

            Statement::Return { ptr } => {
                let ptr_tokens = lower_atom(ptr, None, &locals, pcode_root)?;
                emitted.push(quote! {
                    {
                        let __qcode_ptr = #ptr_tokens;
                        __qcode_builder.push_return(__qcode_ptr);
                    }
                });
                final_expr = None;
            }
        }
    }

    let last = statements.last().expect("non-empty checked above");

    let predecls = exposed.iter().map(|ident| {
        quote! {
            let #ident: #pcode_root::value::InstructionId;
        }
    });

    if is_terminator(last) || matches!(last, Statement::LabelDecl { .. }) {
        // Program ends with a terminator or label — return ()
        Ok(quote! {
            #(#predecls)*
            {
                use ::std::borrow::Cow;
                use #pcode_root::value::Value as _;
                let __qcode_builder = &mut (#builder);
                #(#emitted)*
            }
        })
    } else {
        let Some(final_ident) = final_expr else {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "qcode program must end with an expression or a terminator",
            ));
        };

        Ok(quote! {
            #(#predecls)*
            {
                use ::std::borrow::Cow;
                use #pcode_root::value::Value as _;
                let __qcode_builder = &mut (#builder);
                #(#emitted)*
                #final_ident
            }
        })
    }
}

fn lower_expr(
    expr: &ExprNode,
    locals: &HashMap<String, proc_macro2::Ident>,
    pcode_root: &proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    match expr {
        ExprNode::Atom(atom) => lower_atom(atom, None, locals, pcode_root),

        ExprNode::Unop { op, src } => {
            let src_tokens = lower_atom(src, None, locals, pcode_root)?;

            let call = match op.as_str() {
                "!" => quote! { __qcode_builder.push_bool_not(__qcode_src).id },
                "~" => quote! { __qcode_builder.push_bit_negate(__qcode_src).id },
                "-" => quote! { __qcode_builder.push_neg(__qcode_src).id },
                "f-" => quote! { __qcode_builder.push_fneg(__qcode_src).id },
                "abs" => quote! { __qcode_builder.push_abs(__qcode_src).id },
                "sqrt" => quote! { __qcode_builder.push_sqrt(__qcode_src).id },
                "floor" => quote! { __qcode_builder.push_floor(__qcode_src).id },
                "ceil" => quote! { __qcode_builder.push_ceil(__qcode_src).id },
                "round" => quote! { __qcode_builder.push_round(__qcode_src).id },
                _ => {
                    return Err(syn::Error::new(
                        proc_macro2::Span::call_site(),
                        format!("unsupported unary operator: {op}"),
                    ));
                }
            };

            Ok(quote! {
                {
                    let __qcode_src = #src_tokens;
                    #call
                }
            })
        }

        ExprNode::Binary { lhs, op, rhs } => {
            let lhs_size = size_hint_tokens(lhs, locals, pcode_root)?;
            let rhs_size = size_hint_tokens(rhs, locals, pcode_root)?;

            let lhs_tokens = lower_atom(lhs, rhs_size.clone(), locals, pcode_root)?;
            let rhs_tokens = lower_atom(rhs, lhs_size.clone(), locals, pcode_root)?;

            let call = match op.as_str() {
                "+" => quote! { __qcode_builder.push_add(__qcode_lhs, __qcode_rhs).id },
                "-" => quote! { __qcode_builder.push_sub(__qcode_lhs, __qcode_rhs).id },
                "*" => quote! { __qcode_builder.push_mul(__qcode_lhs, __qcode_rhs).id },
                "/" => quote! { __qcode_builder.push_div(__qcode_lhs, __qcode_rhs).id },
                "&" => quote! { __qcode_builder.push_bit_and(__qcode_lhs, __qcode_rhs).id },
                "|" => quote! { __qcode_builder.push_bit_or(__qcode_lhs, __qcode_rhs).id },
                "^" => quote! { __qcode_builder.push_bit_xor(__qcode_lhs, __qcode_rhs).id },
                "^^" => quote! { __qcode_builder.push_bool_xor(__qcode_lhs, __qcode_rhs).id },
                "&&" => quote! { __qcode_builder.push_bool_and(__qcode_lhs, __qcode_rhs).id },
                "||" => quote! { __qcode_builder.push_bool_or(__qcode_lhs, __qcode_rhs).id },
                "<<" => quote! { __qcode_builder.push_shl(__qcode_lhs, __qcode_rhs).id },
                ">>" => quote! { __qcode_builder.push_shr(__qcode_lhs, __qcode_rhs).id },
                "s>>" => quote! { __qcode_builder.push_sshr(__qcode_lhs, __qcode_rhs).id },
                "==" => quote! { __qcode_builder.push_eq(__qcode_lhs, __qcode_rhs).id },
                "!=" => quote! { __qcode_builder.push_ne(__qcode_lhs, __qcode_rhs).id },
                "<" => quote! { __qcode_builder.push_lt(__qcode_lhs, __qcode_rhs).id },
                "<=" => quote! { __qcode_builder.push_le(__qcode_lhs, __qcode_rhs).id },
                ">" => quote! { __qcode_builder.push_gt(__qcode_lhs, __qcode_rhs).id },
                ">=" => quote! { __qcode_builder.push_ge(__qcode_lhs, __qcode_rhs).id },
                "s<" => quote! { __qcode_builder.push_slt(__qcode_lhs, __qcode_rhs).id },
                "s<=" => quote! { __qcode_builder.push_sle(__qcode_lhs, __qcode_rhs).id },
                "s>" => quote! { __qcode_builder.push_sgt(__qcode_lhs, __qcode_rhs).id },
                "s>=" => quote! { __qcode_builder.push_sge(__qcode_lhs, __qcode_rhs).id },
                "%" => quote! { __qcode_builder.push_mod(__qcode_lhs, __qcode_rhs).id },
                "s/" => quote! { __qcode_builder.push_sdiv(__qcode_lhs, __qcode_rhs).id },
                "s%" => quote! { __qcode_builder.push_smod(__qcode_lhs, __qcode_rhs).id },
                "f+" => quote! { __qcode_builder.push_fadd(__qcode_lhs, __qcode_rhs).id },
                "f-" => quote! { __qcode_builder.push_fsub(__qcode_lhs, __qcode_rhs).id },
                "f*" => quote! { __qcode_builder.push_fmul(__qcode_lhs, __qcode_rhs).id },
                "f/" => quote! { __qcode_builder.push_fdiv(__qcode_lhs, __qcode_rhs).id },
                "f==" => quote! { __qcode_builder.push_feq(__qcode_lhs, __qcode_rhs).id },
                "f!=" => quote! { __qcode_builder.push_fne(__qcode_lhs, __qcode_rhs).id },
                "f<" => quote! { __qcode_builder.push_flt(__qcode_lhs, __qcode_rhs).id },
                "f<=" => quote! { __qcode_builder.push_fle(__qcode_lhs, __qcode_rhs).id },
                "f>" => quote! { __qcode_builder.push_fgt(__qcode_lhs, __qcode_rhs).id },
                "f>=" => quote! { __qcode_builder.push_fge(__qcode_lhs, __qcode_rhs).id },
                _ => {
                    return Err(syn::Error::new(
                        proc_macro2::Span::call_site(),
                        format!("unsupported operator: {op}"),
                    ));
                }
            };

            Ok(quote! {
                {
                    let __qcode_lhs = #lhs_tokens;
                    let __qcode_rhs = #rhs_tokens;
                    let __qcode_lhs_size = __qcode_builder.context().get_value(__qcode_lhs).size();
                    let __qcode_rhs_size = __qcode_builder.context().get_value(__qcode_rhs).size();
                    assert_eq!(
                        __qcode_lhs_size,
                        __qcode_rhs_size,
                        "qcode size mismatch in binary expression: lhs={} rhs={}",
                        __qcode_lhs_size,
                        __qcode_rhs_size
                    );
                    #call
                }
            })
        }

        ExprNode::Cast {
            op,
            size_bytes,
            src,
        } => {
            let src_tokens = lower_atom(src, None, locals, pcode_root)?;
            let size = *size_bytes;

            let call = match op {
                CastOp::Zext => quote! { __qcode_builder.push_zext(__qcode_src, #size).id },
                CastOp::Sext => quote! { __qcode_builder.push_sext(__qcode_src, #size).id },
                CastOp::IntToFloat => {
                    quote! { __qcode_builder.push_int_to_float(__qcode_src, #size).id }
                }
                CastOp::FloatToFloat => {
                    quote! { __qcode_builder.push_float_to_float(__qcode_src, #size).id }
                }
                CastOp::Trunc => quote! { __qcode_builder.push_trunc(__qcode_src, #size).id },
            };

            Ok(quote! {
                {
                    let __qcode_src = #src_tokens;
                    #call
                }
            })
        }

        ExprNode::Load { size_bytes, ptr } => {
            let ptr_tokens = lower_atom(ptr, None, locals, pcode_root)?;
            let size = *size_bytes;

            Ok(quote! {
                {
                    let __qcode_ptr = #ptr_tokens;
                    let __qcode_value = __qcode_builder
                        .push_load::<true>(
                            __qcode_ptr,
                            #size,
                            #pcode_root::space::SPACE_UNIQUE,
                        )
                        .id();

                    match __qcode_value {
                        #pcode_root::value::ValueId::Instruction(id) => id,
                        _ => panic!("qcode load expected instruction result"),
                    }
                }
            })
        }

        ExprNode::Store { ptr, src } => {
            let src_size = size_hint_tokens(src, locals, pcode_root)?;

            let ptr_tokens = lower_atom(ptr, None, locals, pcode_root)?;
            let src_tokens = lower_atom(src, src_size, locals, pcode_root)?;

            Ok(quote! {
                {
                    let __qcode_ptr = #ptr_tokens;
                    let __qcode_src = #src_tokens;
                    __qcode_builder
                        .push_store(__qcode_src, __qcode_ptr, #pcode_root::space::SPACE_UNIQUE)
                        .id
                }
            })
        }

        ExprNode::FuncCall { op, args } => {
            let call = match op.as_str() {
                "nan" => {
                    if args.len() != 1 {
                        return Err(syn::Error::new(
                            proc_macro2::Span::call_site(),
                            "nan expects exactly 1 argument",
                        ));
                    }
                    let src_tokens = lower_atom(&args[0], None, locals, pcode_root)?;
                    quote! {
                        {
                            let __qcode_src = #src_tokens;
                            __qcode_builder.push_is_nan(__qcode_src).id
                        }
                    }
                }
                "popcount" => {
                    if args.len() != 1 {
                        return Err(syn::Error::new(
                            proc_macro2::Span::call_site(),
                            "popcount expects exactly 1 argument",
                        ));
                    }
                    let src_tokens = lower_atom(&args[0], None, locals, pcode_root)?;
                    quote! {
                        {
                            let __qcode_src = #src_tokens;
                            __qcode_builder.push_popcount(__qcode_src, 1).id
                        }
                    }
                }
                "lzcount" => {
                    if args.len() != 1 {
                        return Err(syn::Error::new(
                            proc_macro2::Span::call_site(),
                            "lzcount expects exactly 1 argument",
                        ));
                    }
                    let src_tokens = lower_atom(&args[0], None, locals, pcode_root)?;
                    quote! {
                        {
                            let __qcode_src = #src_tokens;
                            __qcode_builder.push_lzcount(__qcode_src, 1).id
                        }
                    }
                }
                "carry" => {
                    if args.len() != 2 {
                        return Err(syn::Error::new(
                            proc_macro2::Span::call_site(),
                            "carry expects exactly 2 arguments",
                        ));
                    }
                    let lhs_tokens = lower_atom(&args[0], None, locals, pcode_root)?;
                    let rhs_tokens = lower_atom(&args[1], None, locals, pcode_root)?;
                    quote! {
                        {
                            let __qcode_lhs = #lhs_tokens;
                            let __qcode_rhs = #rhs_tokens;
                            __qcode_builder.push_carry(__qcode_lhs, __qcode_rhs).id
                        }
                    }
                }
                "scarry" => {
                    if args.len() != 2 {
                        return Err(syn::Error::new(
                            proc_macro2::Span::call_site(),
                            "scarry expects exactly 2 arguments",
                        ));
                    }
                    let lhs_tokens = lower_atom(&args[0], None, locals, pcode_root)?;
                    let rhs_tokens = lower_atom(&args[1], None, locals, pcode_root)?;
                    quote! {
                        {
                            let __qcode_lhs = #lhs_tokens;
                            let __qcode_rhs = #rhs_tokens;
                            __qcode_builder.push_scarry(__qcode_lhs, __qcode_rhs).id
                        }
                    }
                }
                "sborrow" => {
                    if args.len() != 2 {
                        return Err(syn::Error::new(
                            proc_macro2::Span::call_site(),
                            "sborrow expects exactly 2 arguments",
                        ));
                    }
                    let lhs_tokens = lower_atom(&args[0], None, locals, pcode_root)?;
                    let rhs_tokens = lower_atom(&args[1], None, locals, pcode_root)?;
                    quote! {
                        {
                            let __qcode_lhs = #lhs_tokens;
                            let __qcode_rhs = #rhs_tokens;
                            __qcode_builder.push_sborrow(__qcode_lhs, __qcode_rhs).id
                        }
                    }
                }
                _ => {
                    return Err(syn::Error::new(
                        proc_macro2::Span::call_site(),
                        format!("unsupported function call: {op}"),
                    ));
                }
            };

            Ok(call)
        }
    }
}

fn lower_atom(
    typed: &TypedAtom,
    size_hint: Option<proc_macro2::TokenStream>,
    locals: &HashMap<String, proc_macro2::Ident>,
    pcode_root: &proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    match &typed.atom {
        Atom::External(name) => {
            let ident = format_ident!("{}", name);
            if let Some(expected_size) = typed.size_bytes {
                Ok(quote! {
                    {
                        let __qcode_value_id: #pcode_root::value::ValueId = (#ident).into();
                        let __qcode_expected_size = #expected_size;
                        let __qcode_actual_size = __qcode_builder.context().get_value(__qcode_value_id).size();
                        assert_eq!(
                            __qcode_actual_size,
                            __qcode_expected_size,
                            "qcode size mismatch for value `{}`: expected {} bytes, got {} bytes",
                            stringify!(#ident),
                            __qcode_expected_size,
                            __qcode_actual_size
                        );
                        __qcode_value_id
                    }
                })
            } else {
                Ok(quote! {
                    {
                        let __qcode_value_id: #pcode_root::value::ValueId = (#ident).into();
                        __qcode_value_id
                    }
                })
            }
        }

        Atom::Local(name) => {
            let Some(local_ident) = locals.get(name) else {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("unknown local identifier '{name}'"),
                ));
            };

            if let Some(expected_size) = typed.size_bytes {
                Ok(quote! {
                    {
                        let __qcode_value_id: #pcode_root::value::ValueId = (#local_ident).into();
                        let __qcode_expected_size = #expected_size;
                        let __qcode_actual_size = __qcode_builder.context().get_value(__qcode_value_id).size();
                        assert_eq!(
                            __qcode_actual_size,
                            __qcode_expected_size,
                            "qcode size mismatch for local `{}`: expected {} bytes, got {} bytes",
                            stringify!(#local_ident),
                            __qcode_expected_size,
                            __qcode_actual_size
                        );
                        __qcode_value_id
                    }
                })
            } else {
                Ok(quote! {
                    {
                        let __qcode_value_id: #pcode_root::value::ValueId = (#local_ident).into();
                        __qcode_value_id
                    }
                })
            }
        }

        Atom::Int(value) => {
            let size = if let Some(explicit) = typed.size_bytes {
                quote! { #explicit }
            } else {
                size_hint.unwrap_or_else(|| quote! { 8usize })
            };
            Ok(quote! {
                {
                    let __qcode_size = #size;
                    __qcode_builder.context_mut().get_const(#value, __qcode_size).id()
                }
            })
        }
    }
}

fn size_hint_tokens(
    typed: &TypedAtom,
    locals: &HashMap<String, proc_macro2::Ident>,
    pcode_root: &proc_macro2::TokenStream,
) -> syn::Result<Option<proc_macro2::TokenStream>> {
    if let Some(explicit) = typed.size_bytes {
        return Ok(Some(quote! { #explicit }));
    }

    match &typed.atom {
        Atom::External(name) => {
            let ident = format_ident!("{}", name);
            Ok(Some(quote! {
                {
                    let __qcode_value_id: #pcode_root::value::ValueId = (#ident).into();
                    __qcode_builder.context().get_value(__qcode_value_id).size()
                }
            }))
        }

        Atom::Local(name) => {
            let Some(local_ident) = locals.get(name) else {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("unknown local identifier '{name}'"),
                ));
            };
            Ok(Some(quote! {
                {
                    let __qcode_value_id: #pcode_root::value::ValueId = (#local_ident).into();
                    __qcode_builder.context().get_value(__qcode_value_id).size()
                }
            }))
        }

        Atom::Int(_) => Ok(None),
    }
}
