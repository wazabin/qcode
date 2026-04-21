// [AI Generated]
use qcode_parser::ast::{Atom, CastOp, ExprNode, FnDecl, Label, Statement, TypedAtom};
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::Expr;

/// Tracks whether a local identifier is a varnode (pointer) or an SSA instruction value.
#[derive(Clone)]
enum LocalKind {
    Varnode(proc_macro2::Ident),
    Instruction(proc_macro2::Ident),
}

impl LocalKind {
    fn ident(&self) -> &proc_macro2::Ident {
        match self {
            LocalKind::Varnode(i) | LocalKind::Instruction(i) => i,
        }
    }

    fn is_varnode(&self) -> bool {
        matches!(self, LocalKind::Varnode(_))
    }
}

/// Context in which an atom is being used — controls whether bare varnodes are allowed.
#[derive(Clone, Copy)]
enum AtomContext {
    /// Arithmetic / scalar-value position. Bare varnodes are rejected; use `&name` instead.
    Arithmetic,
    /// Pointer position (load ptr, store ptr, branch target). Bare varnodes are allowed.
    Pointer,
}

/// Compile a function-level program (one or more `fn name: ...` declarations).
pub(crate) fn compile_fn_program(
    ctx: &Expr,
    top_varnodes: &[Statement],
    fns: &[FnDecl],
    pcode_root: &proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    // Emit top-level varnode declarations (no builder needed — use context directly).
    let mut global_varnode_predecls: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut global_varnode_inits: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut global_locals: HashMap<String, LocalKind> = HashMap::new();

    for stmt in top_varnodes {
        let Statement::LocalDecl {
            name, size_bytes, ..
        } = stmt
        else {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "top-level varnode list may only contain varnode declarations",
            ));
        };
        let ident = format_ident!("__qcode_local_{}", name);
        let outer_ident = format_ident!("{}", name);
        let size = *size_bytes;
        global_varnode_predecls.push(quote! { let #outer_ident: #pcode_root::value::VarnodeId; });
        global_varnode_inits.push(quote! {
            let #ident = {
                use ::std::borrow::Cow;
                use #pcode_root::value::Renameable as _;
                let __name = (#ctx).get_unique_name(Cow::Borrowed(#name));
                let __space = (#ctx).make_named_temp_space(__name.clone());
                let __id = #pcode_root::value::Varnode::make(&mut (#ctx), 0, #size, __space).id;
                #pcode_root::value::Varnode::from_id_mut(&mut (#ctx), __id)
                    .rename(__name)
                    .expect("qcode: varnode name conflict");
                __id
            };
            #outer_ident = #ident;
        });
        global_locals.insert(name.clone(), LocalKind::Varnode(ident));
    }

    let mut all_predecls = Vec::new();
    let mut all_bodies = Vec::new();
    // Block names are deduplicated across functions: two functions can share <entry> without
    // conflicting. The `mut` binding lets the last function's assignment win, which is fine
    // since users only reference blocks by the function they care about.
    let mut seen_block_names = std::collections::HashSet::new();
    let mut block_predecls: Vec<proc_macro2::TokenStream> = Vec::new();

    for fn_decl in fns {
        let (predecls, fn_block_idents, body) =
            compile_single_fn(ctx, fn_decl, &global_locals, pcode_root)?;
        all_predecls.extend(predecls);
        for b in fn_block_idents {
            if seen_block_names.insert(b.to_string()) {
                block_predecls.push(quote! { let mut #b: #pcode_root::value::BlockId; });
            }
        }
        all_bodies.push(body);
    }

    Ok(quote! {
        #(#global_varnode_predecls)*
        #(#all_predecls)*
        #(#block_predecls)*
        #(#global_varnode_inits)*
        #(#all_bodies)*
    })
}

fn compile_single_fn(
    ctx: &Expr,
    fn_decl: &FnDecl,
    global_locals: &HashMap<String, LocalKind>,
    pcode_root: &proc_macro2::TokenStream,
) -> syn::Result<(
    Vec<proc_macro2::TokenStream>,
    Vec<proc_macro2::Ident>,
    proc_macro2::TokenStream,
)> {
    let fn_name_str = &fn_decl.name;
    let fn_ident = format_ident!("{}", fn_name_str);
    let statements = &fn_decl.statements;

    // Require the first statement to be a named LabelDecl (entry block).
    let first = statements.first().ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("fn `{fn_name_str}`: function body cannot be empty"),
        )
    })?;
    let entry_name = match first {
        Statement::LabelDecl {
            label: Label::Named { name: n, .. },
            ..
        } => n.clone(),
        _ => {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("fn `{fn_name_str}`: first statement must be a named label like `<entry>`"),
            ));
        }
    };
    let entry_ident = format_ident!("{}", entry_name);

    // Collect named-block labels and SSA variables for predecls.
    let named_blocks: Vec<proc_macro2::Ident> = statements
        .iter()
        .filter_map(|s| {
            if let Statement::LabelDecl {
                label: Label::Named { name: n, .. },
                ..
            } = s
            {
                Some(format_ident!("{}", n))
            } else {
                None
            }
        })
        .collect();

    let exposed_ssas: Vec<proc_macro2::Ident> = statements
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

    let local_decls: Vec<proc_macro2::Ident> = statements
        .iter()
        .filter_map(|s| {
            if let Statement::LocalDecl { name, .. } = s {
                Some(format_ident!("{}", name))
            } else {
                None
            }
        })
        .collect();

    // Non-block predecls: FunctionId, exposed SSAs, VarnodeId locals.
    // Block predecls are returned separately so compile_fn_program can deduplicate them
    // across sibling functions that share block names (e.g. both have <entry>).
    let mut predecls: Vec<proc_macro2::TokenStream> = Vec::new();
    predecls.push(quote! { let #fn_ident: #pcode_root::value::FunctionId; });
    for ssa_ident in &exposed_ssas {
        predecls.push(quote! { let #ssa_ident: #pcode_root::value::InstructionId; });
    }
    for local_ident in &local_decls {
        predecls.push(quote! { let #local_ident: #pcode_root::value::VarnodeId; });
    }

    // Build the statement emissions.
    let mut locals: HashMap<String, LocalKind> = global_locals.clone();
    let mut emitted: Vec<proc_macro2::TokenStream> = Vec::new();

    // Emit statements (skip index 0 — the entry label is created explicitly below).
    for (index, stmt) in statements.iter().enumerate().skip(1) {
        emit_statement(stmt, index, &mut locals, &mut emitted, pcode_root)?;
    }

    let body = quote! {
        {
            use ::std::borrow::Cow;
            use #pcode_root::value::Value as _;
            use #pcode_root::value::Renameable as _;

            // Create the function.
            #fn_ident = {
                let __qcode_fn_id = #pcode_root::value::Function::make(&mut (#ctx), Cow::Borrowed(#fn_name_str))
                    .expect("qcode fn: function name conflict");
                __qcode_fn_id.id
            };

            // Create the entry block and builder.
            let __qcode_root_id = {
                #pcode_root::value::BasicBlock::make(&mut (#ctx))
                    .with_name(Cow::Borrowed(#entry_name))
                    .expect("qcode fn: entry block name conflict")
                    .id
            };
            #entry_ident = __qcode_root_id;

            {
                let mut __qcode_builder = #pcode_root::builder::Builder::from_block(
                    #pcode_root::value::BasicBlock::from_id_mut(&mut (#ctx), __qcode_root_id)
                );

                #(#emitted)*
            }

            #pcode_root::value::Function::from_id_mut(&mut (#ctx), #fn_ident)
                .set_root(__qcode_root_id)
                .unwrap();
        }
    };

    Ok((predecls, named_blocks, body))
}

/// New context-based statement compilation.
/// The program must start with a named label like `<block>` which creates the block.
pub(crate) fn compile_qcode_from_statements_ctx(
    ctx: &Expr,
    statements: &[Statement],
    pcode_root: &proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    // Split off any leading varnode declarations that precede the entry block label.
    let preamble_end = statements
        .iter()
        .position(|s| !matches!(s, Statement::LocalDecl { .. }))
        .unwrap_or(statements.len());
    let (preamble_varnodes, body_statements) = statements.split_at(preamble_end);

    if body_statements.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "qcode program cannot be empty",
        ));
    }

    let entry_name = match body_statements.first() {
        Some(Statement::LabelDecl {
            label: Label::Named { name: n, .. },
            ..
        }) => n.clone(),
        _ => {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "statement program must start with a named label, e.g. `<block>`",
            ));
        }
    };
    let entry_ident = format_ident!("{}", entry_name);

    // Emit preamble varnode declarations against the context (no builder yet).
    let mut preamble_predecls: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut preamble_inits: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut locals: HashMap<String, LocalKind> = HashMap::new();

    for stmt in preamble_varnodes {
        let Statement::LocalDecl {
            name, size_bytes, ..
        } = stmt
        else {
            unreachable!()
        };
        let ident = format_ident!("__qcode_local_{}", name);
        let outer_ident = format_ident!("{}", name);
        let size = *size_bytes;
        preamble_predecls.push(quote! { let #outer_ident: #pcode_root::value::VarnodeId; });
        preamble_inits.push(quote! {
            let #ident = {
                use ::std::borrow::Cow;
                use #pcode_root::value::Renameable as _;
                let __name = (#ctx).get_unique_name(Cow::Borrowed(#name));
                let __space = (#ctx).make_named_temp_space(__name.clone());
                let __id = #pcode_root::value::Varnode::make(&mut (#ctx), 0, #size, __space).id;
                #pcode_root::value::Varnode::from_id_mut(&mut (#ctx), __id)
                    .rename(__name)
                    .expect("qcode: varnode name conflict");
                __id
            };
            #outer_ident = #ident;
        });
        locals.insert(name.clone(), LocalKind::Varnode(ident));
    }

    // Collect ALL named labels for predecls (including entry).
    let named_labels: Vec<proc_macro2::Ident> = body_statements
        .iter()
        .filter_map(|s| {
            if let Statement::LabelDecl {
                label: Label::Named { name: n, .. },
                ..
            } = s
            {
                Some(format_ident!("{}", n))
            } else {
                None
            }
        })
        .collect();

    let exposed_ssas: Vec<proc_macro2::Ident> = body_statements
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

    let local_decls: Vec<proc_macro2::Ident> = body_statements
        .iter()
        .filter_map(|s| {
            if let Statement::LocalDecl { name, .. } = s {
                Some(format_ident!("{}", name))
            } else {
                None
            }
        })
        .collect();

    let mut emitted = Vec::new();

    // Skip the entry label — the block is created explicitly below.
    for (index, stmt) in body_statements.iter().enumerate().skip(1) {
        emit_statement(stmt, index, &mut locals, &mut emitted, pcode_root)?;
    }

    let mut predecls: Vec<proc_macro2::TokenStream> = Vec::new();
    for label in &named_labels {
        predecls.push(quote! { let #label: #pcode_root::value::BlockId; });
    }
    for ssa in &exposed_ssas {
        predecls.push(quote! { let #ssa: #pcode_root::value::InstructionId; });
    }
    for local_ident in &local_decls {
        predecls.push(quote! { let #local_ident: #pcode_root::value::VarnodeId; });
    }

    Ok(quote! {
        #(#preamble_predecls)*
        #(#predecls)*
        #(#preamble_inits)*
        {
            use ::std::borrow::Cow;
            use #pcode_root::value::Value as _;
            use #pcode_root::value::Renameable as _;
            let __qcode_entry_id = {
                #pcode_root::value::BasicBlock::make(&mut (#ctx))
                    .with_name(Cow::Borrowed(#entry_name))
                    .expect("qcode: block name conflict")
                    .id
            };
            #entry_ident = __qcode_entry_id;
            let mut __qcode_builder = #pcode_root::builder::Builder::from_block(
                #pcode_root::value::BasicBlock::from_id_mut(&mut (#ctx), __qcode_entry_id)
            );
            #(#emitted)*
        }
    })
}

/// Shared statement emitter used by both statement-mode and function-mode.
fn emit_statement(
    statement: &Statement,
    index: usize,
    locals: &mut HashMap<String, LocalKind>,
    emitted: &mut Vec<proc_macro2::TokenStream>,
    pcode_root: &proc_macro2::TokenStream,
) -> syn::Result<()> {
    match statement {
        Statement::LocalDecl {
            name, size_bytes, ..
        } => {
            let ident = format_ident!("__qcode_local_{}", name);
            let outer_ident = format_ident!("{}", name);
            let size = *size_bytes;
            emitted.push(quote! {
                let #ident = __qcode_builder
                    .make_named_temp(Cow::Borrowed(#name), #size);
                #outer_ident = #ident;
            });
            locals.insert(name.clone(), LocalKind::Varnode(ident));
        }

        Statement::Assign {
            name, expose, expr, ..
        } => {
            let ident = format_ident!("__qcode_local_{}", name);
            let value_tokens = lower_expr(expr, locals, pcode_root)?;
            if *expose {
                let outer_ident = format_ident!("{}", name);
                emitted.push(quote! {
                    let #ident = #value_tokens;
                    let _ = #pcode_root::value::Instruction::from_id_mut(
                        __qcode_builder.context_mut(), #ident
                    ).rename(Cow::Borrowed(#name));
                    #outer_ident = #ident;
                });
            } else {
                emitted.push(quote! {
                    let #ident = #value_tokens;
                    let _ = #pcode_root::value::Instruction::from_id_mut(
                        __qcode_builder.context_mut(), #ident
                    ).rename(Cow::Borrowed(#name));
                });
            }
            locals.insert(name.clone(), LocalKind::Instruction(ident));
        }

        Statement::Expr(expr) => {
            let value_tokens = lower_expr(expr, locals, pcode_root)?;
            let ident = format_ident!("__qcode_result_{}", index);
            emitted.push(quote! {
                let #ident = #value_tokens;
            });
        }

        Statement::LabelDecl { label, .. } => match label {
            Label::Named { name, .. } => {
                let outer_ident = format_ident!("{}", name);
                emitted.push(quote! {
                    {
                        let __qcode_blk = __qcode_builder
                            .get_or_make_local_label(Cow::Borrowed(#name));
                        __qcode_builder.switch_to_block(__qcode_blk);
                        #outer_ident = __qcode_blk;
                    }
                });
            }
            Label::Address { value: addr, .. } => {
                emitted.push(quote! {
                    {
                        let __qcode_blk = __qcode_builder.get_or_make_block(#addr);
                        __qcode_builder.switch_to_block(__qcode_blk);
                    }
                });
            }
        },

        Statement::Branch { target, .. } => {
            let ts = emit_branch_stmt(target, pcode_root);
            emitted.push(ts);
        }

        Statement::BranchInd { ptr, .. } => {
            let ptr_tokens = lower_atom(ptr, None, AtomContext::Pointer, locals, pcode_root)?;
            emitted.push(quote! {
                {
                    let __qcode_ptr = #ptr_tokens;
                    __qcode_builder.push_branchind(__qcode_ptr);
                }
            });
        }

        Statement::CBranch {
            condition,
            target,
            fallthrough,
            ..
        } => {
            let cond_tokens =
                lower_atom(condition, None, AtomContext::Arithmetic, locals, pcode_root)?;
            let target_ts = label_to_block_id(target, pcode_root);
            let fallthrough_ts = label_to_block_id(fallthrough, pcode_root);
            emitted.push(quote! {
                {
                    let __qcode_cond = #cond_tokens;
                    let __qcode_target = #target_ts;
                    let __qcode_fallthrough = #fallthrough_ts;
                    __qcode_builder.push_cbranch(__qcode_cond, __qcode_target, __qcode_fallthrough);
                    __qcode_builder.switch_to_block(__qcode_fallthrough);
                }
            });
        }

        Statement::Call { target, .. } => {
            let target_ts = match target {
                Label::Named { name, .. } => quote! {
                    __qcode_builder.get_or_make_local_function(Cow::Borrowed(#name))
                },
                Label::Address { .. } => {
                    return Err(syn::Error::new(
                        proc_macro2::Span::call_site(),
                        "call with address target is not supported",
                    ));
                }
            };
            emitted.push(quote! {
                {
                    let __qcode_target = #target_ts;
                    __qcode_builder.push_call(__qcode_target);
                }
            });
        }

        Statement::CallInd { ptr, .. } => {
            let ptr_tokens = lower_atom(ptr, None, AtomContext::Pointer, locals, pcode_root)?;
            emitted.push(quote! {
                {
                    let __qcode_ptr = #ptr_tokens;
                    __qcode_builder.push_call_ind(__qcode_ptr);
                }
            });
        }

        Statement::Return { ptr, .. } => {
            let ptr_tokens = lower_atom(ptr, None, AtomContext::Pointer, locals, pcode_root)?;
            emitted.push(quote! {
                {
                    let __qcode_ptr = #ptr_tokens;
                    __qcode_builder.push_return(__qcode_ptr);
                }
            });
        }
    }
    Ok(())
}

/// Emit code that obtains a `BlockId` from a `Label`, for use in branch targets.
fn label_to_block_id(
    label: &Label,
    _pcode_root: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match label {
        Label::Named { name, .. } => quote! {
            __qcode_builder.get_or_make_local_label(Cow::Borrowed(#name))
        },
        Label::Address { value: addr, .. } => quote! {
            __qcode_builder.get_or_make_block(#addr)
        },
    }
}

/// Emit a branch statement.
fn emit_branch_stmt(
    target: &Label,
    _pcode_root: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match target {
        Label::Named { name, .. } => quote! {
            {
                let __qcode_target = __qcode_builder.get_or_make_local_label(Cow::Borrowed(#name));
                __qcode_builder.push_branch(__qcode_target);
            }
        },
        Label::Address { value: addr, .. } => quote! {
            {
                let __qcode_target = __qcode_builder.get_or_make_block(#addr);
                __qcode_builder.push_branch(__qcode_target);
            }
        },
    }
}

fn lower_expr(
    expr: &ExprNode,
    locals: &HashMap<String, LocalKind>,
    pcode_root: &proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    match expr {
        ExprNode::Atom(atom) => lower_atom(atom, None, AtomContext::Arithmetic, locals, pcode_root),

        ExprNode::Unop { op, src } => {
            let src_tokens = lower_atom(src, None, AtomContext::Arithmetic, locals, pcode_root)?;

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
            // Compile-time size check when both operands have explicit annotations.
            let needs_runtime_check = match (lhs.size_bytes, rhs.size_bytes) {
                (Some(ls), Some(rs)) => {
                    if ls != rs {
                        return Err(syn::Error::new(
                            proc_macro2::Span::call_site(),
                            format!("qcode size mismatch: lhs is {ls} bytes, rhs is {rs} bytes"),
                        ));
                    }
                    false
                }
                _ => true,
            };

            let lhs_size = size_hint_tokens(lhs, locals, pcode_root)?;
            let rhs_size = size_hint_tokens(rhs, locals, pcode_root)?;

            let lhs_tokens = lower_atom(
                lhs,
                rhs_size.clone(),
                AtomContext::Arithmetic,
                locals,
                pcode_root,
            )?;
            let rhs_tokens = lower_atom(
                rhs,
                lhs_size.clone(),
                AtomContext::Arithmetic,
                locals,
                pcode_root,
            )?;

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

            let size_assert = if needs_runtime_check {
                quote! {
                    let __qcode_lhs_size = __qcode_builder.context().get_value(__qcode_lhs).size();
                    let __qcode_rhs_size = __qcode_builder.context().get_value(__qcode_rhs).size();
                    assert_eq!(
                        __qcode_lhs_size,
                        __qcode_rhs_size,
                        "qcode size mismatch in binary expression: lhs={} rhs={}",
                        __qcode_lhs_size,
                        __qcode_rhs_size
                    );
                }
            } else {
                quote! {}
            };

            Ok(quote! {
                {
                    let __qcode_lhs = #lhs_tokens;
                    let __qcode_rhs = #rhs_tokens;
                    #size_assert
                    #call
                }
            })
        }

        ExprNode::Cast {
            op,
            size_bytes,
            src,
        } => {
            let src_tokens = lower_atom(src, None, AtomContext::Arithmetic, locals, pcode_root)?;
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
            let ptr_tokens = lower_atom(ptr, None, AtomContext::Pointer, locals, pcode_root)?;
            let size = *size_bytes;

            Ok(quote! {
                {
                    let __qcode_ptr = #ptr_tokens;
                    let __qcode_load_space = #pcode_root::value::ValueRef::from_id(
                            __qcode_builder.context(),
                            __qcode_ptr,
                        )
                        .space()
                        .map(|s| s.id)
                        .unwrap_or(__qcode_builder.context().default_space);
                    let __qcode_value = __qcode_builder
                        .push_load::<false>(
                            __qcode_ptr,
                            #size,
                            __qcode_load_space,
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

            let ptr_tokens = lower_atom(ptr, None, AtomContext::Pointer, locals, pcode_root)?;
            let src_tokens =
                lower_atom(src, src_size, AtomContext::Arithmetic, locals, pcode_root)?;

            Ok(quote! {
                {
                    let __qcode_ptr = #ptr_tokens;
                    let __qcode_src = #src_tokens;
                    let __qcode_store_space = #pcode_root::value::ValueRef::from_id(
                            __qcode_builder.context(),
                            __qcode_ptr,
                        )
                        .space()
                        .map(|s| s.id)
                        .unwrap_or(__qcode_builder.context().default_space);
                    __qcode_builder
                        .push_store(__qcode_src, __qcode_ptr, __qcode_store_space)
                        .id
                }
            })
        }

        ExprNode::FuncCall { op, args } => {
            let call = match op.as_str() {
                "nan" => {
                    let src = lower_func1(op, args, locals, pcode_root)?;
                    quote! {{ let __qcode_src = #src; __qcode_builder.push_is_nan(__qcode_src).id }}
                }
                "popcount" => {
                    let src = lower_func1(op, args, locals, pcode_root)?;
                    quote! {{ let __qcode_src = #src; __qcode_builder.push_popcount(__qcode_src, 1).id }}
                }
                "lzcount" => {
                    let src = lower_func1(op, args, locals, pcode_root)?;
                    quote! {{ let __qcode_src = #src; __qcode_builder.push_lzcount(__qcode_src, 1).id }}
                }
                "carry" => {
                    let (lhs, rhs) = lower_func2(op, args, locals, pcode_root)?;
                    quote! {{ let __qcode_lhs = #lhs; let __qcode_rhs = #rhs; __qcode_builder.push_carry(__qcode_lhs, __qcode_rhs).id }}
                }
                "scarry" => {
                    let (lhs, rhs) = lower_func2(op, args, locals, pcode_root)?;
                    quote! {{ let __qcode_lhs = #lhs; let __qcode_rhs = #rhs; __qcode_builder.push_scarry(__qcode_lhs, __qcode_rhs).id }}
                }
                "sborrow" => {
                    let (lhs, rhs) = lower_func2(op, args, locals, pcode_root)?;
                    quote! {{ let __qcode_lhs = #lhs; let __qcode_rhs = #rhs; __qcode_builder.push_sborrow(__qcode_lhs, __qcode_rhs).id }}
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

fn lower_func1(
    name: &str,
    args: &[TypedAtom],
    locals: &HashMap<String, LocalKind>,
    pcode_root: &proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    if args.len() != 1 {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("{name} expects exactly 1 argument"),
        ));
    }
    lower_atom(&args[0], None, AtomContext::Arithmetic, locals, pcode_root)
}

fn lower_func2(
    name: &str,
    args: &[TypedAtom],
    locals: &HashMap<String, LocalKind>,
    pcode_root: &proc_macro2::TokenStream,
) -> syn::Result<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
    if args.len() != 2 {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("{name} expects exactly 2 arguments"),
        ));
    }
    Ok((
        lower_atom(&args[0], None, AtomContext::Arithmetic, locals, pcode_root)?,
        lower_atom(&args[1], None, AtomContext::Arithmetic, locals, pcode_root)?,
    ))
}

fn lower_atom(
    typed: &TypedAtom,
    size_hint: Option<proc_macro2::TokenStream>,
    ctx: AtomContext,
    locals: &HashMap<String, LocalKind>,
    pcode_root: &proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    match &typed.atom {
        // External capture `{name}` — resolve to local if declared in this call, else outer scope.
        Atom::External(name) => {
            if let Some(kind) = locals.get(name) {
                // Resolve to the in-call local.
                lower_local_ident(kind.ident(), typed.size_bytes, pcode_root)
            } else {
                // Fall back to outer Rust scope.
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
        }

        Atom::Local(name) => {
            let Some(kind) = locals.get(name) else {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("unknown local identifier '{name}'"),
                ));
            };
            if matches!(ctx, AtomContext::Arithmetic) && kind.is_varnode() {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!(
                        "'{name}' is a varnode (pointer); use `&{name}` for addressof or `load(sz, {name})` to dereference its value"
                    ),
                ));
            }
            lower_local_ident(kind.ident(), typed.size_bytes, pcode_root)
        }

        Atom::AddressOf(name) => {
            let Some(kind) = locals.get(name) else {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("unknown local identifier '{name}' in addressof expression"),
                ));
            };
            if !kind.is_varnode() {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!(
                        "'&{name}' is not valid: `&` (addressof) can only be applied to varnodes, but '{name}' is an SSA value"
                    ),
                ));
            }
            let local_ident = kind.ident();
            if let Some(expected_size) = typed.size_bytes {
                Ok(quote! {
                    {
                        let __qcode_addr_size = #pcode_root::value::Varnode::from_id(
                            __qcode_builder.context(), #local_ident
                        ).space().addr_size;
                        assert_eq!(
                            __qcode_addr_size,
                            #expected_size,
                            "qcode size mismatch for `&{}`: expected {} bytes, got {} bytes (space addr_size)",
                            #name,
                            #expected_size,
                            __qcode_addr_size
                        );
                        let __qcode_value_id: #pcode_root::value::ValueId = (#local_ident).into();
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

fn lower_local_ident(
    local_ident: &proc_macro2::Ident,
    size_bytes: Option<usize>,
    pcode_root: &proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    if let Some(expected_size) = size_bytes {
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

fn size_hint_tokens(
    typed: &TypedAtom,
    locals: &HashMap<String, LocalKind>,
    pcode_root: &proc_macro2::TokenStream,
) -> syn::Result<Option<proc_macro2::TokenStream>> {
    if let Some(explicit) = typed.size_bytes {
        return Ok(Some(quote! { #explicit }));
    }

    match &typed.atom {
        Atom::External(name) => {
            if let Some(kind) = locals.get(name) {
                let local_ident = kind.ident();
                Ok(Some(quote! {
                    {
                        let __qcode_value_id: #pcode_root::value::ValueId = (#local_ident).into();
                        __qcode_builder.context().get_value(__qcode_value_id).size()
                    }
                }))
            } else {
                let ident = format_ident!("{}", name);
                Ok(Some(quote! {
                    {
                        let __qcode_value_id: #pcode_root::value::ValueId = (#ident).into();
                        __qcode_builder.context().get_value(__qcode_value_id).size()
                    }
                }))
            }
        }

        Atom::Local(name) => {
            let Some(kind) = locals.get(name) else {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("unknown local identifier '{name}'"),
                ));
            };
            let local_ident = kind.ident();
            Ok(Some(quote! {
                {
                    let __qcode_value_id: #pcode_root::value::ValueId = (#local_ident).into();
                    __qcode_builder.context().get_value(__qcode_value_id).size()
                }
            }))
        }

        Atom::AddressOf(name) => {
            let Some(kind) = locals.get(name) else {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    format!("unknown local identifier '{name}'"),
                ));
            };
            let local_ident = kind.ident();
            Ok(Some(quote! {
                #pcode_root::value::Varnode::from_id(
                    __qcode_builder.context(), #local_ident
                ).space().addr_size
            }))
        }

        Atom::Int(_) => Ok(None),
    }
}
