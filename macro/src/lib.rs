//! The `qcode!` macro.
//!
//! At compile time it parses the QCode source (via [`qcode_parser`]) and emits
//! Rust code that, at run time, lowers the same source into the caller's
//! `Context` through [`qcode_lower`] — the single shared lowering implementation
//! used by tools and the CLI. The macro's only extra job is to bind each declared
//! name (functions, blocks, SSA values, varnodes, block params) to a Rust `let`
//! so callers can reference them after the invocation, and to forward any
//! `{capture}` atoms from the surrounding Rust scope.

use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use quote::{format_ident, quote};
use syn::{
    Expr, LitStr, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

use qcode_parser::ast::{
    Atom, ExprNode, FnDecl, Label, Program, ProgramKind, Statement, TupleField, TypedAtom,
};

struct QCodeInput {
    expr: Expr,
    program: LitStr,
}

impl Parse for QCodeInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let expr: Expr = input.parse()?;
        let _comma: Token![,] = input.parse()?;
        let program: LitStr = input.parse()?;
        Ok(Self { expr, program })
    }
}

#[proc_macro]
pub fn qcode(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as QCodeInput);
    match compile(&input.expr, &input.program.value()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// The set of names a program binds, partitioned by kind, in first-seen order.
#[derive(Default)]
struct Names {
    functions: Vec<String>,
    blocks: Vec<String>,
    ssa: Vec<String>,
    varnodes: Vec<String>,
    temps: Vec<String>,
    block_params: Vec<String>,
    externals: Vec<String>,
}

fn push_unique(set: &mut Vec<String>, name: &str) {
    if !set.iter().any(|n| n == name) {
        set.push(name.to_owned());
    }
}

fn compile(expr: &Expr, source: &str) -> syn::Result<proc_macro2::TokenStream> {
    let err = |e: String| syn::Error::new(proc_macro2::Span::call_site(), e);
    let program = qcode_parser::qcode_from_str(source).map_err(|e| err(e.to_string()))?;

    let mut names = Names::default();
    collect_program(&program, &mut names);

    // `{capture}` atoms that aren't shadowed by a program-defined name reference
    // Rust values in the caller's scope; everything else resolves internally.
    let defined: std::collections::HashSet<&str> = names
        .functions
        .iter()
        .chain(&names.blocks)
        .chain(&names.ssa)
        .chain(&names.varnodes)
        .chain(&names.temps)
        .chain(&names.block_params)
        .map(String::as_str)
        .collect();
    let captures: Vec<&String> = names
        .externals
        .iter()
        .filter(|n| !defined.contains(n.as_str()))
        .collect();

    let qcode = resolve_crate("qcode")?;

    let capture_inserts = captures.iter().map(|name| {
        let ident = format_ident!("{}", name);
        quote! {
            __qcode_ext.insert(
                ::std::string::String::from(#name),
                #qcode::value::ValueId::from(#ident),
            );
        }
    });

    let bind = |ty: proc_macro2::TokenStream,
                getter: proc_macro2::TokenStream,
                items: &[String]|
     -> Vec<proc_macro2::TokenStream> {
        items
            .iter()
            .map(|name| {
                let ident = format_ident!("{}", name);
                quote! {
                    #[allow(unused_variables)]
                    let #ident: #qcode::value::#ty = __qcode_syms.#getter(#name);
                }
            })
            .collect()
    };

    let fn_binds = bind(quote!(FunctionId), quote!(function), &names.functions);
    let block_binds = bind(quote!(BlockId), quote!(block), &names.blocks);
    let ssa_binds = bind(quote!(InstructionId), quote!(ssa), &names.ssa);
    let varnode_binds = bind(quote!(VarnodeId), quote!(varnode), &names.varnodes);
    let temp_binds = bind(quote!(TempId), quote!(temp), &names.temps);
    let param_binds = bind(
        quote!(BlockParamId),
        quote!(block_param),
        &names.block_params,
    );

    // NB: emit bare `let` statements (no wrapping block) — callers rely on the
    // bound names leaking into their scope, exactly as the old macro did.
    Ok(quote! {
        let mut __qcode_ext: ::std::collections::HashMap<
            ::std::string::String,
            #qcode::value::ValueId,
        > = ::std::collections::HashMap::new();
        #(#capture_inserts)*
        let __qcode_syms = #qcode::lower::lower_str_with_externals(&mut (#expr), #source, __qcode_ext)
            .expect("qcode! lowering failed");
        #(#fn_binds)*
        #(#block_binds)*
        #(#ssa_binds)*
        #(#varnode_binds)*
        #(#temp_binds)*
        #(#param_binds)*
        let _ = &__qcode_syms;
    })
}

fn resolve_crate(name: &str) -> syn::Result<proc_macro2::TokenStream> {
    match crate_name(name) {
        Ok(FoundCrate::Itself) => Ok(quote!(crate)),
        Ok(FoundCrate::Name(found)) => {
            let ident = syn::Ident::new(&found, proc_macro2::Span::call_site());
            Ok(quote!(::#ident))
        }
        Err(e) => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("could not resolve `{name}` crate for qcode! macro: {e}"),
        )),
    }
}

fn collect_program(program: &Program, names: &mut Names) {
    match &program.kind {
        ProgramKind::Statements(stmts) => {
            let split = stmts
                .iter()
                .position(|stmt| !matches!(stmt.inner(), Statement::LocalDecl { .. }))
                .unwrap_or(stmts.len());
            collect_statements(&stmts[..split], names, false);
            collect_statements(&stmts[split..], names, true);
        }
        ProgramKind::Functions { varnodes, fns } => {
            collect_statements(varnodes, names, false);
            for fn_decl in fns {
                collect_fn(fn_decl, names);
            }
        }
    }
}

fn collect_fn(fn_decl: &FnDecl, names: &mut Names) {
    push_unique(&mut names.functions, &fn_decl.name);
    collect_statements(&fn_decl.statements, names, true);
}

fn collect_statements(stmts: &[Statement], names: &mut Names, local_temps: bool) {
    for stmt in stmts {
        collect_statement(stmt.inner(), names, local_temps);
    }
}

fn collect_statement(stmt: &Statement, names: &mut Names, local_temps: bool) {
    match stmt {
        Statement::LocalDecl { name, .. } => {
            if local_temps {
                push_unique(&mut names.temps, name);
            } else {
                push_unique(&mut names.varnodes, name);
            }
        }
        Statement::Assign { name, expr, .. } => {
            push_unique(&mut names.ssa, name);
            collect_expr(expr, names);
        }
        Statement::Expr(expr) => collect_expr(expr, names),
        Statement::LabelDecl {
            label: Label::Named { name, params, .. },
            ..
        } => {
            push_unique(&mut names.blocks, name);
            for p in params {
                push_unique(&mut names.block_params, &p.name);
            }
        }
        Statement::LabelDecl { .. } => {}
        Statement::Branch { args, .. } => {
            for (_, atom) in args {
                collect_atom(atom, names);
            }
        }
        Statement::BranchInd { ptr, .. } => collect_atom(ptr, names),
        Statement::CBranch {
            condition,
            target_args,
            fallthrough_args,
            ..
        } => {
            collect_atom(condition, names);
            for (_, atom) in target_args.iter().chain(fallthrough_args) {
                collect_atom(atom, names);
            }
        }
        Statement::Call { args, .. } => {
            for (_, atom) in args {
                collect_atom(atom, names);
            }
        }
        Statement::CallInd { ptr, args, .. } => {
            collect_atom(ptr, names);
            for atom in args {
                collect_atom(atom, names);
            }
        }
        Statement::Return { ptr, value, .. } => {
            collect_atom(ptr, names);
            if let Some(v) = value {
                collect_atom(v, names);
            }
        }
        Statement::ReturnValue { value, .. } => collect_atom(value, names),
        // Operand-less marker: no names to collect.
        Statement::BadInsn { .. } => {}
        Statement::Assert { condition, .. } => collect_atom(condition, names),
        Statement::Commented { inner, .. } => collect_statement(inner, names, local_temps),
    }
}

fn collect_expr(expr: &ExprNode, names: &mut Names) {
    match expr {
        ExprNode::Atom(a) => collect_atom(a, names),
        ExprNode::Unop { src, .. } => collect_atom(src, names),
        ExprNode::Binary { lhs, rhs, .. } => {
            collect_atom(lhs, names);
            collect_atom(rhs, names);
        }
        ExprNode::Cast { src, .. } => collect_atom(src, names),
        ExprNode::Load { ptr, .. } => collect_atom(ptr, names),
        ExprNode::Store { ptr, src, .. } => {
            collect_atom(ptr, names);
            collect_atom(src, names);
        }
        ExprNode::FuncCall { args, .. }
        | ExprNode::Intrinsic { args, .. }
        | ExprNode::Apply { args, .. } => {
            for a in args {
                collect_atom(a, names);
            }
        }
        ExprNode::Map { src, captures, .. } => {
            collect_atom(src, names);
            for a in captures {
                collect_atom(a, names);
            }
        }
        ExprNode::Scan {
            init,
            src,
            captures,
            ..
        } => {
            collect_atom(init, names);
            collect_atom(src, names);
            for a in captures {
                collect_atom(a, names);
            }
        }
        ExprNode::Tuple { fields } => {
            for TupleField { value, .. } in fields {
                collect_atom(value, names);
            }
        }
        ExprNode::Extract { agg, .. } => collect_atom(agg, names),
        ExprNode::Gep { base, .. } => collect_atom(base, names),
        ExprNode::Range { src, .. } => collect_atom(src, names),
    }
}

fn collect_atom(atom: &TypedAtom, names: &mut Names) {
    if let Atom::External(name) = &atom.atom {
        push_unique(&mut names.externals, name);
    }
}
