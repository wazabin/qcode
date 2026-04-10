// [AI Generated]
use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use qcode_parser::ast::Program;
use quote::quote;
use syn::{
    Expr, LitStr, Token,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
};

mod lower;

struct QCodeInput {
    /// Either a builder expression (statement programs) or a context expression (fn programs).
    expr: Expr,
    program: LitStr,
}

impl Parse for QCodeInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let fork = input.fork();
        if let Ok(_program) = fork.parse::<LitStr>()
            && fork.is_empty()
        {
            // Only a string literal — defer default name choice until we know the program type.
            let program = input.parse::<LitStr>()?;
            return Ok(Self {
                expr: parse_quote!(__qcode_default_placeholder__),
                program,
            });
        }

        let expr: Expr = input.parse()?;
        let _comma: Token![,] = input.parse()?;
        let program: LitStr = input.parse()?;

        Ok(Self { expr, program })
    }
}

#[proc_macro]
pub fn qcode(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as QCodeInput);

    match compile_qcode_from_str(&input.expr, &input.program.value()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn compile_qcode_from_str(expr: &Expr, program: &str) -> syn::Result<proc_macro2::TokenStream> {
    let parsed = qcode_parser::qcode_from_str(program)
        .map_err(|err| syn::Error::new(proc_macro2::Span::call_site(), err.to_string()))?;

    let pcode_root = match crate_name("qcode") {
        Ok(FoundCrate::Itself) => quote!(crate),
        Ok(FoundCrate::Name(name)) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
        Err(err) => {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("could not resolve `qcode` crate for qcode macro: {err}"),
            ));
        }
    };

    match parsed {
        Program::Statements(stmts) => {
            if is_placeholder(expr) {
                // No explicit expression: deprecated builder-based form.
                let builder: Expr = parse_quote!(builder);
                lower::compile_qcode_from_statements(&builder, &stmts, &pcode_root)
            } else {
                // Explicit context expression: new ctx + leading-label form.
                lower::compile_qcode_from_statements_ctx(expr, &stmts, &pcode_root)
            }
        }
        Program::Functions(fns) => {
            // Use `ctx` as default when no explicit expression was given.
            let ctx: Expr = if is_placeholder(expr) {
                parse_quote!(ctx)
            } else {
                expr.clone()
            };
            lower::compile_fn_program(&ctx, &fns, &pcode_root)
        }
    }
}

/// Returns true if the expression is the sentinel placeholder we emit when no
/// explicit expression was provided by the user.
fn is_placeholder(expr: &Expr) -> bool {
    if let Expr::Path(p) = expr {
        p.path.is_ident("__qcode_default_placeholder__")
    } else {
        false
    }
}
