// [AI Generated]
use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use quote::quote;
use syn::{
    Expr, LitStr, Token,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
};

mod lower;

struct QCodeInput {
    builder: Expr,
    program: LitStr,
}

impl Parse for QCodeInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let fork = input.fork();
        if let Ok(_program) = fork.parse::<LitStr>()
            && fork.is_empty()
        {
            let program = input.parse::<LitStr>()?;
            return Ok(Self {
                builder: parse_quote!(builder),
                program,
            });
        }

        let builder: Expr = input.parse()?;
        let _comma: Token![,] = input.parse()?;
        let program: LitStr = input.parse()?;

        Ok(Self { builder, program })
    }
}

#[proc_macro]
pub fn qcode(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as QCodeInput);

    match compile_qcode_from_str(&input.builder, &input.program.value()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn compile_qcode_from_str(builder: &Expr, program: &str) -> syn::Result<proc_macro2::TokenStream> {
    let statements = qcode_parser::qcode_from_str(program)
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

    lower::compile_qcode_from_statements(builder, &statements, &pcode_root)
}
