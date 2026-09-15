//! The `LangRef` derive: exposes an item's rustdoc and `#[langref(…)]`
//! attributes as a runtime table for the language reference generator.
//!
//! ```ignore
//! #[derive(LangRef)]
//! #[langref(category = "Memory")]
//! pub enum Mnemonic {
//!     /// Load a value from a memory space.
//!     #[langref(syntax = "load(<space>:<bytes>, <ptr>)", example = "i32 %v = load(ram:4, i64 @p);")]
//!     Load(Load),
//! }
//! ```
//!
//! On an enum this emits `impl LangRef for Enum { const ENTRIES: … }` with one
//! `InsnDoc` per variant; every variant must carry a `syntax` and a `category`
//! (its own, or the enum's), so an undocumented variant is a compile error
//! rather than a missing page. `syntax` and `example` may repeat. On a struct
//! (an intrinsic definition) it emits `impl LangRefEntry`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Attribute, Data, DeriveInput, Expr, ExprLit, Lit, LitStr, Meta};

/// Fields collected from one `#[langref(…)]` attribute list.
#[derive(Default)]
struct Fields {
    category: Option<LitStr>,
    syntax: Vec<LitStr>,
    examples: Vec<LitStr>,
}

fn parse_fields(attrs: &[Attribute]) -> syn::Result<Fields> {
    let mut fields = Fields::default();
    for attr in attrs.iter().filter(|a| a.path().is_ident("langref")) {
        attr.parse_nested_meta(|meta| {
            let value: LitStr = meta.value()?.parse()?;
            if meta.path.is_ident("category") {
                fields.category = Some(value);
            } else if meta.path.is_ident("syntax") {
                fields.syntax.push(value);
            } else if meta.path.is_ident("example") {
                fields.examples.push(value);
            } else {
                return Err(meta.error("expected `category`, `syntax` or `example`"));
            }
            Ok(())
        })?;
    }
    Ok(fields)
}

/// The `///` lines of `attrs`, joined as one markdown string with the single
/// leading space rustdoc keeps stripped.
fn doc_comment(attrs: &[Attribute]) -> String {
    let mut lines = Vec::new();
    for attr in attrs.iter().filter(|a| a.path().is_ident("doc")) {
        if let Meta::NameValue(nv) = &attr.meta
            && let Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) = &nv.value
        {
            let s = s.value();
            lines.push(s.strip_prefix(' ').unwrap_or(&s).to_owned());
        }
    }
    lines.join("\n").trim().to_owned()
}

fn entry(
    krate: &TokenStream,
    name: &str,
    fallback_category: Option<&LitStr>,
    fields: &Fields,
    doc: &str,
    span: proc_macro2::Span,
) -> syn::Result<TokenStream> {
    let missing =
        |what: &str| syn::Error::new(span, format!("`{name}` needs `#[langref({what} = …)]`"));
    let category = fields
        .category
        .as_ref()
        .or(fallback_category)
        .ok_or_else(|| missing("category"))?;
    if fields.syntax.is_empty() {
        return Err(missing("syntax"));
    }
    let syntax = &fields.syntax;
    let examples = &fields.examples;
    Ok(quote! {
        #krate::langref::InsnDoc {
            name: #name,
            category: #category,
            syntax: &[#(#syntax),*],
            examples: &[#(#examples),*],
            doc: #doc,
        }
    })
}

pub fn derive(input: DeriveInput, krate: TokenStream) -> syn::Result<TokenStream> {
    let ident = &input.ident;
    let top = parse_fields(&input.attrs)?;
    match &input.data {
        Data::Enum(data) => {
            let entries = data
                .variants
                .iter()
                .map(|v| {
                    let fields = parse_fields(&v.attrs)?;
                    let name = v.ident.to_string();
                    let doc = doc_comment(&v.attrs);
                    entry(
                        &krate,
                        &name,
                        top.category.as_ref(),
                        &fields,
                        &doc,
                        v.ident.span(),
                    )
                })
                .collect::<syn::Result<Vec<_>>>()?;
            let intro = doc_comment(&input.attrs);
            let category = match &top.category {
                Some(c) => quote!(Some(#c)),
                None => quote!(None),
            };
            Ok(quote! {
                impl #krate::langref::LangRef for #ident {
                    const ENTRIES: &'static [#krate::langref::InsnDoc] = &[#(#entries),*];
                    const CATEGORY: Option<&'static str> = #category;
                    const INTRO: &'static str = #intro;
                }
            })
        }
        Data::Struct(_) => {
            let entry = entry(
                &krate,
                &ident.to_string(),
                None,
                &top,
                &doc_comment(&input.attrs),
                ident.span(),
            )?;
            Ok(quote! {
                impl #krate::langref::LangRefEntry for #ident {
                    const ENTRY: #krate::langref::InsnDoc = #entry;
                }
            })
        }
        Data::Union(_) => Err(syn::Error::new(
            ident.span(),
            "LangRef cannot be derived for unions",
        )),
    }
}
