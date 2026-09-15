//! Self-describing language reference.
//!
//! Every instruction, operator and intrinsic documents itself where it is
//! defined: its rustdoc is the prose, and a `#[langref(…)]` attribute gives
//! its textual syntax and examples. The [`LangRef`](wazabin_qcode_macro::LangRef)
//! derive collects both into [`InsnDoc`] tables, which [`entries`] returns in
//! reference order. The `qcode` language reference is rendered from these
//! tables, so it cannot drift from the code: a variant without a `syntax` does
//! not compile, and intrinsics implement [`Intrinsic::doc`].
//!
//! [`Intrinsic::doc`]: crate::value::insn::Intrinsic::doc

use crate::{
    context::Context,
    types::{AggregateField, TypeRequest},
    value::insn::{FloatBinop, IntBinop, IntrinsicId, Mnemonic, Unop},
};

/// One reference entry: an instruction, an operator, or an intrinsic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsnDoc {
    /// The Rust name of the item (`"Load"`, `"Add"`, `"Rol"`).
    pub name: &'static str,
    /// The reference section this entry belongs to.
    pub category: &'static str,
    /// The textual forms, with placeholders for operands (`T` a type, `v`,
    /// `a`, `b` values, `bb` a block label, `N` a byte width). Non-empty.
    pub syntax: &'static [&'static str],
    /// Concrete statements in QCode text syntax. Each must parse.
    pub examples: &'static [&'static str],
    /// The item's rustdoc, verbatim markdown.
    pub doc: &'static str,
}

/// An enum whose every variant is a reference entry.
pub trait LangRef {
    const ENTRIES: &'static [InsnDoc];
    /// The enum-level `#[langref(category = …)]`, when the enum is one section.
    const CATEGORY: Option<&'static str>;
    /// The enum's own rustdoc: the section's introduction.
    const INTRO: &'static str;
}

/// A single item (an intrinsic definition) that is a reference entry.
pub trait LangRefEntry {
    const ENTRY: InsnDoc;
}

/// Struct declarations and callees every example may refer to.
pub const EXAMPLE_PRELUDE: &str = "\
type point { x: 4, y: 4 }
fn callee:
<entry @a:i32 @b:i32 @ra:i64>
    return at @ra;
lambda inc:
<entry @v:i32>
    %r = @v + 1;
    return %r;
lambda inc64:
<entry @v:i64>
    %r = @v + 1;
    return %r;
lambda step:
<entry @acc:i64 @e:i64>
    %r = @acc + @e;
    return %r;
";

/// The block every example is placed in: its parameters and the two float
/// values (types carry only a width, so a float operand is one produced by
/// a float instruction) are the free operands the examples use.
pub const EXAMPLE_ENTRY: &str = "\
<entry @x:i32 @y:i32 @p:i64 @n:i64>
    f64 %f = int2float(f64, i32 @x);
    f64 %g = int2float(f64, i32 @y);";

/// The blocks the control-flow examples target.
pub const EXAMPLE_TARGETS: &str = "\
<next>
    return at @p;
<other>
    return at @p;
<loop @i:i32 @acc:i32>
    return at @p;
";

/// The complete program an example is checked in: [`EXAMPLE_PRELUDE`], then
/// `example` as the body of [`EXAMPLE_ENTRY`] in a function that also owns
/// [`EXAMPLE_TARGETS`]. An example that does not end its block falls through
/// to `<next>`.
pub fn example_program(example: &str) -> String {
    let mut program = String::from(EXAMPLE_PRELUDE);
    program.push_str("fn example:\n");
    program.push_str(EXAMPLE_ENTRY);
    program.push('\n');
    for line in example.lines() {
        program.push_str("    ");
        program.push_str(line);
        program.push('\n');
    }
    if !ends_block(example) {
        program.push_str("    goto <next>;\n");
    }
    program.push_str(EXAMPLE_TARGETS);
    program
}

/// Whether the example's last statement is a terminator.
fn ends_block(example: &str) -> bool {
    let last = example.lines().last().unwrap_or("").trim_start();
    [
        "goto ",
        "if ",
        "switch ",
        "call ",
        "tailcall ",
        "return",
        "badinsn",
    ]
    .iter()
    .any(|kw| last.starts_with(kw))
}

/// A context the examples lower in: one with the sequence types the
/// intrinsic examples resolve to. An intrinsic's result type must exist
/// before it is applied; a lift publishes these as it goes, the text lowerer
/// does not.
pub fn example_context() -> Context<'static> {
    let mut ctx = Context::new();
    let types = &mut ctx.shared.types;
    for bytes in [1, 2, 4, 8, 16] {
        types.get_or_make_int(bytes);
    }
    let i32_ty = types.get_or_make_int(4);
    let i64_ty = types.get_or_make_int(8);
    let index_elem = vec![
        AggregateField::new("index", i64_ty),
        AggregateField::new("elem", i64_ty),
    ];
    types.create_requested_types(&[
        TypeRequest::list(i32_ty, None),
        TypeRequest::list(i64_ty, None),
        TypeRequest::array(i64_ty, 1),
        TypeRequest::aggregate(index_elem),
    ]);
    ctx
}

/// The introduction of each section that is one enum: its category and
/// rustdoc.
pub fn category_intros() -> impl Iterator<Item = (&'static str, &'static str)> {
    [
        (Unop::CATEGORY, Unop::INTRO),
        (IntBinop::CATEGORY, IntBinop::INTRO),
        (FloatBinop::CATEGORY, FloatBinop::INTRO),
    ]
    .into_iter()
    .filter_map(|(category, intro)| Some((category?, intro)))
}

/// Every reference entry, in reference order: instructions, then unary and
/// binary operators, then the registered intrinsics sorted by name.
pub fn entries() -> impl Iterator<Item = &'static InsnDoc> {
    Mnemonic::ENTRIES
        .iter()
        .chain(Unop::ENTRIES)
        .chain(IntBinop::ENTRIES)
        .chain(FloatBinop::ENTRIES)
        .chain(IntrinsicId::all().map(|id| id.desc().doc()))
}
