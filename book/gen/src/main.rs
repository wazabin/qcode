//! Renders `book/src/langref.md` from [`qcode::langref::entries`].
//!
//! Usage: `cargo run -p qcode-langref > book/src/langref.md`. Every heading,
//! syntax line, example and paragraph comes from the instruction definitions
//! in `qcode`, so the page cannot drift from the code; `core/tests/langref.rs`
//! checks the examples.

use std::collections::BTreeSet;
use std::fmt::Write;

use qcode::langref::{
    EXAMPLE_ENTRY, EXAMPLE_PRELUDE, EXAMPLE_TARGETS, InsnDoc, category_intros, entries,
};

fn main() {
    print!("{}", render(&entries().collect::<Vec<_>>()));
}

/// Categories in page order; an entry's category not listed here goes last.
const CATEGORY_ORDER: &[&str] = &[
    "Memory",
    "Control flow",
    "Arithmetic",
    "Unary operators",
    "Integer operators",
    "Float operators",
    "Casts",
    "Bit and flag operations",
    "Aggregates",
    "Sequences",
    "Intrinsics",
    "Extensions",
    "Verification",
];

fn category_rank(category: &str) -> usize {
    CATEGORY_ORDER
        .iter()
        .position(|&c| c == category)
        .unwrap_or(CATEGORY_ORDER.len())
}

fn render(entries: &[&InsnDoc]) -> String {
    let names: BTreeSet<&str> = entries.iter().map(|e| e.name).collect();
    let mut categories: Vec<&str> = entries.iter().map(|e| e.category).collect();
    categories.sort_by_key(|c| category_rank(c));
    categories.dedup();

    let mut out = String::new();
    out.push_str(HEADER);
    let _ = writeln!(
        out,
        "```qcode\n{EXAMPLE_PRELUDE}fn example:\n{EXAMPLE_ENTRY}\n    # the example\n{EXAMPLE_TARGETS}```\n"
    );

    // Heading anchors are assigned in page order, and a repeated heading
    // (`Equal` as an integer and as a float operator) gets a `-N` suffix as
    // mdBook gives it.
    let mut anchors = std::collections::HashMap::new();
    let mut anchor_of = |heading: &str| -> String {
        let base = anchor(heading);
        let n = anchors.entry(base.clone()).or_insert(0usize);
        let anchor = if *n == 0 { base } else { format!("{base}-{n}") };
        *n += 1;
        anchor
    };
    out.push_str("## Contents\n\n");
    for category in &categories {
        let _ = writeln!(out, "- [{category}](#{})", anchor_of(category));
        for entry in entries.iter().filter(|e| e.category == *category) {
            let _ = writeln!(out, "  - [`{}`](#{})", entry.name, anchor_of(entry.name));
        }
    }
    out.push('\n');

    let intros: Vec<_> = category_intros().collect();
    for category in &categories {
        let _ = writeln!(out, "## {category}\n");
        if let Some((_, intro)) = intros.iter().find(|(c, _)| c == category) {
            let _ = writeln!(out, "{}\n", rewrite_links(intro, &names));
        }
        for entry in entries.iter().filter(|e| e.category == *category) {
            render_entry(&mut out, entry, &names);
        }
    }
    out
}

fn render_entry(out: &mut String, entry: &InsnDoc, names: &BTreeSet<&str>) {
    let _ = writeln!(out, "### `{}`\n", entry.name);
    out.push_str("#### Syntax\n\n```qcode\n");
    for syntax in entry.syntax {
        let _ = writeln!(out, "{syntax}");
    }
    out.push_str("```\n\n");
    let (overview, semantics) = split_doc(entry.doc);
    out.push_str("#### Overview\n\n");
    let _ = writeln!(out, "{}\n", rewrite_links(overview, names));
    if !semantics.is_empty() {
        out.push_str("#### Semantics\n\n");
        let _ = writeln!(out, "{}\n", rewrite_links(semantics, names));
    }
    if !entry.examples.is_empty() {
        out.push_str("#### Example\n\n```qcode\n");
        for (i, example) in entry.examples.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            let _ = writeln!(out, "{example}");
        }
        out.push_str("```\n\n");
    }
}

/// The doc's first paragraph is the overview; the rest is the semantics.
fn split_doc(doc: &str) -> (&str, &str) {
    match doc.split_once("\n\n") {
        Some((first, rest)) => (first, rest.trim()),
        None => (doc, ""),
    }
}

/// Rewrites rustdoc intra-doc links for the page: a link to another entry
/// becomes an anchor, any other item reference becomes plain code.
fn rewrite_links(doc: &str, names: &BTreeSet<&str>) -> String {
    let mut out = String::with_capacity(doc.len());
    let mut rest = doc;
    while let Some(start) = rest.find("[`") {
        out.push_str(&rest[..start]);
        rest = &rest[start..];
        let Some(close) = rest.find("`]") else {
            break;
        };
        let text = &rest[2..close];
        rest = &rest[close + 2..];
        // An explicit target `(path)` follows the text; its last segment names
        // the item. Without one, the text itself is the path.
        let target = if rest.starts_with('(') {
            let end = rest.find(')').unwrap_or(rest.len() - 1);
            let target = &rest[1..end];
            rest = &rest[end + 1..];
            target
        } else {
            text
        };
        let item = target.rsplit("::").next().unwrap_or(target);
        let item = item.strip_prefix("macro@").unwrap_or(item);
        match names.get(item) {
            Some(name) => {
                let _ = write!(out, "[`{text}`](#{})", anchor(name));
            }
            None => {
                let _ = write!(out, "`{text}`");
            }
        }
    }
    out.push_str(rest);
    out
}

/// The anchor mdBook gives a heading: lowercased, non-alphanumerics dropped
/// or turned into dashes.
fn anchor(heading: &str) -> String {
    let mut anchor = String::new();
    for c in heading.chars() {
        if c.is_alphanumeric() {
            anchor.extend(c.to_lowercase());
        } else if c == ' ' || c == '-' {
            anchor.push('-');
        }
    }
    anchor
}

const HEADER: &str = "\
# Language reference

<!-- Generated by `cargo run -p qcode-langref`; do not edit. -->

This page lists every QCode instruction, operator and intrinsic with its
textual syntax, its meaning, and an example. It is rendered from the
definitions in the `qcode` crate: the prose is each item's documentation, the
syntax and examples are declared next to it, and the examples are checked by
the crate's tests.

## Conventions

A statement binds a result with `T %name = …`, where `T` is the result type:
`iN` an `N`-bit integer, `fN` an `N`-bit float (accepted on input; types carry
only a width, so floats print as `iN`), `bool` a one-byte truth value. Operands
are `T value`, where a value is an SSA result `%v`, a block parameter `@p`, or
a literal such as `0x10`. Terminators bind nothing.

In syntax lines, `T` is a type, `v`, `a`, `b`, `x`, `k` are operands, `ptr` an
address, `bb` a block label, `N` a width in bytes, and `…` a repetition.

Every example is checked inside the program below, so its free operands are
`@x`, `@y` (`i32`), `@p`, `@n` (`i64`), `%f`, `%g` (`f64`), the blocks
`<next>`, `<other>` and `<loop @i @acc>`, and the callees declared first.

";
