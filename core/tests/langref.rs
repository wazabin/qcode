//! The language reference is generated from [`qcode::langref::entries`], so
//! these tests are what keeps it honest: every entry has prose and a syntax,
//! and every example lowers (parses, resolves and types) inside a fixed
//! harness and, where the printer can read its own output back, prints in
//! exactly the form the reference shows.

use qcode::{
    langref::{InsnDoc, entries, example_context, example_program},
    lower::lower_str,
};

fn lower(program: &str) -> Result<String, String> {
    let mut ctx = example_context();
    lower_str(&mut ctx, program).map_err(|e| e.to_string())?;
    Ok(ctx.to_string())
}

/// `text` with runs of whitespace collapsed, so indentation does not matter.
fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn check_example(entry: &InsnDoc, example: &str) {
    let program = example_program(example);
    let printed = lower(&program).unwrap_or_else(|e| {
        panic!(
            "example of `{}` does not lower: {e}\n--- example ---\n{example}\n--- program ---\n{program}",
            entry.name
        )
    });
    // The printer cannot yet render everything the lowerer accepts (sequence
    // types, struct declarations). Where its output reads back, the example
    // must be in canonical form: what the reference shows is what `qcode`
    // prints.
    let reads_back = std::panic::catch_unwind(|| lower(&printed).is_ok()).unwrap_or(false);
    if !reads_back {
        return;
    }
    let printed_lines: Vec<String> = printed.lines().map(normalize).collect();
    for line in example.lines().map(normalize) {
        if printed_lines.contains(&line) {
            continue;
        }
        // A zero-width result (a sequence) prints without its binding: the
        // statement must still be there, as the bare expression.
        if let Some((_, rhs)) = line.split_once(" = ")
            && printed_lines.iter().any(|p| p == rhs)
        {
            continue;
        }
        panic!(
            "example of `{}` is not in canonical form\n--- line ---\n{line}\n--- printed ---\n{printed}",
            entry.name
        );
    }
}

#[test]
fn every_entry_is_documented() {
    for entry in entries() {
        assert!(!entry.doc.is_empty(), "`{}` has no doc comment", entry.name);
        assert!(!entry.syntax.is_empty(), "`{}` has no syntax", entry.name);
        assert!(
            !entry.category.is_empty(),
            "`{}` has no category",
            entry.name
        );
    }
}

#[test]
fn entry_names_are_unique_within_category() {
    let mut seen = std::collections::HashSet::new();
    for entry in entries() {
        assert!(
            seen.insert((entry.category, entry.name)),
            "duplicate entry `{}` in `{}`",
            entry.name,
            entry.category
        );
    }
}

#[test]
fn every_example_lowers_and_is_canonical() {
    for entry in entries() {
        for example in entry.examples {
            check_example(entry, example);
        }
    }
}

#[test]
fn sequence_intrinsics_without_text_examples_are_the_known_ones() {
    // Their operand must already be a fixed array, which only a pass produces.
    let missing: Vec<_> = entries()
        .filter(|e| e.examples.is_empty())
        .map(|e| e.name)
        .collect();
    assert_eq!(missing, ["PCodeOp", "Enumerate", "TakeWhile"]);
}
