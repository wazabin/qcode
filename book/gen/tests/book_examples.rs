//! Every ```qcode fence in the book lowers. A fence that is deliberately a
//! fragment is marked ```qcode,ignore and skipped.

use std::{fs, path::Path};

use qcode::{langref::example_context, lower::lower_str};

fn fences(markdown: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut current: Option<(usize, String)> = None;
    for (i, line) in markdown.lines().enumerate() {
        match &mut current {
            Some((start, body)) => {
                if line.trim_start().starts_with("```") {
                    out.push((*start, body.clone()));
                    current = None;
                } else {
                    body.push_str(line);
                    body.push('\n');
                }
            }
            None => {
                if line.trim() == "```qcode" {
                    current = Some((i + 1, String::new()));
                }
            }
        }
    }
    out
}

fn markdown_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(dir).expect("book/src is readable") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            markdown_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "md") {
            out.push(path);
        }
    }
}

#[test]
fn every_qcode_fence_lowers() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../src");
    let mut files = Vec::new();
    markdown_files(&src, &mut files);
    files.sort();
    let mut checked = 0;
    for file in files {
        // The generated reference is checked by `core/tests/langref.rs`.
        if file.file_name().is_some_and(|n| n == "langref.md") {
            continue;
        }
        let markdown = fs::read_to_string(&file).expect("markdown is readable");
        for (line, body) in fences(&markdown) {
            let mut ctx = example_context();
            if let Err(e) = lower_str(&mut ctx, &body) {
                panic!(
                    "{}:{line}: qcode fence does not lower: {e}\n{body}",
                    file.display()
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "no qcode fences found under {}", src.display());
}
