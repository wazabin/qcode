//! The Embench images, built by `benchmarks/embench/build.sh`.

#![allow(dead_code)]

/// Every `.elf` in the corpus directory, by name, sorted.
///
/// `EMBENCH_DIR` selects an alternative corpus — a build at a larger scale
/// factor, say, where one-time translation is amortised over enough
/// execution to show a steady-state rate rather than a warm-up one.
pub fn images() -> Vec<(String, Vec<u8>)> {
    let corpus = std::env::var("EMBENCH_DIR").unwrap_or_else(|_| "target/embench".to_owned());
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join(corpus);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<_> = entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "elf"))
        .map(|e| {
            (
                e.path().file_stem().unwrap().to_string_lossy().into_owned(),
                std::fs::read(e.path()).expect("image"),
            )
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}
