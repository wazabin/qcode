# The QCode book

The site at [wazabin.github.io/qcode](https://wazabin.github.io/qcode/): the
language reference, the concept pages, the API documentation and the x86-64
playground. Built and deployed by `.github/workflows/pages.yml` on every push
to `main`.

```bash
book/build.sh          # everything into book/book
mdbook serve book      # the book alone, live-reloading (needs src/langref.md)
```

- `src/langref.md` is **generated** by `cargo run -p qcode-langref` from the
  tables in `qcode::langref` and is not committed. Each entry is the rustdoc
  of an instruction, operator or intrinsic plus its `#[langref(syntax,
  example)]` attributes; `core/tests/langref.rs` checks every example.
- `src/concepts/*.md` are hand-written; `book/gen/tests/book_examples.rs`
  lowers every ```qcode fence in them (mark a fragment ```qcode,ignore).
- `src/pkg/` is the wasm lifter built from `web/` by `wasm-pack`.
- `api/` is `cargo doc --workspace`.
