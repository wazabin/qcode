# wazabin-qcode-macro

Procedural macro support for embedding QCode source literals in Rust. This is
normally used through `qcode::qcode!`, which re-exports the macro.

```rust
use qcode::{context::Context, qcode};

let mut context = Context::new();
qcode!(&mut context, "%tmp = 1 + 2");
```

The macro parses and validates the source at compile time, then lowers it into
the supplied `qcode::Context` at runtime. Identifiers declared by the source
are bound in the surrounding Rust scope. Use `{name}` in QCode source to
capture an in-scope Rust `ValueId` named `name`.

For parsing dynamically supplied source, use `qcode::lower::lower_str`
instead.

## License

Licensed under the [MIT License](LICENSE).
