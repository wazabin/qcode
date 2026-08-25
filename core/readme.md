# qcode

`qcode` is a typed, SSA-style p-code intermediate representation for binary
analysis. It models lifted machine-code semantics using values, instructions,
basic blocks, functions, and explicitly addressed memory spaces.

The crate contains the IR and construction/lowering APIs. Optimization,
reconstruction, and other analysis passes are intentionally kept in the
separate `qcode_analysis` crate, so consumers that only need the language do
not take those dependencies.

## Quick start

```rust
use qcode::context::Context;

let mut context = Context::new();
let builder = context.builder_at(0x1000);
// Emit instructions with `builder`, then terminate the block.
# drop(builder);
```

## QCode source

Use `qcode::lower::lower_str` to parse QCode text at runtime. The `qcode!`
macro is re-exported by this crate for embedding source literals:

```rust
use qcode::{context::Context, qcode};

let mut context = Context::new();
qcode!(&mut context, "%tmp = 1 + 2");
```

The macro lowers source into the supplied `Context` and binds identifiers
declared by the source in the surrounding Rust scope.

## Main types

- `Context`: owner of module-wide IR state, interners, spaces, and functions.
- `Builder`: fluent API for emitting IR into a basic block.
- `ValueId`: compact identifier for literals, instructions, varnodes, blocks,
  and functions.
- `Space`: a named, uniformly addressed memory region.
- `FunctionBody`: function-local blocks, instructions, and parameters.

See the API documentation for the QCode text format and the builder methods.
