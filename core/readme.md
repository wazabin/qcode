# qcode

`qcode` is a typed, SSA-style p-code intermediate representation for binary
analysis. It models lifted machine-code semantics using values, instructions,
basic blocks, functions, and explicitly addressed memory spaces.

The crate contains the IR and construction/lowering APIs, and nothing else:
optimization and execution live in separate crates, so consumers that only
need the language do not take those dependencies. See
[`qcode_passes`](https://docs.rs/wazabin-qcode-passes) for block-local cleanup,
[`qcode_emulator`](https://docs.rs/wazabin-qcode-emulator) to interpret it, and
[`qcode_vm`](https://docs.rs/wazabin-qcode-vm) to run a guest program under an MMU.

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

## License

Licensed under the [MIT License](LICENSE).
