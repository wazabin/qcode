# qcode

Core QCode intermediate representation.

## Overview

Defines the IR data structures:
    - values
    - instructions
    - blocks
    - functions
    - value registry
All other `qcode_*` crates depend on this crate for the shared IR types.


## Architecture

This module defines the project’s internal p-code IR and emission machinery.

### Core model

`context::Context` is the central owner of IR state:

- address spaces (`Space`)
- register/bitrange mappings
- named hints for readable output
- value registries (literals, varnodes, instructions, basic blocks, functions)
- block and function lookup by address or name

### Value system

Defined under `value/*` and unified via `ValueId` / `ValueRef`:

- `Literal`: constant immediate values.
- `Varnode`: storage locations (`space`, `address`, `size`).
- `Instruction`: SSA-like computed values with a `Mnemonic`.
- `BasicBlock`: ordered instruction list + optional label.
- `Function`: named grouping of basic blocks with a single entry block and optional binary address.

### Instruction layer

`instruction.rs` defines:

- memory/control/data mnemonics
- integer/boolean/float op families
- custom/user p-code op representation (`PCodeOp`)
- rendering helpers for textual pretty-print output

### Builder API

`builder::Builder` provides ergonomic emission APIs:

- append instructions to current block
- create/switch blocks
- create temporaries
- load/store helpers
- arithmetic/logic/comparison/branch/call/return helpers
- local namespace support for macro expansion

This is the main abstraction used by the SLEIGH walker when lowering constructor semantics.

### QCode integration example

The workspace includes the `qcode` proc macro crate for tiny string-based builder emission.

```rust
use pcode::pcode::{builder::Builder, context::Context, value::ValueId};
use qcode::qcode;

fn add_five<'a>(ctx: &mut Context<'a>, addr: usize, v1: ValueId) -> ValueId {
	let mut builder = Builder::from_context(ctx, addr);

	// One-op expressions only: atom or atom op atom.
	// `{v1}` captures the in-scope Rust ValueId.
	qcode!(&mut builder, "tmp = {v1} + 3; tmp + 2")
}
```

Supported atoms are:

- `{name}`: capture an in-scope Rust variable (as a `ValueId`)
- `name`: local qcode variable created by a prior assignment
- integer literal

### Current state

- The IR is suitable for readable textual output and downstream analysis.
- Some areas are intentionally incomplete/iterative (`analysis`, parts of `module`, TODO paths like `push_addr`).
- The architecture is designed so additional analysis/transforms can be added over `Context` + block/instruction registries.
