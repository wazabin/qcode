# qcode_macro

Procedural macros for embedding QCode at compile time.

## Overview

A `proc-macro` crate that uses `qcode_parser` to parse QCode literals in Rust
source and generate the corresponding IR at compile time.
Enables ergonomic inline QCode in tests and host code.

## Input format

```rust
qcode!(builder_expr, "program")
```

- `builder_expr` is any mutable `Builder` expression (defaults to `builder`).
- `program` is one or more statements separated by `;`.

## Grammar (intentionally minimal)

- `statement := ident = expr | expr`
- `expr := atom | unop atom | atom op atom | float_fn(atom) | misc_fn(args) | cast | memory`
- `atom := {ident} | ident | integer`
- `cast := zext(type, atom) | sext(type, atom) | int2float(type, atom) | float2float(type, atom) | trunc(type, atom)`
- `unop := ! | ~ | - | f-`
- `float_fn := abs | sqrt | floor | ceil | round`
- `misc_fn := nan(src) | popcount(src) | lzcount(src) | carry(lhs, rhs) | scarry(lhs, rhs) | sborrow(lhs, rhs)`
- `op` includes integer/boolean operators (including signed forms `s< s<= s> s>= s>> s/ s%`) plus float operators: `f+ f- f* f/ f== f!= f< f<= f> f>=`
- `memory := load(type, atom) | store(atom, atom)`
- `type := iNN | fNN` (must be byte-aligned)

Only **one operator per expression** is supported.

## Capture behavior

- `{name}` captures an in-scope Rust variable named `name` and treats it as a `ValueId`.
- bare `name` refers to a local qcode variable created by an earlier assignment.

## Return values

- Most expressions return a `ValueId`.
- `load(type, atom)` and `store(atom, atom)` return an `InstructionId`.

## Examples

```rust
fn gen_add_2<'str, 'ctx>(builder: &mut Builder<'str, 'ctx>, v1: ValueId, v2: ValueId) -> ValueId {
    qcode!(builder, "{v1} + {v2}")
}
```

```rust
fn gen_add_const<'str, 'ctx>(builder: &mut Builder<'str, 'ctx>, v1: ValueId) -> ValueId {
    qcode!(builder, "{v1} + 2")
}
```

```rust
fn gen_add_5<'str, 'ctx>(builder: &mut Builder<'str, 'ctx>, v1: ValueId) -> ValueId {
    qcode!(builder, "tmp = {v1} + 3; tmp + 2")
}
```

```rust
fn gen_load<'str, 'ctx>(builder: &mut Builder<'str, 'ctx>, ptr: ValueId) -> InstructionId {
    qcode!(builder, "load(i32, {ptr})")
}
```

```rust
fn gen_store<'str, 'ctx>(builder: &mut Builder<'str, 'ctx>, ptr: ValueId, src: ValueId) -> InstructionId {
    qcode!(builder, "store({ptr}, {src})")
}
```