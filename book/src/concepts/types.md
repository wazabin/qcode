# Values and types

Every value in QCode has a type, and a type is mostly a **width**.

## Kinds of value

| Written | What it is |
| --- | --- |
| `%name` | An SSA value: the result of one statement, defined once. |
| `@name` | A block parameter: a value received on entry to its block. |
| `0x10`, `true` | A literal. |
| `RAX`, `&RAX` | A varnode — a *location*, not a value — and its address. |

An SSA value or parameter is immutable. Mutable state lives in memory spaces
and is reached through `load` and `store`; see [memory spaces](spaces.md).

## Types

| Type | Meaning |
| --- | --- |
| `iN` | An `N`-bit integer: `i8`, `i16`, `i32`, `i64`, `i128`, and wider (the x87 stack uses `i80`). No signedness — the *operator* chooses it. |
| `fN` | Accepted as a spelling of `iN` on input. There is no separate float type: `f+`, `sqrt`, `int2float` say which values are floats. |
| `bool` | A one-byte truth value, `{0, 1}`. Produced only by comparisons and the `true`/`false` literals; `and`/`or`/`xor` over `bool` are logical. |
| `T*` | A pointer to a declared struct `type T { … }`, the base of `gep`. |
| tuples | The result of `pack(a=…, b=…)`, read back with `extract`. |
| sequences | Arrays `[T; N]` and lists `[T; *]` produced by the array intrinsics; see [sequences](sequences.md). |

Because a type is a width, QCode never needs a cast between "kinds" of the
same size: the same `i64` feeds `+`, `f+`, `s<` and a `load` address. The
casts that exist change the width (`zext`, `sext`, `v[a:b]`, `float2float`) or
the representation (`int2float`, `trunc`).

## What the width means

A value of `iN` is `N` bits; arithmetic is modulo `2^N` and the result of an
operator has the width of its operands. Both operands of a binary operator
must have the same width (a literal takes the width of the other operand); a
shift count is no exception, and a count at or past the width shifts every
bit out.

```qcode
fn widths:
<entry @x:i32 @p:i64>
    i32 %wrap = i32 @x * i32 0x10000;
    i64 %wide = zext(i64, i32 @x);
    i64 %prod = i64 %wide * i64 0x10000;
    i32 %sh = i32 @x << i32 0x3;
    i8 %lo = i32 @x[0:1];
    return at i64 @p;
```

`%wrap` keeps 32 bits of the product; `%prod` keeps 64. A sub-range `v[a:b]`
is bytes `a` to `b` (little-endian, `[0:1]` the least significant byte).

## Signedness is in the operator

There is one `+` and one `*`, since two's-complement addition and the low half
of a multiplication do not depend on sign. Everything that does has two forms:

| Unsigned | Signed |
| --- | --- |
| `<`, `<=` | `s<`, `s<=` |
| `/`, `%` | `s/`, `s%` |
| `>>` | `s>>` |
| `zext` | `sext` |
| `carry` | `scarry`, `sborrow` |

## Declared types on input

The printer writes every operand's type; the parser accepts a statement
without them wherever the type follows from the value (`%a + %b` for two
known values). A declared result type is documentation, not a cast: the
result has the type its operator gives it, whatever the prefix says. One
declaration does change a type — `name* %p = …` gives a pointer a struct
type, so that `gep(%p.field)` can name its fields.
