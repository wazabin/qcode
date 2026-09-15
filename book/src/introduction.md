# Introduction

QCode is the intermediate representation of the [wazabin](https://github.com/wazabin)
toolchain: a typed, SSA-style p-code that machine code is lifted into and that
the emulator, the JIT and the analyses all operate on. It is modelled on
Ghidra's p-code — the same SLEIGH processor specifications produce it — with
first-class blocks, block arguments, functions, and a text format.

This site is in three parts.

- The **[language reference](langref.md)** lists every instruction, operator
  and intrinsic with its syntax, semantics and an example. It is generated
  from the `qcode` crate itself: each entry is the documentation written next
  to the instruction's definition, and the examples are checked by the crate's
  tests, so the page cannot fall behind the code.
- The **concept pages** explain what a single entry cannot: how control flow
  and block arguments replace φ-nodes, what a type is, how registers and
  memory are addressed, how calls and lambdas relate, and how loops over
  arrays become `map` and `scanl`.
- The **[playground](playground.md)** answers "what does this instruction
  look like in QCode?": type x86-64 bytes, read the lifted IR.

The [API documentation](api.md) covers the Rust crates.

## A first look

```qcode
fn max:
<entry @a:i64 @b:i64 @ra:i64>
    bool %lt = i64 @a s< i64 @b;
    if bool %lt goto <exit @r=i64 @b> else goto <exit @r=i64 @a>;
<exit @r:i64>
    return i64 @r at i64 @ra;
```

A function is a list of labelled basic blocks. Each statement binds one SSA
value (`%lt`) from an operator over typed operands; a block ends with a
terminator that names its successors and passes them arguments; a block that
receives values declares them as parameters (`@r`). Registers and memory are
not values but *locations* in named spaces, reached through `load` and
`store`; everything else is a pure expression.

## Where it fits

```text
bytes ──SLEIGH──▶ p-code ──lift──▶ QCode ──▶ emulator / JIT / analyses
```

The lifter (`wazabin-qcode-sleigh`) turns flat p-code into QCode one
instruction at a time, on demand, as the virtual machine reaches it. The
passes (`wazabin-qcode-passes`) clean the result up; the analyses in the
private toolchain go further, recovering calls, arguments and loops. The
reference documents what all of them agree on.
