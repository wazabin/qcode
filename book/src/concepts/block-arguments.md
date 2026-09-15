# Blocks and block arguments

QCode is in SSA form: every value is defined exactly once, by the statement
that binds it. The usual price of SSA is the φ-node — a pseudo-instruction at
the top of a block that picks a value depending on which predecessor ran.
QCode has no φ. Instead, as in MLIR and Cranelift, **a block declares
parameters and every branch to it passes arguments**.

## A loop

The factorial of `@n`, as a loop carried in two parameters:

```qcode
fn fact:
<entry @n:i64 @ra:i64>
    goto <head @i=i64 @n @acc=i64 0x1>;
<head @i:i64 @acc:i64>
    bool %done = i64 @i == i64 0x0;
    if bool %done goto <exit @r=i64 @acc> else goto <body @i=i64 @i @acc=i64 @acc>;
<body @i:i64 @acc:i64>
    %next = i64 @acc * i64 @i;
    %dec = i64 @i - i64 0x1;
    goto <head @i=i64 %dec @acc=i64 %next>;
<exit @r:i64>
    return i64 @r at i64 @ra;
```

`<head @i:i64 @acc:i64>` receives its two values from two places: the entry
(`@i=@n`, `@acc=1`) and the back edge (`@i=%dec`, `@acc=%next`). Where LLVM
would write `%acc = phi [1, %entry], [%next, %body]`, QCode puts the choice on
the edge that makes it. The body reads `@i` and `@acc` like any other value.

## Rules

- A branch must supply **every** parameter of its target, by name, with a
  value of the parameter's type. `goto <exit>` to a block with parameters is
  an error.
- Parameters are values only inside their block. Since `@acc` of `<head>` and
  `@acc` of `<body>` are different parameters, the body's branch back to
  `<head>` passes `@acc=i64 %next`, not `@acc`.
- Parameter names are function-scoped: two blocks may not both declare `@i`
  with different meanings — the harness above declares `@i` twice on purpose,
  because the two are the same loop variable, and the printer renames on
  collision otherwise.
- Every terminator that names a block carries arguments: `goto`, both arms
  of `if`, every arm of `switch`, including `default`.

## Why not φ

Two reasons, both practical for a lifter. (For the general case against
φ-nodes, see Filip Pizlo's
[SSA without phi](https://gist.github.com/pizlonator/cf1e72b8600b1437dda8153ea3fdb963);
the trade-offs there are the ones this design makes.)

The first is **locality**. A φ refers to predecessor blocks by name, so adding,
removing or splitting an edge means editing every φ in the successor. With
block arguments the edge owns its values: splitting an edge is inserting a
block whose `goto` forwards the arguments it received, and deleting an edge
deletes its arguments with it. Passes that rewrite control flow — the CFG
straight-line merging in `qcode_passes`, jump-table resolution, function
discovery — never touch anything but the terminator they are rewriting.

The second is **uniformity with calls**. A function's entry block has
parameters like any other block, and a call binds them by name the same way
a branch does:

```qcode,ignore
call fn callee(@a=i32 @x, @b=i32 @y);
```

So argument passing, loop-carried values and the entry of a lambda are one
mechanism, and an analysis that understands `goto <bb @p=v>` already
understands most of `call`.

## What the lifter produces

Freshly lifted machine code has no block arguments: registers flow through
the register space (`load(register:8, i64 RAX)` / `store(…)`), not through
parameters. Block arguments appear when analysis promotes a location to a
value — a loop counter kept in a register becomes a `<head @i>` parameter,
and the stores and loads disappear. The playground shows the former; the
reference examples are mostly written in the latter style, because it is the
form the IR is meant to be read in.
