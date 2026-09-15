# Functions, calls and lambdas

QCode has two kinds of function, and three kinds of transfer between them.

## Machine functions

`fn name:` is a function of the guest program: a body of blocks at machine
addresses, entered by a `call` and left by a `return at ptr`, where `ptr` is
the return address the machine convention put on the stack or in a link
register. A freshly lifted function has no parameters — its inputs are
registers and memory — and a freshly lifted `call fn f()` passes nothing.

```qcode
fn callee:
<entry @a:i32 @b:i32 @ra:i64>
    %sum = i32 @a + i32 @b;
    return i32 %sum at i64 @ra;

fn caller:
<entry @x:i32 @y:i32 @p:i64>
    call fn callee(@a=i32 @x, @b=i32 @y);
<after>
    return at i64 @p;
```

Once analysis has inferred an interface, the call names the callee's entry
parameters (`@a=…`) exactly as a branch names a block's, and `return v at
ptr` carries the value out. A `call` is a terminator: control comes back to
the block that follows it (or to the blocks a `// -> <bb>` hint lists).

The other machine transfers:

- `call [ptr](…)` — through a computed address; the callee is unknown until
  analysis resolves it.
- `tailcall fn f(…)` — control leaves for `f` and never returns here; `f`'s
  return is this function's return. This is how a thunk or a `jmp` to another
  function is encoded, since a block of one function may never `goto` a block
  of another.
- `goto [ptr]` and `switch` — indirect and multi-way branches *within* the
  function.

## Lambdas

`lambda name:` is a pure function of values: no registers, no memory, no
return address. It is entered with `apply` and returns with `return v`. Its
result is an ordinary SSA value at the call site, and evaluating it has no
effect, so the optimizer treats `apply` like any other expression.

```qcode
lambda square:
<entry @v:i64>
    %r = i64 @v * i64 @v;
    return i64 %r;

fn main:
<entry @x:i64 @p:i64>
    i64 %sq = apply square(i64 @x);
    return i64 %sq at i64 @p;
```

Lambdas exist to give a name to a computation that a loop repeats: the body
of a [`map` or `scanl`](sequences.md) is a lambda, applied once per element.
They are also what a pure machine function becomes once analysis has proved it
touches nothing but its arguments.

## Where the boundary is

The IR never stores a block of another function anywhere: a branch target is
always local, and every cross-function edge is a `call`, `tailcall` or
`apply` naming a *function*. That keeps each function body self-contained —
it can be lifted, optimized, serialized and replaced on its own — and keeps
the [block-argument](block-arguments.md) rule simple: the values that enter a
block come from its own function's edges, or from its function's caller.
