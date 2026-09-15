# Sequences: map, scan and the array intrinsics

A loop that walks a buffer is, in the lifted IR, a cycle of blocks with a
counter, a pointer and a store per iteration. Analysis can replace such a
loop by a single value: the whole buffer after the loop, as an **array
value** computed by `map` or `scanl` over a **sequence**. This page is about
those values; the instructions themselves are in the
[reference](../langref.md#sequences).

## Sequences

A sequence is an array `[T; N]` — `N` elements of `T`, laid out
contiguously, its width `N * size(T)` — or a list `[T; *]`, a sequence whose
length is not known statically. Sequences are built and taken apart by the
intrinsics:

| Intrinsic | Type | Meaning |
| --- | --- | --- |
| `$iota(n)` | `i64 -> [i64; *]` | `[0, 1, …, n-1]`, the driver of a counted loop. |
| `$splat(x, n)` | `T, i64 -> [T; n]` | `n` copies of `x`; a zero-filled buffer is `$splat(0, n)`. |
| `$singleton(v)` | `T -> [T; 1]` | `[v]`. |
| `$concat(a, b)` | `Seq<T>, Seq<T> -> Seq<T>` | `a` followed by `b`. |
| `$at(arr, i)` | `[T; N], i64 -> T` | Element `i`, `i` a value. |
| `$insert(arr, i, v)` | `[T; N], i64, T -> [T; N]` | A copy of `arr` with element `i` set to `v`. |
| `$len(seq)` | `Seq<T> -> i64` | The number of elements. |
| `$enumerate(arr)` | `[T; N] -> [(index: i64, elem: T); N]` | Each element paired with its index. |
| `$take_while(arr)` | `[T; N] -> [T; *]` | The prefix before the first zero element: a C string. |

All of them are pure. `$insert` in particular does not modify `arr`; it is the
SSA form of a store into a promoted buffer, and `$at($insert(a, i, v), i)`
simplifies to `v`.

## `map`

`body <$> src` applies the lambda `body` to every element of `src` and is the
array of results:

```qcode
lambda triple:
<entry @x:i64>
    %r = i64 @x * i64 0x3;
    return i64 %r;

fn main:
<entry @p:i64>
    %src = $iota(i64 0x4);
    %m = triple <$> %src;
    return at i64 @p;
```

`%m` is `[0, 3, 6, 9]`. The body sees one element at a time and nothing else,
so `map` is exactly the loop whose iterations are independent. A body that
needs the index maps over `$enumerate(src)`; a body that needs a
loop-invariant value takes it as a capture, `(body %k) <$> src`.

Reading one element of a map does not require computing the whole array:
`$at(body <$> src, i)` is `apply body($at(src, i))`. The passes use this
*projection* to answer a question about one lane without materializing the
buffer.

## `scanl`

When each iteration depends on the previous one, `map` cannot express it.
`scanl @body init src` threads an accumulator through the sequence:

```text
acc_0   = init
acc_i+1 = body(acc_i, src[i])
out[i]  = acc_i+1
```

```qcode
lambda step:
<entry @acc:i64 @x:i64>
    %r = i64 @acc + i64 @x;
    return i64 %r;

fn main:
<entry @p:i64>
    %src = $iota(i64 0x3);
    %s = scanl @step i64 0xa %src;
    return at i64 @p;
```

`%s` is `[10, 11, 13]`: the running sum from 10 over `[0, 1, 2]`. The body is
binary — accumulator first, element second — and its return type is the
accumulator's type. The canonical case is the MT19937 seeding loop
`mt[i] = f(mt[i-1], i)`, a `scanl` over `$iota(624)` with the first element
prepended by `$concat($singleton(seed), …)`.

## What this buys

A loop as blocks is opaque: to know what the buffer holds afterwards you run
it. A loop as `map`/`scanl` is a value: it has a type, it can be compared for
equality with another, it can be projected, folded when its inputs are
constant, and shown in one line. That is the point of the sequence forms —
they are the representation in which "this function fills a table" is a
fact the IR states rather than one an analysis has to rediscover.

The emulator executes them directly (it materializes the array), so a program
that has been rewritten this way still runs, and still runs the same.
