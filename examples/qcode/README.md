# qcode examples

Small, canonical textual-qcode programs for experimenting with passes. They use
the same syntax the [`qcode!`](../../crates/qcode/macro) macro accepts and the
`qcode-pass` filter parses (`#` begins a line comment).

Run any of them through a pass (qcode in on stdin, transformed IR out on stdout):

```sh
cargo run -p qcode_analysis --example qcode-pass -- -p <PASS> < examples/qcode/fibonacci.qcode
cargo run -p qcode_analysis --example qcode-pass -- -p loop_to_recursion -p dce -i examples/qcode/fibonacci.qcode
cargo run -p qcode_analysis --example qcode-pass -- --list      # every registered pass
```

With no `-p`, the tool is a canonicalizing formatter (parse → pretty-print).

## The programs

All four are the same counted-loop shape (`entry → head → {body↺, exit}`), which
is what `loop_to_recursion` and `accumulator_elim` recognize.

| File | Recurrence | `loop_to_recursion` | `accumulator_elim` |
|---|---|---|---|
| `fibonacci.qcode` | `(a,b) -> (b, a+b)` — linear, accumulator-only | ✅ tail recursion | ✅ **3 params → 1**, returns `(a,b)` |
| `factorial.qcode` | `acc -> acc*n` — reads driver `n` | ✅ tail recursion | ❌ declines (acc demoted to driver) |
| `sum_of_range.qcode` | `s -> s+i` — reads driver `i`; `bound` invariant | ✅ tail recursion | ❌ declines |
| `mt19937_recur.qcode` | MT seeding scalar core — reads driver `i` | ✅ tail recursion | ❌ declines |

`accumulator_elim` fires only when an accumulator's update reads *no* driver
(Fibonacci). When the update reads the loop index (`+ i`, `* n`), the classifier
soundly demotes that slot to a driver, so there is nothing to eliminate.

The real MT19937 init additionally *writes an array* (`mt[i] = body(mt[i-1], i)`),
which is impure and element-depends-on-previous: that is a **`scan`**, not a
recursion — see `mt19937_recur.qcode`'s header comment.
