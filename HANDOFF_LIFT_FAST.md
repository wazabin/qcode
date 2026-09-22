# Handoff: making `LiftSession` replay fast

Written 2026-09-22. Everything below is verifiable from this tree and the
two measurement tools in §2; no conversation context is needed. The goal
Jack set is **10 MB/s through `LiftSession` on `/usr/bin/bash`, with no
functionality or guarantee removed and no explosion in memory or code
size**. It is not reached: the branch is at ~8 300 retired instructions
per lifted instruction, and 10 MB/s is ~2 500. §5 says where the rest is
and what has already been tried and rejected, with numbers.

## 1. State of the tree

Branches, each stacked on the one before, all pushed:

| branch | PR | what |
|---|---|---|
| `register-params` | [#38](https://github.com/wazabin/qcode/pull/38) | the lift cache keyed on register fields |
| `session-replay` | [#39](https://github.com/wazabin/qcode/pull/39) | name-table tiers, `FunctionBody::append_insn` |
| `lift-fast` | [#40](https://github.com/wazabin/qcode/pull/40) | everything in §3 |

Work on `lift-fast`. `HANDOFF_LIFT_CACHE.md` §3b and §3c are the same
story from the cache's side and hold the per-step numbers; this file is
what a new agent needs to carry on.

Checks that pass on `lift-fast` (CI's commands):

```sh
cargo test --workspace --all-features            # 820 passed
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --check && cargo doc --workspace --no-deps
```

## 2. How to measure, and how not to

The laptop runs other agents; load average 5–20 is normal, so **wall
clock is noise** — `--warm` passes on `/usr/bin/bash` have come out
anywhere between 0.86 and 2.0 MB/s for the same binary within a minute.
Track **retired instructions per lifted instruction** instead, and use
callgrind for exact per-operation costs.

```sh
B=target/release/examples/lift-throughput
cargo build --release -p wazabin-qcode-sleigh --example lift-throughput

# 1. Retired instructions, net of the example's own setup. `measure.sh`:
#    perf stat -e instructions:u of `--stage none`, of the cold lift pass
#    and of the warm one, differenced and divided by 252 413.
$B /usr/bin/bash --linear --stage lift --session --cache [--warm]

# 2. Exact per-operation cost: one encoding repeated, every lift after the
#    first a hit. A `nop` is one operation, `4885c0` (test rax,rax) is 17,
#    so the pair separates the fixed cost from the per-operation one.
RUSTFLAGS="-C force-frame-pointers=yes -C debuginfo=1" \
  cargo build --release -p wazabin-qcode-sleigh --example lift-throughput \
  --target-dir target-fp
valgrind --tool=callgrind --callgrind-out-file=cg.out \
  --toggle-collect='*LiftSession*lift*' \
  ./target-fp/release/examples/lift-throughput --bytes 4885c0 --iters 20000 \
  --linear --stage lift --cache --session
callgrind_annotate --inclusive=no --threshold=60 cg.out   # divide by 20000*17
```

`--warm` now works with `--session` (the warm pass runs in a throwaway
session, since a session refuses to lift an address twice), and the
example prints the body's block, instruction and name counts.

Two traps. `rtk`'s summary of `cargo test` reports "820 passed" even when
a test binary is **killed** under memory pressure — read the raw log
(`rtk proxy cargo test …`) when a run looks odd; a cache-test kill under
load average 14 looked exactly like a failure. And `perf`'s cycle counts
go haywire (even negative) on a loaded machine; the instruction counts
stay sound.

## 3. What is already done

Per lifted instruction of `/usr/bin/bash` (252 413 instructions, 8.4
operations and 3.8 named operations each), and peak memory of the whole
session:

| | instrs/insn | peak |
|---|---|---|
| before `session-replay` | 28 000 | 695 MB |
| after `session-replay` | 14 400 | 558 MB |
| `lift-fast` | **8 300** | **493 MB** |

- **Layout.** The module-interned ids (`LiteralId`, `VarnodeId`,
  `BytesId`, `PoisonId`) and the body-local temporary ids are `u32`; an
  instruction's links and machine address are packed (`core/src/value/link.rs`);
  terminator, call, intrinsic and tuple argument lists are `Box<[_]>`.
  `Instruction` 160 → 96 bytes, `Mnemonic` 80 → 56, `LocalValueId` 16 → 8.
- **Names** (`core/src/value/name.rs`). A function-local `Name` is eight
  bytes — a `BaseId` into the body's interned bases plus a suffix —
  rendered on demand; a block at an address with no name of its own holds
  none and renders as the address; each base knows the address it spells.
  `Named::name` returns `Cow<str>`. A body carries its bases on the wire
  and rebuilds the table on load, as it already did for use edges.
- **The replay** (`core/src/value/function/prototype.rs`,
  `sleigh/src/cache.rs`). `FunctionBody::append_prototype` takes a
  recorded run — the template's operations with this session's type ids
  and name bases, built once per template and session (`Resolved`) — and
  does per operation only what the body must. Block lists are read and
  written once per run of one block.
- **Bookkeeping.** Use-list heads for literals and varnodes are dense
  vectors (`SharedHeads`), a block's incident edges are inline
  (`EdgeSet`), `add_use` matches once, and a commit skips its
  minted-callee scan when the instruction promised none.
- **A bug fixed on the way**: a replayed branch made no CFG edge, so the
  blocks of a session's hits had no successors. `sleigh/tests/cache.rs`
  `a_session_hit_builds_the_same_edges` compares successor, predecessor
  and use counts of a session of hits against one of misses; it fails on
  the previous replay. Exactness of the text, names included, is
  `a_session_hit_builds_the_same_function_names_included`. **Run both
  after every change**: they are what keeps "no guarantee removed" true.

## 4. Where the 8 300 go

From the pair of synthetic runs in §2: a `nop` costs **3 687** and a
17-operation instruction **12 365**, so **583 per operation and ~3 300
fixed**. Exact self costs, per operation:

| instrs/op | |
|---|---|
| 241 | `append_prototype` itself: the mnemonic clone, the operand walks, the 96-byte arena push |
| 94 | `add_use`, two edges |
| 64 | `StableArena::push` — three vectors (payload, id table, locations) |
| 52 | `RecyclingArena::push` for the use edges |
| 54 | naming (`register_unique` and its probes), for the 45 % that are named |
| 33 | the replay's scratch extends |

And per instruction: the fall-through placeholder block with its label
and index entry ~750, the undecoded key walk ~560, `begin`/`commit`
~300, the replay's prologue ~360, the allocator ~200.

## 5. What to try next, and what not to

**Rejected, measured, do not redo.** A `reserve` on jstd's
`StableArena` and `RecyclingArena`, called once per run, was implemented
(local branch `reserve-arenas` in `~/dev/nochurn/wazabin-jstd`, not
pushed) and measured against the same build with the calls removed:

| | nop | 17 ops |
|---|---|---|
| without reserve | 3 687 | 12 365 |
| with reserve | 3 723 | 12 413 |

It is a **loss**: the arenas grow monotonically here, so doubling is
already amortized to nothing, and three capacity checks per run cost
more than they save. `StableArena::push`'s 64 instructions are the three
pushes themselves, not reallocation. No jstd change is warranted for
this workload.

Also already rejected: merging the prototype's two operand walks into
one (neutral), and a one-entry memo for `Resolved` (adjacent
instructions on linear code rarely share a template).

**Worth trying, in order.**

1. **The use-edge arena** (~50/op). A `Use` already carries
   `next: Option<UseId>`, so `RecyclingArena`'s `Slot` enum — a tag plus
   padding around a 24-byte payload — is redundant: a free list threaded
   through the payload's own link would drop the tag, the enum match and
   8 bytes per edge. That is a jstd design change (a trait for the
   payload's link, or a new type beside `RecyclingArena`), so it needs a
   jstd PR and release; unlike `reserve`, this one removes work rather
   than moving it.
2. **The identifier tables** (~40/op). `StableArena` keeps `slot_ids`
   and `locations` so an id survives a removal. A lift never removes, so
   for the append path those two pushes are pure overhead. A jstd verb
   that appends densely while the arena has had no removal — or a second
   arena type for append-only bookkeeping — would take `push` from three
   vectors to one.
3. **`Mnemonic` down to ~40 bytes** by boxing `CBranch`, `Switch`,
   `Call` and `Scan` (the variants that hold two lists). `Store` at 40
   bytes then sets the floor, and `Instruction` reaches ~80: a tenth off
   every per-operation copy, and off the body's memory.
4. **A leaner `Construction`** for a run that promises nothing: `begin`
   and `commit` are ~300 per instruction of journal bookkeeping for a
   replay that cannot fail in any of the ways the journal guards against.
   Keep the guards for real lowerings.
5. **The fall-through placeholder block** is ~750 per instruction and
   the largest single item: a block made, addressed, labelled and
   indexed for the *next* instruction, which then finds it as its entry.
   Nothing above changes the IR; this one would. Do not touch it without
   asking Jack — the flat lowering is defined to produce that shape, and
   `AddressIndex` is a caller-visible contract. If it is ever on the
   table, the cheap half is having the next instruction's `begin` reuse
   what the previous one just made rather than resolving the address
   again.

Steps 1–4 together look like 8 300 → ~6 500, i.e. ~3 MB/s, not 10. Say
so plainly rather than reporting a number the machine's load produced;
if 10 MB/s is still the bar after 1–4, the remaining factor is in the
IR's shape (8.4 operations and a block per machine instruction), and
that is a design decision, not an optimisation.
