# Handoff: the shape-keyed lift cache (`wazabin_qcode_sleigh::cache`)

Written 2026-09-18, rewritten 2026-09-21. Everything below is verifiable
from this tree; no conversation context is needed. The design itself is
documented in `sleigh/src/cache.rs`'s module docs, which are the source of
truth; this file is the state of the work, the numbers and the follow-ups.

## 1. State of the tree

- Repository `~/dev/nochurn/qcode`, branch `encoding-cache`, rebased on
  `origin/main` at `da90043`. Three commits:
  the exact-encoding cache, the shape keying with combined probes, and the
  cheaper capture.
- Depends on wazabin-sleigh 0.1.12, which ships `Decoder::decode_one_shaped`
  (wazabin/sleigh#10) and `Shape::registers` (#12), both released
  2026-09-21. `sleigh/Cargo.toml` names that version.
- Do not touch `~/dev/juju` or `~/dev/binary`: other agents work there.

Checks that pass as of this handoff (CI's commands):

```sh
cargo test --workspace --all-features            # 814 passed
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --check && cargo doc --workspace --no-deps
```

## 2. What it is, briefly

`LiftCache` keys templates on an encoding's *shape* — the bits the decoder
decided on, the more specific patterns it passed over, and its parameter
fields — in a masked trie per (flat flag, decode context, leading byte),
walked over the byte stream without a decode. A hit is an `Arc<Template>`
and an `Instance` (address delta plus per-parameter deltas); every
constant of the template is affine in those. See the module docs for the
template, the two hit paths (scratch views reading through a template,
`LiftSession` replaying through the builder) and exactness.

A miss lifts for real, then probes: one near lift with the address moved
and a low free bit of every parameter flipped (a different bit each, so no
two deltas coincide), classifying each constant by the unique subset of
deltas summing to its move, and one far lift with every free bit flipped,
verifying the sums. An ambiguous or structure-changing probe falls back to
probing one thing at a time; a nonlinear parameter (`pshufd` lane
selectors) or an instance no perturbation keeps intact (`jmp $`) falls back
to an exact-encoding entry. `LiftCache::validating(true)` lifts every hit
for real as well and compares.

Public surface:

```rust
LiftCache::new(&CompiledSpec) .with_capacity(usize) .validating(bool) .is_validating() .stats() .clear()
ScratchSession::with_cache(Arc<LiftCache>)   LiftSession::with_cache(Arc<LiftCache>)
CacheStats { hits, misses, probes, uncacheable, exact, validation_failures, entries }
```

`QCODE_CACHE_DEBUG=1` prints every miss that made no template and why.

## 3. Measurements

Retired user instructions (`perf stat -e instructions:u`) of
`examples/lift-throughput --stage lift [--linear] [--cache]` over the
`.text` of a binary, this tree. `--linear` computes its offsets with a
decode pass that is inside the count, so linear figures are decode-heavy.

| workload | no cache | cache | shapes (entries) | misses | hits |
|---|---|---|---|---|---|
| `/usr/bin/ls`, every offset | 8.75G | 4.35G | 3 937 | 4 361 | 79 337 |
| `/usr/bin/ls`, linear | 1.79G | 1.01G | 1 018 | 1 075 | 20 860 |
| `/usr/bin/bash`, linear | 14.6G | 3.66G | 1 797 | 1 983 | 250 430 |
| `/usr/bin/bash`, every offset | 95.5G | 16.4G | 12 093 | 14 174 | 952 417 |

Since 2026-09-21 (evening) register fields are parameters too (wazabin/sleigh#12
reports them; see the module docs on registers and coincidences), which is
what took the shape counts from 2 842 / 8 286 (ls / bash linear) to 1 018 /
1 797. Misses exceed entries by the shapes that hold several templates,
one per way their registers coincide. Against disas-bench's xul.dll `.text`
(38 MB, 10.1 M instructions; `--offset 0x400 --len 0x24603E1`) the cached
lift runs at 11.1 MB/s cold (13 505 shapes, was 88 052 and 4.2 MB/s) and
19 MB/s warm (`--warm`, every instruction a hit), against 208 MB/s for iced
decoding alone and 3.2 MB/s for our decoder alone; peak memory 367 MB,
was 927 MB.

Exact-encoding keys, for comparison, missed 77 591 times on bash linear
and reached 10.5G there; shape keys with one probe per parameter reached
6.3G; combining the probes 5.9G; the vector numbering of captures and the
decoder's small maps 5.17G. Probe lifts per miss are about 1.5 (ls
superset: 14 115 over 9 215 misses; 47 % of misses have no parameter, 46 %
one, 5 % two). Validation reports zero disagreements on all four runs.

A decode is ~11k instructions on bash (was ~13k before the small maps),
and dominates everything uncached on linear code. Its remaining profile is
flat: the constructor walk, moving the 160-byte `ConstructorInstance`
through every nesting level (~5 per instruction, memmove ~12 %), and
allocation. Slimming the instance to 80 bytes was estimated at ~5 % of a
decode and not done.

## 3b. The session's replay (2026-09-21, late)

`LiftSession` hits replay a template into a growing `FunctionBody`, and
were bounded by that construction, not the cache: on `/usr/bin/bash` linear
(252 413 instructions, 8.4 operations and 3.8 named operations each) a
hit-only pass cost 4.55 µs per instruction (0.94 MB/s), 28k retired
instructions per replay, against 0.18 µs for a scratch (view-only) hit.
The example's `--warm` now works with `--session` — the warm pass runs in
a throwaway session — and prints the body's block, instruction and name
counts. Wall-clock on this laptop is noisy (another agent runs `juju` on
every core); measure retired instructions net of `--stage none`, or
callgrind over `/usr/bin/ls`.

What was done, all exact (the replayed function renders the same, names
included; `sleigh/tests/cache.rs` checks):

- `NameTable` in three tiers: `base_<n>` names as a vector per base
  indexed by suffix, canonical hex block labels keyed by their value, and
  a string map for everything else. Minting and registering a suffixed
  name is one small-table probe and one slot; growing the table no longer
  re-hashes a hundred thousand heap strings. `register_unique` fuses mint
  and register; serde is custom (a sequence of pairs; round-trip tests).
- `FunctionBody::append_insn`: push, use edges, link, name in one verb,
  behind `Builder::push_mnemonic_with_type_named` (its name is now
  `Option<&str>`). `add_use` matches once and uses `entry` for shared
  values.
- `ReplayScratch` on the session: no per-replay `Vec`s; a per-session
  `TypeKey → TypeId` memo (a scan; a hash cost more than the interner).
- Names baked into the template after probing (`Template::bake_names`,
  lowercase names per register table in `TableView`), so a replay does not
  look the register up and lowercase it per named operation.
- `AddressIndex` remembers the last address registered: the fall-through
  placeholder is the next instruction's entry.
- `suffixed_name`/`hex_name` by hand; `format!` was a third of naming.

Result: 2.4–2.5 µs per instruction hit-only (1.7 MB/s), 15.0k retired
instructions per replay; cold 2.9 µs (1.46 MB/s). Not 10 MB/s. Where the
15k go (exact, from callgrind on `ls`): per operation ~1 000 —
the 160-byte `Instruction` moved a few times (~120), use edges (~300,
half of it the `shared_first_use` hash for literal and varnode operands),
naming (~250 per named: the string allocation and the base probe), the
mnemonic clone and operand remap (~180), the builder and arena layers
(~200); per instruction ~2.5k — the fall-through placeholder block with
its label and index entry (~900), literal interning behind an `RwLock`
(~400), `begin`/`commit` (~700), the lookup (~800). Misses are a floor of
their own: 1 983 on bash at ~110 µs each (three decodes, three lifts, a
capture) is 0.2 s, so cold bash cannot pass ~5 MB/s whatever a hit costs.

Getting the session to 10 MB/s (~1.7k cycles per instruction) is not
more shaving: it needs a body that costs less per operation — prototype
`Instruction`s memcpy'd from the template with fix-up lists, use edges
built from a per-template plan, and no heap string per named operation
(names as `(base, suffix)` rendered on demand). That changes what a
`FunctionBody` stores, which is a design decision, not an optimisation.

## 3c. Making the eager path cheap (2026-09-22)

§3b's two routes were put to Jack, who ruled out deferring anything
("doing things later doesn't speed them up") and kept every guarantee:
the work stays inside the timed loop and the API loses nothing. What
followed is on the branch `lift-fast`, one commit per step, measured
each time (`perf stat -e instructions:u` net of `--stage none`, and
callgrind on a repeated encoding for exact per-operation costs).

Replaying one instruction of `/usr/bin/bash`, retired instructions and
peak memory of the whole session:

| | instrs/insn | peak |
|---|---|---|
| before §3b | 28 000 | 695 MB |
| after §3b (name tiers, append verb) | 14 400 | 558 MB |
| id and link packing | 14 400 | 558 MB |
| names as base + suffix | 11 000 | 484 MB |
| prototype append, dense use heads | 9 400 | — |
| inline edge sets, cheaper labels, batched runs | **8 300** | 493 MB |

What each step did: `LocalValueId` 16 → 8 bytes (the module-interned ids
are `u32`), an instruction's links and address packed (`value::link`),
argument lists `Box<[_]>` rather than `Vec` — `Instruction` 160 → 96,
`Mnemonic` 80 → 56; function-local names as `{BaseId, suffix}` rendered
on demand, with a block at an address holding no name at all
(`value::name`); `FunctionBody::append_prototype`, which takes a
recorded run and does per operation only what the body must, with the
template's operations, type ids and name bases built once per template
and session; use-list heads of literals and varnodes in dense vectors; a
block's incident edges inline; a block's instruction list read and
written once per run.

It also fixed a replay bug found by writing the test for it: a replayed
branch made no CFG edge, so a session's hits had blocks with no
successors (`a_session_hit_builds_the_same_edges`).

Not 10 MB/s: measured hit-only throughput is ~2–2.5 MB/s (the laptop is
usually loaded; instruction counts are the metric to track). Exactly
where the 8 300 go, from callgrind on one repeated encoding — a `nop`
(one operation) costs 3 900 and a 17-operation instruction 13 200, so
**583 per operation and ~3 300 fixed**:

- Per operation: `append_prototype` itself 250 (the mnemonic clone, the
  operand walks, the 96-byte push), `add_use` 94 for two edges,
  `StableArena::push` 63, the use-edge arena 52, naming 54, the replay's
  scratch 33.
- Per instruction: the fall-through placeholder block with its label and
  index entry ~750, the undecoded key walk ~560, `begin`/`commit` ~300,
  the replay's prologue ~360, allocator ~200.

`HANDOFF_LIFT_FAST.md` is the handoff for carrying this on: how to
measure on a loaded machine, what the next moves are worth, and what has
already been measured and rejected — including a `reserve` on jstd's
arenas, which is a small loss rather than the gain it looked like.

## 4. Rejected along the way

- A per-session memo of recent encodings (first eight bytes → template)
  in front of the shared trie: 17–37 % of lookups hit it on xul.dll, but
  the `Arc` traffic of remembering every trie hit cost more than the walk
  it saved.
- Inline storage for `Lifted`'s block and exit lists: kept, but worth
  only ~2 % of the cached lift.

- Classifying constants on the p-code stream instead of the lifted IR:
  branch targets are RAM-space varnodes, not constants, and value
  collisions (`ret 8` against a pop of width 8) made the commonest
  immediates ambiguous.
- Boxing the child instance in wazabin-sleigh's `OperandValue`: under 1 %.
- A probe-only capture that skips building operations: `Mnemonic` is large
  and `non_exhaustive`, so comparing operations without cloning them is
  impractical; the clone is ~6k of a ~45k capture.

## 5. Tests (`sleigh/tests/cache.rs`)

`a_scratch_hit_renders_as_the_uncached_lift` (58 encodings × 5 addresses,
`uncacheable == 0`), `a_variant_of_a_shape_is_served_from_the_first_instance`,
`validation_agrees_on_every_hit`, `a_session_hit_builds_the_same_function_names_included`,
`the_cache_is_shared_across_threads`, `a_full_cache_stops_remembering`,
`a_cache_refuses_another_specification`.

## 6. Tools

```sh
B=target/release/examples/lift-throughput
$B /usr/bin/ls [--linear] --stage lift --cache [--validate]   # prints CacheStats
cargo run --release -p wazabin-qcode-sleigh --example shape-survey -- /usr/bin/ls
cargo run --release -p wazabin-qcode-sleigh --example shape-dump -- <hex>
```

`shape-survey` counts distinct shapes against distinct encodings and
cross-checks the decoder's mask by perturbation. For call graphs, build
with `RUSTFLAGS="-C force-frame-pointers=yes"` into another target
directory and `perf record -g`.

## 7. Follow-ups, by expected payoff

1. **Decode speed** is the floor for everything uncached and for every
   miss (a miss decodes ~3 times: the shaped decode and one per probe).
   The walker's cost is spread out; a real gain is a different matching
   strategy or arena-allocated instances, in wazabin-sleigh.
2. **Cheaper hits.** A scratch hit on real code is a few thousand
   instructions of key walk, `Arc` clone and `lifted_at`'s allocations.
3. **Bulk IR append for `LiftSession` hits**: see §3b — the cheap part is
   done; the rest is a change to what a body stores per operation.
4. **`lift_decoded` through the cache** needs the decode context on
   `sleigh::Instruction`.
