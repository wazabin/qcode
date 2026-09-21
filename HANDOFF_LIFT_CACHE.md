# Handoff: the shape-keyed lift cache (`wazabin_qcode_sleigh::cache`)

Written 2026-09-18, rewritten 2026-09-21. Everything below is verifiable
from this tree; no conversation context is needed. The design itself is
documented in `sleigh/src/cache.rs`'s module docs, which are the source of
truth; this file is the state of the work, the numbers and the follow-ups.

## 1. State of the tree

- Repository `~/dev/nochurn/qcode`, branch `encoding-cache`, rebased on
  `origin/main` at `da90043` (wazabin-sleigh 0.1.10). Three commits:
  the exact-encoding cache, the shape keying with combined probes, and the
  cheaper capture.
- **Depends on unreleased wazabin-sleigh.** `Decoder::decode_one_shaped`
  is in wazabin/sleigh pull request #10 (branch `shaped-decode`, also the
  constructor maps as small vectors). `Cargo.toml` patches
  `wazabin-sleigh` to that branch by git revision so CI can build. To
  merge: land #10, release wazabin-sleigh 0.1.11, bump `sleigh/Cargo.toml`
  to it and drop the patch.
- The sibling checkout `~/dev/nochurn/wazabin-sleigh` holds that branch;
  the path patch it used to be reached through is gone.
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

| workload | no cache | cache | shapes (misses) | hits |
|---|---|---|---|---|
| `/usr/bin/ls`, every offset | 8.75G | 5.87G | 9 215 | 74 483 |
| `/usr/bin/ls`, linear | 1.79G | 1.43G | 2 842 | 19 093 |
| `/usr/bin/bash`, linear | 14.6G | 5.17G | 8 286 | 244 127 |
| `/usr/bin/bash`, every offset | 95.5G | 23.7G | 35 518 | 931 076 |

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

## 4. Rejected along the way

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
3. **Bulk IR append for `LiftSession` hits**: replay is bounded by per-op
   construction in `FunctionBody`; a slab-append verb would make it a few
   memcpys.
4. **`lift_decoded` through the cache** needs the decode context on
   `sleigh::Instruction`.
