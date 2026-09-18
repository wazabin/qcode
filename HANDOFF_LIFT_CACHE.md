# Handoff: the encoding-keyed lift cache (`wazabin_qcode_sleigh::cache`)

Written 2026-09-18. Everything below is verifiable from this tree; no
conversation context is needed.

## 1. State of the tree

- Repository: `~/dev/nochurn/qcode`, branch `encoding-cache`, rebased on
  `origin/main` at `17e46cd` (PR #26, 2026-09-18). **Nothing is committed**:
  `git status` shows the work as modified and untracked files.
- Sibling checkouts in `~/dev/nochurn` (`wazabin-sleigh`, `wazabin-pcode`,
  `wazabin-jstd`, `wazabin-binary`) are not used by this workspace; qcode
  takes its wazabin dependencies from crates.io (`wazabin-sleigh` 0.1.8).
- Do not touch `~/dev/juju` or `~/dev/binary`: other agents work there. The
  consumer that motivated this work (`~/dev/juju/src/summary.rs`, the
  superset-disassembly `Summarizer`) is read-only reference material.
- Open PR #28 (`register-file`) adds a `spec` field to every scratch view in
  `sleigh/src/session.rs`; this branch rewrote those views, so whichever
  lands second needs a small merge.

Files changed (`git diff --stat`, plus untracked):

| file | change |
|---|---|
| `sleigh/src/cache.rs` (new, ~1300 lines) | `LiftCache`, `Template`, `Classifier`, capture/replay/validation |
| `sleigh/src/session.rs` | `with_cache` on both sessions; scratch views read from the store **or** a template; lazy decode of cached hits; per-session length hints |
| `sleigh/src/lib.rs` | `SleighLifter::lower_recording`, `ConstSink`, `flat_control_flow()`, crate docs |
| `sleigh/tests/cache.rs` (new) | 6 tests, see §5 |
| `sleigh/examples/lift-throughput.rs` (new) | throughput probe, see §6 |
| `sleigh/README.md` | one section |
| `core/src/value/insn/mnemonic.rs` | `Mnemonic::map_operands` |
| `core/src/builder.rs` | `push_mnemonic_with_type_named`, `push_temp`, `push_temp_space` |
| `core/src/value/function.rs` | `temp_count`, `temp_space_count` |
| `core/src/lift.rs` | `Lifted::new`, `Exit::new` |

Checks that pass as of this handoff:

```sh
cargo test -p wazabin-qcode-sleigh -p wazabin-qcode      # 576 passed
cargo clippy -p wazabin-qcode-sleigh -p wazabin-qcode --all-targets -- -D warnings
cargo fmt --check -p wazabin-qcode-sleigh -p wazabin-qcode
```

## 2. What it is

`LiftCache` (`sleigh/src/cache.rs`) remembers what every encoding lowered
to, as a `Template`, and serves it again at any address. The key is
`[flat flag] ++ decode-context bytes ++ instruction bytes` (`fn key`), so
sessions with different decode contexts or call lowerings never share
entries. The cache is bound to one `CompiledSpec` fingerprint and returns
`LiftError::IncompatibleSpec` for another lifter. It is `Sync`: 32 shards of
`RwLock<FxHashMap<Box<[u8]>, Entry>>`, atomics for the counters, an
`AtomicUsize` entry count bounded by `with_capacity` (default 1 << 20; when
full, new encodings are lifted and not remembered).

Public surface:

```rust
LiftCache::new(&CompiledSpec) -> Self
    .with_capacity(usize) .validating(bool) .is_validating() .stats() -> CacheStats .clear()
ScratchSession::with_cache(Arc<LiftCache>)   LiftSession::with_cache(Arc<LiftCache>)
ScratchLifted::is_cached()                   // and decoded() now decodes lazily
CacheStats { hits, misses, uncacheable, validation_failures, entries }
```

Only `lift(address, bytes)` goes through the cache. `lift_decoded(...)` does
not: a caller-decoded `Instruction` does not carry the context it was
decoded under, and the key needs it (`sleigh::Instruction` has no context
accessor; adding one to wazabin-sleigh would lift this restriction).

### 2.1 The template

`Template` is the emitted IR of one lift, relative to the address it was
captured at (`base`):

- `blocks`: the instruction's own blocks after the entry, with how the
  emitter named them (`BlockName::{Label, Fallthrough, Other, None}`; the
  names embed the address, so they are re-rendered at replay).
- `externals`: relative addresses of other instructions' blocks the IR
  names, sorted by address (that order is the index scheme), plus the
  fall-through, which `FlatEmitter::new` always resolves even when unnamed.
  `external_order` is the order the emitter made their placeholders in, so
  a replay issues identical block ids.
- `callees`: relative addresses of direct-call targets (`Callee::Minted(k)`
  in the stored mnemonics; resolved with `Construction::callee_at`).
- `temp_spaces` / `temps`: **every** space and temporary the lift appended
  (from the `Marks` taken before the lift), not only the referenced ones —
  the emitter creates sub-range temporaries via `Builder::get_range` that
  nothing references, and ids must come out identical.
- `literals`: `(value, size, TypeKey, relative)`; `TypeKey` is
  `Int(size) | Bool | SpaceAddress(size, space)` because `TypeId`s differ
  between contexts and a comparison's `false` is a `bool`, not an `i8`.
- `types`: result types of the operations, as `TypeKey`.
- `ops`: in instruction-id order (which is emission order, so operands
  always name earlier operations), each `{block, ty, mnemonic, name}`. The
  mnemonic's operands are template indices dressed as `LocalValueId`s:
  `Literal(k)`, `Instruction(k)`, `Temp(k)`, `BasicBlock(k)` (own blocks
  first, then externals), `Load/Store.space = Temp(k)`; varnodes stay
  as they are (architecture-defined, same in every context). `name` is the
  emitter's base name (`base_name`: a `_<n>` suffix is stripped when the
  prefix names a register), so the target body re-numbers it as a fresh lift
  would.
- `exits`: `Lifted`'s exits with sites as op indices and addresses relative.

`Template::capture(ctx, addresses, marks, lifted, classifier)` builds one from
a committed lift; it returns `Err(&'static str)` for anything a template
cannot hold (switches, applies, block params, named temporaries, a callee
without an address, an operand naming a later operation…). Set
`QCODE_CACHE_DEBUG=1` to have every uncacheable encoding and its reason
printed.

### 2.2 The two hit paths

- **Scratch session** (`ScratchSession::lift`, `session.rs`): first tries
  `LiftCache::find_undecoded` with the lengths this session has seen for the
  leading byte (`lengths: Box<[u16; 256]>`, learned after each cached lift).
  Valid encodings are prefix-free under one context, so at most one length
  matches; shortest first. On a hit nothing is decoded or lowered:
  `ScratchLifted` holds `template: Some(Arc<Template>)` plus
  `Lifted::new(...)` synthesised by `Template::lifted_at`, and every view
  (`ScratchBlock`, `ScratchInsn`, `ScratchOperand`) reads through
  `Source::Template { ctx, template, address }` instead of
  `Source::Store(ctx)`. The store is left as it was (no reset). `decoded()`
  decodes on first call (`OnceCell`); `bytes()` slices the stream by the
  template's length. If the undecoded lookup misses, the instruction is
  decoded and `find` (decoded key) / `miss` run as for a LiftSession.
- **LiftSession** (`LiftSession::lift`): `LiftCache::lower` → `find` →
  `Template::replay(target, address)`, which resolves externals and callees,
  interns literals and types once, then pushes spaces, temporaries and
  operations through the builder in template order
  (`push_mnemonic_with_type_named`), and reports blocks and exits to the
  `Emitter`. The result is id-for-id identical to a fresh lift, names
  included (tested).

### 2.3 The miss path and exactness

`LiftCache::miss`:

1. `SleighLifter::lower_recording` lifts for real and returns the constant
   operands the `FlatEmitter` saw in p-code stream order (recorded in
   `PcodeSink::op` inputs and `branch_label` conditions; the emitter's
   thread-local `Workspace` carries the `Vec`).
2. `Classifier::probe` re-decodes the bytes at `address + PROBE_DISTANCE`
   (`0x1_0305_0709_0b0d`, > 4 GiB so a 32-bit truncation cannot pass for an
   offset) and streams the p-code into `ConstSink`. Same count required;
   constant by constant: equal → absolute, moved by exactly the distance
   (masked to the constant's width) → relative, else uncacheable. A value
   shared by an absolute and a relative constant is marked ambiguous.
3. `Template::capture` classifies each QCode literal through
   `Classifier::classify(value, size)`: by (value, size), then by value
   (the builder resizes literals in `coerce_literal_size`), then as a
   sub-range of a constant (`Builder::get_range` folds literal ranges: low
   bytes of a relative constant are relative, higher bytes are refused);
   a value matching nothing is the builder's own and absolute. Ambiguity
   is refused.
4. The entry (template or `Uncacheable`) is inserted; uncacheable
   encodings are never probed again.

The claim this rests on: every QCode literal is a p-code constant, a resize
of one, a sub-range of one, or an address-independent builder constant. It
is documented in the module docs and checked empirically by
`LiftCache::validating(true)`, which on every hit lifts for real into a
thread-local probe store (`with_probe_store`, one `ScratchStore` per
spec per thread) and compares templates (`agrees_with`); a mismatch is
counted, the entry evicted, the instruction lifted for real. The validating
cache never answers the undecoded lookup, so every hit is checked.

## 3. Measurements

Wall time was unusable during this work (load average 8–14 from other
agents' jobs). Everything was measured as retired user instructions, net
of process startup:

```sh
cargo build --release -p wazabin-qcode-sleigh --example lift-throughput
B=./target/release/examples/lift-throughput
instr() { perf stat -e instructions:u "$@" 2>&1 >/dev/null | grep instructions | sed 's/^ *//;s/ .*//;s/,//g'; }
base=$(instr $B --bytes 90 --linear --iters 1 --stage lift)     # startup: spec load etc.
u=$(instr $B --bytes dec9 --linear --iters 3000 --stage lift); c=$(instr $B --bytes dec9 --linear --iters 3000 --stage lift --cache)
echo $(( (u-base)/3000 )) $(( (c-base)/3000 ))
```

Per lift, `origin/main` code, this branch:

| case | uncached | cached hit | notes |
|---|---|---|---|
| `fmulp` (de c9) | 2.52 M | 6.5 K | 1178 QCode insns, 561 temps |
| `mulsd xmm0,[mem]` | 2.33 M | 16.7 K | |
| `addps xmm0,xmm1` | 227 K | 16.7 K | |
| `add rax,rcx` | 127 K | 11.9 K | |
| `nop` | 12.5 K | 2.8 K | fixed per-lift cost |
| `/usr/bin/bash`, every offset (1.08 M offsets, 77 % hits) | 92 K | 41 K | per offset |
| `/usr/bin/bash`, linear (69 % hits) | 62 K | 40 K | |
| `/usr/bin/ls`, every offset (68 % hits) | 92 K | 61 K | |
| `/usr/bin/ls`, linear (56 % hits) | 72 K | 62 K | |

A decode alone is ~22 K instructions on real code (~1.2 µs); a miss costs
about 1.8 lifts (real lift + probe decode + p-code stream + capture). So
single-binary gains are bounded by hit rate; a cache shared across a corpus
(the juju analysis measured 81–86 % distinct-encoding repetition) does
better. Zero uncacheable encodings and zero validation failures were seen
on `ls` and `bash` and on the test corpus.

Profiles (`perf record -g`, `perf report --no-children -g none`) of the
cached superset run are flat: SLEIGH decode (`Walker::try_build_constructor`,
`Tree::get_constructor`), `Template::capture`, malloc, and the miss lifts.

## 4. Where the time went, and what the design rejected

- A template **replayed through the builder** (the LiftSession path) is only
  ~2× cheaper than a lift: `fmulp` 2.52 M → 1.29 M. The cost is the IR data
  structure itself — use edges (`add_use`/`first_use_of`/`set_first_use_of`,
  a `HashMap` insert per use of a shared value), the block linked list,
  `RecyclingArena::push` of ~200-byte `Instruction` records, the
  35-variant operand visitor run twice per op. That is why scratch hits
  stopped materialising.
- A **second full lift** for classification (the first design) made a miss
  2.7× a lift; the p-code-stream probe is ~0.45 of a lift.
- Caching **flat p-code** alone would save ~20 %: for `mulsd` the p-code
  stage is 91 µs of 278 µs, the rest is emission.
- Two apparent hot spots that were not: bincode/serde symbols in profiles
  are the one-time spec load; `FlatEmitter` iteration order is
  deterministic (no `HashMap` iteration in emission).

## 5. Tests (`sleigh/tests/cache.rs`)

- `a_scratch_hit_renders_as_the_uncached_lift`: 58 encodings × 5 addresses
  (`0x1000`, `0x40_1234`, `0x7fff_ffff_f000`, `0x1_2345_6789`,
  `0xffff_ffff_ffff_ff00`), cached vs uncached scratch renders equal;
  asserts `uncacheable == 0` and exact hit/miss counts. The render blanks
  literal ids and other instructions' block ids (numbering schemes differ)
  and compares resolved constants and block addresses instead.
- `validation_agrees_on_every_hit`: same corpus with `validating(true)`,
  `validation_failures == 0`.
- `a_session_hit_builds_the_same_function_names_included`: a 13-instruction
  sequence lifted as one function through `LiftSession`, flat and
  structured lowering, `Lifted`s equal and `ctx.to_string()` equal.
- `the_cache_is_shared_across_threads`, `a_full_cache_stops_remembering`,
  `a_cache_refuses_another_specification`.

`int3` (`cc`) is excluded from the corpus: it trips
`debug_assert!(self.dirty.is_empty(), "a block was left without a spill")`
in `FlatEmitter::enter_block` (`sleigh/src/lib.rs`) on `origin/main` with
no cache involved. That is a pre-existing bug of PR #18's spill logic
(its p-code ends `CallOther; CallInd; Return`), worth its own fix.

## 6. Tools

`sleigh/examples/lift-throughput.rs`:

```sh
$B /usr/bin/ls                       # every byte offset, stages decode / pcode / lift
$B /usr/bin/ls --linear --cache      # linear sweep through the cache, prints CacheStats
$B /usr/bin/ls --cache --validate    # validating cache
$B --bytes dec9 --iters 3000 --linear --stage lift [--cache]   # one encoding in a loop
```

It contains a 30-line ELF `.text` reader (no dependency). `qcode-dump <hex>`
prints text, AST, flat p-code and QCode of one instruction; the Criterion
benches are `cargo bench -p wazabin-qcode-sleigh --bench lift` and
`cargo bench -p wazabin-sleigh --bench decode` in the sleigh checkout.

## 7. Follow-ups, by expected payoff

1. **Cheaper misses.** A miss is real lift (1×) + probe decode (~0.3× on
   real code) + p-code stream (~0.4×) + capture (~0.15×). The probe decode
   exists only because `sleigh::Instruction` binds `inst_start` at decode
   time (disassembly actions compute `reloc = inst_next + simm32` then); a
   wazabin-sleigh API to re-address a decoded instruction, or to stream
   p-code at another address, would remove it. Capture allocates per op
   (`Mnemonic::clone`, `FxHashMap` numbering); in a fresh scratch epoch ids
   are dense so most maps could be vectors.
2. **Shape-level keys.** Mask ModRM/SIB displacement and immediate fields
   so `mov rax,[rip+X]` for every X shares one template. Needs an x86
   length/field decoder (~150 lines, decides nothing about semantics) and a
   second-tier lookup; the classifier already tells which literals are
   address-relative, and the same `validating` mode measures the rest. The
   juju analysis estimated 100 K distinct shapes vs 676 K distinct
   encodings over its corpus.
3. **Bulk IR append for LiftSession hits.** Replay is bounded by per-op
   construction (~600 instructions per entity). A `FunctionBody` verb that
   appends a recorded slab — contiguous ids, precomputed prev/next links
   and use-edge lists for template-internal values, one `shared_first_use`
   update per distinct shared value — would make a replay a few memcpys.
   Touches `core/src/value/function.rs` internals and `arena_integrity.rs`
   checks; the `Template` already has everything such a verb needs.
4. **Fixed per-hit cost** (~2.8 K instructions for `nop`): `lifted_at`
   allocates two `Vec`s and clones `Named` callee strings; the key is
   rebuilt per candidate length; `Arc` clone and shard `RwLock` read.
5. **Decode speed** is now the floor for everything uncached (~22 K
   instructions per real-code instruction in SLEIGH's walker); iced does
   the same work in a few hundred. Out of this crate's scope but the next
   wall.
6. **`lift_decoded` through the cache** needs the decode context on
   `sleigh::Instruction` (see §2).
7. The juju `Summarizer` (not to be edited from here) spends ~19 µs of its
   27 µs per offset in its own `Evaluator::walk` after the lift; a hit-cheap
   lift does not remove that. Its earlier plan to cache summaries per
   encoding is orthogonal and still valid.
