# Handoff: the unpack example (`userland/examples/unpack`)

Written 2026-09-22. Everything below is verifiable from this tree; no
conversation context is needed. `PLAN_UNPACKER_POC.md` is the design and
its history (§0a and §13 are the parts that apply to this branch); this
file is the state of the work, the numbers, the traps and the follow-ups.

## 1. State of the trees

- **This branch**: `emulator-suite`, worktree `~/dev/vm/qcode-suite`, a
  rebase of the old `origin/emulator-suite` onto `main` at `0e30226`, plus
  `5589c32 feat(userland): unpack example, a provenance graph of generated
  code over state-space hooks`. Pushed as **`origin/emulator-suite-unpack`**
  (a new name: the rebase cannot fast-forward the old `origin/emulator-suite`,
  and that branch was left untouched). The old remote lineage holds nothing
  the rebase lost except `013e506`, a four-line tweak to
  `sleigh/tests/x64.rs`. To make `emulator-suite` point here:
  `git push --force-with-lease origin emulator-suite && git branch -u
  origin/emulator-suite`.
- **Against `main`**: `3a73411 fix(vm): absorb straight-line code through
  an interrupt_if split` on `origin/fix/absorb-through-interrupt-if-split`
  (worktree `~/dev/vm/qcode`, checked out on that branch; `main` there is
  one commit behind `origin/main`). See §8; the example here does not need
  it.
- No VM, JIT, emulator, core or `userland/src` change was needed for the
  example. One line outside it: `jit/tests/hooks.rs:488`, a pre-existing
  clippy failure (`vec![..].repeat` → `[..].repeat`).

Checks that pass as of this handoff (CI's commands, run from this worktree):

```sh
cargo test --workspace --all-features            # 59 suites, 0 failures
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check && cargo doc --workspace --no-deps
cargo test -p wazabin-qcode-userland --test unpack   # 9 passed, ~3.5 s
```

## 2. What it is, briefly

Run a static Linux x86-64 ELF under `qcode_userland::Process` with two
compiled hooks, then harvest from the final `Context` a graph of the code
the program *generated* at runtime and of *who wrote it*. The hooks never
leave compiled code: provenance is a store per guest store into a bounded
flat state space, first entries go into a log the same way, and the host
reads everything back once the process has exited. The unified IR is what
makes the harvest cheap: generated code was lifted into the same `Context`
the static engine uses, so `context.bin` is a module holding the unpacker
and everything it unpacked, labelled.

```sh
cargo run --release -p wazabin-qcode-userland --example unpack -- \
    userland/tests/fixtures/unpack/selfdecrypt --out /tmp/u --jit --edges
# exit: 7 / unpacked: stage 2 / gen 1: 4 nodes / gen 2: 2 nodes / regions: 2
```

Flags: `--jit`, `--edges` (observed control-flow edges, +4 IR ops per
entry), `--out DIR` (optional; no harvest without it), `--no-hooks`,
`--budget N`; trailing arguments after the path become the guest's argv
(`busybox echo hi`). `--help` for the rest.

## 3. Layout

```
userland/examples/unpack/main.rs      clap CLI; `#[path = "lib.rs"] mod unpack;`
userland/examples/unpack/lib.rs       pub mod artifact, driver, graph, hooks, layout
userland/examples/unpack/layout.rs    state-space geometry, the two windows, restated constants
userland/examples/unpack/hooks.rs     ProvenanceHook, EntryHook, Recorder, read_log
userland/examples/unpack/driver.rs    Options / Outcome / run / run_keeping over Process
userland/examples/unpack/graph.rs     harvest: nodes, generations, edges, regions, warnings
userland/examples/unpack/artifact.rs  context.bin, context.meta.json, graph.json, regions/*.bin
userland/tests/unpack.rs              `#[path = "../examples/unpack/lib.rs"] mod unpack;`
userland/tests/fixtures/unpack/       selfdecrypt (4.4 KiB), stage0/1/2.S, build.sh, README.md
```

Modules refer to each other with `super::` paths, never `crate::`: the same
files are included from the example root and from the test root. The
example is under `userland/` because it depends on `userland`, which
depends on `jit`.

## 4. Design as built

**State spaces** (`Vm::state_space(name, len)`, read back with
`vm.memory().flat().read_bytes`). A flat space caps at 16 MiB
(`MAX_FLAT_SPACE`, private in `vm/src/flat.rs`, restated in `layout.rs`),
and `userland` places `mmap` at `0x7f00_0000_0000` (`MMAP_BASE`, private in
`userland/src/process.rs`, restated), 139 TB above the image — so the
provenance window is **two ranges**: the image plus the `brk` heap
(`[image_lo, image_lo + 6 MiB)`) and the mmap arena (`[0x7f00_0000_0000,
+2 MiB − 4 KiB)`).

| space | length | contents |
|---|---|---|
| `unpack.shadow` | 16,769,104 | sink `[0,16)`, image-window shadow, mmap-window shadow, 64 B slack; u16 writer-site id per guest byte |
| `unpack.entries` | 1,048,600 | `cursor: u64`, `last: u32`, then 131,073 `(k: u32, pred: u32)` slots |
| `unpack.visited` | 131,072 | one byte per instrumented block (`MAX_BLOCKS = 1 << 17`) |

**Provenance hook**, per guest store site (sites = stores whose instruction
carries a guest address, which excludes the hook's own emitted stores):
`p = zext(ptr, 8)`; per window `Sub, Less, ShiftLeft, Add, zext, Sub, And`;
`Or` of the two candidates (the sink at offset 0 is what "in neither
window" selects to, for free); then the stamp — site id `index + 1`
splatted over `2 × size` bytes in ≤ 8-byte stores. **16 IR ops** for a
1/2/4-byte store, 18 for 8, 22 for 16. No control flow, no interrupt.
Site ids saturate at `0xFFFF` (`flags.sites_saturated`).

**Entry hook**, per `block.entry()`: `k = blocks.len()`; load
`visited[k]`; load `cursor`; store `k` at slot `cursor`; `cursor += 1 −
visited[k]`; store `visited[k] = 1`. **10 ops**, 14 with `--edges` (load
`last`, store it as the slot's `pred`, store `last = k`). A slot only sticks
when the cursor advances, i.e. at a first entry, so the surviving `(k, pred)`
pairs are exactly the first-entered blocks and the block that ran just
before each — the observed edge, indirect jumps included. Entries beyond
`MAX_BLOCKS` are not instrumented (`flags.blocks_saturated`). Nothing
here may stop the run: `Task::handle_interrupt` in `userland` crashes the
task on any explicit `vm.interrupt` it does not own.

**Driver**: `Process::new(bytes, Config { jit, stdio: Captured, argv,
exe_path, .. })`; create the three spaces on `process.vm()`; `add_hook`
provenance then entry; one `process.run(budget)`; `Outcome` carries exit,
captured stdout/stderr, steps, the recorder, `crashed`.

**Harvest** (`graph.rs`), from the final `Context` and the final shadow:
- one node per lifted block with a guest address; `range` from the IR's
  addresses — the `Context` keeps no guest instruction *lengths*, so
  `len(a) = next lifted address − a` when ≤ 15 else 1: exact inside a
  block, a lower bound at the last instruction of a run nothing follows;
- `generated` = any non-zero shadow over the range; `sites` = the distinct
  ids there; `generation` = 0 or `1 + max(generation of the node containing
  each site's pc)` by monotone iteration (cycles → warning, lower bound);
- edges: `control_flow` `static` from `successors()` (walking through the
  address-less pieces a split leaves), `observed` from the entry log with
  `--edges`, `generated_by` per site;
- regions: maximal runs of non-zero shadow touching an executed generated
  node, cut where the writing site's generation changes;
- warnings: an executed node outside both windows; a node holding the
  store that wrote it (self-modification hint). `flags.evicted` is this
  branch's SMC-invalidation counter (`c2298e1`), 0 on everything tested.

**Artifact** (`--out DIR`): `context.bin` (bincode 2 of the serde
`Context`), `context.meta.json`, `graph.json` (`program, io, nodes, edges,
sites, regions, windows, flags, warnings`; every address a `"0x…"` string;
nodes/edges/regions sorted so both strategies write identical bytes),
`regions/<0xstart>-g<gen>.bin`, and a one-screen summary on stdout.

## 5. Numbers (release, `--jit`, medians of 3)

| program | mode | wall | × | steps | × |
|---|---|---|---|---|---|
| selfdecrypt | `--no-hooks` | 79.7 ms | 1.00 | 15,239 | 1.00 |
| selfdecrypt | hooks | 84.2 ms | 1.06 | 21,785 | 1.43 |
| selfdecrypt | hooks `--edges` | 90.0 ms | 1.13 | 23,245 | 1.53 |
| glibc `hello` (`gcc -static -O2`) | `--no-hooks` | 301.6 ms | 1.00 | 1,395,254 | 1.00 |
| glibc `hello` | hooks | 376.8 ms | 1.25 | 1,944,810 | 1.39 |
| glibc `hello` | hooks `--edges` | 384.8 ms | 1.28 | 2,085,918 | 1.49 |

`vm.stats.native_bodies` and `absorbed` are identical with and without
hooks (4/42, 490/4100): the JIT compiles the same blocks and declines none
for hook code. The step deltas are exactly the op counts above times the
site/entry executions (verified on selfdecrypt: 6,546 = 10×365 + 16×181).

selfdecrypt result: 10 nodes lifted, 7 executed; 4 generation-1 nodes in
`0x401000..0x40107f`, each attributed only to site pc `0x4000c0` (stage 0's
`mov [rdi], al`); 2 generation-2 nodes at `0x7f00_0000_0000`, only to
`0x40103a` (stage 1's); regions of exactly 127 and 54 bytes, the stage
images; 4 sites (two ids per pc: the loop body is lifted twice, see §7);
interpreter and JIT byte-identical.

## 6. Tests (`userland/tests/unpack.rs`, fixture-gated)

`the_shadow_covers_both_windows_two_bytes_at_a_time`,
`an_image_wider_than_the_window_is_refused`,
`selfdecrypt_attributes_each_stage_to_the_one_that_wrote_it` (exit,
stdout, per-generation site pcs, region sizes, `generated_by` edges,
`context.bin` round trip, `!crashed`),
`the_two_strategies_write_the_same_graph` (with and without `--edges`),
`edges_link_a_stage_to_the_block_that_jumped_into_it`,
`the_log_holds_each_block_that_ran_exactly_once`,
`a_run_without_hooks_produces_the_same_program_and_no_graph`,
`a_static_glibc_hello_runs_under_the_hooks` (built at test time; skips
without `gcc`; 1519 nodes / 953 executed, nothing generated),
`a_busybox_applet_runs_under_the_hooks` (`$BUSYBOX` or a static busybox in
the usual places; skips otherwise).

## 7. Known limits and deviations

- Two windows (§4); an allocation outside them is untracked and only shows
  as a warning on an executed node. A `Process` accessor for the
  allocations a run made would make this checkable up front.
- `MMAP_BASE` and `MAX_FLAT_SPACE` are private upstream and restated in
  `layout.rs` (const-asserted where possible) — the most fragile constants.
- `graph.json` has no `syscalls` section: `Process` exposes no list of the
  calls a run made (only `Config.trace` to stderr). Unknown syscalls return
  `userland`'s own errno.
- A guest store lifted twice (a loop body absorbed into a run and later
  branched into) gets two site ids; both resolve to the same pc, so the
  graph is right, but `sites` is per lifted copy, not per pc.
- Ranges are address-gap derived (§4); a run's last instruction is
  under-measured by its length minus one.
- Sites saturate at 65,535, blocks at 131,072; bounded by design.
- Overwritten lifted code is invalidated by this branch (`evicted` counts
  it) but not *versioned*: nodes are address-only, there are no `overwrote`
  edges, and the fixture never overwrites lifted code.
- `hello.upx` (a UPX-packed static binary) is the intended second fixture;
  `upx` is not installed on this machine, so no real packer has been run.
- Only static Linux x86-64 ELFs (what `userland` loads).

## 8. The `main`-side fix (`fix/absorb-through-interrupt-if-split`)

Found by the first version of this example, which gated block entries
with `Emitter::interrupt_if`. That split leaves its tail without a guest
address, and `absorb_into_basic_block` refused an address-less head, so
every guest instruction behind such a gate stayed its own block: on glibc
`hello`, `absorbed` 3902 → 0, `native_bodies` 1523 → 9418, 6.9× wall. The
fix lets the tail absorb and drops what it absorbs from the address index
(an address-less block cannot be split, and the emulator would otherwise
enter it at its start for a branch into its middle); `split_tail_address`
recognises the split and refuses tails cut inside an instruction's p-code,
so store/load/compare sites absorb exactly as before. Result 6.4× → 1.9×.
Known limitation: the absorbed addresses stay in the tail's
`extra_addresses`; the VM never rebuilds an index (a stale one errors
loudly), but the clean form wants a `Context` verb to drop them, or
`resolve_block_at` treating address-less blocks as non-targets. The commit
also carries `Emitter::load/store` (this branch already has its own, from
`3efaeb8`), the plan and the fixture. The example here uses no
`interrupt_if`, so it does not need the fix.

## 9. Follow-ups, in order

1. Install `upx`, build and commit `hello.upx`, run it: the first real
   packer.
2. Versioning on `c2298e1`: node = (address, version), `overwrote` edges,
   content-hash identity — the largest missing piece of the graph.
3. Upstream the restated constants and add `Process` accessors for
   allocations and syscalls seen; restore `graph.json`'s `syscalls`.
4. A first static consumer of `context.bin` (resolve indirect targets in
   generated regions with an existing pass).
5. Adaptive tiers: page-level tier 0, and a VM `remove_injector`
   (`hook_del` drops only the callback; injectors and their baked-in
   interrupts stay).
6. Influence taint, designed as precise vs conservative from the start.
7. Execution counts (a compiled counter in a state space).

## 10. Traps for the next agent

- Work on **this** branch/worktree. `main` lacks `userland`, state spaces,
  `Emitter::load/store` and SMC invalidation; the first version of this
  example re-implemented all of them there before anyone noticed.
- `grep` the branch before proposing any VM primitive
  (`git grep <symbol> emulator-suite`).
- Hooks must not stop under `Process` (§4). Anything that needs the host
  mid-run is a design change, not a hook.
- A hook that emits stores to guest RAM offers itself as a store site on
  re-injection; state spaces avoid that, and `BlockView::since` re-offers
  only new instructions anyway. Keep the guest-address filter.
- Arming watchpoints (`WRITE_WATCH`) empties the TLB and makes every memory
  access take the slow path; never on a hot path.
- Flat spaces cap at 16 MiB; a computed access past the bound stops the
  run (AddressOverflow) — keep offsets in bounds by construction.
- `#[path]`-included example modules: `super::`, not `crate::`.
