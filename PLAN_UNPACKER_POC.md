# Plan: provenance-aware unpacker POC (`jit/examples/unpack`)

Written 2026-09-22. Every claim about the tree below was checked against
`main` at `0e30226`. The design decisions come from a /grill-me session and
are recorded in the project memory `unpacker-poc-design`; this file is the
work plan.

## 0. Ground rules

- The unpacker is an **example** (`jit/examples/unpack/`), not VM code.
- The VM changes are two `Emitter` methods, `store` and `load`
  (`vm/src/hook.rs`), and one fix to existing machinery approved during
  implementation: block absorption now continues through the address-less
  tail an `interrupt_if` split leaves (`split_tail_address` in
  `vm/src/hook.rs`; `absorb_into_basic_block`, `settle_absorbed`,
  `drop_absorbed_from_index` in `vm/src/vm.rs`). Everything else uses the
  VM as it is.
- Deferred to a later milestone, deliberately: overwrite versioning
  (`Vm::invalidate`, `BlockExecutor::forget`, content-hash identity),
  `remove_injector`, the SMC code-bitmap hook, dynamic ELF (ld.so, auxv
  beyond the static minimum, TLS), influence taint, execution counts.
- CI gates (unchanged): `cargo fmt --check`, `cargo clippy --workspace
  --all-targets --all-features -- -D warnings`, `cargo test --workspace
  --all-features`, `cargo doc --workspace --no-deps`. Examples compile under
  clippy's `--all-targets`; they are exercised by `jit/tests/unpack.rs`.

## 0a. Branch correction (2026-09-22, after §12)

This pass was built on `main`; the intended base is `emulator-suite`
(worktree `~/dev/vm/qcode-suite`, rebased on main `0e30226`), which already
has the `userland` crate (loader, SysV stack, ~75 syscalls, `cpuid`/`rdtsc`,
processes), `Emitter::load/store`, bounded hook state spaces
(`Vm::state_space`, `Emitter::load_from/store_to`), cheaper stops, the
growing-block re-offer, a revision-keyed JIT cache and SMC invalidation of
lifted code the guest writes over. The POC is therefore being ported to
`userland/examples/unpack/` on that branch, on `userland::Process` plus
state spaces with a zero-stop first-entry log (no `interrupt_if`, so the
absorption issue of §12 does not arise there). What stays on `main` from
this pass: this plan, the fixture (`jit/tests/fixtures/unpack/`), and the
`Emitter::load/store` + absorption-through-split-tails change with its
tests, as a separate PR candidate (the suite has its own `Emitter::load/
store`; the absorption fix is still a real bug for any `interrupt_if`
placed at a block entry).

## 1. VM: `Emitter::store` / `Emitter::load`  (vm/src/hook.rs)

Thin wrappers over `Builder::push_store` / `push_load`, shaped exactly like
`Emitter::binop`: insert before the anchor, no guest address stamped (see
the comment on `binop` for why), default (RAM) space.

```rust
/// Loads `size` bytes of guest RAM at `ptr` before the anchor.
pub fn load(&mut self, ptr: ValueId, size: usize) -> ValueId
/// Stores `value` (its own width) to guest RAM at `ptr` before the anchor.
pub fn store(&mut self, ptr: ValueId, value: ValueId)
```

Facts that make this small: the IR already has `Load`/`Store` mnemonics;
the JIT compiles RAM loads/stores inline (`jit/src/compile.rs:600`,
`:621`); `BlockView::is_ram` shows how the default space is named
(`MemorySpaceId::Shared(ctx.shared.default_space)`).

Tests, in `jit/tests/hooks.rs`, run on both strategies like the existing
ones:
- a store-site hook that increments a guest counter inline; the run never
  exits and the counter equals the number of stores;
- an entry hook gated by `interrupt_if(load(flag) == 0)`; the host sets
  the flag at the first stop; the block re-runs without stopping.
Both must agree interpreter vs JIT (the `drive_with` helper there).

## 2. Example skeleton

```
jit/examples/unpack/main.rs   thin CLI (clap): unpack <elf> --out DIR [--jit] [--edges] [--budget N] [--no-hooks]
jit/examples/unpack/lib.rs    pub mod layout, loader, linux, hooks, driver, graph, artifact
jit/tests/unpack.rs           #[path = "../examples/unpack/lib.rs"] mod unpack;  (fixture-driven tests)
jit/tests/fixtures/unpack/    selfdecrypt.c, build.sh, selfdecrypt (committed), hello.c, hello.upx (committed once upx is available)
```

`jit/Cargo.toml`: add `[[example]] name = "unpack"`; dev-deps add
`wazabin-binary.workspace = true` and `bincode.workspace = true` (both are
already workspace deps).

## 3. Guest memory layout  (layout.rs)

All constants live in one place; the hooks bake them in.

| region | address | purpose |
|---|---|---|
| image | as linked (`ElfBinary.load_address`) | PT_LOAD segments |
| arena | `0x1000_0000_0000 .. +1 GiB` | bump allocator behind `mmap`/`brk` |
| WINDOW | `[image_lo, arena_end)` | the range provenance tracks |
| SHADOW | `0x2000_0000_0000 ..` | `shadow(a) = SHADOW + (a - WINDOW.start) * 2`, u16 per byte |
| SINK | 16 bytes, outside WINDOW | where out-of-window shadow stores land |
| VISITED | 1 MiB, outside WINDOW | one byte per instrumented block |
| LAST | 4 bytes, outside WINDOW | index of the last block entered (`--edges`) |
| stack | `0x7fff_f000_0000 - 8 MiB`, outside WINDOW | untracked on purpose |

Shadow is mapped alongside every mapping inside WINDOW (whole image span
at load, each arena allocation as it happens): a store's shadow write runs
*before* the guest store, so every in-window mapped byte needs shadow or
the hook faults where the guest would not have.

## 4. Static ELF loader  (loader.rs)

`wazabin_binary::elf::ElfBinary::parse` → for each `LoadSegment`:
`mmu.write_unchecked(start, data, perms)` for file-backed bytes and
`mmu.map(start + data.len(), mem_size - data.len(), perms)` for the
zero-fill tail; perms from `executable`/`writable` (`RX_INIT` / `RW_INIT`
/ both). Stack: map, then build argc/argv/envp and the minimal auxv a
static glibc needs (`AT_PHDR AT_PHENT AT_PHNUM AT_PAGESZ AT_RANDOM(16
bytes) AT_ENTRY AT_SECURE AT_UID/EUID/GID/EGID AT_NULL`); set `RSP`,
enter at `analysis.entrypoint` with `Vm::at_address`. Registers are set
with `emulator().set_varnode_by_name` as `qcode-run` does.

## 5. Linux syscalls  (linux.rs)

`vm.hook_insn("syscall", ...)`: nr in `RAX`, args `RDI RSI RDX R10 R8 R9`,
result to `RAX`, `InsnAction::Handled(None)`. Implement: `read` (fd 0 →
0), `write` (fd 1/2 → captured into the artifact), `mmap` (anonymous only;
arena bump + shadow map; file-backed → `-ENOSYS`), `munmap`, `mprotect`
(`mmu.protect`), `brk`, `exit`/`exit_group` (record status, stop the
run), `arch_prctl(ARCH_SET_FS)` (verify the x64 SLEIGH name of the FS base
varnode at implementation time), `set_tid_address`, `set_robust_list`,
`rseq`, `prlimit64`, `getrandom` (deterministic bytes), `uname`. Anything
else: `RAX = -ENOSYS`, continue, and the (nr, pc) goes into
`artifact.missing_syscalls`. Check whether the `syscall` semantic already
clobbers `RCX`/`R11`; if not, do it here.

## 6. Provenance hook  (hooks.rs, `ProvenanceHook`)

`sites` = `block.stores()`. `instrument`, per store site:
1. `site = sites.push({ pc: emit.address(), block: block address })`;
   ids above `0xFFFF` saturate to `0xFFFF` and set `artifact.sites_saturated`.
2. `(ptr, size, _) = emit.store_operands()`; `p = zext(ptr, 8)`.
3. `cond = in_range(p, WINDOW)`; `shadow = SHADOW + ((p - WINDOW.start) << 1)`.
4. Branchless select, no new control flow:
   `mask = 0 - zext(cond, 8)`; `dst = SINK ^ ((shadow ^ SINK) & mask)`.
5. Shadow bytes = `2 * size`; emit stores of ≤ 8 bytes each holding the
   site id splatted as u16 lanes (`constant`, then `store(dst + off, c)`).
About ten IR ops per store site; nothing leaves compiled code.

## 7. First-entry recorder  (hooks.rs, `EntryHook`)

`sites` = `block.entry()`. `instrument`: `k = blocks.push({ addr })`;
`flag = load(VISITED + k, 1)`; `cond = flag == 0`;
`interrupt_if(cond, FIRST_ENTRY, [addr, k])`; with `--edges` also
`store(LAST, k)` **after** the interrupt so the stop still sees the
predecessor. Hot cost after the first entry: one load, one compare, one
predictable branch (plus one store with `--edges`).

Driver loop (`driver.rs`): on `VmExit::Interrupt { kind: Explicit { code:
FIRST_ENTRY }, args: [addr, k] }` → write `visited[k] = 1`; read the
block's bytes and its shadow (`mmu.read`); record node `k` = { addr,
bytes hash, generating sites = distinct shadow ids ≠ 0 }, `generated_by`
edges from each site's block, and (with `--edges`) an observed edge
`LAST → k`; then `vm.resume(None)` and `run` again. Explicit codes below
`TABLE_CODES` reach the caller as `VmExit::Interrupt` (see
`Vm::dispatch`), so no table registration is needed. The run ends on the
exit syscall, a fault, `Unlifted`, or the budget.

## 8. Harvest and artifact  (graph.rs, artifact.rs)

After the run, from the final `Context` (`ctx.blocks()`, `address()`,
`instructions()` → address/size range, `successors()`):
- node per lifted block with an address; `generated = any shadow ≠ 0 over
  its range`; `generation = 0` for image code, else
  `1 + max(generation of generated_by parents)`;
- edges: `control_flow` from `successors()` marked `static` (IR
  terminators) and, with `--edges`, `observed` from the recorder;
  `generated_by` from step 7.
Output `--out DIR`: `context.bin` (bincode of the serde `Context`),
`graph.json` (nodes, edges, sites, regions, syscalls seen/missing,
captured stdout/stderr, exit status), `regions/<addr>-g<gen>.bin` (maximal
runs of generated *and* executed bytes per generation), and a one-screen
text summary on stdout. Note: runtime discovery does not link CFG edges
for indirect targets, so without `--edges` indirect edges are absent, not
wrong.

## 9. Fixtures and tests

- `selfdecrypt.c` (+ `build.sh`): `-static -nostdlib` x86-64, raw
  syscalls only. Stage 0 XOR-decrypts stage 1 into a RWX `.data` buffer
  and jumps; stage 1 decrypts stage 2 into memory it obtains with `mmap`
  and jumps; stage 2 `write`s a line and `exit`s. Two generations, an
  arena allocation, a handful of syscalls, a few KiB, deterministic.
  Committed.
- `hello.upx`: `gcc -static hello.c && upx --best`. `upx` is not installed
  on this machine (`sudo dnf install upx`); commit the binary once built.
- `jit/tests/unpack.rs`:
  1. `selfdecrypt`, JIT: runs to exit; stdout captured; nodes with
     generation 1 and 2 exist; every generation-≥1 node has ≥ 1
     `generated_by` edge; `context.bin` round-trips through serde.
  2. `selfdecrypt`, interpreter vs JIT: identical `graph.json`.
  3. `hello.upx` (skipped when the fixture file is absent): runs to
     `exit_group`, prints `hello`, ≥ 1 generation-1 region.
- Perf number (printed, not gated): `selfdecrypt` and `hello.upx` under
  JIT with hooks vs `--no-hooks`; hookbench's compiled-hook ratios
  (block 1.03×, watch 1.37×) are the reference.

## 10. Order of work and checkpoints

1. §1 Emitter methods + tests → CI green.
2. §2–§4 skeleton, layout, loader; `unpack selfdecrypt --no-hooks` runs
   to exit with the syscalls of §5 → stdout captured.
3. §6 provenance hook; assert shadow is non-zero over the decrypted
   buffer after the run.
4. §7 recorder + driver; generation-1/2 nodes appear.
5. §8 harvest + artifact; tests 1–2 pass.
6. `hello.upx` once `upx` is available; test 3.

## 11. Known limits of the POC

Shadow, VISITED and SINK are guest-visible memory (placed far from the
image). Site ids saturate at 65,535. A sample that overwrites code the VM
has already lifted keeps running the stale IR until the versioning
milestone; harvest re-hashes each lifted range and flags the mismatch.
Stack writes are untracked by construction. Unknown syscalls return
`-ENOSYS`, so guests may take unexpected paths; they are listed.

## 12. State of the tree (2026-09-22, end of the POC pass)

Implemented, uncommitted, on `main` at `0e30226`: §1–§9 plus the absorption
fix. `cargo fmt --all --check`, `cargo clippy --workspace --all-targets
--all-features -- -D warnings`, `cargo test --workspace --all-features` all
pass. Files: `vm/src/{hook,vm,lib}.rs`, `jit/Cargo.toml`, `jit/examples/unpack/`
(`main lib layout loader linux cpu hooks driver graph artifact`),
`jit/tests/{hooks,unpack}.rs`, `jit/tests/fixtures/unpack/` (all-assembly
`selfdecrypt`, 4.4 KiB, two `PT_LOAD`s, reproducible).

Numbers (release, `--jit`):
- `selfdecrypt`: exit 7, `unpacked: stage 2`; 10 nodes lifted / 7 executed;
  4 generation-1 nodes attributed to the single stage-0 store site
  (`0x4000c0`), 2 generation-2 nodes to the stage-1 site (`0x40103a`);
  regions exactly 127 B (gen 1) and 54 B (gen 2); interpreter and JIT write
  byte-identical `graph.json` and `context.bin`; 7 first-entry stops.
- `gcc -static -O2 hello`: `--no-hooks` 0.55 s / 1,373,417 steps; hooks
  0.97 s / 1,598,932 (1.76× wall, 1.16× steps, 879 stops); `--edges` 1.26 s.
  Before the absorption fix hooks were 3.3–3.5 s (6.4–6.9×) because every
  guest instruction behind an entry gate became its own compiled block.
- Provenance hook: 11 IR ops per 1/2/4-byte store site (13 for 8 bytes),
  zero exits. Entry gate: 3 ops on the hot path.

Deviations from the plan that stand: `cpu.rs` answers the `cpuid_*` ops (a
static glibc resolves IFUNCs before `main`); `writev` added; `PT_LOAD`s are
mapped page-rounded (glibc's RELRO `mprotect` needs it); `mmap` honours
`MAP_FIXED`; shadow ids are `index + 1` (0 = never written); `SINK` is 128
bytes; block ranges come from the final IR by address gaps (the `Context`
keeps no guest instruction lengths); the entry recorder records
`sites_at_first_entry` and the harvest recomputes everything from the final
`Context`.

Follow-ups, in priority order:
1. The absorption fix leaves absorbed addresses in a split tail's
   `extra_addresses` while dropping them from the index; a full
   `AddressIndex::analyze` rebuild would re-derive them and could route a
   branch to the tail's start (the VM never rebuilds — a stale index errors
   loudly — but the clean version wants a `Context` verb to drop a block's
   extra addresses, or `resolve_block_at` treating an address-less block as
   not a target).
2. A guest store that is lifted twice (tail duplication behind a gate) gets
   two site ids; both resolve to the same pc, so the graph is right, but
   `sites` could be keyed by pc for the analyst.
3. The 879 first-entry stops on `hello` are most of the remaining 1.76×
   (each is a host round trip with a hash and a shadow read); batching or
   deferring the per-stop work to harvest would bring it near 1.1×.
4. `fstat`/`readlinkat` return `-ENOSYS` (harmless for `hello`); the
   `hello.upx` fixture is absent until `upx` is installed.
5. Then the deferred milestone of §0.

## 13. State on `emulator-suite` (2026-09-22, the port)

Lives in `userland/examples/unpack/` (`main lib layout hooks driver graph
artifact`), tests in `userland/tests/unpack.rs` (9, fixture-gated), fixture
in `userland/tests/fixtures/unpack/`. **No VM or userland source change**;
one pre-existing clippy failure fixed in `jit/tests/hooks.rs:488`.

Built on `userland::Process` (loader, stack, syscalls, `cpuid`) and three
bounded state spaces: `unpack.shadow` (u16 per byte over two windows —
image + brk heap, and the mmap arena at `0x7f00_0000_0000`, since a flat
space caps at 16 MiB and the two are 139 TB apart — with the sink at
offset 0), `unpack.entries` (cursor, last, then `(k, pred)` slots) and
`unpack.visited`. Both hooks are branch-free and never stop the run: 16 IR
ops per 1/2/4-byte store site, 10 per block entry (14 with `--edges`),
verified against step counts. Observed edges come from the entry log's
`(k, last)` pairs; nodes, generations, regions and hashes are computed at
harvest from the final `Context` and shadow.

Numbers (release, `--jit`, medians): `selfdecrypt` 1.06× wall (1.13× with
`--edges`), same attribution/regions as §12, interpreter = JIT byte for
byte; glibc `hello` 1.25× wall / 1.39× steps (1.28× / 1.49× with edges),
`native_bodies` and `absorbed` identical with and without hooks, `evicted`
0. A BusyBox applet runs under the hooks when a static busybox is present.

Follow-ups here: expose `MMAP_BASE` (private in `userland`) and
`MAX_FLAT_SPACE` (private in `vm/src/flat.rs`) — both restated in
`layout.rs`; a `Process` accessor for the allocations a run made and the
syscalls it saw (`graph.json` lost its `syscalls` section); then the
deferred milestone of §0, for which this branch's SMC invalidation
(`c2298e1`) is the prerequisite.
