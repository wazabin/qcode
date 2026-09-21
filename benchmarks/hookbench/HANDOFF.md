# Handoff: evaluating QCode's hooks for the article

Written 2026-09-18. Everything here lives on the `emulator-suite` branch,
checked out as a worktree at `~/dev/vm/qcode-suite` (main stays at
`~/dev/vm/qcode`). Nothing is pushed: the branch was rebased, so updating
`origin/emulator-suite` is a force-push and is left to you.

## The artifact we are working on

The results page: **https://claude.ai/artifact/7ch4BjnCMKhTTtSg49VhWB**
("QCode Hook Costs", version 2). It is generated, not hand-written: `make_page.py
target/results out.html` in `benchmarks/hookbench` rebuilds it from the sweep's
JSON, and republishing the same file to that URL updates it. The Markdown twin of
the same numbers is `benchmarks/hookbench/RESULTS.md`, committed.

## What the article wants to say, and what the numbers support

The claim: hooks compiled to QCode are cheap, and adding them is easy where
patching native binaries is hard. What we measured, on the 17 Embench-IoT images
built freestanding for x86-64 (geometric mean, min of 5 runs, quiet machine):

| instrumentation | QCode JIT compiled | QCode JIT callback | icicle | Unicorn C | Unicorn Python | AFL++ QEMU | native compiler |
|---|---:|---:|---:|---:|---:|---:|---:|
| block counter | 1.26× (hook space) / 1.41× (guest RAM) | 5.47× | 1.04× | 1.13× | 16× | | 2.34× |
| instruction counter | 1.33× / 1.85× | 14.7× | 1.08× | 3.71× | 118× | | |
| AFL edge map | 1.55× | | | | | 1.00× | 3.13× |
| write watch, 32 bytes | 1.71× | 11.3× | 1.11× | 1.87× | 2.2× | | 1.99× (hw watchpoints) |
| compare log | 6.91× | fails | | 1.13× (cmp only) | | | 1.96× |

Per host callback: QCode 400 to 650 ns, icicle and Unicorn's C API 3 to 8 ns,
Unicorn's Python binding about 1.2 µs (Qiling's regime).

The supportable statements:

1. Compiling the hook is a 4× to 8× difference under QCode. That is the
   article's point, and it holds.
2. QCode's compiled counters now cost what icicle's injected p-code costs, once
   the hook keeps its state in a flat hook space rather than guest RAM. Icicle
   remains the closest competitor; QCode's edge over it is the hook layer
   (typed sites, idempotence, conditions as IR), not raw speed.
3. QCode's callback path is a machine stop and is two orders of magnitude
   slower than icicle's or Unicorn's callbacks. Say so; the compiled path is
   the answer to it.
4. AFL++ QEMU mode's edge instrumentation is free on top of QEMU, but QEMU's
   base is 5 to 10× slower than native; QCode's edge map costs 1.55× on a base
   that is also emulation. Put the baselines side by side rather than the
   ratios alone.
5. The compare hook is not comparable across engines: QCode's site is every
   p-code comparison (about 1800 per image, every flag computation), Unicorn's
   TCG hook sees 343 `cmp` instructions on crc32.

Caveat on baselines: native and AFL++ time one warm iteration inside the
process; QCode, icicle and Unicorn run the image once from `main`, which
includes translation. The page says so; the article should too.

## What we did, in order

1. **Compared the tools' hook mechanisms** from source. Unicorn: C callbacks
   per instruction in a hooked range, memory hooks force the slow path for
   every access to a hooked page. Qiling: Python dispatch on top. Icicle:
   hooks are p-code ops injected by a `CodeInjector`; memory hooks are MMU-side.
   QCode: hooks rewrite the lifted IR at typed sites (`vm/src/hook.rs`).
2. **Counted interfaces**: Unicorn 67 C entry points, Qiling about 120, QCode
   66 (21 of them the hook emitter). Missing in QCode relative to Unicorn:
   named register access, CPU context save and restore (now covered by the
   branch's snapshots), `until`/timeout on `run`, MMIO, other architectures.
3. **Rebased `emulator-suite` onto main** (12 userland commits carried; PRs #2
   and #3 on main were squashes of the first 8 branch commits, dropped).
   Conflicts were in the JIT chaining condition, the discovery loop, the
   manifests, and main's linked-list instruction order, which needed a port of
   the snapshot code (`6387c3f`). All 783 workspace tests pass.
4. **Built `benchmarks/hookbench`**, a standalone crate (excluded from the
   workspace) that runs the same instrumentation under qcode-interp, qcode-jit,
   Unicorn (Rust binding, path dep on `~/dev/unicorn`, needs `-latomic`),
   icicle (`~/dev/icicle-emu`, needs `GHIDRA_SRC` pointing at a dir with
   `Ghidra/Processors`; a symlink to pypcode's processors dir works), Unicorn's
   Python binding (`python/unicorn_py.py`), AFL++ QEMU mode
   (`~/dev/AFLplusplus/afl-qemu-trace`, built from a fresh checkout), and native
   binaries with gcc sanitizer coverage and perf hardware watchpoints
   (`native/`). `run-all.sh` is the sweep, `report.py` and `make_page.py` the
   outputs. The README covers the build.
5. **Fixed what the harness found** (each its own commit):
   - `39be029` hooks can emit guest loads and stores; block-entry sites anchor
     on the first *guest* instruction, so a hook that emits IR at the entry is
     not re-applied every time the block grows (it was, tripling the count).
   - `3347004` cheaper stops: no diagnostic string per interrupt, O(1)
     position lookup instead of a linked-list walk, no instruction-list rebuild
     per block. Callback cost went from about 5 µs to 0.4–0.6 µs.
   - `e1170a9` compiled code is resumed only from an instruction boundary:
     re-entering the JIT in the middle of an instruction's p-code produced
     wrong state (edn faulted, huffbench hung) where the interpreter was
     right. This is a workaround; the JIT bug is open.
   - `605e4b6` a flat hook state space (`Emitter::state_space`, `load_from`,
     `store_to`), and spaces configured after every injection (a space first
     touched by compiled code was unconfigured and read as unwritten).
6. **Ran the sweep twice.** The first ran on a machine with load 20 from an
   unrelated job and hung overnight on the JIT bug above; the numbers in
   `RESULTS.md` and on the page are from the second run on a quiet machine
   (load 1.6), with one process and a timeout per run.

## What we learnt

- **Where hook state lives decides the compiled cost.** Guest RAM means the
  JIT's inline TLB lookup, permission check and init-bit update on every
  access. A flat space is a base pointer plus constant offset. That single
  change took the per-instruction counter from 1.85× to 1.33× (1.07–1.13× on
  most images) and is why icicle looked faster.
- **Block granularity is subtle.** Lifting is per instruction and absorption
  rebuilds basic blocks; hooks are applied at injection, which can be before
  absorption. The anchoring fix makes block-entry hooks count basic blocks
  (350k on crc32, matching Unicorn's TB count within TB-size differences).
- **A callback under QCode is a stop**: exit compiled code, describe the
  interrupt, dispatch, resume, re-enter. The accidental costs are gone; what
  is left is the design. Do not promise callback parity with Unicorn.
- **The JIT's part-way re-entry is fragile mid-instruction.** Symptoms: wrong
  base register a few instructions after a store hook's interrupt. Interpreting
  the rest of the block is correct and is the current behaviour.
- **`cmp-cb` fails everywhere** with `Error("value 0 is too large to
  represent")` from the interpreter's array-typed load path after resuming past
  an interrupt placed before a comparison. Not diagnosed. Reproduce with
  `hookbench --engine qcode-jit --instr cmp-cb --only crc32`.
- **Icicle's own injector faulted once** (block-ir on sglib-combined,
  ReadUnmapped); likely the harness's temp allocation in
  `src/icicle_backend.rs`, not icicle.
- Unicorn's `until` needs the sentinel page mapped; icicle reports the
  sentinel as an ExecViolation; QCode as `Unlifted` at the sentinel.

## Done since: the JIT cache (2026-09-18, later)

Steps 1 and 2 below had one cause, found by tracing the store callbacks
under both engines (`HOOKBENCH_TRACE=1`) to the first divergence: the JIT
validated its block cache by block id and instruction count, and a block
the VM empties and lifts again from the same bytes has as many
instructions as before under new ids. The cached code imported the stack
pointer from the block's previous life. Commit `c83141b`: every function
body stamps a block with a fresh revision on any edit to its instructions
(`BasicBlock::revision`, bumped by the link verbs and the operand
replacers in `core/src/value/function.rs`), the JIT keys both caches on
it, and `Vm::resume` offers the executor the rest of the block from
anywhere. `watch-cb` and `cmp-cb` verify on all 17 images under the JIT;
edn's `watch-cb` run went from 2.1 s to 0.56 s. The re-measured QCode JIT
rows are in `target/results-fix1` (with a `notes.txt`); `make_page.py`
and `report.py` take later runs as `label=dir` arguments and show every
run side by side in a progress section, so the page keeps its history.

## Done since: bounded hook spaces (2026-09-21)

Step 3 below. `Vm::state_space(name, len)` makes a flat hook space of a
fixed length (`FlatSpace::bound`: allocated whole, growth past it refused
under interpreter and JIT alike). Given the bound, the JIT compiles a
load or store at a *computed* address into the space as one unsigned
compare against `bound - size` and the access off the base pointer; a
failed check stops the block with `BLOCK_OVERFLOW` before the access and
the VM reports the interpreter's `AddressOverflow`. A computed address
into an unbounded space is still declined. Commit `c3ad0a6`, tests in
`jit/tests/hooks.rs` (map indexed natively, identical under both
strategies; an overrun stops both alike).

The harness (`1a76249`) moved `edge-ir` and `cmp-ir` into bounded spaces
and kept the RAM versions as `edge-ram` and `cmp-ram`. The first two
runs' `edge-ir`/`cmp-ir` measured the RAM layout, so `relabel.txt` in
`target/results` and `target/results-fix1` renames them to `edge-ram`/
`cmp-ram` when the report and the page load them (the loaders honour
that file; it is untracked, like the JSON).

**The machine was loaded (load 15–25 from other jobs) and would not go
quiet**, so the re-measure was made two ways that survive load
(`ae4317c`, README): thread CPU time (`HOOKBENCH_CPUTIME=1`, rows carry
`"clock": "cpu"`) and retired user instructions (`perf stat -e
instructions:u`, `instructions.txt` in the run dir; deterministic to six
digits under any load). Such a run stays out of the charts and appears in
the progress table only. Results in `target/results-fix2` (qcode-jit,
`none edge-ir edge-ram cmp-ir cmp-ram`, 17 images, all verified):

| kind | CPU time, geomean | instructions, geomean |
|---|---:|---:|
| edge-ram (was `edge-ir`) | 1.48× | 1.35× |
| edge-ir, bounded space | 1.12× | 1.14× |
| cmp-ram (was `cmp-ir`) | 4.08× | 5.01× |
| cmp-ir, bounded space | 2.52× | 2.14× |

The edge map went from 1.5× to about 1.1×, which is what step 3
promised and where icicle's trace store sits. The compare log halved in
time and is 2.3× fewer instructions; what remains is the site count
(every flag computation: nettle-sha256 alone is 20× in instructions,
the median image 1.5×), not the store path. The page and `RESULTS.md`
carry the fix2 column; the chart rows for the hook-space edge map and
compare log stay empty until a quiet wall-clock run (step 8).

## Done since: the growing-block cost (2026-09-21, later)

Asked why the compiled block counter sat at 1.14× where icicle's is
1.04×: the geomean was carried by nettle-sha256 at 3.07× (every other
image 1.02–1.12×). perf showed 56% of that run walking the interpreter's
instruction list. Straight-line code is discovered an instruction at a
time, each absorbed into the block being run, and the injectors are
offered the block after every absorption; sha256's compression is one
block of ~10k p-code ops, offered ~2,200 times, and each offer (a)
dropped the interpreter's list unconditionally in `Vm::inject`, so the
next step walked back to position ~9,900, and (b) had the hook's
`addresses()`/`stores()`/`compares()` walk the whole block. Both
quadratic in the block's length. Commit `085e89a`: `inject` reports
whether the block's revision moved and the list is dropped only then
(the absorption path lets `resume_after` extend it from the tail);
`HookInjector` keeps a per-block frontier and offers the block from its
successor (`BlockView::since`; ids are never reused). Test: a
65-instruction run offers a hook ≤ 195 sites, where it offered 2,145.
`HOOKBENCH_DECLINES=1` (`57ad395`) lists what a fresh JIT declines.

`target/results-fix3`: every QCode JIT row, still on a loaded machine
(load 11 → 4), so read the instruction column:

| kind | fix1, wall (quiet) | fix3, instructions |
|---|---:|---:|
| block-ir | 1.14× | 1.03× |
| insn-ir | 1.20× | 1.05× |
| edge-ir | (1.52× in RAM) | 1.08× |
| watch-ir | 1.58× | 1.37× |
| cmp-ir | (6.39× in RAM) | 1.99× |

The worst image for the counters is now 1.05–1.06×. What is left on
`cmp-ir` for sha256 (8.6× in instructions) is Cranelift compiling a
300k-op function — `regalloc2::domtree::merge_sets` — once; a cap on
compiled block size, or splitting huge blocks, would be the fix, and is
not a hook-layer matter. The chart rows still await a quiet wall-clock
run; the load was 3.7 when this was written.

## Potential next steps

1. ~~Fix the JIT's mid-instruction re-entry~~ — done, see above.
2. ~~Diagnose `cmp-cb`~~ — same bug, done.
3. ~~Dynamic offsets into flat spaces in the JIT~~ — done, see above.
4. **Fold hook conditions.** The cleanup passes run at lift time, before
   injection; running `remove_dead_insns` and constant folding after injection
   would simplify a write watch whose store address is constant, and is the
   "same IR as the analysis tool" argument made concrete.
5. **Per-block instruction counting**: `insn-ir` bumps once per instruction;
   a hook that adds the block's instruction count once per block entry is the
   obvious optimisation and a good demonstration of hook logic as IR.
6. **A typed register API** generated from `wazabin-sleigh`'s register
   iterator (`runtime.rs` `registers()`), giving `Vm::reg_read`/`reg_write`;
   `userland/src/regs.rs` already does the resolve-once pattern by hand.
7. **Qiling proper**: install into a venv and run its shellcode mode on the
   images, or accept the Python-binding floor as its stand-in (documented).
8. **Re-run with `--repeat 5` on a quiet machine before publishing**; check
   `target/results/load.txt`. The interpreter was run on four images only.
   The hook-space `edge-ir` and `cmp-ir` have no wall-clock numbers yet
   (see above); that run fills the chart rows.
9. Push the branch (force) and open a PR; the commit messages carry the
   reasoning.

## Where things are

- Worktree: `~/dev/vm/qcode-suite` (branch `emulator-suite`, 30 commits over
  main).
- Harness: `benchmarks/hookbench/`; results in `target/results/`
  (untracked), tables in `RESULTS.md`, page generator `make_page.py`.
- Images: `~/dev/vm/qcode/target/embench` (from `benchmarks/embench/build.sh`,
  Embench checkout at `~/dev/embench-iot`); native binaries in
  `benchmarks/hookbench/target/native`.
- External engines: `~/dev/unicorn`, `~/dev/icicle-emu`, `~/dev/AFLplusplus`
  (with `afl-qemu-trace` built), Ghidra specs via
  `~/.virtualenvs/angr/.../pypcode/processors` symlinked as `Ghidra/Processors`.
- Memory notes for future sessions are under the project's memory directory
  (`qcode-article-evaluation`, `hookbench-harness`).
