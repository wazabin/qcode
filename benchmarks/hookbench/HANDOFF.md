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

## Done since: the rebase onto main, and the JIT warm-up (2026-09-21, evening)

**Rebased onto main at `0e30226`** (13 commits: the lifting consolidation,
p-code uniques kept as SSA values within a block, memoized lifts, the
Criterion bench in CI). Three textual conflicts (the lock, the manifest,
`forward_temp_stores`) and one commit of follow-ups (`359fe8d`): main now
guards the address index with a shape clock and keeps the body verbs behind
`QCodeMut`, so the VM's eviction of written-over code uncovers absorbed
addresses through a new `BlockMut::uncover_absorbed`, clears the block
through `QCodeMut::clear_block_instructions`, and vouches for the index it
maintained by hand — without that, every lift after an eviction was refused
with `ForeignIndex`. A cloned body keeps its revision counter. The
repeated-restore test settles the code before comparing two runs of one
budget: a budget is exact to the block under an executor, and main's
lifting changed the loop block's length. All 870 tests pass.

`target/results-main2` re-measures every QCode JIT row after the rebase
(loaded machine, CPU time and instructions). Main's changes moved the
baseline itself: 5–10% fewer instructions per run, and the compare sites
halved (nettle-sha256 23,640 → 11,123: fewer redundant flag comparisons
once uniques are SSA), which took `cmp-ir` from 1.99× to 1.62× in
instructions. The counters stayed at 1.03×/1.05×.

**Where the base run's time goes.** Asked what would make hook placement
faster, the profile said the hooks are no longer where the time is: on
`nsichneu` (`none`), 37% of the run was Cranelift, 19% malloc under it,
11% `Jit::run_block` itself (the per-block entry: base pointers, imports,
exports) and 7% the compiled code. `QCODE_JIT_TRACE`-style timing of every
compile (not committed) gave: nsichneu 1,670 compiles for 1,038 blocks,
632 blocks compiled more than once, 322 ms of a 540 ms run; nettle-sha256
130 compiles for 80 blocks, 84 ms of 235 ms. A trivial block costs
Cranelift 40 µs, a typical 20–30-op block 200 µs (opt_level `none`
saves a fifth of that and was not worth the slower code). Two causes: a
block is compiled from the shape it has when it is entered, and a VM that
lifts on demand grows a block by absorbing the instructions it discovers
after it, so straight-line code is compiled after each absorption; and
code that runs once is compiled for nothing.

**The warm-up** (`7a925de`): the JIT's cache slot counts entries at a
revision and compiles on the second (`Jit::set_warm_up`, default 2; 1 is
the old behaviour). A rebuilt block starts over, which is the point — the
second entry to the same revision is what says the block has stopped
changing. `try_compile` still compiles at once, for tooling; a
continuation after an interrupt is compiled at once, as before. A third
entry gains nothing more. `target/results-warm` re-measures every row
(loaded machine, load 2–6): CPU time over `results-main2`, geomean of 17
images:

| kind | faster by | kind | faster by |
|---|---:|---|---:|
| none | 1.27× | edge-ir | 1.35× |
| block-ir | 1.27× | watch-ir | 1.28× |
| insn-ir | 1.31× | cmp-ir | 1.30× |
| block-cb | 1.09× | watch-cb | 1.17× |
| insn-cb | 1.06× | cmp-cb | 1.03× |

The ratios over the baseline are unchanged for the compiled hooks
(block 1.03×, insn 1.05×, edge 1.07×, watch 1.37×, cmp 1.65× in
instructions) and a little higher for the callbacks, whose absolute cost
did not move while the baseline shrank.

**The quiet wall-clock run** (`target/results-warm-wall`, `--repeat 5`,
load 1–2; a few rows a burst of load inflated were re-measured, see its
`notes.txt`) fills the chart rows at last — step 8 of the list below. QCode
JIT, geomean over 17 images, over its own uninstrumented run:

| block-ir | insn-ir | edge-ir | watch-ir | cmp-ir | block-cb | insn-cb | watch-cb | cmp-cb |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.05× | 1.10× | 1.13× | 1.65× | 1.94× | 5.9× | 17× | 3.9× | 78× |

against icicle's 1.04× / 1.08× / – / 1.11× and Unicorn's 1.13× (block
callback) / 3.71× (instruction callback) / 1.87× (write hook). The
baseline itself: crc32 24.8 ms (was 33.0 in the first sweep), nsichneu
322 ms (was 448), nettle-sha256 171 ms (was 210); icicle is 10.8, 215
and 142 ms on those. `RESULTS.md` and the page (version 6) carry all of
this; the progress table has every run.

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
   reasoning. The warm-up (`7a925de`) and the post-rebase fixes (`359fe8d`)
   are VM and JIT changes that belong on main; cherry-pick them ahead of
   the bench.

Where the base run's time goes now, and what would move it (measured on
`nsichneu`, the block-hopping image, after the warm-up):

10. **Compilation is still half the run on short images.** nsichneu: 197 ms
    of 371 compiling 1,009 blocks, ~195 µs each, in Cranelift's codegen
    (`define_function`: egraph, lowering, regalloc2); translation is 10%
    of that, `finalize_definitions` 7%. Levers, in order of promise:
    tiering (opt_level `none` first, `speed` on a hot count — `none` saved
    a fifth of codegen here); compiling off the critical path on another
    thread, running the interpreter until the code lands; and compiling a
    region — a loop, a function — as one Cranelift function instead of a
    block, which also removes the next item.
11. **The per-block round trip.** `Jit::run_block` is 11% self time and
    `Context::block` 4% on nsichneu, against 7% in compiled code: every
    block entry resolves the cache slot, takes a base pointer per flat
    space, copies imports in and exports out, and hands the terminator to
    the interpreter, which resolves the successor. Chaining decides direct
    and indirect branches inside `run_block` but still returns per block.
    Emitting the terminator natively (a direct jump to the successor's
    code when it is compiled, patched in when it is) is the classic fix.
12. **A cap on compiled block size.** `cmp-ir` on nettle-sha256 is 3.7× in
    instructions, mostly Cranelift's register allocator on a 68k-op block
    (a single compression function absorbed whole); compiling such a block
    in pieces, or declining above a size and interpreting it, bounds that.
13. **Hook injection itself is now cheap** (linear in what a block gains,
    `085e89a`); the remaining cost of a compiled hook is its operations,
    as the instruction ratios show. What would still help the article's
    hooks: folding hook conditions after injection (item 4), and the
    per-block instruction counter (item 5).

## Where things are

- Worktree: `~/dev/vm/qcode-suite` (branch `emulator-suite`, rebased onto
  main `0e30226` on 2026-09-21; 34 commits over main).
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
