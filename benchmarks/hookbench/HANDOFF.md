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

## Potential next steps

1. **Fix the JIT's mid-instruction re-entry** (`jit/src/compile.rs`
   `translate_body` with `start > 0`, and `jit/src/jit.rs` imports). Then drop
   the boundary rule in `Vm::resume` and re-measure `watch-cb`, which is
   currently the interpreter's cost.
2. **Diagnose `cmp-cb`.** Start from the interpreter's `ValueError(0)` sites
   in `emulator/src/concrete.rs` (array-typed loads) and the IR dump
   (`HOOKBENCH_DUMP=file`).
3. **Dynamic offsets into flat spaces in the JIT**, bounds-checked, so the
   edge map and the compare log can leave guest RAM. Icicle's trace store does
   this; it would bring `edge-ir` from 1.55× toward 1.1×.
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
9. Push the branch (force) and open a PR; the commit messages carry the
   reasoning.

## Where things are

- Worktree: `~/dev/vm/qcode-suite` (branch `emulator-suite`, 19 commits over
  main, clean tree).
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
