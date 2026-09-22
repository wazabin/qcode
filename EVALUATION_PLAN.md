# Evaluation plan: provenance graphs over a shared static/dynamic IR

Written 2026-09-22. Companion to `PLAN_UNPACKER_POC.md` (design) and
`HANDOFF_UNPACK.md` (state). Runs from the `emulator-suite` branch once the
2026-09-22 working tree is committed.

## 0. Claims and research questions

The VM runs the same IR as the static analysis tool. Four claims follow;
every experiment below is tied to one.

| id | claim | evidence |
|---|---|---|
| C1 | Hooks compiled into the IR cost less than callback hooks in Unicorn, Qiling and icicle. | per-hook ratios (hookbench, done) and whole-task overhead (E3) |
| C2 | Instrumentation is easier to write: a hook is a few IR ops, not a native patch or a marshalled callback. | instrumentation size and shape per engine on one task (E3) |
| C3 | A provenance graph over that IR recovers what a static CFG cannot: code produced at runtime, and control flow split across processes. | byte-exact regions, CFG recall, cross-process edges (E1, E2, E4) |
| C4 | Generated code is lifted into the same `Context`, so the static tool consumes the result directly, with no rebuilt binary. | a static pass over `context.bin` yields facts a static run on the original file cannot (E5) |

- **RQ1** How complete and exact is the recovered code, per generator? (C3)
- **RQ2** How much executed control flow does the dynamic graph recover, and what does the static CFG miss? (C3, C4)
- **RQ3** What does recording cost, and how does the same task compare on the other engines? (C1, C2)
- **RQ4** Does the process model hold on real split-control-flow binaries, and at what cost? (C3)
- **RQ5** Is the result identical across interpreter and JIT, and stable across runs? (soundness of all of the above)

## 1. Corpus

Static Linux x86-64 ELFs, or dynamic ones run through the system `ld.so`
as the program. Each entry ships with source, build command, and ground
truth (§2). Target: 40 to 60 binaries in five families.

### 1.1 Compressed executables (RQ1, RQ2, RQ3)

Seeds, five: `hw` (write-only glibc hello), `tiny` (exit code only), three
Embench kernels rebuilt as userland binaries whose `main` prints a checksum
(crc32, nbody, sha256), plus BusyBox `echo` and `md5sum`.

Compressors: UPX 5.2.1 (`~/dev/upx/build/upx`) at default, `--best`,
`--lzma`, `--nrv2b`, `--ultra-brute`, `--no-filter`; UPX 4.2 release at
default and `--best` for the `/proc/self/exe` stub path without memfd;
other Linux ELF compressors best effort, each dropped and listed if its
output does not run natively. The `puts` hello crashing under the 5.2.1
beta is recorded as a tool bug, not hidden.

### 1.2 Runtime code generation with known provenance (RQ1, RQ2)

Tigress 4.0.10 (`~/Downloads/tigress/4.0.10`) on the same seeds:

- `Jit`: one function compiled at runtime into an anonymous mapping; the
  writer site, generation 1, and region bounds are known from the source.
- `JitDynamic`: re-generated on each call; exercises SMC invalidation and
  versioning (`evicted > 0`, one version per call).
- `EncodeData` + `Virtualize`: negative control, heavy data traffic, no
  code generation; must yield zero regions.
- `Split` + `Flatten` + `AddOpaque`: static-CFG stress control.

### 1.3 Self-modifying fixtures (RQ1, RQ5)

`selfdecrypt` (two generations, exists), plus three new fixtures: a stage
that overwrites code it already executed (versioning), a decryptor using
16-byte stores (site-size coverage), and a stage placed outside the mmap
window (must produce the documented warning, not silence).

### 1.4 Split control flow across processes (RQ4)

- Corpus guests, freestanding, native oracle: `nanomite.c` (exists: int3
  and fault, tracer rewrites rip), plus variants with `ud2`, single-step,
  SIGFPE in an in-process handler, and a two-thread hand-off over a futex.
- Real binaries from `~/dev/vm/nanomites`: `voracious.bin` (static, 15
  traced children, works), `nano.bin` and `ringgit` (need glibc clone
  flags and SIGFPE/SIGSEGV delivery to guest handlers). `ch12` is i386 and
  excluded. Run only under the emulator.

### 1.5 Negative controls (all RQs)

Unmodified static glibc hello, BusyBox applets (`echo`, `ls`, `sh -c`
pipeline), and the Embench userland builds: no regions, no generated
nodes, and the cost of the hooks on ordinary code.

## 2. Ground truth

| family | code ground truth | control-flow ground truth | provenance ground truth |
|---|---|---|---|
| 1.1 compressed | the seed's segments (byte-exact against the original ELF, as the UPX test does today) | static CFG of the seed from the same IR, plus a reference dynamic run of the seed (entry log, interpreter) | the stub's stores: all generation 1 |
| 1.2 Tigress | the Tigress-emitted buffer, dumped by an instrumented seed build at the generation call | the same, per version for `JitDynamic` | the emitting call site from the source; generation per version |
| 1.3 fixtures | the plaintext stage blobs (build.sh) | hand-listed edges | hand-listed sites |
| 1.4 split | none generated except the decrypted arena | the native oracle's stage order, and for the corpus guests the tracer table itself | tracer block to chosen stage |
| 1.5 controls | none | reference dynamic run | none |

The reference dynamic run is the seed executed under the same hooks: it
gives the set of executed edges the recovered graph must reach.

## 3. Metrics

Code (RQ1): bytes recovered over bytes expected per segment; mismatching
bytes; extra bytes (the stub trampoline counts as extra, and is reported);
regions expected vs found; generation depth correct.

Control flow (RQ2): recall = executed edges of the reference run found in
the graph; static-miss = edges in the graph absent from the static CFG of
the same file; for split control flow, cross-process edges found over
cross-process edges expected. Reported per binary and aggregated.

Provenance (RQ1, RQ4): site attribution correct (writer pc and generation
match ground truth); `sites` per pc vs per lifted copy (the known
duplication) reported as a ratio.

Cost (RQ3): wall time and retired operations with and without hooks, with
and without `--edges`, medians of 5 after one warm-up; `native_bodies`
and `absorbed` with and without hooks (must be equal); switch cost on
fork-heavy inputs (BusyBox pipeline, voracious: switches per run, bytes
copied per switch).

Instrumentation (C2): lines of hook code per engine, number of host
round-trips per guest store and per block entry, and whether the hook
can run without leaving compiled code.

Determinism (RQ5): graph bytes identical between interpreter and JIT
(already known to fail by one node on UPX: quantify and explain), and
across five repeated runs.

## 4. Baselines

Same task on each engine: a shadow byte per guest store into a bounded
window, a first-entry log per block, harvest of generated regions after
exit. Same corpus, same machine, same input.

| engine | how | expected shape |
|---|---|---|
| Qiling (Python) | `hook_mem_write` + `hook_block`, its own Linux loader and syscalls | one Python callback per store and per block |
| Unicorn (Rust binding, `~/dev/unicorn`) | same hooks, loader and syscalls borrowed from the hookbench harness | one native callback per event |
| icicle (`~/dev/icicle-emu`) | its injector API, as in hookbench | closest competitor, block-level |
| QCode | the unpack example | no callbacks; two compiled hooks |

Fairness rules: the baseline records the same facts, not more; each engine
gets its fastest supported configuration; a baseline that cannot run a
family (fork, ptrace) is marked unsupported for that family rather than
timed on a partial run; the harness reruns any point with more than 10 %
spread.

## 5. Experiments

Each experiment names its inputs, procedure, output table, and pass
criterion. All run through the harness of §6; nothing is timed by hand.

**E1 Code recovery (RQ1).** Families 1.1, 1.2, 1.3 under `--jit --edges`.
Per binary: bytes expected, recovered, mismatching, extra; regions;
generations. Pass: every code segment byte-exact; every Tigress region
attributed to its emitting site with the right generation; controls
yield zero regions. Output: Table 1 (per family), Figure 1 (bytes
recovered vs expected).

**E2 Control-flow recovery (RQ2).** Families 1.1, 1.2, 1.4. Per binary:
edges in the reference run, edges recovered, recall; edges the static CFG
of the same file lacks. Pass: recall 1.0 on families 1.1 and 1.3; on
1.4, every cross-process edge in the oracle present. Output: Table 2,
plus one drawn graph for voracious (the fan-out from the single dispatch
site) and one for `nanomite.c`.

**E3 Cost and instrumentation, four engines (RQ3, C1, C2).** Families
1.1 and 1.5 on QCode, icicle, Unicorn, Qiling; the same task per engine
(§4). Per engine and binary: wall with and without hooks, ratio; retired
guest instructions where the engine reports them; hook source size.
Pass: QCode's ratio at or below every baseline's on every binary, with
the baseline that cannot run a binary marked, not omitted. Output:
Table 3 (ratios), Figure 2 (wall time per engine, log scale), a short
listing of each engine's store hook side by side.

**E4 Process model (RQ4).** Family 1.4. Per binary: children, traps
serviced, cross-process edges expected and found, switches per run and
bytes copied per switch, wall with and without hooks. Pass: oracle
output and exit code reproduced for every corpus guest; voracious
reaches its exit with all 15 fan-out edges; nano.bin and ringgit run to
their verdict once the two process-model additions land. Output: Table 4.

**E5 Static consumption (C4).** Families 1.1 and 1.2. Run an existing
static pass (indirect-target resolution, or the CFG builder) on
`context.bin` and on the original file, and count what it resolves in
each. Pass: the pass resolves targets inside generated regions from
`context.bin` that it cannot see in the original file, with no code
change to the pass. Output: Table 5.

**E6 Determinism and stability (RQ5).** Every family, interpreter vs JIT,
five runs each. Per binary: graph bytes equal, node and edge deltas,
step delta. Pass: identical or the delta explained by lifted-copy
duplication only. Output: one paragraph and a footnote-sized table.

**E7 Ablation.** UPX seeds and voracious with: hooks off, provenance only,
provenance + entries, + `--edges`; the JIT warm-up threshold 1 vs 2; the
shared-memory switch copy vs a page-dirty diff once implemented. Output:
Figure 3, stacked cost per feature.

## 6. Infrastructure

- `benchmarks/unpackbench/` on `emulator-suite`, beside `hookbench`:
  `build-corpus.sh` (seeds, compressors, Tigress, fixtures, corpus
  guests; writes `corpus/manifest.json` with source, command, sha256 and
  ground-truth paths), `run-all.sh` (`ENGINES=`, `FAMILIES=`, `REPS=`),
  `truth.py` (byte and edge comparison against the manifest), `report.py`
  (Markdown tables and figures; every run appended to a progress section,
  never overwritten, as the project prefers).
- Ground-truth extraction: `dump-segments.py` for family 1.1; an
  `LD_PRELOAD`-free instrumented build flag for the Tigress seeds that
  writes the emitted buffer to a file at the generation call; the entry
  log of a reference run for edges.
- Baseline drivers under `unpackbench/engines/{qiling,unicorn,icicle}`,
  each a single file implementing the task of §4; their line counts are
  themselves a result.
- CI: a smoke subset (selfdecrypt, hello.upx, nanomite.c) on every push
  through the existing corpus and unpack suites; the full sweep is manual.
- Machine: one quiet host, CPU frequency pinned, `perf stat` for
  instruction counts where used, all versions recorded in the manifest.

## 7. Threats to validity, to be stated in the paper

- Corpus scope: Linux x86-64, static or via `ld.so`; no Windows, no
  32-bit, no threads. Single machine.
- Windows of provenance: two address ranges, 16 MiB flat-space cap; an
  allocation outside them is a warning, not a region.
- Sites are per lifted copy, not per pc; ranges are address-gap derived.
- The interpreter/JIT graph difference on UPX is a lifting-shape
  artefact, not a semantic one; it is measured, not assumed.
- Switch cost is linear in shared bytes and resident pages; acceptable
  for the corpus, not a general-purpose scheduler figure.
- Baselines are our implementations of the task on other engines;
  their source is published so the comparison can be checked.

## 8. Prerequisites and schedule

Prerequisites, in order: commit and push the 2026-09-22 tree in four
reviewable pieces (VM park/unpark and MMU fixes; single-VM scheduler;
ptrace, waitid and shared anonymous memory; fixtures and tests);
glibc clone flags and SIGFPE/SIGSEGV delivery for nano.bin and ringgit;
the `Process` accessors for allocations and syscalls (to restore the
`syscalls` section of `graph.json`); a data-region output next to code
regions, so whole binaries can be reconstructed.

| week | work | output |
|---|---|---|
| 1 | commits, CI, `build-corpus.sh`, families 1.1 and 1.3, `truth.py` | E1 on compressed seeds |
| 2 | Tigress family, instrumented seed builds, E1 complete, E6 | Table 1, determinism paragraph |
| 3 | reference runs, static CFG export, E2 | Table 2, the two drawn graphs |
| 4 | baseline drivers (icicle first, then Unicorn, then Qiling), E3 | Table 3, Figure 2 |
| 5 | nano.bin and ringgit support, corpus-guest variants, E4 | Table 4 |
| 6 | static consumer, E5, E7 ablation | Table 5, Figure 3 |
| 7 | reruns on the quiet host, figures, threats section | camera-ready numbers |

Six to seven weeks of evaluation work, one person, with weeks 4 and 5
parallelisable if a second person takes the baseline drivers.
