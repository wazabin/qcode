# JIT and VM handoff

The JIT is correct on all 17 Embench-IoT benchmarks and worth about 1.3–31× on
them. It compiles 99% of the blocks a real program lifts. What is left is
throughput inside the code it already takes, not coverage.

## Start here: run the divergence harness on every program

```sh
git clone --depth 1 https://github.com/embench/embench-iot.git
EMBENCH=/path/to/embench-iot ./benchmarks/embench/build.sh

cargo test --release -p qcode_jit --test divergence   -- --ignored --nocapture
cargo test --release -p qcode_jit --test embench      -- --ignored --nocapture
```

Both must be green before touching performance. `divergence` is the stronger
check and the one to run first after *any* change to the JIT, the lifter, or the
VM's optimizer:

- `embench` runs each benchmark twice and asks it to verify its own result. That
  is one number at the end of tens of millions of operations.
- `divergence` records the block and a digest of the guest register file at
  **every block entry**, interpreted and jitted, and names the first entry where
  they disagree — plus the block that produced the bad value, and which register
  bytes differ. `B=<name>` narrows it to one program; `BUDGET=<n>` shortens a run.

It is deliberately unchained (`run_block(..., false)`), so it isolates compiled
code from the decision to stay in it. Current state: all 17 agree, 351k–1.79M
block entries each.

`guest_memory` is the unit-level companion, and the one to extend when touching
the RAM path: it runs each block *twice*, because which of the two memory paths
executes depends on history — a page is only reachable inline once an access to
it has been served the slow way.

Every correctness bug in this area was found by one of these two and would not
have been found by the other. Three times, a green result turned out to be the
harness lying — see "Lessons" below.

## What was fixed, and what it cost

| | |
|---|---|
| `cb88060` | The JIT compiled **guest RAM** at a constant address as flat storage. A RIP-relative operand resolves to a perfectly constant address and is still RAM, so it read a fabricated zero-filled space instead of guest memory. `VmMemory::is_flat` already existed for this; the JIT never called it. Fixed 7 of 9 wrong benchmarks. |
| `7552962` | `remove_dead_load_insns_block` treats nothing as live at a block's exit — sound for the pipeline that hands it an alias result, false for the VM. It dropped stores the next block reads. Fixed the last 2. |
| `276e767` | `forward_temp_stores` applied rewrites in program order, so a forwarded value that was itself a load being removed left a reference to a deleted instruction. This was the `dead or unknown id` panic. Pre-existing. |
| `361fc76` | An over-wide shift. QCode follows p-code (result is 0, or the sign); Cranelift *masks* the amount. The old comment claimed both behaved alike. |
| `3d20d27` | The VM's lifter now lowers guest `call`/`ret` as jumps (`with_flat_control_flow`). This is what made Embench run at all — see below. |
| `33fe87a` | Guest RAM is compiled, through an inlined software TLB and an inlined per-byte permission check. Block coverage went from ~24% to 99.1% and every benchmark crossed 1.0×. |

### Dead stores: use the pass that exists, and expect ~9%

`qcode_analysis::remove_dead_stores_with_liveness(ctx, func)` runs the real
thing — the alias oracle (`AliasAnalysis`) and the memory liveness
(`compute_memory_liveness`) that `remove_dead_load_insns` needs to be sound.
`remove_dead_load_insns` still accepts `aliases: None`, and with `None` its scan
treats nothing as live at a block's exit, which is exactly the unsoundness
`7552962` removed; the new entry point exists so that choice cannot be made by
accident.

Measured on the code each benchmark discovers (`DEAD_STORES=1` on the
`run-embench` example):

```
nettle-sha256  38758 -> 35419 insns  ( 8.6%)  438ms
nsichneu       20086 -> 18653 insns  ( 7.1%)  510ms
qrduino        47891 -> 43199 insns  ( 9.8%)  231ms
matmult-int     2297 ->  2029 insns  (11.7%)    4ms
```

Two things follow. The prize is **~9% of instructions, not the ~2x** the "12 of
26 flag stores" note below implies — that anecdote was one hand-picked block,
not a program. And the cost is *more than the whole run* on the larger
programs, because the VM lifts with flat control flow, so the "function" the
alias analysis must chew through is the entire discovered program. This is a
one-shot "the program is now known, optimise it properly" step for a
long-running workload, never a per-lift one.

### The conservative dead-store rule does not work as stated

This handoff used to suggest recovering `7552962` with "a strictly conservative
rule — remove a store only when a later store *in the same block* covers every
byte with no intervening read". That was tried and **reverted**; it makes
`qrduino` diverge (`B=qrduino` names the block), and the interpreter alone
retires 13% fewer operations while still finishing, so the pass really does find
the work — it just is not sound in that form. Two holes, both found the hard way:

1. **Slots are keyed by start address, and x86 registers nest.** `AH`, `AX`,
   `EAX` and `RAX` are the same bytes filed under three different keys, so a
   read of one does not mark the others as read. Any such rule has to invalidate
   by *byte range*, not by slot key.
2. **Only `Load` counts as a read.** A user p-code op reads registers without
   ever being a `Load`, so a side-effecting instruction between the two stores
   is an invisible reader. Anything not understood has to count as a reader of
   every flat space.

Fixing (1) alone was not enough. Do not extend the rule with a third special
case — use `remove_dead_stores_with_liveness` above, which is what the project
already had. Writing a new pass instead of looking for the existing one is the
same mistake recorded in "Lessons" below, made twice.

`7552962` gives up a real optimization: on a merged straight-line block, dead
flag stores were 12 of 26, and removing them took one block from 96 QCode
instructions to 40. Recovering it needs either liveness supplied to that pass
(`crate::mem::mem_liveness`) or a strictly conservative rule — remove a store
only when a later store *in the same block* covers every byte with no
intervening read, which never depends on what is live at the exit.

## Two throughput numbers, and which one you mean

Quoting one figure for "how fast is it" hides the thing that matters most here,
because Embench at its stock scale factor executes only ~3M guest instructions
per program — less work than it takes to translate it.

* **Warm-up** (stock corpus, `target/embench`): ~18M guest-insn/s aggregate.
  Dominated by one-time cost — SLEIGH decoding, the block cleanup, and Cranelift
  compiling blocks that then run for a few milliseconds.
* **Steady state** (`benchmarks/embench/build.sh` with `GLOBAL_SCALE_FACTOR=20`,
  run with `EMBENCH_DIR=target/embench-x20`): **55M guest-insn/s aggregate,
  median 63M/s**, ranging 13–152M/s, all 17 verifying. This is the rate compiled
  code actually sustains once translation is amortised.

  Measure it on *CPU* time (`/usr/bin/time -f %U`), not wall time: the emulator
  is single-threaded, so CPU time stays honest on a machine that is busy with
  something else. Guest-instruction counts at this scale are derived as
  `steps ÷ p-code-per-instruction`, the ratio taken from the stock corpus; that
  was checked against an exact count for `statemate` and came within 0.9%.

Neither is wrong; they answer different questions. Optimise against the one that
matches the workload — a fuzzing harness that re-lifts constantly lives in the
first, a long-running emulation in the second.

**`nsichneu` is the outlier worth chasing**: 13M/s even at steady state, five
times worse than anything else. Its guest basic blocks average ~1.8 guest
instructions, so it leaves and re-enters compiled code constantly and pays
per-block dispatch rather than compilation. Everything else clears 34M/s.

A caution when reading `stats.steps` rates: `nettle-sha256` retires 2.7 *billion*
p-code operations per second, which is not one interpreted operation per
0.4 cycles — it is Cranelift deleting most of them. x86 lifting emits ~20 p-code
ops per guest instruction, mostly flag computation nothing reads, and the
backend's own DCE removes them. That is also why shrinking QCode helps
*compile* time far more than it helps execution.

## Where the performance actually is

```
xgboost 31.5x   md5sum 25.9x   nettle-aes 23.1x   huffbench 19.0x   statemate 14.1x
qrduino 9.1x   edn 7.0x   nettle-sha256 5.8x   picojpeg 5.5x   aha-mont64 4.8x
sglib 4.5x   nsichneu 2.8x   depthconv 2.5x   crc32 1.9x   ud 1.8x
tarfind 1.6x   matmult-int 1.3x
```

Nothing is below 1.0× any more. The two that used to be — `crc32` at 0.95× and
`nsichneu` at 0.96× — were paying a `resolve` per execution for blocks that then
ran on the interpreter anyway; they are now 1.9× and 2.8×.

Microbenchmark (`vm_integration jit_throughput`): a two-instruction loop that
touches no memory. It predicted none of the above and still does not; the
benchmark that means something is `embench`, which reports these numbers itself.

### How guest RAM is compiled

Following icicle (`~/dev/icicle-emu/icicle-jit/src/translate/mem.rs`), with one
deliberate departure. The `Mmu` stays the only implementation of what an access
*means*; what is inlined is its answer.

- `vm/src/tlb.rs` — a 64-entry direct-mapped table of
  `TlbEntry { tag, guest_to_host_offset }`, 16 bytes so indexing is a shift and
  a mask. An entry says only *where a resident page lives*; it says nothing
  about permissions.
- A page is now one allocation, `PageData { data, perm }`, `#[repr(C)]`, so the
  permission byte for a data byte is a fixed `PAGE_PERM_OFFSET` away. That
  layout is an interface — `mmu::tests::the_permission_array_follows_the_data_array`
  is what holds it.
- The hot path (`compile::inline_access`): same-page check, index, tag compare,
  add, then a per-byte permission test that loads `size` permission bytes as one
  integer and checks `required & !held == 0`. Anything it cannot settle branches
  to a `set_cold_block` calling `vm/src/jit_abi.rs`, which serves the access
  through the real `Mmu` and caches the page on the way out.

**The departure from icicle: one table, not separate read and write ones.**
Separate tables let the table itself carry the coarse permission. Here
permissions are per *byte*, so the byte scan happens either way and a second
table would only be a second thing to fill and invalidate.

Three things hold this together, and each is a way to get it silently wrong:

1. **Invalidation is blunt on purpose.** Every operation that can add, drop or
   replace a page flushes the whole table (`map`, `unmap`, `protect`,
   `write_unchecked`, `restore`), and a clone starts empty. An entry is a raw
   pointer into a page allocation; a mis-scoped invalidation is a use-after-free
   in compiled code, not a stale read.
2. **The dynamic checks are opted out of, not approximated.** `check_uninit` and
   `watchpoints_armed` both turn a *set* bit into a fault, which the inline
   "these bits are all present" test cannot express. So `cache_translation`
   refuses to cache anything while either is on, and both setters flush. That is
   why they are behind setters now rather than public fields.
3. **An inline store sets `INIT`.** It is bookkeeping nobody reads on a normal
   run, which is exactly why omitting it would go unnoticed until a
   `check_uninit` run reported bytes the guest had plainly written.

A faulting access stops the block where it happened and returns `BLOCK_FAULT`;
the fault itself is left on the `VmMemory` for the VM to turn into a
`VmExit::Fault`, exactly as an interpreted one is. The stores that already ran
stay applied — which is what the interpreter leaves too, and the reason guest
RAM and the flat spaces are compiled *without* alias regions: Cranelift must not
reorder one store past another, or that prefix stops being a prefix.

## What is next

Coverage is no longer the lever. Of 3,733 blocks lifted across Embench, 34
decline: 33 for a 16-byte operand width and one for integer division.

- **Register caching in SSA** (icicle's `TranslatorCtx::active_vars`). Guest
  state still round-trips through memory between every instruction in a block.
  This is also what would recover the dead-flag-store optimization `7552962`
  gave up, structurally: a value that never reaches memory needs no proving dead.
- **128-bit operand widths**, which is the whole remaining decline list and
  overlaps with lifting SSE.
- The TLB is direct-mapped with 64 entries and no measurement behind that number.
  Before tuning it, count misses — `jit_abi` is the one place they pass through.

## Other things worth knowing

**Why the lifter lowers calls as jumps.** A guest `call` is a push and a jump; a
`ret` is a pop and an indirect jump. SLEIGH already emits both stack halves as
ordinary p-code, so the QCode `Call`/`Return` mnemonics contributed only
*function structure* — a callee `FunctionId` entered through its root block. An
emulator has no use for that, and it is unsound for on-demand discovery: guest
code branches across function boundaries freely, and a block belongs to exactly
one function's arena. Before `3d20d27`, every benchmark died at
`initialise_board` with "function FunctionId(1) has no root block". The
decompiler keeps real calls; `with_flat_control_flow` is opt-in.

If per-function lifting is ever wanted, `Context::rehome_owned_blocks` is the
machinery, and its doc states the precondition (strip cross-function edges and
rewrite foreign terminator targets to `TailCall`s first).

**Basic-block absorption.** `Vm::discover` folds a discovered straight-line run
into one block (`absorb_straight_line`), so a compiled unit is a guest basic
block rather than a guest instruction. Three invariants hold it together, each
learned from a bug:

1. Only a successor whose code is *already lifted* may be absorbed — an empty
   successor is an unlifted placeholder, and absorbing one drops the terminator
   that reaches it (`70b7d34`).
2. Only *into* a block that starts at a machine address — a split re-lifts both
   halves, and a p-code-internal block has no address to lift from (`c2fa7d9`).
3. Every address absorbed must be repointed in the index, not just the one being
   discovered — a run absorbs a chain (`6d718fe`).

A later branch into the middle of an absorbed run splits it by *invalidating*
both halves (`Context::split_block_at_address`) and letting them be lifted
again. It does not move instructions: the blocks are optimized in place, so
after forwarding and DCE no instruction "is" the start of a given address any
more.

## Known-broken, untouched

- **`concrete::tests::x87_context_uses_rounding_control_and_makes_apfloat_exceptions_sticky`**
  fails, and failed before any of this work. Unrelated.
- **SSE is not lifted** — `invalid bit range [0, 32] for 128-bit storage`. This is
  why `benchmarks/embench/build.sh` passes `-mno-sse -mno-sse2 -mno-mmx
  -fno-tree-vectorize`, and why 2 of the 19 benchmarks are skipped (`wikisort`
  returns a float in an SSE register; `slre` wants glibc's ctype tables).
  Lifting SSE would let Embench build at stock settings.
- **`fib-static` crashes the interpreter** when run as a program (as opposed to a
  decode corpus): a dead arena id while *formatting* an emulator error. Present
  at the original baseline. Formatting a diagnostic should not be able to panic.

## Lessons that cost time

- **A benchmark that only compares two implementations passes when both are
  wrong.** `the_jit_does_not_change_what_a_program_computes` was green while the
  countdown loop returned `ECX=999999`. It now asserts stated answers (`30cb084`).
- **A run that stops early leaves noise in the result register.** The first
  Embench harness reported all 17 as passing because `EAX == 0` happened to hold
  after an early fault. It now requires the run to have reached the sentinel.
- **Report which side is wrong.** For a while the harness printed the
  interpreted exit on every failure, so `qrduino` looked like an interpreter bug
  when the interpreter was fine (`d516d80`). All 9 real failures turned out to be
  the JIT.
- **A benchmark script that ignores exit status reports crashes as records.**
  A run that panics exits early, and `sglib-combined` duly "improved" from 230ms
  to 40ms. This is the same lesson as the Embench sentinel below, relearned
  through a shell script. Check the exit status *and* grep for `panic`.
- **Take min-of-three.** This machine varies by ±20% run to run, which is wider
  than most of the wins here. Two optimizations that looked good on one reading
  (`rposition` for terminator removal, reusing Cranelift's contexts) measured
  *worse* when A/B'd properly, and were dropped.
- **The two harnesses really do catch different things.** Caching a branch
  target with the compiled code passed `divergence` and hung `embench`:
  divergence runs unchained, so it cannot see a chaining bug by construction.
  A cached target is also wrong on its own terms — discovery deletes blocks.
- **Run the JIT's tests in debug too.** Cranelift's FunctionBuilder checks "you
  have to fill your block before switching" only under `debug_assertions`. The
  fault epilogue was built by switching away from a half-emitted block, and
  `--release` — which is how every harness here is run — passed all 17 programs
  on it. Only `cargo test` without `--release` said anything.
- **Check the claim in the comment.** Both the shift bug and the RAM bug were
  comments asserting a property the code did not enforce.
- **Look for the pass before writing it.** Dead-store elimination already
  existed (`analysis/src/dce/dead_load.rs`), and a redundant reimplementation was
  written and deleted before that was noticed.
