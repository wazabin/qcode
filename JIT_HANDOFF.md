# JIT and VM handoff

The JIT is correct on all 17 Embench-IoT benchmarks and worth about 1.0–4.8× on
them. The next work is coverage: it declines two thirds of real code, and where
it declines it costs more than it saves.

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

`7552962` gives up a real optimization: on a merged straight-line block, dead
flag stores were 12 of 26, and removing them took one block from 96 QCode
instructions to 40. Recovering it needs either liveness supplied to that pass
(`crate::mem::mem_liveness`) or a strictly conservative rule — remove a store
only when a later store *in the same block* covers every byte with no
intervening read, which never depends on what is live at the exit.

## Where the performance actually is

```
aha-mont64 4.81x   huffbench 1.38x   md5sum 1.31x   picojpeg 1.16x
sglib 1.13x   qrduino 1.12x   depthconv 1.11x   ...   crc32 0.95x   nsichneu 0.96x
```

Microbenchmark (`vm_integration jit_throughput`): ~43ms, ~46M guest-insn/s,
`native_bodies=3` — a million-iteration loop runs inside one entry into the JIT.
Treat that number with suspicion: it is a two-instruction loop that touches no
memory, and it predicted none of the above.

**The ceiling is memory access.** Measured over `fib-static.ins` (3,779 real
instructions from icicle's corpus, `~/dev/icicle-emu/icicle-test/tests/`):

```
64.4% of instructions compile
96% of all declines are "non-constant address" — i.e. guest RAM
```

Per *block* it is far worse (~24% on a real kernel), because one unsupported
instruction declines the whole block, and basic-block absorption made blocks
bigger. A declined block still pays a `resolve` on every execution, which is why
`crc32` and `nsichneu` are *below* 1.0×.

### The next change: inline the MMU

Follow icicle (`~/dev/icicle-emu/icicle-jit/src/translate/mem.rs`). It inlines a
software TLB and keeps the real MMU as a cold fallback:

- `TLBEntry { tag: u64, guest_to_host_offset: u64 }` — 16 bytes, so indexing is a
  shift and a mask.
- **Separate read and write tables**, so the coarse permission check *is* which
  table the entry was found in.
- Hot path: index, load tag, compare, miss branches to a `set_cold_block` that
  calls the Rust MMU; hit adds `guest_to_host_offset` and does the access.
- Plus `check_alignment` and `check_same_page` (both elided for 1-byte), and a
  per-byte `check_perm`. `tlb_lookup_const` folds the index when the address is
  constant. TLB loads are tagged `AliasRegion::Vmctx`.

After that, register caching in SSA (icicle's `TranslatorCtx::active_vars`) is
what would recover the dead-flag-store win structurally, without needing a
dead-store pass at all: values that never reach memory need no proving dead.

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
- **Check the claim in the comment.** Both the shift bug and the RAM bug were
  comments asserting a property the code did not enforce.
- **Look for the pass before writing it.** Dead-store elimination already
  existed (`analysis/src/dce/dead_load.rs`), and a redundant reimplementation was
  written and deleted before that was noticed.
