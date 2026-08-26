# Physical x87/MMX handoff

This document is the current handoff for x87/MMX SLEIGH, Aegis, Binit, and the
QCode concrete emulator. The physical-file transport and its initial hardware
corpus are complete. Arithmetic semantics and the first full environment-memory
restore corpus are replay-clean; remaining work is exceptional control-word
behavior, full-tag propagation, and broader physical alias coverage.

## Do not lose the model

There is one canonical **physical** x87/MMX file, R0–R7:

- SLEIGH stores R0–R7 as ten-byte slots in private `x87` RAM at offsets
  `0, 10, …, 70`.
- `FPUStatusWord.TOP` maps logical `ST(i)` to physical `R[(TOP + i) & 7]`.
- `FPUTagWord` is a physical-order architectural **full** tag word: two bits
  per R slot. It is not FXSAVE's abridged format.
- MMX `MM(i)` is the low 64-bit view of physical R(i). An MMX write sets that
  slot's high 16 bits to `0xffff`, marks all tags valid, and clears TOP.
  The TOP clearing was observed in the hardware alias corpus; do not remove it.
- `EMMS` leaves payload bytes alone and empties all physical full tags.

Do not reintroduce `ST0`–`ST7` or `MM0`–`MM7` architectural register aliases,
or rotate payloads when TOP changes.

## Physical JSON / Binit contract

The stable external state format is physical:

```text
x87_r0 … x87_r7  20 lowercase hex digits each, raw little-endian f80 bytes
x87_status        full status word; TOP is bits 11..13
x87_top           optional redundant input integer 0..7 (not recorded by Aegis)
x87_tag           FXSAVE abridged tag byte in physical R order
x87_control, x87_opcode, x87_ip, x87_dp
mm0 … mm7         optional low-64 physical R views
scratch_memory    optional 1024-digit lowercase hex string: 512 raw bytes at mem0
```

`scratch_memory` is mutually exclusive with `mem0_value` and `mem1_value`. It
is the bounded environment-memory transport; byte 0 is `mem0`, and byte 256
is the old `mem1` location.

Consumers derive a logical view themselves:

```text
ST(i) = x87_r[(x87_top + i) & 7]
```

`Aegis` accepts `x87_rN`, preserves/snapshots physical tags, and derives TOP
from `x87_status`. It accepts `x87_top` only as a validated redundant input and
does not serialize it in results. It still accepts legacy `x87_stN` input for
protocol compatibility, but it rejects mixing logical and physical payload
fields.

`Binit` is stricter: its replay boundary accepts only `x87_rN`. A legacy
hand-written fixture must be explicitly converted before insertion with:

```python
from db import logical_x87_to_physical_state
physical_state = logical_x87_to_physical_state(logical_state)
```

That helper maps payloads and the legacy logical abridged tag through TOP and
adds `x87_top`. It is an import boundary, not a Binit runtime mode.

### Important FXSAVE fact

The FXSAVE/FXRSTOR **payload slots** are logical `ST(0)`…`ST(7)` order, while
its abridged tag byte is physical R order. Aegis converts payload order at the
FXSAVE boundary in both directions while keeping `CpuState.fpu.registers`
physical internally. This previously caused the first physical replay
mismatch; do not change it back to direct physical slot copies.

## Completed implementation

### SLEIGH

`wazabin-sleigh/precompile/open_sleigh/src/x86/ia.sinc` has:

- private `x87` RAM, TOP-derived `fregop`/`stN` operands, and physical stack
  push/pop/TOP macros;
- physical full-tag helpers; non-rotating `FLD`, `FST*`, `FXCH`, `FFREE*`,
  `FINIT`/`FNINIT`, arithmetic-pop, and environment/save/restore accesses;
- MMX low-64 R views, high-16 write behavior, physical tag effects, and EMMS;
- FXSAVE/FXRSTOR abridged physical-tag conversion, logical FXSAVE payload
  ordering, and 64-bit FIP/FDP fields in the long-mode image;
- explicit zero extension of legacy 32-bit FRSTOR/FLDENV FIP/FDP fields into
  x86-64's pointer registers (without consuming adjacent FOP bytes);
- MMX operand constructors that clear TOP (`FPUStatusWord & 0xc7ff`).

Useful source regions:

| Area | Approximate location in `ia.sinc` |
| --- | --- |
| MMX physical operands | 1160 |
| TOP/full-tag/stack macros | 2500–2720 |
| environment save/restore | 6200–6500 |
| FXSAVE/FXRSTOR | 6740–6850 |

### Aegis

Relevant files:

- `aegis/libaegis/src/cpu.rs`: `FpuState` is the physical file and converts
  FXSAVE logical payload order at its boundary.
- `aegis/client/src/main.rs`: physical JSON parsing/serialization, TOP
  validation, and MMX/R overlap validation.
- `aegis/README.md`: current protocol documentation.

The client has tests for physical JSON round-tripping and rejecting conflicting
`mmN`/`x87_rN` input.

### Binit and QCode

Relevant files:

- `binit/db/__init__.py`: physical fixture conversion helper and lossless f80
  JSON parsing.
- `binit/generator/admit_x87_physical_cases.py`: idempotent insertion tool for
  focused physical alias/environment rows.
- `wazabin-qcode/sleigh/tests/binit.rs`: physical-only DB replay; it seeds and
  snapshots raw R slots and maps physical abridged tags to SLEIGH full tags.
- `wazabin-qcode/emulator/src/concrete.rs`: treats private `x87` RAM as
  zero-filled architectural state.
- `wazabin-qcode/emulator/src/concrete/float80.rs`: APFloat f80 arithmetic,
  currently fixed at nearest-ties-to-even.

## Local hardware corpus

The local PostgreSQL DB (`x86db`) currently has these physical x87/MMX rows.
Each has four hardware states with TOP 0 and TOP 3.

| ID | Instruction | Coverage |
| ---: | --- | --- |
| 11666 | `FLD ST1` | physical push/TOP remap |
| 11667 | `FST ST1` | TOP-derived x87 write |
| 11668 | `FSTP ST1` | write plus physical pop/tag |
| 11669 | `FXCH` | payload and full-tag swap |
| 11670 | `FFREE ST1` | physical full-tag clear |
| 11671 | `FNINIT` | environment reset without payload rotation |
| 11672 | `MOVQ MM0, MM1` | R-file input followed by MMX write; captures both views |
| 11673 | `FST ST2` | x87 write observed through the destination MMX view |
| 11674 | `FXSAVE [RBX]` | physical TOP/tag save header at `mem0` |
| 11675 | `MOVD MM0, EAX` | GPR-to-MMX physical R0 write |
| 11676 | `MOVD EAX, MM1` | MMX physical R1 source observation |
| 11677 | `PADDB MM0, MM1` | packed-byte R0/R1 read-modify-write |
| 11678 | `FCOM ST1` | comparison condition bits with nonzero TOP |
| 11679 | `FXAM` | classification condition bits with nonzero TOP |
| 11680 | `FST ST3` | store into initially empty physical destination |
| 11681 | `FSTP ST3` | store/pop into initially empty physical destination |
| 11682 | `FINCSTP` | defined C1 clearing |
| 11683 | `FDECSTP` | defined C1 clearing |
| 11684 | `FXCH ST1` | defined C1 clearing |
| 11685 | `FNSTENV [RBX]` | full tag word for zero/infinity at TOP 0/3 |
| 11686 | `FNSAVE [RBX]` | full tag word/save image for zero/infinity at TOP 0/3 |
| 11698 | `FXRSTOR [RBX]` | 4 images: TOP 0/3, logical f80 slots, physical abridged tags |
| 11699 | `FRSTOR [RBX]` | 4 images: TOP 0/3, explicit physical full tags, logical f80 slots |
| 11700 | `FLDENV [RBX]` | 4 images: TOP 0/3, explicit physical full tags, payload preservation |
| 11701–11706 | `FRSTOR; operation; FNSAVE` | 24 states: full-tag copy, pop, push, swap, arithmetic, and FCMOV audit |

All 12 restore states in `11698`–`11700` and all 24 full-tag propagation
states in `11701`–`11706` were captured by Aegis and replay cleanly in strict
QCode. They include +0/-0, normal, denormal, both infinities,
qNaN, an unsupported f80 encoding, and an empty physical slot; every image has
RC=up, PC=single, sticky/C1 condition state, and distinct FIP/FDP. `11698`
separates logical FXSAVE payload order from physical abridged tags. `11699` and
`11700` supply full tag pairs directly rather than deriving classes from f80
payloads.

The corpus exposed two pointer-width issues in SLEIGH: long-mode FXSAVE/
FXRSTOR FIP/FDP are 64-bit fields at +8/+16, while legacy FRSTOR/FLDENV FIP/FDP
are 32-bit fields and must be explicitly zero-extended. A bare `*:4` assignment
to a 64-bit SLEIGH register widened the *memory load* and consumed adjacent
FOP/selector bytes; use `zext(*:4 ...)` for the legacy forms.

`11678`–`11686` fixed comparison/status preservation, FXAM classification,
FST/FSTP occupancy, defined C1 clearing, and synthesized full environment
tags. `11677` exposed and fixed raw p-code lowering for bit-range writes into
private-memory loads (the MMX packed-lane form).

The arithmetic hardware corpus is IDs **11687–11697**: 346 captured and
QCode-replay-clean states covering FADD/FSUB/FMUL/FDIV, FRNDINT, FILD,
FISTP/FISTTP, f32 FLD/FSTP, and FCOM. It sweeps RC and PC where applicable,
uses TOP 0 and 3, and includes infinity/zero invalid operations, divide by
zero, qNaN comparison, f32 denormal loads, and exact f80↔f32 boundaries.

The first alias case found the MMX TOP-clearing behavior described above. This
is why physical overlap cases are valuable; retain and expand them rather than
relying only on unit tests.

`11666`–`11671` were migrated in the live local DB from old logical `x87_stN`
fields. This was a local DB data migration, not a committed SQL migration.
New data producers must emit physical fields directly.

## Immediate next-agent task

Read [`X87_NEXT_BATCH_HANDOFF.md`](X87_NEXT_BATCH_HANDOFF.md) first. It
tracks the hardware-captured MMX/x87 rows awaiting replay and defines the next
Binit/Aegis-only memory, conversion, condition, and stack-boundary batch.

The environment restore corpus is complete: `11698`–`11700`, 12 hardware
states, strict-replay-clean. Read
[`X87_ENVIRONMENT_RESTORE_HANDOFF.md`](X87_ENVIRONMENT_RESTORE_HANDOFF.md)
for its state and transport invariants before expanding this area.

Do not start another arithmetic admission batch until a follow-up design covers
exceptional precision-control or trap policy; the current arithmetic/control
word corpus is the existing 346-state baseline.

## Follow-up work

### x87 control word and exception status

The concrete emulator now has a contextual f80 path in
`emulator/src/concrete.rs`'s `StandaloneEmulator`. It recognizes contexts
with named `FPUControlWord` and `FPUStatusWord` varnodes and applies:

- `RC` to f80 add/sub/mul/div, FRNDINT, FISTP, and f80→f32/f64 stores;
- `PC` to normal finite arithmetic results by rounding the raw f80
  significand in place, preserving the extended exponent range (FILD itself
  remains an exact extended conversion);
- APFloat invalid, divide-by-zero, overflow, underflow, and inexact status to
  the matching x87 sticky bits, plus denormal-operand and C1 rounded-up state;
- x87's masked-invalid indefinite QNaN result for invalid arithmetic.

The current explicit policy is **non-trapping**: results are always produced;
unmasked exceptions additionally set ES. Architectural exception delivery is
not modelled yet, so do not add hardware corpus cases that expect a trap.
The corpus covers f32 denormal loads and the principal masked invalid/divide
cases. Additional exceptional precision-control and unmasked-trap cases remain
follow-up work.

`DomainValue` remains context-free and generic float operations still flow
through `emulator/src/lib.rs`; only f80 instructions in a context exposing the
x87 words take this concrete contextual path.

The baseline arithmetic/conversion admission is IDs 11687–11697; follow-up
exceptional precision-control and trap-policy rows should be added only after
the environment restore corpus is clean.

### Environment-memory transport

The bounded raw-byte transport is implemented as `scratch_memory`: exactly 512
bytes (1024 lowercase hex digits), rooted at `mem0`. It is carried through
`libaegis::CpuState`, the hardware harness, Aegis JSON, Binit parsing, and
QCode seed/snapshot/comparison/CSV diagnostics. It is intentionally exclusive
with the legacy `mem0_value`/`mem1_value` words. Aegis kernel images must be
rebuilt before hardware capture because `CpuState` changed.

Hardware restore rows are now `11698`–`11700`: FXRSTOR and FRSTOR use
logical-order memory payloads at TOP 0/3; FRSTOR and FLDENV carry deliberately
supplied physical full-tag words. FLDENV confirms that its environment-only
restore leaves the supplied physical payload file intact. Keep the FXSAVE rule
in mind: memory payload is logical order even though the Binit state stays
physical.

### 3. Expand physical alias cases

Continue using one-instruction rows with a canonical physical initial state
and redundant requested MMX views for observability. Good next cases include
other MMX source/destination forms, `MOVD`, arithmetic MMX writes, and `EMMS`
with a nonzero initial TOP. For x87-to-MMX observation, include the relevant
`mmN` field along with its matching `x87_rN` field in the initial state.
Aegis will reject mismatching overlap bits.

### 4. Full-tag class propagation audit

Aegis only has FXSAVE's abridged tag byte, so it cannot oracle valid/zero/
special full-tag classes. Audit SLEIGH copy/result forms when richer full-tag
input becomes available (for example via an x87 environment path). In
particular, verify class propagation for `FST ST(i)`, `FSTP ST(i)`, `FLD ST(i)`,
conditional moves, and arithmetic results. Do not regress the current
physical valid/empty behavior while doing so.

## Commands

Run the physical corpus through QCode:

```bash
cd /home/jack/dev/binary/wazabin-qcode
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
PCODE_FUZZ_TEST_CASE_IDS=11666,11667,11668,11669,11670,11671,11672,11673,11674 \
PCODE_FUZZ_THREADS=1 \
PCODE_FUZZ_OUTPUT=/tmp/x87-qcode.csv \
cargo test -p wazabin-qcode-sleigh --test binit binit_full -- --ignored --nocapture
```

Insert only missing focused physical cases:

```bash
cd /home/jack/dev/binary/binit
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
python3 generator/admit_x87_physical_cases.py --insert
```

Capture rows on hardware. Add `--ignore-completed` only when intentionally
replacing an existing result:

```bash
cd /home/jack/dev/aegis
rm -f /tmp/serial.sock /dev/shm/ivshmem
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
just run --test-case-id ID [--test-case-id ID ...]
```

Useful validation:

```bash
cd /home/jack/dev/aegis
cargo test -p client
cargo test -p libaegis --features std

cd /home/jack/dev/binary/binit
python3 -m pytest generator/test_admit_sleigh_candidates.py generator/test_admit_x87_physical_cases.py

cd /home/jack/dev/binary/wazabin-qcode
cargo test -p wazabin-qcode-sleigh --test x64
```

## Checkout warning

All four checkouts already contain unrelated/uncommitted work. In particular,
`wazabin-sleigh/precompile/open_sleigh` is a dirty nested checkout, and Binit's
generator files are untracked local work. Do not `git clean`, reset, or update
submodules. QCode builds the nested
`wazabin-sleigh/precompile/open_sleigh` source, **not**
`/home/jack/dev/binary/open_sleigh`.
