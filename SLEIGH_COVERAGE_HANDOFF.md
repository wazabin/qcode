# SLEIGH coverage handoff

## Current state

Live x86-64 SLEIGH constructor coverage is **802/5719 (14.02%)**:

```text
instruction table: 684/4574 (14.95%)
11,656 of 11,656 Binit cases decoded
```

The latest report is:

```text
/tmp/sleigh-constructor-coverage-after-mmx-batch-1.json
```

The prior batch-5 report is retained at:

```text
/tmp/sleigh-constructor-coverage-after-admission-batch-5.json
```

This session added 42 database cases in five safe, state-policy-reviewed
batches. Every new case has complete Aegis hardware results (17,755 new state
results total). **41/42 are also QCode replay-clean.** Do not describe the one
remaining case as QCode-validated; see `BSF blocker` below.

## Completed work

### String operations

The original `MOVSB` p-code lift error (`p-code expression has no known byte
width`) is fixed. All admitted non-REP string forms replay cleanly against
Aegis:

- `MOVS`, `CMPS`, `LODS`, `SCAS`, `STOS`;
- byte, word, dword, and qword forms;
- 1,440 targeted replay states across 20 test cases.

Successful strict replay output:

```text
/tmp/wazabin-string-qcode-validated.csv
```

`wazabin-qcode/sleigh/src/lib.rs` has a regression test covering all 20
non-REP string encodings.

### Width constraint propagation

The p-code lowerer now performs a local-width constraint prepass before it
allocates unique varnodes. It resolves widths supplied only by later consumers
(such as `word << v`, `zext(CF)`, and x86-64 `imm32` register writes). It also
propagates a carry-family operand width to nested extensions and treats integer
literals as width-polymorphic at their consuming operation.

Implementation:

```text
/home/jack/dev/binary/wazabin-pcode/src/instruction.rs
```

This fixed QCode lift failures for the admitted `SHLD`, `MOV imm32`, `SBB`,
and `ADC` cases.

### Admissions

The Binit admission tool was run only on its narrow safe subset. The resulting
plans are in `/tmp`:

| Batch | IDs | Cases | Targeted QCode result |
| --- | --- | --- | --- |
| 1 | 11614–11646 | 33 | 32 clean; `11615` pending |
| 2 | 11647–11650 | 4 | clean |
| 3 | 11651 | 1 | clean |
| 4 | 11652–11653 | 2 | clean |
| 5 | 11654–11655 | 2 | clean |

The admissions include integer/GPR, simple modeled-memory, `PAUSE`,
`PREFETCH*`, and `BTS` forms. Hardware completion was checked by comparing
`jsonb_array_length(test_cases.initial_states)` with `count(test_results)` for
each selected ID.

Relevant artifacts:

```text
/tmp/sleigh-constructor-candidates-batch-{2,3,4,5}.json
/tmp/sleigh-admission-batch-{2,3,4,5}-inserted.json
/tmp/wazabin-admission-batch-{1,2,3,4,5}.csv
/tmp/wazabin-admission-batch-1-validated.csv
```

## MMX and control-flow policy implementation

The agreed phase-one policy is now implemented:

- Aegis restores the legacy x87/MMX/SSE image with `FXRSTOR64` before every
  one-instruction test and captures it with `FXSAVE64` on the exception path.
  `CpuState.mmx` is seeded into, and extracted from, the shared x87 physical
  register-file slots. This bounds each MMX test without admitting x87 stack
  instructions. x87 logical stack/control/tag comparison is still deferred.
- The admission tool permits a narrow, explicit MMX mnemonic allowlist with
  GPR/simple-memory operands. It seeds `mm0`–`mm7`; `EMMS` uses three bounded
  whole-file states rather than a Cartesian product.
- QCode replay now seeds, snapshots, and compares `mm0`–`mm7`.
- The admission tool permits only direct `JMP` and `Jcc` encodings whose
  non-negative relative offset is within the fire page. It records final `RIP`;
  QCode stops on and compares the same successor block. Indirect flow, calls,
  returns, loops, REP, and stack-mutating transfers remain rejected.

New admissions:

| IDs | Cases | Aegis | QCode |
| --- | --- | --- | --- |
| 11656–11658 | `MOVQ`, `EMMS`, `PXOR` MMX smoke | complete (1,803 states) | clean |
| 11659–11660 | direct `JMP 0`, `JZ 0` | complete (9 states) | clean |
| 11661–11665 | `PUNPCKLBW`, `PSADBW`, `PSUBD`, `PADDUSW`, `PAVGB` | complete (1,890 states) | `PUNPCKLBW`/`PSUBD` clean; three blockers below |

Do not call all of batch 11661–11665 QCode-validated. Case 11662 requires
user-op `psadbw`, 11664 requires `paddusw`, and 11665 still has a width-less
p-code expression. Their recorded output is:

```text
/tmp/wazabin-mmx-batch-1.csv
```

Aegis changes are in:

```text
/home/jack/dev/aegis/libaegis/src/cpu.rs
/home/jack/dev/aegis/aegis/src/testing/harness.rs
/home/jack/dev/aegis/aegis/src/kernel/interrupts.rs
```

## BSF blocker

Case **11615** is:

```text
BSF R8D,dword ptr [R8 + R9*1 + -32]
opcode: 470fbc4408e0
```

It has all 30 Aegis results, but QCode emulation fails at the zero-memory state:

```text
emulator error: value 0 is too large to represent
store(register:4, i32 R8D <- i32 %tmp11)
```

Its SLEIGH p-code initializes a unique temporary `v0`, loops over it, then
uses `v0` after the loop. `FlatEmitter` currently replaces a repeated unique
varnode offset with its latest SSA value globally. On the zero-input branch,
the loop-body definition was not executed, so the final register store reads
an unavailable value. A real fix needs loop-carried/merge (`phi`) handling for
reassigned SLEIGH unique temporaries. This is an architectural decision; do
not silently classify it as a successful QCode regression.

## Remaining policy boundary: x87

x87 remains deferred. Before admitting `FADD`, `FLD`, `FIST*`, or any `st(i)`
form, decide the x87 numeric domain/oracle: NaNs, infinities, denormals,
rounding, exceptions, raw 80-bit versus normalized values, and status/tag
words. `CpuState` does not yet serialize the logical x87 stack or its
control/status/tag state. Do not broaden the MMX admission policy to x87 until
that model is specified.

The admission tool continues to reject indirect branches/calls, `CALL`, `RET`,
far transfers, `IRET`, `LOOP`, `JCXZ` variants, REP forms, and self-loops.

## Commands

### Discover and plan a safe batch

```bash
cd /home/jack/dev/binary/wazabin-qcode
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
PCODE_SLEIGH_CANDIDATE_BUDGET=100000 \
PCODE_SLEIGH_CANDIDATES_OUTPUT=/tmp/sleigh-constructor-candidates-next.json \
cargo test -p wazabin-qcode-sleigh --test binit \
  binit_constructor_candidates -- --ignored --nocapture

cd /home/jack/dev/binary/binit
/home/jack/dev/binit/.venv/bin/python generator/admit_sleigh_candidates.py \
  /tmp/sleigh-constructor-candidates-next.json \
  --output /tmp/sleigh-admission-next.json
```

Insert only after reviewing the plan:

```bash
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
/home/jack/dev/binit/.venv/bin/python generator/admit_sleigh_candidates.py \
  /tmp/sleigh-constructor-candidates-next.json \
  --output /tmp/sleigh-admission-next-inserted.json --insert
```

### Aegis and QCode gate

Use only newly inserted IDs; do not run the entire corpus merely to execute
highest IDs:

```bash
cd /home/jack/dev/aegis
rm -f /tmp/serial.sock /dev/shm/ivshmem
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
timeout --foreground 180s just run --test-case-id ID [--test-case-id ID ...]

cd /home/jack/dev/binary/wazabin-qcode
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
PCODE_FUZZ_TEST_CASE_IDS=ID,ID \
PCODE_FUZZ_THREADS=1 \
PCODE_FUZZ_OUTPUT=/tmp/wazabin-admission-next.csv \
cargo test -p wazabin-qcode-sleigh --test binit \
  binit_full -- --ignored --nocapture
```

`PCODE_FUZZ_FAIL_QUIETLY=1` is investigation-only. Remove it for a completed
regression gate.

### Measure coverage

```bash
cd /home/jack/dev/binary/wazabin-qcode
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
PCODE_SLEIGH_COVERAGE_OUTPUT=/tmp/sleigh-constructor-coverage-next.json \
cargo test -p wazabin-qcode-sleigh --test binit \
  binit_constructor_coverage -- --ignored --nocapture
```

## Validation completed this session

```bash
cd /home/jack/dev/binary/wazabin-pcode
cargo fmt --check
cargo test -p wazabin-pcode

cd /home/jack/dev/binary/wazabin-qcode
cargo fmt --check
cargo test -p wazabin-qcode-sleigh
```

Both passed before the MMX/control-flow work. The current strict constructor
report is `/tmp/sleigh-constructor-coverage-after-mmx-batch-1.json`.

## Working-tree warning

`/home/jack/dev/binary/wazabin-pcode/src/instruction.rs` is currently an
untracked file in a checkout with other pre-existing untracked p-code files.
Do not clean that tree or switch branches without preserving it. The Binit
admission tool/tests and Aegis targeted-run/two-word changes are likewise
pre-existing uncommitted work in their respective checkouts.
