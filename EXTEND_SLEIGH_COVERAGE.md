# Extend x86-64 SLEIGH coverage

This document is the operating guide for extending the live Binit/Aegis corpus
and measuring `wazabin-sleigh` constructor coverage. Work spans three nearby
checkouts:

| Checkout | Purpose |
| --- | --- |
| `/home/jack/dev/binary/wazabin-qcode` | SLEIGH coverage collector and QCode differential replay |
| `/home/jack/dev/binary/binit` | Binit generator, candidate-admission tool, PostgreSQL schema |
| `/home/jack/dev/aegis` | hardware oracle, client, and guest kernel |

`binit/aegis` is an uninitialized gitlink. Use `/home/jack/dev/aegis`.

## Current baseline

At the latest successful run:

```text
x86-64 constructors: 747/5719 (13.06%)
11604 of 11604 Binit cases decoded
instruction table: 649/4574 (14.19%)
```

The original live baseline was 685/5719. The report after adding string
operations is:

```text
/tmp/sleigh-constructor-coverage-after-strings.json
```

Earlier reports and candidate artifacts are also in `/tmp`:

```text
/tmp/sleigh-constructor-candidates.json
/tmp/sleigh-constructor-coverage-after-*.json
/tmp/sleigh-admission-*.json
```

The mutator artifact has 536 candidates that originally covered 605 missing
constructor keys. It is discovery input, not a trusted corpus.

## Measure coverage

The metric counts every recursive constructor match selected by every Binit
opcode, divided by every constructor in every public x86-64 SLEIGH table. It
includes operand-table and delay-slot matches, not just the root instruction
constructor.

```bash
cd /home/jack/dev/binary/wazabin-qcode
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
PCODE_SLEIGH_COVERAGE_OUTPUT=/tmp/sleigh-constructor-coverage.json \
cargo test -p wazabin-qcode-sleigh --test binit \
  binit_constructor_coverage -- --ignored --nocapture
```

Use the JSON witness keys `(table,index)` to measure deltas. Prefer candidates
that cover a new `instruction` constructor, then rare operand-table entries.

To discover more candidates:

```bash
cd /home/jack/dev/binary/wazabin-qcode
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
PCODE_SLEIGH_CANDIDATE_BUDGET=100000 \
PCODE_SLEIGH_CANDIDATES_OUTPUT=/tmp/sleigh-constructor-candidates.json \
cargo test -p wazabin-qcode-sleigh --test binit \
  binit_constructor_candidates -- --ignored --nocapture
```

The mutation seed is deterministic. Encodings are canonicalized to decoded
instruction bytes and a candidate is retained only for a constructor absent
from the corpus at the beginning of the run.

## Required admission gate

For every admission:

1. Classify the candidate. Reject privileged, host-I/O, uncontrolled loop,
   unmodelled segment, and unsafe-memory instructions.
2. Generate architecture-aware initial states; never insert `[{}]`.
3. Insert only selected cases into PostgreSQL.
4. Run Aegis only for those IDs and confirm every state has a result.
5. Run targeted QCode replay and fix or record every lift/emulation failure.
6. Re-run constructor coverage and save a before/after report.

Aegis result rows are the hardware oracle. A constructor witness without them
is not a regression case.

### Candidate-admission tool

`/home/jack/dev/binary/binit/generator/admit_sleigh_candidates.py` reads the
mutator JSON, emits a reviewable plan, and only inserts when `--insert` is
explicit. It accepts a deliberately narrow GPR and simple-memory subset,
reuses Binit's state domains, and rejects unsupported implicit memory,
privileged state, vector state, segments, stack/control flow, and unsafe
addressing.

Example:

```bash
cd /home/jack/dev/binary/binit
/home/jack/dev/binit/.venv/bin/python \
  generator/admit_sleigh_candidates.py \
  /tmp/sleigh-constructor-candidates.json \
  --opcode 4380f208 \
  --output /tmp/sleigh-admission.json \
  --insert
```

The output plan records the opcode, decoded text, constructor path, decision,
reason, and generated initial states. The tool and its tests are currently
untracked changes in the Binit checkout; preserve them before changing
branches or cleaning the tree.

## Run only selected Aegis cases

Aegis now accepts repeatable `--test-case-id` options. The `just run` wrapper
starts the guest, waits two seconds for UART initialization, then starts the
client. It also conditionally enables AVX-512 XCR0 state: this host lacks
AVX-512, so unconditionally setting ZMM XCR0 bits raises `#GP` before tests
begin.

```bash
cd /home/jack/dev/aegis
rm -f /tmp/serial.sock /dev/shm/ivshmem
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
just run --test-case-id 11585 --test-case-id 11595
```

Use a shell timeout during investigation, for example
`timeout --foreground 90s ...`. Do not run the entire corpus merely to execute
new highest-ID cases.

Check completion directly:

```sql
SELECT test_case_id, count(*)
FROM test_results
WHERE test_case_id = ANY (ARRAY[11585, 11595])
GROUP BY test_case_id;
```

The live string inserts have non-contiguous IDs because a failed transaction
consumed sequence values. Select by actual IDs/opcodes, not an assumed range.

## Two-word memory model and string operations

The old corpus used one eight-byte modeled word:

```text
mem0_value @ 0x666666010100
```

String support adds:

```text
mem1_value @ 0x666666010200
```

Both addresses are in the existing mapped page. `mem0_value` is the string
source and `mem1_value` its destination. The generator fixes `RSI` and `RDI`
to those addresses, uses source/destination values `{0, 1, u64::MAX}`, and
uses both DF=0 and DF=1. This covers non-REP `MOVS`, `CMPS`, `LODS`, `SCAS`,
and `STOS` byte/word/dword/qword forms. Port-string `INS`/`OUTS` remain
excluded because they need an I/O-port model.

Relevant implementation files:

- `binit/generator/make_test_cases.py`
  - state policy, two words, string encodings, and generator inclusion;
  - Keystone cannot reliably encode zero-operand string mnemonics, so the
    architectural bytes are explicitly mapped (`MOVSB=a4`, `CMPSQ=48a7`, etc.).
- `aegis/libaegis/src/cpu.rs`
  - `CpuState.mem0` and `CpuState.mem1`.
- `aegis/aegis/src/testing/harness.rs`
  - maps, seeds, and snapshots both words.
- `aegis/client/src/main.rs`
  - serializes `mem1_value` to/from PostgreSQL.
- `wazabin-qcode/sleigh/tests/binit.rs`
  - seeds, snapshots, compares both words; includes DF; supports
    `PCODE_FUZZ_TEST_CASE_IDS` for targeted replay.

Generator checks:

```bash
cd /home/jack/dev/binary/binit
/home/jack/dev/binit/.venv/bin/python -m pytest \
  generator/test_string_memory.py generator/test_admit_sleigh_candidates.py -q
```

## Targeted QCode replay

The replay runner has an optional comma-separated case filter. It avoids
replaying the full database for each new encoding:

```bash
cd /home/jack/dev/binary/wazabin-qcode
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
PCODE_FUZZ_TEST_CASE_IDS=11585 \
PCODE_FUZZ_THREADS=1 \
PCODE_FUZZ_OUTPUT=/tmp/wazabin-string-mismatches.csv \
PCODE_FUZZ_FAIL_QUIETLY=1 \
cargo test -p wazabin-qcode-sleigh --test binit \
  binit_full -- --ignored --nocapture
```

`PCODE_FUZZ_FAIL_QUIETLY` is for investigation only. Remove it for a completed
regression gate.

### Known blocker: MOVSB lift

A targeted `MOVSB` replay currently reaches Aegis and has all 18 hardware
results, but QCode lifting fails before emulation:

```text
lift failed: p-code expression has no known byte width
```

The recorded CSV is currently:

```text
/home/jack/dev/binary/wazabin-qcode/sleigh/candidates/wazabin_mismatches.csv
```

Do not mark string admissions as QCode-validated until this is fixed. Start by
decoding `a4`, inspecting its flat p-code operands, and finding the expression
whose width is absent in the SLEIGH-to-QCode lifter. Preserve the two-word
memory comparison while fixing it; reducing the test to one word hides the
actual `MOVS` behavior.

## Useful current commands

Build/check the touched code:

```bash
cd /home/jack/dev/aegis
cargo fmt --all -- --check
cargo check -p client --release
(cd aegis && cargo check --release)

cd /home/jack/dev/binary/wazabin-qcode
cargo fmt --check
cargo test -p wazabin-qcode-sleigh --test binit --no-run
```

## Next work order

1. Fix the width-less p-code expression exposed by `MOVSB`.
2. Run targeted QCode replay for every admitted string form; fix semantics or
   report unsupported p-code separately from hardware results.
3. Add non-REP string forms through the normal generator after preserving the
   current live database or creating a migration/import path. The generator's
   current `main()` deliberately clears the database.
4. Continue candidate batches with safe new `instruction` constructors and
   use Aegis partial mode plus targeted QCode replay after each batch.
5. Add REP/REPE/REPNE cases only with a bounded RCX policy and explicit loop
   semantics. Do not admit arbitrary mutator REP encodings.
