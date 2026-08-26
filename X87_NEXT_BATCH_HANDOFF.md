# x87/MMX next Binit/Aegis admission batch

## Status at handoff

The physical x87/MMX transport, environment restore corpus, and full-tag
propagation sequence corpus are implemented and strict-QCode-replay-clean
through IDs **11706**.

Recent **hardware-captured but not yet strict-QCode-replayed** Binit rows are:

| IDs | States | Coverage |
| --- | ---: | --- |
| 11707–11715 | 36 | MMX POR/PSUBW/PCMPEQB/PUNPCKLBW and x87 FCHS/FABS/FTST/FLD1/FLDZ |
| 11716–11721 | 24 | MMX PADDSB/PSUBUSW/PMULLW/PSLLW/PACKSSWB/PUNPCKHBW |
| 11722–11726 | 20 | x87 FSQRT/FXTRACT/FSCALE/FPREM/FPREM1 |

All listed states have hardware results in `x86db`. Do not edit their initial
states or overwrite their results. Their generator is
`binit/generator/admit_x87_mmx_special_cases.py`; its current Binit commits are
`61f7de5` and `3635990`.

## Immediate task

Add a **new** Binit generator (do not keep growing the broad special-case
script) and focused pytest coverage for the following hardware-only admission
batch. Build every memory image as `bytearray(512)` and serialize only raw
little-endian bits: do not use host floating-point serialization.

1. `FLD double ptr [RBX]` and `FSTP double ptr [RBX]`.
2. `FILD`/`FISTP` word and qword memory forms.
3. `FUCOM` and `FUCOMI` condition variants, including ordered, unordered, and
   nonzero-TOP cases.
4. MMX `MOVQ` qword memory load and store forms.
5. Empty-stack underflow and full-stack overflow cases.
6. Explicit +0/-0, normal, denormal, +infinity/-infinity, qNaN, unsupported
   f80 (nonzero exponent with integer bit clear), and empty-slot vectors.

Use `qcode-dump` to record exact instruction spellings/opcodes before adding a
row. For every x87 row retain canonical physical `x87_r0`…`x87_r7`, physical
abridged `x87_tag`, and TOP 0/3. Make all raw payload slots distinct. For MMX,
include matching `mmN` views whenever they make an R-file alias observable.

## Required workflow

```bash
# Review, then insert only new rows.
cd /home/jack/dev/binary/binit
python3 generator/NEW_GENERATOR.py --output /tmp/x87-next.json
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
  python3 generator/NEW_GENERATOR.py --insert

# Capture every printed ID. Aegis is the separate checkout.
cd /home/jack/dev/aegis
rm -f /tmp/serial.sock /dev/shm/ivshmem
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
  just run --test-case-id ID [--test-case-id ID ...]
```

Verify each inserted row has exactly the expected number of `test_results` and
no null final state. Commit Binit generator/tests separately. Do **not** change
QCode or SLEIGH during this admission phase.

Only after this batch is captured should a later agent run strict QCode replay
for IDs **11707 onward**, fixing only hardware-confirmed mismatches in small
SLEIGH/QCode commits.

## Invariants

- `scratch_memory` is exactly 512 bytes / 1024 lowercase hex digits at
  `0x6666_6601_0100`, exclusive with `mem0_value`/`mem1_value`.
- Binit state is physical; never use `x87_stN` fields.
- `ST(i) = R[(TOP+i)&7]`; never rotate JSON payload fields.
- FXSAVE payload order is logical while its abridged tag is physical.
- Do not reset/clean the dirty unrelated checkouts or blindly update the
  OpenSleigh submodule.
