# x87 environment restore handoff

## Goal

The next task is to add a **hardware-captured, QCode-replay-clean restore
corpus** for x87 environment images. The physical x87/MMX transport, bounded
raw scratch-memory transport, and arithmetic corpus are already in place; do
not redesign them.

The immediate deliverable is a Binit generator analogous to
`binit/generator/admit_x87_arithmetic_cases.py`, followed by Aegis capture and
strict QCode replay for:

1. `FXRSTOR [RBX]` (`0f ae 0b`), including the FXSAVE header and all eight
   16-byte ST/MM payload slots;
2. `FRSTOR [RBX]` (`dd 23`), including its full 16-bit tag word and logical
   f80 payload order;
3. `FLDENV [RBX]` (`d9 23`), including control/status/TOP/full-tag fields.

Use the exact decoded spellings from `qcode-dump` before inserting a row. The
opcode table above is for the x86-64 RBX addressing form.

## Non-negotiable state model

- Canonical Binit/Aegis state is physical `x87_r0`…`x87_r7`; each is a raw,
  20-digit, little-endian f80 string.
- `ST(i)` is physical `R[(TOP + i) & 7]`; do **not** rotate the JSON payload
  fields when TOP changes.
- `x87_tag` in Binit/Aegis JSON is FXSAVE's **physical-order abridged byte**:
  set bit = nonempty physical slot. QCode translates it to/from SLEIGH's
  physical-order full tag register.
- `scratch_memory` is exactly 512 bytes / 1024 lowercase hexadecimal digits
  rooted at `0x6666_6601_0100`. It is mutually exclusive with `mem0_value`
  and `mem1_value`.
- FXSAVE/FXRSTOR payloads are **logical ST0…ST7 order**, even though their
  abridged tag is physical order. This boundary conversion is already
  implemented in Aegis; test it rather than changing it.

The full protocol is in `X87_HANDOFF.md` and Aegis's `README.md`.

## Required state matrix

For every restore instruction, create at least four input images:

| TOP | required physical payloads | tag classes to make observable |
| --- | --- | --- |
| 0 | all eight R slots distinct | valid, zero, special, empty |
| 3 | all eight R slots distinct | valid, zero, special, empty |

Across the suite, ensure that logical slots restored from memory include:

- +0 and -0;
- a normal finite value;
- a denormal;
- +infinity/-infinity;
- qNaN and an unsupported f80 encoding (nonzero exponent with integer bit
  clear);
- an empty physical slot.

For legacy `FRSTOR`/`FLDENV`, set full tag pairs deliberately in the memory
image. Do not infer their classes from the payload in the test generator: the
point is to verify that SLEIGH restores the supplied full tag word. For
FXRSTOR, test the abridged-to-full synthesis separately from payload ordering.

Seed non-default control/status values too: at minimum RC=up, PC=single,
several sticky exception bits, C1, and distinct instruction/data pointers.
That makes accidentally omitted environment fields observable.

## Image layouts to seed and verify

These are the layouts currently used by
`wazabin-sleigh/precompile/open_sleigh/src/x86/ia.sinc`; hardware capture is
the oracle if a manual/architecture detail disagrees.

### FXSAVE / FXRSTOR

- CW: `+0` (u16), SW: `+2` (u16), abridged FTW: `+4` (u8), FOP: `+6` (u16)
- instruction pointer: `+8`; data pointer: `+16`; MXCSR: `+24`
- logical ST(i) payloads: `+32 + i*16`, ten bytes per slot

The existing save-only row is **11674** (`FXSAVE [RBX]`). It verifies the
header but is not restore coverage.

### FSAVE/FNSAVE / FRSTOR and FSTENV/FNSTENV / FLDENV

The current SLEIGH constructors use the legacy 28-byte environment:

- CW: `+0`, SW: `+4`, full tag word: `+8`
- instruction pointer: `+12`, FOP: `+18`, data pointer: `+20`
- logical ST(i) payloads in save/restore images: `+28 + i*10`

The existing save rows are **11685** (`FNSTENV [RBX]`) and **11686**
(`FNSAVE [RBX]`). They are output-only baseline cases, not restore inputs.

## Implementation path

1. Add `binit/generator/admit_x87_environment_restore_cases.py` and focused
   pytest coverage. Build scratch images with `bytearray(512)`, then write
   fields with `to_bytes`; never use host-float serialization.
2. Give every state `rbx = MEM0_ADDRESS`, `scratch_memory`, all physical
   `x87_rN` inputs (with deliberately different pre-restore values), and
   nontrivial x87 control/status/tag fields. The final hardware snapshot must
   demonstrate that the memory image, not the original file, won.
3. Insert only new rows. Never mutate or overwrite 11674–11697.
4. Capture with Aegis, verify every state has a result, then run strict QCode
   replay. Fix SLEIGH/QCode only when the captured result identifies a real
   mismatch; do not normalize away payload, TOP, tag, or scratch differences.
5. Update `X87_HANDOFF.md` with IDs, state count, exact coverage, and any
   discovered architectural behavior. Commit Binit and QCode changes in small
   separate commits.

## Commands

```bash
# Generate reviewable inputs, then insert only after review.
cd /home/jack/dev/binary/binit
python3 generator/admit_x87_environment_restore_cases.py --output /tmp/x87-env.json
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
  python3 generator/admit_x87_environment_restore_cases.py --insert

# Capture only the newly printed IDs.
cd /home/jack/dev/aegis
rm -f /tmp/serial.sock /dev/shm/ivshmem
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
  just run --test-case-id ID [--test-case-id ID ...]

# Strict replay; do not set PCODE_FUZZ_FAIL_QUIETLY.
cd /home/jack/dev/binary/wazabin-qcode
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
PCODE_FUZZ_TEST_CASE_IDS=ID,ID \
PCODE_FUZZ_THREADS=1 \
PCODE_FUZZ_OUTPUT=/tmp/x87-env-qcode.csv \
  cargo test -p wazabin-qcode-sleigh --test binit binit_full -- --ignored --nocapture
```

## Current baseline

- Physical transport and save-side corpus: IDs **11666–11686**, clean.
- Arithmetic/conversion corpus: IDs **11687–11697**, 346 hardware states,
  clean in strict QCode replay.
- Relevant committed changes: Binit `1003b12`; QCode `c5c7b6a` and
  `16da489`.

Do not `git clean`, reset, or update the SLEIGH submodule blindly. QCode uses
`/home/jack/dev/binary/wazabin-sleigh/precompile/open_sleigh`, not a separate
checkout named `open_sleigh`.
