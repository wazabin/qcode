# Binit corpus QCode replay status

## The corpus was regenerated — old IDs are gone

Binit's corpus is now built from `generator/data/insn.json`; the
`admit_x87_*_cases.py` scripts and the ID ranges every earlier revision of this
document referenced (11666–11891) have been retired, and the generator cleared
the database. Those IDs now hold unrelated `sub` cases.

**Never select replay cases by ID.** IDs are not stable across a regeneration.
Select by mnemonic:

```sql
select string_agg(id::text, ',' order by id) from test_cases
where instruction ~* '^(f|maskmov|p(add|sub|cmp|unpck|ack|sll|srl|sra|mul|avg|madd|sad|or|and|xor))';
```

The database currently holds 12881 cases / 7.02M captured result states, all
with hardware results. That selector matches 841 x87/MMX cases / 4.94M states.

## Replay status — 2026-08-28

Always replay the **whole** corpus before deciding what to work on. A run over
the x87/MMX subset alone ranked the work wrongly: every finding it could
surface was an x87 finding, so the largest item in the corpus — a binit fixture
gap gating 624 `BTS`/`BTC` cases — was invisible, and the x87 lift failures
looked like the cheapest available fix when they are worth 8-12 cases each.

Full corpus, 12881 cases / 7021675 states, 12 threads: **11847 OK, 40 state
mismatches, 10 lift failures, 14 unsupported ops**, plus 967 cases skipped
after another case with the same instruction had already failed. Case counts
are a floor, not a verdict: fix one failure and its skipped siblings become new
signal.

All 40 state mismatches are x87/MMX. Nothing outside x87 regressed — in
particular the architecture-independent p-code shift-amount fix holds across
the whole corpus.

This run included the then-uncommitted `ia.sinc` FOP / B-bit work, which is
partial: it appears in the failures below rather than being inert.

### Priority by cases unblocked

| Fix | Cases gated |
| --- | ---: |
| `BTS` + `BTC` fixture window (binit) | 624 |
| `PSHUFW` lift failure | 96 |
| `CRC32` | 40 |
| `XCHG` LOCK | 32 |
| any single x87 family | 6-12 |

### Fixture limitations (3) — binit, not lifter bugs

The harness seeds only the 8-byte `mem0` window and zeroes the rest of the
page, so these cases diverge on memory the fixture never modelled; the lifted
QCode is correct. `fixture_limitation_reason` in `sleigh/tests/binit.rs`
classifies them.

- `bts`/`btc qword ptr [...], rbx` — a register bit index addresses
  `base + (index s>> 3)`, outside the modelled 8 bytes. **624 cases.**
- `cmpxchg16b xmmword ptr [...]` — a 128-bit operand exceeds the 8-byte
  window. 7 cases.

### Lift failures (10)

Eight are x87, all `unresolved field reached p-code lowering`: `ffree st(1)`,
`ffreep st(1)`, `fxch`, `fsub st(0),st(1)`, `fsubr st(0),st(1)`, `fcom dword
ptr`, `ficomp dword ptr`, `fisub dword ptr`. Same class as the already-fixed
`mmxreg2op_m64` bug: a constructor matching a raw field whose body exports a
subtable.

Two are not:

- `pshufw mm0, mm1, 0x0` — `invalid bit range [0, 64] for 8-bit storage`.
  **96 cases.**
- `call 0x666666661042` — `raw p-code lowering does not support address-of a
  non-...`.

### Unsupported ops (14)

Six are the deferred transcendentals below. The other eight: `swapgs`,
`rdpmc`, `crc32` (40 cases), `invlpg`, `sfence`, `lfence`, `mfence`, and
`LOCK` on `xchg qword ptr` (32 cases). The fences and `LOCK` are plausibly
cheap — no observable state in this model; `crc32` is real work.

### State mismatches (40), by cluster

All x87/MMX.
- **B bit not mirrored** (7) — expected `0x8081`, got `0x81`: `fdiv`, `fdivr`,
  `fidiv`, `fimul`, `frndint`, `fist`, `fistp`. The uncommitted
  `fpu_refresh_error_summary` change is not reaching these paths.
- **FOP wrong** (2) — `fsqrt` and `fidivr` both report a constant `0x111`
  against expected `0x1fa` / `0x63d`. `fpu_record_opcode` is wired into too few
  constructors, and computes the wrong value in those it does reach.
- **FCMOV** (8) — status/tag left at `0x0` where hardware leaves `0x41` / `0x1`
  across the whole conditional-move family.
- **Significand LSB** (7) — `x87_r0`/`x87_r1` off by `0x01` in the low
  significand byte: `fadd`, `faddp`, `fiadd`, `fisubr`, `fsubp`, `fsubrp`,
  `fcmovnbe`.
- **C2 partial reduction** (3) — `fprem`, `fprem1`, `fscale` expect `0x2`, get
  `0x0`. Earlier revisions listed this path as implemented but uncovered by any
  vector; the regenerated corpus reaches it and it is wrong.
- **Singles** — `fxsave` scratch-memory layout, `fbld`, `fbstp`, `fstp st(1)`,
  `fmul`, `fisttp`, `fld st(1)`, `fst st(1)`, and C0 on `fxtract` / `fyl2x` /
  `fyl2xp1`.

### Transcendentals — deferred indefinitely

`f2xm1`, `fsin`, `fcos`, `fsincos`, `fptan`, `fpatan` report
`unsupported_pcode_op` and **will not be implemented**. Reproducing hardware
bit-for-bit needs an accuracy model, a different kind of commitment from
everything else here: FSCALE, FXTRACT, the BCD conversions and the 80-bit
square root were implemented because they are *exact*. Treat these 6 cases as
permanently out of scope, not as outstanding work.

## Workflow

Replay the whole corpus. Leaving `PCODE_FUZZ_TEST_CASE_IDS` unset selects every
case that has results (AVX-tagged instructions excluded by the query):

```bash
cd /home/jack/dev/binary/wazabin-qcode
X86DB_DSN=postgresql://x86db:x86db@localhost:5432/x86db \
PCODE_FUZZ_THREADS=12 \
PCODE_FUZZ_OUTPUT=/tmp/full-corpus.csv \
cargo test --release -p wazabin-qcode-sleigh --test binit binit_full -- --ignored --nocapture
```

Release build and 12 threads put the full 7.02M-state sweep in minutes, so
there is no reason to sample. The single-threaded debug invocation earlier
revisions prescribed does not scale to the regenerated corpus. Set
`PCODE_FUZZ_TEST_CASE_IDS` only to iterate on one instruction while fixing it,
never to decide what to fix.

Retain the CSV: `diff_json` names the offending field per mismatch. To rank
findings by leverage, count the sibling cases each failing instruction gates:

```sql
with f as (select distinct instruction_id from test_cases where id in (<failing ids>))
select i.name, count(*) from test_cases c
  join f on f.instruction_id = c.instruction_id
  join instructions i on i.id = c.instruction_id
group by i.name order by 2 desc;
```

Commit Binit generator/test changes separately from SLEIGH/QCode fixes.

## History

Fixed in earlier passes, each confirmed against a hardware capture:

### SLEIGH

- FLDL2T/FLDL2E/FLDPI/FLDLG2/FLDLN2 load exact 80-bit constants; they were
  f64 literals widened by `float2float`, which zeroed the low 11 significand
  bits of every one.
- C1 is cleared where it is architecturally defined clear (FABS, FCHS) and
  *before* the operation where it is a rounding indicator the operation sets
  (FADD, FSQRT, FSCALE). Clearing after discarded what the operation reported.
- The FCOMI family no longer clears C1 at all: it reports only through
  ZF/PF/CF, leaving C0-C3 alone.
- The ordered and unordered compares are separate. FCOM and FUCOM produced
  identical p-code, so the interpreter applied FCOM's quiet-NaN rule to both;
  `fcom`/`fcomi` now signal invalid for a quiet NaN and `fucom`/`fucomi` do
  not.
- Stack underflow is modelled: referencing an empty register sets IE and SF
  and clears C1. A new `fregidx` subtable exports an operand's logical index,
  since the raw `freg` field is consumed by `fregop`.
- FLD m32fp/m64fp, FILD m16/m32/m64 and FBLD open-coded `fdec()` and a store,
  bypassing `fpushv` and its overflow detection.
- A stack fault takes precedence in FXTRACT, aborting the extraction.
- The MMX shift family shares one 64-bit count clamped to the lane width;
  PSLLD and PSRAD had used a separate per-lane count from each half of the
  source, and an oversized count wrapped instead of emptying the lane.
- `packsswb`/`packssdw` passed a destination bit range as a macro output
  parameter — an rvalue there — so no saturated lane was ever written.
- `mmxreg2op_m64` matched the raw field while its body exported the subtable,
  leaving it unresolved at lowering.
- FSTENV/FNSTENV write the selector fields at +16 and +24.
- FPREM/FPREM1 lower to their own user-ops instead of `x - trunc(x/y)*y`.

### QCode

- A p-code shift amount is the full unsigned value of input1 and empties the
  operand when out of range. It had been masked to input0's width, truncated
  to `u32`, then reduced modulo 128. **This is architecture-independent**; a
  full replay of the whole corpus confirmed no regression outside x87/MMX.
- Packed MMX user-ops: `pavgb`, `pavgw`, `pmulhuw`, `pmaddwd`, and the eight
  saturating add/subtract forms.
- FIST/FISTP/FISTTP store the integer indefinite on an invalid conversion,
  not APFloat's saturated bound.
- A narrowing f80 store no longer raises the denormal-operand exception, and
  an invalid one stores the *signed* indefinite QNaN.
- Comparisons are quiet: only a signalling NaN raises invalid.
- FSCALE via APFloat's `scalbn`; FPREM/FPREM1 with the quotient's low three
  bits in C0/C3/C1 and C2 for an incomplete reduction; FBLD/FBSTP packed
  decimal; FXTRACT's significand and exponent.
- A correctly rounded 80-bit square root computed on the integer significand.
  APFloat has none, and the generic path routed f80 through f64, losing
  eleven significand bits.


## Invariants

- `scratch_memory` is exactly 512 bytes / 1024 lowercase hex digits at
  `0x6666_6601_0100`, exclusive with `mem0_value`/`mem1_value`.
- Binit state is physical; never use `x87_stN` fields.
- `ST(i) = R[(TOP+i)&7]`; never rotate JSON payload fields.
- FXSAVE payload order is logical while its abridged tag is physical.
- Do not reset/clean the dirty unrelated checkouts or blindly update the
  OpenSleigh submodule.
