# Binit corpus QCode replay status

## SLEIGH transfer fixes — 2026-09-07

The August status below is historical. The September 3 CSV had 40 x87
first-difference state mismatches; it is not the post-fix backlog.

Changed `wazabin-sleigh/precompile/open_sleigh/src/x86/ia.sinc` (the vendored
specification, not the sibling top-level `open_sleigh` checkout). Spec commit
`8602251` is pinned by `wazabin-sleigh` commit `2cb0031`; standalone tests are
in QCode commit `13b4551`:

- `FLD ST(i)` checks the source tag, not the old ST(0) tag, and checks the
  push destination for overflow before changing TOP. Masked faults push the
  indefinite value; unmasked invalid preserves payloads, tags, and TOP.
- Register `FST`/`FSTP` report empty-source IE|SF, write an occupied indefinite
  destination when masked, and do not store/pop on unmasked invalid. Every
  register form clears C1, including ST(0) aliases and the previously missed
  FSTP constructors.
- `FXCH` substitutes indefinite only for empty operands before exchanging,
  propagates the corresponding full tags, clears C1, and leaves both operands
  alone on unmasked invalid. The implicit and explicit forms share the logic.
- `FISTTP` and `FBSTP` clear stale C1 and report empty-stack faults without
  overwriting the source payload. Conversion is staged before storing/popping;
  unmasked invalid leaves memory and TOP alone.
- `FFREE`/`FFREEP` clear C1 consistently. FFREE condition codes are
  architecturally undefined; zero is an explicit hardware-compatible choice,
  not a newly claimed architectural guarantee.

**Important correction to the old exception diagnosis:** not every unmasked
exception aborts without producing a result. A native
`FXRSTOR; FISTTP/FBSTP; FXSAVE; FNINIT` check with ST(0)=0.5 and CW=0x35f
(unmasked precision) stores zero and pops: SW=0x88a0 from TOP=0. With +infinity
and CW=0x37e (unmasked invalid), memory/TOP stay unchanged: SW=0x8081.
The standalone regression tests cover this distinction; the existing Binit
FISTTP/FBSTP inputs only use CW=0x37f. General arithmetic exception delivery
and exception-priority handling are **not** fixed by this batch. Nor is this
complete FBSTP rounding-control coverage: the `to_bcd` user-op still does not
provide a rounded-up C1 result; this batch clears the stale bit for the exact,
round-down, and invalid/empty cases covered here.

Regression tests are in `sleigh/tests/x64.rs`: every ST(i), every TOP,
masked/unmasked stack faults, FLD overflow/source-destination aliases, full
payload/tag preservation, all three FISTTP widths, and BCD stores. ST(0)
constructors are tested even though this database snapshot lacks most of them.

Focused selection (resolve IDs from the current database; do not hard-code):

```sql
select string_agg(id::text, ',' order by id) from test_cases
where instruction ~* '^(ffree|fisttp|fbstp|fld st|fstp? st|fxch)';
```

The first focused replay went from 8 failing families / 0 clean cases to
**53 clean cases / 69,144 states, no mismatches**. Expanding the selection to
memory FLD/FST/FSTP exposes remaining faults rather than register regressions:

- `FLD m64fp`, input signaling NaN `0x7ff0000000000001`: the widened payload
  is correct but IE is missing (SW=0x3800 versus 0x3801).
- Memory `FST`/`FSTP`, minimum f80 subnormal and CW=0x340: the floating-point
  conversion reports PE as well as unmasked UE (0x80b0 versus 0x8090); FSTP
  also pops where hardware does not. These constructors were not changed.

### Post-fix verification

- `cargo test -p wazabin-qcode-sleigh --test x64`: 66 passed, 11 pre-existing
  ignored; `--test x86`: 7 passed. Six new regression tests exercise 1,736
  transfer/store scenarios without the database.
- `binit_undefined_flags_are_explicit_undef_assignments`: passed.
- `binit_smoke`: passed.
- Final focused replay: all 53 cases / 69,144 states passed, output
  `/tmp/x87-spec-targeted-final.csv` (header only).
- Full scalar corpus: 13,079 cases / 9,869,419 recorded states, 12 threads,
  315.71 seconds in release mode. **12,792 clean cases**, 238 skipped behind
  a same-instruction failure, **35 state mismatches**, 13 unsupported,
  1 fixture limitation. The test exits failing for the remaining 35 state
  mismatches and the fixture limitation; it does not waive them.

```sh
# From wazabin-qcode/. Use release mode for the full multi-million-state run.
PCODE_FUZZ_OUTPUT=/tmp/binit-x87-spec-fixes-full.csv \
  cargo test --release -p wazabin-qcode-sleigh --test binit binit_full \
  -- --ignored --nocapture
```

The full CSV has 49 rows (including the 13 ignored unsupported records),
versus 54 in `/tmp/binit-full-20260903-155700.csv`. State mismatches fall from
**40 to 35**: `FBSTP`, `FFREE`, `FFREEP`, `FISTTP`, and `FXCH` disappear.
`FLD`, `FST`, and `FSTP` now fail on the untouched memory forms rather than
register transfers. No newly failing mnemonic and no non-x87 state mismatch
appeared. This is still a first-difference/sibling-suppressed report, not a
claim that every remaining field or opcode is correct. The remaining
arithmetic, denormal, payload/precision, transcendental, and FXSAVE issues
need further work; the entire suite is **not green**.

## The corpus is generated — never select cases by ID

Binit's corpus is built from `generator/data/insn.json`, and regenerating it
clears the database and reassigns every ID. Any document that names IDs is
wrong the moment the corpus is rebuilt; an earlier revision of this one pointed
at 11666-11891, which by then held unrelated `sub` cases.

Select by mnemonic instead:

```sql
select string_agg(id::text, ',' order by id) from test_cases
where instruction ~* '^(f|maskmov|p(add|sub|cmp|unpck|ack|sll|srl|sra|mul|avg|madd|sad|or|and|xor))';
```

The database currently holds 12881 cases / 7014763 captured result states, all
with hardware results. Regenerating and recapturing takes minutes:
`echo yes | uv run --extra generator python generator/make_test_cases.py`, then
`cd aegis && just run`. Back the database up first - `pg_dump -Fc` is ~84MB -
because the generator clears it before any capture has happened.

## Replay status — 2026-08-28

**12665 of 12881 cases clean.** Full corpus, 7014763 states, 12 threads.

Always replay the whole corpus before deciding what to work on. Ranking work
from an x87-only subset put a binit fixture gap worth 624 cases out of view
entirely and made the x87 lift failures look like the cheapest fix available
when they were worth 8-12 cases each.

Case counts are a floor, not a verdict, in two ways. A failing case suppresses
its same-instruction siblings, so fixing one converts several. And the harness
reports only the *first* differing field, so fixing a field reveals the next
one underneath: this session closed FOP entirely and the state-mismatch count
went *up*, because seven instructions moved from `backend_error` to
`state_mismatch`.

| | Session start | Now |
| --- | ---: | ---: |
| Cases OK | 11847 | **12665** |
| Skipped behind a failure | 967 | 164 |
| State mismatches | 40 | 39 |
| Lift failures | 10 | **0** |
| Unsupported ops | 14 | 12 |
| Fixture limitations | 3 | 1 |

There are **no lift failures left in the corpus**. Every remaining state
mismatch is x87.

## What remains

### The unmasked-exception abort (at least 4 cases)

`fmulp`, `fdivp`, `fdivrp` and `frndint` all differ because an *unmasked*
exception aborts the instruction on hardware: it produces no result, sets no
result-derived flag, and does not pop. `record_x87_status` deliberately does
not model this - "an unmasked exception sets ES, but does not yet transfer
control to a hardware exception handler; that deliberately non-trapping policy
keeps the generic emulator API intact until architectural trap delivery is
modelled". These are the first cases where the simplification is observably
wrong: TOP has advanced where hardware left it alone, and `frndint` reports a
precision result hardware never computed.

This is a design decision, not a flag fix, and it likely reaches beyond x87.

### Denormal-operand exceptions (5 cases)

- **Missing DE** - `fprem`, `fprem1`, `fscale`. Implemented as user-ops, so
  they never reach `record_x87_status` and report no denormal operand at all.
- **Spurious DE** - `fist`, `fistp`. Architecturally FIST raises invalid and
  precision only. The DE comes from the *decomposition*, not the architecture:
  the constructor is `trunc(round(x))` and it is `round()` - `FloatRound` -
  that reports the denormal. It cannot simply be silenced there, because
  `FRNDINT` is the same p-code operation and does raise DE. Either FIST stops
  going through `round()`, or the conversion path stops inheriting its status.

### Stack faults not reported (3-4 cases)

`fld st(1)`, `fst st(2)`, `fbstp` (and `fxch`, seen while other fields were
being fixed) expect IE|SF from referencing an empty register. The machinery
exists and works - `fpu_stack_underflow_out` plus `fpu_underflow_indefinite`,
as used by the FCMOV family - these constructors just do not call it.

### Smaller

- **Missing IE** - `fxtract`, `fyl2x` (2).
- **C1 not cleared** - `fstp st(1)`, `fisttp` (2). The register-store path
  still leaves the previous instruction's rounding answer standing.
- **`fcom dword ptr`** (1) reports DE where hardware reports IE alone; its
  expected status already has IE, so this is an unsupported operand being
  classed as a denormal rather than a missing flag.
- **`x87_r0`/`x87_r1`/`x87_r7`** (18) and **`fxsave` scratch layout** (1) are
  unanalysed. The register cluster is largely a low significand byte differing
  by `0x01` under a narrowed precision control.

### Fixture limitation (1) — binit, not a lifter bug

`cmpxchg16b xmmword ptr [...]` has a 128-bit operand and the harness models an
8-byte window, so the unmodelled high half drives the divergence. The fix has
the same shape as the bit-index one already landed: seed it through the
512-byte `scratch_memory` transport instead.

### Unsupported ops (12)

Six are the deferred transcendentals below. The rest: `swapgs`, `rdpmc`,
`invlpg`, `sfence`, `lfence`, `mfence`. The three fences are plausibly the same
no-op argument that settled `LOCK`; the privileged three may have no meaningful
replay semantics.

### Transcendentals — deferred indefinitely

`f2xm1`, `fsin`, `fcos`, `fsincos`, `fptan`, `fpatan` report
`unsupported_pcode_op` and **will not be implemented**. Reproducing hardware
bit-for-bit needs an accuracy model, a different kind of commitment from
everything else here: FSCALE, FXTRACT, the BCD conversions and the 80-bit
square root were implemented because they are *exact*. Treat these 6 cases as
permanently out of scope, not as outstanding work.

### Latent, and invisible to the corpus

`imm8:8` appears on 34 lines of `ia.sinc` - `pshufhw`, `pshuflw`, `mpsadbw`,
`dpps`, `dppd`, `blendps`, `blendpd`, `pblendw`, `roundps/ss/pd/sd`,
`insertps`, `extractps`. Every one is the bug that broke PSHUFW: an 8-bit field
cast to 8 bytes. They do not fail today only because the corpus contains 7 XMM
cases and none of those mnemonics. Correcting the cast alone would only move
them from `backend_error` to `unsupported_pcode_op` - none of the user-ops is
implemented - so the real fix is to widen binit to SSE2/SSE4.1 shuffle, blend
and round forms, recapture, and then lower the whole class against real signal.

## Traps this session hit

Worth knowing before touching the same machinery.

- **Address-of is a disassembly-time fold**, not a p-code operation, and needs
  a symbol with a static address. It is *not* the way to name a memory
  operand's address. The `m*` subtables already export it: `m16` is literally
  `export *:2 Mem`, so `Mem` is the address and `fpu_record_data_pointer(Mem)`
  is how FDP is recorded. Lowering `&` over a load to a runtime pointer makes
  the spec uncompilable by Ghidra, and the corpus cannot see the problem
  because the spec and the lifter then agree with each other.
- **Raw `mod`, `reg_opcode` and `r_m` do not survive to p-code lowering** -
  they are consumed by the addressing subtables. `fregidx` exists for exactly
  this reason, and `modidx`/`regidx`/`rmidx` now do the same job. Reading them
  directly fails with `unresolved field reached p-code lowering`.
- **Check the register name.** `fpu_record_opcode` wrote to `FPUOpcode`; the
  register is `FPULastInstructionOpcode`. The macro had never had any effect,
  and nothing complained.
- **The WAIT prefix is not the opcode.** `FSTCW` is `9B D9 /7`; deriving FOP
  from the first `byte=0x..` in the pattern picks up the `9B`.
- **Macros must be defined before use**, and a textual sweep that rewrites
  `FPUStatusWord = FPUStatusWord & 0xfdff;` will also rewrite the body of
  `fpu_clear_c1` into a call to itself.

## FOP and FDP

`FOP = ((first opcode byte & 7) << 8) | second byte`, where the second byte is
the modrm byte or the fixed second opcode byte. FIP updates on every x87
instruction; **FDP and FOP update together and only when an unmasked exception
is pending**. That gate was confirmed against hardware rather than assumed: for
`fdiv st(0),st(1)`, FOP holds its stale value in every captured state except
those whose status has ES set.

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

A status-word mismatch is rarely one bug. XOR actual against expected and group
by the differing bits - that is what separated the 19 `x87_status` cases into
six independent causes, including a denormal-exception cluster that fails in
*both* directions (three instructions never raise DE, four raise it when
hardware does not). The bits: C0 `0x0100`, C1 `0x0200`, C2 `0x0400`, TOP
`0x3800`, C3 `0x4000`, B `0x8000`, and IE/DE/ZE/OE/UE/PE in the low six with
SF `0x40` and ES `0x80`.

To decide whether a field updates unconditionally or only under some condition,
group the captured states rather than reasoning from the manual:

```sql
select (r.final_state->>'x87_status')::bigint status,
       r.final_state->>'x87_opcode' fop, count(*)
from test_cases c join test_results r on r.test_case_id = c.id
where c.id = <case> group by 1, 2 order by 1;
```

Commit Binit generator/test changes separately from SLEIGH/QCode fixes.

## History

Fixed in earlier passes, each confirmed against a hardware capture.

### This session (11847 -> 12665 cases)

- **binit fixture: the bit-index memory window** (+622). `BT`/`BTC`/`BTR`/`BTS`
  with a register index reach `base + (index s>> 3)`, tens of bytes outside the
  8-byte `mem0_value` word, so every such state diverged on memory the fixture
  never modelled. They now use the 512-byte `scratch_memory` transport with the
  operand centred in it. The index pool is bounded to +-256 to match, which
  loses nothing: against a register destination the index is masked to the
  operand width, and a test asserts the bounded pool still covers every masked
  residue.
- **PSHUFW lowered to p-code** (+96), replacing `pshufw(..., imm8:8)` - an
  8-bit field cast to 8 bytes. Each destination word shifts the source down by
  `selector * 16`; the source is copied first because the destination may alias
  it.
- **CRC32 lowered to p-code** (+40) as the reflected CRC-32C update, one
  `crc32_byte` macro for all six constructors. The step function was checked in
  isolation first - CRC-32C of `"123456789"` is `0xe3069283` - before any
  corpus run.
- **LOCK/UNLOCK as emulator no-ops** (+31). The markers stay in the IR for
  analysis consumers; the prefix orders an access against other bus agents and
  constrains nothing about the resulting state.
- **FCMOV faults on an empty operand** (8 cases). It references ST(0) and ST(i)
  whether or not it moves. `FCMOVNBE` also skipped on `CF & ZF` where NBE moves
  only when both are clear, so it must skip when either is set.
- **FOP recorded on every x87 instruction**, and written to the register that
  exists. This also fixed all 8 pre-existing x87 lift failures, which were the
  earlier FOP calls reading raw modrm fields.
- **FDP recorded for every memory operand**, from the `Mem` subtable.
- **Address-of an address literal folds** (`wazabin-pcode`), which was the last
  `backend_error` in the corpus - `push88(&:8 inst_next)` in CALL.
- **B mirrors ES** in the emulator. The spec-side change alone could not work:
  for `FMUL m64fp` and friends the spec never touches the status word at all,
  and the emulator raises the sticky bits from APFloat.
- **C1 is written on every operation that reports a rounding direction**, and
  cleared in FFREE/FFREEP/FBSTP. It is an operation result, not a sticky bit,
  so an exact result clears it rather than leaving the previous answer.

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
