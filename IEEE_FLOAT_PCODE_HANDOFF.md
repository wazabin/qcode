# IEEE floating p-code handoff

## Decision

Do not encode architecture-specific floating exception policy in the concrete
emulator.  Generic IEEE arithmetic is exposed to SLEIGH through explicit
p-code operations with an explicit rounding mode:

```sleigh
result = float_add(a, b, rounding_mode);
flags  = float_add_flags(a, b, rounding_mode);
```

Equivalent pairs are required for subtraction, multiplication, division, and
later unary/conversion operations.  `flags` contains only IEEE-754 facts:
invalid, divide-by-zero, overflow, underflow, inexact, and (where needed) a
rounded-up indication.  The operations must work for their operand format
(f32, f64, f80) and must not know about x87, SSE, ARM, or RISC-V.

The two operations are deterministic evaluations of the same operands, format,
and rounding mode.  SLEIGH macros may wrap them for readable constructors:

```sleigh
macro ieee_add(a, b, round, result, flags) {
    result = float_add(a, b, round);
    flags = float_add_flags(a, b, round);
}
```

## x87 policy belongs in `ia.sinc`

x87 SLEIGH consumes the generic result and flags and performs all of:

- mapping IEEE flags to x87 status bits;
- denormal-operand detection (DE);
- mask and priority handling;
- C1, ES, and B updates;
- invalid-result selection;
- result-store and pop suppression.

The QCode emulator must not directly mutate `FPUStatusWord`, choose x87
indefinite values, or decide x87 commit/pop behavior for these operations.

## Current transitional state

`float_add_flags`, `float_sub_flags`, `float_mul_flags`, and
`float_div_flags` were introduced as an f80/nearest-rounding prototype. They
are **not ready to wire into constructors**: they lack an explicit rounding
argument, are not format-generic, and current QCode still directly records
x87 status for generic floating instructions.

Before wiring x87 constructors:

1. Replace the prototype with `float_{add,sub,mul,div}` and matching
   `*_flags`, all accepting an IEEE rounding-mode argument.
2. Implement these generically in QCode for f32/f64/f80.
3. Add SLEIGH IEEE helper macros.
4. Migrate x87 constructors and remove their contextual emulator status side
effects one operation family at a time, replaying Binit after each family.
5. Keep x87 invalid-result payload selection and exception priority explicit
in SLEIGH; capture more hardware cases where payload sign differs.

## Current implementation status

The declarations/macros from the sibling `/home/jack/dev/binary/open_sleigh`
checkout were integrated into the vendored submodule:

- `float_{add,sub,mul,div}` and matching `*_flags` declarations are in
  `precompile/open_sleigh/src/x86/ia.sinc`.
- `ieee_{add,sub,mul,div}` macros are in
  `precompile/open_sleigh/src/x86/macros.sinc` and call the operations with
  `(a, b, round)`.

They are intentionally **not wired** into x87 constructors yet. Wiring them
before removing QCode's current contextual x87 status recording would duplicate
architectural effects.

## IEEE operation implementation — complete

`wazabin-qcode/emulator/src/concrete.rs` now implements:

- three-argument `float_{add,sub,mul,div}(a, b, rounding_mode)` result ops;
- matching three-argument `*_flags` ops returning architecture-neutral IEEE
  masks;
- f32, f64, and f80 evaluation for every IEEE rounding mode;
- legacy two-argument f80 flag ops for existing callers.

The implementation has result/flag agreement tests across operations, formats,
and rounding modes. Reported verification: `cargo test -p qcode_emulator --lib`
passes 56 tests.

## Migration status

The x87 add (`FADD`, `FADDP`, `FIADD`), subtraction (`FSUB`, `FSUBR`,
`FSUBP`, `FSUBRP`, `FISUB`, `FISUBR`), and multiplication (`FMUL`, `FMULP`,
`FIMUL`) families now use `ieee_add`, `ieee_sub`, and `ieee_mul` through
helpers in the vendored `ia.sinc`. The helpers pass `(FPUControlWord >> 10) &
3`, map generic flags into the x87 status word, report DE from the source
encodings, compute C1 against a toward-zero result, select x87 invalid
payloads, and apply existing mask/commit logic. These constructors no longer
use generic `f+`, `f-`, or `f*`, so QCode's contextual arithmetic paths are
not reached for them.

Next, replay focused Binit arithmetic cases and then the full corpus. Migrate
division one family at a time only after that succeeds. Do not replace generic
floating operators globally: non-x87 specifications keep them.

## Relevant commits

- open-sleigh `76c76b2`: original f80 flag-op prototype.
- open-sleigh `4d33b85`: sibling generic IEEE declarations/macros integrated.
- open-sleigh `c3f54b7`: removes duplicate declarations.
- wazabin-sleigh `25161ec`: pins the integration.
- wazabin-qcode `22f7402`: f80 flag-op prototype.
- wazabin-qcode `14d3c0d`: original design handoff.

The arithmetic commit/status experiments are separate later open-sleigh commits.

## Precision control, division, and the removal of the contextual path

Three further changes completed the arithmetic migration.

**Precision control as an IEEE operation.** `float_round_to_precision(value,
precision_bits, round)` and its `*_flags` twin round a value's significand to a
narrower precision *within its own exponent range*. The precision is an
explicit operand (24, 53, or 64 significand bits, integer bit included; 64 is
the identity), so the operation carries no x87 policy. Only the 80-bit format
is implemented — f32/f64 operands return no result and fall through to the
generic interpreter. `ia.sinc`'s `fpu_precision_control` maps the architectural
`(FPUControlWord >> 8) & 3` field onto it, treating the reserved `01` encoding
as extended, and `fpu_ieee_{add,sub,mul,div}_result` apply it to both the
result and the toward-zero value the C1 comparison uses.

**Division.** `fpu_ieee_div_result` mirrors the other families and adds x87's
exponent-wrapped delivery: on an unmasked overflow or underflow the true
exponent is biased by -24576 or +24576 so a result the format cannot hold is
still delivered, with its real rounding indication and inexactness. SLEIGH
reproduces that by rescaling whichever operand has the exponent room by the
same factor and dividing again; with no unmasked range exception the operands
are unchanged, so one evaluation serves both cases. `fpu_ieee_arithmetic_status`
gained a `wrapped` argument that keeps C1 and PE for such a delivered result.
The division constructors also report stack faults (`fpu_stack_underflow`).
`FYL2X` and `FYL2XP1` moved to `fpu_ieee_mul_result` at the same time.

**Emulator cleanup.** No constructor now applies generic `f+ f- f* f/` to an
f80 with x87 semantics, so `interpret_x87_float` no longer interprets
arithmetic: its `Binop::Float` arm handles only comparisons (a signalling NaN
still raises invalid there). The legacy two-argument `_flags` prototype is
gone, and `float80::{add,sub,mul,div}` are now plain architecture-neutral
IEEE fallbacks; their `*_contextual` forms were deleted. `apply_precision`,
`arithmetic`, and `invalid_result` survive only because `from_i128_contextual`
and `scale_contextual` still use them; they should disappear when the
conversion and FSCALE families migrate.
