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

They are intentionally **not wired** into constructors yet. The current QCode
prototype supports only two-argument f80 `*_flags` queries and directly
mutates x87 status for ordinary `f+` etc.; wiring the macros now would produce
unsupported p-code operations and duplicate x87 effects.

## Immediate implementation task

Refactor `wazabin-qcode/emulator/src/concrete.rs`, in
`StandaloneEmulator::interpret_packed_pcode_op` around the current
`let [lhs, rhs] = op.args.as_slice()` dispatch:

1. Accept both `[lhs, rhs]` (legacy p-code ops) and
   `[lhs, rhs, rounding_mode]` (new IEEE ops).
2. Implement `float_{add,sub,mul,div}` returning the f80 result, and matching
   `*_flags` returning the IEEE bit mask, using the same explicit rounding
   mode for both calls.
3. Generalize the implementation beyond f80 before claiming the operations
   architecture-independent; f32/f64 must use their corresponding IEEE
   formats.
4. Add focused emulator tests proving result/flags agreement for every
   rounding mode.
5. Only then migrate one x87 family at a time to `ieee_*`, move status mapping
   into SLEIGH, and remove the corresponding contextual x87 mutation from
   QCode. Replay Binit after each family.

Do not replace generic `f+` globally: non-x87 specifications keep it. The new
operations are explicit supplemental IEEE evaluation used only where a spec
needs flags/rounding control.

## Relevant commits

- open-sleigh `76c76b2`: original f80 flag-op prototype.
- open-sleigh `4d33b85`: sibling generic IEEE declarations/macros integrated.
- open-sleigh `c3f54b7`: removes duplicate declarations.
- wazabin-sleigh `25161ec`: pins the integration.
- wazabin-qcode `22f7402`: f80 flag-op prototype.
- wazabin-qcode `14d3c0d`: original design handoff.

The arithmetic commit/status experiments are separate later open-sleigh commits.
