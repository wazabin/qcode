# wazabin-qcode-sleigh

High-performance adapter from decoded `wazabin-sleigh` instructions to QCode.

It consumes `Instruction::pcode_ops()`—the flat, Ghidra-style p-code sequence—rather than the SLEIGH source AST. `SleighLifter` caches the specification-derived QCode context (spaces, registers, and user operations); reuse its indexed lifting API and one `AddressIndex` for a full lifting session.

## Binit/Aegis differential tests

`tests/binit.rs` replays Binit's Aegis-generated x86-64 input/output states through the embedded x86-64 specification from `sleigh-precompile`, this lifter, and `qcode_emulator`. Database tests are ignored unless explicitly requested:

```bash
# Start and populate ../binit's PostgreSQL database first.
cargo test -p wazabin-qcode-sleigh --test binit binit_smoke -- --ignored
# Replay the complete corpus.
cargo test -p wazabin-qcode-sleigh --test binit binit_full -- --ignored
```

Set `X86DB_DSN` to use a non-default Binit database. `PCODE_FUZZ_THREADS`, `PCODE_FUZZ_OUTPUT`, and `PCODE_FUZZ_FAIL_QUIETLY` respectively control replay parallelism, the mismatch CSV path, and whether mismatches fail the test.

To report constructor coverage without replaying the emulator:

```bash
cargo test -p wazabin-qcode-sleigh --test binit binit_constructor_coverage -- --ignored --nocapture
```

The metric counts every constructor selected by Binit's concrete opcode cases, including recursive operand-table access, divided by all constructors in the precompiled x86-64 specification. It counts cases independently of their generated input states and separately reports malformed opcodes and decode failures. Set `PCODE_SLEIGH_COVERAGE_OUTPUT` to write a JSON report with per-table totals and a witness opcode for every covered constructor.

A deterministic coverage-guided mutator can propose encodings that reach constructors not currently covered by Binit:

```bash
PCODE_SLEIGH_CANDIDATE_BUDGET=100000 \
  PCODE_SLEIGH_CANDIDATES_OUTPUT=candidates/sleigh_constructor_candidates.json \
  cargo test -p wazabin-qcode-sleigh --test binit \
    binit_constructor_candidates -- --ignored --nocapture
```

Candidates are emitted as JSON records containing a canonical opcode, rendered instruction, complete constructor path, and newly covered locations. Review them, generate suitable Binit initial states, then execute them with Aegis before inserting results into the shared database. Flattened p-code operation coverage is the next follow-up.

CI restores `tests/fixtures/binit-smoke.sql` after creating Binit's schema. It is a small Aegis-compatible corpus, so ordinary CI needs only PostgreSQL—not KVM. Refreshing or expanding hardware results remains a separate job for a KVM-capable runner.

## License

Licensed under the [MIT License](LICENSE).
