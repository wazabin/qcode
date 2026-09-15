# API documentation

The Rust API documentation for every crate of the workspace is built with
`cargo doc` alongside this book:

- [`qcode`](api/qcode/index.html) — the IR: values, instructions, blocks,
  functions, memory spaces; the `langref` module this reference is rendered
  from.
- [`qcode_passes`](api/qcode_passes/index.html) — block-local cleanup.
- [`qcode_emulator`](api/qcode_emulator/index.html) — the concrete
  interpreter.
- [`qcode_vm`](api/qcode_vm/index.html) — the virtual machine: MMU,
  permissions, faults, on-demand lifting.
- [`qcode_jit`](api/qcode_jit/index.html) — the Cranelift backend.
- [`wazabin_qcode_sleigh`](api/wazabin_qcode_sleigh/index.html) — the SLEIGH
  p-code lifter.
- [`wazabin_qcode_parser`](api/wazabin_qcode_parser/index.html) — the text
  format parser.
- [`wazabin_qcode_macro`](api/wazabin_qcode_macro/index.html) — `qcode!` and
  the `LangRef` derive.

The published crates are on [crates.io](https://crates.io/crates/wazabin-qcode)
and [docs.rs](https://docs.rs/wazabin-qcode).
