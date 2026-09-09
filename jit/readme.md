# wazabin-qcode-jit

A [Cranelift](https://cranelift.dev/) JIT backend for
[QCode](https://docs.rs/wazabin-qcode) — an execution strategy alongside the
interpreter, not a replacement for it.

[`qcode_emulator`](https://docs.rs/wazabin-qcode-emulator) interprets QCode one
operation at a time: every intermediate value is materialised into its value
table and every operand resolved through the module. Compiled code does
neither. A QCode block is already SSA, so it maps onto Cranelift's SSA
directly, and values consumed inside a block stay in machine registers.

The backend is deliberately **partial**. A block it declines to compile is run
by the interpreter instead, so coverage can grow over time without ever
becoming a correctness question. `compile::Unsupported` names what was
declined and why.

Install it on a [`qcode_vm`](https://docs.rs/wazabin-qcode-vm) machine:

```rust
vm.set_block_executor(Box::new(qcode_jit::Jit::new()));
```

The test suite runs programs both ways and requires identical results.

Developed by [Thalium](https://blog.thalium.re/about/).

## License

Licensed under the [MIT License](LICENSE).
