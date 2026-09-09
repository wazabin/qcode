# qcode_vm

The machine layer over the [QCode](https://docs.rs/qcode) interpreter.

[`qcode_emulator`](https://docs.rs/qcode_emulator) evaluates QCode over a
pre-lifted, immutable module: it answers *what does this IR compute*. This
crate adds what a machine needs on top of that, so a guest program can be
**run** rather than merely evaluated:

- **Mapped memory with permissions** — an MMU with pages, `READ`/`WRITE`/`EXEC`
  bits, and a translation cache.
- **Faults as values** — a bad access is delivered to the caller, not raised as
  an abort, so a harness can observe it and carry on.
- **On-demand lifting** — code is discovered and lifted as the guest reaches
  it, so there is no up-front CFG recovery step.
- **Snapshots** — capture and restore machine state.

Blocks are cleaned up as they are lifted (see
[`qcode_passes`](https://docs.rs/qcode_passes)): SLEIGH emits a great deal of
temporary traffic that the interpreter would otherwise re-execute on every pass
over a block.

Execution strategy is pluggable via `set_block_executor`, which is how
[`qcode_jit`](https://docs.rs/qcode_jit) is installed.

Developed by [Thalium](https://blog.thalium.re/about/).

## License

Licensed under the [MIT License](../LICENSE).
