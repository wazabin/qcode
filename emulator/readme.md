# wazabin-qcode-emulator

A concrete interpreter for the [QCode](https://docs.rs/wazabin-qcode) IR.

It executes QCode against concrete machine state over a pre-lifted, immutable
module — it answers *what does this IR compute*. Use it to verify that lifted
code faithfully reproduces the original binary's semantics, and as the
reference execution strategy that
[`qcode_jit`](https://docs.rs/wazabin-qcode-jit) is checked against.

For mapped memory, page permissions, faults, on-demand lifting and snapshots —
everything needed to *run* a guest program rather than evaluate a module — see
[`qcode_vm`](https://docs.rs/wazabin-qcode-vm), which layers those on top of this
crate.

Floating point is evaluated with [`rustc_apfloat`](https://docs.rs/rustc_apfloat),
including correctly rounded 80-bit x87 extended precision, so results match
hardware rather than the host's `f64`.

Developed by [Thalium](https://blog.thalium.re/about/).

## License

Licensed under the [MIT License](../LICENSE).
