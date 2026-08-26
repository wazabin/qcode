# QCode

Typed SSA-style p-code IR, analysis passes, emulator, parser, and SLEIGH adapter used by Harbinger.

## Development

Run every workspace test with:

```bash
cargo test --workspace --all-features
```

The workspace uses the sibling `wazabin-*`, `jstd`, and `wazabin` checkouts declared in `Cargo.toml`.
