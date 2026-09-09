# wazabin-qcode-passes

Block-local cleanup passes over the [QCode](https://docs.rs/wazabin-qcode) IR.

These are the transforms a *lifter* wants while it is still building a
function: cheap, local, and safe to run on a partially discovered CFG. They
need no alias analysis, no calling convention, and no knowledge of the target
architecture — which is what lets them sit below `qcode_analysis` and be used
on their own by consumers that only ever want the cleanup, never the
decompiler.

- `dce` — remove pure instructions nothing reads, and unused parameters of
  blocks with no predecessors.
- `cfg` — merge straight-line blocks, drop empty forwarding blocks, fold
  degenerate branches. `absorb_straight_line` is the incremental variant that
  is safe to run while a lifter holds a position inside the block it is
  extending.

Developed by [Thalium](https://blog.thalium.re/about/).

## License

Licensed under the [MIT License](../LICENSE).
