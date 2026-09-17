# hookbench

What instrumentation costs under QCode, Unicorn, icicle-emu and native
execution, on the Embench-IoT images.

Each instrumentation is written the way each engine allows: as IR the
engine compiles along with the guest (`-ir`), and as a host callback the
engine stops for (`-cb`). The harness reports wall time, the number of
times the host was entered on the hook's behalf, the events counted, and
whether the benchmark verified its own result.

| kind | what it does | QCode | icicle | Unicorn | native |
|---|---|---|---|---|---|
| block-ir | `counter += 1` in guest memory at every block entry | IR | p-code injector | — | `-fsanitize-coverage=trace-pc` |
| block-cb | a host callback at every block entry | interrupt | `Op::Hook` | `UC_HOOK_BLOCK` | — |
| insn-ir | `counter += 1` before every guest instruction | IR | p-code injector | — | — |
| insn-cb | a host callback before every guest instruction | interrupt | `Op::Hook` | `UC_HOOK_CODE` | — |
| edge-ir | an AFL-style edge map in guest memory | IR | — | — | trace-pc + map |
| watch-ir | stores to 32 bytes: range check as IR, host on a hit | IR + interrupt | MMU hook | `UC_HOOK_MEM_WRITE` | 4 hardware watchpoints |
| watch-cb | stores to 32 bytes: host at every store | interrupt | (same) | (same) | — |
| cmp-ir | every integer comparison's operands to a ring buffer | IR | — | — | `-fsanitize-coverage=trace-cmp` |
| cmp-cb | every integer comparison's operands to the host | interrupt | — | `UC_HOOK_TCG_OPCODE` (cmp only) | — |

QCode's "block" is the lifted block after absorption, which is the guest's
basic block; its "comparison" is every integer comparison in the p-code,
which for x86 includes every flag computation, so `cmp-*` instruments an
order of magnitude more sites than Unicorn's `cmp`-instruction hook.

## Building

```sh
# The images: benchmarks/embench/build.sh, from an embench-iot checkout.
EMBENCH=~/dev/embench-iot ../embench/build.sh          # -> target/embench (workspace root)
# The native binaries, five variants each.
EMBENCH=~/dev/embench-iot OUT=target/native native/build.sh
# The harness, with the two external engines.
cargo build --release --features unicorn,icicle
```

`unicorn-engine` and `icicle-vm` are path dependencies on checkouts of
the two projects (see `Cargo.toml`); Unicorn builds its QEMU with cmake
and needs `libatomic`. icicle wants Ghidra's processor specifications:
point `GHIDRA_SRC` at a directory holding `Ghidra/Processors` (pypcode's
`processors/` directory, symlinked, does).

## Running

```sh
./target/release/hookbench --engine qcode-jit --instr watch-ir --only crc32 --repeat 5
GHIDRA_SRC=... REPEAT=5 ./run-all.sh      # everything, JSON under target/results
./report.py target/results > RESULTS.md   # the tables
```

`HOOKBENCH_STATS=1` prints the VM's counters after each QCode run and
`HOOKBENCH_DUMP=file` writes the instrumented IR.
