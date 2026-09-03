# Embench-IoT under the VM

Real C benchmarks to judge the emulator and the JIT on, in place of the
single-function loops that flattered both.

## Building

```sh
git clone --depth 1 https://github.com/embench/embench-iot.git
EMBENCH=/path/to/embench-iot ./benchmarks/embench/build.sh
cargo test --release -p qcode_jit --test embench -- --ignored --nocapture
```

Images land in `target/embench/`. Embench itself is not vendored — it is
GPL-3.0, and pinning a checkout is the caller's choice.

## What this build is

Freestanding static x86-64: no libc, no startup code, no syscalls. Entered
directly at `main`, which the harness gives a stack and a sentinel return
address, so `main` returning is what ends the run. Its return value — 0 when the
benchmark verified itself — is read out of `RAX`.

`boardsupport.c` supplies the board hooks Embench expects (all no-ops: the VM is
the board, and the harness times the run from the host) along with the handful
of libc routines GCC lowers to regardless of `-ffreestanding`.

## Why SSE is off

The lifter does not yet cover 128-bit storage — `invalid bit range [0, 32] for
128-bit storage` — and GCC will autovectorise these loops given the chance. So
the build passes `-mno-sse -mno-sse2 -mno-mmx -fno-tree-vectorize`. Lifting SSE
is what would let these be built at stock settings.

17 of the 19 benchmarks build. Two do not:

- `wikisort` returns a float in an SSE register, which `-mno-sse` forbids.
- `slre` wants glibc's locale-internal ctype tables (`__ctype_b_loc`).
