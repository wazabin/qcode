# QCode

[![CI](https://github.com/wazabin/qcode/actions/workflows/ci.yml/badge.svg)](https://github.com/wazabin/qcode/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/qcode.svg)](https://crates.io/crates/qcode)

**Decode a binary to p-code, then actually run it — in Rust, with no JVM.**

QCode lifts machine code into a typed, SSA-style IR and executes it: as an
interpreter, or through a Cranelift JIT, behind a virtual machine with mapped
memory, page permissions, faults, and snapshots. Instruction semantics come
from [SLEIGH](https://github.com/wazabin/sleigh) specifications — the same
processor definitions Ghidra uses — compiled natively, so nothing here shells
out to Java.

Developed by [Thalium](https://blog.thalium.re/about/).

## What this is for

Emulating code you cannot or would rather not run natively: unpacking, malware
triage, fuzzing harnesses, firmware for hardware you do not have, and
differential testing of a lifter against real silicon. Because execution is
just a strategy over the IR, the same program can be interpreted for fidelity
or JIT-compiled for speed, and the two are checked against each other.

## Running a program

```rust
use qcode_jit::Jit;
use qcode_vm::{Vm, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

// x86-64 semantics, from a precompiled SLEIGH specification.
let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
let ctx = source.new_context();

let mut memory = VmMemory::new();
memory.mmu.write_unchecked(0x1000, code, perm::READ | perm::EXEC);
memory.mmu.map(0x20000, 0x2000, perm::RW_INIT).unwrap();

let mut vm = Vm::at_address(ctx, 0x1000, source, memory)?;
vm.set_block_executor(Box::new(Jit::new())); // optional; omit to interpret
vm.run(budget);

let rax = vm.emulator().read_varnode_by_name(&ctx, "RAX");
```

Code is discovered and lifted lazily as the guest reaches it, so there is no
up-front CFG recovery step. Faults are delivered as values rather than aborts,
which is what lets a harness observe a bad access instead of dying on it.

## The crates

| Crate | What it is |
| --- | --- |
| [`qcode`](core) | The IR: values, instructions, blocks, functions, memory spaces |
| [`qcode_passes`](passes) | Block-local cleanup — dead code, CFG straight-line merging |
| [`qcode_emulator`](emulator) | Concrete interpreter over a pre-lifted module |
| [`qcode_vm`](vm) | The machine: MMU, permissions, faults, on-demand lifting, snapshots |
| [`qcode_jit`](jit) | Cranelift JIT backend — an execution strategy, not a replacement |
| [`wazabin-qcode-sleigh`](sleigh) | Lifts `wazabin-sleigh` flat p-code into QCode |
| [`wazabin-qcode-parser`](parser) | Parser for the QCode text format |
| [`wazabin-qcode-macro`](macro) | `qcode!` — embed QCode source in Rust |

The JIT is deliberately partial: a block it declines is interpreted instead, so
its coverage can grow without ever becoming a correctness question.

## Writing IR by hand

QCode has a text format, which is how most of the test suite is written:

```rust
use qcode::{context::Context, qcode};

let mut ctx = Context::new();
qcode!(ctx, "%sum = 1 + 2");
```

## Tools

```bash
# Decode one x86-64 instruction to text, SLEIGH AST, flat p-code, and QCode.
cargo run -p wazabin-qcode-sleigh --example qcode-dump -- 4889d8
```

## Correctness

The x86-64 lifter is replayed against **hardware-recorded** input/output states
— real CPU execution captured by [Binit](https://github.com/wazabin/binit) —
and every divergence is a test failure. See [`sleigh/README.md`](sleigh) for
running the corpus and reporting SLEIGH constructor coverage.

## Development

```bash
cargo test --workspace --all-features
```

The workspace uses the sibling `wazabin-sleigh`, `wazabin-pcode`,
`wazabin-binary`, and `jstd` checkouts declared in `Cargo.toml`.

## License

Licensed under the [MIT License](LICENSE).
