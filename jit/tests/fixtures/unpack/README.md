# `selfdecrypt` — two-generation self-decrypting fixture

A tiny static, libc-less x86-64 Linux ELF that produces two generations of
dynamically generated code. It is the workhorse fixture for the unpacker POC
(`jit/tests/unpack.rs`): the VM maps its `PT_LOAD` segments, runs it, and the
provenance hooks must observe code being written and then executed.

## Layers

| stage | where it runs | what it does |
|---|---|---|
| **stage 0** | `_start`, in `.text` | XOR-decrypts (key `0x5A`) the embedded encrypted stage 1 **in place** inside the RWX `.rwx` section, then `jmp`s into it. |
| **stage 1** | the RWX `.rwx` buffer (1st generation) | `mmap`s a fresh `RWX` page, XOR-decrypts (key `0xA7`) the embedded encrypted stage 2 into it, then `jmp`s into the mapping. |
| **stage 2** | the anonymous `mmap` page (2nd generation) | `write(1, "unpacked: stage 2\n", 18)` then `exit_group(7)`. |

Stages 1 and 2 are hand-written position-independent assembly (RIP-relative or
register-based only), so stage 2 runs correctly wherever `mmap` places it.

## Expected output

```
$ ./selfdecrypt; echo $?
unpacked: stage 2
7
```

## Segments (`readelf -l selfdecrypt`)

Two `PT_LOAD` segments, and nothing else the loader must handle:

- `LOAD  R E`  at `0x400000` — `.text` (stage 0 / `_start`).
- `LOAD  RWE`  at `0x401000` — `.rwx`, the writable+executable buffer holding
  the (initially encrypted) stage-1 image. This is the segment that must be
  `PF_R|PF_W|PF_X`; the VM maps `PT_LOAD` with the ELF flags, so the self-
  modification in stage 0 needs it writable *and* executable.

The `ld.bfd` "LOAD segment with RWX permissions" warning during the build is
expected and desired.

## Syscalls used

Raw `syscall` only, no libc. Exactly three, by x86-64 number:

| syscall | nr | stage |
|---|---|---|
| `mmap` | 9 | stage 1 |
| `write` | 1 | stage 2 |
| `exit_group` | 231 | stage 2 |

Stage 0 issues no syscalls.

## XOR keys and stage sizes

| item | value |
|---|---|
| stage-1 key (stage 0 → stage 1) | `0x5A` |
| stage-2 key (stage 1 → stage 2) | `0xA7` |
| stage-1 plaintext blob | 127 bytes |
| stage-2 plaintext blob | 54 bytes |

## Rebuilding

```
./build.sh
```

The build is reproducible: fixed keys, `--build-id=none`, no timestamps, and a
final `strip -s`. Building twice yields byte-identical `selfdecrypt`
(sha256 `952d33454c2b0429314e8808b9b83d2af2c62e7a10c076fdef4673cd6633dfb7`).
`build.sh` removes all intermediates, leaving only the sources, the script, and
the committed `selfdecrypt` binary.

Sources:
- `stage0.S` — `_start`; decrypts stage 1 in place, `.incbin`s `stage1.enc`
  into the RWX `.rwx` section.
- `stage1.S` — mmap + decrypt loop; `.incbin`s the encrypted `stage2.enc`.
- `stage2.S` — the payload.

Toolchain: `gcc`, `objcopy`, `strip`, `python3` (the XOR step). Built with GCC
16.2.1 / binutils on Fedora; any recent GNU toolchain reproduces the same
bytes given the same compiler and linker versions.
