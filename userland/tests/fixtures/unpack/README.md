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

# `hello.upx` — a UPX-packed static glibc `hello`

The first real packer. `hello.c` is a static glibc program that writes
`hello\n` with the `write` system call and exits 0; `hello.upx` is that
program packed with UPX. The stub's fold decompresses the whole first
`PT_LOAD` into a `MAP_SHARED` mapping of a `memfd` at the image base, then
maps the file back over it read-only and executable, so the unpacked program
sits byte for byte where the loader would have put it (`0x400000`) and every
byte of it was written by the decompressor's stores.

## Building

```
gcc -static -O2 -o hello hello.c        # GCC 16.2.1, glibc-static 2.43 (Fedora 44)
cp hello hello.upx && upx -q hello.upx  # upx 5.2.1-devel.15+git-6dbfb688, defaults
```

sha256 of the committed `hello.upx`:
`3abd35b48512a57ddc44f4240a3a11381bd130883986015516714f0e8ca28c09`.

The program uses `write`, not `puts`: with this UPX the `puts` build traps in
the stub *natively* (`Trace/breakpoint trap`, exit 133) before any of our
code is involved, so it is no fixture. The `write` build runs natively and
under `userland` alike.

## What the stub needs from `userland`

`memfd_create` (319), `ftruncate` (77), `msync` (26), and `MAP_SHARED` file
mappings whose stores reach the file — `userland` maps files as snapshots and
carries a shared mapping's bytes back at `msync`, at `munmap`, and before the
file is mapped again. Without them the stub falls back to `/dev/shm`, then
`hlt`s.

## Expected

```
$ ./hello.upx; echo $?
hello
0
```

Under the hooks: one generation-1 region of 503,296 bytes at `0x400008`, no
second generation.

### Byte-exact oracle

`unpack.rs` checks the harvested region against the binary that was packed:
bytes `0x8..0x7adfd` of the unpacked `hello` (its whole `R E` segment past
the ELF magic) have sha256
`9400e9e66d468ef4af501831d7cca8a21e3db2440a715c7f99d9abb9767d87bc`, and the
region continues with the stub's 11-byte exit trampoline
(`f3 0f 1e fa 0f 05 5a 58 3e ff e0`). Recompute with
`python3 -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()[8:0x7adfd]).hexdigest())" hello`
after rebuilding `hello` from `hello.c`; a different toolchain gives a
different segment and a different digest, so re-pack and update both.
