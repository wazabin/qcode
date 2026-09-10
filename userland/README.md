# wazabin-qcode-userland

A Linux x86-64 userland environment on top of the [`qcode_vm`](https://github.com/wazabin/qcode/tree/main/vm)
emulator: it loads a static ELF executable, builds the SysV process stack, and
services the program's Linux system calls on the host. The guest runs either on
the QCode interpreter or on the Cranelift JIT.

```
cargo run --bin run-elf -- [--jit] [--trace] [--budget N] [--root DIR] [--env K=V]... PROG [ARGS...]
```

`run-elf` exits with the guest's status. A crash prints the guest `pc`, the
fault and a register dump, and exits with `128 + signal`. `--trace` prints one
`strace`-style line per system call to stderr; `--root DIR` resolves every
guest path under `DIR` (with `..` unable to climb above it); `--budget N`
stops after `N` p-code operations.

As a library:

```rust
use qcode_userland::{Config, Process, ProcessExit, fs::Stdio};

let image = std::fs::read("prog")?;
let config = Config { stdio: Stdio::Captured, ..Config::default() };
let mut process = Process::new(&image, config)?;
match process.run(u64::MAX) {
    ProcessExit::Exited(code) => println!("{code}: {:?}", process.files.stdout()),
    ProcessExit::Crashed(crash) => eprintln!("{crash}"),
    ProcessExit::Budget => eprintln!("out of budget"),
}
```

## How system calls are intercepted

The `syscall` instruction lifts, in the SLEIGH x86 specification, to a
user-defined p-code operation (`define pcodeop syscall`). The QCode emulator
has no semantics for it, so `Vm::run` returns `VmExit::Interrupt` naming the
operation, with the machine positioned **at** it: everything before it in the
block has retired and nothing after it — including the block's terminator —
has run. This is the same exit an explicit `vm.interrupt` op raises, and it
holds with and without the JIT: compiled code runs the prefix of the block
natively and hands the op to the interpreter.

To resume, the environment supplies the operation's effect and calls
`Vm::resume(value)`:

- `syscall` produces no value: the environment writes `RAX` and resumes with
  `None`.
- `rdtsc` (an `i64`) and the `cpuid_*` family (an `i128` packed as
  `EAX | EBX<<32 | EDX<<64 | ECX<<96`) produce a value, handed to `resume`,
  which files it under the op's instruction so the register stores that
  follow it in the same block read it. `rdtsc` counts retired operations
  (reproducible); `cpuid` reports a baseline "GenuineIntel" with SSE2 and
  nothing newer.
- `ud2` surfaces as the `invalidInstructionException` op and is reported as a
  crash with signal 4.

## What is supported

- **Loader**: ELF64 x86-64, `ET_EXEC` and static `ET_DYN` (static PIE, placed
  at `0x5555_5555_0000`). Per-segment permissions, page-aligned mappings,
  zeroed `.bss`; `AT_PHDR`/`AT_PHNUM`/`AT_PHENT` from `PT_PHDR` or the covering
  `PT_LOAD`; `PT_TLS` is recorded. No relocation processing — a static PIE
  must apply its own `R_X86_64_RELATIVE` relocations (glibc's does; a
  freestanding one must avoid absolute pointers in data).
- **Stack**: 8 MiB below `0x7fff_f000_0000`, with `argc`, `argv`, `envp`, and an
  auxiliary vector (`AT_PHDR`, `AT_PHENT`, `AT_PHNUM`, `AT_PAGESZ`, `AT_BASE`,
  `AT_FLAGS`, `AT_ENTRY`, `AT_UID/EUID/GID/EGID`, `AT_PLATFORM`, `AT_HWCAP`,
  `AT_HWCAP2`, `AT_CLKTCK`, `AT_SECURE=0`, `AT_RANDOM`, `AT_EXECFN`,
  `AT_NULL`), 16-byte aligned.
- **System calls**: `read`, `write`, `readv`, `writev`, `pread64`, `pwrite64`,
  `open`, `openat`, `close`, `lseek`, `stat`, `lstat`, `fstat`, `newfstatat`
  (including `AT_EMPTY_PATH`), `access`, `faccessat`, `faccessat2`,
  `readlink`, `readlinkat` (`/proc/self/exe` resolves to the program), `getcwd`,
  `chdir`, `dup`, `dup2`, `dup3`, `fcntl` (`F_DUPFD`, `F_DUPFD_CLOEXEC`,
  `F_GETFD`, `F_SETFD`, `F_GETFL`, `F_SETFL`), `getdents64`, `ioctl`
  (`TCGETS`, `TIOCGWINSZ` when the host descriptor is a terminal, else
  `ENOTTY`), `brk`, `mmap` (anonymous and file-backed, `MAP_FIXED`,
  `MAP_FIXED_NOREPLACE`, hints), `munmap`, `mprotect`, `madvise`,
  `arch_prctl` (`ARCH_SET_FS`/`ARCH_GET_FS` on the SLEIGH `FS_OFFSET`
  register, and `GS`), `set_tid_address`, `set_robust_list`, `rseq`
  (`ENOSYS`), `futex` (`WAIT` is `EAGAIN` on mismatch and returns at once
  otherwise; `WAKE` wakes nobody), `rt_sigaction` (stored and returned, never
  delivered), `rt_sigprocmask`, `sigaltstack`, `kill`/`tgkill` to self,
  `nanosleep`, `clock_nanosleep`, `sched_yield` (no-ops), `getpid`,
  `getppid`, `gettid`, `getuid`, `geteuid`, `getgid`, `getegid`, `uname`,
  `getrandom` (deterministic), `clock_gettime`, `clock_getres`,
  `gettimeofday`, `time`, `prlimit64`, `exit`, `exit_group`. Anything else
  returns `-ENOSYS` and logs a warning.
- **Files**: stdin/stdout/stderr on the host's, or captured to buffers
  (`Stdio::Captured`) for harnesses. Other descriptors are host files and
  directories, optionally under a sandbox root.
- Every copy to or from guest memory goes through the MMU; an unmapped or
  unwritable buffer yields `-EFAULT`.

## Limitations

- **No SSE, so no glibc.** The emulator does not lift SSE, and every static
  glibc uses it in start-up code (`memcpy`, `strlen`, IFUNC selection). The
  corpus therefore uses a freestanding runtime with raw `syscall` wrappers
  (`tests/corpus/sys.h`); build guests with `-nostdlib -nostartfiles -mno-sse
  -mno-sse2 -mno-mmx`.
- Single thread: no `clone`, no `fork`/`execve`; `futex` never blocks.
- Signals are recorded but never delivered; a fault or `ud2` ends the process
  with a diagnostic instead.
- `mmap` of a file is a private snapshot: writes never reach the host file.
- `pipe` and sockets are not provided.
- `int 0x80` is not handled (the emulator panics on it); use `syscall`.
- x86-64 Linux only.

## Tests

```
cargo test
```

`tests/corpus.rs` builds the C programs in `tests/corpus/` with the host `gcc`
(skipping if there is none) and runs each interpreted and with the JIT,
checking stdout and the exit status. One is built as a static PIE.

The Embench harnesses live here too, on the `bare` module, which runs a
freestanding image with no process around it: `tests/embench.rs` verifies
every benchmark under both strategies, `tests/divergence.rs` finds the first
block a compiled run computes differently, and `tests/guest_rate_probe.rs`
reports guest instructions per second. All three are ignored by default and
need the images from `benchmarks/embench/build.sh`.
