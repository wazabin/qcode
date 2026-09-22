# Nanomite support: the ptrace subset `userland` needs

The oracle for this is `corpus/nanomite.c` (test
`a_nanomite_is_steered_by_its_tracer`, `#[ignore]` until this lands). It runs
natively today: stdout `stage 1\nstage 2\nstage 3\ntraced ok\n`, exit `7`.
All ABI numbers below were read from this machine's headers
(`sys/ptrace.h`, `sys/user.h`, `bits/siginfo-consts.h`, `sys/wait.h`) and the
disassembly of `~/dev/vm/nanomites/voracious.bin`.

## The model

A nanomite binary splits control flow across two processes. The tracee's
jumps are replaced by traps; only the tracer holds the map from a trap's
address to the real target. To run one, the emulator's tasks (already
cooperative, one `Vm`, see the scheduler work) must let one task **trace**
another: a trap in the tracee stops it instead of crashing, the stop is
reported to the tracer through `wait`, and the tracer reads and rewrites the
tracee's registers and continues it.

Today (file:line): `rt_sigaction` records handlers but nothing is ever
delivered (`userland/src/syscall.rs:693`); an explicit trap crashes the task
(`userland/src/process.rs:420-460`, `handle_interrupt`); a fatal fault is
`TaskExit::Crashed` with a signal (`process.rs:396`); there is no `ptrace`
(101) and no `waitid` (247). `wait4` (61) reports only exited children
(`syscall.rs:465`).

## Syscalls

| nr | name | needed for |
|---|---|---|
| 101 | `ptrace` | the whole mechanism |
| 247 | `waitid` | reporting a stop, not just an exit |
| 57 | `fork` | already present |
| 61 | `wait4` | already present; extend to report stops |

### `ptrace(request, pid, addr, data)`

The kernel argument order is `rdi=request, rsi=pid, rdx=addr, r10=data`
(syscall arg 4 is `r10`, not `rcx`). `voracious` passes the register buffer in
`r10` with `rdx=0`; `nanomite.c` does the same. Requests actually used:

| request | value | semantics to implement |
|---|---|---|
| `PTRACE_TRACEME` | 0 | the **caller** (the child) marks itself traced by its parent; its next trap/fault becomes a stop delivered to the tracer instead of its default action. `data`/`addr` ignored. Returns 0. |
| `PTRACE_CONT` | 7 | resume the stopped tracee `pid`; `data` is a signal number to deliver, `0` meaning "swallow the trap". Returns 0. The nanomite continues at whatever `rip` SETREGS left. |
| `PTRACE_GETREGS` | 12 | write the tracee's `user_regs_struct` (216 bytes) to guest address `data`. Returns 0, or `-ESRCH` if `pid` is not a stopped tracee of the caller. |
| `PTRACE_SETREGS` | 13 | read a `user_regs_struct` from `data` into the tracee. Returns 0. The tracer redirects control by changing `rip` here. |
| `PTRACE_SEIZE` | 0x4206 | **voracious only** (not `nanomite.c`): attach to an already-running child `pid` without stopping it; `data` is an options mask. The tracee's later traps then stop and report. Needed only when targeting `voracious`, whose children do not call `TRACEME`. |

Errors: `-ESRCH` (3) for a `pid` that is not a live, stopped tracee of the
caller (the guest relies on this: a `GETREGS` after the child exits must
return `-ESRCH`); `-EPERM` (1) for tracing a non-child you have not seized;
`-EIO` (5) for a bad request. Only `-ESRCH` is exercised by the oracle.

A tracee must be **stopped and already waited on** before `GETREGS`/`SETREGS`/
`CONT` operate on it — the same ordering Linux enforces.

### `waitid(idtype, id, infop, options, rusage)`

`nanomite.c` calls `waitid(P_ALL=0, 0, &si, WEXITED|WSTOPPED, 0)`.
`voracious` calls `waitid(P_PID=1, child_pid, &si, WSTOPPED=2, 0)` and loops
until `si_status == SIGTRAP`. Constants: `P_ALL=0 P_PID=1`; `WEXITED=4
WSTOPPED=2 WCONTINUED=8 WNOHANG=1`. Fill the first 128 bytes of `infop`
(`siginfo_t`) and return 0.

`siginfo_t` field offsets (x86-64):

| field | offset | value on a report |
|---|---|---|
| `si_signo` | 0 | `SIGCHLD` (17) |
| `si_code` | 8 | `CLD_EXITED`=1, `CLD_KILLED`=2, `CLD_DUMPED`=3, `CLD_TRAPPED`=4, `CLD_STOPPED`=5 |
| `si_pid` | 16 | the child's pid |
| `si_status` | 24 | on exit: the exit code; on a trap/stop: the signal (`SIGTRAP`=5, `SIGSEGV`=11) |

The guest branches on `si_code`: `CLD_EXITED` ends the loop, `CLD_TRAPPED`
is a nanomite to service.

## `wait4` status word (for the existing call)

`wait4` returns the status as an `int`, not a `siginfo`:

| outcome | encoding | macros |
|---|---|---|
| exited, code `c` | `(c & 0xff) << 8` | `WIFEXITED`, `WEXITSTATUS` |
| killed by signal `s` | `s` (0x01..0x7e) | `WIFSIGNALED`, `WTERMSIG` |
| stopped by signal `s` | `(s << 8) \| 0x7f` | `WIFSTOPPED`, `WSTOPSIG` |

`userland` already produces the exit and signal forms; add the **stopped**
form (`(sig << 8) | 0x7f`) so a tracer using `wait4` instead of `waitid`
sees stops. A SIGTRAP stop is `0x057f`.

## `user_regs_struct` (216 bytes, 27 × u64) → emulator registers

Offsets are byte offsets; index = offset/8. Map each to the name
`userland/src/regs.rs` resolves (`Regs::resolve`): the GPRs by name, the two
segment bases to `FS_OFFSET`/`GS_OFFSET`. `rip` is the one the tracer
rewrites; `orig_rax`, the segment selectors (`cs/ss/ds/es/fs/gs`) and
`eflags` have no backing register in the spec and can be zero on GETREGS and
ignored on SETREGS.

| idx | field | maps to | idx | field | maps to |
|---|---|---|---|---|---|
| 0 | r15 | `R15` | 14 | rdi | `RDI` |
| 1 | r14 | `R14` | 15 | orig_rax | — (0) |
| 2 | r13 | `R13` | 16 | **rip** | machine position |
| 3 | r12 | `R12` | 17 | cs | — (0) |
| 4 | rbp | `RBP` | 18 | eflags | — (0) |
| 5 | rbx | `RBX` | 19 | rsp | `RSP` |
| 6 | r11 | `R11` | 20 | ss | — (0) |
| 7 | r10 | `R10` | 21 | fs_base | `FS_OFFSET` |
| 8 | r9 | `R9` | 22 | gs_base | `GS_OFFSET` |
| 9 | r8 | `R8` | 23 | ds | — (0) |
| 10 | rax | `RAX` | 24 | es | — (0) |
| 11 | rcx | `RCX` | 25 | fs | — (0) |
| 12 | rdx | `RDX` | 26 | gs | — (0) |
| 13 | rsi | `RSI` | | | |

Writing `rip` on SETREGS is a reposition of the task at an instruction
boundary — the VM `position_at` the scheduler work adds. The GPRs are the
register-space image the scheduler already saves per task.

## Where the trap comes from, and the two rip conventions

- **int3** (`0xCC`) raises `SIGTRAP`; the reported `rip` is **one past** the
  byte. The tracer's table is keyed by the trap address, so it computes
  `trap = rip - 1`.
- **a faulting access** (the oracle reads address 0) raises `SIGSEGV`; the
  reported `rip` is **at** the faulting instruction, so `trap = rip`.

Both must be delivered as a stop to the tracer, not a crash, *when the task
is traced*. An untraced task keeps today's behaviour (crash with the signal).
The tracer swallows the signal (`PTRACE_CONT` with `data = 0`) and the task
resumes at the redirected `rip`, so the fault is never re-raised.

## Smallest implementation shape

1. A `traced_by: Option<pid>` and a `stopped: Option<stop_signal>` on `Task`.
2. In `handle_interrupt`/the fault path: if the task is traced, park it as
   stopped with the signal and yield to the scheduler instead of
   `TaskExit::Crashed`.
3. `ptrace` (101): TRACEME sets `traced_by = ppid`; GETREGS/SETREGS read or
   write the target task's register-space image (it is a sibling task in the
   same `Process`); CONT clears `stopped` and reschedules the tracee; return
   `-ESRCH` when the target is not a stopped tracee.
4. `waitid` (247) and the stopped form of `wait4`: report a stopped tracee
   with `si_code = CLD_TRAPPED`, `si_status = signal`.
5. For `voracious` only: `PTRACE_SEIZE` (0x4206) to attach a running child.

`nanomite.c` needs items 1-4; `voracious.bin` additionally needs 5.
