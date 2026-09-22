/* A self-contained nanomite: the same shape as ~/dev/vm/nanomites/voracious.
 *
 * The parent forks a child, the child places itself under the parent with
 * PTRACE_TRACEME, and the child's control flow is cut at two points by a
 * deliberate trap instead of a jump. Each trap stops the child and is
 * reported to the parent through waitid; the parent reads the child's
 * registers, looks the trap's address up in a nanomite table, writes the
 * real target into the child's rip, and continues it. The child never holds
 * the addresses of its own next stages: only the tracer does.
 *
 * Two trap kinds, one each, exactly as a real nanomite mixes them:
 *   - int3 (0xCC) raises SIGTRAP; the reported rip is one byte past it.
 *   - a read of address 0 raises SIGSEGV; the reported rip is at the load.
 *
 * The child prints "stage 1/2/3" as it is steered through them and exits 7;
 * the parent verifies both redirects happened, that a GETREGS on the exited
 * child returns ESRCH, and exits with the child's code. Oracle:
 *   stdout: "stage 1\nstage 2\nstage 3\ntraced ok\n"
 *   exit:   7
 */
#include "sys.h"

#define SYS_fork 57
#define SYS_kill 62
#define SYS_ptrace 101
#define SYS_waitid 247

/* ptrace requests (asm/ptrace-abi.h). */
#define PTRACE_CONT 7
#define PTRACE_GETREGS 12
#define PTRACE_SETREGS 13

/* waitid (sys/wait.h). */
#define P_ALL 0
#define WEXITED 4
#define WSTOPPED 2
#define CLD_EXITED 1
#define CLD_TRAPPED 4

#define SIGTRAP 5
#define SIGSEGV 11

/* The child body, in assembly so the trap sites and stage entries are named
 * symbols at fixed addresses the parent can put in its table. Raw syscalls
 * only; it never returns. */
extern void child_body(void);
extern char nm_trap1[], nm_stage2[], nm_trap2[], nm_stage3[];
__asm__(
    ".text\n"
    ".globl child_body, nm_trap1, nm_stage2, nm_trap2, nm_stage3\n"
    "child_body:\n"
    "  xor %edi, %edi\n"        /* PTRACE_TRACEME = 0 */
    "  xor %esi, %esi\n"
    "  xor %edx, %edx\n"
    "  xor %r10d, %r10d\n"
    "  mov $101, %eax\n"        /* ptrace */
    "  syscall\n"
    "  mov $1, %eax\n"          /* write(1, "stage 1\n", 8) */
    "  mov $1, %edi\n"
    "  lea nm_msg1(%rip), %rsi\n"
    "  mov $8, %edx\n"
    "  syscall\n"
    "nm_trap1:\n"
    "  int3\n"                  /* SIGTRAP; parent redirects rip to nm_stage2 */
    "  mov $1, %eax\n"          /* fall-through only if the redirect failed */
    "  mov $1, %edi\n"
    "  lea nm_bad(%rip), %rsi\n"
    "  mov $4, %edx\n"
    "  syscall\n"
    "  mov $1, %edi\n"
    "  mov $231, %eax\n"        /* exit_group(1) */
    "  syscall\n"
    "nm_stage2:\n"
    "  mov $1, %eax\n"          /* write(1, "stage 2\n", 8) */
    "  mov $1, %edi\n"
    "  lea nm_msg2(%rip), %rsi\n"
    "  mov $8, %edx\n"
    "  syscall\n"
    "nm_trap2:\n"
    "  mov 0, %rax\n"           /* SIGSEGV at 0; parent redirects to nm_stage3 */
    "  mov $1, %eax\n"          /* fall-through only if the redirect failed */
    "  mov $1, %edi\n"
    "  lea nm_bad(%rip), %rsi\n"
    "  mov $4, %edx\n"
    "  syscall\n"
    "  mov $1, %edi\n"
    "  mov $231, %eax\n"        /* exit_group(1) */
    "  syscall\n"
    "nm_stage3:\n"
    "  mov $1, %eax\n"          /* write(1, "stage 3\n", 8) */
    "  mov $1, %edi\n"
    "  lea nm_msg3(%rip), %rsi\n"
    "  mov $8, %edx\n"
    "  syscall\n"
    "  mov $7, %edi\n"
    "  mov $231, %eax\n"        /* exit_group(7): the child's success code */
    "  syscall\n"
    ".section .rodata\n"
    "nm_msg1: .ascii \"stage 1\\n\"\n"
    "nm_msg2: .ascii \"stage 2\\n\"\n"
    "nm_msg3: .ascii \"stage 3\\n\"\n"
    "nm_bad:  .ascii \"BAD\\n\"\n"
    ".text\n");

/* x86-64 user_regs_struct is 27 u64 (216 bytes); rip is field 16. */
#define REG_RIP 16
typedef unsigned long regs_t[27];

static i64 ptrace_(i64 req, i64 pid, i64 addr, i64 data) {
    return sys4(SYS_ptrace, req, pid, addr, data);
}

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;

    /* trap instruction address -> the real next stage. The parent normalizes
     * the reported rip (one past the int3, at the faulting load) back to the
     * trap instruction below, so both keys are the bare trap label. */
    struct { u64 at; u64 target; } table[2] = {
        { (u64)nm_trap1, (u64)nm_stage2 },
        { (u64)nm_trap2, (u64)nm_stage3 },
    };

    i64 pid = sys0(SYS_fork);
    if (pid < 0) { puts_("fork BAD\n"); return 1; }
    if (pid == 0) { child_body(); return 1; /* unreachable */ }

    int redirects = 0;
    for (;;) {
        /* siginfo_t: si_code at 8, si_pid at 16, si_status at 24. */
        u8 si[128];
        memset(si, 0, sizeof si);
        if (sys6(SYS_waitid, P_ALL, 0, (i64)si, WEXITED | WSTOPPED, 0, 0) != 0) {
            puts_("waitid BAD\n");
            return 1;
        }
        int code = *(int *)(si + 8);
        int status = *(int *)(si + 24);
        if (code == CLD_EXITED) {
            /* The child is gone; GETREGS on it must fail with ESRCH. */
            regs_t dead;
            if (ptrace_(PTRACE_GETREGS, pid, 0, (i64)dead) != -3) {
                puts_("ESRCH BAD\n");
                return 1;
            }
            if (redirects != 2 || status != 7) {
                puts_("stages BAD\n");
                return 1;
            }
            puts_("traced ok\n");
            return status;
        }
        if (code != CLD_TRAPPED) { puts_("code BAD\n"); return 1; }

        regs_t regs;
        if (ptrace_(PTRACE_GETREGS, pid, 0, (i64)regs) != 0) {
            puts_("getregs BAD\n");
            return 1;
        }
        u64 rip = regs[REG_RIP];
        u64 trap = (status == SIGTRAP) ? rip - 1 : rip;
        u64 target = 0;
        for (int i = 0; i < 2; i++)
            if (table[i].at == trap) target = table[i].target;
        if (target == 0) { puts_("table BAD\n"); return 1; }

        regs[REG_RIP] = target;
        if (ptrace_(PTRACE_SETREGS, pid, 0, (i64)regs) != 0) {
            puts_("setregs BAD\n");
            return 1;
        }
        redirects++;
        /* Continue, swallowing the trap (signal 0) so it is not delivered. */
        if (ptrace_(PTRACE_CONT, pid, 0, 0) != 0) {
            puts_("cont BAD\n");
            return 1;
        }
    }
}
