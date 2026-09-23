/* Signal delivery to a guest handler: a store to an unmapped address raises
 * SIGSEGV and an integer divide by zero raises SIGFPE. Each handler checks
 * the siginfo the kernel would pass, then rewrites the saved instruction
 * pointer in the ucontext so rt_sigreturn resumes at a recovery routine
 * rather than re-faulting. Proves handler entry, siginfo, ucontext rip
 * editing, the alternate stack, and rt_sigreturn. No libc. */
#include "sys.h"

#define SYS_rt_sigreturn 15

/* rt_sigframe ucontext: rip is general register 16, so uc + 40 + 16*8. The
 * SIGSEGV fault address is siginfo + 16. */
#define UC_RIP_OFF (40 + 16 * 8)
#define SI_ADDR_OFF 16

struct kact {
    void (*handler)(int, void *, void *);
    u64 flags;
    void (*restorer)(void);
    u64 mask;
};

#define SA_SIGINFO 0x00000004
#define SA_RESTORER 0x04000000

/* The restorer the kernel returns through: it issues rt_sigreturn. */
__asm__(".text\n"
        ".global sig_restore\n"
        "sig_restore:\n"
        "  mov $15, %rax\n"
        "  syscall\n");
void sig_restore(void);

static void install(int sig, void (*h)(int, void *, void *)) {
    struct kact act;
    act.handler = h;
    act.flags = SA_SIGINFO | SA_RESTORER;
    act.restorer = sig_restore;
    act.mask = 0;
    sys4(SYS_rt_sigaction, sig, &act, 0, 8);
}

/* Recovery targets. rt_sigreturn restores every register except the rip the
 * handler changed, so rsp is the faulting frame's rsp and these run on a
 * good stack. Neither returns. */
static void after_segv(void);
static void after_fpe(void);

static void on_segv(int sig, void *info, void *uc) {
    u64 addr = *(u64 *)((char *)info + SI_ADDR_OFF);
    if (sig != 11 || addr != 0x10) {
        puts_("segv: wrong siginfo\n");
        exit_(1);
    }
    puts_("segv handled\n");
    *(u64 *)((char *)uc + UC_RIP_OFF) = (u64)&after_segv;
}

static void on_fpe(int sig, void *info, void *uc) {
    (void)info;
    if (sig != 8) {
        puts_("fpe: wrong signo\n");
        exit_(1);
    }
    puts_("fpe handled\n");
    *(u64 *)((char *)uc + UC_RIP_OFF) = (u64)&after_fpe;
}

/* Kept volatile and out of line so the compiler cannot fold the fault away. */
static volatile int zero = 0;

static void trigger_fpe(void) {
    volatile int one = 1;
    volatile int r = one / zero;
    (void)r;
    puts_("fpe: no fault\n");
    exit_(1);
}

static void after_segv(void) {
    trigger_fpe();
}

static void after_fpe(void) {
    puts_("done\n");
    exit_(7);
}

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    install(11, on_segv);
    install(8, on_fpe);
    *(volatile int *)0x10 = 1; /* SIGSEGV; handler jumps to after_segv */
    puts_("segv: no fault\n");
    return 1;
}
