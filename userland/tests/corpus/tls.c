/* arch_prctl(ARCH_SET_FS) and %fs-relative addressing. */
#include "sys.h"

#define ARCH_SET_FS 0x1002
#define ARCH_GET_FS 0x1003

static u64 tcb[4] __attribute__((aligned(16)));

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    tcb[0] = (u64)tcb; /* self pointer, as glibc's TCB has */
    tcb[1] = 42;
    tcb[2] = 0xdeadbeef;
    if (sys2(SYS_arch_prctl, ARCH_SET_FS, tcb) != 0) { puts_("set_fs BAD\n"); return 1; }
    u64 self, v1, v2;
    __asm__ volatile("mov %%fs:0, %0" : "=r"(self));
    __asm__ volatile("mov %%fs:8, %0" : "=r"(v1));
    __asm__ volatile("mov %%fs:16, %0" : "=r"(v2));
    if (self != (u64)tcb || v1 != 42 || v2 != 0xdeadbeef) { puts_("fs read BAD\n"); return 1; }
    /* A store through %fs lands in the block. */
    __asm__ volatile("movq $7, %%fs:24" ::: "memory");
    if (tcb[3] != 7) { puts_("fs write BAD\n"); return 1; }
    u64 got = 0;
    if (sys2(SYS_arch_prctl, ARCH_GET_FS, &got) != 0 || got != (u64)tcb) { puts_("get_fs BAD\n"); return 1; }
    if (sys2(SYS_arch_prctl, 0x9999, 0) != -22) { puts_("bad code BAD\n"); return 1; }
    puts_("tls ok\n");
    return 0;
}
