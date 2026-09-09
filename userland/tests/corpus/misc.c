/* The small syscalls a runtime's startup makes, plus cpuid/rdtsc. */
#include "sys.h"

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    i64 ts[2];
    if (sys2(SYS_clock_gettime, 1, ts) != 0 || ts[0] <= 0) { puts_("clock BAD\n"); return 1; }
    if (sys2(SYS_clock_gettime, 0, ts) != 0 || ts[0] < 1600000000) { puts_("realtime BAD\n"); return 1; }
    u8 rnd[16] = {0};
    if (sys3(SYS_getrandom, rnd, 16, 0) != 16) { puts_("getrandom BAD\n"); return 1; }
    int nonzero = 0;
    for (int i = 0; i < 16; i++) nonzero |= rnd[i];
    if (!nonzero) { puts_("random zero BAD\n"); return 1; }
    u64 act[4] = {1, 2, 3, 4}, old[4] = {0};
    if (sys4(SYS_rt_sigaction, 11, act, 0, 8) != 0) { puts_("sigaction BAD\n"); return 1; }
    if (sys4(SYS_rt_sigaction, 11, 0, old, 8) != 0 || old[0] != 1 || old[3] != 4) { puts_("sigaction old BAD\n"); return 1; }
    u64 set = 0xff, oldset = 1;
    if (sys4(SYS_rt_sigprocmask, 0, &set, &oldset, 8) != 0 || oldset != 0) { puts_("sigprocmask BAD\n"); return 1; }
    if (sys4(SYS_rt_sigprocmask, 0, 0, &oldset, 8) != 0 || oldset != 0xff) { puts_("sigprocmask old BAD\n"); return 1; }
    int tid = 0;
    if (sys1(SYS_set_tid_address, &tid) <= 0) { puts_("set_tid BAD\n"); return 1; }
    if (sys4(SYS_rseq, 0, 0, 0, 0) != -38) { puts_("rseq BAD\n"); return 1; }
    u32 word = 5;
    if (sys6(SYS_futex, (i64)&word, 0, 4, 0, 0, 0) != -11) { puts_("futex mismatch BAD\n"); return 1; }
    if (sys6(SYS_futex, (i64)&word, 0, 5, 0, 0, 0) != 0) { puts_("futex match BAD\n"); return 1; }
    if (sys6(SYS_futex, (i64)&word, 1 | 128, 1, 0, 0, 0) != 0) { puts_("futex wake BAD\n"); return 1; }
    if (sys0(9999) != -38) { puts_("unknown BAD\n"); return 1; }
    /* A write from an unmapped buffer is EFAULT, not a crash. */
    if (sys3(SYS_write, 1, 0x7000000000UL, 4) != -14) { puts_("efault BAD\n"); return 1; }
    /* ioctl on a captured stdout is not a tty. */
    u8 ws[8];
    if (sys3(SYS_ioctl, 1, 0x5413, ws) != -25) { puts_("ioctl BAD\n"); return 1; }
    u32 a, b, c, d;
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(0), "c"(0));
    char vendor[13];
    memcpy(vendor, &b, 4);
    memcpy(vendor + 4, &d, 4);
    memcpy(vendor + 8, &c, 4);
    vendor[12] = 0;
    puts_("cpuid: ");
    puts_(vendor);
    puts_("\n");
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(1), "c"(0));
    if (!(d & (1 << 26))) { puts_("no sse2 BAD\n"); return 1; }
    u32 lo1, hi1, lo2, hi2;
    __asm__ volatile("rdtsc" : "=a"(lo1), "=d"(hi1));
    __asm__ volatile("rdtsc" : "=a"(lo2), "=d"(hi2));
    u64 t1 = ((u64)hi1 << 32) | lo1, t2 = ((u64)hi2 << 32) | lo2;
    if (t2 <= t1) { puts_("rdtsc BAD\n"); return 1; }
    puts_("misc ok\n");
    return 0;
}
