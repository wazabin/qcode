/* mmap/mprotect/munmap: anonymous, hinted, fixed. */
#include "sys.h"

#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define MAP_PRIVATE 2
#define MAP_FIXED 0x10
#define MAP_ANONYMOUS 0x20

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    i64 p = sys6(SYS_mmap, 0, 3 * 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p < 0 || (p & 4095)) { puts_("mmap BAD\n"); return 1; }
    volatile u32 *m = (u32 *)p;
    m[0] = 0x12345678;
    m[3 * 1024 - 1] = 0x9abcdef0;
    if (m[0] != 0x12345678 || m[3 * 1024 - 1] != 0x9abcdef0 || m[500] != 0) { puts_("contents BAD\n"); return 1; }
    /* A second map does not overlap the first. */
    i64 q = sys6(SYS_mmap, 0, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (q < 0 || (q >= p && q < p + 3 * 4096)) { puts_("overlap BAD\n"); return 1; }
    /* Read-only after mprotect: reads still work. */
    if (sys3(SYS_mprotect, p, 4096, PROT_READ) != 0) { puts_("mprotect BAD\n"); return 1; }
    if (m[0] != 0x12345678) { puts_("ro read BAD\n"); return 1; }
    /* mprotect on unmapped memory fails. */
    if (sys3(SYS_mprotect, 0x7000000000UL, 4096, PROT_READ) != -12) { puts_("mprotect unmapped BAD\n"); return 1; }
    /* MAP_FIXED at a chosen address. */
    i64 f = sys6(SYS_mmap, 0x10000000, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (f != 0x10000000) { puts_("fixed BAD\n"); return 1; }
    *(volatile u64 *)f = 7;
    /* A hint that is free is honoured. */
    i64 h = sys6(SYS_mmap, 0x20000000, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (h != 0x20000000) { puts_("hint BAD\n"); return 1; }
    /* A hint that is taken is not. */
    i64 h2 = sys6(SYS_mmap, 0x20000000, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (h2 < 0 || h2 == 0x20000000) { puts_("taken hint BAD\n"); return 1; }
    if (sys2(SYS_munmap, p, 3 * 4096) != 0 || sys2(SYS_munmap, f, 8192) != 0) { puts_("munmap BAD\n"); return 1; }
    /* Unmapped memory can be mapped again at the same place. */
    i64 r = sys6(SYS_mmap, p, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (r != p || m[0] != 0) { puts_("remap BAD\n"); return 1; }
    puts_("mmap ok\n");
    return 0;
}
