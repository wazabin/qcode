/* Shared anonymous memory across fork: a MAP_SHARED|MAP_ANONYMOUS page is
 * one page for parent and child, a MAP_PRIVATE one is copied. The child
 * writes both and exits; the parent waits and reads them. */
#include "sys.h"

#define SYS_fork 57
#define SYS_wait4 61
#define PROT_RW 3
#define MAP_SHARED 0x1
#define MAP_PRIVATE 0x2
#define MAP_ANONYMOUS 0x20

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    i64 s = sys6(SYS_mmap, 0, 4096, PROT_RW, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    i64 p = sys6(SYS_mmap, 0, 4096, PROT_RW, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (s < 0 || p < 0) { puts_("mmap BAD\n"); return 1; }
    volatile u32 *shared = (u32 *)s;
    volatile u32 *private_ = (u32 *)p;
    shared[0] = 1;
    private_[0] = 1;
    i64 pid = sys0(SYS_fork);
    if (pid < 0) { puts_("fork BAD\n"); return 1; }
    if (pid == 0) {
        /* The child sees the parent's values, then changes both. */
        if (shared[0] != 1 || private_[0] != 1) { puts_("child view BAD\n"); sys1(SYS_exit, 9); }
        shared[0] = 0x1234;
        shared[1000] = 0x77;
        private_[0] = 0x5678;
        sys1(SYS_exit, 3);
    }
    int status = 0;
    if (sys4(SYS_wait4, pid, (i64)&status, 0, 0) != pid) { puts_("wait BAD\n"); return 1; }
    if ((status >> 8) != 3) { puts_("status BAD\n"); return 1; }
    if (shared[0] != 0x1234 || shared[1000] != 0x77) { puts_("shared BAD\n"); return 1; }
    if (private_[0] != 1) { puts_("private leaked BAD\n"); return 1; }
    puts_("shared ok\n");
    return 0;
}
