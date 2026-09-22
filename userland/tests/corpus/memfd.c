/* memfd_create, ftruncate, and MAP_SHARED mappings whose stores reach the
 * file: at msync, at munmap, and before the file is mapped again. */
#include "sys.h"

#define SYS_msync 26
#define SYS_ftruncate 77
#define SYS_memfd_create 319
#define PROT_READ 1
#define PROT_WRITE 2
#define MAP_SHARED 1
#define MAP_PRIVATE 2
#define MS_SYNC 4

/* st_size sits at byte 48 of the x86-64 struct stat. */
static i64 size_of(i64 fd) {
    char st[144];
    if (sys2(SYS_fstat, fd, st) != 0) return -1;
    return *(i64 *)(st + 48);
}

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    if (sys2(SYS_memfd_create, "x", 0x100) != -22) { puts_("flags BAD\n"); return 1; }
    i64 fd = sys2(SYS_memfd_create, "upx", 0);
    if (fd < 0) { puts_("memfd BAD\n"); return 1; }
    if (sys3(SYS_write, fd, "abcd", 4) != 4 || size_of(fd) != 4) { puts_("write BAD\n"); return 1; }
    if (sys2(SYS_ftruncate, fd, 8192) != 0 || size_of(fd) != 8192) { puts_("ftruncate BAD\n"); return 1; }

    /* A shared mapping starts with the file's bytes. */
    i64 p = sys6(SYS_mmap, 0, 8192, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p < 0) { puts_("shared mmap BAD\n"); return 1; }
    volatile char *m = (char *)p;
    if (m[0] != 'a' || m[3] != 'd' || m[4] != 0) { puts_("shared contents BAD\n"); return 1; }

    /* Stores reach the file at msync ... */
    char buf[8];
    m[0] = 'A';
    m[4096] = 'Z';
    if (sys3(SYS_msync, p, 8192, MS_SYNC) != 0) { puts_("msync BAD\n"); return 1; }
    if (sys4(SYS_pread64, fd, buf, 4, 0) != 4 || buf[0] != 'A' || buf[1] != 'b') { puts_("msync write-back BAD\n"); return 1; }

    /* ... and at munmap. */
    m[1] = 'B';
    if (sys2(SYS_munmap, p, 8192) != 0) { puts_("munmap BAD\n"); return 1; }
    if (sys4(SYS_pread64, fd, buf, 4, 0) != 4 || buf[1] != 'B') { puts_("munmap write-back BAD\n"); return 1; }

    /* A private mapping sees them, and keeps its own stores to itself. */
    i64 q = sys6(SYS_mmap, 0, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (q < 0) { puts_("private mmap BAD\n"); return 1; }
    volatile char *n = (char *)q;
    if (n[0] != 'A' || n[1] != 'B' || n[4096] != 'Z') { puts_("private view BAD\n"); return 1; }
    n[2] = 'C';
    if (sys2(SYS_munmap, q, 8192) != 0) { puts_("private munmap BAD\n"); return 1; }
    if (sys4(SYS_pread64, fd, buf, 4, 0) != 4 || buf[2] != 'c') { puts_("private leak BAD\n"); return 1; }

    /* A shared mapping written and the file mapped again with no msync in
     * between: the new mapping sees the bytes. */
    i64 r = sys6(SYS_mmap, 0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (r < 0) { puts_("second shared mmap BAD\n"); return 1; }
    ((volatile char *)r)[3] = 'D';
    i64 s = sys6(SYS_mmap, 0, 4096, PROT_READ, MAP_PRIVATE, fd, 0);
    if (s < 0 || ((volatile char *)s)[3] != 'D') { puts_("remap BAD\n"); return 1; }

    /* The descriptor may go; the mapping still reaches the file. */
    if (sys1(SYS_close, fd) != 0) { puts_("close BAD\n"); return 1; }
    ((volatile char *)r)[0] = 'E';
    if (sys3(SYS_msync, r, 4096, MS_SYNC) != 0) { puts_("late msync BAD\n"); return 1; }
    i64 t = sys6(SYS_mmap, 0, 4096, PROT_READ, MAP_PRIVATE, fd, 0);
    if (t != -9) { puts_("closed fd mmap BAD\n"); return 1; }

    puts_("memfd ok\n");
    return 0;
}
