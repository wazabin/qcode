/* Freestanding Linux x86-64 runtime for the corpus: raw syscalls, a _start,
 * and just enough formatting to print results. No libc. */
#ifndef SYS_H
#define SYS_H

typedef unsigned long u64;
typedef long i64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

#define SYS_read 0
#define SYS_write 1
#define SYS_open 2
#define SYS_close 3
#define SYS_stat 4
#define SYS_fstat 5
#define SYS_lseek 8
#define SYS_mmap 9
#define SYS_mprotect 10
#define SYS_munmap 11
#define SYS_brk 12
#define SYS_rt_sigaction 13
#define SYS_rt_sigprocmask 14
#define SYS_ioctl 16
#define SYS_pread64 17
#define SYS_writev 20
#define SYS_access 21
#define SYS_dup 32
#define SYS_dup2 33
#define SYS_getpid 39
#define SYS_exit 60
#define SYS_uname 63
#define SYS_getcwd 79
#define SYS_readlink 89
#define SYS_getuid 102
#define SYS_arch_prctl 158
#define SYS_futex 202
#define SYS_getdents64 217
#define SYS_set_tid_address 218
#define SYS_clock_gettime 228
#define SYS_exit_group 231
#define SYS_openat 257
#define SYS_newfstatat 262
#define SYS_readlinkat 267
#define SYS_getrandom 318
#define SYS_rseq 334

static inline i64 sys6(i64 n, i64 a, i64 b, i64 c, i64 d, i64 e, i64 f) {
    i64 ret;
    register i64 r10 __asm__("r10") = d;
    register i64 r8 __asm__("r8") = e;
    register i64 r9 __asm__("r9") = f;
    __asm__ volatile("syscall"
                     : "=a"(ret)
                     : "a"(n), "D"(a), "S"(b), "d"(c), "r"(r10), "r"(r8), "r"(r9)
                     : "rcx", "r11", "memory");
    return ret;
}
#define sys0(n) sys6(n, 0, 0, 0, 0, 0, 0)
#define sys1(n, a) sys6(n, (i64)(a), 0, 0, 0, 0, 0)
#define sys2(n, a, b) sys6(n, (i64)(a), (i64)(b), 0, 0, 0, 0)
#define sys3(n, a, b, c) sys6(n, (i64)(a), (i64)(b), (i64)(c), 0, 0, 0)
#define sys4(n, a, b, c, d) sys6(n, (i64)(a), (i64)(b), (i64)(c), (i64)(d), 0, 0)

static inline u64 strlen_(const char *s) {
    u64 n = 0;
    while (s[n]) n++;
    return n;
}

static inline void write_all(int fd, const void *buf, u64 len) {
    const char *p = buf;
    while (len) {
        i64 n = sys3(SYS_write, fd, p, len);
        if (n <= 0) return;
        p += n;
        len -= n;
    }
}

static inline void puts_(const char *s) { write_all(1, s, strlen_(s)); }

static inline void put_num(i64 v) {
    char buf[24];
    int i = 23;
    buf[i] = 0;
    u64 u = v < 0 ? (u64)(-v) : (u64)v;
    do {
        buf[--i] = '0' + (u % 10);
        u /= 10;
    } while (u);
    if (v < 0) buf[--i] = '-';
    puts_(buf + i);
}

static inline void put_hex(u64 v) {
    char buf[19];
    int i = 18;
    buf[i] = 0;
    do {
        buf[--i] = "0123456789abcdef"[v & 15];
        v >>= 4;
    } while (v);
    buf[--i] = 'x';
    buf[--i] = '0';
    puts_(buf + i);
}

static inline __attribute__((noreturn)) void exit_(int code) {
    sys1(SYS_exit_group, code);
    for (;;) {}
}

/* Freestanding objects: the compiler may still emit these. */
void *memset(void *d, int c, u64 n) {
    u8 *p = d;
    while (n--) *p++ = (u8)c;
    return d;
}
void *memcpy(void *d, const void *s, u64 n) {
    u8 *p = d;
    const u8 *q = s;
    while (n--) *p++ = *q++;
    return d;
}
int memcmp(const void *a, const void *b, u64 n) {
    const u8 *p = a, *q = b;
    for (u64 i = 0; i < n; i++)
        if (p[i] != q[i]) return p[i] - q[i];
    return 0;
}

int main_(int argc, char **argv, char **envp);

void start_c(long *sp) {
    int argc = (int)sp[0];
    char **argv = (char **)(sp + 1);
    char **envp = argv + argc + 1;
    exit_(main_(argc, argv, envp));
}

__asm__(".text\n"
        ".global _start\n"
        "_start:\n"
        "  mov %rsp, %rdi\n"
        "  and $-16, %rsp\n"
        "  call start_c\n"
        "  hlt\n");

#endif
