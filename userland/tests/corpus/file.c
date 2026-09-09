/* Files under the sandbox root: open, fstat, read, lseek, getdents64. */
#include "sys.h"

#define O_RDONLY 0
#define O_DIRECTORY 0200000
#define AT_FDCWD -100

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    i64 fd = sys4(SYS_openat, AT_FDCWD, "/data/input.txt", O_RDONLY, 0);
    if (fd < 0) { puts_("open BAD\n"); return 1; }
    u8 st[144];
    if (sys2(SYS_fstat, fd, st) != 0) { puts_("fstat BAD\n"); return 1; }
    i64 size = *(i64 *)(st + 48);
    u32 mode = *(u32 *)(st + 24);
    puts_("size=");
    put_num(size);
    puts_(" regular=");
    put_num((mode & 0170000) == 0100000);
    puts_("\n");
    char buf[64];
    i64 n = sys3(SYS_read, fd, buf, sizeof buf);
    if (n != size) { puts_("read BAD\n"); return 1; }
    write_all(1, buf, n);
    /* Seek back and read the first word again. */
    if (sys3(SYS_lseek, fd, 0, 0) != 0) { puts_("lseek BAD\n"); return 1; }
    n = sys3(SYS_read, fd, buf, 4);
    write_all(1, buf, n);
    puts_("\n");
    i64 fd2 = sys1(SYS_dup, fd);
    if (fd2 <= fd) { puts_("dup BAD\n"); return 1; }
    if (sys1(SYS_close, fd) != 0 || sys1(SYS_close, fd2) != 0 || sys1(SYS_close, fd) != -9) { puts_("close BAD\n"); return 1; }
    /* Missing files, and the sandbox: .. cannot escape the root. */
    if (sys3(SYS_open, "/data/missing", O_RDONLY, 0) != -2) { puts_("ENOENT BAD\n"); return 1; }
    if (sys3(SYS_open, "/../../../../etc/passwd", O_RDONLY, 0) != -2) { puts_("escape BAD\n"); return 1; }
    /* stat by path. */
    if (sys4(SYS_newfstatat, AT_FDCWD, "/data/input.txt", st, 0) != 0 || *(i64 *)(st + 48) != size) { puts_("newfstatat BAD\n"); return 1; }
    /* The directory listing. */
    i64 dir = sys3(SYS_open, "/data", O_RDONLY | O_DIRECTORY, 0);
    if (dir < 0) { puts_("opendir BAD\n"); return 1; }
    u8 dents[512];
    i64 len = sys3(SYS_getdents64, dir, dents, sizeof dents);
    int entries = 0, saw_input = 0;
    for (i64 off = 0; off < len;) {
        u16 reclen = *(u16 *)(dents + off + 16);
        const char *name = (const char *)(dents + off + 19);
        if (memcmp(name, "input.txt", 10) == 0) saw_input = 1;
        entries++;
        off += reclen;
    }
    if (sys3(SYS_getdents64, dir, dents, sizeof dents) != 0) { puts_("getdents end BAD\n"); return 1; }
    puts_("entries=");
    put_num(entries);
    puts_(saw_input ? " input seen\n" : " input MISSING\n");
    /* /proc/self/exe resolves to something. */
    n = sys3(SYS_readlink, "/proc/self/exe", buf, sizeof buf);
    puts_(n > 0 && buf[0] == '/' ? "exe ok\n" : "exe BAD\n");
    n = sys2(SYS_getcwd, buf, sizeof buf);
    puts_(n == 2 && buf[0] == '/' && buf[1] == 0 ? "cwd ok\n" : "cwd BAD\n");
    return 0;
}
