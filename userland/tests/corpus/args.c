/* argc/argv/envp/auxv layout, and the exit status. */
#include "sys.h"

int main_(int argc, char **argv, char **envp) {
    puts_("argc=");
    put_num(argc);
    puts_("\n");
    for (int i = 0; i < argc; i++) {
        puts_("argv[");
        put_num(i);
        puts_("]=");
        puts_(argv[i]);
        puts_("\n");
    }
    int nenv = 0;
    for (char **e = envp; *e; e++) {
        nenv++;
        if (memcmp(*e, "FOO=", 4) == 0) {
            puts_("FOO is ");
            puts_(*e + 4);
            puts_("\n");
        }
    }
    puts_("nenv=");
    put_num(nenv);
    puts_("\n");
    /* auxv follows the envp terminator */
    u64 *auxv = (u64 *)(envp + nenv + 1);
    u64 pagesz = 0, entry = 0, random = 0, phdr = 0, phnum = 0;
    for (u64 *a = auxv; a[0]; a += 2) {
        if (a[0] == 6) pagesz = a[1];
        if (a[0] == 9) entry = a[1];
        if (a[0] == 25) random = a[1];
        if (a[0] == 3) phdr = a[1];
        if (a[0] == 5) phnum = a[1];
    }
    puts_("pagesz=");
    put_num(pagesz);
    puts_("\n");
    extern char _start;
    puts_(entry == (u64)&_start ? "entry ok\n" : "entry BAD\n");
    puts_(random && ((u8 *)random)[0] | ((u8 *)random)[1] ? "random ok\n" : "random BAD\n");
    /* phdr points at a table whose first entry is a valid p_type */
    u32 p_type = phdr ? *(u32 *)phdr : 0;
    puts_(phnum > 0 && p_type >= 1 && p_type <= 7 ? "phdr ok\n" : "phdr BAD\n");
    return 3;
}
