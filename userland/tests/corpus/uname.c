/* uname, and the identity syscalls. */
#include "sys.h"

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    char uts[65 * 6];
    if (sys1(SYS_uname, uts) != 0) { puts_("uname BAD\n"); return 1; }
    puts_(uts);
    puts_(" ");
    puts_(uts + 65 * 4);
    puts_("\n");
    puts_("pid>0=");
    put_num(sys0(SYS_getpid) > 0);
    puts_(" uid=");
    put_num(sys0(SYS_getuid));
    puts_("\n");
    return 0;
}
