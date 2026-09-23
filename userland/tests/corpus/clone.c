/* glibc issues fork through clone with the flags below, not the bare SIGCHLD
 * the environment used to require. This exercises that path: a fork-like
 * clone with CHILD_SETTID and CHILD_CLEARTID, the child exits with a known
 * status, and the parent reaps it with wait4 and decodes the status. No
 * libc. */
#include "sys.h"

#define SYS_clone 56
#define SYS_wait4 61

/* CLONE_CHILD_CLEARTID | CLONE_CHILD_SETTID | SIGCHLD, the glibc fork mask. */
#define CLONE_FORK 0x01200011

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;

    volatile long ctid = -1;
    /* clone(flags, child_stack=0, parent_tid=0, child_tid=&ctid, tls=0):
     * a zero stack means the child runs on a copy of the parent's, as fork
     * does. */
    long rc = sys6(SYS_clone, CLONE_FORK, 0, 0, (i64)&ctid, 0, 0);

    if (rc == 0) {
        /* Child: a copy-on-write clone of the parent, exiting with a status
         * the parent will decode. */
        exit_(5);
    }

    if (rc < 0) {
        puts_("clone failed\n");
        return 1;
    }

    long status = 0;
    long w = sys4(SYS_wait4, rc, (i64)&status, 0, 0);
    if (w != rc) {
        puts_("wait4 wrong pid\n");
        return 1;
    }
    /* WIFEXITED and WEXITSTATUS. */
    if ((status & 0x7f) != 0 || ((status >> 8) & 0xff) != 5) {
        puts_("bad child status\n");
        return 1;
    }
    puts_("clone ok\n");
    return 0;
}
