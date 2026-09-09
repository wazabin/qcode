/* A crash: the environment must report it, not die. */
#include "sys.h"

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    puts_("about to fault\n");
    *(volatile int *)0x10 = 1;
    puts_("survived?!\n");
    return 0;
}
