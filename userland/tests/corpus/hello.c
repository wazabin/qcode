/* The first program: write, then exit with a status. */
#include "sys.h"

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    puts_("hi\n");
    return 0;
}
