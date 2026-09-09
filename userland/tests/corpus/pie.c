/* Built as a static PIE: everything is RIP-relative and the loader picks the
 * base. Reports where it was loaded. */
#include "sys.h"

static int counter = 5;

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    counter += 1;
    puts_("pie counter=");
    put_num(counter);
    puts_("\n");
    /* The image sits at the PIE base, not at a fixed address. */
    puts_(((u64)&counter >> 32) == 0x5555 ? "base ok\n" : "base BAD\n");
    return 0;
}
