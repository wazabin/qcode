/* The program break: grow, use, shrink. */
#include "sys.h"

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    u64 start = sys1(SYS_brk, 0);
    if (start == 0) { puts_("brk(0) BAD\n"); return 1; }
    /* Growing below the start is refused: brk stays. */
    if ((u64)sys1(SYS_brk, start - 4096) != start) { puts_("shrink-below BAD\n"); return 1; }
    u64 end = sys1(SYS_brk, start + 65536);
    if (end != start + 65536) { puts_("grow BAD\n"); return 1; }
    volatile u8 *heap = (u8 *)start;
    heap[0] = 0xaa;
    heap[65535] = 0x55;
    if (heap[0] != 0xaa || heap[65535] != 0x55 || heap[100] != 0) { puts_("heap BAD\n"); return 1; }
    /* A second grow keeps the old contents. */
    if ((u64)sys1(SYS_brk, start + 131072) != start + 131072) { puts_("regrow BAD\n"); return 1; }
    if (heap[0] != 0xaa) { puts_("regrow contents BAD\n"); return 1; }
    heap[131071] = 1;
    if ((u64)sys1(SYS_brk, start + 4096) != start + 4096) { puts_("shrink BAD\n"); return 1; }
    if (heap[0] != 0xaa) { puts_("shrink contents BAD\n"); return 1; }
    puts_("brk ok\n");
    return 0;
}
