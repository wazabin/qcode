/* x87 arithmetic straight from process start: the control, status and tag
 * words must be what a fresh process gets, or the first fld overflows the
 * stack and every x87 instruction after it does nothing. Built with -mno-sse,
 * so all of this is x87. */
#include "sys.h"

volatile double d5 = 5.0, d3 = 3.0, d0 = 0.0;
volatile long double keep;

int main_(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    long double y = d5, t = d3;
    keep = y;
    const u8 *p = (const u8 *)&keep;
    if (p[9] != 0x40 || p[8] != 0x01 || p[7] != 0xa0) { puts_("fstpt BAD\n"); return 1; }
    if (y * t != 15.0L) { puts_("fmul BAD\n"); return 1; }
    if (y + t != 8.0L || y - t != 2.0L) { puts_("fadd/fsub BAD\n"); return 1; }
    if (!(y / t > 1.6L && y / t < 1.7L)) { puts_("fdiv BAD\n"); return 1; }
    if ((long)(y * 2.5L) != 12) { puts_("fistp BAD\n"); return 1; }
    long double m;
    __asm__("fprem" : "=t"(m) : "0"(y), "u"(t));
    if (m != 2.0L) { puts_("fprem BAD\n"); return 1; }
    long double z = d0;
    if (z != z || y != y) { puts_("nan BAD\n"); return 1; }
    if (!(z < y) || z > y) { puts_("compare BAD\n"); return 1; }
    double back = (double)(y * t);
    if (back != 15.0) { puts_("fstpl BAD\n"); return 1; }
    puts_("fpu ok\n");
    return 0;
}
