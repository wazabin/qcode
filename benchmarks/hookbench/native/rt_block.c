/* -fsanitize-coverage=trace-pc: the compiler calls this at every basic block. */
#include <stdint.h>
static uint64_t counter;
__attribute__((no_sanitize("coverage"))) void __sanitizer_cov_trace_pc(void) { counter++; }
uint64_t rt_events(void) { return counter; }
void rt_setup(void) { counter = 0; }
void rt_teardown(void) {}
