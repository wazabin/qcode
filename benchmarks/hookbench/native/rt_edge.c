/* -fsanitize-coverage=trace-pc: an AFL-style edge map keyed on the return
   address, the same shape as the emulated edge-ir. */
#include <stdint.h>
static uint8_t map[65536];
static uint64_t prev;
__attribute__((no_sanitize("coverage"))) void __sanitizer_cov_trace_pc(void) {
  uintptr_t pc = (uintptr_t)__builtin_return_address(0);
  uint64_t cur = ((pc >> 4) ^ (pc >> 12) ^ (pc * 0x9e3779b9ull)) & 0xffff;
  map[(cur ^ prev) & 0xffff]++;
  prev = cur >> 1;
}
uint64_t rt_events(void) { uint64_t n = 0; for (int i = 0; i < 65536; i++) n += map[i] != 0; return n; }
void rt_setup(void) {}
void rt_teardown(void) {}
