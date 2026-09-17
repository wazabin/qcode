/* -fsanitize-coverage=trace-cmp: the compiler calls these at every integer
   comparison with the operands. Logged to a ring buffer like cmp-ir. */
#include <stdint.h>
static uint64_t log_[4096][2];
static uint64_t idx;
#define LOG(a, b) do { log_[idx & 4095][0] = (a); log_[idx & 4095][1] = (b); idx++; } while (0)
__attribute__((no_sanitize("coverage"))) void __sanitizer_cov_trace_cmp1(uint8_t a, uint8_t b) { LOG(a, b); }
__attribute__((no_sanitize("coverage"))) void __sanitizer_cov_trace_cmp2(uint16_t a, uint16_t b) { LOG(a, b); }
__attribute__((no_sanitize("coverage"))) void __sanitizer_cov_trace_cmp4(uint32_t a, uint32_t b) { LOG(a, b); }
__attribute__((no_sanitize("coverage"))) void __sanitizer_cov_trace_cmp8(uint64_t a, uint64_t b) { LOG(a, b); }
__attribute__((no_sanitize("coverage"))) void __sanitizer_cov_trace_const_cmp1(uint8_t a, uint8_t b) { LOG(a, b); }
__attribute__((no_sanitize("coverage"))) void __sanitizer_cov_trace_const_cmp2(uint16_t a, uint16_t b) { LOG(a, b); }
__attribute__((no_sanitize("coverage"))) void __sanitizer_cov_trace_const_cmp4(uint32_t a, uint32_t b) { LOG(a, b); }
__attribute__((no_sanitize("coverage"))) void __sanitizer_cov_trace_const_cmp8(uint64_t a, uint64_t b) { LOG(a, b); }
__attribute__((no_sanitize("coverage"))) void __sanitizer_cov_trace_switch(uint64_t v, uint64_t *cases) { LOG(v, cases[0]); }
uint64_t rt_events(void) { return idx; }
void rt_setup(void) { idx = 0; }
void rt_teardown(void) {}
