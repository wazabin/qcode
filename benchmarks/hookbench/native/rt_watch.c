/* A hardware watchpoint through perf, the closest native analogue of a
   memory write hook: four debug registers cover 32 bytes at the data
   symbol the emulated watch starts at (WATCH_SYMBOL, from build.sh). Every write traps into the kernel, which counts
   it; the process itself is not notified. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#ifndef WATCH_SYMBOL
#define WATCH_SYMBOL __bss_start
#endif
extern char WATCH_SYMBOL[];
static int fds[4];
static uint64_t hits;
void rt_setup(void) {
  for (int i = 0; i < 4; i++) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof pe);
    pe.type = PERF_TYPE_BREAKPOINT;
    pe.size = sizeof pe;
    pe.bp_type = HW_BREAKPOINT_W;
    pe.bp_addr = ((uintptr_t)WATCH_SYMBOL & ~7ull) + 8 * i;
    pe.bp_len = HW_BREAKPOINT_LEN_8;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    fds[i] = syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
    if (fds[i] < 0) { perror("perf_event_open"); }
    else ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
  }
}
void rt_teardown(void) {
  hits = 0;
  for (int i = 0; i < 4; i++) {
    if (fds[i] < 0) continue;
    ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);
    uint64_t n = 0;
    if (read(fds[i], &n, sizeof n) == sizeof n) hits += n;
    close(fds[i]);
  }
}
uint64_t rt_events(void) { return hits; }
