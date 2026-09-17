/* Native harness for the Embench benchmarks: the same sources, the same
   flags as the emulated images, run in-process and timed per iteration.

   Prints one line: "<name> <variant> <min_ns> <verified> <events>". */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "support.h"

void initialise_board(void) {}
void start_trigger(void) {}
void stop_trigger(void) {}

/* Instrumentation runtimes. Each defines rt_setup/rt_teardown and reports
   the events it counted. */
uint64_t rt_events(void);
void rt_setup(void);
void rt_teardown(void);

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

int main(int argc, char **argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 50;
  const char *name = argc > 2 ? argv[2] : "?";
  const char *variant = argc > 3 ? argv[3] : "?";
  initialise_benchmark();
  warm_caches(WARMUP_HEAT);
  rt_setup();
  uint64_t best = UINT64_MAX;
  volatile int result = 0;
  for (int i = 0; i < reps; i++) {
    uint64_t t0 = now_ns();
    result = benchmark();
    uint64_t t1 = now_ns();
    if (t1 - t0 < best) best = t1 - t0;
  }
  rt_teardown();
  int correct = verify_benchmark(result);
  printf("%s %s %llu %d %llu\n", name, variant, (unsigned long long)best, correct,
         (unsigned long long)rt_events());
  return !correct;
}
