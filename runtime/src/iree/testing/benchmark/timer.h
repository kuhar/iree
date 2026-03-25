// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_BENCHMARK_TIMER_H_
#define IREE_TESTING_BENCHMARK_TIMER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Timer abstraction
//===----------------------------------------------------------------------===//

// A pluggable timer. Custom timers (TSC, GPU timers) can implement this.
typedef struct {
  uint64_t (*now_ns)(void* user_data);
  void* user_data;
} iree_bench_timer_t;

// Returns a wall-clock timer using iree_time_now() (steady/monotonic clock).
iree_bench_timer_t iree_bench_timer_wall(void);

// Returns a CPU-thread timer (measures only the calling thread's CPU time).
// Falls back to wall clock on platforms without thread CPU time.
iree_bench_timer_t iree_bench_timer_cpu_thread(void);

// Returns a CPU-process timer (measures total process CPU time).
// Falls back to wall clock on platforms without process CPU time.
iree_bench_timer_t iree_bench_timer_cpu_process(void);

// Measures the timer's resolution by calling it in a tight loop and finding
// the minimum observable delta. Returns nanoseconds. The runner calls this
// once at startup. Tests never call it (they inject a synthetic value).
uint64_t iree_bench_detect_clock_resolution(const iree_bench_timer_t* timer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TESTING_BENCHMARK_TIMER_H_
