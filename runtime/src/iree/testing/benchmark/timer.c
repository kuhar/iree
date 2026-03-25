// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark/timer.h"

#include "iree/base/api.h"

//===----------------------------------------------------------------------===//
// Wall clock timer (uses iree_time_now)
//===----------------------------------------------------------------------===//

static uint64_t iree_bench_timer_wall_now(void* user_data) {
  (void)user_data;
  iree_time_t t = iree_time_now();
  return (uint64_t)t;
}

iree_bench_timer_t iree_bench_timer_wall(void) {
  iree_bench_timer_t timer = {iree_bench_timer_wall_now, NULL};
  return timer;
}

//===----------------------------------------------------------------------===//
// CPU thread timer
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
#include <time.h>

static uint64_t iree_bench_timer_cpu_thread_now(void* user_data) {
  (void)user_data;
  struct timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t iree_bench_timer_cpu_process_now(void* user_data) {
  (void)user_data;
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

#elif defined(IREE_PLATFORM_APPLE)
#include <mach/mach_time.h>
#include <time.h>

static uint64_t iree_bench_timer_cpu_thread_now(void* user_data) {
  (void)user_data;
  struct timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t iree_bench_timer_cpu_process_now(void* user_data) {
  (void)user_data;
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

#elif defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>

static uint64_t iree_bench_timer_cpu_thread_now(void* user_data) {
  (void)user_data;
  FILETIME creation, exit, kernel, user;
  GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user);
  // FILETIME is in 100ns units.
  uint64_t kernel_ns =
      ((uint64_t)kernel.dwHighDateTime << 32 | kernel.dwLowDateTime) * 100;
  uint64_t user_ns =
      ((uint64_t)user.dwHighDateTime << 32 | user.dwLowDateTime) * 100;
  return kernel_ns + user_ns;
}

static uint64_t iree_bench_timer_cpu_process_now(void* user_data) {
  (void)user_data;
  FILETIME creation, exit, kernel, user;
  GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user);
  uint64_t kernel_ns =
      ((uint64_t)kernel.dwHighDateTime << 32 | kernel.dwLowDateTime) * 100;
  uint64_t user_ns =
      ((uint64_t)user.dwHighDateTime << 32 | user.dwLowDateTime) * 100;
  return kernel_ns + user_ns;
}

#else
// Fallback: use wall clock.
static uint64_t iree_bench_timer_cpu_thread_now(void* user_data) {
  return iree_bench_timer_wall_now(user_data);
}

static uint64_t iree_bench_timer_cpu_process_now(void* user_data) {
  return iree_bench_timer_wall_now(user_data);
}
#endif  // platform

iree_bench_timer_t iree_bench_timer_cpu_thread(void) {
  iree_bench_timer_t timer = {iree_bench_timer_cpu_thread_now, NULL};
  return timer;
}

iree_bench_timer_t iree_bench_timer_cpu_process(void) {
  iree_bench_timer_t timer = {iree_bench_timer_cpu_process_now, NULL};
  return timer;
}

//===----------------------------------------------------------------------===//
// Clock resolution detection
//===----------------------------------------------------------------------===//

uint64_t iree_bench_detect_clock_resolution(const iree_bench_timer_t* timer) {
  // Find the minimum observable tick delta across multiple attempts.
  // This gives us the timer's effective resolution.
  uint64_t min_delta = UINT64_MAX;
  for (int attempt = 0; attempt < 100; ++attempt) {
    uint64_t t0 = timer->now_ns(timer->user_data);
    uint64_t t1;
    // Spin until the timer ticks.
    do {
      t1 = timer->now_ns(timer->user_data);
    } while (t1 == t0);
    uint64_t delta = t1 - t0;
    if (delta < min_delta) {
      min_delta = delta;
    }
  }
  // Clamp to at least 1ns to avoid division by zero.
  return min_delta > 0 ? min_delta : 1;
}
