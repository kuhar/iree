// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_BENCHMARK_STATE_H_
#define IREE_TESTING_BENCHMARK_STATE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/testing/benchmark/result.h"
#include "iree/testing/benchmark/strategy.h"
#include "iree/testing/benchmark/timer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Benchmark state: wraps strategy + timer for the iteration loop
//===----------------------------------------------------------------------===//

#define IREE_BENCH_MAX_COUNTERS 16

struct iree_bench_state_t {
  // Owning strategy (the state drives it).
  iree_bench_strategy_t* strategy;

  // Timer to use for measurements.
  iree_bench_timer_t timer;

  // Current action from the strategy.
  iree_bench_action_t current_action;

  // Iteration tracking within current epoch.
  uint64_t iterations_remaining;
  uint64_t total_iterations;  // Across all epochs.

  // Timing state.
  uint64_t epoch_start_ns;
  uint64_t pause_start_ns;
  uint64_t paused_duration_ns;
  bool timer_started;
  bool paused;
  bool done;
  bool first_call;

  // Manual timing accumulator.
  double manual_time_ns;
  bool uses_manual_time;

  // User-set metrics.
  int64_t bytes_processed;
  int64_t items_processed;
  iree_bench_counter_result_t counters[IREE_BENCH_MAX_COUNTERS];
  size_t counter_count;

  // Skip state.
  bool skipped;
  const char* skip_message;
};

// Initializes a benchmark state. Does not take ownership of |strategy|.
void iree_bench_state_init(iree_bench_state_t* state,
                           iree_bench_strategy_t* strategy,
                           iree_bench_timer_t timer);

// The core loop function. Returns true while the benchmark should keep running.
// On the first call, starts the timer and gets the first action from the
// strategy. On subsequent calls, checks if the current epoch's iterations are
// done, measures elapsed time, feeds it to the strategy, and gets the next
// action.
bool iree_bench_state_keep_running(iree_bench_state_t* state,
                                   uint64_t batch_count);

// Pauses the timer. Use for excluding setup work within the loop.
void iree_bench_state_pause(iree_bench_state_t* state);

// Resumes the timer after a pause.
void iree_bench_state_resume(iree_bench_state_t* state);

// Adds to the manual timing accumulator for the current epoch. Times
// accumulate across calls within an epoch and are reset at epoch boundary.
// Call once per iteration with that iteration's measured time.
void iree_bench_state_set_iteration_time(iree_bench_state_t* state,
                                         double seconds);

// Sets bytes processed for throughput reporting.
void iree_bench_state_set_bytes(iree_bench_state_t* state, int64_t bytes);

// Sets items processed for throughput reporting.
void iree_bench_state_set_items(iree_bench_state_t* state, int64_t items);

// Adds a named counter.
void iree_bench_state_add_counter(iree_bench_state_t* state, const char* name,
                                  double value,
                                  iree_bench_counter_flags_t flags);

// Skips the benchmark with a message.
void iree_bench_state_skip(iree_bench_state_t* state, const char* message);

// Returns the total iterations executed across all completed epochs.
uint64_t iree_bench_state_iterations(const iree_bench_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TESTING_BENCHMARK_STATE_H_
