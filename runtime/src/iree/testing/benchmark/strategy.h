// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_BENCHMARK_STRATEGY_H_
#define IREE_TESTING_BENCHMARK_STRATEGY_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Maximum number of samples a strategy can collect.
#define IREE_BENCH_MAX_SAMPLES 256

//===----------------------------------------------------------------------===//
// Action types and sample storage
//===----------------------------------------------------------------------===//

// What the strategy tells the runner to do next.
typedef enum {
  // Run the benchmark for iteration_count iterations. Feed the elapsed time
  // back via iree_bench_strategy_observe().
  IREE_BENCH_ACTION_TRIAL = 0,

  // The last observation was a valid epoch. It has been recorded internally.
  // Now run another trial (iteration_count tells you how many iterations).
  IREE_BENCH_ACTION_RECORD_AND_CONTINUE = 1,

  // All epochs collected. The strategy is done.
  IREE_BENCH_ACTION_DONE = 2,
} iree_bench_action_type_t;

// Action returned by strategy begin/observe.
typedef struct {
  iree_bench_action_type_t type;
  uint64_t iteration_count;
} iree_bench_action_t;

// A single timing sample/epoch.
typedef struct {
  uint64_t iterations;
  double real_time_ns;
  double cpu_time_ns;
  double manual_time_ns;  // 0 if not manual timing.
} iree_bench_sample_t;

//===----------------------------------------------------------------------===//
// Strategy vtable and base type
//===----------------------------------------------------------------------===//

// Forward declaration.
typedef struct iree_bench_strategy_t iree_bench_strategy_t;

// Strategy vtable. Concrete strategies implement these functions.
typedef struct {
  iree_bench_action_t (*begin)(iree_bench_strategy_t* strategy);
  iree_bench_action_t (*observe)(iree_bench_strategy_t* strategy,
                                 uint64_t iterations_run, double elapsed_ns);
  const iree_bench_sample_t* (*samples)(const iree_bench_strategy_t* strategy,
                                        size_t* out_count);
  void (*destroy)(iree_bench_strategy_t* strategy);
} iree_bench_strategy_vtable_t;

// Base strategy type. Concrete strategies embed this as their first member.
struct iree_bench_strategy_t {
  const iree_bench_strategy_vtable_t* vtable;
};

//===----------------------------------------------------------------------===//
// Strategy dispatch (inline for zero overhead)
//===----------------------------------------------------------------------===//

// Gets the first action (before any observations).
static inline iree_bench_action_t iree_bench_strategy_begin(
    iree_bench_strategy_t* strategy) {
  return strategy->vtable->begin(strategy);
}

// Feeds an observation (iterations actually run + elapsed time), gets the next
// action.
static inline iree_bench_action_t iree_bench_strategy_observe(
    iree_bench_strategy_t* strategy, uint64_t iterations_run,
    double elapsed_ns) {
  return strategy->vtable->observe(strategy, iterations_run, elapsed_ns);
}

// After DONE: retrieves collected epochs.
static inline const iree_bench_sample_t* iree_bench_strategy_samples(
    const iree_bench_strategy_t* strategy, size_t* out_count) {
  return strategy->vtable->samples(strategy, out_count);
}

// Destroys a strategy and frees its resources.
static inline void iree_bench_strategy_destroy(
    iree_bench_strategy_t* strategy) {
  if (strategy && strategy->vtable->destroy) {
    strategy->vtable->destroy(strategy);
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TESTING_BENCHMARK_STRATEGY_H_
