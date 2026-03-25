// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_BENCHMARK_STRATEGY_AUTO_H_
#define IREE_TESTING_BENCHMARK_STRATEGY_AUTO_H_

#include "iree/testing/benchmark/strategy.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parameters for the AUTO strategy.
typedef struct {
  uint64_t iteration_hint;             // Starting iterations. 0 -> 1.
  uint32_t epoch_count;                // Number of epochs to collect. 0 -> 11.
  uint32_t clock_resolution_multiple;  // target = resolution * this. 0 -> 1000.
  uint64_t max_iterations;             // Safety cap. 0 -> 1<<40.
} iree_bench_strategy_auto_params_t;

// Creates a clock-resolution-adaptive strategy.
// |clock_resolution_ns| is injected (not measured internally).
// The caller must destroy the returned strategy with
// iree_bench_strategy_destroy().
iree_bench_strategy_t* iree_bench_strategy_auto_create(
    uint64_t clock_resolution_ns,
    const iree_bench_strategy_auto_params_t* params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TESTING_BENCHMARK_STRATEGY_AUTO_H_
