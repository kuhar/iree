// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_BENCHMARK_STRATEGY_FIXED_H_
#define IREE_TESTING_BENCHMARK_STRATEGY_FIXED_H_

#include "iree/testing/benchmark/strategy.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a fixed-iteration strategy.
// Runs exactly |iteration_count| iterations per epoch, for |epoch_count|
// epochs. No calibration phase. |epoch_count| of 0 defaults to 1.
iree_bench_strategy_t* iree_bench_strategy_fixed_create(
    uint64_t iteration_count, uint32_t epoch_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TESTING_BENCHMARK_STRATEGY_FIXED_H_
