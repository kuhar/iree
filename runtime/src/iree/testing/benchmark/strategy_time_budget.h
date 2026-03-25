// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_BENCHMARK_STRATEGY_TIME_BUDGET_H_
#define IREE_TESTING_BENCHMARK_STRATEGY_TIME_BUDGET_H_

#include "iree/testing/benchmark/strategy.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a time-budget strategy. Calibrates iterations like AUTO (using
// clock resolution), then keeps collecting epochs until cumulative elapsed
// time >= |budget_ns| AND epochs >= |min_epoch_count|.
// |min_epoch_count| of 0 defaults to 3.
iree_bench_strategy_t* iree_bench_strategy_time_budget_create(
    uint64_t clock_resolution_ns, int64_t budget_ns, uint32_t min_epoch_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TESTING_BENCHMARK_STRATEGY_TIME_BUDGET_H_
