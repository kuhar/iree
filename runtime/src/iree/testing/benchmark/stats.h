// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_BENCHMARK_STATS_H_
#define IREE_TESTING_BENCHMARK_STATS_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Pure statistical functions
// All operate on caller-provided buffers. No heap allocation.
//===----------------------------------------------------------------------===//

// Computes the arithmetic mean of |values|.
double iree_bench_stats_mean(const double* values, size_t count);

// Computes the population standard deviation of |values| (divides by N, not
// N-1). Appropriate for benchmark epochs which represent the full measurement.
double iree_bench_stats_stddev(const double* values, size_t count);

// Computes the median of |values|. Modifies |values| in-place (partial sort).
double iree_bench_stats_median(double* values, size_t count);

// Computes the Median Absolute Deviation of |values|.
// Modifies |scratch| in-place (must be at least |count| elements).
double iree_bench_stats_mad(double* values, size_t count, double* scratch);

// Computes the Mean Absolute Percentage Error (relative to median) of
// per-iteration times. Each value[i] = elapsed_ns / iterations for that epoch.
// Result = mean(|x_i - median| / median). Used as a stability indicator.
// Modifies |scratch| in-place (must be at least |count| elements).
double iree_bench_stats_mape(double* values, size_t count, double* scratch);

// Returns the minimum value.
double iree_bench_stats_min(const double* values, size_t count);

// Returns the maximum value.
double iree_bench_stats_max(const double* values, size_t count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TESTING_BENCHMARK_STATS_H_
