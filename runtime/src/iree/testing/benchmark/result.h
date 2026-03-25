// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_BENCHMARK_RESULT_H_
#define IREE_TESTING_BENCHMARK_RESULT_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/testing/benchmark/config.h"
#include "iree/testing/benchmark/strategy.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Counter result
//===----------------------------------------------------------------------===//

typedef enum {
  IREE_BENCH_COUNTER_FLAG_NONE = 0,
  IREE_BENCH_COUNTER_FLAG_RATE = 1u << 0,         // Display as /s.
  IREE_BENCH_COUNTER_FLAG_AVG_THREADS = 1u << 1,  // Average over threads.
} iree_bench_counter_flags_t;

typedef struct {
  const char* name;
  double value;
  iree_bench_counter_flags_t flags;
} iree_bench_counter_result_t;

//===----------------------------------------------------------------------===//
// Benchmark result
//===----------------------------------------------------------------------===//

typedef struct {
  const char* name;
  iree_bench_unit_t unit;

  // Raw epoch/sample data.
  const iree_bench_sample_t* samples;
  size_t sample_count;

  // Computed statistics (per-iteration times in ns).
  double median_ns;
  double mean_ns;
  double mad_ns;
  double stddev_ns;
  double min_ns;
  double max_ns;
  double mape;

  // User counters.
  const iree_bench_counter_result_t* counters;
  size_t counter_count;

  // Bytes and items processed (for throughput reporting).
  int64_t bytes_processed;
  int64_t items_processed;

  // Skip information.
  bool skipped;
  const char* skip_message;
} iree_bench_result_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TESTING_BENCHMARK_RESULT_H_
