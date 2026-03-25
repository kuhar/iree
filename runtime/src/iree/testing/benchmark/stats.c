// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark/stats.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// Helper: in-place partial sort for median
//===----------------------------------------------------------------------===//

static int iree_bench_compare_double(const void* a, const void* b) {
  double da = *(const double*)a;
  double db = *(const double*)b;
  if (da < db) return -1;
  if (da > db) return 1;
  return 0;
}

//===----------------------------------------------------------------------===//
// Statistical functions
//===----------------------------------------------------------------------===//

double iree_bench_stats_mean(const double* values, size_t count) {
  if (count == 0) return 0.0;
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    sum += values[i];
  }
  return sum / (double)count;
}

double iree_bench_stats_stddev(const double* values, size_t count) {
  if (count <= 1) return 0.0;
  double mean = iree_bench_stats_mean(values, count);
  double sum_sq = 0.0;
  for (size_t i = 0; i < count; ++i) {
    double diff = values[i] - mean;
    sum_sq += diff * diff;
  }
  return sqrt(sum_sq / (double)count);
}

double iree_bench_stats_median(double* values, size_t count) {
  if (count == 0) return 0.0;
  if (count == 1) return values[0];
  qsort(values, count, sizeof(double), iree_bench_compare_double);
  if (count % 2 == 1) {
    return values[count / 2];
  }
  return (values[count / 2 - 1] + values[count / 2]) / 2.0;
}

double iree_bench_stats_mad(double* values, size_t count, double* scratch) {
  if (count == 0) return 0.0;
  // Compute median first (sorts values).
  double med = iree_bench_stats_median(values, count);
  // Compute absolute deviations into scratch.
  for (size_t i = 0; i < count; ++i) {
    scratch[i] = fabs(values[i] - med);
  }
  // MAD = median of absolute deviations.
  return iree_bench_stats_median(scratch, count);
}

double iree_bench_stats_mape(double* values, size_t count, double* scratch) {
  if (count == 0) return 0.0;
  // Compute median of per-iteration times.
  // Copy to scratch first to preserve original values for error computation.
  memcpy(scratch, values, count * sizeof(double));
  double med = iree_bench_stats_median(scratch, count);
  if (med == 0.0) return 0.0;
  // Compute absolute percentage errors.
  double sum_ape = 0.0;
  for (size_t i = 0; i < count; ++i) {
    sum_ape += fabs(values[i] - med) / med;
  }
  return sum_ape / (double)count;
}

double iree_bench_stats_min(const double* values, size_t count) {
  if (count == 0) return 0.0;
  double min_val = values[0];
  for (size_t i = 1; i < count; ++i) {
    if (values[i] < min_val) min_val = values[i];
  }
  return min_val;
}

double iree_bench_stats_max(const double* values, size_t count) {
  if (count == 0) return 0.0;
  double max_val = values[0];
  for (size_t i = 1; i < count; ++i) {
    if (values[i] > max_val) max_val = values[i];
  }
  return max_val;
}
