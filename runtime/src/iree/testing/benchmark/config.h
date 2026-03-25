// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_BENCHMARK_CONFIG_H_
#define IREE_TESTING_BENCHMARK_CONFIG_H_

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/testing/benchmark/strategy.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Unit and timing mode enums
//===----------------------------------------------------------------------===//

typedef enum {
  IREE_BENCH_UNIT_DEFAULT = 0,  // Auto-scale based on measurement.
  IREE_BENCH_UNIT_NANOSECOND,
  IREE_BENCH_UNIT_MICROSECOND,
  IREE_BENCH_UNIT_MILLISECOND,
  IREE_BENCH_UNIT_SECOND,
} iree_bench_unit_t;

typedef enum {
  IREE_BENCH_TIMING_WALL = 0,
  IREE_BENCH_TIMING_CPU_THREAD,
  IREE_BENCH_TIMING_CPU_PROCESS,
  IREE_BENCH_TIMING_MANUAL,
} iree_bench_timing_mode_t;

typedef enum {
  IREE_BENCH_ITER_AUTO = 0,
  IREE_BENCH_ITER_FIXED,
  IREE_BENCH_ITER_TIME_BUDGET,
} iree_bench_iter_strategy_t;

//===----------------------------------------------------------------------===//
// Forward declarations
//===----------------------------------------------------------------------===//

typedef struct iree_bench_reporter_t iree_bench_reporter_t;

//===----------------------------------------------------------------------===//
// Benchmark definition
//===----------------------------------------------------------------------===//

typedef struct iree_bench_state_t iree_bench_state_t;  // From state.h.
typedef struct iree_bench_def_t iree_bench_def_t;

// Benchmark function signature for the new framework.
// Takes the new iree_bench_state_t directly.
typedef iree_status_t (*iree_bench_fn_t)(const iree_bench_def_t* def,
                                         iree_bench_state_t* state);

struct iree_bench_def_t {
  iree_bench_timing_mode_t timing_mode;
  iree_bench_unit_t time_unit;
  iree_bench_iter_strategy_t iter_strategy;
  uint64_t iteration_count;  // For FIXED. 0 = use global/auto.
  int64_t time_budget_ns;    // For TIME_BUDGET. 0 = use global/auto.
  uint64_t iteration_hint;   // Starting hint for AUTO. 0 = start at 1.
  int64_t warmup_ns;         // -1 = use global default. Not yet implemented.
  iree_bench_fn_t run;
  void* user_data;
};

// Returns a definition with sensible defaults.
static inline iree_bench_def_t iree_bench_def_default(iree_bench_fn_t fn) {
  iree_bench_def_t def;
  memset(&def, 0, sizeof(def));
  def.timing_mode = IREE_BENCH_TIMING_WALL;
  def.time_unit = IREE_BENCH_UNIT_DEFAULT;
  def.iter_strategy = IREE_BENCH_ITER_AUTO;
  def.warmup_ns = -1;
  def.run = fn;
  return def;
}

//===----------------------------------------------------------------------===//
// Suite-level configuration
//===----------------------------------------------------------------------===//

typedef struct {
  // Filter benchmarks by substring. NULL = run all.
  const char* filter;

  // Global iteration override. 0 = per-benchmark or auto.
  uint64_t iteration_count;

  // Global time budget override. 0 = per-benchmark or auto. In nanoseconds.
  int64_t time_budget_ns;

  // Global unit override. DEFAULT = per-benchmark or auto-scale.
  iree_bench_unit_t time_unit;

  // Number of epochs for AUTO strategy. 0 = 11.
  uint32_t epoch_count;

  // Reporter callbacks. NULL = default console reporter.
  const iree_bench_reporter_t* reporter;

  // Emit raw JSON alongside console output. NULL = no. Not yet implemented.
  const char* json_output_path;

  // Verbosity. 0 = normal, 1 = show trial runs, 2+ = debug. Not yet
  // implemented.
  int verbosity;
} iree_bench_config_t;

// Returns a config with sensible defaults.
static inline iree_bench_config_t iree_bench_config_default(void) {
  iree_bench_config_t config;
  memset(&config, 0, sizeof(config));
  return config;
}

//===----------------------------------------------------------------------===//
// Strategy resolution
//===----------------------------------------------------------------------===//

// Resolves which strategy type should be used given config and def overrides.
// Pure function: (config, def) -> strategy type.
static inline iree_bench_iter_strategy_t iree_bench_resolve_strategy_type(
    const iree_bench_config_t* config, const iree_bench_def_t* def) {
  if (config->iteration_count > 0) return IREE_BENCH_ITER_FIXED;
  if (config->time_budget_ns > 0) return IREE_BENCH_ITER_TIME_BUDGET;
  if (def->iter_strategy == IREE_BENCH_ITER_FIXED) return IREE_BENCH_ITER_FIXED;
  if (def->iter_strategy == IREE_BENCH_ITER_TIME_BUDGET)
    return IREE_BENCH_ITER_TIME_BUDGET;
  return IREE_BENCH_ITER_AUTO;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TESTING_BENCHMARK_CONFIG_H_
