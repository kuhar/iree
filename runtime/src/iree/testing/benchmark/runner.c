// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark/runner.h"

#include <string.h>

#include "iree/testing/benchmark/result.h"
#include "iree/testing/benchmark/state.h"
#include "iree/testing/benchmark/stats.h"
#include "iree/testing/benchmark/strategy_auto.h"
#include "iree/testing/benchmark/strategy_fixed.h"
#include "iree/testing/benchmark/strategy_time_budget.h"
#include "iree/testing/benchmark/timer.h"

//===----------------------------------------------------------------------===//
// Strategy factory
//===----------------------------------------------------------------------===//

static iree_bench_strategy_t* iree_bench_create_strategy(
    const iree_bench_config_t* config, const iree_bench_def_t* def,
    uint64_t clock_resolution_ns) {
  iree_bench_iter_strategy_t type =
      iree_bench_resolve_strategy_type(config, def);

  switch (type) {
    case IREE_BENCH_ITER_FIXED: {
      uint64_t iters = config->iteration_count > 0 ? config->iteration_count
                                                   : def->iteration_count;
      if (iters == 0) iters = 1;
      uint32_t epochs = config->epoch_count > 0 ? config->epoch_count : 1;
      return iree_bench_strategy_fixed_create(iters, epochs);
    }
    case IREE_BENCH_ITER_TIME_BUDGET: {
      int64_t budget = config->time_budget_ns > 0 ? config->time_budget_ns
                                                  : def->time_budget_ns;
      if (budget <= 0) budget = 500000000;  // 500ms default.
      return iree_bench_strategy_time_budget_create(clock_resolution_ns, budget,
                                                    3);
    }
    case IREE_BENCH_ITER_AUTO:
    default: {
      iree_bench_strategy_auto_params_t params;
      memset(&params, 0, sizeof(params));
      params.iteration_hint = def->iteration_hint;
      params.epoch_count = config->epoch_count;
      return iree_bench_strategy_auto_create(clock_resolution_ns, &params);
    }
  }
}

//===----------------------------------------------------------------------===//
// Timer selection
//===----------------------------------------------------------------------===//

static iree_bench_timer_t iree_bench_select_timer(
    iree_bench_timing_mode_t mode) {
  switch (mode) {
    case IREE_BENCH_TIMING_CPU_THREAD:
      return iree_bench_timer_cpu_thread();
    case IREE_BENCH_TIMING_CPU_PROCESS:
      return iree_bench_timer_cpu_process();
    case IREE_BENCH_TIMING_MANUAL:
    case IREE_BENCH_TIMING_WALL:
    default:
      return iree_bench_timer_wall();
  }
}

//===----------------------------------------------------------------------===//
// Runner
//===----------------------------------------------------------------------===//

void iree_bench_run_one(const iree_bench_config_t* config,
                        const iree_bench_registry_entry_t* entry,
                        uint64_t clock_resolution_ns,
                        const iree_bench_reporter_t* reporter) {
  const iree_bench_def_t* def = &entry->def;

  // Create strategy.
  iree_bench_strategy_t* strategy =
      iree_bench_create_strategy(config, def, clock_resolution_ns);
  if (!strategy) return;

  // Create timer and state.
  iree_bench_timer_t timer = iree_bench_select_timer(def->timing_mode);
  iree_bench_state_t bench_state;
  iree_bench_state_init(&bench_state, strategy, timer);
  if (def->timing_mode == IREE_BENCH_TIMING_MANUAL) {
    bench_state.uses_manual_time = true;
  }

  // Run the benchmark function with the new state directly.
  if (def->run) {
    iree_status_t status = def->run(def, &bench_state);
    if (!iree_status_is_ok(status)) {
      iree_bench_state_skip(&bench_state, "benchmark returned error");
      iree_status_ignore(status);
    }
  }

  // Build result.
  iree_bench_result_t result;
  memset(&result, 0, sizeof(result));
  result.name = entry->name;
  result.unit = def->time_unit;
  result.skipped = bench_state.skipped;
  result.skip_message = bench_state.skip_message;
  result.bytes_processed = bench_state.bytes_processed;
  result.items_processed = bench_state.items_processed;
  result.counters = bench_state.counters;
  result.counter_count = bench_state.counter_count;

  // Get samples and compute stats.
  size_t sample_count = 0;
  const iree_bench_sample_t* samples =
      iree_bench_strategy_samples(strategy, &sample_count);
  result.samples = samples;
  result.sample_count = sample_count;

  if (sample_count > 0 && !result.skipped) {
    // Compute per-iteration times.
    double per_iter[IREE_BENCH_MAX_SAMPLES];
    double scratch[IREE_BENCH_MAX_SAMPLES];
    for (size_t i = 0; i < sample_count; ++i) {
      per_iter[i] =
          samples[i].iterations > 0
              ? samples[i].real_time_ns / (double)samples[i].iterations
              : 0.0;
    }

    // Copy for median (which sorts in-place).
    double work[IREE_BENCH_MAX_SAMPLES];
    memcpy(work, per_iter, sample_count * sizeof(double));
    result.median_ns = iree_bench_stats_median(work, sample_count);

    result.mean_ns = iree_bench_stats_mean(per_iter, sample_count);
    result.stddev_ns = iree_bench_stats_stddev(per_iter, sample_count);
    result.min_ns = iree_bench_stats_min(per_iter, sample_count);
    result.max_ns = iree_bench_stats_max(per_iter, sample_count);

    memcpy(work, per_iter, sample_count * sizeof(double));
    result.mad_ns = iree_bench_stats_mad(work, sample_count, scratch);
    result.mape = iree_bench_stats_mape(per_iter, sample_count, scratch);
  }

  // Report.
  if (reporter && reporter->report) {
    reporter->report(reporter->user_data, &result);
  }

  iree_bench_strategy_destroy(strategy);
}

void iree_bench_run(const iree_bench_config_t* config,
                    iree_bench_registry_t* registry) {
  // Detect clock resolution once.
  iree_bench_timer_t wall_timer = iree_bench_timer_wall();
  uint64_t clock_resolution_ns =
      iree_bench_detect_clock_resolution(&wall_timer);

  // Resolve reporter.
  const iree_bench_reporter_t* reporter = config->reporter;

  // Begin reporting.
  if (reporter && reporter->begin) {
    reporter->begin(reporter->user_data, config);
  }

  // Run each matching benchmark.
  for (size_t i = 0; i < iree_bench_registry_count(registry); ++i) {
    const iree_bench_registry_entry_t* entry =
        iree_bench_registry_get(registry, i);
    if (!iree_bench_registry_matches_filter(entry->name, config->filter)) {
      continue;
    }
    iree_bench_run_one(config, entry, clock_resolution_ns, reporter);
  }

  // End reporting.
  if (reporter && reporter->end) {
    reporter->end(reporter->user_data);
  }
}
