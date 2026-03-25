// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark/strategy_time_budget.h"

#include <stdlib.h>

//===----------------------------------------------------------------------===//
// TIME_BUDGET strategy: calibrate then collect until budget met
//===----------------------------------------------------------------------===//

typedef enum {
  IREE_BENCH_TB_CALIBRATING = 0,
  IREE_BENCH_TB_COLLECTING = 1,
} iree_bench_tb_phase_t;

typedef struct {
  iree_bench_strategy_t base;
  iree_bench_tb_phase_t phase;
  uint64_t calibration_target_ns;  // clock_resolution * 1000.
  int64_t budget_ns;
  uint32_t min_epoch_count;
  uint64_t current_iterations;
  uint64_t max_iterations;
  double total_elapsed_ns;
  uint32_t epochs_collected;
  iree_bench_sample_t samples[IREE_BENCH_MAX_SAMPLES];
} iree_bench_strategy_time_budget_t;

static iree_bench_action_t iree_bench_strategy_tb_begin(
    iree_bench_strategy_t* strategy) {
  iree_bench_strategy_time_budget_t* s =
      (iree_bench_strategy_time_budget_t*)strategy;
  iree_bench_action_t action;
  action.type = IREE_BENCH_ACTION_TRIAL;
  action.iteration_count = s->current_iterations;
  return action;
}

static iree_bench_action_t iree_bench_strategy_tb_observe(
    iree_bench_strategy_t* strategy, uint64_t iterations_run,
    double elapsed_ns) {
  iree_bench_strategy_time_budget_t* s =
      (iree_bench_strategy_time_budget_t*)strategy;
  iree_bench_action_t action;

  if (s->phase == IREE_BENCH_TB_CALIBRATING) {
    if (elapsed_ns >= (double)s->calibration_target_ns ||
        s->current_iterations >= s->max_iterations) {
      // Calibration complete. Record as first epoch.
      s->phase = IREE_BENCH_TB_COLLECTING;
      s->samples[s->epochs_collected].iterations = iterations_run;
      s->samples[s->epochs_collected].real_time_ns = elapsed_ns;
      s->samples[s->epochs_collected].cpu_time_ns = 0;
      s->samples[s->epochs_collected].manual_time_ns = 0;
      s->epochs_collected++;
      s->total_elapsed_ns += elapsed_ns;

      if (s->total_elapsed_ns >= (double)s->budget_ns &&
          s->epochs_collected >= s->min_epoch_count) {
        action.type = IREE_BENCH_ACTION_DONE;
      } else {
        action.type = IREE_BENCH_ACTION_RECORD_AND_CONTINUE;
      }
      action.iteration_count = s->current_iterations;
    } else {
      // Double iterations and try again (with overflow guard).
      if (s->current_iterations > s->max_iterations / 2) {
        s->current_iterations = s->max_iterations;
      } else {
        s->current_iterations *= 2;
      }
      action.type = IREE_BENCH_ACTION_TRIAL;
      action.iteration_count = s->current_iterations;
    }
  } else {
    // COLLECTING phase: record and check budget.
    if (s->epochs_collected < IREE_BENCH_MAX_SAMPLES) {
      s->samples[s->epochs_collected].iterations = iterations_run;
      s->samples[s->epochs_collected].real_time_ns = elapsed_ns;
      s->samples[s->epochs_collected].cpu_time_ns = 0;
      s->samples[s->epochs_collected].manual_time_ns = 0;
      s->epochs_collected++;
    }
    s->total_elapsed_ns += elapsed_ns;

    if (s->total_elapsed_ns >= (double)s->budget_ns &&
        s->epochs_collected >= s->min_epoch_count) {
      action.type = IREE_BENCH_ACTION_DONE;
    } else {
      action.type = IREE_BENCH_ACTION_RECORD_AND_CONTINUE;
    }
    action.iteration_count = s->current_iterations;
  }

  return action;
}

static const iree_bench_sample_t* iree_bench_strategy_tb_samples(
    const iree_bench_strategy_t* strategy, size_t* out_count) {
  const iree_bench_strategy_time_budget_t* s =
      (const iree_bench_strategy_time_budget_t*)strategy;
  *out_count = s->epochs_collected;
  return s->samples;
}

static void iree_bench_strategy_tb_destroy(iree_bench_strategy_t* strategy) {
  free(strategy);
}

static const iree_bench_strategy_vtable_t iree_bench_strategy_tb_vtable = {
    .begin = iree_bench_strategy_tb_begin,
    .observe = iree_bench_strategy_tb_observe,
    .samples = iree_bench_strategy_tb_samples,
    .destroy = iree_bench_strategy_tb_destroy,
};

iree_bench_strategy_t* iree_bench_strategy_time_budget_create(
    uint64_t clock_resolution_ns, int64_t budget_ns, uint32_t min_epoch_count) {
  iree_bench_strategy_time_budget_t* s =
      (iree_bench_strategy_time_budget_t*)calloc(1, sizeof(*s));
  if (!s) return NULL;

  s->base.vtable = &iree_bench_strategy_tb_vtable;
  s->phase = IREE_BENCH_TB_CALIBRATING;
  s->calibration_target_ns = clock_resolution_ns * 1000;
  s->budget_ns = budget_ns;
  s->min_epoch_count = min_epoch_count > 0 ? min_epoch_count : 3;
  s->current_iterations = 1;
  s->max_iterations = (uint64_t)1 << 40;
  s->total_elapsed_ns = 0;
  s->epochs_collected = 0;

  return &s->base;
}
