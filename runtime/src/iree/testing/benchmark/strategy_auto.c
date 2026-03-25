// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark/strategy_auto.h"

#include <stdlib.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// AUTO strategy: clock-resolution-adaptive
//===----------------------------------------------------------------------===//

typedef enum {
  IREE_BENCH_AUTO_CALIBRATING = 0,
  IREE_BENCH_AUTO_COLLECTING = 1,
} iree_bench_auto_phase_t;

typedef struct {
  iree_bench_strategy_t base;
  iree_bench_auto_phase_t phase;
  uint64_t target_ns;
  uint64_t current_iterations;
  uint64_t max_iterations;
  uint32_t epoch_count;
  uint32_t epochs_collected;
  iree_bench_sample_t samples[IREE_BENCH_MAX_SAMPLES];
} iree_bench_strategy_auto_t;

static iree_bench_action_t iree_bench_strategy_auto_begin(
    iree_bench_strategy_t* strategy) {
  iree_bench_strategy_auto_t* s = (iree_bench_strategy_auto_t*)strategy;
  iree_bench_action_t action;
  action.type = IREE_BENCH_ACTION_TRIAL;
  action.iteration_count = s->current_iterations;
  return action;
}

static iree_bench_action_t iree_bench_strategy_auto_observe(
    iree_bench_strategy_t* strategy, uint64_t iterations_run,
    double elapsed_ns) {
  iree_bench_strategy_auto_t* s = (iree_bench_strategy_auto_t*)strategy;
  iree_bench_action_t action;

  if (s->phase == IREE_BENCH_AUTO_CALIBRATING) {
    if (elapsed_ns >= (double)s->target_ns ||
        s->current_iterations >= s->max_iterations) {
      // Calibration complete. Record this observation as the first epoch.
      s->phase = IREE_BENCH_AUTO_COLLECTING;
      s->samples[s->epochs_collected].iterations = iterations_run;
      s->samples[s->epochs_collected].real_time_ns = elapsed_ns;
      s->samples[s->epochs_collected].cpu_time_ns = 0;
      s->samples[s->epochs_collected].manual_time_ns = 0;
      s->epochs_collected++;

      if (s->epochs_collected >= s->epoch_count) {
        action.type = IREE_BENCH_ACTION_DONE;
        action.iteration_count = s->current_iterations;
      } else {
        action.type = IREE_BENCH_ACTION_RECORD_AND_CONTINUE;
        action.iteration_count = s->current_iterations;
      }
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
    // COLLECTING phase: record every observation as an epoch.
    if (s->epochs_collected < IREE_BENCH_MAX_SAMPLES) {
      s->samples[s->epochs_collected].iterations = iterations_run;
      s->samples[s->epochs_collected].real_time_ns = elapsed_ns;
      s->samples[s->epochs_collected].cpu_time_ns = 0;
      s->samples[s->epochs_collected].manual_time_ns = 0;
      s->epochs_collected++;
    }

    if (s->epochs_collected >= s->epoch_count) {
      action.type = IREE_BENCH_ACTION_DONE;
      action.iteration_count = s->current_iterations;
    } else {
      action.type = IREE_BENCH_ACTION_RECORD_AND_CONTINUE;
      action.iteration_count = s->current_iterations;
    }
  }

  return action;
}

static const iree_bench_sample_t* iree_bench_strategy_auto_samples(
    const iree_bench_strategy_t* strategy, size_t* out_count) {
  const iree_bench_strategy_auto_t* s =
      (const iree_bench_strategy_auto_t*)strategy;
  *out_count = s->epochs_collected;
  return s->samples;
}

static void iree_bench_strategy_auto_destroy(iree_bench_strategy_t* strategy) {
  free(strategy);
}

static const iree_bench_strategy_vtable_t iree_bench_strategy_auto_vtable = {
    .begin = iree_bench_strategy_auto_begin,
    .observe = iree_bench_strategy_auto_observe,
    .samples = iree_bench_strategy_auto_samples,
    .destroy = iree_bench_strategy_auto_destroy,
};

iree_bench_strategy_t* iree_bench_strategy_auto_create(
    uint64_t clock_resolution_ns,
    const iree_bench_strategy_auto_params_t* params) {
  iree_bench_strategy_auto_t* s =
      (iree_bench_strategy_auto_t*)calloc(1, sizeof(*s));
  if (!s) return NULL;

  s->base.vtable = &iree_bench_strategy_auto_vtable;
  s->phase = IREE_BENCH_AUTO_CALIBRATING;

  uint64_t hint = params->iteration_hint;
  if (hint == 0) hint = 1;
  s->current_iterations = hint;

  uint32_t epoch_count = params->epoch_count;
  if (epoch_count == 0) epoch_count = 11;
  if (epoch_count > IREE_BENCH_MAX_SAMPLES)
    epoch_count = IREE_BENCH_MAX_SAMPLES;
  s->epoch_count = epoch_count;

  uint32_t multiple = params->clock_resolution_multiple;
  if (multiple == 0) multiple = 1000;
  s->target_ns = clock_resolution_ns * multiple;

  uint64_t max_iters = params->max_iterations;
  if (max_iters == 0) max_iters = (uint64_t)1 << 40;
  s->max_iterations = max_iters;

  s->epochs_collected = 0;

  return &s->base;
}
