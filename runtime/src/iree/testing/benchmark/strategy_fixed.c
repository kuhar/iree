// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark/strategy_fixed.h"

#include <stdlib.h>

//===----------------------------------------------------------------------===//
// FIXED strategy: exact iteration count, no calibration
//===----------------------------------------------------------------------===//

typedef struct {
  iree_bench_strategy_t base;
  uint64_t iteration_count;
  uint32_t epoch_count;
  uint32_t epochs_collected;
  iree_bench_sample_t samples[IREE_BENCH_MAX_SAMPLES];
} iree_bench_strategy_fixed_t;

static iree_bench_action_t iree_bench_strategy_fixed_begin(
    iree_bench_strategy_t* strategy) {
  iree_bench_strategy_fixed_t* s = (iree_bench_strategy_fixed_t*)strategy;
  iree_bench_action_t action;
  action.type = IREE_BENCH_ACTION_TRIAL;
  action.iteration_count = s->iteration_count;
  return action;
}

static iree_bench_action_t iree_bench_strategy_fixed_observe(
    iree_bench_strategy_t* strategy, uint64_t iterations_run,
    double elapsed_ns) {
  iree_bench_strategy_fixed_t* s = (iree_bench_strategy_fixed_t*)strategy;
  iree_bench_action_t action;

  // Record every observation as an epoch.
  if (s->epochs_collected < IREE_BENCH_MAX_SAMPLES) {
    s->samples[s->epochs_collected].iterations = iterations_run;
    s->samples[s->epochs_collected].real_time_ns = elapsed_ns;
    s->samples[s->epochs_collected].cpu_time_ns = 0;
    s->samples[s->epochs_collected].manual_time_ns = 0;
    s->epochs_collected++;
  }

  if (s->epochs_collected >= s->epoch_count) {
    action.type = IREE_BENCH_ACTION_DONE;
  } else {
    action.type = IREE_BENCH_ACTION_RECORD_AND_CONTINUE;
  }
  action.iteration_count = s->iteration_count;
  return action;
}

static const iree_bench_sample_t* iree_bench_strategy_fixed_samples(
    const iree_bench_strategy_t* strategy, size_t* out_count) {
  const iree_bench_strategy_fixed_t* s =
      (const iree_bench_strategy_fixed_t*)strategy;
  *out_count = s->epochs_collected;
  return s->samples;
}

static void iree_bench_strategy_fixed_destroy(iree_bench_strategy_t* strategy) {
  free(strategy);
}

static const iree_bench_strategy_vtable_t iree_bench_strategy_fixed_vtable = {
    .begin = iree_bench_strategy_fixed_begin,
    .observe = iree_bench_strategy_fixed_observe,
    .samples = iree_bench_strategy_fixed_samples,
    .destroy = iree_bench_strategy_fixed_destroy,
};

iree_bench_strategy_t* iree_bench_strategy_fixed_create(
    uint64_t iteration_count, uint32_t epoch_count) {
  iree_bench_strategy_fixed_t* s =
      (iree_bench_strategy_fixed_t*)calloc(1, sizeof(*s));
  if (!s) return NULL;

  s->base.vtable = &iree_bench_strategy_fixed_vtable;
  s->iteration_count = iteration_count;
  s->epoch_count = epoch_count > 0 ? epoch_count : 1;
  if (s->epoch_count > IREE_BENCH_MAX_SAMPLES)
    s->epoch_count = IREE_BENCH_MAX_SAMPLES;
  s->epochs_collected = 0;

  return &s->base;
}
