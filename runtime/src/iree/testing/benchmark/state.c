// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark/state.h"

#include <string.h>

void iree_bench_state_init(iree_bench_state_t* state,
                           iree_bench_strategy_t* strategy,
                           iree_bench_timer_t timer) {
  memset(state, 0, sizeof(*state));
  state->strategy = strategy;
  state->timer = timer;
  state->first_call = true;
}

bool iree_bench_state_keep_running(iree_bench_state_t* state,
                                   uint64_t batch_count) {
  if (state->done || state->skipped) return false;

  if (state->first_call) {
    // First call: get the initial action from the strategy, start timing,
    // and return true for the first iteration without consuming.
    state->first_call = false;
    state->current_action = iree_bench_strategy_begin(state->strategy);
    state->iterations_remaining = state->current_action.iteration_count;
    state->epoch_start_ns = state->timer.now_ns(state->timer.user_data);
    state->paused_duration_ns = 0;
    state->timer_started = true;
    return true;
  }

  // Account for the work the caller did since the last true return.
  uint64_t consume = batch_count;
  if (consume > state->iterations_remaining) {
    consume = state->iterations_remaining;
  }
  state->iterations_remaining -= consume;
  state->total_iterations += consume;

  if (state->iterations_remaining > 0) {
    return true;
  }

  // Epoch complete. Compute elapsed time and feed to strategy.
  uint64_t now = state->timer.now_ns(state->timer.user_data);
  double elapsed_ns;
  if (state->uses_manual_time) {
    elapsed_ns = state->manual_time_ns;
    state->manual_time_ns = 0;
  } else {
    elapsed_ns =
        (double)(now - state->epoch_start_ns - state->paused_duration_ns);
  }

  state->current_action = iree_bench_strategy_observe(
      state->strategy, state->current_action.iteration_count, elapsed_ns);

  if (state->current_action.type == IREE_BENCH_ACTION_DONE) {
    state->done = true;
    return false;
  }

  // Set up next epoch and return true for its first iteration.
  state->iterations_remaining = state->current_action.iteration_count;
  state->epoch_start_ns = state->timer.now_ns(state->timer.user_data);
  state->paused_duration_ns = 0;
  return true;
}

void iree_bench_state_pause(iree_bench_state_t* state) {
  if (!state->paused) {
    state->paused = true;
    state->pause_start_ns = state->timer.now_ns(state->timer.user_data);
  }
}

void iree_bench_state_resume(iree_bench_state_t* state) {
  if (state->paused) {
    state->paused = false;
    uint64_t now = state->timer.now_ns(state->timer.user_data);
    state->paused_duration_ns += now - state->pause_start_ns;
  }
}

void iree_bench_state_set_iteration_time(iree_bench_state_t* state,
                                         double seconds) {
  state->uses_manual_time = true;
  state->manual_time_ns += seconds * 1e9;
}

void iree_bench_state_set_bytes(iree_bench_state_t* state, int64_t bytes) {
  state->bytes_processed = bytes;
}

void iree_bench_state_set_items(iree_bench_state_t* state, int64_t items) {
  state->items_processed = items;
}

void iree_bench_state_add_counter(iree_bench_state_t* state, const char* name,
                                  double value,
                                  iree_bench_counter_flags_t flags) {
  if (state->counter_count < IREE_BENCH_MAX_COUNTERS) {
    state->counters[state->counter_count].name = name;
    state->counters[state->counter_count].value = value;
    state->counters[state->counter_count].flags = flags;
    state->counter_count++;
  }
}

void iree_bench_state_skip(iree_bench_state_t* state, const char* message) {
  state->skipped = true;
  state->skip_message = message;
}

uint64_t iree_bench_state_iterations(const iree_bench_state_t* state) {
  return state->total_iterations;
}
