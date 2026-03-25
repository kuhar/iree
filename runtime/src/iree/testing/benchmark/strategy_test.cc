// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark/strategy.h"

#include "iree/testing/benchmark/config.h"
#include "iree/testing/benchmark/strategy_auto.h"
#include "iree/testing/benchmark/strategy_fixed.h"
#include "iree/testing/benchmark/strategy_time_budget.h"
#include "iree/testing/gtest.h"

namespace {

//===----------------------------------------------------------------------===//
// AUTO strategy tests
//===----------------------------------------------------------------------===//

TEST(StrategyAutoTest, FastBenchmarkConvergesViaDoubling) {
  // Clock resolution = 100ns, multiple = 1000, target = 100,000ns.
  // Benchmark takes 10ns/iter.
  iree_bench_strategy_auto_params_t p = {};
  p.iteration_hint = 1;
  p.epoch_count = 3;
  p.clock_resolution_multiple = 1000;

  iree_bench_strategy_t* s = iree_bench_strategy_auto_create(100, &p);
  ASSERT_NE(s, nullptr);

  iree_bench_action_t a = iree_bench_strategy_begin(s);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_TRIAL);
  EXPECT_EQ(a.iteration_count, 1u);

  // Trial 1: 1 iter * 10ns = 10ns < 100,000ns target -> double.
  a = iree_bench_strategy_observe(s, 1, 10.0);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_TRIAL);
  EXPECT_EQ(a.iteration_count, 2u);

  // Trial 2: 2 * 10ns = 20ns < target -> double.
  a = iree_bench_strategy_observe(s, 2, 20.0);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_TRIAL);
  EXPECT_EQ(a.iteration_count, 4u);

  // Keep doubling: 4->8->16->...->16384.
  uint64_t iters = 4;
  while (iters * 10.0 < 100000.0) {
    a = iree_bench_strategy_observe(s, iters, iters * 10.0);
    EXPECT_EQ(a.type, IREE_BENCH_ACTION_TRIAL);
    iters = a.iteration_count;
  }
  // iters is now 16384. 16384 * 10ns = 163,840ns >= 100,000ns.

  // Epoch 1: passes threshold -> recorded.
  a = iree_bench_strategy_observe(s, iters, iters * 10.0);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_RECORD_AND_CONTINUE);
  EXPECT_EQ(a.iteration_count, iters);

  // Epoch 2.
  a = iree_bench_strategy_observe(s, iters, iters * 10.0);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_RECORD_AND_CONTINUE);

  // Epoch 3 (last).
  a = iree_bench_strategy_observe(s, iters, iters * 10.0);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_DONE);

  size_t count = 0;
  const iree_bench_sample_t* samples = iree_bench_strategy_samples(s, &count);
  EXPECT_EQ(count, 3u);
  for (size_t i = 0; i < count; i++) {
    EXPECT_EQ(samples[i].iterations, iters);
    EXPECT_NEAR(samples[i].real_time_ns, iters * 10.0, 1.0);
  }

  iree_bench_strategy_destroy(s);
}

TEST(StrategyAutoTest, HintSkipsCalibration) {
  // hint=16384 already exceeds target (100*1000=100000), 16384*10=163840.
  iree_bench_strategy_auto_params_t p = {};
  p.iteration_hint = 16384;
  p.epoch_count = 2;
  p.clock_resolution_multiple = 1000;

  iree_bench_strategy_t* s = iree_bench_strategy_auto_create(100, &p);
  iree_bench_action_t a = iree_bench_strategy_begin(s);
  EXPECT_EQ(a.iteration_count, 16384u);

  // First observation already meets target.
  a = iree_bench_strategy_observe(s, 16384, 163840.0);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_RECORD_AND_CONTINUE);

  a = iree_bench_strategy_observe(s, 16384, 163840.0);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_DONE);

  iree_bench_strategy_destroy(s);
}

TEST(StrategyAutoTest, SlowBenchmarkNoDoubling) {
  // 1 iteration takes 500,000ns >> 100,000ns target.
  iree_bench_strategy_auto_params_t p = {};
  p.iteration_hint = 1;
  p.epoch_count = 3;
  p.clock_resolution_multiple = 1000;

  iree_bench_strategy_t* s = iree_bench_strategy_auto_create(100, &p);
  iree_bench_action_t a = iree_bench_strategy_begin(s);
  EXPECT_EQ(a.iteration_count, 1u);

  // 1 iter takes 500,000ns >> target. Should immediately record.
  a = iree_bench_strategy_observe(s, 1, 500000.0);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_RECORD_AND_CONTINUE);
  EXPECT_EQ(a.iteration_count, 1u);

  a = iree_bench_strategy_observe(s, 1, 510000.0);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_RECORD_AND_CONTINUE);

  a = iree_bench_strategy_observe(s, 1, 490000.0);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_DONE);

  iree_bench_strategy_destroy(s);
}

TEST(StrategyAutoTest, ZeroElapsedHitsMaxIterations) {
  iree_bench_strategy_auto_params_t p = {};
  p.iteration_hint = 1;
  p.epoch_count = 1;
  p.clock_resolution_multiple = 1000;
  p.max_iterations = 1 << 20;  // 1M cap.

  iree_bench_strategy_t* s = iree_bench_strategy_auto_create(100, &p);
  iree_bench_action_t a = iree_bench_strategy_begin(s);

  // Every observation reports 0ns. Strategy should keep doubling
  // until it hits the max_iterations cap, then give up and record.
  for (int i = 0; i < 30; i++) {
    a = iree_bench_strategy_observe(s, a.iteration_count, 0.0);
    if (a.type != IREE_BENCH_ACTION_TRIAL) break;
  }
  // Should have hit the cap and forced an epoch.
  EXPECT_NE(a.type, IREE_BENCH_ACTION_TRIAL);

  iree_bench_strategy_destroy(s);
}

//===----------------------------------------------------------------------===//
// FIXED strategy tests
//===----------------------------------------------------------------------===//

TEST(StrategyFixedTest, ExactIterations) {
  iree_bench_strategy_t* s = iree_bench_strategy_fixed_create(42, 2);
  ASSERT_NE(s, nullptr);

  iree_bench_action_t a = iree_bench_strategy_begin(s);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_TRIAL);
  EXPECT_EQ(a.iteration_count, 42u);

  // Time doesn't matter for FIXED.
  a = iree_bench_strategy_observe(s, 42, 999.0);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_RECORD_AND_CONTINUE);
  EXPECT_EQ(a.iteration_count, 42u);

  a = iree_bench_strategy_observe(s, 42, 1.0);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_DONE);

  size_t count = 0;
  const iree_bench_sample_t* samples = iree_bench_strategy_samples(s, &count);
  EXPECT_EQ(count, 2u);
  EXPECT_EQ(samples[0].iterations, 42u);
  EXPECT_DOUBLE_EQ(samples[0].real_time_ns, 999.0);
  EXPECT_EQ(samples[1].iterations, 42u);
  EXPECT_DOUBLE_EQ(samples[1].real_time_ns, 1.0);

  iree_bench_strategy_destroy(s);
}

TEST(StrategyFixedTest, SingleEpochDefault) {
  // epoch_count=0 defaults to 1.
  iree_bench_strategy_t* s = iree_bench_strategy_fixed_create(100, 0);
  iree_bench_action_t a = iree_bench_strategy_begin(s);
  EXPECT_EQ(a.iteration_count, 100u);

  a = iree_bench_strategy_observe(s, 100, 500.0);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_DONE);

  size_t count = 0;
  iree_bench_strategy_samples(s, &count);
  EXPECT_EQ(count, 1u);

  iree_bench_strategy_destroy(s);
}

//===----------------------------------------------------------------------===//
// TIME_BUDGET strategy tests
//===----------------------------------------------------------------------===//

TEST(StrategyTimeBudgetTest, CollectsUntilBudgetMet) {
  iree_bench_strategy_t* s =
      iree_bench_strategy_time_budget_create(100, 1000000, 2);
  ASSERT_NE(s, nullptr);

  iree_bench_action_t a = iree_bench_strategy_begin(s);
  EXPECT_EQ(a.type, IREE_BENCH_ACTION_TRIAL);

  // Calibration: double until we exceed calibration target (100*1000=100000ns).
  uint64_t iters = a.iteration_count;
  while (true) {
    double elapsed = iters * 10.0;  // 10ns/iter.
    a = iree_bench_strategy_observe(s, iters, elapsed);
    if (a.type != IREE_BENCH_ACTION_TRIAL) break;
    iters = a.iteration_count;
  }

  // Now in collection phase. Keep observing until DONE.
  while (a.type != IREE_BENCH_ACTION_DONE) {
    a = iree_bench_strategy_observe(s, a.iteration_count,
                                    a.iteration_count * 10.0);
  }

  // Must have at least min_epoch_count (2) epochs.
  size_t count = 0;
  iree_bench_strategy_samples(s, &count);
  EXPECT_GE(count, 2u);

  iree_bench_strategy_destroy(s);
}

//===----------------------------------------------------------------------===//
// Strategy resolution tests
//===----------------------------------------------------------------------===//

TEST(StrategyResolutionTest, DefaultIsAuto) {
  iree_bench_config_t config = iree_bench_config_default();
  iree_bench_def_t def = iree_bench_def_default(nullptr);
  EXPECT_EQ(iree_bench_resolve_strategy_type(&config, &def),
            IREE_BENCH_ITER_AUTO);
}

TEST(StrategyResolutionTest, PerBenchmarkFixed) {
  iree_bench_config_t config = iree_bench_config_default();
  iree_bench_def_t def = iree_bench_def_default(nullptr);
  def.iter_strategy = IREE_BENCH_ITER_FIXED;
  def.iteration_count = 500;
  EXPECT_EQ(iree_bench_resolve_strategy_type(&config, &def),
            IREE_BENCH_ITER_FIXED);
}

TEST(StrategyResolutionTest, GlobalOverrideWins) {
  iree_bench_config_t config = iree_bench_config_default();
  config.iteration_count = 1000;
  iree_bench_def_t def = iree_bench_def_default(nullptr);
  def.iter_strategy = IREE_BENCH_ITER_TIME_BUDGET;
  def.time_budget_ns = 500000;
  // Global iteration_count > 0 -> FIXED, overriding per-benchmark TIME_BUDGET.
  EXPECT_EQ(iree_bench_resolve_strategy_type(&config, &def),
            IREE_BENCH_ITER_FIXED);
}

TEST(StrategyResolutionTest, GlobalTimeBudget) {
  iree_bench_config_t config = iree_bench_config_default();
  config.time_budget_ns = 2000000000;  // 2s.
  iree_bench_def_t def = iree_bench_def_default(nullptr);
  EXPECT_EQ(iree_bench_resolve_strategy_type(&config, &def),
            IREE_BENCH_ITER_TIME_BUDGET);
}

}  // namespace
