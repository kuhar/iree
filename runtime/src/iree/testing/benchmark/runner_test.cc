// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark/runner.h"

#include <cstring>
#include <vector>

#include "iree/testing/benchmark/config.h"
#include "iree/testing/benchmark/registry.h"
#include "iree/testing/benchmark/reporter.h"
#include "iree/testing/benchmark/result.h"
#include "iree/testing/benchmark/state.h"
#include "iree/testing/gtest.h"

namespace {

// A collecting reporter for testing. Deep-copies samples since the strategy
// (which owns the sample buffer) is destroyed after the report callback.
struct TestReporter {
  std::vector<iree_bench_result_t> results;
  std::vector<std::vector<iree_bench_sample_t>> sample_storage;
  bool began = false;
  bool ended = false;
};

static void test_reporter_begin(void* user_data,
                                const iree_bench_config_t* config) {
  static_cast<TestReporter*>(user_data)->began = true;
}

static void test_reporter_report(void* user_data,
                                 const iree_bench_result_t* result) {
  auto* r = static_cast<TestReporter*>(user_data);
  iree_bench_result_t copy = *result;
  // Deep-copy samples so they survive strategy destruction.
  r->sample_storage.emplace_back(result->samples,
                                 result->samples + result->sample_count);
  copy.samples = r->sample_storage.back().data();
  r->results.push_back(copy);
}

static void test_reporter_end(void* user_data) {
  static_cast<TestReporter*>(user_data)->ended = true;
}

static iree_bench_reporter_t make_test_reporter(TestReporter* state) {
  iree_bench_reporter_t r;
  r.begin = test_reporter_begin;
  r.report = test_reporter_report;
  r.end = test_reporter_end;
  r.user_data = state;
  return r;
}

// A trivial benchmark that does nothing.
static iree_status_t trivial_benchmark(const iree_bench_def_t* def,
                                       iree_bench_state_t* state) {
  while (iree_bench_state_keep_running(state, 1)) {
    // Do nothing.
  }
  return iree_ok_status();
}

TEST(RunnerTest, FixedStrategyProducesExactCount) {
  iree_bench_registry_t registry;
  iree_bench_registry_init(&registry);

  iree_bench_def_t def = iree_bench_def_default(nullptr);
  def.iter_strategy = IREE_BENCH_ITER_FIXED;
  def.iteration_count = 100;
  def.run = trivial_benchmark;
  iree_bench_registry_add(&registry, "BM_Trivial", &def);

  TestReporter test_reporter;
  iree_bench_reporter_t reporter = make_test_reporter(&test_reporter);

  iree_bench_config_t config = iree_bench_config_default();
  config.reporter = &reporter;
  // Use a synthetic clock resolution.
  const iree_bench_registry_entry_t* entry =
      iree_bench_registry_get(&registry, 0);
  iree_bench_run_one(&config, entry, /*clock_resolution_ns=*/100, &reporter);

  ASSERT_EQ(test_reporter.results.size(), 1u);
  EXPECT_STREQ(test_reporter.results[0].name, "BM_Trivial");
  EXPECT_FALSE(test_reporter.results[0].skipped);
  EXPECT_EQ(test_reporter.results[0].sample_count, 1u);
  if (test_reporter.results[0].sample_count > 0) {
    EXPECT_EQ(test_reporter.results[0].samples[0].iterations, 100u);
  }

  iree_bench_registry_destroy(&registry);
}

TEST(RunnerTest, AutoStrategyProducesDefaultEpochs) {
  iree_bench_registry_t registry;
  iree_bench_registry_init(&registry);

  iree_bench_def_t def = iree_bench_def_default(nullptr);
  def.run = trivial_benchmark;
  iree_bench_registry_add(&registry, "BM_Auto", &def);

  TestReporter test_reporter;
  iree_bench_reporter_t reporter = make_test_reporter(&test_reporter);

  iree_bench_config_t config = iree_bench_config_default();
  config.reporter = &reporter;

  const iree_bench_registry_entry_t* entry =
      iree_bench_registry_get(&registry, 0);
  iree_bench_run_one(&config, entry, /*clock_resolution_ns=*/100, &reporter);

  ASSERT_EQ(test_reporter.results.size(), 1u);
  // AUTO default is 11 epochs.
  EXPECT_EQ(test_reporter.results[0].sample_count, 11u);
  EXPECT_GT(test_reporter.results[0].median_ns, 0.0);

  iree_bench_registry_destroy(&registry);
}

TEST(RunnerTest, FilterExcludesBenchmarks) {
  iree_bench_registry_t registry;
  iree_bench_registry_init(&registry);

  iree_bench_def_t def = iree_bench_def_default(nullptr);
  def.iter_strategy = IREE_BENCH_ITER_FIXED;
  def.iteration_count = 10;
  def.run = trivial_benchmark;
  iree_bench_registry_add(&registry, "BM_Include", &def);
  iree_bench_registry_add(&registry, "BM_Exclude", &def);

  TestReporter test_reporter;
  iree_bench_reporter_t reporter = make_test_reporter(&test_reporter);

  iree_bench_config_t config = iree_bench_config_default();
  config.filter = "Include";
  config.reporter = &reporter;

  iree_bench_run(&config, &registry);

  EXPECT_TRUE(test_reporter.began);
  EXPECT_TRUE(test_reporter.ended);
  ASSERT_EQ(test_reporter.results.size(), 1u);
  EXPECT_STREQ(test_reporter.results[0].name, "BM_Include");

  iree_bench_registry_destroy(&registry);
}

}  // namespace
