// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark/reporter.h"

#include <cstdio>
#include <cstring>
#include <string>

#include "iree/testing/benchmark/config.h"
#include "iree/testing/benchmark/result.h"
#include "iree/testing/benchmark/strategy.h"
#include "iree/testing/gtest.h"

namespace {

// Helper to capture reporter output to a string.
class OutputCapture {
 public:
  OutputCapture() { f_ = tmpfile(); }
  ~OutputCapture() {
    if (f_) fclose(f_);
  }
  FILE* file() { return f_; }
  std::string contents() {
    fflush(f_);
    fseek(f_, 0, SEEK_END);
    long size = ftell(f_);
    fseek(f_, 0, SEEK_SET);
    std::string result(size, '\0');
    fread(&result[0], 1, size, f_);
    return result;
  }

 private:
  FILE* f_;
};

static iree_bench_result_t make_test_result(const char* name,
                                            const iree_bench_sample_t* samples,
                                            size_t sample_count,
                                            double median_ns, double mad_ns,
                                            double mean_ns, double mape) {
  iree_bench_result_t r;
  memset(&r, 0, sizeof(r));
  r.name = name;
  r.unit = IREE_BENCH_UNIT_DEFAULT;
  r.samples = samples;
  r.sample_count = sample_count;
  r.median_ns = median_ns;
  r.mad_ns = mad_ns;
  r.mean_ns = mean_ns;
  r.mape = mape;
  return r;
}

//===----------------------------------------------------------------------===//
// Console reporter tests
//===----------------------------------------------------------------------===//

TEST(ConsoleReporterTest, PrintsHeader) {
  OutputCapture capture;
  iree_bench_console_reporter_state_t state;
  iree_bench_console_reporter_init(&state, capture.file());
  iree_bench_reporter_t reporter = iree_bench_reporter_console(&state);

  iree_bench_config_t config = iree_bench_config_default();
  reporter.begin(reporter.user_data, &config);

  std::string output = capture.contents();
  EXPECT_NE(output.find("Benchmark"), std::string::npos);
  EXPECT_NE(output.find("Median"), std::string::npos);
  EXPECT_NE(output.find("MAD"), std::string::npos);
}

TEST(ConsoleReporterTest, AutoScalesUnits) {
  OutputCapture capture;
  iree_bench_console_reporter_state_t state;
  iree_bench_console_reporter_init(&state, capture.file());
  iree_bench_reporter_t reporter = iree_bench_reporter_console(&state);

  iree_bench_config_t config = iree_bench_config_default();
  reporter.begin(reporter.user_data, &config);

  // median=2340ns -> should show as us.
  iree_bench_sample_t samples[1] = {{4096, 2340.0 * 4096, 0, 0}};
  iree_bench_result_t result =
      make_test_result("BM_Test", samples, 1, 2340.0, 40.0, 2350.0, 0.017);
  reporter.report(reporter.user_data, &result);

  std::string output = capture.contents();
  EXPECT_NE(output.find("us"), std::string::npos);
}

TEST(ConsoleReporterTest, ShowsSkipped) {
  OutputCapture capture;
  iree_bench_console_reporter_state_t state;
  iree_bench_console_reporter_init(&state, capture.file());
  iree_bench_reporter_t reporter = iree_bench_reporter_console(&state);

  iree_bench_config_t config = iree_bench_config_default();
  reporter.begin(reporter.user_data, &config);

  iree_bench_result_t result;
  memset(&result, 0, sizeof(result));
  result.name = "BM_Skipped";
  result.skipped = true;
  result.skip_message = "not supported";
  reporter.report(reporter.user_data, &result);

  std::string output = capture.contents();
  EXPECT_NE(output.find("not supported"), std::string::npos);
}

//===----------------------------------------------------------------------===//
// JSON reporter tests
//===----------------------------------------------------------------------===//

TEST(JsonReporterTest, ValidStructure) {
  OutputCapture capture;
  iree_bench_json_reporter_state_t state;
  iree_bench_json_reporter_init(&state, capture.file());
  iree_bench_reporter_t reporter = iree_bench_reporter_json(&state);

  iree_bench_config_t config = iree_bench_config_default();
  reporter.begin(reporter.user_data, &config);

  iree_bench_sample_t samples[2] = {
      {1000, 5000.0, 4800.0, 0},
      {1000, 5100.0, 4900.0, 0},
  };
  iree_bench_result_t result =
      make_test_result("BM_Json", samples, 2, 5.0, 0.05, 5.05, 0.01);
  reporter.report(reporter.user_data, &result);
  reporter.end(reporter.user_data);

  std::string output = capture.contents();
  EXPECT_NE(output.find("\"benchmarks\""), std::string::npos);
  EXPECT_NE(output.find("\"BM_Json\""), std::string::npos);
  EXPECT_NE(output.find("\"samples\""), std::string::npos);
  EXPECT_NE(output.find("\"stats\""), std::string::npos);
  EXPECT_NE(output.find("\"median_ns\""), std::string::npos);
}

TEST(JsonReporterTest, MultipleBenchmarks) {
  OutputCapture capture;
  iree_bench_json_reporter_state_t state;
  iree_bench_json_reporter_init(&state, capture.file());
  iree_bench_reporter_t reporter = iree_bench_reporter_json(&state);

  iree_bench_config_t config = iree_bench_config_default();
  reporter.begin(reporter.user_data, &config);

  iree_bench_sample_t s1[1] = {{100, 1000.0, 0, 0}};
  iree_bench_result_t r1 =
      make_test_result("BM_A", s1, 1, 10.0, 1.0, 10.0, 0.01);
  reporter.report(reporter.user_data, &r1);

  iree_bench_sample_t s2[1] = {{200, 2000.0, 0, 0}};
  iree_bench_result_t r2 =
      make_test_result("BM_B", s2, 1, 10.0, 1.0, 10.0, 0.01);
  reporter.report(reporter.user_data, &r2);

  reporter.end(reporter.user_data);

  std::string output = capture.contents();
  EXPECT_NE(output.find("\"BM_A\""), std::string::npos);
  EXPECT_NE(output.find("\"BM_B\""), std::string::npos);
}

TEST(JsonReporterTest, SkippedBenchmark) {
  OutputCapture capture;
  iree_bench_json_reporter_state_t state;
  iree_bench_json_reporter_init(&state, capture.file());
  iree_bench_reporter_t reporter = iree_bench_reporter_json(&state);

  iree_bench_config_t config = iree_bench_config_default();
  reporter.begin(reporter.user_data, &config);

  iree_bench_result_t result;
  memset(&result, 0, sizeof(result));
  result.name = "BM_Skip";
  result.skipped = true;
  result.skip_message = "not available";
  reporter.report(reporter.user_data, &result);

  reporter.end(reporter.user_data);

  std::string output = capture.contents();
  EXPECT_NE(output.find("\"skipped\": true"), std::string::npos);
}

}  // namespace
