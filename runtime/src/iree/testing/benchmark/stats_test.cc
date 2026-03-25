// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark/stats.h"

#include "iree/testing/gtest.h"

namespace {

TEST(StatsTest, MeanBasic) {
  double values[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  EXPECT_DOUBLE_EQ(iree_bench_stats_mean(values, 5), 3.0);
}

TEST(StatsTest, MeanSingle) {
  double values[] = {42.0};
  EXPECT_DOUBLE_EQ(iree_bench_stats_mean(values, 1), 42.0);
}

TEST(StatsTest, MeanEmpty) {
  EXPECT_DOUBLE_EQ(iree_bench_stats_mean(nullptr, 0), 0.0);
}

TEST(StatsTest, StddevBasic) {
  // Values: 2, 4, 4, 4, 5, 5, 7, 9. Mean = 5. Population stddev = 2.
  double values[] = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
  EXPECT_NEAR(iree_bench_stats_stddev(values, 8), 2.0, 0.01);
}

TEST(StatsTest, StddevSingle) {
  double values[] = {5.0};
  EXPECT_DOUBLE_EQ(iree_bench_stats_stddev(values, 1), 0.0);
}

TEST(StatsTest, MedianOdd) {
  double values[] = {3.0, 1.0, 2.0};
  EXPECT_DOUBLE_EQ(iree_bench_stats_median(values, 3), 2.0);
}

TEST(StatsTest, MedianEven) {
  double values[] = {4.0, 1.0, 3.0, 2.0};
  EXPECT_DOUBLE_EQ(iree_bench_stats_median(values, 4), 2.5);
}

TEST(StatsTest, MedianSingle) {
  double values[] = {7.0};
  EXPECT_DOUBLE_EQ(iree_bench_stats_median(values, 1), 7.0);
}

TEST(StatsTest, MadBasic) {
  // Values: 1, 1, 2, 2, 4, 6, 9
  // Median = 2. Deviations: 1, 1, 0, 0, 2, 4, 7. MAD = median of deviations
  // = 1.
  double values[] = {1.0, 1.0, 2.0, 2.0, 4.0, 6.0, 9.0};
  double scratch[7];
  EXPECT_DOUBLE_EQ(iree_bench_stats_mad(values, 7, scratch), 1.0);
}

TEST(StatsTest, MapeStableValues) {
  // All the same -> MAPE = 0.
  double values[] = {10.0, 10.0, 10.0};
  double scratch[3];
  EXPECT_DOUBLE_EQ(iree_bench_stats_mape(values, 3, scratch), 0.0);
}

TEST(StatsTest, MapeWithVariation) {
  // Values: 9, 10, 11. Median = 10.
  // APE: 1/10, 0, 1/10. Mean APE = 2/30 ≈ 0.0667.
  double values[] = {9.0, 10.0, 11.0};
  double scratch[3];
  EXPECT_NEAR(iree_bench_stats_mape(values, 3, scratch), 0.0667, 0.001);
}

TEST(StatsTest, MinMax) {
  double values[] = {5.0, 2.0, 8.0, 1.0, 9.0};
  EXPECT_DOUBLE_EQ(iree_bench_stats_min(values, 5), 1.0);
  EXPECT_DOUBLE_EQ(iree_bench_stats_max(values, 5), 9.0);
}

TEST(StatsTest, MinMaxSingle) {
  double values[] = {3.0};
  EXPECT_DOUBLE_EQ(iree_bench_stats_min(values, 1), 3.0);
  EXPECT_DOUBLE_EQ(iree_bench_stats_max(values, 1), 3.0);
}

}  // namespace
