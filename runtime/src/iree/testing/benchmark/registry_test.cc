// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark/registry.h"

#include "iree/testing/gtest.h"

namespace {

TEST(RegistryTest, InitAndDestroy) {
  iree_bench_registry_t registry;
  iree_bench_registry_init(&registry);
  EXPECT_EQ(iree_bench_registry_count(&registry), 0u);
  iree_bench_registry_destroy(&registry);
}

TEST(RegistryTest, AddAndGet) {
  iree_bench_registry_t registry;
  iree_bench_registry_init(&registry);

  iree_bench_def_t def = iree_bench_def_default(nullptr);
  def.iteration_count = 42;

  size_t idx = iree_bench_registry_add(&registry, "BM_Test", &def);
  EXPECT_EQ(idx, 0u);
  EXPECT_EQ(iree_bench_registry_count(&registry), 1u);

  const iree_bench_registry_entry_t* entry =
      iree_bench_registry_get(&registry, 0);
  ASSERT_NE(entry, nullptr);
  EXPECT_STREQ(entry->name, "BM_Test");
  EXPECT_EQ(entry->def.iteration_count, 42u);

  iree_bench_registry_destroy(&registry);
}

TEST(RegistryTest, MultipleEntries) {
  iree_bench_registry_t registry;
  iree_bench_registry_init(&registry);

  iree_bench_def_t def = iree_bench_def_default(nullptr);
  iree_bench_registry_add(&registry, "BM_First", &def);
  iree_bench_registry_add(&registry, "BM_Second", &def);
  iree_bench_registry_add(&registry, "BM_Third", &def);

  EXPECT_EQ(iree_bench_registry_count(&registry), 3u);
  EXPECT_STREQ(iree_bench_registry_get(&registry, 0)->name, "BM_First");
  EXPECT_STREQ(iree_bench_registry_get(&registry, 1)->name, "BM_Second");
  EXPECT_STREQ(iree_bench_registry_get(&registry, 2)->name, "BM_Third");

  iree_bench_registry_destroy(&registry);
}

TEST(RegistryTest, OutOfBoundsReturnsNull) {
  iree_bench_registry_t registry;
  iree_bench_registry_init(&registry);
  EXPECT_EQ(iree_bench_registry_get(&registry, 0), nullptr);
  EXPECT_EQ(iree_bench_registry_get(&registry, 100), nullptr);
  iree_bench_registry_destroy(&registry);
}

TEST(RegistryTest, FilterMatching) {
  EXPECT_TRUE(iree_bench_registry_matches_filter("BM_Dispatch/1/8", nullptr));
  EXPECT_TRUE(
      iree_bench_registry_matches_filter("BM_Dispatch/1/8", "Dispatch"));
  EXPECT_TRUE(iree_bench_registry_matches_filter("BM_Dispatch/1/8", "1/8"));
  EXPECT_FALSE(
      iree_bench_registry_matches_filter("BM_Dispatch/1/8", "NonExistent"));
  EXPECT_TRUE(iree_bench_registry_matches_filter("BM_Dispatch/1/8", "BM_"));
}

TEST(RegistryTest, GrowsBeyondInitialCapacity) {
  iree_bench_registry_t registry;
  iree_bench_registry_init(&registry);

  iree_bench_def_t def = iree_bench_def_default(nullptr);
  for (int i = 0; i < 100; ++i) {
    char name[32];
    snprintf(name, sizeof(name), "BM_%d", i);
    iree_bench_registry_add(&registry, name, &def);
  }
  EXPECT_EQ(iree_bench_registry_count(&registry), 100u);

  iree_bench_registry_destroy(&registry);
}

}  // namespace
