// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_BENCHMARK_RUNNER_H_
#define IREE_TESTING_BENCHMARK_RUNNER_H_

#include "iree/testing/benchmark/config.h"
#include "iree/testing/benchmark/registry.h"
#include "iree/testing/benchmark/reporter.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Runner: orchestrates strategy <-> benchmark function
//===----------------------------------------------------------------------===//

// Runs all registered benchmarks matching the config filter.
// For each benchmark:
//   1. Resolves strategy from config + def + clock resolution.
//   2. Creates state wrapping strategy + timer.
//   3. Calls the benchmark function.
//   4. Computes statistics over collected samples.
//   5. Emits result to reporter(s).
void iree_bench_run(const iree_bench_config_t* config,
                    iree_bench_registry_t* registry);

// Runs a single benchmark entry. Used internally and for testing.
// |clock_resolution_ns| is pre-measured; pass a synthetic value in tests.
void iree_bench_run_one(const iree_bench_config_t* config,
                        const iree_bench_registry_entry_t* entry,
                        uint64_t clock_resolution_ns,
                        const iree_bench_reporter_t* reporter);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TESTING_BENCHMARK_RUNNER_H_
