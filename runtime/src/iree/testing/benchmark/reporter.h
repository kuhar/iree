// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_BENCHMARK_REPORTER_H_
#define IREE_TESTING_BENCHMARK_REPORTER_H_

#include <stdbool.h>
#include <stdio.h>

#include "iree/testing/benchmark/config.h"
#include "iree/testing/benchmark/result.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Reporter interface
//===----------------------------------------------------------------------===//

struct iree_bench_reporter_t {
  // Called once before any benchmarks run.
  void (*begin)(void* user_data, const iree_bench_config_t* config);

  // Called once per completed benchmark.
  void (*report)(void* user_data, const iree_bench_result_t* result);

  // Called once after all benchmarks complete.
  void (*end)(void* user_data);

  void* user_data;
};

//===----------------------------------------------------------------------===//
// Console reporter
//===----------------------------------------------------------------------===//

// State for the console reporter.
typedef struct {
  FILE* output;
  bool use_color;
} iree_bench_console_reporter_state_t;

// Initializes a console reporter state.
void iree_bench_console_reporter_init(
    iree_bench_console_reporter_state_t* state, FILE* output);

// Returns a reporter vtable backed by console output.
iree_bench_reporter_t iree_bench_reporter_console(
    iree_bench_console_reporter_state_t* state);

//===----------------------------------------------------------------------===//
// JSON reporter
//===----------------------------------------------------------------------===//

typedef struct {
  FILE* output;
  bool first_result;
} iree_bench_json_reporter_state_t;

// Initializes a JSON reporter state.
void iree_bench_json_reporter_init(iree_bench_json_reporter_state_t* state,
                                   FILE* output);

// Returns a reporter vtable backed by JSON output.
iree_bench_reporter_t iree_bench_reporter_json(
    iree_bench_json_reporter_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TESTING_BENCHMARK_REPORTER_H_
