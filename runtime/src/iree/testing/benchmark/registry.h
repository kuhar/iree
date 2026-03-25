// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_BENCHMARK_REGISTRY_H_
#define IREE_TESTING_BENCHMARK_REGISTRY_H_

#include <stdbool.h>
#include <stddef.h>

#include "iree/testing/benchmark/config.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Registry: dynamic array of benchmark definitions
//===----------------------------------------------------------------------===//

// An entry in the registry: a name + definition.
typedef struct {
  const char* name;  // Owned copy.
  iree_bench_def_t def;
} iree_bench_registry_entry_t;

typedef struct {
  iree_bench_registry_entry_t* entries;
  size_t count;
  size_t capacity;
} iree_bench_registry_t;

// Initializes an empty registry.
void iree_bench_registry_init(iree_bench_registry_t* registry);

// Destroys the registry and frees all owned memory.
void iree_bench_registry_destroy(iree_bench_registry_t* registry);

// Registers a benchmark. The name is copied. Returns the index of the entry.
size_t iree_bench_registry_add(iree_bench_registry_t* registry,
                               const char* name, const iree_bench_def_t* def);

// Returns the number of registered benchmarks.
size_t iree_bench_registry_count(const iree_bench_registry_t* registry);

// Returns the entry at the given index. NULL if out of bounds.
const iree_bench_registry_entry_t* iree_bench_registry_get(
    const iree_bench_registry_t* registry, size_t index);

// Returns true if |name| contains |filter| as a substring.
// If |filter| is NULL, always returns true.
bool iree_bench_registry_matches_filter(const char* name, const char* filter);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TESTING_BENCHMARK_REGISTRY_H_
