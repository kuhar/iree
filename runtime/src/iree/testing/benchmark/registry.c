// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark/registry.h"

#include <stdlib.h>
#include <string.h>

void iree_bench_registry_init(iree_bench_registry_t* registry) {
  memset(registry, 0, sizeof(*registry));
}

void iree_bench_registry_destroy(iree_bench_registry_t* registry) {
  for (size_t i = 0; i < registry->count; ++i) {
    free((void*)registry->entries[i].name);
  }
  free(registry->entries);
  memset(registry, 0, sizeof(*registry));
}

size_t iree_bench_registry_add(iree_bench_registry_t* registry,
                               const char* name, const iree_bench_def_t* def) {
  if (registry->count >= registry->capacity) {
    size_t new_capacity = registry->capacity == 0 ? 16 : registry->capacity * 2;
    iree_bench_registry_entry_t* new_entries =
        (iree_bench_registry_entry_t*)realloc(
            registry->entries, new_capacity * sizeof(*new_entries));
    if (!new_entries) return (size_t)-1;
    registry->entries = new_entries;
    registry->capacity = new_capacity;
  }

  size_t index = registry->count;
  size_t name_len = strlen(name);
  char* name_copy = (char*)malloc(name_len + 1);
  if (!name_copy) return (size_t)-1;
  memcpy(name_copy, name, name_len + 1);

  registry->entries[index].name = name_copy;
  registry->entries[index].def = *def;
  registry->count++;
  return index;
}

size_t iree_bench_registry_count(const iree_bench_registry_t* registry) {
  return registry->count;
}

const iree_bench_registry_entry_t* iree_bench_registry_get(
    const iree_bench_registry_t* registry, size_t index) {
  if (index >= registry->count) return NULL;
  return &registry->entries[index];
}

bool iree_bench_registry_matches_filter(const char* name, const char* filter) {
  if (!filter) return true;
  return strstr(name, filter) != NULL;
}
