// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "iree/testing/benchmark/reporter.h"

//===----------------------------------------------------------------------===//
// JSON reporter implementation
//===----------------------------------------------------------------------===//

static void iree_bench_json_begin(void* user_data,
                                  const iree_bench_config_t* config) {
  iree_bench_json_reporter_state_t* state =
      (iree_bench_json_reporter_state_t*)user_data;
  fprintf(state->output, "{\n  \"benchmarks\": [\n");
  state->first_result = true;
}

// Writes a JSON-safe string (escaping special characters).
static void iree_bench_json_write_string(FILE* f, const char* s) {
  fputc('"', f);
  for (const char* p = s; *p; ++p) {
    switch (*p) {
      case '"':
        fputs("\\\"", f);
        break;
      case '\\':
        fputs("\\\\", f);
        break;
      case '\n':
        fputs("\\n", f);
        break;
      case '\r':
        fputs("\\r", f);
        break;
      case '\t':
        fputs("\\t", f);
        break;
      default:
        fputc(*p, f);
        break;
    }
  }
  fputc('"', f);
}

static void iree_bench_json_report(void* user_data,
                                   const iree_bench_result_t* result) {
  iree_bench_json_reporter_state_t* state =
      (iree_bench_json_reporter_state_t*)user_data;
  FILE* f = state->output;

  if (!state->first_result) {
    fprintf(f, ",\n");
  }
  state->first_result = false;

  fprintf(f, "    {\n");
  fprintf(f, "      \"name\": ");
  iree_bench_json_write_string(f, result->name);
  fprintf(f, ",\n");

  if (result->skipped) {
    fprintf(f, "      \"skipped\": true,\n");
    fprintf(f, "      \"skip_message\": ");
    iree_bench_json_write_string(
        f, result->skip_message ? result->skip_message : "");
    fprintf(f, "\n");
    fprintf(f, "    }");
    return;
  }

  // Raw samples.
  fprintf(f, "      \"samples\": [\n");
  for (size_t i = 0; i < result->sample_count; ++i) {
    const iree_bench_sample_t* s = &result->samples[i];
    fprintf(f,
            "        {\"iterations\": %" PRIu64
            ", \"real_time_ns\": %.1f, \"cpu_time_ns\": %.1f}",
            s->iterations, s->real_time_ns, s->cpu_time_ns);
    if (i + 1 < result->sample_count) fprintf(f, ",");
    fprintf(f, "\n");
  }
  fprintf(f, "      ],\n");

  // Stats.
  fprintf(f, "      \"stats\": {\n");
  fprintf(f, "        \"median_ns\": %.3f,\n", result->median_ns);
  fprintf(f, "        \"mean_ns\": %.3f,\n", result->mean_ns);
  fprintf(f, "        \"mad_ns\": %.3f,\n", result->mad_ns);
  fprintf(f, "        \"stddev_ns\": %.3f,\n", result->stddev_ns);
  fprintf(f, "        \"min_ns\": %.3f,\n", result->min_ns);
  fprintf(f, "        \"max_ns\": %.3f,\n", result->max_ns);
  fprintf(f, "        \"mape\": %.6f\n", result->mape);
  fprintf(f, "      }");

  // Counters.
  if (result->counter_count > 0) {
    fprintf(f, ",\n      \"counters\": {");
    for (size_t i = 0; i < result->counter_count; ++i) {
      if (i > 0) fprintf(f, ",");
      fprintf(f, "\n        ");
      iree_bench_json_write_string(f, result->counters[i].name);
      fprintf(f, ": %.6g", result->counters[i].value);
    }
    fprintf(f, "\n      }");
  }

  // Throughput.
  if (result->bytes_processed > 0 || result->items_processed > 0) {
    fprintf(f, ",\n      \"throughput\": {");
    bool first = true;
    if (result->bytes_processed > 0) {
      fprintf(f, "\n        \"bytes_processed\": %" PRId64,
              result->bytes_processed);
      first = false;
    }
    if (result->items_processed > 0) {
      if (!first) fprintf(f, ",");
      fprintf(f, "\n        \"items_processed\": %" PRId64,
              result->items_processed);
    }
    fprintf(f, "\n      }");
  }

  fprintf(f, "\n    }");
}

static void iree_bench_json_end(void* user_data) {
  iree_bench_json_reporter_state_t* state =
      (iree_bench_json_reporter_state_t*)user_data;
  fprintf(state->output, "\n  ]\n}\n");
  fflush(state->output);
}

void iree_bench_json_reporter_init(iree_bench_json_reporter_state_t* state,
                                   FILE* output) {
  memset(state, 0, sizeof(*state));
  state->output = output;
  state->first_result = true;
}

iree_bench_reporter_t iree_bench_reporter_json(
    iree_bench_json_reporter_state_t* state) {
  iree_bench_reporter_t reporter;
  reporter.begin = iree_bench_json_begin;
  reporter.report = iree_bench_json_report;
  reporter.end = iree_bench_json_end;
  reporter.user_data = state;
  return reporter;
}
