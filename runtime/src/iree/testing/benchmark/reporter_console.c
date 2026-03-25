// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "iree/testing/benchmark/reporter.h"

#if !defined(IREE_PLATFORM_WINDOWS)
#include <unistd.h>
#endif

//===----------------------------------------------------------------------===//
// Auto-scaling unit selection
//===----------------------------------------------------------------------===//

static void iree_bench_auto_scale(double value_ns, iree_bench_unit_t hint,
                                  double* out_value, const char** out_suffix) {
  if (hint != IREE_BENCH_UNIT_DEFAULT) {
    switch (hint) {
      case IREE_BENCH_UNIT_NANOSECOND:
        *out_value = value_ns;
        *out_suffix = "ns";
        return;
      case IREE_BENCH_UNIT_MICROSECOND:
        *out_value = value_ns / 1e3;
        *out_suffix = "us";
        return;
      case IREE_BENCH_UNIT_MILLISECOND:
        *out_value = value_ns / 1e6;
        *out_suffix = "ms";
        return;
      case IREE_BENCH_UNIT_SECOND:
        *out_value = value_ns / 1e9;
        *out_suffix = "s";
        return;
      default:
        break;
    }
  }

  // Auto-scale: pick unit where value is in [1, 1000).
  if (value_ns < 1e3) {
    *out_value = value_ns;
    *out_suffix = "ns";
  } else if (value_ns < 1e6) {
    *out_value = value_ns / 1e3;
    *out_suffix = "us";
  } else if (value_ns < 1e9) {
    *out_value = value_ns / 1e6;
    *out_suffix = "ms";
  } else {
    *out_value = value_ns / 1e9;
    *out_suffix = "s";
  }
}

//===----------------------------------------------------------------------===//
// Console reporter implementation
//===----------------------------------------------------------------------===//

static void iree_bench_console_begin(void* user_data,
                                     const iree_bench_config_t* config) {
  iree_bench_console_reporter_state_t* state =
      (iree_bench_console_reporter_state_t*)user_data;
  fprintf(state->output, "%-40s %12s %12s %12s %8s %8s\n", "Benchmark",
          "Median", "MAD", "Mean", "Iters", "MAPE");
  fprintf(state->output,
          "----------------------------------------------"
          "----------------------------------------------\n");
  fflush(state->output);
}

static void iree_bench_console_report(void* user_data,
                                      const iree_bench_result_t* result) {
  iree_bench_console_reporter_state_t* state =
      (iree_bench_console_reporter_state_t*)user_data;

  if (result->skipped) {
    fprintf(state->output, "%-40s %s\n", result->name,
            result->skip_message ? result->skip_message : "SKIPPED");
    fflush(state->output);
    return;
  }

  // Determine unit from median, apply consistently to all columns.
  double median_val;
  const char* unit_suffix;
  iree_bench_auto_scale(result->median_ns, result->unit, &median_val,
                        &unit_suffix);
  double scale =
      (result->median_ns > 0.0) ? median_val / result->median_ns : 1.0;
  double mad_val = result->mad_ns * scale;
  double mean_val = result->mean_ns * scale;

  // Determine iteration count from first sample.
  uint64_t iters = 0;
  if (result->sample_count > 0) {
    iters = result->samples[0].iterations;
  }

  // MAPE warning.
  const char* color_start = "";
  const char* color_end = "";
  char mape_buf[32] = "";
  if (result->mape > 0.0) {
    snprintf(mape_buf, sizeof(mape_buf), "%.1f%%", result->mape * 100.0);
    if (result->mape > 0.05 && state->use_color) {
      color_start = "\033[33m";  // Yellow.
      color_end = "\033[0m";
    }
  }

  fprintf(state->output,
          "%-40s %9.2f %-2s %9.2f %-2s %9.2f %-2s %8" PRIu64 " %s%s%s\n",
          result->name, median_val, unit_suffix, mad_val, unit_suffix, mean_val,
          unit_suffix, iters, color_start, mape_buf, color_end);

  // Throughput labels. bytes/items_processed are totals across all iterations
  // (matching Google Benchmark convention), so divide by total epoch time.
  if (result->bytes_processed > 0 && iters > 0) {
    double total_ns = result->median_ns * (double)iters;
    double bytes_per_sec = (double)result->bytes_processed / (total_ns * 1e-9);
    if (bytes_per_sec >= 1e9) {
      fprintf(state->output, "  %.2f GB/s\n", bytes_per_sec / 1e9);
    } else if (bytes_per_sec >= 1e6) {
      fprintf(state->output, "  %.2f MB/s\n", bytes_per_sec / 1e6);
    } else {
      fprintf(state->output, "  %.2f KB/s\n", bytes_per_sec / 1e3);
    }
  }
  if (result->items_processed > 0 && iters > 0) {
    double total_ns = result->median_ns * (double)iters;
    double items_per_sec = (double)result->items_processed / (total_ns * 1e-9);
    if (items_per_sec >= 1e6) {
      fprintf(state->output, "  %.2f M items/s\n", items_per_sec / 1e6);
    } else if (items_per_sec >= 1e3) {
      fprintf(state->output, "  %.2f K items/s\n", items_per_sec / 1e3);
    } else {
      fprintf(state->output, "  %.2f items/s\n", items_per_sec);
    }
  }

  fflush(state->output);
}

static void iree_bench_console_end(void* user_data) { (void)user_data; }

void iree_bench_console_reporter_init(
    iree_bench_console_reporter_state_t* state, FILE* output) {
  memset(state, 0, sizeof(*state));
  state->output = output;
#if !defined(IREE_PLATFORM_WINDOWS)
  state->use_color = isatty(fileno(output)) != 0;
#else
  state->use_color = false;
#endif
}

iree_bench_reporter_t iree_bench_reporter_console(
    iree_bench_console_reporter_state_t* state) {
  iree_bench_reporter_t reporter;
  reporter.begin = iree_bench_console_begin;
  reporter.report = iree_bench_console_report;
  reporter.end = iree_bench_console_end;
  reporter.user_data = state;
  return reporter;
}
