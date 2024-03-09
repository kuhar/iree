#! /usr/bin/env bash

set -euo pipefail

readonly INPUT="$(realpath "$1")"
readonly DEVICE="$2"
shift 2

readonly NAME="$(basename "$INPUT" .mlir)"

# ./unet.sh "$INPUT" -o "${NAME}.vmfb" --compile-from=executable-sources > /dev/null 2>&1
timeout 10s ./tools/iree-benchmark-module --device="rocm://${DEVICE}" --module="${INPUT}" \
  --batch_size=1000 --benchmark_repetitions=3 > "benchmark_log_${DEVICE}.out" 2>&1

MEAN_TIME="$(grep real_time_mean "benchmark_log_${DEVICE}.out" | awk '{print $2}')"
printf "%s: Mean Time: %.1f\n" "$(shasum -b "$INPUT")" "$MEAN_TIME"
