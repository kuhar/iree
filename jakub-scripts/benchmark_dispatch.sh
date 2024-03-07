#! /usr/bin/env bash

set -xeuo pipefail

readonly INPUT="$(realpath "$1")"
readonly DEVICE="$2"
shift 2

readonly NAME="$(basename "$INPUT" .mlir)"

./unet.sh "$INPUT" -o "${NAME}.vmfb" --compile-from=executable-sources > /dev/null
./tools/iree-benchmark-module --device="rocm://${DEVICE}" --module="${NAME}.vmfb" --benchmark_repetitions=5 | tee benchmark_log.out

MEAN_TIME="$(grep real_time_mean benchmark_log.out | awk '{print $2}')"
printf "Mean Time: %.1f\n" "$MEAN_TIME"
