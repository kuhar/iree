#! /usr/bin/env bash

set -eou pipefail

readonly INPUT="$1"
readonly DIR="$(dirname "$INPUT")"
readonly BASENAME="$(basename "$INPUT" .mlir)"
readonly OUT="${DIR}/compiled/${BASENAME}.vmfb"

timeout 20s ./unet.sh "$INPUT" -o "$OUT" --compile-from=executable-sources 2>/dev/null || (mv "$INPUT" "$DIR/failed" && exit 1)
tools/iree-dump-module "$OUT" | grep -q 'rocm-hsaco-fb' || (mv "$INPUT" "$DIR/failed" && rm -f "$OUT" && exit 1)
echo "${INPUT}: success"
