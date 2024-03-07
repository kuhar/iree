#! /usr/bin/env bash

set -xeuo pipefail

readonly INPUT="$(realpath "$1")"
shift

tools/iree-compile \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target-chip=gfx942 \
    --iree-rocm-link-bc=true \
    --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
    --iree-global-opt-propagate-transposes=true \
    --iree-opt-outer-dim-concat=true \
    --iree-opt-const-eval=false \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-rocm-waves-per-eu=2 \
    --iree-preprocessing-pass-pipeline="builtin.module(iree-preprocessing-transpose-convolution-pipeline)" \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-codegen-transform-dialect-library="$(realpath ~/iree/models/attention_mfma_transform_64_spec.mlir)" \
    "$INPUT" $@

# --iree-hal-dump-executable-files-to=dump-unet \
