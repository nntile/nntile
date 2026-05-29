#!/usr/bin/env bash
# Configure and build Catch2 once for CI (see build-test-prerequisites job).
set -euo pipefail

build_dir="${1:-build}"

common_flags=(
    -S .
    -B "$build_dir"
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DUSE_CUDA=OFF
    -DBUILD_NNTILE=OFF
    -DBUILD_TESTS=OFF
    -DBUILD_EXAMPLES=OFF
    -DBUILD_PYTHON_WRAPPERS=OFF
)

cmake "${common_flags[@]}"
cmake --build "$build_dir" --target nntile_test_prerequisites -j"$(nproc)"
