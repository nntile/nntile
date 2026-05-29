#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/cmake-build-test-prerequisites.sh
# Configure and build Catch2 once for CI (see build-test-prerequisites job).
#
#
# @version 1.1.0
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
