#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/cmake-build-examples.sh
#
# Configure and build all C++ examples (local or CI helper).
#
# @version 1.1.0
set -euo pipefail

build_dir="${1:-build}"

cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF \
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER:-gcc}" \
    -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-g++}" \
    -DNNTILE_PRESET=full \
    -DBUILD_NNTILE=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=ON \
    -DBUILD_PYTHON_WRAPPERS=OFF -DBUILD_TESTS_PYTORCH=OFF

cmake --build "$build_dir" --target nntile_all_examples -j"$(nproc)"
