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
# Configure and build libtorch_nntile C++ examples (local or CI helper).
#
# @version 1.1.0
set -euo pipefail

build_dir="${1:-build}"

TORCH_PREFIX="${TORCH_PREFIX:-$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)' 2>/dev/null || true)}"

cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF \
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER:-gcc}" \
    -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-g++}" \
    -DBUILD_NNTILE=ON -DBUILD_TESTS=OFF \
    -DBUILD_TORCH_NNTILE=ON -DBUILD_TORCH_NNTILE_EXAMPLES=ON \
    ${TORCH_PREFIX:+-DCMAKE_PREFIX_PATH="${TORCH_PREFIX}"}

cmake --build "$build_dir" --target torch_nntile -j"$(nproc)"
