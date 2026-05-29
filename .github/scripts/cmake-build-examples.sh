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
# Configure and link all C++ examples against prebuilt libnntile (CI).
#
# @version 1.1.0
set -euo pipefail

build_dir="${1:-build}"

chmod +x .github/scripts/nntile-prebuilt-lib-path.sh
_lib=$(.github/scripts/nntile-prebuilt-lib-path.sh "$build_dir")

cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF \
    -DNNTILE_PRESET=full -DNNTILE_PREBUILT_LIBRARY="${_lib}" \
    -DBUILD_NNTILE=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=ON \
    -DBUILD_PYTHON_WRAPPERS=OFF -DBUILD_TESTS_PYTORCH=OFF

echo "=== ninja plan (examples only; libnntile is prebuilt) ==="
ninja -C "$build_dir" -n nntile_all_examples 2>&1 | tee /tmp/examples-ninja-plan.txt
if grep -E 'CXX_COMPILER|CUDA_COMPILER' /tmp/examples-ninja-plan.txt \
    | grep -qE 'nntile/src/'; then
    echo "unexpected library source compile (use prebuilt libnntile only)" >&2
    grep -E 'CXX_COMPILER|CUDA_COMPILER' /tmp/examples-ninja-plan.txt \
        | grep 'nntile/src/' >&2 || true
    exit 1
fi

cmake --build "$build_dir" --target nntile_all_examples -j"$(nproc)"
