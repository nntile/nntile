#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/verify-catch2-deps-for-compile-check.sh
#
# Verify build/_deps has Catch2 sources and generated headers (no Catch2 build here).
#
# @version 1.1.0
set -euo pipefail

build_dir="${1:-build}"
src="${build_dir}/_deps/catch2-src/src/catch2/catch_test_macros.hpp"
gen="${build_dir}/_deps/catch2-build/generated-includes/catch2/catch_user_config.hpp"

missing=0
for path in "$src" "$gen"; do
    if [ ! -f "$path" ]; then
        echo "missing: $path" >&2
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "Catch2 deps incomplete; run build-test-prerequisites first" >&2
    exit 1
fi
echo "ok: Catch2 headers ready under ${build_dir}/_deps"
