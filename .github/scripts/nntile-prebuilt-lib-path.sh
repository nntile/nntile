#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/nntile-prebuilt-lib-path.sh
#
# Print absolute path to libnntile.so after build-nntile (for -DNNTILE_PREBUILT_LIBRARY).
#
# @version 1.1.0
set -euo pipefail

build_dir="${1:-build}"
for candidate in \
    "${build_dir}/lib/libnntile.so" \
    "${build_dir}/libnntile.so"; do
    if [ -f "$candidate" ]; then
        readlink -f "$candidate"
        exit 0
    fi
done
found=$(find "$build_dir" -name 'libnntile.so' -type f 2>/dev/null | head -1)
if [ -n "$found" ]; then
    readlink -f "$found"
    exit 0
fi
echo "libnntile.so not found under ${build_dir}" >&2
exit 1
