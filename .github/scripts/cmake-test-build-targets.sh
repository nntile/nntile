#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/cmake-test-build-targets.sh
#
# List CMake executable targets for one test subsystem (for selective test-build).
#
# @version 1.1.0
set -euo pipefail
sub="${1:?subsystem name required}"
build_dir="${2:-build}"

re="$(.github/scripts/ctest-run-subsystem.sh "$sub")"
mapfile -t targets < <(
    cd "$build_dir"
    ctest -N -R "$re" 2>/dev/null \
        | sed -n 's/^[[:space:]]*Test[[:space:]]*#[0-9]*:[[:space:]]*\([^[:space:]]*\).*/\1/p'
)

if ((${#targets[@]} == 0)); then
    echo "No CTest targets matched subsystem '${sub}' (regex: ${re})" >&2
    exit 1
fi

printf '%s\n' "${targets[@]}"
