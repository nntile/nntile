#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/cmake-build-tests-subsystem.sh
#
# Print -DNNTILE_TEST_SUBSYSTEM=... for CMake (see nntile/cmake/NNTileTests.cmake).
#
# @version 1.1.0
set -euo pipefail
sub="${1:?subsystem name required}"

case "$sub" in
    kernel|starpu|core|tile|tensor|nn|module|model|io)
        printf '%s\n' "-DNNTILE_TEST_SUBSYSTEM=${sub}"
        ;;
    runtime|optim|dataset)
        echo "No dedicated test tree for subsystem ${sub}" >&2
        exit 1
        ;;
    *)
        echo "unknown subsystem: $sub" >&2
        exit 1
        ;;
esac
