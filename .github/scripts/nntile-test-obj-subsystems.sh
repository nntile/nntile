#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/nntile-test-obj-subsystems.sh
#
# Print test OBJECT subsystems to restore for build-tests (matches
# nntile/cmake/NNTileTests.cmake NNTILE_TEST_SUBSYSTEM deps).
#
# @version 1.1.0
set -euo pipefail

sub="${1:?subsystem name required}"

case "$sub" in
    kernel)
        printf '%s\n' kernel
        ;;
    starpu)
        printf '%s\n' kernel starpu
        ;;
    core)
        printf '%s\n' kernel starpu core
        ;;
    tile|tensor|nn|module|model|io)
        printf '%s\n' "$sub"
        ;;
    *)
        echo "unknown subsystem: $sub" >&2
        exit 1
        ;;
esac
