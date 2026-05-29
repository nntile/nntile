#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/ctest-label-exclude-subsystem.sh
#
# Print a semicolon-separated ctest -LE label list for a test subsystem job.
#
# @version 1.1.0
set -euo pipefail
sub="${1:?subsystem name required}"
_le='NotImplemented'
case "$sub" in
    model) _le='NotImplemented;FixtureData' ;;
    kernel|starpu|core|tile|tensor|nn|module|io) ;;
    *)
        echo "unknown test subsystem: $sub" >&2
        exit 1
        ;;
esac
printf '%s\n' "$_le"
