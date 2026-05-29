#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/touch-nntile-objs.sh
#
# Refresh mtimes on restored OBJECT files so Ninja skips recompilation.
#
# @version 1.1.0
set -euo pipefail

build_dir="${1:-build}"
find "${build_dir}/nntile/src/CMakeFiles" \
    -path '*/nntile_objs_*.dir/*.o' -print0 |
    xargs -0 -r touch
