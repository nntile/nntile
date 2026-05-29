#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/ci-subsystems-for-diff.sh
#
# Map changed paths to subsystem names (with ancestors) for CI path filters.
#
# @version 1.1.0
set -euo pipefail

base="${1:-origin/graph_api}"
subsystems=()

add_sub() {
    local s="$1"
    subsystems+=("$s")
}

changed=$(git diff --name-only "$base"...HEAD)

if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/kernel/'; then
    add_sub kernel
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/starpu/'; then
    add_sub starpu
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/core/'; then
    add_sub core
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/tile/'; then
    add_sub tile
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/tensor/'; then
    add_sub tensor
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/nn/'; then
    add_sub nn
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/runtime'; then
    add_sub runtime
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/module/'; then
    add_sub module
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/optim/'; then
    add_sub optim
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/io/'; then
    add_sub io
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/dataset/'; then
    add_sub dataset
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/model/'; then
    add_sub model
fi

if [ ${#subsystems[@]} -eq 0 ]; then
    echo "kernel starpu core tile tensor nn runtime module optim io dataset model"
else
    printf '%s\n' "${subsystems[@]}" | sort -u | tr '\n' ' '
fi
