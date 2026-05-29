#!/usr/bin/env bash
# Map changed paths to subsystem names (with ancestors) for CI path filters.
set -euo pipefail

base="${1:-origin/graph_api}"
subsystems=()

# Headers: nntile/include/nntile/<layer>/; sources: nntile/src/<layer>/.

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
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/tile/'; then
    add_sub tile
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/tile_graph/'; then
    add_sub tile_graph
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/tensor_graph/'; then
    add_sub tensor_graph
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/nn_graph/'; then
    add_sub nn_graph
fi
if echo "$changed" | grep -qE '^nntile/(src|include/nntile)/runtime/'; then
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
    echo "kernel starpu tile tile_graph tensor_graph nn_graph runtime module optim io dataset model"
else
    printf '%s\n' "${subsystems[@]}" | sort -u | tr '\n' ' '
fi
