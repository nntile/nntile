#!/usr/bin/env bash
# Map changed paths to subsystem names (with ancestors) for CI path filters.
set -euo pipefail

base="${1:-origin/graph_api}"
subsystems=()

add_sub() {
    local s="$1"
    subsystems+=("$s")
}

if git diff --name-only "$base"...HEAD | grep -qE '^nntile/(src|include)/kernel/'; then
    add_sub kernel
fi
if git diff --name-only "$base"...HEAD | grep -qE '^nntile/(src|include)/starpu/'; then
    add_sub starpu
fi
if git diff --name-only "$base"...HEAD | grep -qE '^nntile/(src|include)/tile/'; then
    add_sub tile
fi
if git diff --name-only "$base"...HEAD | grep -qE '^nntile/(src|include)/tile_graph/'; then
    add_sub tile_graph
fi
if git diff --name-only "$base"...HEAD | grep -qE '^nntile/(src|include)/tensor_graph/'; then
    add_sub tensor_graph
fi
if git diff --name-only "$base"...HEAD | grep -qE '^nntile/(src|include)/nn_graph/'; then
    add_sub nn_graph
fi

if [ ${#subsystems[@]} -eq 0 ]; then
    echo "kernel starpu tile tile_graph tensor_graph nn_graph module model"
else
    printf '%s\n' "${subsystems[@]}" | sort -u | tr '\n' ' '
fi
