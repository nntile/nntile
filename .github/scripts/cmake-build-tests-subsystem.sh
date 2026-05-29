#!/usr/bin/env bash
# Print -DBUILD_TESTS_* flags: enable tests only for SUBSYSTEM (inclusive layer).
set -euo pipefail
sub="${1:?subsystem name required}"

all=(kernel starpu tile tensor tile_graph tensor_graph nn_graph module model io)
flags=()
for s in "${all[@]}"; do
    u=$(echo "$s" | tr '[:lower:]' '[:upper:]')
    case "$s" in
        tile_graph) u=TILE_GRAPH ;;
        tensor_graph) u=TENSOR_GRAPH ;;
        nn_graph) u=NN_GRAPH ;;
    esac
    flags+=("-DBUILD_TESTS_${u}=OFF")
done

enable() {
    local s=$1
    local u
    u=$(echo "$s" | tr '[:lower:]' '[:upper:]')
    case "$s" in
        tile_graph) u=TILE_GRAPH ;;
        tensor_graph) u=TENSOR_GRAPH ;;
        nn_graph) u=NN_GRAPH ;;
    esac
    flags+=("-DBUILD_TESTS_${u}=ON")
}

case "$sub" in
    kernel) enable kernel ;;
    starpu) enable kernel; enable starpu ;;
    tile) enable kernel; enable starpu; enable tile ;;
    tensor) enable kernel; enable starpu; enable tile; enable tensor ;;
    tile_graph) enable tile_graph ;;
    tensor_graph) enable tensor_graph ;;
    nn_graph) enable nn_graph ;;
    module) enable module ;;
    model) enable model ;;
    io) enable io ;;
    runtime|optim|dataset)
        echo "No dedicated test tree for subsystem ${sub}" >&2
        exit 1
        ;;
    *)
        echo "unknown subsystem: $sub" >&2
        exit 1
        ;;
esac

printf '%s\n' "${flags[@]}"
