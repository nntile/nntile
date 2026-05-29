#!/usr/bin/env bash
# Print extra -D flags to enable subsystems through LAST (inclusive).
set -euo pipefail
last="${1:?subsystem name required}"
flags=()
enable() { flags+=("-DNNTILE_BUILD_${1}=ON"); }
disable() { flags+=("-DNNTILE_BUILD_${1}=OFF"); }

for s in KERNEL STARPU TILE TENSOR LOGGER GRAPH_BASE TILE_GRAPH TENSOR_GRAPH \
    NN_GRAPH RUNTIME MODULE OPTIM IO DATASET MODEL; do
    disable "$s"
done

case "$last" in
    kernel) enable KERNEL; enable LOGGER ;;
    starpu) enable KERNEL; enable STARPU; enable LOGGER ;;
    tile) enable KERNEL; enable STARPU; enable TILE; enable LOGGER ;;
    tile_graph)
        enable KERNEL; enable STARPU; enable TILE; enable GRAPH_BASE
        enable TILE_GRAPH; enable LOGGER ;;
    tensor_graph)
        enable KERNEL; enable STARPU; enable TILE; enable GRAPH_BASE
        enable TILE_GRAPH; enable TENSOR_GRAPH; enable LOGGER ;;
    nn_graph)
        enable KERNEL; enable STARPU; enable TILE; enable GRAPH_BASE
        enable TILE_GRAPH; enable TENSOR_GRAPH; enable NN_GRAPH; enable LOGGER ;;
    runtime)
        enable KERNEL; enable STARPU; enable TILE; enable GRAPH_BASE
        enable TILE_GRAPH; enable TENSOR_GRAPH; enable NN_GRAPH
        enable RUNTIME; enable LOGGER ;;
    module)
        enable KERNEL; enable STARPU; enable TILE; enable GRAPH_BASE
        enable TILE_GRAPH; enable TENSOR_GRAPH; enable NN_GRAPH
        enable MODULE; enable LOGGER ;;
    optim)
        enable KERNEL; enable STARPU; enable TILE; enable GRAPH_BASE
        enable TILE_GRAPH; enable TENSOR_GRAPH; enable NN_GRAPH
        enable OPTIM; enable LOGGER ;;
    io)
        enable KERNEL; enable STARPU; enable TILE; enable GRAPH_BASE
        enable IO; enable LOGGER ;;
    dataset)
        enable KERNEL; enable STARPU; enable TILE; enable GRAPH_BASE
        enable IO; enable DATASET; enable LOGGER ;;
    model)
        enable KERNEL; enable STARPU; enable TILE; enable GRAPH_BASE
        enable TILE_GRAPH; enable TENSOR_GRAPH; enable NN_GRAPH
        enable RUNTIME; enable MODULE; enable MODEL; enable LOGGER ;;
    *)
        echo "unknown subsystem: $last" >&2
        exit 1
        ;;
esac

printf '%s\n' "${flags[@]}"
