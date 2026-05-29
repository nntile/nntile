#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/cmake-test-subsystem.sh
#
# Print extra -D flags to enable subsystems through LAST (inclusive).
#
# @version 1.1.0
set -euo pipefail
last="${1:?subsystem name required}"
flags=()
enable() { flags+=("-DNNTILE_BUILD_${1}=ON"); }
disable() { flags+=("-DNNTILE_BUILD_${1}=OFF"); }

for s in KERNEL STARPU CORE LOGGER GRAPH_BASE TILE TENSOR NN RUNTIME MODULE \
    OPTIM IO DATASET MODEL; do
    disable "$s"
done

case "$last" in
    kernel) enable KERNEL; enable LOGGER ;;
    starpu) enable KERNEL; enable STARPU; enable LOGGER ;;
    core) enable KERNEL; enable STARPU; enable CORE; enable LOGGER ;;
    tile)
        enable KERNEL; enable STARPU; enable CORE; enable GRAPH_BASE
        enable TILE; enable LOGGER ;;
    tensor)
        enable KERNEL; enable STARPU; enable CORE; enable GRAPH_BASE
        enable TILE; enable TENSOR; enable LOGGER ;;
    nn)
        enable KERNEL; enable STARPU; enable CORE; enable GRAPH_BASE
        enable TILE; enable TENSOR; enable NN; enable LOGGER ;;
    runtime)
        enable KERNEL; enable STARPU; enable CORE; enable GRAPH_BASE
        enable TILE; enable TENSOR; enable NN
        enable RUNTIME; enable LOGGER ;;
    module)
        enable KERNEL; enable STARPU; enable CORE; enable GRAPH_BASE
        enable TILE; enable TENSOR; enable NN
        enable MODULE; enable LOGGER ;;
    optim)
        enable KERNEL; enable STARPU; enable CORE; enable GRAPH_BASE
        enable TILE; enable TENSOR; enable NN
        enable OPTIM; enable LOGGER ;;
    io)
        enable KERNEL; enable STARPU; enable CORE; enable GRAPH_BASE
        enable IO; enable LOGGER ;;
    dataset)
        enable KERNEL; enable STARPU; enable CORE; enable GRAPH_BASE
        enable IO; enable DATASET; enable LOGGER ;;
    model)
        enable KERNEL; enable STARPU; enable CORE; enable GRAPH_BASE
        enable TILE; enable TENSOR; enable NN
        enable RUNTIME; enable MODULE; enable MODEL; enable LOGGER ;;
    *)
        echo "unknown subsystem: $last" >&2
        exit 1
        ;;
esac

printf '%s\n' "${flags[@]}"
