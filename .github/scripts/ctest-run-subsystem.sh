#!/usr/bin/env bash
# Print a ctest -R regex for the given test subsystem.
set -euo pipefail
sub="${1:?subsystem name required}"
case "$sub" in
    kernel) echo 'tests_core_kernel_|tests_nntile_constants' ;;
    starpu) echo 'tests_core_starpu_' ;;
    core) echo 'tests_core_tile_' ;;
    tile) echo 'tests_graph_tile_' ;;
    tensor) echo 'tests_graph_tensor_' ;;
    nn) echo 'tests_graph_nn_' ;;
    module) echo 'tests_graph_module_' ;;
    model) echo 'tests_graph_model_' ;;
    io) echo 'tests_graph_io_' ;;
    *)
        echo "unknown test subsystem: $sub" >&2
        exit 1
        ;;
esac
