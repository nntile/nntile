#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/ctest-run-subsystem.sh
#
# Print a ctest -R regex for the given test subsystem.
#
# @version 1.1.0
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
