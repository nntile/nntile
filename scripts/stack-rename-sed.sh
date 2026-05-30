#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file scripts/stack-rename-sed.sh
#
# @version 1.1.0
set -euo pipefail
cd /workspace

mapfile -t FILES < <(git ls-files -z | tr '\0' '\n' | grep -E '\.(cc|hh|h|cmake|md|py|yml|sh|txt|in)$' | grep -v '^build' || true)

apply_sed() {
    local expr=$1
    for f in "${FILES[@]}"; do
        [[ -f "$f" ]] || continue
        sed -i "$expr" "$f"
    done
}

echo "=== Include path placeholders ==="
apply_sed 's|nntile/tile_graph/|nntile/__GRAPH_TILE__/|g'
apply_sed 's|nntile/tensor_graph/|nntile/__GRAPH_TENSOR__/|g'
apply_sed 's|nntile/nn_graph/|nntile/__GRAPH_NN__/|g'
apply_sed 's|<nntile/tile_graph\.hh>|<nntile/__GRAPH_TILE__.hh>|g'
apply_sed 's|<nntile/tensor_graph\.hh>|<nntile/__GRAPH_TENSOR__.hh>|g'

echo "=== Eager tile -> core paths ==="
apply_sed 's|nntile/tile/|nntile/core/|g'
apply_sed 's|<nntile/tile\.hh>|<nntile/core.hh>|g'
apply_sed 's|"tile/|"core/|g'
apply_sed 's|/tile/|/core/|g'

echo "=== Restore graph paths ==="
apply_sed 's|nntile/__GRAPH_TILE__/|nntile/tile/|g'
apply_sed 's|nntile/__GRAPH_TENSOR__/|nntile/tensor/|g'
apply_sed 's|nntile/__GRAPH_NN__/|nntile/nn/|g'
apply_sed 's|<nntile/__GRAPH_TILE__\.hh>|<nntile/tile.hh>|g'
apply_sed 's|<nntile/__GRAPH_TENSOR__\.hh>|<nntile/tensor.hh>|g'

echo "=== Namespace placeholders ==="
apply_sed 's|nntile::tile_graph|nntile::__GRAPH_TILE_NS__|g'
apply_sed 's|nntile::tensor_graph|nntile::__GRAPH_TENSOR_NS__|g'
apply_sed 's|nntile::nn_graph|nntile::__GRAPH_NN_NS__|g'
apply_sed 's|namespace tile_graph_bind_detail|namespace tile_bind_detail|g'
apply_sed 's|namespace tile_graph_layout_io|namespace tile_layout_io|g'
apply_sed 's|tile_graph_bind_detail|tile_bind_detail|g'
apply_sed 's|tile_graph_layout_io|tile_layout_io|g'

echo "=== Eager tile namespace -> core ==="
apply_sed 's|nntile::tile|nntile::core|g'
apply_sed 's|namespace nntile::core_graph|namespace nntile::tile|g'  # undo false positive if any

echo "=== Restore graph namespaces ==="
apply_sed 's|nntile::__GRAPH_TILE_NS__|nntile::tile|g'
apply_sed 's|nntile::__GRAPH_TENSOR_NS__|nntile::tensor|g'
apply_sed 's|nntile::__GRAPH_NN_NS__|nntile::nn|g'

echo "=== CMake / CI tokens ==="
apply_sed 's|NNTILE_BUILD_TILE_GRAPH|NNTILE_BUILD___GRAPH_TILE__|g'
apply_sed 's|NNTILE_BUILD_TENSOR_GRAPH|NNTILE_BUILD___GRAPH_TENSOR__|g'
apply_sed 's|NNTILE_BUILD_NN_GRAPH|NNTILE_BUILD___GRAPH_NN__|g'
apply_sed 's|BUILD_TESTS_TILE_GRAPH|BUILD_TESTS___GRAPH_TILE__|g'
apply_sed 's|BUILD_TESTS_TENSOR_GRAPH|BUILD_TESTS___GRAPH_TENSOR__|g'
apply_sed 's|BUILD_TESTS_NN_GRAPH|BUILD_TESTS___GRAPH_NN__|g'
apply_sed 's|nntile_objs_tile_graph|nntile_objs___GRAPH_TILE__|g'
apply_sed 's|nntile_objs_tensor_graph|nntile_objs___GRAPH_TENSOR__|g'
apply_sed 's|nntile_objs_nn_graph|nntile_objs___GRAPH_NN__|g'
apply_sed 's|nntile_compile_check_tile_graph|nntile_compile_check___GRAPH_TILE__|g'
apply_sed 's|nntile_compile_check_tensor_graph|nntile_compile_check___GRAPH_TENSOR__|g'
apply_sed 's|nntile_compile_check_nn_graph|nntile_compile_check___GRAPH_NN__|g'
apply_sed 's|nntile_compile_check_tests_tile_graph|nntile_compile_check_tests___GRAPH_TILE__|g'
apply_sed 's|nntile_compile_check_tests_tensor_graph|nntile_compile_check_tests___GRAPH_TENSOR__|g'
apply_sed 's|nntile_compile_check_tests_nn_graph|nntile_compile_check_tests___GRAPH_NN__|g'

# Eager TILE build flag -> CORE (only where not graph placeholder)
apply_sed 's|NNTILE_BUILD_TILE|NNTILE_BUILD_CORE|g'
apply_sed 's|BUILD_TESTS_TILE|BUILD_TESTS_CORE|g'
apply_sed 's|nntile_objs_tile|nntile_objs_core|g'
apply_sed 's|nntile_compile_check_tile|nntile_compile_check_core|g'
apply_sed 's|nntile_compile_check_tests_tile|nntile_compile_check_tests_core|g'

# Restore graph cmake tokens
apply_sed 's|NNTILE_BUILD___GRAPH_TILE__|NNTILE_BUILD_TILE|g'
apply_sed 's|NNTILE_BUILD___GRAPH_TENSOR__|NNTILE_BUILD_TENSOR|g'
apply_sed 's|NNTILE_BUILD___GRAPH_NN__|NNTILE_BUILD_NN|g'
apply_sed 's|BUILD_TESTS___GRAPH_TILE__|BUILD_TESTS_TILE|g'
apply_sed 's|BUILD_TESTS___GRAPH_TENSOR__|BUILD_TESTS_TENSOR|g'
apply_sed 's|BUILD_TESTS___GRAPH_NN__|BUILD_TESTS_NN|g'
apply_sed 's|nntile_objs___GRAPH_TILE__|nntile_objs_tile|g'
apply_sed 's|nntile_objs___GRAPH_TENSOR__|nntile_objs_tensor|g'
apply_sed 's|nntile_objs___GRAPH_NN__|nntile_objs_nn|g'
apply_sed 's|nntile_compile_check___GRAPH_TILE__|nntile_compile_check_tile|g'
apply_sed 's|nntile_compile_check___GRAPH_TENSOR__|nntile_compile_check_tensor|g'
apply_sed 's|nntile_compile_check___GRAPH_NN__|nntile_compile_check_nn|g'
apply_sed 's|nntile_compile_check_tests___GRAPH_TILE__|nntile_compile_check_tests_tile|g'
apply_sed 's|nntile_compile_check_tests___GRAPH_TENSOR__|nntile_compile_check_tests_tensor|g'
apply_sed 's|nntile_compile_check_tests___GRAPH_NN__|nntile_compile_check_tests_nn|g'

echo "=== sed done (CMake/src lists updated separately) ==="
