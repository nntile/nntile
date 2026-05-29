#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file scripts/stack-rename.sh
#
# Restructure nntile stack: kernel -> starpu -> core -> tile -> tensor -> nn
# @version 1.1.0
set -euo pipefail
cd /workspace

echo "=== Phase 0: remove legacy ==="
git rm -rf wrappers/python notebooks 2>/dev/null || true
rm -rf wrappers/python notebooks
git rm -rf nntile/src/tensor nntile/include/nntile/tensor nntile/tests/tensor 2>/dev/null || true

echo "=== Phase 1: eager tile -> core ==="
git mv nntile/src/tile nntile/src/core
git mv nntile/include/nntile/tile nntile/include/nntile/core
git mv nntile/tests/tile nntile/tests/core
git mv nntile/include/nntile/tile.hh nntile/include/nntile/core_eager.hh

echo "=== Phase 2: tile_graph -> tile ==="
git mv nntile/src/tile_graph nntile/src/tile
git mv nntile/include/nntile/tile_graph nntile/include/nntile/tile
git mv nntile/tests/tile_graph nntile/tests/tile
git mv nntile/include/nntile/tile_graph.hh nntile/include/nntile/tile.hh

echo "=== Phase 3: tensor_graph -> tensor ==="
rm -f nntile/include/nntile/tensor.hh 2>/dev/null || true
git rm -f nntile/include/nntile/tensor.hh 2>/dev/null || true
git mv nntile/src/tensor_graph nntile/src/tensor
git mv nntile/include/nntile/tensor_graph nntile/include/nntile/tensor
git mv nntile/tests/tensor_graph nntile/tests/tensor
git mv nntile/include/nntile/tensor_graph.hh nntile/include/nntile/tensor.hh

echo "=== Phase 4: nn_graph -> nn ==="
git mv nntile/tests/nn.cc nntile/tests/nn_graph/nn_graph.cc
git mv nntile/src/nn_graph nntile/src/nn
git mv nntile/include/nntile/nn_graph nntile/include/nntile/nn
git mv nntile/tests/nn_graph nntile/tests/nn

echo "=== Phase 5: core_eager.hh -> core.hh (replace umbrella) ==="
git rm -f nntile/include/nntile/core.hh 2>/dev/null || true
git mv nntile/include/nntile/core_eager.hh nntile/include/nntile/core.hh

echo "=== Done with git moves ==="
