/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/lower_from_tensor.hh
 * Lower TensorGraph ops to TileGraph ops (elementwise, GEMM, optimizer steps).
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tile/lowering_context.hh>

namespace nntile::graph
{

//! Append tile-level ops for all tensor ops in \p tg. \p out must already
//! contain per-tile storage (see \c lower_to_tile and tensor-to-tile mapping).
//! @param tg Source tensor graph
//! @param out Output tile graph (tile nodes must already exist in \p tile_map)
//! @param tile_map Mapping from each tensor data node to its tile nodes
void lower_tensor_ops_to_tile_graph(
    const TensorGraph& tg,
    TileGraph& out,
    const TensorNodeToTileMap& tile_map);

} // namespace nntile::graph
