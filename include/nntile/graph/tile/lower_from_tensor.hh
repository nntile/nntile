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

#include <nntile/graph/tile/lowering_context.hh>

namespace nntile::graph
{

//! Append tile-level ops for all tensor ops in tg. out must already contain tiles.
void lower_tensor_ops_to_tile_graph(
    const TensorGraph& tg,
    TileGraph& out,
    const TensorNodeToTileMap& tile_map);

} // namespace nntile::graph
