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

//! Append tile-level ops for tensor ops \p tg.ops()[op_begin:op_end).
//! @param tg Source tensor graph
//! @param out Output tile graph (tile nodes must exist in \p tile_map)
//! @param tile_map Mapping from each tensor data node to tile nodes
//! @param op_begin First op index (inclusive)
//! @param op_end One past last op index (exclusive)
void lower_tensor_ops_to_tile_graph(
    const TensorGraph& tg,
    TileGraph& out,
    const TensorNodeToTileMap& tile_map,
    size_t op_begin,
    size_t op_end);

//! Lower all ops (same as op_begin=0, op_end=tg.num_ops()).
void lower_tensor_ops_to_tile_graph(
    const TensorGraph& tg,
    TileGraph& out,
    const TensorNodeToTileMap& tile_map);

} // namespace nntile::graph
