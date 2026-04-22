/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/lowering_context.hh
 * Context for TensorGraph::OpNode::lower_to_tile.
 *
 * @version 1.1.0
 * */

#pragma once

//! Full tile graph before <map> (libc++ can otherwise instantiate TileGraph too early).
#include <nntile/graph/tile/graph.hh>

#include <map>
#include <vector>

namespace nntile::graph
{

//! Maps each tensor data node to tile nodes in row-major grid order.
using TensorNodeToTileMap =
    std::map<const TensorGraph::TensorNode*, std::vector<TileGraph::TileNode*>>;

//! Passed to each tensor op when lowering to TileGraph.
struct LoweringContext
{
    TileGraph& out;
    const TensorNodeToTileMap& tile_map;
    const TensorGraphTiling& tiling;
};

} // namespace nntile::graph
