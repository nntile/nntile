/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/lower_from_tensor.cc
 * TileGraph lower from tensor operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/lower_from_tensor.hh"
#include <nntile/graph/tensor/graph.hh>

#include <stdexcept>
#include <string>

namespace nntile::graph
{

void lower_tensor_ops_to_tile_graph(
    const TensorGraph& tg,
    TileGraph& out,
    const TensorNodeToTileMap& tile_map)
{
    const TensorGraphTiling* tsch = out.tiling_scheme();
    if(tsch == nullptr)
    {
        throw std::runtime_error(
            "lower_tensor_ops_to_tile_graph: TileGraph has no tiling_scheme");
    }

    const LoweringContext ctx{out, tile_map, *tsch};
    for(const auto& op : tg.ops())
    {
        op->lower_to_tile(ctx);
    }
}

} // namespace nntile::graph
