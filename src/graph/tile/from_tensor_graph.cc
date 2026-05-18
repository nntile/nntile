/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/from_tensor_graph.cc
 * TileGraph::from_tensor_graph - convert TensorGraph to TileGraph.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/graph.hh"

#include <sstream>
#include <stdexcept>

#include "nntile/graph/tensor/graph.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tile/lower_from_tensor.hh"

namespace nntile::graph
{

namespace
{

std::string internal_tile_name(const std::string& tensor_name, Index linear,
                                Index grid_volume)
{
    if(grid_volume == 1)
    {
        return tensor_name;
    }
    return tensor_name + "__t" + std::to_string(static_cast<long long>(linear));
}

} // namespace

TileGraph TileGraph::from_tensor_graph(const TensorGraph& tg)
{
    TensorGraphTiling tiling = TensorGraphTiling::from_tensor_graph(tg);
    return from_tensor_graph(tg, tiling);
}

TileGraph TileGraph::from_tensor_graph(
    const TensorGraph& tg, const TensorGraphTiling& tiling)
{
    auto scheme = std::make_shared<TensorGraphTiling>(tiling);
    TileGraph tile_graph(tg.name() + "_tile");
    tile_graph.set_tiling_scheme(scheme);

    TensorNodeToTileMap node_map;

    for(const auto& tensor_node : tg.tensor_nodes())
    {
        const TensorAxisLayout* lay = scheme->find(tensor_node.get());
        if(lay == nullptr)
        {
            throw std::runtime_error(
                "TileGraph::from_tensor_graph: missing tiling for tensor '" +
                tensor_node->name() + "'");
        }

        const Index vol = lay->grid_volume();
        std::vector<TileGraph::TileNode*> tiles;
        tiles.reserve(static_cast<size_t>(vol));

        std::vector<Index> grid_coord;
        for(Index lin = 0; lin < vol; ++lin)
        {
            lay->grid_coord_from_linear(lin, grid_coord);
            const std::vector<Index> tile_shape = lay->tile_shape_at(grid_coord);
            const std::string tname =
                internal_tile_name(tensor_node->name(), lin, vol);

            TileNode* tile_node = tile_graph.data(
                tile_shape,
                tname,
                tensor_node->dtype());

            if(tensor_node->is_input())
            {
                tile_node->mark_input(true);
            }
            if(tensor_node->is_output())
            {
                tile_node->mark_output(true);
            }

            tiles.push_back(tile_node);
        }

        TileGraph::TensorDescriptor desc;
        desc.tensor_name = tensor_node->name();
        desc.tensor_shape = tensor_node->shape();
        desc.tile_shape = lay->max_tile_extents();
        desc.grid_shape = lay->grid_shape();
        desc.dtype = tensor_node->dtype();
        desc.tiles = tiles;
        desc.source_node = tensor_node.get();

        TileGraph::TensorDescriptor* desc_ptr =
            tile_graph.add_tensor_descriptor(std::move(desc));

        for(Index lin = 0; lin < vol; ++lin)
        {
            lay->grid_coord_from_linear(lin, grid_coord);
            tiles[static_cast<size_t>(lin)]->set_tensor_info(
                desc_ptr, grid_coord);
        }

        node_map[tensor_node.get()] = std::move(tiles);
    }

    lower_tensor_ops_to_tile_graph(tg, tile_graph, node_map);

    return tile_graph;
}

} // namespace nntile::graph
