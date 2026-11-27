#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/src/tile/lower_staging_tensor.cc
 *
 * @version 1.1.0
 * */

#include "nntile/tile/lower_staging_tensor.hh"

#include <stdexcept>

#include "nntile/tile/graph.hh"

namespace nntile
{

namespace
{

std::string tile_node_name(
    std::string const &logical_name,
    Index group_id,
    Index linear,
    Index grid_volume)
{
    std::string base =
        logical_name + "__g" + std::to_string(static_cast<long long>(group_id));
    if (grid_volume == 1)
    {
        return base;
    }
    return base + "__t" + std::to_string(static_cast<long long>(linear));
}

std::vector<TileGraph::TileNode *> build_tile_nodes(
    TileGraph &tile_graph,
    TensorGraph::TensorNode const *tensor_node,
    TensorAxisLayout const &lay,
    Index group_id)
{
    const Index vol = lay.grid_volume();
    std::vector<TileGraph::TileNode *> tiles;
    tiles.reserve(static_cast<size_t>(vol));
    std::vector<Index> grid_coord;
    for (Index lin = 0; lin < vol; ++lin)
    {
        lay.grid_coord_from_linear(lin, grid_coord);
        const std::vector<Index> tile_shape =
            lay.tile_shape_at(grid_coord);
        const std::string tname =
            tile_node_name(tensor_node->name(), group_id, lin, vol);
        TileGraph::TileNode *tile_node_ptr = tile_graph.data(
            tile_shape,
            tname,
            tensor_node->dtype());
        tiles.push_back(tile_node_ptr);
    }

    TileGraph::TensorDescriptor desc;
    desc.tensor_name = tensor_node->name();
    desc.tensor_shape = tensor_node->shape();
    desc.tile_shape = lay.max_tile_extents();
    desc.grid_shape = lay.grid_shape();
    desc.dtype = tensor_node->dtype();
    desc.tiles = tiles;
    desc.source_node = const_cast<TensorGraph::TensorNode *>(tensor_node);

    TileGraph::TensorDescriptor *desc_ptr =
        tile_graph.add_tensor_descriptor(std::move(desc));

    for (Index lin = 0; lin < vol; ++lin)
    {
        lay.grid_coord_from_linear(lin, grid_coord);
        tiles[static_cast<size_t>(lin)]->set_tensor_info(
            desc_ptr,
            grid_coord);
    }
    return tiles;
}

} // namespace

void lower_staging_tensor_immediate(
    TensorGraph const &tg,
    TensorGraph::TensorNode const *staging,
    std::shared_ptr<TensorGraphTiling const> tiling,
    TileGraph &tile_graph,
    TileGraphIncrementalState &state,
    TensorNodeToTileMap &tile_map)
{
    (void)tg;
    if (staging == nullptr)
    {
        throw std::invalid_argument(
            "lower_staging_tensor_immediate: staging must be non-null");
    }
    if (tiling == nullptr)
    {
        throw std::invalid_argument(
            "lower_staging_tensor_immediate: tiling must be non-null");
    }
    const TensorAxisLayout *lay = tiling->find(staging);
    if (lay == nullptr)
    {
        throw std::runtime_error(
            "lower_staging_tensor_immediate: missing tiling for staging '"
            + staging->name() + "'");
    }
    if (lay->grid_volume() != 1)
    {
        throw std::runtime_error(
            "lower_staging_tensor_immediate: staging must be single-tile");
    }

    tile_graph.set_tiling_scheme(tiling);

    std::uint64_t const fp = lay->layout_fingerprint_hash();
    std::uint64_t const *fp_ptr = state.tensor_layout_fp.try_get(staging);
    const bool have_tiles =
        state.tensor_to_tiles.contains(staging) && fp_ptr != nullptr;

    if (!have_tiles)
    {
        const Index gid = state.next_tile_group_id++;
        std::vector<TileGraph::TileNode *> tiles =
            build_tile_nodes(tile_graph, staging, *lay, gid);
        state.tensor_to_tiles[staging] = tiles;
        state.tensor_layout_fp[staging] = fp;
        tile_map[staging] = tiles;
        return;
    }

    if (*fp_ptr != fp)
    {
        throw std::runtime_error(
            "lower_staging_tensor_immediate: staging '" + staging->name()
            + "' uses a different tiling than in an earlier phase");
    }

    tile_map[staging] = state.tensor_to_tiles[staging];
}

} // namespace nntile
