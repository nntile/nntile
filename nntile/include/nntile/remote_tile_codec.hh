/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/remote_tile_codec.hh
 * Classic TILE_* encode / apply for the Unix-socket driver.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/graph.hh>

#include <nlohmann/json.hpp>

namespace nntile
{

nlohmann::json encode_tile_op_attrs(TileGraph::OpNode const &op);

void apply_tile_op(TileGraph &graph, nlohmann::json const &op);

TileGraph::TileNode *tile_node_by_id(
    TileGraph &graph, TileGraph::NodeId id);

bool tile_graph_has_node(
    TileGraph const &graph, TileGraph::NodeId id);

} // namespace nntile
