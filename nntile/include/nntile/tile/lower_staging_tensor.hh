/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/tile/lower_staging_tensor.hh
 * Lower a single-tile io_staging tensor into a TileGraph immediately.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/graph_decl.hh>
#include <nntile/tensor/tensor_graph_tiling.hh>
#include <nntile/tile/append_tensor_graph_phase.hh>
#include <nntile/tile/graph_decl.hh>

namespace nntile
{

//! Ensure \p staging (single-tile) has tile nodes in \p tile_graph.
//! Reuses \p state when layout fingerprint matches.
void lower_staging_tensor_immediate(
    TensorGraph const &tg,
    TensorGraph::TensorNode const *staging,
    TensorGraphTiling const &tiling,
    TileGraph &tile_graph,
    TileGraphIncrementalState &state,
    TensorNodeToTileMap &tile_map);

} // namespace nntile
