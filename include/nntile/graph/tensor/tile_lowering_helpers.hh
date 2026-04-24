/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/tile_lowering_helpers.hh
 * Shared helpers for tensor-to-tile lowering implementations.
 *
 * @version 1.1.0
 * */

#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <nntile/graph/tile/lowering_context.hh>

namespace nntile::graph::tile_lower
{

void assert_same_elementwise_layout(
    const TensorGraph::TensorNode* a,
    const TensorGraph::TensorNode* b,
    const char* ctx);

const std::vector<TileGraph::TileNode*>& tiles_of(
    const TensorNodeToTileMap& m,
    const TensorGraph::TensorNode* n);

std::vector<TileGraph::TileNode*> copy_tiles(
    const TensorNodeToTileMap& m,
    const TensorGraph::TensorNode* n);

void lower_unary2(
    const TensorGraph::TensorNode* src,
    const TensorGraph::TensorNode* dst,
    const TensorNodeToTileMap& m,
    const char* ctx,
    void (*fn)(TileGraph::TileNode*, TileGraph::TileNode*));

void lower_inplace1(
    const TensorGraph::TensorNode* x,
    const TensorNodeToTileMap& m,
    const char* ctx,
    void (*fn)(TileGraph::TileNode*));

void lower_backward3(
    const TensorGraph::TensorNode* x,
    const TensorGraph::TensorNode* dy,
    const TensorGraph::TensorNode* dx,
    const TensorNodeToTileMap& m,
    const char* ctx,
    void (*fn)(TileGraph::TileNode*, TileGraph::TileNode*, TileGraph::TileNode*));

} // namespace nntile::graph::tile_lower
