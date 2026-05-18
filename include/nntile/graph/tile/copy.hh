/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/copy.hh
 * TileGraph copy operation: dst = src
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Copy operation at tile level: dst = src
struct TileCopyOp : TileGraph::OpNode
{
    TileGraph::TileNode* src = nullptr;
    TileGraph::TileNode* dst = nullptr;
    TileCopyOp() = default;
    TileCopyOp(TileGraph::TileNode* s, TileGraph::TileNode* d) : src(s), dst(d)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_COPY"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileCopyOp>(*this);
    }
};

//! Copy: dst = src (uses existing output)
void copy(TileGraph::TileNode* src, TileGraph::TileNode* dst);
} // namespace nntile::graph::tile_graph
