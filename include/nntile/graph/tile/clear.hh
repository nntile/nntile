/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/clear.hh
 * TileGraph clear operation: x = 0
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Clear operation at tile level: x = 0
struct TileClearOp : TileGraph::OpNode
{
    TileGraph::TileNode* x = nullptr;

    TileClearOp() = default;
    explicit TileClearOp(TileGraph::TileNode* x_)
        : x(x_)
    {
        outputs_ = {x};
    }

    std::string op_name() const override { return "TILE_CLEAR"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileClearOp>(*this);
    }
};

//! Clear tile: x = 0
void clear(TileGraph::TileNode* x);

} // namespace nntile::graph::tile_graph
