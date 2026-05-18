/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/rope_backward.hh
 * TileGraph rope_backward operation: (sin, cos, dy, dx)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! RoPE backward operation: dx = rope_backward(sin, cos, dy)
struct TileRopeBackwardOp : TileGraph::OpNode
{
    TileGraph::TileNode* sin = nullptr, * cos = nullptr, * dy = nullptr, * dx = nullptr;
    TileRopeBackwardOp() = default;
    TileRopeBackwardOp(TileGraph::TileNode* si, TileGraph::TileNode* co, TileGraph::TileNode* y, TileGraph::TileNode* x) : sin(si), cos(co), dy(y), dx(x)
    {
        inputs_ = {sin, cos, dy};
        outputs_ = {dx};
    }
    std::string op_name() const override { return "TILE_ROPE_BACKWARD"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileRopeBackwardOp>(*this);
    }
};
void rope_backward(TileGraph::TileNode* sin, TileGraph::TileNode* cos, TileGraph::TileNode* dy, TileGraph::TileNode* dx);
} // namespace nntile::graph::tile_graph
