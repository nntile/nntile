/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/relu_backward.hh
 * TileGraph ReLU backward: dx = relu_backward(x, dy) (accumulates into dx)
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

struct TileReluBackwardOp : TileGraph::OpNode
{
    TileGraph::TileNode* x = nullptr;
    TileGraph::TileNode* dy = nullptr;
    TileGraph::TileNode* dx = nullptr;

    TileReluBackwardOp() = default;
    TileReluBackwardOp(
        TileGraph::TileNode* x_, TileGraph::TileNode* dy_, TileGraph::TileNode* dx_)
        : x(x_), dy(dy_), dx(dx_)
    {
        inputs_ = {x, dy, dx};
        outputs_ = {dx};
    }

    std::string op_name() const override { return "TILE_RELU_BACKWARD"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileReluBackwardOp>(*this);
    }
};

void relu_backward(
    TileGraph::TileNode* x, TileGraph::TileNode* dy, TileGraph::TileNode* dx);

} // namespace nntile::graph::tile_graph
