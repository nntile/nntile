/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/ops/relu_backward.hh
 * TileGraph ReLU backward: dx = alpha * grad_func(x) * dy + beta * dx
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/tile/graph.hh>

namespace nntile::tile
{

//! ReLU backward operation: dx = alpha * grad_func(x) * dy + beta * dx
struct TileReluBackwardOp : TileGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    TileGraph::TileNode* x = nullptr;
    TileGraph::TileNode* dy = nullptr;
    TileGraph::TileNode* dx = nullptr;

    TileReluBackwardOp() = default;
    TileReluBackwardOp(
        TileGraph::TileNode* x_, TileGraph::TileNode* dy_, TileGraph::TileNode* dx_,
        Scalar alpha_, Scalar beta_)
        : alpha(alpha_), beta(beta_), x(x_), dy(dy_), dx(dx_)
    {
        if(beta == Scalar{0.0})
        {
            inputs_ = {x, dy};
        }
        else
        {
            inputs_ = {x, dy, dx};
        }
        outputs_ = {dx};
    }

    std::string op_name() const override { return "TILE_RELU_BACKWARD"; }

    void execute(Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileReluBackwardOp>(*this);
    }
};

//! ReLU backward: dx = alpha * grad_func(x) * dy + beta * dx
void relu_backward(Scalar alpha, TileGraph::TileNode* x, TileGraph::TileNode* dy,
    Scalar beta, TileGraph::TileNode* dx);

} // namespace nntile::tile
