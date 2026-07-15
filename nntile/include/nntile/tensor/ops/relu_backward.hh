/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/ops/relu_backward.hh
 * TensorGraph relu_backward operation: dx = alpha * grad_func(x) * dy + beta * dx
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/tensor/graph.hh>

namespace nntile
{
struct LoweringContext;
}

namespace nntile::tensor
{

//! ReLU backward operation: dx = alpha * grad_func(x) * dy + beta * dx
struct TensorReluBackwardOp : TensorGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    TensorGraph::TensorNode *x = nullptr;
    TensorGraph::TensorNode *dy = nullptr;
    TensorGraph::TensorNode *dx = nullptr;

    TensorReluBackwardOp() = default;
    TensorReluBackwardOp(TensorGraph::TensorNode *x_,
        TensorGraph::TensorNode *dy_,
        TensorGraph::TensorNode *dx_,
        Scalar alpha_, Scalar beta_) :
        alpha(alpha_), beta(beta_), x(x_), dy(dy_), dx(dx_)
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

    std::string op_name() const override { return "RELU_BACKWARD"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorReluBackwardOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

//! ReLU backward: dx = alpha * grad_func(x) * dy + beta * dx (creates output, beta=0)
TensorGraph::TensorNode *relu_backward(
    Scalar alpha, TensorGraph::TensorNode *x, TensorGraph::TensorNode *dy);

//! ReLU backward: dx = alpha * grad_func(x) * dy + beta * dx (uses existing output)
void relu_backward(Scalar alpha, TensorGraph::TensorNode *x,
    TensorGraph::TensorNode *dy,
    Scalar beta, TensorGraph::TensorNode *dx);

} // namespace nntile::tensor
