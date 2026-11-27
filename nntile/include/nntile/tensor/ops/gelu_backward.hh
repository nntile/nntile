/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/ops/gelu_backward.hh
 * TensorGraph GeLU backward operation: dx = alpha * grad_func(x) * dy + beta * dx
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/tensor/graph.hh>

namespace nntile
{
struct LoweringContext;
}

namespace nntile::tensor
{

//! GeLU backward operation: dx = alpha * grad_func(x) * dy + beta * dx
struct TensorGeluBackwardOp : TensorGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    TensorGraph::TensorNode* x = nullptr;
    TensorGraph::TensorNode* dy = nullptr;
    TensorGraph::TensorNode* dx = nullptr;

    TensorGeluBackwardOp() = default;
    TensorGeluBackwardOp(
        TensorGraph::TensorNode* x_,
        TensorGraph::TensorNode* dy_,
        TensorGraph::TensorNode* dx_,
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

    std::string op_name() const override { return "GELU_BACKWARD"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorGeluBackwardOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

//! GeLU backward: dx = alpha * grad_func(x) * dy + beta * dx (creates output, beta=0)
TensorGraph::TensorNode* gelu_backward(
    Scalar alpha, TensorGraph::TensorNode* x, TensorGraph::TensorNode* dy);

//! GeLU backward: dx = alpha * grad_func(x) * dy + beta * dx (uses existing output)
void gelu_backward(
    Scalar alpha,
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    Scalar beta,
    TensorGraph::TensorNode* dx);

} // namespace nntile::tensor
