/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/gelu_backward.hh
 * TensorGraph GeLU backward operation: dx += gelu_backward(x, dy)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{
struct LoweringContext;
}

namespace nntile::graph::tensor
{

//! GeLU backward operation at tensor level: dx += gelu_backward(x, dy)
struct TensorGeluBackwardOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* x = nullptr;
    TensorGraph::TensorNode* dy = nullptr;
    TensorGraph::TensorNode* dx = nullptr;

    TensorGeluBackwardOp() = default;
    TensorGeluBackwardOp(
        TensorGraph::TensorNode* x_,
        TensorGraph::TensorNode* dy_,
        TensorGraph::TensorNode* dx_)
        : x(x_), dy(dy_), dx(dx_)
    {
        inputs_ = {x, dy, dx};
        outputs_ = {dx};
    }

    std::string op_name() const override { return "GELU_BACKWARD"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorGeluBackwardOp>(*this);
    }
    void lower_to_tile(const LoweringContext& ctx) const override;

};

//! GeLU backward: dx += gelu_backward(x, dy)
//! @param x Input tensor (forward pass activation)
//! @param dy Gradient of output (upstream gradient)
//! @param dx Gradient tensor to accumulate into (gradient of input)
void gelu_backward(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    TensorGraph::TensorNode* dx);

} // namespace nntile::graph::tensor
