/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/relu_backward.hh
 * TensorGraph relu_backward operation: dx = relu_backward(x, dy)
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! ReLU backward operation: dx = relu_backward(x, dy)
struct TensorReluBackwardOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* x = nullptr;
    TensorGraph::TensorNode* dy = nullptr;
    TensorGraph::TensorNode* dx = nullptr;

    TensorReluBackwardOp() = default;
    TensorReluBackwardOp(TensorGraph::TensorNode* x_,
                         TensorGraph::TensorNode* dy_,
                         TensorGraph::TensorNode* dx_)
        : x(x_), dy(dy_), dx(dx_)
    {
        inputs_ = {x, dy, dx};
        outputs_ = {dx};
    }

    std::string op_name() const override { return "RELU_BACKWARD"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorReluBackwardOp>(*this);
    }
};

//! ReLU backward: dx = relu_backward(x, dy) (creates output)
TensorGraph::TensorNode* relu_backward(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    const std::string& output_name);

//! ReLU backward: dx = relu_backward(x, dy) (uses existing output)
void relu_backward(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    TensorGraph::TensorNode* dx);

} // namespace nntile::graph
