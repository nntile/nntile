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

#include <nntile/graph/tensor_graph.hh>
#include <nntile/graph/base_op_node.hh>

namespace nntile::graph
{

//! GeLU backward operation at tensor level: dx += gelu_backward(x, dy)
struct TensorGeluBackwardOp : BaseOpNode<TensorGraph, TensorGraphNode>
{
    TensorGraph::DataNode* x = nullptr;
    TensorGraph::DataNode* dy = nullptr;
    TensorGraph::DataNode* dx = nullptr;

    TensorGeluBackwardOp() = default;
    TensorGeluBackwardOp(
        TensorGraph::DataNode* x_,
        TensorGraph::DataNode* dy_,
        TensorGraph::DataNode* dx_)
        : x(x_), dy(dy_), dx(dx_)
    {
        inputs_ = {x, dy, dx};
        outputs_ = {dx};
    }

    std::string op_name() const override { return "GELU_BACKWARD"; }

    void execute(ExecutionContext<TensorGraph::DataNode>& ctx) const override;

    std::shared_ptr<BaseOpNode<TensorGraph, TensorGraphNode>> clone() const override
    {
        return std::make_shared<TensorGeluBackwardOp>(*this);
    }
};

//! GeLU backward: dx += gelu_backward(x, dy)
//! @param x Input tensor (forward pass activation)
//! @param dy Gradient of output (upstream gradient)
//! @param dx Gradient tensor to accumulate into (gradient of input)
void gelu_backward(
    TensorGraph::DataNode* x,
    TensorGraph::DataNode* dy,
    TensorGraph::DataNode* dx);

} // namespace nntile::graph
