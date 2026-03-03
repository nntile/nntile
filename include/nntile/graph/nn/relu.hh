/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/relu.hh
 * NNGraph ReLU autograd operation.
 *
 * Forward: y = relu(x)
 * Backward: grad_x += relu_backward(x, grad_y)
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/relu.hh>

namespace nntile::graph
{

//! ReLU op: y = relu(x). PyTorch-style: outputs created in forward().
struct NNReluOp : NNGraph::OpNode
{
    NNGraph::TensorNode* x = nullptr;

    NNReluOp() = default;
    explicit NNReluOp(NNGraph::TensorNode* x_)
        : x(x_)
    {
        inputs_ = {x};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* relu(
    NNGraph::TensorNode* x,
    const std::string& output_name);

} // namespace nntile::graph
