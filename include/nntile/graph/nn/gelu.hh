/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/gelu.hh
 * NNGraph GELU autograd operation.
 *
 * Forward: y = gelu(x)
 * Backward: grad_x += gelu_backward(x, grad_y)
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/gelu.hh>

namespace nntile::graph
{

//! GELU op: y = gelu(x). PyTorch-style: outputs created in forward().
struct NNGeluOp : NNGraph::OpNode
{
    NNGraph::TensorNode* x = nullptr;

    NNGeluOp() = default;
    explicit NNGeluOp(NNGraph::TensorNode* x_)
        : x(x_)
    {
        inputs_ = {x};
    }

    NNGraph::TensorNode* forward(const std::string& output_name) override;
    void backward() const override;
};

NNGraph::TensorNode* gelu(
    NNGraph::TensorNode* x,
    const std::string& output_name);

} // namespace nntile::graph
