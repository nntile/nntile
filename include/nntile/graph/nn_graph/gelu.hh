/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/gelu.hh
 * NNGraph GELU autograd operation.
 *
 * Forward: y = gelu(x)
 * Backward: grad_x += gelu_backward(x, grad_y)
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/tensor/gelu.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! GELU op: y = gelu(x). Self-contained: holds tensors.
struct NNGeluOp : NNBaseOpNode
{
    NNGraph::TensorNode* x = nullptr;
    NNGraph::TensorNode* y = nullptr;

    NNGeluOp() = default;
    NNGeluOp(NNGraph::TensorNode* x_, NNGraph::TensorNode* y_)
        : x(x_), y(y_)
    {
        inputs_ = {x};
        outputs_ = {y};
    }

    void add_forward_to_tensor_graph(NNGraph& graph) override;
    void backward() override;
};

NNGraph::TensorNode* gelu(
    NNGraph::TensorNode* x,
    const std::string& output_name);

} // namespace nntile::graph
