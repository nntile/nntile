/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/mse_loss.hh
 * NNGraph mse_loss autograd operation.
 *
 * Forward: loss = scale * ||x||^2 = scale * sum(x^2)
 * scale=1.0 gives total loss, scale=1/num_values gives mean loss.
 *
 * Backward: grad_x += 2 * scale * x * grad_loss
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/clear.hh>
#include <nntile/graph/tensor/multiply.hh>
#include <nntile/graph/tensor/norm.hh>

namespace nntile::graph
{

//! MseLoss op: loss = scale * ||x||^2 (scalar). PyTorch-style.
struct NNMseLossOp : NNGraph::OpNode
{
    Scalar scale;
    NNGraph::TensorNode* x = nullptr;

    NNMseLossOp(NNGraph::TensorNode* x_, Scalar scale_ = 1.0)
        : scale(scale_), x(x_)
    {
        inputs_ = {x};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* mse_loss(
    NNGraph::TensorNode* x,
    const std::string& output_name,
    Scalar scale = 1.0);

} // namespace nntile::graph
