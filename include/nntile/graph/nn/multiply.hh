/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/multiply.hh
 * NNGraph multiply autograd operation.
 *
 * Forward: output = alpha * x * y
 * Backward: grad_x += alpha * grad_out * y, grad_y += alpha * grad_out * x
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/multiply.hh>

namespace nntile::graph
{

//! Multiply op: output = alpha*x*y. PyTorch-style.
struct NNMultiplyOp : NNGraph::OpNode
{
    Scalar alpha;
    NNGraph::TensorNode* x = nullptr;
    NNGraph::TensorNode* y = nullptr;

    NNMultiplyOp(NNGraph::TensorNode* x_,
                 NNGraph::TensorNode* y_,
                 Scalar alpha_ = 1.0)
        : alpha(alpha_), x(x_), y(y_)
    {
        inputs_ = {x, y};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* multiply(
    NNGraph::TensorNode* x,
    NNGraph::TensorNode* y,
    const std::string& output_name,
    Scalar alpha = 1.0);

} // namespace nntile::graph
