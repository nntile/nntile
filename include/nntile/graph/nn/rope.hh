/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/rope.hh
 * NNGraph RoPE (Rotary Position Embedding) autograd operation.
 *
 * Forward: y = rope(sin, cos, x)
 * Backward: grad_x = rope_backward(sin, cos, grad_y)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/graph/nn/graph_op_node.hh>

namespace nntile::graph
{

//! RoPE op: y = rope(sin, cos, x). PyTorch-style: outputs created in forward().
//! sin and cos are positional embeddings (typically no grad); only x gets grad.
struct NNRopeOp : NNGraph::OpNode
{
    NNGraph::TensorNode* sin = nullptr;
    NNGraph::TensorNode* cos = nullptr;
    NNGraph::TensorNode* x = nullptr;

    NNRopeOp() = default;
    NNRopeOp(NNGraph::TensorNode* sin_,
             NNGraph::TensorNode* cos_,
             NNGraph::TensorNode* x_)
        : sin(sin_), cos(cos_), x(x_)
    {
        inputs_ = {sin, cos, x};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* rope(
    NNGraph::TensorNode* sin,
    NNGraph::TensorNode* cos,
    NNGraph::TensorNode* x,
    const std::string& output_name);

} // namespace nntile::graph
