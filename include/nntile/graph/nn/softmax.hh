/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/softmax.hh
 * NNGraph softmax autograd operation.
 *
 * Forward: y = softmax(x, axis) via maxsumexp + softmax_inplace
 * Backward: grad_x = y * (grad_y - sumprod_slice(y, grad_y))
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/nn/graph_op_node.hh>

namespace nntile::graph
{

//! Softmax op: y = softmax(x, axis). PyTorch-style: outputs created in forward().
struct NNSoftmaxOp : NNGraph::OpNode
{
    Index axis;
    int redux;
    NNGraph::TensorNode* x = nullptr;

    NNSoftmaxOp() = default;
    explicit NNSoftmaxOp(NNGraph::TensorNode* x_, Index axis_ = 0,
                        int redux_ = 0)
        : axis(axis_), redux(redux_), x(x_)
    {
        inputs_ = {x};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* softmax(
    NNGraph::TensorNode* x,
    const std::string& output_name,
    Index axis = 0,
    int redux = 0);

} // namespace nntile::graph
