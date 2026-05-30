/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn/ops/mse_loss.hh
 * NNGraph mse_loss autograd operation.
 *
 * Forward: loss = scale * ||x||^2 = scale * sum(x^2)
 * scale=1.0 gives total loss, scale=1/num_values gives mean loss.
 *
 * Backward: grad_x += 2 * scale * x
 * (Gradient over the loss value is implicitly 1.0.)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/nn/graph_op_node.hh>
#include <nntile/tensor/ops/clear.hh>
#include <nntile/tensor/ops/multiply.hh>
#include <nntile/tensor/ops/norm.hh>

namespace nntile
{

//! MseLoss op: loss = scale * ||x||^2 (scalar). PyTorch-style.
struct NNMseLossOp : NNGraph::OpNode
{
    Scalar scale;
    NNGraph::TensorNode *x = nullptr;

    NNMseLossOp(NNGraph::TensorNode *x_, Scalar scale_ = 1.0) :
        scale(scale_), x(x_)
    {
        inputs_ = {x};
    }

    NNGraph::TensorNode *forward();
    void backward() const override;
};

NNGraph::TensorNode *mse_loss(NNGraph::TensorNode *x, Scalar scale = 1.0);

} // namespace nntile
