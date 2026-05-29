/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn_graph/ops/add_fiber.hh
 * NNGraph add_fiber autograd operation.
 *
 * Forward: output = alpha * fiber + beta * tensor
 * Backward: grad_fiber += alpha * sum_fiber(grad_out), grad_tensor += beta *
 * grad_out
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/nn_graph/graph_op_node.hh>
#include <nntile/tensor_graph/ops/add_fiber.hh>

namespace nntile
{

//! AddFiber op: output = alpha*fiber + beta*tensor. PyTorch-style: outputs in
//! forward().
struct NNAddFiberOp : NNGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    Index axis;
    Index batch_ndim;
    NNGraph::TensorNode *fiber = nullptr;
    NNGraph::TensorNode *tensor = nullptr;

    NNAddFiberOp(NNGraph::TensorNode *fiber_,
        NNGraph::TensorNode *tensor_,
        Scalar alpha_,
        Scalar beta_,
        Index axis_,
        Index batch_ndim_) :
        alpha(alpha_),
        beta(beta_),
        axis(axis_),
        batch_ndim(batch_ndim_),
        fiber(fiber_),
        tensor(tensor_)
    {
        inputs_ = {fiber, tensor};
    }

    NNGraph::TensorNode *forward();
    void backward() const override;
};

NNGraph::TensorNode *add_fiber(Scalar alpha,
    NNGraph::TensorNode *fiber,
    Scalar beta,
    NNGraph::TensorNode *tensor,
    Index axis,
    Index batch_ndim);

} // namespace nntile
