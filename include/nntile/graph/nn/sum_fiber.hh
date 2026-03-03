/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/sum_fiber.hh
 * NNGraph sum_fiber autograd operation.
 *
 * Forward: y = alpha * sum_fiber(x) + beta * y
 * Backward: grad_x += alpha * add_fiber_inplace(grad_y)
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/sum_fiber.hh>

namespace nntile::graph
{

//! SumFiber op: y = alpha*sum_fiber(x) + beta*y. PyTorch-style: outputs in forward().
struct NNSumFiberOp : NNGraph::OpNode
{
    Scalar alpha = 1.0;
    Scalar beta = 0.0;
    Index axis = 0;
    Index batch_ndim = 0;
    int redux = 0;
    NNGraph::TensorNode* x = nullptr;

    NNSumFiberOp() = default;
    NNSumFiberOp(NNGraph::TensorNode* x_,
                 Index axis_, Index batch_ndim_,
                 int redux_, Scalar alpha_, Scalar beta_)
        : alpha(alpha_), beta(beta_), axis(axis_), batch_ndim(batch_ndim_)
        , redux(redux_), x(x_)
    {
        inputs_ = {x};
    }

    NNGraph::TensorNode* forward(const std::string& output_name) override;
    void backward() const override;
};

NNGraph::TensorNode* sum_fiber(
    NNGraph::TensorNode* x,
    const std::string& output_name,
    Index axis = 0,
    Index batch_ndim = 0,
    int redux = 0,
    Scalar alpha = 1.0,
    Scalar beta = 0.0);

} // namespace nntile::graph
