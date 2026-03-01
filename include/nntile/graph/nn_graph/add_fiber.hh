/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/add_fiber.hh
 * NNGraph add_fiber autograd operation.
 *
 * Forward: output = alpha * fiber + beta * tensor
 * Backward: grad_fiber += alpha * sum_fiber(grad_out), grad_tensor += beta * add_fiber_inplace(grad_out)
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/autograd_function.hh>
#include <nntile/graph/logical/add_fiber.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! AddFiber: build_forward does logical op + bookkeeping; build_backward for grad.
struct AddFiber
{
    static NNGraph::TensorNode* build_forward(
        Scalar alpha,
        NNGraph::TensorNode* fiber,
        Scalar beta,
        NNGraph::TensorNode* tensor,
        const std::string& output_name,
        Index axis = 0,
        Index batch_ndim = 0);

    static void build_backward(const NNGraph::OpNode* op);
};

//! Convenience free function
inline NNGraph::TensorNode* add_fiber(
    Scalar alpha,
    NNGraph::TensorNode* fiber,
    Scalar beta,
    NNGraph::TensorNode* tensor,
    const std::string& output_name,
    Index axis = 0,
    Index batch_ndim = 0)
{
    return AddFiber::build_forward(alpha, fiber, beta, tensor, output_name,
                                  axis, batch_ndim);
}

} // namespace nntile::graph
