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

//! AddFiber functor: operator() does bookkeeping; build_forward does logical op only.
struct AddFiber : AutogradFunction<AddFiber>
{
    //! Forward: logical op only
    static ForwardResult build_forward(
        Scalar alpha,
        NNGraph::TensorNode* fiber,
        Scalar beta,
        NNGraph::TensorNode* tensor,
        const std::string& output_name,
        Index axis = 0,
        Index batch_ndim = 0);

    //! Backward: grad_fiber += alpha*sum_fiber(grad_out), grad_tensor += beta*add_fiber_inplace(grad_out)
    static void build_backward(const NNGraph::OpNode* op);
};

//! Convenience free function (single output)
inline NNGraph::TensorNode* add_fiber(
    Scalar alpha,
    NNGraph::TensorNode* fiber,
    Scalar beta,
    NNGraph::TensorNode* tensor,
    const std::string& output_name,
    Index axis = 0,
    Index batch_ndim = 0)
{
    std::vector<NNGraph::TensorNode*> outs =
        AddFiber()(alpha, fiber, beta, tensor, output_name, axis, batch_ndim);
    return outs.empty() ? nullptr : outs[0];
}

} // namespace nntile::graph
