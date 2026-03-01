/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/sum_fiber.hh
 * NNGraph sum_fiber autograd operation.
 *
 * Forward: y = alpha * sum_fiber(x) + beta * y
 * Backward: grad_x += alpha * add_fiber_inplace(grad_y) (broadcast grad_y back)
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/autograd_function.hh>
#include <nntile/graph/logical/sum_fiber.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! SumFiber functor: operator() does bookkeeping; build_forward does logical op only.
struct SumFiber : AutogradFunction<SumFiber>
{
    //! Forward: logical op only
    static ForwardResult build_forward(
        NNGraph::TensorNode* x,
        const std::string& output_name,
        Index axis = 0,
        Index batch_ndim = 0,
        int redux = 0,
        Scalar alpha = 1.0,
        Scalar beta = 0.0);

    //! Backward: grad_x += alpha * add_fiber_inplace(grad_y)
    static void build_backward(const NNGraph::OpNode* op);
};

//! Convenience free function (single output)
inline NNGraph::TensorNode* sum_fiber(
    NNGraph::TensorNode* x,
    const std::string& output_name,
    Index axis = 0,
    Index batch_ndim = 0,
    int redux = 0,
    Scalar alpha = 1.0,
    Scalar beta = 0.0)
{
    std::vector<NNGraph::TensorNode*> outs =
        SumFiber()(x, output_name, axis, batch_ndim, redux, alpha, beta);
    return outs.empty() ? nullptr : outs[0];
}

} // namespace nntile::graph
