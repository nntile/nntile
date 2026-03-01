/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/gelu.hh
 * NNGraph GELU autograd operation.
 *
 * Forward: y = gelu(x)
 * Backward: grad_x += gelu_backward(x, grad_y)
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/autograd_function.hh>
#include <nntile/graph/logical/gelu.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! Gelu functor: operator() does bookkeeping; build_forward does logical op only.
struct Gelu : AutogradFunction<Gelu>
{
    //! Forward: logical op only
    static ForwardResult build_forward(
        NNGraph::TensorNode* x,
        const std::string& output_name);

    //! Backward: grad_x += gelu_backward(x, grad_y)
    static void build_backward(const NNGraph::OpNode* op);
};

//! Convenience free function
inline NNGraph::TensorNode* gelu(
    NNGraph::TensorNode* x,
    const std::string& output_name)
{
    return Gelu()(x, output_name);
}

} // namespace nntile::graph
