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

#include <nntile/graph/logical/gelu.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! Gelu functor: callable, forward and backward in one place
struct Gelu
{
    //! Callable: y = gelu(x)
    NNGraph::TensorNode* operator()(
        NNGraph::TensorNode* x,
        const std::string& output_name) const
    {
        return build_forward(x, output_name);
    }

    //! Forward: y = gelu(x)
    static NNGraph::TensorNode* build_forward(
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
    return Gelu::build_forward(x, output_name);
}

} // namespace nntile::graph
