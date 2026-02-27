/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/add.hh
 * NNGraph add operation - out-of-place z = alpha*x + beta*y.
 *
 * Add functor: build_forward() and build_backward() keep forward-backward
 * logic together; backward is registered for TensorNode::backward() dispatch.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/logical/add.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! Add functor: forward and backward in one place
struct Add
{
    //! Forward: z = alpha * x + beta * y
    static NNGraph::TensorNode& build_forward(
        Scalar alpha,
        NNGraph::TensorNode& x,
        Scalar beta,
        NNGraph::TensorNode& y,
        const std::string& output_name);

    //! Backward: grad_x += alpha*grad_z, grad_y += beta*grad_z
    //! Graph, inputs, output, and gradients deduced from op.
    static void build_backward(const NNGraph::OpNode* op);
};

//! Convenience free function
inline NNGraph::TensorNode& add(
    Scalar alpha,
    NNGraph::TensorNode& x,
    Scalar beta,
    NNGraph::TensorNode& y,
    const std::string& output_name)
{
    return Add::build_forward(alpha, x, beta, y, output_name);
}

} // namespace nntile::graph
