/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/add_fiber_inplace.hh
 * NNGraph add_fiber_inplace operation overload.
 *
 * @version 1.1.0
 * */

#pragma once

// Include other NNTile headers
#include <nntile/graph/logical/add_fiber_inplace.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! Add along fibers in-place: tensor = alpha * fiber + beta * tensor
//! Overload for NNGraph::TensorNode
inline void add_fiber_inplace(
    Scalar alpha,
    NNGraph::TensorNode& fiber,
    Scalar beta,
    NNGraph::TensorNode& tensor,
    Index axis,
    Index batch_ndim = 0)
{
    add_fiber_inplace(alpha, fiber.data(), beta, tensor.data(), axis,
                     batch_ndim);
}

} // namespace nntile::graph
