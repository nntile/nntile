/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/sum_fiber.hh
 * NNGraph sum_fiber operation overload.
 *
 * @version 1.1.0
 * */

#pragma once

// Include other NNTile headers
#include <nntile/graph/logical/sum_fiber.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! Sum along fibers: y = alpha * sum_fiber(x) + beta * y
//! Overload for NNGraph::TensorNode
inline void sum_fiber(
    NNGraph::TensorNode& x,
    NNGraph::TensorNode& y,
    Index axis = 0,
    Index batch_ndim = 0,
    int redux = 0,
    Scalar alpha = 1.0,
    Scalar beta = 0.0)
{
    sum_fiber(x.data(), y.data(), axis, batch_ndim, redux, alpha, beta);
}

} // namespace nntile::graph
