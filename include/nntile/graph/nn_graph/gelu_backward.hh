/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/gelu_backward.hh
 * NNGraph GELU backward operation overload.
 *
 * @version 1.1.0
 * */

#pragma once

// Include other NNTile headers
#include <nntile/graph/logical/gelu_backward.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! GeLU backward: dx += gelu_backward(x, dy)
//! Overload for NNGraph::TensorNode
inline void gelu_backward(
    NNGraph::TensorNode& x,
    NNGraph::TensorNode& dy,
    NNGraph::TensorNode& dx)
{
    gelu_backward(x.data(), dy.data(), dx.data());
}

} // namespace nntile::graph
