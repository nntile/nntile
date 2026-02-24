/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/gelu.hh
 * NNGraph GELU operation overload.
 *
 * @version 1.1.0
 * */

#pragma once

// Include other NNTile headers
#include <nntile/graph/logical/gelu.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! GeLU activation into pre-created output: y = gelu(x)
//! Overload for NNGraph::TensorNode
inline void gelu(
    NNGraph::TensorNode& x,
    NNGraph::TensorNode& y)
{
    gelu(x.data(), y.data());
}

} // namespace nntile::graph
