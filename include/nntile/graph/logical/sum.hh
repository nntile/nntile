/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/sum.hh
 * Logical graph sum operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include other NNTile headers
#include <nntile/graph/logical/norm.hh>
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Total sum of all elements: y = alpha * sum(x) + beta * y
//! @param x Input tensor
//! @param y Output tensor to accumulate into
//! @param alpha Scaling factor for sum (default: 1.0)
//! @param beta Scaling factor for existing y (default: 0.0)
void sum(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha = 1.0,
    Scalar beta = 0.0
);

} // namespace nntile::graph
