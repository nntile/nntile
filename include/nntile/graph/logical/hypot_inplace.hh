/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/hypot_inplace.hh
 * Logical graph hypot_inplace operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Hypot in-place: y = hypot(alpha * x, beta * y)
//! @param x First input tensor
//! @param y Second input/output tensor (modified in-place)
//! @param alpha Scaling factor for x (default: 1.0)
//! @param beta Scaling factor for y (default: 1.0)
void hypot_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha = 1.0,
    Scalar beta = 1.0
);

} // namespace nntile::graph
