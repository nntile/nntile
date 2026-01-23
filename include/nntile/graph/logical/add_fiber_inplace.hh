/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/add_fiber_inplace.hh
 * Logical graph add fiber in-place operation.
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

//! Add along fibers in-place: y = alpha * x + beta * y
//! @param x Input tensor
//! @param y Input/output tensor (modified in-place)
//! @param axis Axis along which to perform fiber-wise operation
//! @param alpha Scaling factor for x
//! @param beta Scaling factor for y
void add_fiber_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis,
    Scalar alpha = 1.0,
    Scalar beta = 1.0
);

} // namespace nntile::graph