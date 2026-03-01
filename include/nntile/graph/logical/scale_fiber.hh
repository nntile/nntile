/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/scale_fiber.hh
 * Logical graph scale_fiber operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include other NNTile headers
#include <nntile/graph/logical/sum_fiber.hh>
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Scale along fibers: y = alpha * scale_fiber(x, y)
//! @param x Scaling tensor (broadcasted along fibers)
//! @param y Input/output tensor (modified in-place)
//! @param alpha Scaling factor
//! @param axis Axis along which to broadcast scaling
//! @param batch_ndim Number of trailing batch dimensions
void scale_fiber(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha = 1.0,
    Index axis = 0,
    Index batch_ndim = 0
);

} // namespace nntile::graph
