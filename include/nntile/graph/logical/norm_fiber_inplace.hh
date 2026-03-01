/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/norm_fiber_inplace.hh
 * Logical graph norm_fiber_inplace operation.
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

//! Norm along fibers (in-place): y = alpha * norm_fiber(x) + beta * y
//! @param x Input tensor
//! @param y Output tensor to accumulate into
//! @param axis Axis along which to compute norm
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
//! @param redux Whether to use reduction (default: 0)
//! @param alpha Scaling factor for norm (default: 1.0)
//! @param beta Scaling factor for existing y (default: 0.0)
void norm_fiber_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis,
    Index batch_ndim = 0,
    int redux = 0,
    Scalar alpha = 1.0,
    Scalar beta = 0.0
);

} // namespace nntile::graph
