/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/norm_slice.hh
 * Logical graph norm_slice operation.
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

//! Norm along slices (out-of-place): z = alpha * norm_slice(x) + beta * y
//! Use norm_slice_inplace when y and z should be the same tensor.
//! @param x Input tensor
//! @param y Accumulation input (must be different from z)
//! @param z Output tensor (must be different from y)
//! @param axis Axis along which to compute norm
//! @param redux Whether to use reduction (default: 0)
//! @param alpha Scaling factor for norm (default: 1.0)
//! @param beta Scaling factor for y (default: 0.0)
void norm_slice(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    LogicalGraph::TensorNode& z,
    Index axis,
    int redux = 0,
    Scalar alpha = 1.0,
    Scalar beta = 0.0
);

} // namespace nntile::graph
