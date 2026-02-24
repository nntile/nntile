/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/sumprod_slice.hh
 * Logical graph sumprod_slice operation.
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

//! Sum of products along slices: y = alpha * sum_slice(x1 * x2) + beta * y
//! @param x1 First input tensor
//! @param x2 Second input tensor
//! @param y Output tensor to accumulate into
//! @param axis Axis along which to sum
//! @param redux Whether to use reduction (default: 0)
//! @param alpha Scaling factor for sum (default: 1.0)
//! @param beta Scaling factor for existing y (default: 0.0)
void sumprod_slice(
    LogicalGraph::TensorNode& x1,
    LogicalGraph::TensorNode& x2,
    LogicalGraph::TensorNode& y,
    Index axis,
    int redux = 0,
    Scalar alpha = 1.0,
    Scalar beta = 0.0
);

} // namespace nntile::graph
