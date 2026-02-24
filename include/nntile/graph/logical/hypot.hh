/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/hypot.hh
 * Logical graph hypot operation.
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

//! Hypot operation: z = hypot(alpha * x, beta * y)
//! @param x First input tensor
//! @param y Second input tensor
//! @param output_name Name for the output tensor
//! @param alpha Scaling factor for x (default: 1.0)
//! @param beta Scaling factor for y (default: 1.0)
//! @return Reference to the output tensor
LogicalGraph::TensorNode& hypot(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    const std::string& output_name,
    Scalar alpha = 1.0,
    Scalar beta = 1.0
);

} // namespace nntile::graph
