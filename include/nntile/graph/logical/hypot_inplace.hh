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
#include <nntile/graph/logical/add.hh>
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Hypot in-place: y = hypot(alpha * x, beta * y)
//! @param alpha Scaling factor for x
//! @param x First input tensor
//! @param beta Scaling factor for y
//! @param y Second input/output tensor (modified in-place)
void hypot_inplace(
    Scalar alpha,
    LogicalGraph::TensorNode& x,
    Scalar beta,
    LogicalGraph::TensorNode& y
);

} // namespace nntile::graph
