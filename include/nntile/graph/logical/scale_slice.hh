/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/scale_slice.hh
 * Logical graph scale_slice operation.
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

//! Scale along slices: y = alpha * scale_slice(x, y)
//! @param x Scaling tensor (broadcasted along slices)
//! @param y Input/output tensor (modified in-place)
//! @param alpha Scaling factor
//! @param axis Axis along which to broadcast scaling
void scale_slice(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha = 1.0,
    Index axis = 0
);

} // namespace nntile::graph