/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/scale_inplace.hh
 * Logical graph scale in-place operation.
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

//! Scale in-place: x = alpha * x
//! @param x Input/output tensor (modified in-place)
//! @param alpha Scaling factor (default: 1.0)
void scale_inplace(
    LogicalGraph::TensorNode& x,
    Scalar alpha = 1.0
);

} // namespace nntile::graph
