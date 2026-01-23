/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/mask_scalar.hh
 * Logical graph mask_scalar operation.
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

//! Mask scalar operation: conditionally set values based on mask
//! @param mask Boolean mask tensor
//! @param x Input/output tensor (modified in-place)
//! @param val Value to set where mask is true (default: 0.0)
//! @param batch_ndim Number of batch dimensions (default: 0)
void mask_scalar(
    LogicalGraph::TensorNode& mask,
    LogicalGraph::TensorNode& x,
    Scalar val = 0.0,
    Index batch_ndim = 0
);

} // namespace nntile::graph