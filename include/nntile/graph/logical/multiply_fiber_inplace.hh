/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/multiply_fiber_inplace.hh
 * Logical graph multiply fiber in-place operation.
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

//! Multiply along fibers in-place: tensor = alpha * fiber * tensor
//! @param alpha Scaling factor for fiber * tensor product
//! @param fiber Input fiber tensor (1D)
//! @param tensor Input/output tensor (modified in-place)
//! @param axis Axis along which to perform fiber-wise operation
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
void multiply_fiber_inplace(
    Scalar alpha,
    LogicalGraph::TensorNode& fiber,
    LogicalGraph::TensorNode& tensor,
    Index axis,
    Index batch_ndim = 0
);

} // namespace nntile::graph
