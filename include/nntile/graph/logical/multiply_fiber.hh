/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/multiply_fiber.hh
 * Logical graph multiply fiber operation.
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

//! Multiply along fibers: output = alpha * fiber * tensor
//! @param alpha Scaling factor for fiber * tensor product
//! @param fiber Input fiber tensor (1D)
//! @param tensor Input tensor
//! @param output_name Name for the output tensor
//! @param axis Axis along which to broadcast the fiber
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
//! @return Reference to the output tensor
LogicalGraph::TensorNode& multiply_fiber(
    Scalar alpha,
    LogicalGraph::TensorNode& fiber,
    LogicalGraph::TensorNode& tensor,
    const std::string& output_name,
    Index axis,
    Index batch_ndim = 0
);

} // namespace nntile::graph
