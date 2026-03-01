/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/softmax.hh
 * Logical graph softmax operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include other NNTile headers
#include <nntile/graph/logical/logsumexp.hh>
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Softmax operation: y = softmax(maxsumexp, x, alpha)
//! @param maxsumexp Max and sum tensor from maxsumexp operation
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @param alpha Scaling factor (default: 1.0)
//! @param axis Axis along which to compute softmax (default: -1, last axis)
//! @return Reference to the output tensor
LogicalGraph::TensorNode& softmax(
    LogicalGraph::TensorNode& maxsumexp,
    LogicalGraph::TensorNode& x,
    const std::string& output_name,
    Scalar alpha = 1.0,
    Index axis = -1
);

} // namespace nntile::graph
