/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/add_slice.hh
 * Logical graph add slice operation.
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

//! Add along slices: output = alpha * slice + beta * tensor
//! @param alpha Scaling factor for slice
//! @param slice Input slice tensor
//! @param beta Scaling factor for tensor
//! @param tensor Input tensor
//! @param output_name Name for the output tensor
//! @param axis Axis along which to perform slice-wise operation
//! @return Reference to the output tensor
LogicalGraph::TensorNode& add_slice(
    Scalar alpha,
    LogicalGraph::TensorNode& slice,
    Scalar beta,
    LogicalGraph::TensorNode& tensor,
    const std::string& output_name,
    Index axis
);

} // namespace nntile::graph
