/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/add_slice_inplace.hh
 * Logical graph add slice in-place operation.
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

//! Add along slices in-place: tensor = alpha * slice + beta * tensor
//! @param alpha Scaling factor for slice
//! @param slice Input slice tensor
//! @param beta Scaling factor for tensor
//! @param tensor Input/output tensor (modified in-place)
//! @param axis Axis along which to perform slice-wise operation
void add_slice_inplace(
    Scalar alpha,
    LogicalGraph::TensorNode& slice,
    Scalar beta,
    LogicalGraph::TensorNode& tensor,
    Index axis
);

} // namespace nntile::graph