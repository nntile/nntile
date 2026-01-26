/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/copy_intersection.hh
 * Logical graph copy_intersection operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>
#include <vector>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Copy intersection operation: copy overlapping regions between tensors
//! @param src Source tensor
//! @param src_offset Offset in source tensor
//! @param dst Destination tensor (modified in-place)
//! @param dst_offset Offset in destination tensor
void copy_intersection(
    LogicalGraph::TensorNode& src,
    const std::vector<Index>& src_offset,
    LogicalGraph::TensorNode& dst,
    const std::vector<Index>& dst_offset
);

} // namespace nntile::graph