/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/copy_intersection.cc
 * Logical graph copy_intersection operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/copy_intersection.hh"

// Include standard headers
#include <stdexcept>
#include <utility>
#include <vector>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Copy intersection operation: copy overlapping regions between tensors
void copy_intersection(
    LogicalGraph::TensorNode& src,
    const std::vector<Index>& src_offset,
    LogicalGraph::TensorNode& dst,
    const std::vector<Index>& dst_offset)
{
    if(&src.graph() != &dst.graph())
    {
        throw std::invalid_argument(
            "copy_intersection: tensors must belong to the same graph");
    }

    if(src.dtype() != dst.dtype())
    {
        throw std::invalid_argument(
            "copy_intersection: tensors must have the same dtype");
    }

    auto attrs = std::make_shared<CopyIntersectionAttrs>(CopyIntersectionAttrs{src_offset, dst_offset});
    src.graph().add_op(
        OpType::COPY_INTERSECTION,
        attrs,
        {&src, &dst},
        {&dst}
    );
}

} // namespace nntile::graph
