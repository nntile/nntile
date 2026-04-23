/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/copy_intersection.cc
 * TensorGraph copy_intersection operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/copy_intersection.hh"

#include <stdexcept>

#include "nntile/graph/tensor.hh"
#include "nntile/tensor/copy_intersection.hh"

namespace nntile::graph::tensor
{



void copy_intersection(TensorGraph::TensorNode* src,
                       const std::vector<Index>& src_offset,
                       TensorGraph::TensorNode* dst,
                       const std::vector<Index>& dst_offset)
{
    if(src == nullptr || dst == nullptr)
        throw std::invalid_argument(
            "copy_intersection: tensors must be non-null");
    if(src->graph() != dst->graph())
        throw std::invalid_argument(
            "copy_intersection: tensors must belong to same graph");
    if(src->dtype() != dst->dtype())
        throw std::invalid_argument(
            "copy_intersection: tensors must have same dtype");
    if(src_offset.size() != src->ndim() || dst_offset.size() != dst->ndim())
        throw std::invalid_argument(
            "copy_intersection: offset sizes must match tensor ndim");
    auto op = std::make_shared<TensorCopyIntersectionOp>(
        src, src_offset, dst, dst_offset);
    dst->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
