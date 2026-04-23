/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/norm_slice_inplace.cc
 * TensorGraph norm_slice_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/norm_slice_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/norm_slice_inplace.hh"

namespace nntile::graph::tensor
{



void norm_slice_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    Scalar beta,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "norm_slice_inplace: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "norm_slice_inplace: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "norm_slice_inplace: input tensors must have the same dtype");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "norm_slice_inplace: src and dst must be distinct tensors");
    }
    validate_slice_shape_and_merge(dst, src, axis, "norm_slice_inplace");

    auto op = std::make_shared<TensorNormSliceInplaceOp>(
        alpha, beta, src, dst, axis, redux);
    src->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
