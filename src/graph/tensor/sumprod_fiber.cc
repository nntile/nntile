/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sumprod_fiber.cc
 * TensorGraph sumprod_fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sumprod_fiber.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sumprod_fiber.hh"

namespace nntile::graph::tensor
{



void sumprod_fiber(
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(src1 == nullptr || src2 == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "sumprod_fiber: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph() || src1->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "sumprod_fiber: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype() || src1->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "sumprod_fiber: input tensors must have the same dtype");
    }
    if(dst->ndim() != 1)
    {
        throw std::invalid_argument(
            "sumprod_fiber: dst must be 1-dimensional");
    }
    if(axis < 0 || axis >= src1->ndim())
    {
        throw std::invalid_argument(
            "sumprod_fiber: axis out of range");
    }
    if(src1 == src2 || src1 == dst || src2 == dst)
    {
        throw std::invalid_argument(
            "sumprod_fiber: src1, src2, and dst must be distinct tensors");
    }

    validate_same_shape_and_merge(src1, src2, "sumprod_fiber");
    // Merge dst (reduced fiber) axis with src1 axis
    merge_axis(dst->mutable_axes()[0], src1->mutable_axes()[axis]);

    auto op = std::make_shared<TensorSumprodFiberOp>(
        src1, src2, dst, axis, redux, alpha, beta);
    src1->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
