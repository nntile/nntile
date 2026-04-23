/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/multiply_fiber_inplace.cc
 * TensorGraph multiply_fiber_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/multiply_fiber_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/multiply_fiber_inplace.hh"

namespace nntile::graph::tensor
{



void multiply_fiber_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "multiply_fiber_inplace: input tensors must be non-null");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "multiply_fiber_inplace: src and dst must be distinct tensors");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "multiply_fiber_inplace: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "multiply_fiber_inplace: input tensors must have the same dtype");
    }
    validate_fiber_shape_and_merge(src, dst, axis, 0,
                                  "multiply_fiber_inplace");

    auto op = std::make_shared<TensorMultiplyFiberInplaceOp>(
        alpha, src, dst, axis);
    src->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
