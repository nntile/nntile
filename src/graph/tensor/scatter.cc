/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/scatter.cc
 * TensorGraph scatter operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/scatter.hh"

#include <stdexcept>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/scatter.hh"

namespace nntile::graph::tensor
{



void scatter(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "scatter: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "scatter: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "scatter: input tensors must have the same dtype");
    }
    if(src->shape() != dst->shape())
    {
        throw std::invalid_argument(
            "scatter: src and dst must have the same shape");
    }

    auto op = std::make_shared<TensorScatterOp>(src, dst);
    src->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
