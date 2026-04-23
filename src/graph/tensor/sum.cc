/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sum.cc
 * TensorGraph sum operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sum.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sum.hh"

namespace nntile::graph::tensor
{



void sum(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Scalar alpha,
    Scalar beta)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "sum: input tensors must be non-null");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "sum: src and dst must be distinct tensors");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "sum: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "sum: input tensors must have the same dtype");
    }
    if(dst->ndim() != 0)
    {
        throw std::invalid_argument(
            "sum: dst must be a scalar (0-dimensional tensor)");
    }

    auto op = std::make_shared<TensorSumOp>(src, dst, alpha, beta);
    src->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
