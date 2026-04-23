/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/softmax.cc
 * TensorGraph softmax operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/softmax.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/softmax.hh"

namespace nntile::graph::tensor
{



TensorGraph::TensorNode* softmax(
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Scalar alpha,
    Index axis)
{
    if(maxsumexp == nullptr || src == nullptr)
    {
        throw std::invalid_argument(
            "softmax: input tensors must be non-null");
    }
    if(maxsumexp->graph() != src->graph())
    {
        throw std::invalid_argument(
            "softmax: input tensors must belong to the same graph");
    }
    if(maxsumexp->dtype() != src->dtype())
    {
        throw std::invalid_argument(
            "softmax: input tensors must have the same dtype");
    }
    // maxsumexp has shape with 2 at axis, src has full shape

    TensorGraph::TensorNode* dst = src->graph()->data(
        src->shape(),
        output_name,
        src->dtype());
    dst->set_axes(src->axes());

    softmax(maxsumexp, src, dst, alpha, axis);

    return dst;
}

void softmax(
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Scalar alpha,
    Index axis)
{
    if(maxsumexp == nullptr || src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "softmax: input tensors must be non-null");
    }
    if(maxsumexp->graph() != src->graph() || maxsumexp->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "softmax: input tensors must belong to the same graph");
    }
    if(maxsumexp->dtype() != src->dtype() || maxsumexp->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "softmax: input tensors must have the same dtype");
    }
    validate_same_shape_and_merge(src, dst, "softmax");
    validate_maxsumexp_shape_and_merge(src, maxsumexp, axis, "softmax");

    auto op = std::make_shared<TensorSoftmaxOp>(
        maxsumexp, src, dst, alpha, axis);
    src->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
