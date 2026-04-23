/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/total_sum_accum.cc
 * TensorGraph total_sum_accum operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/total_sum_accum.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/total_sum_accum.hh"

namespace nntile::graph::tensor
{



void total_sum_accum(
    Scalar alpha,
    TensorGraph::TensorNode* logsumexp,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* class_labels,
    TensorGraph::TensorNode* val,
    Index ignore_index)
{
    if(logsumexp == nullptr || src == nullptr || class_labels == nullptr ||
       val == nullptr)
    {
        throw std::invalid_argument(
            "total_sum_accum: input tensors must be non-null");
    }
    if(logsumexp->graph() != src->graph() ||
       logsumexp->graph() != class_labels->graph() ||
       logsumexp->graph() != val->graph())
    {
        throw std::invalid_argument(
            "total_sum_accum: input tensors must belong to the same graph");
    }
    if(logsumexp->dtype() != src->dtype())
    {
        throw std::invalid_argument(
            "total_sum_accum: logsumexp and src must have the same dtype");
    }
    if(class_labels->dtype() != DataType::INT64)
    {
        throw std::invalid_argument(
            "total_sum_accum: class_labels must have INT64 dtype");
    }
    if(val->dtype() != DataType::FP32)
    {
        throw std::invalid_argument(
            "total_sum_accum: val must have FP32 dtype");
    }
    validate_same_shape_and_merge(logsumexp, class_labels, "total_sum_accum");
    validate_logsumexp_shape_and_merge(src, logsumexp, "total_sum_accum");

    auto op = std::make_shared<TensorTotalSumAccumOp>(
        alpha, logsumexp, src, class_labels, val, ignore_index);
    logsumexp->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
