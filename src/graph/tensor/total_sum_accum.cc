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

namespace
{

template<typename T>
void run_total_sum_accum(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Index ignore_index,
    TensorGraph::TensorNode* logsumexp,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* class_labels,
    TensorGraph::TensorNode* val)
{
    auto& logsumexp_t = runtime.get_tensor<T>(logsumexp);
    auto& src_t = runtime.get_tensor<T>(src);
    auto& class_labels_t = runtime.get_tensor<int64_t>(class_labels);
    auto& val_t = runtime.get_tensor<nntile::fp32_t>(val);
    nntile::tensor::total_sum_accum<T>(
        alpha, logsumexp_t, src_t, class_labels_t, val_t, ignore_index);
}

} // namespace

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

    // logsumexp and class_labels: same shape
    if(logsumexp->ndim() == class_labels->ndim())
    {
        for(Index i = 0; i < logsumexp->ndim(); ++i)
        {
            merge_axis(logsumexp->mutable_axes()[i],
                       class_labels->mutable_axes()[i]);
        }
    }
    // src.dim[1:] matches logsumexp (src has extra first dim for classes)
    if(src->ndim() == logsumexp->ndim() + 1)
    {
        for(Index i = 0; i < logsumexp->ndim(); ++i)
        {
            merge_axis(src->mutable_axes()[static_cast<size_t>(i + 1)],
                       logsumexp->mutable_axes()[i]);
        }
    }

    auto op = std::make_shared<TensorTotalSumAccumOp>(
        alpha, logsumexp, src, class_labels, val, ignore_index);
    logsumexp->graph()->add_op(op);
}

void TensorTotalSumAccumOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(logsumexp);

    switch(dtype)
    {
        case DataType::FP32:
            run_total_sum_accum<nntile::fp32_t>(
                runtime, alpha, ignore_index,
                logsumexp, src, class_labels, val);
            break;
        case DataType::FP32_FAST_TF32:
            run_total_sum_accum<nntile::fp32_fast_tf32_t>(
                runtime, alpha, ignore_index,
                logsumexp, src, class_labels, val);
            break;
        case DataType::FP32_FAST_FP16:
            run_total_sum_accum<nntile::fp32_fast_fp16_t>(
                runtime, alpha, ignore_index,
                logsumexp, src, class_labels, val);
            break;
        case DataType::FP32_FAST_BF16:
            run_total_sum_accum<nntile::fp32_fast_bf16_t>(
                runtime, alpha, ignore_index,
                logsumexp, src, class_labels, val);
            break;
        case DataType::FP64:
            run_total_sum_accum<nntile::fp64_t>(
                runtime, alpha, ignore_index,
                logsumexp, src, class_labels, val);
            break;
        case DataType::FP16:
            run_total_sum_accum<nntile::fp16_t>(
                runtime, alpha, ignore_index,
                logsumexp, src, class_labels, val);
            break;
        case DataType::BF16:
            run_total_sum_accum<nntile::bf16_t>(
                runtime, alpha, ignore_index,
                logsumexp, src, class_labels, val);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for total_sum_accum operation");
        default:
            throw std::runtime_error(
                "Unsupported data type for total_sum_accum");
    }
}

} // namespace nntile::graph::tensor
