/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/logsumexp.cc
 * TensorGraph logsumexp operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/logsumexp.hh"

#include <stdexcept>
#include <utility>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/logsumexp.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_logsumexp(
    TensorGraph::Runtime& runtime,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::logsumexp<T>(src_t, dst_t);
}

} // namespace

TensorGraph::TensorNode* logsumexp(
    TensorGraph::TensorNode* src,
    const std::string& output_name)
{
    if(src == nullptr)
    {
        throw std::invalid_argument(
            "logsumexp: input tensor must be non-null");
    }

    // dst shape: src.shape[1:] (src has shape[0]=2 for maxsumexp format)
    std::vector<Index> output_shape;
    if(src->ndim() > 1)
    {
        output_shape.assign(src->shape().begin() + 1, src->shape().end());
    }

    TensorGraph::TensorNode* dst = src->graph()->data(
        std::move(output_shape),
        output_name,
        src->dtype());

    logsumexp(src, dst);

    return dst;
}

void logsumexp(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "logsumexp: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "logsumexp: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "logsumexp: input tensors must have the same dtype");
    }

    auto op = std::make_shared<TensorLogsumexpOp>(src, dst);
    src->graph()->add_op(op);
}

void TensorLogsumexpOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_logsumexp<nntile::fp32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_logsumexp<nntile::fp32_fast_tf32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_logsumexp<nntile::fp32_fast_fp16_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_logsumexp<nntile::fp32_fast_bf16_t>(runtime, src, dst);
            break;
        case DataType::FP64:
            run_logsumexp<nntile::fp64_t>(runtime, src, dst);
            break;
        case DataType::FP16:
            run_logsumexp<nntile::fp16_t>(runtime, src, dst);
            break;
        case DataType::BF16:
            run_logsumexp<nntile::bf16_t>(runtime, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for logsumexp operation");
        default:
            throw std::runtime_error("Unsupported data type for logsumexp");
    }
}

} // namespace nntile::graph::tensor
