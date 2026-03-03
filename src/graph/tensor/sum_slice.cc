/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sum_slice.cc
 * TensorGraph sum_slice operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sum_slice.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sum_slice.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_sum_slice(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    Index axis, int redux,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::sum_slice<T>(alpha, src_t, beta, dst_t, axis, redux);
}

} // namespace

void sum_slice(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "sum_slice: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "sum_slice: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "sum_slice: input tensors must have the same dtype");
    }
    if(src->ndim() - 1 != dst->ndim())
    {
        throw std::invalid_argument(
            "sum_slice: dst must have ndim = src.ndim - 1");
    }
    if(axis < 0 || axis >= src->ndim())
    {
        throw std::invalid_argument(
            "sum_slice: axis out of range");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "sum_slice: src and dst must be distinct tensors");
    }

    auto op = std::make_shared<TensorSumSliceOp>(
        src, dst, axis, redux, alpha, beta);
    src->graph()->add_op(op);
}

void TensorSumSliceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_sum_slice<nntile::fp32_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_sum_slice<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_sum_slice<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_sum_slice<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP64:
            run_sum_slice<nntile::fp64_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP16:
            run_sum_slice<nntile::fp16_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::BF16:
            run_sum_slice<nntile::bf16_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sum_slice operation");
        default:
            throw std::runtime_error("Unsupported data type for sum_slice");
    }
}

} // namespace nntile::graph
