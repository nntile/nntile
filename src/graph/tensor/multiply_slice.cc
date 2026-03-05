/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/multiply_slice.cc
 * TensorGraph multiply_slice operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/multiply_slice.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/multiply_slice.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_multiply_slice(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Index axis,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::multiply_slice<T>(alpha, src_t, dst_t, axis);
}

} // namespace

void multiply_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "multiply_slice: input tensors must be non-null");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "multiply_slice: src and dst must be distinct tensors");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "multiply_slice: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "multiply_slice: input tensors must have the same dtype");
    }
    if(dst->ndim() != src->ndim() + 1)
    {
        throw std::invalid_argument(
            "multiply_slice: dst must have ndim = src.ndim + 1");
    }
    if(axis < 0 || axis >= dst->ndim())
    {
        throw std::invalid_argument(
            "multiply_slice: axis out of range");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(dst->shape()[i] != src->shape()[i])
        {
            throw std::invalid_argument(
                "multiply_slice: dst.shape[i] must match src.shape[i] for i < axis");
        }
    }
    for(Index i = axis + 1; i < dst->ndim(); ++i)
    {
        if(dst->shape()[i] != src->shape()[i - 1])
        {
            throw std::invalid_argument(
                "multiply_slice: dst.shape[i] must match src.shape[i-1] for i > axis");
        }
    }

    auto op = std::make_shared<TensorMultiplySliceOp>(alpha, src, dst, axis);
    src->graph()->add_op(op);
}

void TensorMultiplySliceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_multiply_slice<nntile::fp32_t>(runtime, alpha, axis, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_multiply_slice<nntile::fp32_fast_tf32_t>(runtime, alpha, axis, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_multiply_slice<nntile::fp32_fast_fp16_t>(runtime, alpha, axis, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_multiply_slice<nntile::fp32_fast_bf16_t>(runtime, alpha, axis, src, dst);
            break;
        case DataType::FP64:
            run_multiply_slice<nntile::fp64_t>(runtime, alpha, axis, src, dst);
            break;
        case DataType::FP16:
            run_multiply_slice<nntile::fp16_t>(runtime, alpha, axis, src, dst);
            break;
        case DataType::BF16:
            run_multiply_slice<nntile::bf16_t>(runtime, alpha, axis, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for multiply_slice operation");
        default:
            throw std::runtime_error("Unsupported data type for multiply_slice");
    }
}

} // namespace nntile::graph::tensor
