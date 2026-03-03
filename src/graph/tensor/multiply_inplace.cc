/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/multiply_inplace.cc
 * TensorGraph multiply_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/multiply_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/multiply_inplace.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_multiply_inplace(
    TensorGraph::ExecutionContext& ctx,
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = ctx.get_tensor<T>(src);
    auto& dst_t = ctx.get_tensor<T>(dst);
    nntile::tensor::multiply_inplace<T>(alpha, src_t, dst_t);
}

} // namespace

void multiply_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "multiply_inplace: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "multiply_inplace: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "multiply_inplace: input tensors must have the same dtype");
    }
    if(src->shape() != dst->shape())
    {
        throw std::invalid_argument(
            "multiply_inplace: input tensors must have the same shape");
    }

    auto op = std::make_shared<TensorMultiplyInplaceOp>(src, dst, alpha);
    src->graph()->add_op(op);
}

void TensorMultiplyInplaceOp::execute(
    TensorGraph::ExecutionContext& ctx) const
{
    DataType dtype = ctx.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_multiply_inplace<nntile::fp32_t>(ctx, alpha, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_multiply_inplace<nntile::fp32_fast_tf32_t>(ctx, alpha, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_multiply_inplace<nntile::fp32_fast_fp16_t>(ctx, alpha, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_multiply_inplace<nntile::fp32_fast_bf16_t>(ctx, alpha, src, dst);
            break;
        case DataType::FP64:
            run_multiply_inplace<nntile::fp64_t>(ctx, alpha, src, dst);
            break;
        case DataType::FP16:
            run_multiply_inplace<nntile::fp16_t>(ctx, alpha, src, dst);
            break;
        case DataType::BF16:
            run_multiply_inplace<nntile::bf16_t>(ctx, alpha, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for multiply_inplace");
        default:
            throw std::runtime_error("Unsupported data type for multiply_inplace");
    }
}

} // namespace nntile::graph
