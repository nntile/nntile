/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/norm_slice.cc
 * TensorGraph norm_slice operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/norm_slice.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/norm_slice.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_norm_slice(
    TensorGraph::ExecutionContext& ctx,
    Scalar alpha, Scalar beta,
    Index axis, int redux,
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst)
{
    auto& src1_t = ctx.get_tensor<T>(src1);
    auto& src2_t = ctx.get_tensor<T>(src2);
    auto& dst_t = ctx.get_tensor<T>(dst);
    nntile::tensor::norm_slice<T>(
        alpha, src1_t, beta, src2_t, dst_t, axis, redux);
}

} // namespace

TensorGraph::TensorNode* norm_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    const std::string& output_name,
    Index axis,
    int redux)
{
    if(src1 == nullptr || src2 == nullptr)
    {
        throw std::invalid_argument(
            "norm_slice: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph())
    {
        throw std::invalid_argument(
            "norm_slice: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype())
    {
        throw std::invalid_argument(
            "norm_slice: input tensors must have the same dtype");
    }

    std::vector<Index> output_shape = src2->shape();
    TensorGraph::TensorNode* dst = src1->graph()->data(
        std::move(output_shape),
        output_name,
        src1->dtype());

    norm_slice(alpha, src1, beta, src2, dst, axis, redux);

    return dst;
}

void norm_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux)
{
    if(src1 == nullptr || src2 == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "norm_slice: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph() || src1->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "norm_slice: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype() || src1->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "norm_slice: input tensors must have the same dtype");
    }
    if(src2->shape() != dst->shape())
    {
        throw std::invalid_argument(
            "norm_slice: dst shape must match src2 shape");
    }

    auto op = std::make_shared<TensorNormSliceOp>(
        alpha, beta, src1, src2, dst, axis, redux);
    src1->graph()->add_op(op);
}

void TensorNormSliceOp::execute(
    TensorGraph::ExecutionContext& ctx) const
{
    DataType dtype = ctx.get_dtype(src1);

    switch(dtype)
    {
        case DataType::FP32:
            run_norm_slice<nntile::fp32_t>(
                ctx, alpha, beta, axis, redux, src1, src2, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_norm_slice<nntile::fp32_fast_tf32_t>(
                ctx, alpha, beta, axis, redux, src1, src2, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_norm_slice<nntile::fp32_fast_fp16_t>(
                ctx, alpha, beta, axis, redux, src1, src2, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_norm_slice<nntile::fp32_fast_bf16_t>(
                ctx, alpha, beta, axis, redux, src1, src2, dst);
            break;
        case DataType::FP64:
            run_norm_slice<nntile::fp64_t>(
                ctx, alpha, beta, axis, redux, src1, src2, dst);
            break;
        case DataType::FP16:
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for norm_slice operation");
        case DataType::BF16:
            run_norm_slice<nntile::bf16_t>(
                ctx, alpha, beta, axis, redux, src1, src2, dst);
            break;
        default:
            throw std::runtime_error("Unsupported data type for norm_slice");
    }
}

} // namespace nntile::graph
