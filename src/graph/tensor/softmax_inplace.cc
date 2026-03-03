/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/softmax_inplace.cc
 * TensorGraph softmax_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/softmax_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/softmax_inplace.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_softmax_inplace(
    TensorGraph::ExecutionContext& ctx,
    Scalar alpha, Index axis,
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* dst)
{
    auto& maxsumexp_t = ctx.get_tensor<T>(maxsumexp);
    auto& dst_t = ctx.get_tensor<T>(dst);
    nntile::tensor::softmax_inplace<T>(maxsumexp_t, alpha, dst_t, axis);
}

} // namespace

void softmax_inplace(
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* dst,
    Scalar alpha,
    Index axis)
{
    if(maxsumexp == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "softmax_inplace: input tensors must be non-null");
    }
    if(maxsumexp->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "softmax_inplace: input tensors must belong to the same graph");
    }
    if(maxsumexp->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "softmax_inplace: input tensors must have the same dtype");
    }
    // maxsumexp has shape with 2 at axis, dst has full shape

    auto op = std::make_shared<TensorSoftmaxInplaceOp>(
        maxsumexp, dst, alpha, axis);
    maxsumexp->graph()->add_op(op);
}

void TensorSoftmaxInplaceOp::execute(
    TensorGraph::ExecutionContext& ctx) const
{
    DataType dtype = ctx.get_dtype(maxsumexp);

    switch(dtype)
    {
        case DataType::FP32:
            run_softmax_inplace<nntile::fp32_t>(
                ctx, alpha, axis, maxsumexp, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_softmax_inplace<nntile::fp32_fast_tf32_t>(
                ctx, alpha, axis, maxsumexp, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_softmax_inplace<nntile::fp32_fast_fp16_t>(
                ctx, alpha, axis, maxsumexp, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_softmax_inplace<nntile::fp32_fast_bf16_t>(
                ctx, alpha, axis, maxsumexp, dst);
            break;
        case DataType::FP64:
            run_softmax_inplace<nntile::fp64_t>(
                ctx, alpha, axis, maxsumexp, dst);
            break;
        case DataType::FP16:
            run_softmax_inplace<nntile::fp16_t>(
                ctx, alpha, axis, maxsumexp, dst);
            break;
        case DataType::BF16:
            run_softmax_inplace<nntile::bf16_t>(
                ctx, alpha, axis, maxsumexp, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for softmax_inplace operation");
        default:
            throw std::runtime_error(
                "Unsupported data type for softmax_inplace");
    }
}

} // namespace nntile::graph
