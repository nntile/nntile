/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/conv2d_inplace.cc
 * TensorGraph conv2d_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/conv2d_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/conv2d_inplace.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_conv2d_inplace(TensorGraph::ExecutionContext& ctx,
                       Scalar alpha, TensorGraph::TensorNode* X,
                       TensorGraph::TensorNode* C, Scalar beta,
                       TensorGraph::TensorNode* Y,
                       const std::array<Index, 2>& padding,
                       const std::array<Index, 2>& stride,
                       const std::array<Index, 2>& dilation)
{
    auto& X_t = ctx.get_tensor<T>(X);
    auto& C_t = ctx.get_tensor<T>(C);
    auto& Y_t = ctx.get_tensor<T>(Y);
    nntile::tensor::conv2d_inplace<T>(
        alpha, X_t, C_t, beta, Y_t, padding, stride, dilation);
}

} // namespace

void conv2d_inplace(Scalar alpha,
                    TensorGraph::TensorNode* X,
                    TensorGraph::TensorNode* C,
                    Scalar beta,
                    TensorGraph::TensorNode* Y,
                    std::array<Index, 2> padding,
                    std::array<Index, 2> stride,
                    std::array<Index, 2> dilation)
{
    if(X == nullptr || C == nullptr || Y == nullptr)
        throw std::invalid_argument("conv2d_inplace: tensors must be non-null");
    if(X->graph() != C->graph() || C->graph() != Y->graph())
        throw std::invalid_argument(
            "conv2d_inplace: tensors must belong to same graph");
    if(X->dtype() != C->dtype() || C->dtype() != Y->dtype())
        throw std::invalid_argument(
            "conv2d_inplace: tensors must have same dtype");
    auto op = std::make_shared<TensorConv2dInplaceOp>(
        alpha, X, C, beta, Y, padding, stride, dilation);
    Y->graph()->add_op(op);
}

void TensorConv2dInplaceOp::execute(
    TensorGraph::ExecutionContext& ctx) const
{
    DataType dtype = ctx.get_dtype(X);
    switch(dtype)
    {
        case DataType::FP32:
            run_conv2d_inplace<nntile::fp32_t>(
                ctx, alpha, X, C, beta, Y, padding, stride, dilation);
            break;
        case DataType::FP32_FAST_TF32:
            run_conv2d_inplace<nntile::fp32_fast_tf32_t>(
                ctx, alpha, X, C, beta, Y, padding, stride, dilation);
            break;
        case DataType::FP32_FAST_FP16:
            run_conv2d_inplace<nntile::fp32_fast_fp16_t>(
                ctx, alpha, X, C, beta, Y, padding, stride, dilation);
            break;
        case DataType::FP32_FAST_BF16:
            run_conv2d_inplace<nntile::fp32_fast_bf16_t>(
                ctx, alpha, X, C, beta, Y, padding, stride, dilation);
            break;
        case DataType::FP64:
            run_conv2d_inplace<nntile::fp64_t>(
                ctx, alpha, X, C, beta, Y, padding, stride, dilation);
            break;
        case DataType::FP16:
            throw std::runtime_error(
                "FP16 not supported for conv2d_inplace (use FP32_FAST_FP16)");
            break;
        case DataType::BF16:
            run_conv2d_inplace<nntile::bf16_t>(
                ctx, alpha, X, C, beta, Y, padding, stride, dilation);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for conv2d_inplace");
        default:
            throw std::runtime_error(
                "Unsupported data type for conv2d_inplace");
    }
}

} // namespace nntile::graph
