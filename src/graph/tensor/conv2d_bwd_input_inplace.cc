/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/conv2d_bwd_input_inplace.cc
 * TensorGraph conv2d_bwd_input_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/conv2d_bwd_input_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/conv2d_bwd_input_inplace.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_conv2d_bwd_input_inplace(TensorGraph::Runtime& runtime,
                                 Scalar alpha, TensorGraph::TensorNode* dY,
                                 TensorGraph::TensorNode* kernel, Scalar beta,
                                 TensorGraph::TensorNode* dX,
                                 const std::array<Index, 2>& padding,
                                 const std::array<Index, 2>& stride,
                                 const std::array<Index, 2>& dilation)
{
    auto& dY_t = runtime.get_tensor<T>(dY);
    auto& kernel_t = runtime.get_tensor<T>(kernel);
    auto& dX_t = runtime.get_tensor<T>(dX);
    nntile::tensor::conv2d_bwd_input_inplace<T>(
        alpha, dY_t, kernel_t, beta, dX_t, padding, stride, dilation);
}

} // namespace

void conv2d_bwd_input_inplace(Scalar alpha,
                              TensorGraph::TensorNode* dY,
                              TensorGraph::TensorNode* kernel,
                              Scalar beta,
                              TensorGraph::TensorNode* dX,
                              std::array<Index, 2> padding,
                              std::array<Index, 2> stride,
                              std::array<Index, 2> dilation)
{
    if(dY == nullptr || kernel == nullptr || dX == nullptr)
        throw std::invalid_argument(
            "conv2d_bwd_input_inplace: tensors must be non-null");
    if(dY->graph() != kernel->graph() || kernel->graph() != dX->graph())
        throw std::invalid_argument(
            "conv2d_bwd_input_inplace: tensors must belong to same graph");
    if(dY->dtype() != kernel->dtype() || kernel->dtype() != dX->dtype())
        throw std::invalid_argument(
            "conv2d_bwd_input_inplace: tensors must have same dtype");
    auto op = std::make_shared<TensorConv2dBwdInputInplaceOp>(
        alpha, dY, kernel, beta, dX, padding, stride, dilation);
    dX->graph()->add_op(op);
}

void TensorConv2dBwdInputInplaceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(dY);
    switch(dtype)
    {
        case DataType::FP32:
            run_conv2d_bwd_input_inplace<nntile::fp32_t>(
                runtime, alpha, dY, kernel, beta, dX, padding, stride, dilation);
            break;
        case DataType::FP32_FAST_TF32:
            run_conv2d_bwd_input_inplace<nntile::fp32_fast_tf32_t>(
                runtime, alpha, dY, kernel, beta, dX, padding, stride, dilation);
            break;
        case DataType::FP32_FAST_FP16:
            run_conv2d_bwd_input_inplace<nntile::fp32_fast_fp16_t>(
                runtime, alpha, dY, kernel, beta, dX, padding, stride, dilation);
            break;
        case DataType::FP32_FAST_BF16:
            run_conv2d_bwd_input_inplace<nntile::fp32_fast_bf16_t>(
                runtime, alpha, dY, kernel, beta, dX, padding, stride, dilation);
            break;
        case DataType::FP64:
            run_conv2d_bwd_input_inplace<nntile::fp64_t>(
                runtime, alpha, dY, kernel, beta, dX, padding, stride, dilation);
            break;
        case DataType::FP16:
            throw std::runtime_error(
                "FP16 not supported for conv2d_bwd_input_inplace "
                "(use FP32_FAST_FP16)");
            break;
        case DataType::BF16:
            run_conv2d_bwd_input_inplace<nntile::bf16_t>(
                runtime, alpha, dY, kernel, beta, dX, padding, stride, dilation);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for conv2d_bwd_input_inplace");
        default:
            throw std::runtime_error(
                "Unsupported data type for conv2d_bwd_input_inplace");
    }
}

} // namespace nntile::graph
