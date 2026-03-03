/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/gelu_backward.cc
 * TensorGraph GeLU backward operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/gelu_backward.hh"

#include <stdexcept>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/gelu_backward.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_gelu_backward(
    TensorGraph::ExecutionContext& ctx,
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    TensorGraph::TensorNode* dx)
{
    auto& x_t = ctx.get_tensor<T>(x);
    auto& dy_t = ctx.get_tensor<T>(dy);
    auto& dx_t = ctx.get_tensor<T>(dx);
    nntile::tensor::gelu_backward<T>(x_t, dy_t, dx_t);
}

} // namespace

void gelu_backward(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    TensorGraph::TensorNode* dx)
{
    if(x == nullptr || dy == nullptr || dx == nullptr)
    {
        throw std::invalid_argument(
            "gelu_backward: input tensors must be non-null");
    }
    if(x->graph() != dy->graph() || x->graph() != dx->graph())
    {
        throw std::invalid_argument(
            "gelu_backward: input tensors must belong to the same graph");
    }
    if(x->dtype() != dy->dtype() || x->dtype() != dx->dtype())
    {
        throw std::invalid_argument(
            "gelu_backward: input tensors must have the same dtype");
    }
    if(x->shape() != dy->shape() || x->shape() != dx->shape())
    {
        throw std::invalid_argument(
            "gelu_backward: all tensors must have the same shape");
    }

    auto op = std::make_shared<TensorGeluBackwardOp>(x, dy, dx);
    x->graph()->add_op(op);
}

void TensorGeluBackwardOp::execute(
    TensorGraph::ExecutionContext& ctx) const
{
    DataType dtype = ctx.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_gelu_backward<nntile::fp32_t>(ctx, x, dy, dx);
            break;
        case DataType::FP32_FAST_TF32:
            run_gelu_backward<nntile::fp32_fast_tf32_t>(ctx, x, dy, dx);
            break;
        case DataType::FP32_FAST_FP16:
            run_gelu_backward<nntile::fp32_fast_fp16_t>(ctx, x, dy, dx);
            break;
        case DataType::FP32_FAST_BF16:
            run_gelu_backward<nntile::fp32_fast_bf16_t>(ctx, x, dy, dx);
            break;
        case DataType::FP64:
            run_gelu_backward<nntile::fp64_t>(ctx, x, dy, dx);
            break;
        case DataType::FP16:
            run_gelu_backward<nntile::fp16_t>(ctx, x, dy, dx);
            break;
        case DataType::BF16:
            run_gelu_backward<nntile::bf16_t>(ctx, x, dy, dx);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gelu_backward operation");
        default:
            throw std::runtime_error("Unsupported data type for gelu_backward");
    }
}

} // namespace nntile::graph
