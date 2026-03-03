/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/silu_backward.cc
 * TensorGraph silu_backward operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/silu_backward.hh"

#include <stdexcept>
#include <utility>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/silu_backward.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_silu_backward(
    TensorGraph::ExecutionContext& ctx,
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    TensorGraph::TensorNode* dx)
{
    auto& x_t = ctx.get_tensor<T>(x);
    auto& dy_t = ctx.get_tensor<T>(dy);
    auto& dx_t = ctx.get_tensor<T>(dx);
    nntile::tensor::silu_backward<T>(x_t, dy_t, dx_t);
}

} // namespace

TensorGraph::TensorNode* silu_backward(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    const std::string& output_name)
{
    if(x == nullptr || dy == nullptr)
    {
        throw std::invalid_argument(
            "silu_backward: input tensors must be non-null");
    }
    if(x->graph() != dy->graph())
    {
        throw std::invalid_argument(
            "silu_backward: input tensors must belong to the same graph");
    }
    if(x->dtype() != dy->dtype())
    {
        throw std::invalid_argument(
            "silu_backward: input tensors must have the same dtype");
    }
    if(x->shape() != dy->shape())
    {
        throw std::invalid_argument(
            "silu_backward: input tensors must have the same shape");
    }

    std::vector<Index> output_shape = x->shape();
    TensorGraph::TensorNode* dx = x->graph()->data(
        std::move(output_shape),
        output_name,
        x->dtype());

    silu_backward(x, dy, dx);

    return dx;
}

void silu_backward(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    TensorGraph::TensorNode* dx)
{
    if(x == nullptr || dy == nullptr || dx == nullptr)
    {
        throw std::invalid_argument(
            "silu_backward: input tensors must be non-null");
    }
    if(x->graph() != dy->graph() || x->graph() != dx->graph())
    {
        throw std::invalid_argument(
            "silu_backward: input tensors must belong to the same graph");
    }
    if(x->dtype() != dy->dtype() || x->dtype() != dx->dtype())
    {
        throw std::invalid_argument(
            "silu_backward: input tensors must have the same dtype");
    }
    if(x->shape() != dx->shape())
    {
        throw std::invalid_argument(
            "silu_backward: dx must have the same shape as x");
    }

    auto op = std::make_shared<TensorSiluBackwardOp>(x, dy, dx);
    x->graph()->add_op(op);
}

void TensorSiluBackwardOp::execute(
    TensorGraph::ExecutionContext& ctx) const
{
    DataType dtype = ctx.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_silu_backward<nntile::fp32_t>(ctx, x, dy, dx);
            break;
        case DataType::FP32_FAST_TF32:
            run_silu_backward<nntile::fp32_fast_tf32_t>(ctx, x, dy, dx);
            break;
        case DataType::FP32_FAST_FP16:
            run_silu_backward<nntile::fp32_fast_fp16_t>(ctx, x, dy, dx);
            break;
        case DataType::FP32_FAST_BF16:
            run_silu_backward<nntile::fp32_fast_bf16_t>(ctx, x, dy, dx);
            break;
        case DataType::FP64:
            run_silu_backward<nntile::fp64_t>(ctx, x, dy, dx);
            break;
        case DataType::FP16:
            run_silu_backward<nntile::fp16_t>(ctx, x, dy, dx);
            break;
        case DataType::BF16:
            run_silu_backward<nntile::bf16_t>(ctx, x, dy, dx);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for silu_backward operation");
        default:
            throw std::runtime_error("Unsupported data type for silu_backward");
    }
}

} // namespace nntile::graph
