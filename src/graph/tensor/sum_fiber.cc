/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sum_fiber.cc
 * TensorGraph sum_fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sum_fiber.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sum_fiber.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_sum_fiber(
    TensorGraph::ExecutionContext& ctx,
    Scalar alpha, Scalar beta,
    Index axis, Index batch_ndim, int redux,
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y)
{
    auto& x_t = ctx.get_tensor<T>(x);
    auto& y_t = ctx.get_tensor<T>(y);
    nntile::tensor::sum_fiber<T>(
        alpha, x_t, beta, y_t, axis, batch_ndim, redux);
}

} // namespace

void sum_fiber(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument(
            "sum_fiber: input tensors must be non-null");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "sum_fiber: input tensors must belong to the same graph");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "sum_fiber: input tensors must have the same dtype");
    }

    auto op = std::make_shared<TensorSumFiberOp>(
        x, y, axis, batch_ndim, redux, alpha, beta);

    x->graph()->add_op(op);
}

void TensorSumFiberOp::execute(
    TensorGraph::ExecutionContext& ctx) const
{
    DataType dtype = ctx.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_sum_fiber<nntile::fp32_t>(
                ctx, alpha, beta, axis, batch_ndim, redux, x, y);
            break;
        case DataType::FP32_FAST_TF32:
            run_sum_fiber<nntile::fp32_fast_tf32_t>(
                ctx, alpha, beta, axis, batch_ndim, redux, x, y);
            break;
        case DataType::FP32_FAST_FP16:
            run_sum_fiber<nntile::fp32_fast_fp16_t>(
                ctx, alpha, beta, axis, batch_ndim, redux, x, y);
            break;
        case DataType::FP32_FAST_BF16:
            run_sum_fiber<nntile::fp32_fast_bf16_t>(
                ctx, alpha, beta, axis, batch_ndim, redux, x, y);
            break;
        case DataType::FP64:
            run_sum_fiber<nntile::fp64_t>(
                ctx, alpha, beta, axis, batch_ndim, redux, x, y);
            break;
        case DataType::FP16:
            run_sum_fiber<nntile::fp16_t>(
                ctx, alpha, beta, axis, batch_ndim, redux, x, y);
            break;
        case DataType::BF16:
            run_sum_fiber<nntile::bf16_t>(
                ctx, alpha, beta, axis, batch_ndim, redux, x, y);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sum_fiber operation");
        default:
            throw std::runtime_error("Unsupported data type for sum_fiber");
    }
}

} // namespace nntile::graph
